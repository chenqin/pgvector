#include "postings_eval.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>

extern "C" {
#include "filter.h"
}

#define PCUDA(x)                                                                                   \
  do {                                                                                             \
    cudaError_t e_ = (x);                                                                          \
    if (e_ != cudaSuccess) {                                                                       \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));           \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

/* Sets the bits of every row in the ranges: range i is postings[first_i .. first_i + count_i). */
__global__ void scatter_rows(const uint32_t* postings, const uint64_t* firsts, const uint64_t* ends,
                             const uint64_t* offsets, uint32_t nranges, uint64_t total, uint32_t* words)
{
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < total;
       i += uint64_t(gridDim.x) * blockDim.x) {
    /* The range holding output position i: offsets is the running total of counts. */
    uint32_t lo = 0, hi = nranges;
    while (hi - lo > 1) {
      uint32_t mid = (lo + hi) / 2;
      if (offsets[mid] <= i) lo = mid; else hi = mid;
    }
    uint32_t row = postings[firsts[lo] + (i - offsets[lo])];
    atomicOr(&words[row >> 5], 1u << (row & 31));
  }
}

/* a = op(a, b) word by word; op 5 AND, 6 OR, 7 NOT a (b unused), 100: a = ~a & ~b (negated leaf). */
__global__ void combine(uint32_t* a, const uint32_t* b, size_t nwords, int op, uint32_t tail_mask)
{
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < nwords;
       i += size_t(gridDim.x) * blockDim.x) {
    uint32_t x = a[i], y = b ? b[i] : 0u;
    uint32_t r = op == 5 ? (x & y) : op == 6 ? (x | y) : op == 7 ? ~x : (~x & ~y);
    if (i == nwords - 1) r &= tail_mask;
    a[i] = r;
  }
}

__global__ void fill(uint32_t* a, size_t nwords, uint32_t value, uint32_t tail_mask)
{
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < nwords;
       i += size_t(gridDim.x) * blockDim.x)
    a[i] = i == nwords - 1 ? (value & tail_mask) : value;
}

PostingsColumn postings_build(const int32_t* data, const uint32_t* offsets, uint64_t rows)
{
  PostingsColumn c;
  c.multi = offsets != nullptr;
  const uint64_t n = c.multi ? offsets[rows] : rows;
  /* Distinct values, then a dense id per element by binary search. */
  std::vector<int32_t> all(data, data + n);
  std::sort(all.begin(), all.end());
  all.erase(std::unique(all.begin(), all.end()), all.end());
  c.distinct = all;
  std::vector<uint32_t> id(n);
  std::vector<uint64_t> counts(c.distinct.size() + 1, 0);
  auto id_of = [&](int32_t v) {
    return uint32_t(std::lower_bound(c.distinct.begin(), c.distinct.end(), v) - c.distinct.begin());
  };
  /* A row appears once per distinct element: drop repeats within a row. */
  std::vector<uint8_t> keep(n, 1);
  for (uint64_t r = 0; r < rows; r++) {
    uint64_t b = c.multi ? offsets[r] : r, e = c.multi ? offsets[r + 1] : r + 1;
    for (uint64_t i = b; i < e; i++) {
      id[i] = id_of(data[i]);
      for (uint64_t j = b; j < i; j++)
        if (data[j] == data[i]) keep[i] = 0;
      if (keep[i]) counts[id[i] + 1]++;
    }
  }
  std::partial_sum(counts.begin(), counts.end(), counts.begin());
  c.starts = counts;
  std::vector<uint32_t> post(counts.back());
  std::vector<uint64_t> at(counts.begin(), counts.end() - 1);
  for (uint64_t r = 0; r < rows; r++) {
    uint64_t b = c.multi ? offsets[r] : r, e = c.multi ? offsets[r + 1] : r + 1;
    for (uint64_t i = b; i < e; i++)
      if (keep[i]) post[at[id[i]]++] = uint32_t(r);
  }
  PCUDA(cudaMalloc(&c.postings, std::max<size_t>(1, post.size()) * sizeof(uint32_t)));
  PCUDA(cudaMemcpy(c.postings, post.data(), post.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
  /* NULL rows: a single-valued NULL, or a NULL array (its sole element INT32_MIN). */
  size_t nwords = (rows + 31) / 32;
  std::vector<uint32_t> nulls(nwords, 0);
  for (uint64_t r = 0; r < rows; r++) {
    bool is_null = c.multi ? (offsets[r + 1] - offsets[r] == 1 && data[offsets[r]] == ANN_FILTER_NULL_VALUE)
                           : data[r] == ANN_FILTER_NULL_VALUE;
    if (is_null) { nulls[r / 32] |= 1u << (r % 32); c.null_count++; }
  }
  PCUDA(cudaMalloc(&c.null_words, nwords * sizeof(uint32_t)));
  PCUDA(cudaMemcpy(c.null_words, nulls.data(), nwords * sizeof(uint32_t), cudaMemcpyHostToDevice));
  return c;
}

void postings_scratch_init(PostingsScratch& s, uint64_t rows)
{
  s.rows   = rows;
  s.nwords = (rows + 31) / 32;
  s.stack.assign(ANN_FILTER_MAX_DEPTH, nullptr);
}

static uint32_t* slot(PostingsScratch& s, int depth)
{
  if (!s.stack[depth]) PCUDA(cudaMalloc(&s.stack[depth], s.nwords * sizeof(uint32_t)));
  return s.stack[depth];
}

void postings_eval(const std::vector<PostingsColumn*>& columns, const AnnUdfStep* steps, uint32_t nsteps,
                   const int32_t* values, PostingsScratch& s, uint32_t* out, cudaStream_t stream)
{
  const uint32_t tail = s.rows % 32 ? (1u << (s.rows % 32)) - 1 : 0xffffffffu;
  const int threads = 256, blocks_words = int(std::min<size_t>((s.nwords + threads - 1) / threads, 4096));
  int depth = 0;
  for (uint32_t i = 0; i < nsteps; i++) {
    const AnnUdfStep& st = steps[i];
    if (st.op == ANN_FILTER_AND || st.op == ANN_FILTER_OR) {
      combine<<<blocks_words, threads, 0, stream>>>(slot(s, depth - 2), slot(s, depth - 1), s.nwords, st.op,
                                                    tail);
      depth--;
      continue;
    }
    if (st.op == ANN_FILTER_NOT) {
      combine<<<blocks_words, threads, 0, stream>>>(slot(s, depth - 1), nullptr, s.nwords, 7, tail);
      continue;
    }
    uint32_t* w = slot(s, depth++);
    if (st.op == ANN_FILTER_ALL || st.op == ANN_FILTER_NONE) {
      fill<<<blocks_words, threads, 0, stream>>>(w, s.nwords, st.op == ANN_FILTER_ALL ? ~0u : 0u, tail);
      continue;
    }
    const PostingsColumn& c = *columns[st.column];
    /* The leaf's runs of postings, planned on the host from the distinct values. */
    s.host_ranges.clear();
    auto add_value_range = [&](uint64_t a, uint64_t b) {  // distinct indexes [a, b)
      if (a < b && c.starts[b] > c.starts[a]) {
        s.host_ranges.push_back(c.starts[a]);
        s.host_ranges.push_back(c.starts[b]);
      }
    };
    auto index_of = [&](int64_t v) {
      return uint64_t(std::lower_bound(c.distinct.begin(), c.distinct.end(), v) - c.distinct.begin());
    };
    if (st.op == ANN_FILTER_NULL) {
      /* NULL: the column's NULL mask is the answer (negated: its complement). */
      PCUDA(cudaMemcpyAsync(w, c.null_words, s.nwords * 4, cudaMemcpyDeviceToDevice, stream));
      if (st.negate) combine<<<blocks_words, threads, 0, stream>>>(w, nullptr, s.nwords, 7, tail);
      continue;
    }
    if (st.op == ANN_FILTER_RANGE) {
      int64_t low = std::max<int64_t>(st.low, int64_t(ANN_FILTER_NULL_VALUE) + 1);
      if (low <= st.high) add_value_range(index_of(low), index_of(int64_t(st.high) + 1));
    } else {
      for (uint32_t v = 0; v < st.count; v++) {
        uint64_t j = index_of(values[st.first + v]);
        if (j < c.distinct.size() && c.distinct[j] == values[st.first + v]) add_value_range(j, j + 1);
      }
    }
    PCUDA(cudaMemsetAsync(w, 0, s.nwords * 4, stream));
    const uint32_t nranges = uint32_t(s.host_ranges.size() / 2);
    if (nranges) {
      /* firsts, ends, offsets (running totals) in one upload. */
      std::vector<uint64_t> buf(3 * size_t(nranges));
      uint64_t total = 0;
      for (uint32_t r = 0; r < nranges; r++) {
        buf[r]               = s.host_ranges[2 * r];
        buf[nranges + r]     = s.host_ranges[2 * r + 1];
        buf[2 * nranges + r] = total;
        total += s.host_ranges[2 * r + 1] - s.host_ranges[2 * r];
      }
      if (buf.size() > s.range_capacity) {
        if (s.ranges) PCUDA(cudaFree(s.ranges));
        s.range_capacity = std::max<size_t>(buf.size(), 4096);
        PCUDA(cudaMalloc(&s.ranges, s.range_capacity * sizeof(uint64_t)));
      }
      PCUDA(cudaMemcpyAsync(s.ranges, buf.data(), buf.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
      /* The upload's host buffer must outlive the copy. */
      PCUDA(cudaStreamSynchronize(stream));
      int blocks = int(std::min<uint64_t>((total + threads - 1) / threads, 8192));
      scatter_rows<<<blocks, threads, 0, stream>>>(c.postings, s.ranges, s.ranges + nranges,
                                                   s.ranges + 2 * size_t(nranges), nranges, total, w);
    }
    /* A negated comparison matches the non-NULL rows the leaf does not. */
    if (st.negate) combine<<<blocks_words, threads, 0, stream>>>(w, c.null_words, s.nwords, 100, tail);
  }
  if (depth != 1) { fprintf(stderr, "postings_eval: program left %d results\n", depth); exit(1); }
  PCUDA(cudaMemcpyAsync(out, slot(s, 0), s.nwords * 4, cudaMemcpyDeviceToDevice, stream));
}
