/*
 * Two-hop CAGRA search on one FP16 shard with three kinds of filter, measured on the same queries:
 *
 *   bitset1  one query per call, its bitset already on the device (the worker's same-filter batch)
 *   bitsetq  BATCH queries per call, each with its own device bitset (query_bitsets, --mixed-graph)
 *   udf      BATCH queries per call, each with its own predicate, evaluated row by row in the walk by
 *            the JIT-LTO filter UDF of ann/server/udf.c (no bitset)
 *   udf1     one query per call with the UDF
 *   udfv     as udf, with value bitsets: each multi-valued column's elements as dense 16-bit ids
 *            (ranks among its distinct values), and each leaf's values as a bitset over those ids,
 *            built on the host for the call, so a row test is the row's ids and one bit each
 *   udfv1    one query per call with the value-bitset UDF
 *   udfb     control: a UDF testing each query's device bitset (bitsetq's), one load a test, to
 *            separate the cost of cuVS' UDF path from the predicate's
 *   udfq     as udfp, with one 32-byte descriptor a query (its column's packed ids and offsets, its
 *            value bitset) so a single-leaf program's test is three loads; others interpret as udfv
 *   udfp     as udfv, each row's first eight ids packed into 16 bytes beside the row (one read a
 *            test, as a bitset's); rows with more than eight take udfv's offsets
 *   build1   one query per call, its bitset built on the GPU for this call from device postings
 *            (postings_eval.cu), as a filter-cache miss would, then searched as bitset1
 *   buildq   BATCH queries per call, each bitset built for the call, then searched as bitsetq
 *   twol     as bitsetq, each query's bitset stored two-level (a bit per 1,024-item block, words of
 *            occupied blocks only; prototype cuVS flag CUVS_TWO_LEVEL); use with --reorder
 *   scanc    one query per call, its bitset built by cuVS, searched by cuVS' FP8 filtered scan (codes
 *            made at load from the FP16 rows; candidates k + k * --scan-slack / 100); needs a cuVS
 *            with the filtered scan
 *   buildc   as buildq, each bitset built by cuVS (cuvs::core::postings::evaluate, the worker's
 *            build) from the same postings; programs of one VALUES or OVERLAP leaf only
 *   stock1   control: cuVS' own filtered walk (filter_expansion SCORE_ALL, algo AUTO), one query per
 *            call with its device bitset, the filtering rate left for cuVS to count
 *   stockr1  as stock1, the filtering rate given (no count of the bitset per call)
 *   stockb   BATCH queries per call that share one filter (the first query's), as cuVS' own filtered
 *            search takes them: SCORE_ALL, algo AUTO, the filtering rate given
 *   bitsetb  the same shared-filter batches with the two-hop walk (a plain bitset_filter, no
 *            query_bitsets)
 *   linked1  as bitset1, the bitset declared as a bitset of bytes, so the kernel calls the linked
 *            sample_filter for every test instead of reading the words itself
 *   linkedb  as bitsetb, through the linked sample_filter
 *   nofilter BATCH queries per call with no filter at all (MULTI_CTA, the same settings); use with
 *            --random-pass 1 so that recall is measured against every row
 *   ivfpqb   control: an IVF-PQ index over the same rows (built from vectors.f32, or loaded from
 *            --ivf-index), searched with cuVS' bitset filter in batches that share one filter, as
 *            stockb; --ivf-probes lists a query, k * --ivf-refine candidates re-scored exactly
 *            from the FP32 rows on the device (cuvs::neighbors::refine)
 *   ivfpq1   the same, one query per call with its own filter; --ivf-lut 32|16|8 and --ivf-internal 32|16 set the
 *            precision of the search's lookup tables and internal distances (default 32: cuVS' defaults);
 *            --ivf-refine-host re-scores on the host (cuVS' device refine only takes k <= 256)
 *
 * With a build mode, a check pass first builds every case's bitset on the GPU, compares it with the
 * host interpreter's bit for bit and reports the build time.
 *
 * Reports per call latency, queries per second (--threads host threads, each with its own stream,
 * share the index), recall against an exact scan of each query's passing rows, rows returned that
 * fail their predicate, and the first UDF call's time (the JIT compile).
 *
 *   cagra_twohop_udf_probe INDEX_DIR CASES [--k K] [--batch B] [--reps R] [--threads T]
 *                          [--itopk N] [--iterations N] [--samplings N] [--width N] [--block N]
 *                          [--no-pitch] [--reorder] [--random-pass RATE] [--modes bitset1,bitsetq,udf,udf1,udfv,udfv1,udfp,udfb,udfq,twol,scanc,build1,buildq,buildc,stock1,stockr1,stockb,bitsetb,linked1,linkedb,nofilter] [--no-recall]
 *
 * CASES (benchmarks/ann/twohop_udf_cases.py): u32 count, u32 dims, then per case u32 k (unused:
 * --k sets k), u32 program length, the predicate program (ann/common/filter.h) and the FP32 query.
 * INDEX_DIR is a worker index directory with one CAGRA shard holding its FP16 dataset. One JSON
 * object per line on stdout.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuvs/core/postings.hpp>
#include <cuvs/neighbors/cagra.hpp>
#if __has_include(<cuvs/preprocessing/quantize/fp8.hpp>)
#include <cuvs/preprocessing/quantize/fp8.hpp>
#define PROBE_HAS_FP8_SCAN 1  // a cuVS with the FP8 filtered scan (the fork's scan pull request)
#endif
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/refine.hpp>
#include <raft/core/bitset.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <algorithm>
#include <array>
#include <set>
#include <unordered_map>
#include <fstream>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern "C" {
#include "filter.h"
#include "udf.h"
}
#include "postings_eval.cuh"

namespace cagra = cuvs::neighbors::cagra;

#define CUDA(x)                                                                                    \
  do {                                                                                             \
    cudaError_t e_ = (x);                                                                          \
    if (e_ != cudaSuccess) {                                                                       \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));           \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

/*
 * The value-bitset UDF (udfv): ann_predicate_udfv(query, row, params) over ann_udfv_params. Leaves
 * on multi-valued columns read the row's dense ids (dense[column]) and test them against the
 * leaf's aux: a bitset of ids in arena (VALUES, OVERLAP), an id interval (RANGE), or the NULL id.
 * Other leaves, and the stack, are the interpreter's (ann/server/udf.c).
 */
static const char kUdfvSource[] = R"UDF(
struct ann_udf_step {
  unsigned char op, column, negate, multi;
  int low, high;
  unsigned int first, count;
};
struct ann_udfv_aux {
  unsigned int a, b, null_id, kind;  /* kind 0: bitset arena[a, a + b); 1: ids [a, b) */
};
struct ann_udfv_params {
  const int* columns[8];
  const ann_udf_step* steps;
  const int* values;
  const unsigned int* starts;
  unsigned long long row_base;
  const unsigned int* offsets[8];
  const unsigned short* dense[8];
  const ann_udfv_aux* aux;
  const unsigned int* arena;
  const uint4* packed[8];  /* udfp: 8 ids a row, 0xffff past the last, 0xfffe first: overflow */
};

__device__ bool ann_udfv_among(const ann_udf_step& s, const int* values, int v)
{
  unsigned int lo = s.first, hi = s.first + s.count;
  while (lo < hi) {
    unsigned int mid = lo + (hi - lo) / 2;
    if (values[mid] < v) lo = mid + 1; else hi = mid;
  }
  return lo < s.first + s.count && values[lo] == v;
}

__device__ bool ann_udfv_single(const ann_udf_step& s, const int* values, int v)
{
  const int null_value = (int)0x80000000;
  bool match;
  if (s.op == 3) match = v == null_value;
  else if (v == null_value) return false;
  else if (s.op == 2) match = s.low <= v && v <= s.high;
  else match = ann_udfv_among(s, values, v);
  return s.negate ? !match : match;
}

__device__ bool ann_udfv_multi(const ann_udf_step& s, const ann_udfv_aux& x, const unsigned int* arena,
                               const unsigned short* ids, unsigned int begin, unsigned int end)
{
  const bool is_null = end - begin == 1 && (unsigned int)ids[begin] == x.null_id;
  bool match = false;
  if (s.op == 3) {
    match = is_null;
  } else if (is_null) {
    return false;
  } else if (x.kind == 1) {
    for (unsigned int i = begin; i < end && !match; i++) match = ids[i] >= x.a && ids[i] < x.b;
  } else {
    for (unsigned int i = begin; i < end && !match; i++) {
      unsigned int id = ids[i];
      match = (id >> 5) < x.b && ((__ldg(arena + x.a + (id >> 5)) >> (id & 31)) & 1u);
    }
  }
  return s.negate ? !match : match;
}

__device__ bool ann_udfv_test(const ann_udfv_aux& x, unsigned int id, const unsigned int* arena)
{
  if (x.kind == 1) return id >= x.a && id < x.b;
  return (id >> 5) < x.b && ((__ldg(arena + x.a + (id >> 5)) >> (id & 31)) & 1u);
}

/* udfp: the row's packed ids; -1 when it has more than fit (take the offsets). */
__device__ int ann_udfp_multi(const ann_udf_step& s, const ann_udfv_aux& x, const unsigned int* arena,
                              const uint4* packed, unsigned long long row)
{
  const uint4 w = __ldg(packed + row);
  const unsigned int words[4] = {w.x, w.y, w.z, w.w};
  if ((w.x & 0xffffu) == 0xfffeu) return -1;
  unsigned int n = 0;
  bool match = false;
  for (int k = 0; k < 8; k++) {
    unsigned int id = (words[k >> 1] >> ((k & 1) * 16)) & 0xffffu;
    if (id == 0xffffu) break;
    n++;
    if (!match && s.op != 3) match = ann_udfv_test(x, id, arena);
  }
  const bool is_null = n == 1 && (w.x & 0xffffu) == x.null_id;
  if (s.op == 3) match = is_null;
  else if (is_null) return 0;
  return (s.negate ? !match : match) ? 1 : 0;
}

__device__ bool ann_udfv_leaf(const ann_udfv_params* p, unsigned int i, unsigned long long row)
{
  const ann_udf_step s = p->steps[i];
  if (s.multi && p->packed[s.column]) {
    const int r = ann_udfp_multi(s, p->aux[i], p->arena, p->packed[s.column], row);
    if (r >= 0) return r == 1;
  }
  if (s.multi)
    return ann_udfv_multi(s, p->aux[i], p->arena, p->dense[s.column], p->offsets[s.column][row],
                          p->offsets[s.column][row + 1]);
  return ann_udfv_single(s, p->values, p->columns[s.column][row]);
}

__device__ bool ann_predicate_udfv(uint32_t query_id, source_index_t source_id, void* filter_data)
{
  const ann_udfv_params* p = static_cast<const ann_udfv_params*>(filter_data);
  const unsigned long long row = p->row_base + source_id;
  const unsigned int first = p->starts[query_id], end = p->starts[query_id + 1];
  if (end - first == 1) return ann_udfv_leaf(p, first, row);  /* one leaf: no stack */
  unsigned long long stack = 0;
  int depth = 0;
  for (unsigned int i = first; i < end; i++) {
    const unsigned char op = p->steps[i].op;
    unsigned long long bit;
    if (op == 5 || op == 6) {
      unsigned long long b = (stack >> (depth - 1)) & 1ull, a = (stack >> (depth - 2)) & 1ull;
      depth -= 2;
      bit = op == 5 ? (a & b) : (a | b);
    } else if (op == 7) {
      bit = ((stack >> (depth - 1)) & 1ull) ^ 1ull;
      depth--;
    } else if (op == 8) {
      bit = 1;
    } else if (op == 9) {
      bit = 0;
    } else {
      bit = ann_udfv_leaf(p, i, row) ? 1ull : 0ull;
    }
    stack = (stack & ~(1ull << depth)) | (bit << depth);
    depth++;
  }
  return depth == 1 && (stack & 1ull);
}
)UDF";

static const char kUdfbSource[] = R"UDF(
__device__ bool ann_predicate_udfb(uint32_t query_id, source_index_t source_id, void* filter_data)
{
  const unsigned int* const* words = static_cast<const unsigned int* const*>(filter_data);
  const unsigned int* w = words[query_id];
  return (__ldg(w + (source_id >> 5)) >> (source_id & 31)) & 1u;
}
)UDF";

/*
 * udfq: a query's descriptor. kind 1: one leaf OVERLAP/VALUES on a multi-valued column, value bitset
 * at bits[0, nwords); kind 0: interpret through generic (an ann_udfv_params) as udfv.
 */
static const char kUdfqSource[] = R"UDF(
struct ann_udfq_desc {
  const uint4* packed;
  const unsigned int* bits;
  const unsigned int* offsets;
  const unsigned short* ids;
  unsigned int nwords, kind, negate, null_id;
  const void* generic;
};
)UDF";

struct UdfqDesc {
  const void* packed;
  const uint32_t* bits;
  const uint32_t* offsets;
  const uint16_t* ids;
  uint32_t nwords, kind, negate, null_id;
  const void* generic;
};
static const char kUdfqPredicate[] = R"UDF(
__device__ bool ann_udfq_bit(const ann_udfq_desc& d, unsigned int id)
{
  return (id >> 5) < d.nwords && ((__ldg(d.bits + (id >> 5)) >> (id & 31)) & 1u);
}

__device__ bool ann_predicate_udfq(uint32_t query_id, source_index_t source_id, void* filter_data)
{
  const ann_udfq_desc d = static_cast<const ann_udfq_desc*>(filter_data)[query_id];
  if (d.kind == 0) return ann_predicate_udfv(query_id, source_id, const_cast<void*>(d.generic));
  const uint4 w = __ldg(d.packed + source_id);
  bool match = false;
  unsigned int n = 0;
  if ((w.x & 0xffffu) == 0xfffeu) {
    const unsigned int b = d.offsets[source_id], e = d.offsets[source_id + 1];
    n = e - b;
    for (unsigned int i = b; i < e && !match; i++) match = ann_udfq_bit(d, d.ids[i]);
    if (n == 1 && d.ids[b] == d.null_id) return false;
  } else {
    /* Every slot's test is issued unconditionally, so their loads overlap. */
    const unsigned int words[4] = {w.x, w.y, w.z, w.w};
    unsigned int hit = 0;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const unsigned int id = (words[k >> 1] >> ((k & 1) * 16)) & 0xffffu;
      const unsigned int valid = id != 0xffffu;
      n += valid;
      hit |= valid & (unsigned int)ann_udfq_bit(d, valid ? id : 0u);
    }
    match = hit != 0;
    if (n == 1 && (w.x & 0xffffu) == d.null_id) return false;
  }
  return d.negate ? !match : match;
}
)UDF";

struct UdfvAux {
  uint32_t a, b, null_id, kind;
};
struct UdfvParams {
  const int32_t* columns[ANN_UDF_MAX_COLUMNS];
  const AnnUdfStep* steps;
  const int32_t* values;
  const uint32_t* starts;
  uint64_t row_base;
  const uint32_t* offsets[ANN_UDF_MAX_COLUMNS];
  const uint16_t* dense[ANN_UDF_MAX_COLUMNS];
  const UdfvAux* aux;
  const uint32_t* arena;
  const void* packed[ANN_UDF_MAX_COLUMNS];
};

static void fail(const std::string& why)
{
  fprintf(stderr, "cagra_twohop_udf_probe: %s\n", why.c_str());
  exit(1);
}

static double now_ms()
{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

static const void* map_file(const std::string& path, size_t* size)
{
  int fd = open(path.c_str(), O_RDONLY);
  struct stat st;
  if (fd < 0 || fstat(fd, &st) != 0) fail("cannot open " + path);
  void* p = st.st_size ? mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0) : nullptr;
  close(fd);
  if (st.st_size && p == MAP_FAILED) fail("cannot map " + path);
  *size = st.st_size;
  return p;
}

struct Manifest {
  uint64_t rows = 0;
  uint32_t dims = 0;
  std::string shard;
  std::map<std::string, std::string> attributes, multivalued;
};

static Manifest read_manifest(const std::string& dir)
{
  Manifest m;
  FILE* f = fopen((dir + "/manifest").c_str(), "r");
  char line[4096];
  if (!f) fail("no manifest in " + dir);
  while (fgets(line, sizeof(line), f)) {
    std::istringstream in(line);
    std::string key, a, b, c, d;
    in >> key >> a >> b >> c >> d;
    if (key == "rows") m.rows = std::stoull(a);
    if (key == "dimensions") m.dims = std::stoul(a);
    if (key == "attribute") m.attributes[a] = b;
    if (key == "multivalued") m.multivalued[a] = b;
    if (key == "shard") {
      if (m.shard.size()) fail("more than one shard");
      m.shard = d;
    }
  }
  fclose(f);
  if (!m.rows || !m.dims || m.shard.empty() || m.shard == "-") fail("need one CAGRA shard: " + dir);
  return m;
}

struct Case {
  std::vector<unsigned char> program;
  std::vector<float> query;
  uint32_t first_step = 0, nsteps = 0;
  std::vector<uint32_t> words;  // host bitset over the shard's rows
  uint64_t count = 0;
  std::vector<uint32_t> truth;  // exact k nearest passing rows
  std::vector<uint32_t> truth_shared;  // the same under the filter of the first case of its batch
};

/* AnnUdfCompile's column map: the columns programs name, in slots as first used. */
struct Columns {
  const Manifest* manifest;
  std::vector<std::string> names;
};

static int column_slot(void* arg, const char* name)
{
  auto* c = static_cast<Columns*>(arg);
  if (!c->manifest->attributes.count(name)) return -1;
  int slot = int(std::find(c->names.begin(), c->names.end(), name) - c->names.begin());
  if (slot == int(c->names.size())) {
    if (slot == ANN_UDF_MAX_COLUMNS) return -1;
    c->names.push_back(name);
  }
  return c->manifest->multivalued.count(name) ? (slot | ANN_UDF_MULTI) : slot;
}

/*
 * --analyze-h: structure of H, the graph PASSING_TWO_HOP searches, for each case's own filter (no search).
 * H's vertices are the passing rows P; u -> x when x passes and x is a graph neighbour of u, or a graph
 * neighbour of a filtered-out graph neighbour of u. Per case, one JSON line:
 *   s          pass rate
 *   p_in       fraction of the graph edges of sampled passing rows that land in P (homophily)
 *   deg1       mean number of passing graph neighbours of a passing row
 *   degH       mean out-degree in H; degH0 = fraction of passing rows with none; degH_med = median
 *   b2         mean number of distinct rows within two graph hops (E|B2|, a property of the graph)
 *   dead_H / dead_P / dead_G: among the 100 passing rows nearest to the query after its 10 nearest, the
 *              fraction with no closer out-neighbour in H, in the one-hop graph on P, and in the graph
 *   comp_*     (cases 0, 64, 128, ...): weak components of H, the largest one's share of P, rows in
 *              components of size 1, and how many of the query's 10 nearest passing rows lie in the largest.
 */
static void analyze_two_hop(const std::vector<uint32_t>& g, size_t deg, const float* vec, uint32_t dims, uint64_t rows,
                            const std::vector<Case>& cases, int samples)
{
  auto l2 = [&](const std::vector<float>& q, uint32_t r) {
    const float* x = vec + size_t(r) * dims;
    float d        = 0;
    for (uint32_t j = 0; j < dims; j++) {
      const float t = x[j] - q[j];
      d += t * t;
    }
    return d;
  };
  std::atomic<int> next{0};
  std::mutex out_mu;
  unsigned nthreads = std::max(1u, std::thread::hardware_concurrency() / 2);
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < nthreads; t++)
    pool.emplace_back([&]() {
      std::vector<uint32_t> cand, ball;
      for (int i; (i = next++) < int(cases.size());) {
        const Case& c = cases[i];
        auto passes = [&](uint32_t r) { return r < rows && ((c.words[r >> 5] >> (r & 31)) & 1u); };
        if (c.count == 0) continue;
        // H out-neighbours of u (distinct, without u), and optionally the two-hop ball of the graph.
        auto h_out = [&](uint32_t u, std::vector<uint32_t>& out, std::vector<uint32_t>* b2) {
          out.clear();
          if (b2) b2->clear();
          for (size_t j = 0; j < deg; j++) {
            const uint32_t v = g[size_t(u) * deg + j];
            if (v >= rows || v == u) continue;
            if (b2) b2->push_back(v);
            if (passes(v)) { out.push_back(v); continue; }
            for (size_t l = 0; l < deg; l++) {
              const uint32_t x = g[size_t(v) * deg + l];
              if (x >= rows || x == u) continue;
              if (b2) b2->push_back(x);
              if (passes(x)) out.push_back(x);
            }
          }
          if (b2 && !b2->empty()) {
            for (size_t j = 0; j < deg; j++) {  // the ball also holds the passing neighbours' neighbours
              const uint32_t v = g[size_t(u) * deg + j];
              if (v >= rows || v == u || !passes(v)) continue;
              for (size_t l = 0; l < deg; l++) {
                const uint32_t x = g[size_t(v) * deg + l];
                if (x < rows && x != u) b2->push_back(x);
              }
            }
            std::sort(b2->begin(), b2->end());
            b2->erase(std::unique(b2->begin(), b2->end()), b2->end());
          }
          std::sort(out.begin(), out.end());
          out.erase(std::unique(out.begin(), out.end()), out.end());
        };
        // 1. Degrees over sampled passing rows.
        uint64_t rng = 0x9e3779b97f4a7c15ull ^ uint64_t(i) * 0x2545f4914f6cdd1dull;
        auto next_rand = [&]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
        double sum_deg1 = 0, sum_degh = 0, sum_b2 = 0, edges = 0, in_edges = 0;
        int zero = 0, n = 0;
        std::vector<uint32_t> degs;
        for (int tries = 0; n < samples && tries < samples * 200000; tries++) {
          const uint32_t u = uint32_t(next_rand() % rows);
          if (!passes(u)) continue;
          n++;
          int d1 = 0;
          for (size_t j = 0; j < deg; j++) {
            const uint32_t v = g[size_t(u) * deg + j];
            if (v >= rows) continue;
            edges++;
            if (passes(v)) { d1++; in_edges++; }
          }
          h_out(u, cand, &ball);
          sum_deg1 += d1; sum_degh += cand.size(); sum_b2 += ball.size(); zero += cand.empty();
          degs.push_back(uint32_t(cand.size()));
        }
        std::sort(degs.begin(), degs.end());
        // 2. Dead ends near the query: the passing rows ranked 11..110 by distance to the query.
        std::vector<std::pair<float, uint32_t>> near;
        near.reserve(size_t(std::min<uint64_t>(c.count, 1u << 24)));
        for (size_t w = 0; w < c.words.size(); w++)
          for (uint32_t bits = c.words[w]; bits; bits &= bits - 1) {
            const uint32_t r = uint32_t(w * 32 + __builtin_ctz(bits));
            if (r < rows) near.emplace_back(l2(c.query, r), r);
          }
        const size_t keep = std::min<size_t>(110, near.size());
        std::partial_sort(near.begin(), near.begin() + keep, near.end());
        int dead_h = 0, dead_p = 0, dead_g = 0, checked = 0;
        for (size_t a = 10; a < keep; a++) {
          const uint32_t u = near[a].second;
          const float du   = near[a].first;
          checked++;
          h_out(u, cand, nullptr);
          bool h = false, pp = false, gg = false;
          for (uint32_t x : cand) if (l2(c.query, x) < du) { h = true; break; }
          for (size_t j = 0; j < deg; j++) {
            const uint32_t v = g[size_t(u) * deg + j];
            if (v >= rows) continue;
            const bool closer = l2(c.query, v) < du;
            gg |= closer;
            pp |= closer && passes(v);
          }
          dead_h += !h; dead_p += !pp; dead_g += !gg;
        }
        // 3. Components of H (one case per batch of 64).
        std::string comp;
        if (i % 64 == 0) {
          std::vector<uint32_t> parent;
          std::vector<uint32_t> members;
          members.reserve(size_t(c.count));
          for (size_t w = 0; w < c.words.size(); w++)
            for (uint32_t bits = c.words[w]; bits; bits &= bits - 1) {
              const uint32_t r = uint32_t(w * 32 + __builtin_ctz(bits));
              if (r < rows) members.push_back(r);
            }
          std::unordered_map<uint32_t, uint32_t> slot;
          slot.reserve(members.size() * 2);
          for (uint32_t k2 = 0; k2 < members.size(); k2++) slot[members[k2]] = k2;
          parent.resize(members.size());
          for (uint32_t k2 = 0; k2 < parent.size(); k2++) parent[k2] = k2;
          auto find = [&](uint32_t a) { while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
          for (uint32_t k2 = 0; k2 < members.size(); k2++) {
            h_out(members[k2], cand, nullptr);
            for (uint32_t x : cand) {
              const uint32_t a = find(k2), b = find(slot[x]);
              if (a != b) parent[a] = b;
            }
          }
          std::unordered_map<uint32_t, uint32_t> size;
          for (uint32_t k2 = 0; k2 < members.size(); k2++) size[find(k2)]++;
          uint32_t best = 0, best_root = 0, singles = 0;
          for (auto& kv : size) {
            if (kv.second > best) { best = kv.second; best_root = kv.first; }
            singles += kv.second == 1;
          }
          int top_in = 0;
          for (size_t a = 0; a < std::min<size_t>(10, keep); a++) top_in += find(slot[near[a].second]) == best_root;
          char buf[256];
          snprintf(buf, sizeof(buf), ", \"comp_count\": %zu, \"comp_largest\": %.4f, \"comp_singletons\": %.4f, \"top10_in_largest\": %d",
                   size.size(), double(best) / members.size(), double(singles) / members.size(), top_in);
          comp = buf;
        }
        std::lock_guard<std::mutex> lk(out_mu);
        printf("{\"analyze_h\": %d, \"s\": %.6f, \"count\": %llu, \"p_in\": %.4f, \"deg1\": %.2f, \"degH\": %.2f, \"degH0\": %.4f, "
               "\"degH_med\": %u, \"b2\": %.1f, \"dead_H\": %.3f, \"dead_P\": %.3f, \"dead_G\": %.3f, \"samples\": %d%s}\n",
               i, double(c.count) / rows, (unsigned long long)c.count, edges ? in_edges / edges : 0.0, n ? sum_deg1 / n : 0.0,
               n ? sum_degh / n : 0.0, n ? double(zero) / n : 0.0, degs.empty() ? 0u : degs[degs.size() / 2],
               n ? sum_b2 / n : 0.0, checked ? double(dead_h) / checked : 0.0, checked ? double(dead_p) / checked : 0.0,
               checked ? double(dead_g) / checked : 0.0, n, comp.c_str());
        fflush(stdout);
      }
    });
  for (auto& th : pool) th.join();
}

int main(int argc, char** argv)
{
  if (argc < 3) fail("usage: cagra_twohop_udf_probe INDEX_DIR CASES [options]");
  std::string dir = argv[1], cases_path = argv[2];
  int k = 300, batch = 32, reps = 3, threads = 1, itopk = 1536, iterations = 18, samplings = 2,
      width = 4, block = 64;
  bool recall = true;
  bool pitch = true;
  bool reorder = false;  // items renumbered by their smallest multi-valued value (product group)
  int scan_slack = 300;  // scanc: candidates k + k * scan_slack / 100 (the worker's --slack-ratio)
  // Above 0: each case's filter is replaced by rows drawn independently at this rate (the bitset
  // modes only: the programs no longer describe the filters).
  double random_pass = 0;
  // ivfpq modes: lists, probed lists, PQ dimension, candidates per result re-scored exactly, and
  // a file the built index is saved to and loaded from.
  int ivf_lists = 8192, ivf_probes = 50, ivf_pq_dim = 32, ivf_refine = 4;
  // ivfpq modes: bits of the search's lookup tables (32, 16 or 8) and of its internal distances (32 or 16).
  int ivf_lut = 32, ivf_internal = 32;
  // ivfpq modes: re-score on the host (cuvs::neighbors::refine with host views; the device refine
  // only takes k <= 256).
  bool ivf_refine_host = false;
  int analyze_h = 0;
  // --pool-gb N: give each thread's resources a pooled workspace of N GiB, so that the search's temporary
  // buffers do not go through cudaMalloc / cudaFree on every call (default 0: the library default).
  int pool_gb = 0;
  // --workspace-pool-gb N: only the pooled workspace (RAFT's recommended setting), not the device resource.
  int workspace_pool_gb = 0;
  bool auto_expansion   = false;
  // --ivf-int8: the ivfpq modes build, search and re-score on int8 rows and queries (one affine map
  // from the 0.5%..99.5% value range to [-128, 127]); recall is still against the exact FP32 answers.
  bool ivf_int8 = false;
  // --analyze-ivf: for each case's own filter, how its passing rows spread over the IVF lists and how
  // many lists / rows IVF-PQ must scan to cover the exact top-k, nearest-centroid order against
  // skipping lists without passing rows (exact coverage, no PQ error); then exit.
  bool analyze_ivf = false;
  // --analyze-ivf-order: reorder the rows inside each list by the filters of half the cases (rows that
  // passed the same filters become adjacent), then count, for the other half, the 32-row groups that
  // hold a passing row in the lists each query needs: today's order, the learned order, a perfect order.
  bool analyze_ivf_order = false;
  // --dump-cases DIR: write each case's query (queries.f32, [cases, dims]) and filter bitset over the
  // rows (words.u32, [cases, words]) for offline analysis, then exit.
  std::string dump_cases;
  // --ivf-sort-lists: after the IVF-PQ index is ready, sort the rows inside each list by their smallest
  // value of the first multi-valued column (codes and ids move together; ids and lists are unchanged),
  // so that rows sharing the filter attribute fill the same 32-row code groups.
  bool ivf_sort_lists = false;
  // --ivf-renumber: give every row of the IVF-PQ index the id of its position, list by list, each list
  // starting at a multiple of 32, and search with each case's bitset permuted to those ids, so that
  // the 32 rows of a code group read one bitset word (results are mapped back before the re-score).
  bool ivf_renumber = false;
  // --rand-xor-mask X: cagra::search_params::rand_xor_mask (the seed of the random entry points).
  uint64_t rand_xor_mask = 0;
  bool rand_xor_mask_set = false;  // --auto-expansion: the two-hop modes search with filter_expansion_mode::AUTO  // --analyze-h N: report the structure of the two-hop graph (N sampled rows a case) and exit
  std::string ivf_index_path;
  std::string modes = "bitset1,bitsetq,udf,udf1";
  for (int i = 3; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> int {
      if (i + 1 >= argc) fail("missing value for " + a);
      return atoi(argv[++i]);
    };
    if (a == "--k") k = next();
    else if (a == "--batch") batch = next();
    else if (a == "--reps") reps = next();
    else if (a == "--threads") threads = next();
    else if (a == "--itopk") itopk = next();
    else if (a == "--iterations") iterations = next();
    else if (a == "--samplings") samplings = next();
    else if (a == "--width") width = next();
    else if (a == "--block") block = next();
    else if (a == "--modes" && i + 1 < argc) modes = argv[++i];
    else if (a == "--no-recall") recall = false;
    else if (a == "--no-pitch") pitch = false;
    else if (a == "--reorder") reorder = true;
    else if (a == "--scan-slack") scan_slack = std::stoi(argv[++i]);
    else if (a == "--random-pass" && i + 1 < argc) random_pass = std::stod(argv[++i]);
    else if (a == "--ivf-lists") ivf_lists = next();
    else if (a == "--ivf-probes") ivf_probes = next();
    else if (a == "--ivf-pq-dim") ivf_pq_dim = next();
    else if (a == "--ivf-refine") ivf_refine = next();
    else if (a == "--ivf-lut") ivf_lut = next();
    else if (a == "--ivf-internal") ivf_internal = next();
    else if (a == "--ivf-refine-host") ivf_refine_host = true;
    else if (a == "--analyze-h") analyze_h = next();
    else if (a == "--pool-gb") pool_gb = next();
    else if (a == "--workspace-pool-gb") workspace_pool_gb = next();
    else if (a == "--auto-expansion") auto_expansion = true;
    else if (a == "--ivf-int8") ivf_int8 = true;
    else if (a == "--analyze-ivf") analyze_ivf = true;
    else if (a == "--analyze-ivf-order") analyze_ivf = analyze_ivf_order = true;
    else if (a == "--dump-cases" && i + 1 < argc) dump_cases = argv[++i];
    else if (a == "--ivf-sort-lists") ivf_sort_lists = true;
    else if (a == "--ivf-renumber") ivf_renumber = true;
    else if (a == "--rand-xor-mask" && i + 1 < argc) { rand_xor_mask = std::stoull(argv[++i], nullptr, 0); rand_xor_mask_set = true; }
    else if (a == "--ivf-index" && i + 1 < argc) ivf_index_path = argv[++i];
    else fail("unknown option " + a);
  }

  Manifest m = read_manifest(dir);
  const uint64_t rows = m.rows;
  const size_t nwords = (rows + 31) / 32;

  /* Cases and their compiled predicates. */
  std::vector<Case> cases;
  std::vector<AnnUdfStep> steps(1 << 20);
  std::vector<int32_t> values(1 << 22);
  uint32_t nsteps = 0, nvalues = 0;
  Columns columns{&m, {}};
  {
    FILE* f = fopen(cases_path.c_str(), "rb");
    uint32_t n, dims;
    if (!f || fread(&n, 4, 1, f) != 1 || fread(&dims, 4, 1, f) != 1) fail("cannot read " + cases_path);
    if (dims != m.dims) fail("cases have " + std::to_string(dims) + " dimensions, the index " +
                             std::to_string(m.dims));
    for (uint32_t i = 0; i < n; i++) {
      Case c;
      uint32_t ck, len;
      if (fread(&ck, 4, 1, f) != 1 || fread(&len, 4, 1, f) != 1) fail("short cases file");
      c.program.resize(len);
      c.query.resize(dims);
      if (fread(c.program.data(), 1, len, f) != len ||
          fread(c.query.data(), sizeof(float), dims, f) != dims)
        fail("short cases file");
      c.first_step = nsteps;
      int status  = AnnUdfCompile(c.program.data(), len, column_slot, &columns, steps.data(),
                                 uint32_t(steps.size()), &nsteps, values.data(),
                                 uint32_t(values.size()), &nvalues);
      if (status != ANN_FILTER_OK) fail("case " + std::to_string(i) + ": compile status " +
                                        std::to_string(status));
      c.nsteps = nsteps - c.first_step;
      cases.push_back(std::move(c));
    }
    fclose(f);
  }
  const int ncases = int(cases.size());
  if (ncases == 0) fail("no cases");

  /* Attribute columns on the host (mapped) and the device. */
  std::vector<const int32_t*> host_columns(ANN_UDF_MAX_COLUMNS, nullptr);
  std::vector<const uint32_t*> host_offsets(ANN_UDF_MAX_COLUMNS, nullptr);
  AnnUdfParams params_template;
  memset(&params_template, 0, sizeof(params_template));
  size_t column_bytes = 0;
  for (size_t c = 0; c < columns.names.size(); c++) {
    const std::string& name = columns.names[c];
    size_t size;
    host_columns[c] =
      static_cast<const int32_t*>(map_file(dir + "/" + m.attributes[name], &size));
    void* d;
    CUDA(cudaMalloc(&d, size));
    CUDA(cudaMemcpy(d, host_columns[c], size, cudaMemcpyHostToDevice));
    params_template.columns[c] = static_cast<const int32_t*>(d);
    column_bytes += size;
    if (m.multivalued.count(name)) {
      host_offsets[c] =
        static_cast<const uint32_t*>(map_file(dir + "/" + m.multivalued[name], &size));
      if (size != (rows + 1) * sizeof(uint32_t)) fail(name + ": offsets are not rows + 1 bounds");
      CUDA(cudaMalloc(&d, size));
      CUDA(cudaMemcpy(d, host_offsets[c], size, cudaMemcpyHostToDevice));
      params_template.offsets[c] = static_cast<const uint32_t*>(d);
      column_bytes += size;
    }
  }

  /*
   * --reorder: items renumbered by the smallest value of the first multi-valued column (NULL and empty
   * arrays last): order[new] = old, inv[old] = new. Columns are permuted here, before any filter is
   * evaluated; the rows and the graph after the index loads.
   */
  std::vector<uint32_t> order, inv;
  std::vector<std::vector<int32_t>> owned_columns(ANN_UDF_MAX_COLUMNS);
  std::vector<std::vector<uint32_t>> owned_offsets(ANN_UDF_MAX_COLUMNS);
  if (reorder) {
    int mc = -1;
    for (size_t c = 0; c < columns.names.size() && mc < 0; c++)
      if (host_offsets[c]) mc = int(c);
    if (mc < 0) fail("--reorder needs a multi-valued column");
    std::vector<int64_t> key(rows);
    for (uint64_t r = 0; r < rows; r++) {
      int64_t k = INT64_MAX;
      for (uint32_t i = host_offsets[mc][r]; i < host_offsets[mc][r + 1]; i++)
        if (host_columns[mc][i] != ANN_FILTER_NULL_VALUE) k = std::min<int64_t>(k, host_columns[mc][i]);
      key[r] = k;
    }
    order.resize(rows);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return key[a] < key[b]; });
    inv.resize(rows);
    for (uint64_t i = 0; i < rows; i++)
      inv[order[i]] = uint32_t(i);
    for (size_t c = 0; c < columns.names.size(); c++) {
      if (host_offsets[c]) {
        auto& o = owned_offsets[c];
        auto& e = owned_columns[c];
        o.assign(rows + 1, 0);
        for (uint64_t i = 0; i < rows; i++)
          o[i + 1] = o[i] + (host_offsets[c][order[i] + 1] - host_offsets[c][order[i]]);
        e.resize(o[rows]);
        for (uint64_t i = 0; i < rows; i++)
          std::copy(host_columns[c] + host_offsets[c][order[i]], host_columns[c] + host_offsets[c][order[i] + 1],
                    e.begin() + o[i]);
        CUDA(cudaMemcpy(const_cast<uint32_t*>(params_template.offsets[c]), o.data(), o.size() * 4,
                        cudaMemcpyHostToDevice));
        host_offsets[c] = o.data();
      } else {
        auto& e = owned_columns[c];
        e.resize(rows);
        for (uint64_t i = 0; i < rows; i++)
          e[i] = host_columns[c][order[i]];
      }
      CUDA(cudaMemcpy(const_cast<int32_t*>(params_template.columns[c]), owned_columns[c].data(),
                      owned_columns[c].size() * 4, cudaMemcpyHostToDevice));
      host_columns[c] = owned_columns[c].data();
    }
  }

  /* Modes whose batches share their first case's filter need exact answers under that filter. */
  auto is_shared = [](const std::string& mode) {
    return mode == "stockb" || mode == "bitsetb" || mode == "linkedb" || mode == "ivfpqb";
  };
  const bool shared_truth = modes.find("stockb") != std::string::npos ||
                            modes.find("bitsetb") != std::string::npos ||
                            modes.find("linkedb") != std::string::npos ||
                            modes.find("ivfpqb") != std::string::npos;

  /* Each case's passing rows on the host, from the same interpreter the UDF runs. */
  double t0 = now_ms();
  {
    std::vector<std::thread> pool;
    std::atomic<int> next{0};
    unsigned nthreads = std::max(1u, std::thread::hardware_concurrency() / 2);
    for (unsigned t = 0; t < nthreads; t++)
      pool.emplace_back([&]() {
        for (int i; (i = next++) < ncases;) {
          Case& c = cases[i];
          c.words.assign(nwords, 0);
          if (random_pass > 0) {
            // xorshift64, one stream a case.
            uint64_t state           = 0x9E3779B97F4A7C15ull * uint64_t(i + 1);
            // 2^64 does not fit: a rate of 1 or more passes every row.
            const uint64_t threshold =
              random_pass >= 1.0 ? ~uint64_t(0) : uint64_t(random_pass * 18446744073709551615.0);
            for (uint64_t r = 0; r < rows; r++) {
              state ^= state << 13;
              state ^= state >> 7;
              state ^= state << 17;
              if (state <= threshold) {
                c.words[r / 32] |= 1u << (r % 32);
                c.count++;
              }
            }
            continue;
          }
          for (uint64_t r = 0; r < rows; r++)
            if (AnnUdfMatchMulti(steps.data() + c.first_step, c.nsteps, values.data(),
                                 host_columns.data(), host_offsets.data(), r)) {
              c.words[r / 32] |= 1u << (r % 32);
              c.count++;
            }
        }
      });
    for (auto& t : pool)
      t.join();
  }
  double eval_ms = now_ms() - t0;

  /* Exact answers from vectors.f32 over each case's passing rows. */
  if (recall) {
    size_t size;
    auto* vectors = static_cast<const float*>(map_file(dir + "/vectors.f32", &size));
    if (size != rows * m.dims * sizeof(float)) fail("vectors.f32 size");
    std::vector<std::thread> pool;
    std::atomic<int> next{0};
    unsigned nthreads = std::max(1u, std::thread::hardware_concurrency() / 2);
    for (unsigned t = 0; t < nthreads; t++)
      pool.emplace_back([&]() {
        std::vector<std::pair<float, uint32_t>> d;
        for (int i; (i = next++) < ncases;) {
          Case& c = cases[i];
          // The case's own filter, then (for the shared-filter modes) its batch's first filter.
          for (int shared = 0; shared < (shared_truth ? 2 : 1); shared++) {
            const std::vector<uint32_t>& words = shared ? cases[(i / batch) * batch].words : c.words;
            d.clear();
            for (size_t w = 0; w < nwords; w++)
              for (uint32_t bits = words[w]; bits; bits &= bits - 1) {
                uint32_t r   = uint32_t(w * 32 + __builtin_ctz(bits));
                const float* x = vectors + size_t(reorder ? order[r] : r) * m.dims;
                float s        = 0;
                for (uint32_t j = 0; j < m.dims; j++) {
                  float t = x[j] - c.query[j];
                  s += t * t;
                }
                d.emplace_back(s, r);
              }
            size_t keep = std::min<size_t>(k, d.size());
            std::partial_sort(d.begin(), d.begin() + keep, d.end());
            auto& out = shared ? c.truth_shared : c.truth;
            out.resize(keep);
            for (size_t j = 0; j < keep; j++)
              out[j] = d[j].second;
          }
        }
      });
    for (auto& t : pool)
      t.join();
  }
  double truth_ms = now_ms() - t0 - eval_ms;
  if (!dump_cases.empty()) {
    std::ofstream qf(dump_cases + "/queries.f32", std::ios::binary), wf(dump_cases + "/words.u32", std::ios::binary);
    for (const Case& c : cases) {
      qf.write(reinterpret_cast<const char*>(c.query.data()), m.dims * sizeof(float));
      wf.write(reinterpret_cast<const char*>(c.words.data()), c.words.size() * sizeof(uint32_t));
    }
    printf("{\"dumped_cases\": %d, \"dims\": %u, \"words\": %zu}\n", ncases, m.dims, cases.empty() ? 0 : cases[0].words.size());
    return 0;
  }
  {
    // How many of the cases' filters are distinct (the workload's unique filters among these queries).
    std::set<std::vector<uint32_t>> distinct;
    for (const Case& c : cases) distinct.insert(c.words);
    printf("{\"distinct_filters\": %zu, \"cases\": %d}\n", distinct.size(), ncases);
  }

  /* The index, its FP16 dataset kept alive beside it. */
  raft::device_resources load_res;
  cagra::device_padded_index<half> index(load_res);
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<half, int64_t>> dataset;
  t0 = now_ms();
  cagra::deserialize(load_res, dir + "/" + m.shard, &index, &dataset);
  load_res.sync_stream();
  double load_ms = now_ms() - t0;
  if (uint64_t(index.size()) != rows || uint32_t(index.dim()) != m.dims) fail("index shape");
  if (analyze_h > 0) {
    if (reorder) fail("--analyze-h does not take --reorder");
    const size_t deg = size_t(index.graph().extent(1));
    std::vector<uint32_t> hg(size_t(rows) * deg);
    CUDA(cudaMemcpy(hg.data(), index.graph().data_handle(), hg.size() * 4, cudaMemcpyDeviceToHost));
    size_t vsz;
    const auto* vec = static_cast<const float*>(map_file(dir + "/vectors.f32", &vsz));
    if (vsz != rows * m.dims * sizeof(float)) fail("vectors.f32 size");
    analyze_two_hop(hg, deg, vec, m.dims, rows, cases, analyze_h);
    return 0;
  }

  /*
   * cuVS searches one query per launch when the dataset's row pitch exceeds its dimensions (65
   * dimensions at pitch 72 here). As the worker does (--cagra-pitch), view the padded rows as
   * pitch-wide vectors with the padding zeroed and pad the queries: same distances, one launch a
   * batch. --no-pitch keeps the per-query launches.
   */
  const uint32_t stride = uint32_t(dataset->view().data_const_view().stride(0));
  const uint32_t qdim   = pitch ? stride : m.dims;
  std::optional<cagra::device_padded_index<half>> wide;
  if (pitch && stride > m.dims) {
    half* rows_ptr = const_cast<half*>(dataset->view().data_const_view().data_handle());
    CUDA(cudaMemset2D(rows_ptr + m.dims, size_t(stride) * sizeof(half), 0,
                      size_t(stride - m.dims) * sizeof(half), rows));
    auto wide_rows = cuvs::neighbors::make_device_padded_dataset_view(
      load_res, raft::make_device_matrix_view<const half, int64_t>(rows_ptr, int64_t(rows), int64_t(stride)));
    wide.emplace(load_res, cuvs::distance::DistanceType::L2Expanded, wide_rows,
                 raft::make_const_mdspan(index.graph()));
    load_res.sync_stream();
  }
  std::optional<cagra::device_padded_index<half>> permuted;
  half* permuted_rows       = nullptr;
  uint32_t* permuted_graph  = nullptr;
  if (reorder) {
    const auto rv   = dataset->view().data_const_view();
    const size_t st = size_t(rv.stride(0));
    std::vector<half> hrows(size_t(rows) * st), prow(size_t(rows) * st);
    CUDA(cudaMemcpy(hrows.data(), rv.data_handle(), hrows.size() * sizeof(half), cudaMemcpyDeviceToHost));
    const size_t deg = size_t(index.graph().extent(1));
    std::vector<uint32_t> hg(size_t(rows) * deg), pg(size_t(rows) * deg);
    CUDA(cudaMemcpy(hg.data(), index.graph().data_handle(), hg.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<std::thread> pool;
    unsigned nthreads = std::max(1u, std::thread::hardware_concurrency() / 2);
    for (unsigned t = 0; t < nthreads; t++)
      pool.emplace_back([&, t]() {
        for (uint64_t i = t; i < rows; i += nthreads) {
          std::copy(hrows.begin() + size_t(order[i]) * st, hrows.begin() + size_t(order[i] + 1) * st,
                    prow.begin() + i * st);
          for (size_t j = 0; j < deg; j++) {
            uint32_t nb      = hg[size_t(order[i]) * deg + j];
            pg[i * deg + j] = nb < rows ? inv[nb] : nb;
          }
        }
      });
    for (auto& th : pool)
      th.join();
    CUDA(cudaMalloc(&permuted_rows, prow.size() * sizeof(half)));
    CUDA(cudaMemcpy(permuted_rows, prow.data(), prow.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA(cudaMalloc(&permuted_graph, pg.size() * 4));
    CUDA(cudaMemcpy(permuted_graph, pg.data(), pg.size() * 4, cudaMemcpyHostToDevice));
    const uint32_t width = pitch ? uint32_t(st) : m.dims;
    if (pitch && st > m.dims)
      CUDA(cudaMemset2D(permuted_rows + m.dims, st * sizeof(half), 0, (st - m.dims) * sizeof(half), rows));
    auto prows = cuvs::neighbors::make_device_padded_dataset_view(
      load_res, raft::make_device_matrix_view<const half, int64_t>(permuted_rows, int64_t(rows), int64_t(st)));
    if (width != st) fail("--reorder needs the pitch view (drop --no-pitch)");
    permuted.emplace(load_res, cuvs::distance::DistanceType::L2Expanded, prows,
                     raft::make_device_matrix_view<const uint32_t, int64_t>(permuted_graph, int64_t(rows), int64_t(deg)));
    load_res.sync_stream();
  }
  auto& searched = permuted ? *permuted : (wide ? *wide : index);
#ifdef PROBE_HAS_FP8_SCAN
  std::optional<cuvs::preprocessing::quantize::fp8::quantizer> scan_codes;
  if (modes.find("scanc") != std::string::npos) {
    namespace fp8 = cuvs::preprocessing::quantize::fp8;
    const auto rv = searched.dataset().data_const_view();
    const uint32_t d = uint32_t(searched.dim());
    scan_codes.emplace(load_res, int64_t(rows), d);
    if (uint32_t(rv.stride(0)) != d) fail("scanc needs dense rows: keep the pitch view");
    fp8::transform(load_res,
                   raft::make_device_matrix_view<const half, int64_t>(rv.data_handle(), int64_t(rows), int64_t(d)),
                   *scan_codes);
    searched.update_scan_codes(load_res, scan_codes->view());
    load_res.sync_stream();
  }
#else
  if (modes.find("scanc") != std::string::npos) fail("scanc needs a cuVS with the FP8 filtered scan");
#endif

  /* Device copies: queries as FP16, bitsets, compiled predicates. */
  std::vector<half> hq(size_t(ncases) * qdim, __float2half(0.0f));
  for (int i = 0; i < ncases; i++)
    for (uint32_t j = 0; j < m.dims; j++)
      hq[size_t(i) * qdim + j] = __float2half(cases[i].query[j]);
  half* dq;
  CUDA(cudaMalloc(&dq, hq.size() * sizeof(half)));
  CUDA(cudaMemcpy(dq, hq.data(), hq.size() * sizeof(half), cudaMemcpyHostToDevice));
  std::vector<uint32_t*> dbits(ncases);
  for (int i = 0; i < ncases; i++) {
    CUDA(cudaMalloc(&dbits[i], nwords * sizeof(uint32_t)));
    CUDA(cudaMemcpy(dbits[i], cases[i].words.data(), nwords * 4, cudaMemcpyHostToDevice));
  }
  /* ivfpq modes: the IVF-PQ index, the queries and (for the exact re-score) the rows as FP32. */
  namespace ivf_pq = cuvs::neighbors::ivf_pq;
  std::optional<ivf_pq::index<int64_t>> ivf;
  uint64_t ivf_new_rows = 0;            // --ivf-renumber: size of the new id space,
  std::vector<int64_t> ivf_new2old;     // the row of each new id (-1 for the padding of a list),
  std::vector<uint32_t*> ivf_dbits;     // and each case's bitset over the new ids
  float* dqf  = nullptr;
  float* dvec = nullptr;
  const float* hvec = nullptr;  // the rows and the queries as FP32 on the host (--ivf-refine-host)
  std::vector<float> hqf;
  std::vector<int8_t> h8, hq8;  // --ivf-int8: rows and queries as int8 on the host
  int8_t* dvec8 = nullptr;      // and on the device
  int8_t* dq8   = nullptr;
  // Candidates ivf_pq::search returned that do not pass the filter, and the queries all of whose
  // candidates are such rows.
  std::atomic<uint64_t> ivf_filtered_out_candidates{0};
  std::vector<char> ivf_no_candidate(size_t(ncases), 0);
  double ivf_build_ms = 0;
  if (modes.find("ivfpq") != std::string::npos || analyze_ivf) {
    if (reorder) fail("the ivfpq modes do not take --reorder");
    raft::device_resources ivf_res;
    size_t vsize;
    auto* vectors = static_cast<const float*>(map_file(dir + "/vectors.f32", &vsize));
    if (vsize != rows * m.dims * sizeof(float)) fail("vectors.f32 size");
    hvec = vectors;
    hqf.assign(size_t(ncases) * m.dims, 0.0f);
    for (int i = 0; i < ncases; i++)
      std::copy(cases[i].query.begin(), cases[i].query.begin() + m.dims, hqf.begin() + size_t(i) * m.dims);
    CUDA(cudaMalloc(&dqf, hqf.size() * sizeof(float)));
    CUDA(cudaMemcpy(dqf, hqf.data(), hqf.size() * sizeof(float), cudaMemcpyHostToDevice));
    if (ivf_int8) {
      // One affine map for rows and queries, from a sample of the values: L2 distances keep their order
      // up to rounding.
      std::vector<float> sample;
      const uint64_t step = std::max<uint64_t>(1, rows / 100000);
      for (uint64_t r = 0; r < rows; r += step)
        sample.insert(sample.end(), vectors + r * m.dims, vectors + (r + 1) * m.dims);
      auto at = [&](double q) {
        std::nth_element(sample.begin(), sample.begin() + size_t(q * (sample.size() - 1)), sample.end());
        return sample[size_t(q * (sample.size() - 1))];
      };
      const float lo = at(0.005), hi = at(0.995);
      auto q8 = [&](float x) {
        const float v = std::nearbyint((x - lo) / (hi - lo) * 255.0f - 128.0f);
        return int8_t(std::min(127.0f, std::max(-128.0f, v)));
      };
      h8.resize(size_t(rows) * m.dims);
      {
        std::vector<std::thread> pool;
        const unsigned nt = std::max(1u, std::thread::hardware_concurrency() / 2);
        for (unsigned t = 0; t < nt; t++)
          pool.emplace_back([&, t]() {
            for (size_t i = t; i < h8.size(); i += nt) h8[i] = q8(vectors[i]);
          });
        for (auto& th : pool) th.join();
      }
      hq8.resize(hqf.size());
      for (size_t i = 0; i < hqf.size(); i++) hq8[i] = q8(hqf[i]);
      CUDA(cudaMalloc(&dvec8, h8.size()));
      CUDA(cudaMemcpy(dvec8, h8.data(), h8.size(), cudaMemcpyHostToDevice));
      CUDA(cudaMalloc(&dq8, hq8.size()));
      CUDA(cudaMemcpy(dq8, hq8.data(), hq8.size(), cudaMemcpyHostToDevice));
      printf("{\"ivf_int8\": true, \"lo\": %.5f, \"hi\": %.5f}\n", lo, hi);
    }
    if (ivf_refine > 1 && !ivf_refine_host && !ivf_int8) {
      CUDA(cudaMalloc(&dvec, vsize));
      CUDA(cudaMemcpy(dvec, vectors, vsize, cudaMemcpyHostToDevice));
    }
    double b0 = now_ms();
    std::ifstream saved(ivf_index_path);
    if (!ivf_index_path.empty() && saved.good()) {
      ivf.emplace(ivf_res);
      ivf_pq::deserialize(ivf_res, ivf_index_path, &*ivf);
    } else {
      ivf_pq::index_params ip;
      ip.metric                   = cuvs::distance::DistanceType::L2Expanded;
      ip.n_lists                  = uint32_t(ivf_lists);
      ip.pq_dim                   = uint32_t(ivf_pq_dim);
      ip.pq_bits                  = 8;
      ip.kmeans_trainset_fraction = 0.1;
      if (ivf_int8)
        ivf.emplace(ivf_pq::build(
          ivf_res, ip, raft::make_device_matrix_view<const int8_t, int64_t>(dvec8, int64_t(rows), int64_t(m.dims))));
      else
        ivf.emplace(ivf_pq::build(
          ivf_res, ip, raft::make_host_matrix_view<const float, int64_t>(vectors, int64_t(rows), int64_t(m.dims))));
      ivf_res.sync_stream();
      ivf_build_ms = now_ms() - b0;
      if (!ivf_index_path.empty()) ivf_pq::serialize(ivf_res, ivf_index_path, *ivf);
    }
    ivf_res.sync_stream();
    if (ivf_sort_lists) {
      const double s0 = now_ms();
      int mc = -1;
      for (size_t cc = 0; cc < columns.names.size() && mc < 0; cc++)
        if (host_offsets[cc]) mc = int(cc);
      if (mc < 0) fail("--ivf-sort-lists needs a multi-valued column");
      std::vector<int64_t> key(rows);
      for (uint64_t r = 0; r < rows; r++) {
        int64_t kk = INT64_MAX;
        for (uint32_t a = host_offsets[mc][r]; a < host_offsets[mc][r + 1]; a++)
          if (host_columns[mc][a] != ANN_FILTER_NULL_VALUE) kk = std::min<int64_t>(kk, host_columns[mc][a]);
        key[r] = kk;
      }
      const uint32_t nl = ivf->n_lists(), pqd = ivf->pq_dim();
      std::vector<uint32_t> sizes(nl);
      std::vector<int64_t*> ptrs(nl);
      CUDA(cudaMemcpy(sizes.data(), ivf->list_sizes().data_handle(), nl * 4, cudaMemcpyDeviceToHost));
      CUDA(cudaMemcpy(ptrs.data(), ivf->inds_ptrs().data_handle(), nl * sizeof(int64_t*), cudaMemcpyDeviceToHost));
      const uint32_t maxn = *std::max_element(sizes.begin(), sizes.end());
      auto dcodes = raft::make_device_matrix<uint8_t, uint32_t>(ivf_res, maxn, pqd);
      std::vector<uint8_t> hc(size_t(maxn) * pqd), pc(size_t(maxn) * pqd);
      std::vector<int64_t> ids(maxn), pids(maxn);
      std::vector<uint32_t> perm(maxn);
      uint64_t moved = 0;
      for (uint32_t l = 0; l < nl; l++) {
        const uint32_t n = sizes[l];
        if (n < 2) continue;
        CUDA(cudaMemcpy(ids.data(), ptrs[l], n * sizeof(int64_t), cudaMemcpyDeviceToHost));
        perm.resize(n);
        std::iota(perm.begin(), perm.end(), 0u);
        std::stable_sort(perm.begin(), perm.end(), [&](uint32_t a, uint32_t b) { return key[ids[a]] < key[ids[b]]; });
        auto view = raft::make_device_matrix_view<uint8_t, uint32_t>(dcodes.data_handle(), n, pqd);
        ivf_pq::helpers::codepacker::unpack_list_data(ivf_res, *ivf, view, l, 0);
        CUDA(cudaMemcpyAsync(hc.data(), dcodes.data_handle(), size_t(n) * pqd, cudaMemcpyDeviceToHost,
                             raft::resource::get_cuda_stream(ivf_res).get()));
        ivf_res.sync_stream();
        for (uint32_t a = 0; a < n; a++) {
          std::copy(hc.begin() + size_t(perm[a]) * pqd, hc.begin() + size_t(perm[a] + 1) * pqd, pc.begin() + size_t(a) * pqd);
          pids[a] = ids[perm[a]];
          moved += perm[a] != a;
        }
        CUDA(cudaMemcpy(dcodes.data_handle(), pc.data(), size_t(n) * pqd, cudaMemcpyHostToDevice));
        ivf_pq::helpers::codepacker::pack_list_data(
          ivf_res, &*ivf, raft::make_device_matrix_view<const uint8_t, uint32_t>(dcodes.data_handle(), n, pqd), l, 0);
        ivf_res.sync_stream();
        CUDA(cudaMemcpy(ptrs[l], pids.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice));
      }
      printf("{\"ivf_sort_lists\": true, \"rows_moved\": %llu, \"ms\": %.0f}\n", (unsigned long long)moved, now_ms() - s0);
    }
    if (ivf_renumber) {
      if (ivf_refine < 2) fail("--ivf-renumber needs --ivf-refine 2 or more (ids are mapped back on the host)");
      const double s0 = now_ms();
      const uint32_t nl = ivf->n_lists();
      std::vector<uint32_t> sizes(nl);
      std::vector<int64_t*> ptrs(nl);
      CUDA(cudaMemcpy(sizes.data(), ivf->list_sizes().data_handle(), nl * 4, cudaMemcpyDeviceToHost));
      CUDA(cudaMemcpy(ptrs.data(), ivf->inds_ptrs().data_handle(), nl * sizeof(int64_t*), cudaMemcpyDeviceToHost));
      uint64_t total = 0;
      for (uint32_t l = 0; l < nl; l++) total += (uint64_t(sizes[l]) + 31) / 32 * 32;
      ivf_new_rows = total;
      ivf_new2old.assign(total, -1);
      std::vector<int64_t> ids;
      uint64_t base = 0;
      for (uint32_t l = 0; l < nl; l++) {
        const uint32_t n = sizes[l];
        ids.resize(n);
        if (n) CUDA(cudaMemcpy(ids.data(), ptrs[l], n * sizeof(int64_t), cudaMemcpyDeviceToHost));
        for (uint32_t a = 0; a < n; a++) {
          ivf_new2old[base + a] = ids[a];
          ids[a]                = int64_t(base + a);
        }
        if (n) CUDA(cudaMemcpy(ptrs[l], ids.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice));
        base += (uint64_t(n) + 31) / 32 * 32;
      }
      const size_t nw = (total + 31) / 32;
      ivf_dbits.assign(size_t(ncases), nullptr);
      std::vector<std::thread> workers;
      const int nth = 32;
      std::vector<std::vector<uint32_t>> scratch(nth, std::vector<uint32_t>(nw));
      for (int w = 0; w < nth; w++)
        workers.emplace_back([&, w] {
          for (int c = w; c < ncases; c += nth) {
            auto& out = scratch[w];
            std::fill(out.begin(), out.end(), 0u);
            const uint32_t* in = cases[c].words.data();
            for (uint64_t x = 0; x < total; x++) {
              const int64_t o = ivf_new2old[x];
              if (o >= 0 && ((in[o >> 5] >> (o & 31)) & 1u)) out[x >> 5] |= 1u << (x & 31);
            }
            CUDA(cudaMalloc(&ivf_dbits[c], nw * 4));
            CUDA(cudaMemcpy(ivf_dbits[c], out.data(), nw * 4, cudaMemcpyHostToDevice));
          }
        });
      for (auto& w : workers) w.join();
      printf("{\"ivf_renumber\": true, \"id_space\": %llu, \"ms\": %.0f}\n", (unsigned long long)total, now_ms() - s0);
    }
    printf("{\"ivfpq_index\": true, \"lists\": %d, \"pq_dim\": %d, \"build_ms\": %.0f, \"ready_ms\": %.0f}\n",
           ivf_lists, ivf_pq_dim, ivf_build_ms, now_ms() - b0);
    fflush(stdout);
    if (analyze_ivf) {
      // List membership (row -> list) and centroids on the host.
      const uint32_t nl = ivf->n_lists();
      std::vector<uint32_t> sizes(nl);
      std::vector<int64_t*> ptrs(nl);
      CUDA(cudaMemcpy(sizes.data(), ivf->list_sizes().data_handle(), nl * 4, cudaMemcpyDeviceToHost));
      CUDA(cudaMemcpy(ptrs.data(), ivf->inds_ptrs().data_handle(), nl * sizeof(int64_t*), cudaMemcpyDeviceToHost));
      std::vector<uint16_t> list_of(rows, 0xffff);
      std::vector<int64_t> ids;
      for (uint32_t l = 0; l < nl; l++) {
        ids.resize(sizes[l]);
        if (sizes[l]) CUDA(cudaMemcpy(ids.data(), ptrs[l], sizes[l] * 8, cudaMemcpyDeviceToHost));
        for (int64_t r : ids) if (r >= 0 && uint64_t(r) < rows) list_of[r] = uint16_t(l);
      }
      // Rows of each list in today's order, and each row's position in its list.
      std::vector<std::vector<uint32_t>> list_rows(nl);
      std::vector<uint32_t> pos_now(rows, 0);
      for (uint32_t l = 0; l < nl; l++) {
        ids.resize(sizes[l]);
        if (sizes[l]) CUDA(cudaMemcpy(ids.data(), ptrs[l], sizes[l] * 8, cudaMemcpyDeviceToHost));
        for (uint32_t a = 0; a < sizes[l]; a++)
          if (ids[a] >= 0 && uint64_t(ids[a]) < rows) { list_rows[l].push_back(uint32_t(ids[a])); pos_now[ids[a]] = a; }
      }
      // Learned order: the distinct filters of the even-numbered cases (up to 128) are the past traffic.
      std::vector<uint32_t> pos_learn, pos_attr;
      std::vector<const std::vector<uint32_t>*> train;
      if (analyze_ivf_order) {
        for (int i = 0; i < ncases && train.size() < 128; i += 2) {
          bool dup = false;
          for (auto* t : train) dup |= (*t == cases[i].words);
          if (!dup && cases[i].count) train.push_back(&cases[i].words);
        }
        std::vector<std::array<uint64_t, 2>> sig(rows, {0, 0});
        for (size_t t = 0; t < train.size(); t++)
          for (size_t w = 0; w < train[t]->size(); w++)
            for (uint32_t bits = (*train[t])[w]; bits; bits &= bits - 1) {
              const uint32_t r = uint32_t(w * 32 + __builtin_ctz(bits));
              if (r < rows) sig[r][t / 64] |= 1ull << (63 - t % 64);  // filter 0 most significant
            }
        pos_learn.assign(rows, 0);
        for (uint32_t l = 0; l < nl; l++) {
          std::vector<uint32_t> v = list_rows[l];
          // Rows that passed the same past filters together (lexicographic on the signature, filters
          // seen first lead), rows that passed none last; ties keep today's order.
          std::stable_sort(v.begin(), v.end(), [&](uint32_t a, uint32_t b) {
            const bool za = !(sig[a][0] | sig[a][1]), zb = !(sig[b][0] | sig[b][1]);
            if (za != zb) return zb;
            if (sig[a][0] != sig[b][0]) return sig[a][0] > sig[b][0];
            return sig[a][1] > sig[b][1];
          });
          for (uint32_t a = 0; a < v.size(); a++) pos_learn[v[a]] = a;
        }
        // Learning-free comparison: each list sorted by the row's smallest value of the first
        // multi-valued column (the attribute the filters read).
        int mc = -1;
        for (size_t cc = 0; cc < columns.names.size() && mc < 0; cc++)
          if (host_offsets[cc]) mc = int(cc);
        pos_attr.assign(rows, 0);
        if (mc >= 0) {
          std::vector<int64_t> key(rows);
          for (uint64_t r = 0; r < rows; r++) {
            int64_t kk = INT64_MAX;
            for (uint32_t a = host_offsets[mc][r]; a < host_offsets[mc][r + 1]; a++)
              if (host_columns[mc][a] != ANN_FILTER_NULL_VALUE) kk = std::min<int64_t>(kk, host_columns[mc][a]);
            key[r] = kk;
          }
          for (uint32_t l = 0; l < nl; l++) {
            std::vector<uint32_t> v = list_rows[l];
            std::stable_sort(v.begin(), v.end(), [&](uint32_t a, uint32_t b) { return key[a] < key[b]; });
            for (uint32_t a = 0; a < v.size(); a++) pos_attr[v[a]] = a;
          }
        }
        printf("{\"ivf_order_train_filters\": %zu, \"attribute_column\": %d}\n", train.size(), mc);
        fflush(stdout);
      }
      const uint32_t dext = ivf->centers().extent(1);
      std::vector<float> cen(size_t(nl) * dext);
      CUDA(cudaMemcpy(cen.data(), ivf->centers().data_handle(), cen.size() * 4, cudaMemcpyDeviceToHost));
      std::atomic<int> next{0};
      std::mutex mu;
      std::vector<std::thread> pool;
      const unsigned nt = std::max(1u, std::thread::hardware_concurrency() / 2);
      for (unsigned t = 0; t < nt; t++)
        pool.emplace_back([&]() {
          for (int i; (i = next++) < ncases;) {
            const Case& c = cases[i];
            if (c.count < uint64_t(k)) continue;
            const float* q = hqf.data() + size_t(i) * m.dims;
            // Passing rows per list, and the exact top-k passing rows.
            std::vector<uint32_t> pass(nl, 0);
            std::vector<std::pair<float, uint32_t>> near;
            for (size_t w = 0; w < c.words.size(); w++)
              for (uint32_t bits = c.words[w]; bits; bits &= bits - 1) {
                const uint32_t r = uint32_t(w * 32 + __builtin_ctz(bits));
                if (r >= rows) continue;
                pass[list_of[r]]++;
                const float* x = vectors + size_t(r) * m.dims;
                float d = 0;
                for (uint32_t j = 0; j < m.dims; j++) { const float e = x[j] - q[j]; d += e * e; }
                near.emplace_back(d, r);
              }
            std::partial_sort(near.begin(), near.begin() + k, near.end());
            // Lists in nearest-centroid order.
            std::vector<std::pair<float, uint32_t>> order(nl);
            for (uint32_t l = 0; l < nl; l++) {
              float d = 0;
              for (uint32_t j = 0; j < m.dims; j++) { const float e = cen[size_t(l) * dext + j] - q[j]; d += e * e; }
              order[l] = {d, l};
            }
            std::sort(order.begin(), order.end());
            std::vector<uint32_t> rank(nl);
            for (uint32_t a = 0; a < nl; a++) rank[order[a].second] = a;
            std::vector<uint32_t> top_rank;
            for (int a = 0; a < k; a++) top_rank.push_back(rank[list_of[near[a].second]]);
            std::sort(top_rank.begin(), top_rank.end());
            // Concentration of the passing rows over the lists.
            std::vector<uint32_t> sorted_pass = pass;
            std::sort(sorted_pass.rbegin(), sorted_pass.rend());
            uint64_t tot = 0, top1 = 0, top10 = 0;
            uint32_t nonempty = 0;
            for (uint32_t a = 0; a < nl; a++) {
              tot += sorted_pass[a]; nonempty += sorted_pass[a] > 0;
              if (a < nl / 100) top1 += sorted_pass[a];
              if (a < nl / 10) top10 += sorted_pass[a];
            }
            // Cost of covering `need` of the top-k: all lists up to the needed rank (nearest-centroid
            // order), the non-empty ones among them (skip empty lists), and their passing rows.
            auto cost = [&](int need) {
              const uint32_t last = top_rank[need - 1];
              uint64_t lists_std = last + 1, lists_skip = 0, rows_std = 0, rows_skip = 0, rows_pass = 0;
              for (uint32_t a = 0; a <= last; a++) {
                const uint32_t l = order[a].second;
                rows_std += sizes[l];
                if (pass[l]) { lists_skip++; rows_skip += sizes[l]; rows_pass += pass[l]; }
              }
              char b[256];
              snprintf(b, sizeof(b), "{\"lists\": %llu, \"lists_nonempty\": %llu, \"rows\": %llu, \"rows_nonempty\": %llu, \"passing_rows\": %llu}",
                       (unsigned long long)lists_std, (unsigned long long)lists_skip, (unsigned long long)rows_std,
                       (unsigned long long)rows_skip, (unsigned long long)rows_pass);
              return std::string(b);
            };
            if (analyze_ivf_order) {
              if (i % 2 == 0) continue;  // the even cases trained the order
              bool seen = false;
              for (auto* t : train) seen |= (*t == c.words);
              // Rows read when the scan skips 32-row groups without a passing row, over the lists the
              // query needs to cover its exact top-k (nearest-centroid order), and over 400 lists.
              auto groups = [&](uint32_t last) {
                uint64_t full = 0, now = 0, learn = 0, attr = 0, perfect = 0, passing = 0;
                std::vector<char> g_now, g_learn, g_attr;
                for (uint32_t a = 0; a <= last; a++) {
                  const uint32_t l = order[a].second;
                  full += sizes[l];
                  if (!pass[l]) continue;
                  const uint32_t ng = (sizes[l] + 31) / 32;
                  g_now.assign(ng, 0); g_learn.assign(ng, 0); g_attr.assign(ng, 0);
                  for (uint32_t r : list_rows[l])
                    if ((c.words[r >> 5] >> (r & 31)) & 1u) {
                      g_now[pos_now[r] / 32] = 1; g_learn[pos_learn[r] / 32] = 1; g_attr[pos_attr[r] / 32] = 1;
                    }
                  for (uint32_t g = 0; g < ng; g++) {
                    const uint32_t len = std::min<uint32_t>(32, sizes[l] - g * 32);
                    now += g_now[g] ? len : 0; learn += g_learn[g] ? len : 0; attr += g_attr[g] ? len : 0;
                  }
                  perfect += std::min<uint64_t>(sizes[l], uint64_t((pass[l] + 31) / 32) * 32);
                  passing += pass[l];
                }
                char b[256];
                snprintf(b, sizeof(b), "{\"lists\": %u, \"full\": %llu, \"now\": %llu, \"learned\": %llu, \"attribute\": %llu, \"perfect\": %llu, \"passing\": %llu}",
                         last + 1, (unsigned long long)full, (unsigned long long)now, (unsigned long long)learn,
                         (unsigned long long)attr, (unsigned long long)perfect, (unsigned long long)passing);
                return std::string(b);
              };
              std::lock_guard<std::mutex> lk(mu);
              printf("{\"ivf_order\": %d, \"seen\": %d, \"s\": %.5f, \"cover\": %s, \"probes400\": %s}\n", i, int(seen),
                     double(c.count) / rows, groups(top_rank[k - 1]).c_str(), groups(399).c_str());
              fflush(stdout);
              continue;
            }
            std::lock_guard<std::mutex> lk(mu);
            printf("{\"analyze_ivf\": %d, \"s\": %.5f, \"lists_with_passing\": %u, \"top1pct_share\": %.4f, \"top10pct_share\": %.4f, "
                   "\"cover_9\": %s, \"cover_10\": %s}\n",
                   i, double(c.count) / rows, nonempty, double(top1) / std::max<uint64_t>(tot, 1),
                   double(top10) / std::max<uint64_t>(tot, 1), cost(std::max(1, k - k / 10)).c_str(), cost(k).c_str());
            fflush(stdout);
          }
        });
      for (auto& th : pool) th.join();
      return 0;
    }
  }
  /* twol: each case's bitset as a summary bit per 1,024-item block plus the occupied blocks' words. */
  struct TwoLevel {
    const uint32_t* summary;
    const uint32_t* prefix;
    const uint32_t* words;
    const uint32_t* blocks;
    uint32_t n_blocks;
  };
  std::vector<const TwoLevel*> dtwol(ncases, nullptr);
  double occupied_sum = 0;
  if (modes.find("twol") != std::string::npos) {
    const size_t nblocks = (rows + 1023) / 1024, nsum = (nblocks + 31) / 32;
    for (int i = 0; i < ncases; i++) {
      const auto& w = cases[i].words;
      std::vector<uint32_t> summary(nsum, 0), prefix(nsum, 0), words, blocks;
      for (size_t b = 0; b < nblocks; b++) {
        bool any = false;
        for (size_t k = 0; k < 32 && b * 32 + k < w.size() && !any; k++)
          any = w[b * 32 + k] != 0;
        if (!any) continue;
        summary[b / 32] |= 1u << (b % 32);
        blocks.push_back(uint32_t(b));
        for (size_t k = 0; k < 32; k++)
          words.push_back(b * 32 + k < w.size() ? w[b * 32 + k] : 0u);
      }
      for (size_t j = 1; j < nsum; j++)
        prefix[j] = prefix[j - 1] + __builtin_popcount(summary[j - 1]);
      occupied_sum += double(blocks.size()) / double(nblocks);
      uint32_t *ds, *dp, *dw, *db;
      CUDA(cudaMalloc(&ds, nsum * 4)); CUDA(cudaMemcpy(ds, summary.data(), nsum * 4, cudaMemcpyHostToDevice));
      CUDA(cudaMalloc(&dp, nsum * 4)); CUDA(cudaMemcpy(dp, prefix.data(), nsum * 4, cudaMemcpyHostToDevice));
      CUDA(cudaMalloc(&dw, std::max<size_t>(words.size(), 1) * 4));
      CUDA(cudaMemcpy(dw, words.data(), words.size() * 4, cudaMemcpyHostToDevice));
      CUDA(cudaMalloc(&db, std::max<size_t>(blocks.size(), 1) * 4));
      CUDA(cudaMemcpy(db, blocks.data(), blocks.size() * 4, cudaMemcpyHostToDevice));
      TwoLevel h{ds, dp, dw, db, uint32_t(blocks.size())};
      TwoLevel* d;
      CUDA(cudaMalloc(&d, sizeof(TwoLevel)));
      CUDA(cudaMemcpy(d, &h, sizeof(h), cudaMemcpyHostToDevice));
      dtwol[i] = d;
    }
    printf("{\"two_level\": true, \"occupied_blocks_mean\": %.4f}\n", occupied_sum / ncases);
  }
  AnnUdfStep* dsteps;
  int32_t* dvalues;
  CUDA(cudaMalloc(&dsteps, std::max<size_t>(1, nsteps) * sizeof(AnnUdfStep)));
  CUDA(cudaMalloc(&dvalues, std::max<size_t>(1, nvalues) * sizeof(int32_t)));
  CUDA(cudaMemcpy(dsteps, steps.data(), nsteps * sizeof(AnnUdfStep), cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(dvalues, values.data(), nvalues * sizeof(int32_t), cudaMemcpyHostToDevice));

  /* udfv: each multi-valued column's distinct values and its elements as dense 16-bit ids. */
  const bool use_udfv = modes.find("udfv") != std::string::npos;
  std::vector<std::vector<int32_t>> dict(ANN_UDF_MAX_COLUMNS);
  UdfvParams udfv_template;
  memset(&udfv_template, 0, sizeof(udfv_template));
  double dense_ms = 0;
  std::vector<const void*> packed_columns(ANN_UDF_MAX_COLUMNS, nullptr);
  const bool use_udfv_any =
    use_udfv || modes.find("udfp") != std::string::npos || modes.find("udfq") != std::string::npos;
  if (use_udfv_any) {
    t0 = now_ms();
    for (size_t c = 0; c < columns.names.size(); c++) {
      if (!host_offsets[c]) continue;
      const uint64_t n = host_offsets[c][rows];
      std::vector<int32_t> all(host_columns[c], host_columns[c] + n);
      std::sort(all.begin(), all.end());
      all.erase(std::unique(all.begin(), all.end()), all.end());
      if (all.size() > 65535) fail(columns.names[c] + ": more than 65,535 distinct values");
      dict[c] = all;
      std::vector<uint16_t> ids(n);
      std::vector<std::thread> pool;
      unsigned nthreads = std::max(1u, std::thread::hardware_concurrency() / 2);
      for (unsigned t = 0; t < nthreads; t++)
        pool.emplace_back([&, t]() {
          for (uint64_t i = t; i < n; i += nthreads)
            ids[i] = uint16_t(std::lower_bound(dict[c].begin(), dict[c].end(), host_columns[c][i]) -
                              dict[c].begin());
        });
      for (auto& th : pool)
        th.join();
      void* d;
      CUDA(cudaMalloc(&d, std::max<uint64_t>(n, 1) * sizeof(uint16_t)));
      CUDA(cudaMemcpy(d, ids.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice));
      udfv_template.dense[c] = static_cast<const uint16_t*>(d);
      if (all.size() > 65533) fail(columns.names[c] + ": ids 0xfffe and 0xffff are markers");
      /* udfp: eight ids a row in 16 bytes. */
      std::vector<uint16_t> packed(size_t(rows) * 8, 0xffff);
      uint64_t overflow = 0;
      for (uint64_t r = 0; r < rows; r++) {
        uint32_t b = host_offsets[c][r], e = host_offsets[c][r + 1];
        if (e - b > 8) {
          packed[r * 8] = 0xfffe;
          overflow++;
          continue;
        }
        for (uint32_t i = b; i < e; i++)
          packed[r * 8 + (i - b)] = ids[i];
      }
      CUDA(cudaMalloc(&d, packed.size() * sizeof(uint16_t)));
      CUDA(cudaMemcpy(d, packed.data(), packed.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      packed_columns[c] = d;
      printf("{\"packed\": \"%s\", \"values\": %zu, \"overflow_rows\": %llu}\n",
             columns.names[c].c_str(), all.size(), (unsigned long long)overflow);
    }
    for (int c = 0; c < ANN_UDF_MAX_COLUMNS; c++) {
      udfv_template.columns[c] = params_template.columns[c];
      udfv_template.offsets[c] = params_template.offsets[c];
    }
    udfv_template.steps  = dsteps;
    udfv_template.values = dvalues;
    dense_ms             = now_ms() - t0;
  }
  /* A step's aux for udfv, its value bitset appended to arena (host, per call). */
  auto udfv_aux = [&](const AnnUdfStep& st, std::vector<uint32_t>& arena) -> UdfvAux {
    UdfvAux x{0, 0, 0xffffffffu, 0};
    if (!st.multi) return x;
    const std::vector<int32_t>& d = dict[st.column];
    if (!d.empty() && d[0] == ANN_FILTER_NULL_VALUE) x.null_id = 0;
    if (st.op == ANN_FILTER_RANGE) {
      int64_t lo = std::max<int64_t>(st.low, int64_t(ANN_FILTER_NULL_VALUE) + 1);
      x.kind     = 1;
      x.a = uint32_t(std::lower_bound(d.begin(), d.end(), lo) - d.begin());
      x.b = uint32_t(std::upper_bound(d.begin(), d.end(), int64_t(st.high)) - d.begin());
      if (lo > st.high) x.b = x.a;
      return x;
    }
    if (st.op == ANN_FILTER_NULL) return x;
    const uint32_t words = uint32_t((d.size() + 31) / 32);
    x.a = uint32_t(arena.size());
    x.b = words;
    arena.resize(arena.size() + words, 0);
    for (uint32_t v = 0; v < st.count; v++) {
      auto it = std::lower_bound(d.begin(), d.end(), values[st.first + v]);
      if (it != d.end() && *it == values[st.first + v]) {
        uint32_t id = uint32_t(it - d.begin());
        arena[x.a + id / 32] |= 1u << (id % 32);
      }
    }
    return x;
  };

  /* Device postings of every column the programs name, for the build modes. */
  const bool use_postings = modes.find("build") != std::string::npos || modes.find("scanc") != std::string::npos;
  std::vector<PostingsColumn> postings(columns.names.size());
  std::vector<PostingsColumn*> postings_ptrs;
  double postings_ms = 0;
  if (use_postings) {
    t0 = now_ms();
    for (size_t c = 0; c < columns.names.size(); c++) {
      postings[c] = postings_build(host_columns[c], host_offsets[c], rows);
      postings_ptrs.push_back(&postings[c]);
    }
    postings_ms = now_ms() - t0;
  }

  /*
   * One call: queries [first, first + n). Per-query data (bitset pointers, UDF starts and params)
   * goes to buffers of the calling thread.
   */
  struct Thread {
    raft::device_resources res;
    uint32_t* neighbors;
    float* distances;
    const uint32_t** bitset_ptrs;
    uint32_t* starts;
    AnnUdfParams* params;
    UdfvParams* vparams;
    UdfvAux* vaux = nullptr;
    uint32_t* varena = nullptr;
    size_t vaux_cap = 0, varena_cap = 0;
    std::vector<uint32_t> out;
    PostingsScratch scratch;
    uint32_t* built = nullptr;  // batch bitsets built for the call
    uint32_t* cworkspace = nullptr;  // buildc: cuVS postings workspace
    int64_t cworkspace_words = 0;
    int64_t* ivf_cand = nullptr;  // ivfpq: k * refine candidates a query, their PQ distances,
    float* ivf_cand_d = nullptr;  // and the k re-scored results
    int64_t* ivf_nb   = nullptr;
    std::vector<int64_t> ivf_host;
  };
  // --pool-gb: the current device resource becomes a pool as well (the search also allocates from it).
  if (pool_gb > 0)
    rmm::mr::set_current_device_resource(
      rmm::mr::pool_memory_resource(rmm::mr::cuda_memory_resource{}, size_t(pool_gb) << 30));
  auto make_thread = [&]() {
    auto t = std::make_unique<Thread>();
    if (pool_gb > 0) raft::resource::set_workspace_to_pool_resource(t->res, size_t(pool_gb) << 30);
    if (workspace_pool_gb > 0)
      raft::resource::set_workspace_to_pool_resource(t->res, size_t(workspace_pool_gb) << 30);
    CUDA(cudaMalloc(&t->neighbors, size_t(batch) * k * sizeof(uint32_t)));
    CUDA(cudaMalloc(&t->distances, size_t(batch) * k * sizeof(float)));
    CUDA(cudaMalloc(&t->bitset_ptrs, size_t(batch) * sizeof(void*)));
    CUDA(cudaMalloc(&t->starts, size_t(batch + 1) * sizeof(uint32_t)));
    CUDA(cudaMalloc(&t->params, sizeof(AnnUdfParams)));
    CUDA(cudaMalloc(&t->vparams, sizeof(UdfvParams)));
    t->out.resize(size_t(batch) * k);
    if (ivf) {
      const size_t kc = size_t(k) * ivf_refine;
      CUDA(cudaMalloc(&t->ivf_cand, size_t(batch) * kc * sizeof(int64_t)));
      CUDA(cudaMalloc(&t->ivf_cand_d, size_t(batch) * kc * sizeof(float)));
      CUDA(cudaMalloc(&t->ivf_nb, size_t(batch) * k * sizeof(int64_t)));
      t->ivf_host.resize(size_t(batch) * kc);
    }
    if (use_postings) {
      postings_scratch_init(t->scratch, rows);
      CUDA(cudaMalloc(&t->built, size_t(batch) * nwords * sizeof(uint32_t)));
    }
    return t;
  };
  auto run_call = [&](Thread& t, const std::string& mode, int first, int n) {
    cagra::search_params sp;
    sp.algo                 = cagra::search_algo::MULTI_CTA;
    sp.itopk_size           = std::max(itopk, k);
    sp.max_iterations       = iterations;
    sp.search_width         = width;
    sp.thread_block_size    = block;
    sp.num_random_samplings = samplings;
    if (rand_xor_mask_set) sp.rand_xor_mask = rand_xor_mask;
    sp.filter_expansion     = cagra::filter_expansion_mode::PASSING_TWO_HOP;
    // --auto-expansion: filter_expansion_mode::AUTO (value 100; needs a cuVS that has it).
    if (auto_expansion) sp.filter_expansion = static_cast<cagra::filter_expansion_mode>(100);
    cudaStream_t stream     = raft::resource::get_cuda_stream(t.res).get();
    auto queries = raft::make_device_matrix_view<const half, int64_t>(dq + size_t(first) * qdim,
                                                                     n, qdim);
    auto nb      = raft::make_device_matrix_view<uint32_t, int64_t>(t.neighbors, n, k);
    auto ds      = raft::make_device_matrix_view<float, int64_t>(t.distances, n, k);
    if (mode == "ivfpqb" || mode == "ivfpq1") {
      ivf_pq::search_params isp;
      isp.n_probes     = uint32_t(ivf_probes);
      isp.lut_dtype    = ivf_lut == 16 ? CUDA_R_16F : ivf_lut == 8 ? CUDA_R_8U : CUDA_R_32F;
      isp.internal_distance_dtype = ivf_internal == 16 ? CUDA_R_16F : CUDA_R_32F;
      const int64_t kc = int64_t(k) * ivf_refine;
      auto fq = raft::make_device_matrix_view<const float, int64_t>(dqf + size_t(first) * m.dims, n, m.dims);
      // One bitset a call: the batch's first filter (ivfpqb), or the single query's own (ivfpq1).
      cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> filter(
        ivf_renumber ? raft::core::bitset_view<uint32_t, int64_t>(ivf_dbits[first], int64_t(ivf_new_rows))
                     : raft::core::bitset_view<uint32_t, int64_t>(dbits[first], int64_t(rows)));
      auto q8v = raft::make_device_matrix_view<const int8_t, int64_t>(dq8 + size_t(first) * m.dims, n, m.dims);
      if (ivf_int8)
        ivf_pq::search(t.res, isp, *ivf, q8v,
                       raft::make_device_matrix_view<int64_t, int64_t>(t.ivf_cand, n, kc),
                       raft::make_device_matrix_view<float, int64_t>(t.ivf_cand_d, n, kc), filter);
      else
        ivf_pq::search(t.res, isp, *ivf, fq,
                       raft::make_device_matrix_view<int64_t, int64_t>(t.ivf_cand, n, kc),
                       raft::make_device_matrix_view<float, int64_t>(t.ivf_cand_d, n, kc), filter);
      const int64_t* result = t.ivf_cand;
      int64_t width_out     = kc;
      if (ivf_refine > 1) {
        // A query with fewer passing rows in its probed lists than kc leaves slots without a
        // passing row (ivf_pq::search fills them with ids of filtered-out rows of those lists,
        // which the exact re-score would otherwise rank among the results): give those the
        // query's first passing candidate (duplicates are dropped below), since the re-score
        // reads every candidate's row.
        CUDA(cudaMemcpyAsync(t.ivf_host.data(), t.ivf_cand, size_t(n) * kc * sizeof(int64_t),
                             cudaMemcpyDeviceToHost, stream));
        t.res.sync_stream();
        bool patched = false;
        if (ivf_renumber) {
          for (size_t j = 0; j < size_t(n) * kc; j++) {
            int64_t& c = t.ivf_host[j];
            c          = (c >= 0 && uint64_t(c) < ivf_new_rows) ? ivf_new2old[c] : -1;
          }
          patched = true;
        }
        const std::vector<uint32_t>& words = cases[first].words;
        auto usable = [&](int64_t r) {
          return r >= 0 && uint64_t(r) < rows && ((words[size_t(r) / 32] >> (r % 32)) & 1);
        };
        for (int q = 0; q < n; q++) {
          int64_t* c     = t.ivf_host.data() + size_t(q) * kc;
          int64_t filler = -1;
          for (int64_t j = 0; j < kc && filler < 0; j++)
            if (usable(c[j])) filler = c[j];
          ivf_no_candidate[size_t(first) + q] = filler < 0;
          for (int64_t j = 0; j < kc; j++)
            if (!usable(c[j])) {
              if (c[j] >= 0 && uint64_t(c[j]) < rows) ivf_filtered_out_candidates++;
              c[j]    = filler < 0 ? 0 : filler;
              patched = true;
            }
        }
        if (ivf_refine_host) {
          // Exact re-score on the host from the mapped rows; the results land in ivf_host.
          std::vector<int64_t> nb(size_t(n) * k);
          std::vector<float> dist(size_t(n) * k);
          if (ivf_int8)
            cuvs::neighbors::refine(
              t.res, raft::make_host_matrix_view<const int8_t, int64_t>(h8.data(), int64_t(rows), int64_t(m.dims)),
              raft::make_host_matrix_view<const int8_t, int64_t>(hq8.data() + size_t(first) * m.dims, n, m.dims),
              raft::make_host_matrix_view<const int64_t, int64_t>(t.ivf_host.data(), n, kc),
              raft::make_host_matrix_view<int64_t, int64_t>(nb.data(), n, k),
              raft::make_host_matrix_view<float, int64_t>(dist.data(), n, k), cuvs::distance::DistanceType::L2Expanded);
          else
            cuvs::neighbors::refine(
              t.res, raft::make_host_matrix_view<const float, int64_t>(hvec, int64_t(rows), int64_t(m.dims)),
              raft::make_host_matrix_view<const float, int64_t>(hqf.data() + size_t(first) * m.dims, n, m.dims),
              raft::make_host_matrix_view<const int64_t, int64_t>(t.ivf_host.data(), n, kc),
              raft::make_host_matrix_view<int64_t, int64_t>(nb.data(), n, k),
              raft::make_host_matrix_view<float, int64_t>(dist.data(), n, k), cuvs::distance::DistanceType::L2Expanded);
          std::copy(nb.begin(), nb.end(), t.ivf_host.begin());
          for (int q = 0; q < n; q++) {
            const int64_t* c = t.ivf_host.data() + size_t(q) * k;
            uint32_t* o      = t.out.data() + size_t(q) * k;
            int filled       = 0;
            for (int j = 0; j < k && filled < k; j++) {
              if (c[j] < 0 || uint64_t(c[j]) >= rows || ivf_no_candidate[size_t(first) + q]) continue;
              const uint32_t r = uint32_t(c[j]);
              if (std::find(o, o + filled, r) == o + filled) o[filled++] = r;
            }
            std::fill(o + filled, o + k, 0xffffffffu);
          }
          return;
        }
        if (patched)
          CUDA(cudaMemcpyAsync(t.ivf_cand, t.ivf_host.data(), size_t(n) * kc * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream));
        if (ivf_int8)
          cuvs::neighbors::refine(
            t.res, raft::make_device_matrix_view<const int8_t, int64_t>(dvec8, int64_t(rows), int64_t(m.dims)),
            q8v, raft::make_device_matrix_view<const int64_t, int64_t>(t.ivf_cand, n, kc),
            raft::make_device_matrix_view<int64_t, int64_t>(t.ivf_nb, n, k), ds,
            cuvs::distance::DistanceType::L2Expanded);
        else
          cuvs::neighbors::refine(
            t.res, raft::make_device_matrix_view<const float, int64_t>(dvec, int64_t(rows), int64_t(m.dims)),
            fq, raft::make_device_matrix_view<const int64_t, int64_t>(t.ivf_cand, n, kc),
            raft::make_device_matrix_view<int64_t, int64_t>(t.ivf_nb, n, k), ds,
            cuvs::distance::DistanceType::L2Expanded);
        result    = t.ivf_nb;
        width_out = k;
      }
      CUDA(cudaMemcpyAsync(t.ivf_host.data(), result, size_t(n) * width_out * sizeof(int64_t),
                           cudaMemcpyDeviceToHost, stream));
      t.res.sync_stream();
      for (int q = 0; q < n; q++) {
        const int64_t* c = t.ivf_host.data() + size_t(q) * width_out;
        uint32_t* o      = t.out.data() + size_t(q) * k;
        int filled       = 0;
        for (int64_t j = 0; j < width_out && filled < k; j++) {
          if (c[j] < 0 || uint64_t(c[j]) >= rows) continue;
          if (ivf_refine > 1 && ivf_no_candidate[size_t(first) + q]) continue;
          const uint32_t r = uint32_t(c[j]);
          if (std::find(o, o + filled, r) == o + filled) o[filled++] = r;
        }
        std::fill(o + filled, o + k, 0xffffffffu);
      }
      return;
    }
    if (mode == "udfv" || mode == "udfv1" || mode == "udfp") {
      /* The call's value bitsets and aux, built here as a search with new filters would. */
      std::vector<uint32_t> hs(n + 1), arena;
      std::vector<UdfvAux> aux;
      for (int q = 0; q < n; q++) {
        const Case& c = cases[first + q];
        hs[q]         = uint32_t(aux.size());
        for (uint32_t i = 0; i < c.nsteps; i++)
          aux.push_back(udfv_aux(steps[c.first_step + i], arena));
      }
      hs[n] = uint32_t(aux.size());
      /* Steps are indexed per call: point steps at the first query's, aux in call order. */
      if (aux.size() > t.vaux_cap) {
        if (t.vaux) CUDA(cudaFree(t.vaux));
        t.vaux_cap = aux.size() * 2;
        CUDA(cudaMalloc(&t.vaux, t.vaux_cap * sizeof(UdfvAux)));
      }
      if (arena.size() + 1 > t.varena_cap) {
        if (t.varena) CUDA(cudaFree(t.varena));
        t.varena_cap = (arena.size() + 1) * 2;
        CUDA(cudaMalloc(&t.varena, t.varena_cap * sizeof(uint32_t)));
      }
      UdfvParams p = udfv_template;
      p.steps      = dsteps + cases[first].first_step;
      p.starts     = t.starts;
      p.aux        = t.vaux;
      p.arena      = t.varena;
      p.row_base   = 0;
      for (int c = 0; c < ANN_UDF_MAX_COLUMNS; c++)
        p.packed[c] = mode == "udfp" ? packed_columns[c] : nullptr;
      CUDA(cudaMemcpyAsync(t.starts, hs.data(), (n + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice,
                           stream));
      CUDA(cudaMemcpyAsync(t.vaux, aux.data(), aux.size() * sizeof(UdfvAux), cudaMemcpyHostToDevice,
                           stream));
      if (!arena.empty())
        CUDA(cudaMemcpyAsync(t.varena, arena.data(), arena.size() * sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream));
      CUDA(cudaMemcpyAsync(t.vparams, &p, sizeof(p), cudaMemcpyHostToDevice, stream));
      cuvs::neighbors::filtering::udf_filter filter(kUdfvSource, t.vparams, -1.0f, "ann_predicate_udfv");
      cagra::search(t.res, sp, searched, queries, nb, ds, filter);
    } else if (mode == "udfq") {
      static const std::string source = std::string(kUdfvSource) + kUdfqSource + kUdfqPredicate;
      std::vector<uint32_t> hs(n + 1), arena;
      std::vector<UdfvAux> aux;
      std::vector<UdfqDesc> desc(n);
      std::vector<uint32_t> bit_at(n, 0);
      for (int q = 0; q < n; q++) {
        const Case& c = cases[first + q];
        hs[q]         = uint32_t(aux.size());
        for (uint32_t i = 0; i < c.nsteps; i++)
          aux.push_back(udfv_aux(steps[c.first_step + i], arena));
        const AnnUdfStep& st = steps[c.first_step];
        desc[q]              = UdfqDesc{};
        if (c.nsteps == 1 && st.multi && (st.op == ANN_FILTER_VALUES || st.op == ANN_FILTER_OVERLAP)) {
          desc[q].kind    = 1;
          desc[q].packed  = packed_columns[st.column];
          desc[q].offsets = params_template.offsets[st.column];
          desc[q].ids     = udfv_template.dense[st.column];
          desc[q].nwords  = aux.back().b;
          desc[q].negate  = st.negate;
          desc[q].null_id = aux.back().null_id;
          bit_at[q]       = aux.back().a;
        }
      }
      hs[n]          = uint32_t(aux.size());
      const size_t D = (size_t(n) * sizeof(UdfqDesc) + 15) / 16 * 16;
      if (aux.size() + D / sizeof(UdfvAux) + 1 > t.vaux_cap) {
        if (t.vaux) CUDA(cudaFree(t.vaux));
        t.vaux_cap = (aux.size() + D / sizeof(UdfvAux) + 1) * 2;
        CUDA(cudaMalloc(&t.vaux, t.vaux_cap * sizeof(UdfvAux)));
      }
      if (arena.size() + 1 > t.varena_cap) {
        if (t.varena) CUDA(cudaFree(t.varena));
        t.varena_cap = (arena.size() + 1) * 2;
        CUDA(cudaMalloc(&t.varena, t.varena_cap * sizeof(uint32_t)));
      }
      UdfvParams p = udfv_template;
      p.steps      = dsteps + cases[first].first_step;
      p.starts     = t.starts;
      p.aux        = t.vaux;
      p.arena      = t.varena;
      for (int c = 0; c < ANN_UDF_MAX_COLUMNS; c++)
        p.packed[c] = packed_columns[c];
      auto* ddesc = reinterpret_cast<UdfqDesc*>(t.vaux + aux.size() + 1);
      for (int q = 0; q < n; q++) {
        desc[q].generic = t.vparams;
        if (desc[q].kind == 1) desc[q].bits = t.varena + bit_at[q];
      }
      CUDA(cudaMemcpyAsync(t.starts, hs.data(), (n + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice,
                           stream));
      CUDA(cudaMemcpyAsync(t.vaux, aux.data(), aux.size() * sizeof(UdfvAux), cudaMemcpyHostToDevice,
                           stream));
      if (!arena.empty())
        CUDA(cudaMemcpyAsync(t.varena, arena.data(), arena.size() * sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream));
      CUDA(cudaMemcpyAsync(t.vparams, &p, sizeof(p), cudaMemcpyHostToDevice, stream));
      CUDA(cudaMemcpyAsync(ddesc, desc.data(), n * sizeof(UdfqDesc), cudaMemcpyHostToDevice, stream));
      cuvs::neighbors::filtering::udf_filter filter(source, ddesc, -1.0f, "ann_predicate_udfq");
      cagra::search(t.res, sp, searched, queries, nb, ds, filter);
    } else if (mode == "udfb") {
      std::vector<const uint32_t*> hp(n);
      for (int q = 0; q < n; q++)
        hp[q] = dbits[first + q];
      CUDA(cudaMemcpyAsync(t.bitset_ptrs, hp.data(), n * sizeof(void*), cudaMemcpyHostToDevice, stream));
      cuvs::neighbors::filtering::udf_filter filter(kUdfbSource, (void*)t.bitset_ptrs, -1.0f,
                                                    "ann_predicate_udfb");
      cagra::search(t.res, sp, searched, queries, nb, ds, filter);
    } else if (mode == "udf" || mode == "udf1") {
      /* Cases were compiled in order, so query q's steps end where query q + 1's begin. */
      std::vector<uint32_t> hs(n + 1);
      for (int q = 0; q < n; q++)
        hs[q] = cases[first + q].first_step;
      hs[n] = cases[first + n - 1].first_step + cases[first + n - 1].nsteps;
      AnnUdfParams p = params_template;
      p.steps        = dsteps;
      p.values       = dvalues;
      p.starts       = t.starts;
      p.rowBase      = 0;
      CUDA(cudaMemcpyAsync(t.starts, hs.data(), (n + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice,
                           stream));
      CUDA(cudaMemcpyAsync(t.params, &p, sizeof(p), cudaMemcpyHostToDevice, stream));
      cuvs::neighbors::filtering::udf_filter filter(AnnUdfDeviceSource(), t.params, -1.0f,
                                                    "ann_predicate_udf");
      cagra::search(t.res, sp, searched, queries, nb, ds, filter);
    } else {
      std::vector<const uint32_t*> hp(n);
      const bool build = mode == "build1" || mode == "buildq";
      if (mode == "buildc" || mode == "scanc") {
        namespace postings_api = cuvs::core::postings;
        std::vector<raft::device_vector_view<const uint32_t, int64_t>> lists;
        for (auto* c : postings_ptrs)
          lists.push_back(raft::make_device_vector_view<const uint32_t, int64_t>(
            c->postings, int64_t(c->starts.back())));
        for (int q = 0; q < n; q++) {
          const Case& c       = cases[first + q];
          const AnnUdfStep& st = steps[c.first_step];
          if (c.nsteps != 1 || !(st.op == ANN_FILTER_VALUES || st.op == ANN_FILTER_OVERLAP) || st.negate)
            fail("buildc takes programs of one VALUES or OVERLAP leaf");
          const PostingsColumn& col = *postings_ptrs[st.column];
          std::vector<postings_api::run> runs;
          for (uint32_t v = 0; v < st.count; v++) {
            auto it = std::lower_bound(col.distinct.begin(), col.distinct.end(), values[st.first + v]);
            if (it == col.distinct.end() || *it != values[st.first + v]) continue;
            size_t j = size_t(it - col.distinct.begin());
            runs.push_back(postings_api::run{st.column, int64_t(col.starts[j]),
                                             int64_t(col.starts[j + 1] - col.starts[j])});
          }
          postings_api::step leaf{postings_api::step_kind::LEAF, 0, int64_t(runs.size())};
          auto prog  = raft::make_host_vector_view<const postings_api::step, int64_t>(&leaf, 1);
          auto rview = raft::make_host_vector_view<const postings_api::run, int64_t>(runs.data(), int64_t(runs.size()));
          int64_t need = postings_api::workspace_size(prog, rview.extent(0), int64_t(rows));
          if (need > t.cworkspace_words) {
            if (t.cworkspace) CUDA(cudaFree(t.cworkspace));
            t.cworkspace_words = need * 2;
            CUDA(cudaMalloc(&t.cworkspace, t.cworkspace_words * sizeof(uint32_t)));
          }
          postings_api::evaluate(
            t.res, prog, rview,
            raft::make_host_vector_view<const raft::device_vector_view<const uint32_t, int64_t>, int64_t>(
              lists.data(), int64_t(lists.size())),
            cuvs::core::bitset_view<uint32_t, int64_t>(t.built + size_t(q) * nwords, int64_t(rows)), 0,
            raft::make_device_vector_view<uint32_t, int64_t>(t.cworkspace, t.cworkspace_words));
        }
      }
      for (int q = 0; q < n; q++) {
        if (build) {
          const Case& c = cases[first + q];
          postings_eval(postings_ptrs, steps.data() + c.first_step, c.nsteps, values.data(), t.scratch,
                        t.built + size_t(q) * nwords, stream);
        }
        hp[q] = mode == "twol" ? reinterpret_cast<const uint32_t*>(dtwol[first + q])
                : (build || mode == "buildc" || mode == "scanc") ? t.built + size_t(q) * nwords
                : is_shared(mode) ? dbits[first] : dbits[first + q];
      }
#ifdef PROBE_HAS_FP8_SCAN
      if (mode == "scanc") {
        sp.filtered_route  = cagra::filtered_route_mode::SCAN;
        sp.filter_expansion = cagra::filter_expansion_mode::SCORE_ALL;
        sp.scan_candidates  = uint32_t(std::min(4096, k + k * scan_slack / 100));
      }
#endif
      if (mode == "bitsetq" || mode == "buildq" || mode == "buildc" || mode == "twol") {
        CUDA(cudaMemcpyAsync(t.bitset_ptrs, hp.data(), n * sizeof(void*), cudaMemcpyHostToDevice,
                             stream));
        sp.query_bitsets = t.bitset_ptrs;
      }
      if (mode == "stock1" || mode == "stockr1" || mode == "stockb") {
        // cuVS' own filtered walk: every neighbour scored, the list lengthened by the filtering rate.
        sp.algo             = cagra::search_algo::AUTO;
        sp.filter_expansion = cagra::filter_expansion_mode::SCORE_ALL;
        if (mode != "stock1")
          sp.filtering_rate = float(std::min(0.999, 1.0 - double(cases[first].count) / double(rows)));
      }
      if (mode == "nofilter") {
        sp.filter_expansion = cagra::filter_expansion_mode::SCORE_ALL;
        cagra::search(t.res, sp, searched, queries, nb, ds);
        CUDA(cudaMemcpyAsync(t.out.data(), t.neighbors, size_t(n) * k * sizeof(uint32_t),
                             cudaMemcpyDeviceToHost, stream));
        t.res.sync_stream();
        return;
      }
      // Declared as a bitset of bytes, the same words pass the same rows through the linked filter.
      const bool linked = mode == "linked1" || mode == "linkedb";
      auto view = linked ? raft::core::bitset_view<uint32_t, int64_t>(const_cast<uint32_t*>(hp[0]),
                                                                    int64_t(rows), 8)
                         : raft::core::bitset_view<uint32_t, int64_t>(
                             const_cast<uint32_t*>(mode == "twol" ? dbits[first] : hp[0]), int64_t(rows));
      cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> filter(view);
      cagra::search(t.res, sp, searched, queries, nb, ds, filter);
    }
    CUDA(cudaMemcpyAsync(t.out.data(), t.neighbors, size_t(n) * k * sizeof(uint32_t),
                         cudaMemcpyDeviceToHost, stream));
    t.res.sync_stream();
  };

  printf("{\"setup\": true, \"rows\": %llu, \"dims\": %u, \"cases\": %d, \"columns\": %zu, "
         "\"column_mb\": %.0f, \"host_eval_ms\": %.0f, \"truth_ms\": %.0f, \"load_ms\": %.0f, "
         "\"pass_rate_median\": %.5f}\n",
         (unsigned long long)rows, m.dims, ncases, columns.names.size(), column_bytes / 1048576.0,
         eval_ms, truth_ms, load_ms,
         [&]() {
           std::vector<double> r;
           for (auto& c : cases)
             r.push_back(double(c.count) / double(rows));
           std::sort(r.begin(), r.end());
           return r[r.size() / 2];
         }());
  fflush(stdout);

  /* Build every case's bitset on the GPU once: exactness against the host and the build time. */
  if (use_postings) {
    auto t        = make_thread();
    cudaStream_t stream = raft::resource::get_cuda_stream(t->res).get();
    std::vector<uint32_t> back(nwords);
    std::vector<double> ms;
    uint64_t mismatched = 0;
    for (int i = 0; i < ncases; i++) {
      const Case& c = cases[i];
      CUDA(cudaStreamSynchronize(stream));
      double s0 = now_ms();
      postings_eval(postings_ptrs, steps.data() + c.first_step, c.nsteps, values.data(), t->scratch, t->built,
                    stream);
      CUDA(cudaStreamSynchronize(stream));
      ms.push_back(now_ms() - s0);
      CUDA(cudaMemcpy(back.data(), t->built, nwords * 4, cudaMemcpyDeviceToHost));
      mismatched += back != c.words;
    }
    std::vector<double> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    printf("{\"build_check\": true, \"postings_ms\": %.0f, \"cases\": %d, \"mismatched\": %llu, "
           "\"build_ms_p50\": %.3f, \"build_ms_p99\": %.3f, \"build_ms_max\": %.3f}\n",
           postings_ms, ncases, (unsigned long long)mismatched, sorted[sorted.size() / 2],
           sorted[std::min(sorted.size() - 1, size_t(0.99 * sorted.size()))], sorted.back());
    fflush(stdout);
  }

  std::vector<std::string> mode_list;
  for (std::stringstream in(modes); in.good();) {
    std::string mode;
    std::getline(in, mode, ',');
    if (!mode.empty()) mode_list.push_back(mode);
  }
  for (const std::string& mode : mode_list) {
    const int per_call =
      (mode == "bitset1" || mode == "udf1" || mode == "udfv1" || mode == "build1" || mode == "scanc" ||
       mode == "stock1" || mode == "stockr1" || mode == "linked1" || mode == "ivfpq1")
        ? 1
        : batch;
    const int calls    = ncases / per_call;
    if (calls == 0) continue;
    if (mode == "twol") setenv("CUVS_TWO_LEVEL", "1", 1); else unsetenv("CUVS_TWO_LEVEL");
    std::vector<std::unique_ptr<Thread>> pool;
    for (int t = 0; t < threads; t++)
      pool.push_back(make_thread());

    /* First call alone: for the UDF it includes the JIT compile. */
    double c0 = now_ms();
    run_call(*pool[0], mode, 0, per_call);
    double first_ms = now_ms() - c0;

    /* Quality from one pass over every call. */
    double recall_sum = 0, recall_min = 1;
    uint64_t failing = 0, short_results = 0, checked = 0;
    for (int call = 0; call < calls; call++) {
      run_call(*pool[0], mode, call * per_call, per_call);
      for (int q = 0; q < per_call; q++) {
        const Case& own = cases[call * per_call + q];
        // The filter this query was searched under: its own, or its batch's first.
        const Case& c   = is_shared(mode) ? cases[call * per_call] : own;
        const uint32_t* got = pool[0]->out.data() + size_t(q) * k;
        uint64_t valid = 0, hit = 0;
        std::vector<uint32_t> truth = is_shared(mode) ? own.truth_shared : own.truth;
        std::sort(truth.begin(), truth.end());
        for (int j = 0; j < k; j++) {
          uint32_t r = got[j];
          if (r >= rows) continue;
          if (!((c.words[r / 32] >> (r % 32)) & 1)) {
            failing++;
            continue;
          }
          valid++;
          hit += std::binary_search(truth.begin(), truth.end(), r);
        }
        short_results += valid < std::min<uint64_t>(k, c.count);
        if (recall && !truth.empty()) {
          double rr = double(hit) / double(truth.size());
          recall_sum += rr;
          recall_min = std::min(recall_min, rr);
        }
        checked++;
      }
    }

    /* Timing: reps passes over every call, threads taking calls in turn. */
    std::vector<double> lat;
    std::mutex lat_mutex;
    double w0 = now_ms();
    for (int rep = 0; rep < reps; rep++) {
      std::atomic<int> next{0};
      std::vector<std::thread> workers;
      for (int t = 0; t < threads; t++)
        workers.emplace_back([&, t]() {
          std::vector<double> mine;
          for (int call; (call = next++) < calls;) {
            double s = now_ms();
            run_call(*pool[t], mode, call * per_call, per_call);
            mine.push_back(now_ms() - s);
          }
          std::lock_guard<std::mutex> g(lat_mutex);
          lat.insert(lat.end(), mine.begin(), mine.end());
        });
      for (auto& w : workers)
        w.join();
    }
    double wall = now_ms() - w0;
    std::sort(lat.begin(), lat.end());
    auto pct = [&](double p) { return lat[std::min(lat.size() - 1, size_t(p * lat.size()))]; };
    printf("{\"mode\": \"%s\", \"k\": %d, \"queries_per_call\": %d, \"threads\": %d, \"calls\": %d, "
           "\"first_call_ms\": %.1f, \"call_ms_p50\": %.3f, \"call_ms_p99\": %.3f, "
           "\"queries_per_s\": %.0f, \"recall_mean\": %.4f, \"recall_min\": %.4f, "
           "\"failing_rows\": %llu, \"short_results\": %llu, \"checked\": %llu, "
           "\"itopk\": %d, \"iterations\": %d, \"samplings\": %d}\n",
           mode.c_str(), k, per_call, threads, calls, first_ms, pct(0.5), pct(0.99),
           double(calls) * per_call * reps / (wall / 1000.0),
           recall && checked ? recall_sum / checked : -1.0, recall ? recall_min : -1.0,
           (unsigned long long)failing, (unsigned long long)short_results,
           (unsigned long long)checked, std::max(itopk, k), iterations, samplings);
    if (mode.rfind("ivfpq", 0) == 0) {
      // Over every call of the mode (the timed repetitions included). int8 = --ivf-int8.
      printf("{\"ivfpq_note\": \"%s\", \"filtered_out_candidates_returned\": %llu, \"probes\": %d, \"refine\": %d, "
             "\"lut_bits\": %d, \"internal_bits\": %d}\n",
             mode.c_str(), (unsigned long long)ivf_filtered_out_candidates.exchange(0), ivf_probes, ivf_refine,
             ivf_lut, ivf_internal);
    }
    fflush(stdout);
  }
  return 0;
}
