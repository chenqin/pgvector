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
 *   buildc   as buildq, each bitset built by cuVS (cuvs::core::postings::evaluate, the worker's
 *            build) from the same postings; programs of one VALUES or OVERLAP leaf only
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
 *                          [--no-pitch] [--modes bitset1,bitsetq,udf,udf1,udfv,udfv1,udfp,udfb,udfq,build1,buildq,buildc] [--no-recall]
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
#include <cuvs/neighbors/common.hpp>
#include <raft/core/bitset.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <algorithm>
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

int main(int argc, char** argv)
{
  if (argc < 3) fail("usage: cagra_twohop_udf_probe INDEX_DIR CASES [options]");
  std::string dir = argv[1], cases_path = argv[2];
  int k = 300, batch = 32, reps = 3, threads = 1, itopk = 1536, iterations = 18, samplings = 2,
      width = 4, block = 64;
  bool recall = true;
  bool pitch = true;
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
          d.clear();
          for (size_t w = 0; w < nwords; w++)
            for (uint32_t bits = c.words[w]; bits; bits &= bits - 1) {
              uint32_t r   = uint32_t(w * 32 + __builtin_ctz(bits));
              const float* x = vectors + size_t(r) * m.dims;
              float s        = 0;
              for (uint32_t j = 0; j < m.dims; j++) {
                float t = x[j] - c.query[j];
                s += t * t;
              }
              d.emplace_back(s, r);
            }
          size_t keep = std::min<size_t>(k, d.size());
          std::partial_sort(d.begin(), d.begin() + keep, d.end());
          c.truth.resize(keep);
          for (size_t j = 0; j < keep; j++)
            c.truth[j] = d[j].second;
        }
      });
    for (auto& t : pool)
      t.join();
  }
  double truth_ms = now_ms() - t0 - eval_ms;

  /* The index, its FP16 dataset kept alive beside it. */
  raft::device_resources load_res;
  cagra::device_padded_index<half> index(load_res);
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<half, int64_t>> dataset;
  t0 = now_ms();
  cagra::deserialize(load_res, dir + "/" + m.shard, &index, &dataset);
  load_res.sync_stream();
  double load_ms = now_ms() - t0;
  if (uint64_t(index.size()) != rows || uint32_t(index.dim()) != m.dims) fail("index shape");

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
  auto& searched = wide ? *wide : index;

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
  const bool use_postings = modes.find("build") != std::string::npos;
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
  };
  auto make_thread = [&]() {
    auto t = std::make_unique<Thread>();
    CUDA(cudaMalloc(&t->neighbors, size_t(batch) * k * sizeof(uint32_t)));
    CUDA(cudaMalloc(&t->distances, size_t(batch) * k * sizeof(float)));
    CUDA(cudaMalloc(&t->bitset_ptrs, size_t(batch) * sizeof(void*)));
    CUDA(cudaMalloc(&t->starts, size_t(batch + 1) * sizeof(uint32_t)));
    CUDA(cudaMalloc(&t->params, sizeof(AnnUdfParams)));
    CUDA(cudaMalloc(&t->vparams, sizeof(UdfvParams)));
    t->out.resize(size_t(batch) * k);
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
    sp.filter_expansion     = cagra::filter_expansion_mode::PASSING_TWO_HOP;
    cudaStream_t stream     = raft::resource::get_cuda_stream(t.res).get();
    auto queries = raft::make_device_matrix_view<const half, int64_t>(dq + size_t(first) * qdim,
                                                                     n, qdim);
    auto nb      = raft::make_device_matrix_view<uint32_t, int64_t>(t.neighbors, n, k);
    auto ds      = raft::make_device_matrix_view<float, int64_t>(t.distances, n, k);
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
      if (mode == "buildc") {
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
        hp[q] = (build || mode == "buildc") ? t.built + size_t(q) * nwords : dbits[first + q];
      }
      if (mode == "bitsetq" || mode == "buildq" || mode == "buildc") {
        CUDA(cudaMemcpyAsync(t.bitset_ptrs, hp.data(), n * sizeof(void*), cudaMemcpyHostToDevice,
                             stream));
        sp.query_bitsets = t.bitset_ptrs;
      }
      auto view = raft::core::bitset_view<uint32_t, int64_t>(const_cast<uint32_t*>(hp[0]), int64_t(rows));
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
      (mode == "bitset1" || mode == "udf1" || mode == "udfv1" || mode == "build1") ? 1 : batch;
    const int calls    = ncases / per_call;
    if (calls == 0) continue;
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
        const Case& c = cases[call * per_call + q];
        const uint32_t* got = pool[0]->out.data() + size_t(q) * k;
        uint64_t valid = 0, hit = 0;
        std::vector<uint32_t> truth = c.truth;
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
    fflush(stdout);
  }
  return 0;
}
