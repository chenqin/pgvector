/*
 * Blackwell scan module: quantized gather scans for the cuVS worker.
 *
 * Eight threads score one listed row, each taking 16-dimension chunks with
 * one 8-byte (NVFP4) or 16-byte (FP8) load and, for NVFP4, one block-scale
 * decode, so a warp scores four rows. That reads 4 or 8 bits per dimension
 * where the cuVS backend's gather kernel reads 16 or 32. Decoding uses Blackwell's conversion instructions, which
 * need an architecture-specific target (sm_100a, sm_120a); the module is
 * compiled once per device with NVRTC and only on compute capability 10.x
 * or 12.x.
 */
#include "blackwell.h"
#include "nvrtc.h"

/* The driver API header is not ISO C pedantic-clean. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <cuda.h>
#pragma GCC diagnostic pop
#include <cuda_runtime_api.h>
#include <pthread.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ANN_BLACKWELL_MAX_DEVICES 64
#define ANN_BLACKWELL_BLOCK 256
#define ANN_BLACKWELL_GROUP 8	/* threads per scored row, as GROUP below */

/* Pieces stay under ISO C's string literal limit; joined before compiling. */
/* Dynamic shared memory every grouped kernel may use on every device with a
 * loaded module: the device's opt-in limit less the kernel's static shared
 * memory, at most 160 KB. */
static size_t sharedMax = ANN_BLACKWELL_SHARED_MAX;

static const char *const BlackwellSource[] = {
	"__device__ __forceinline__ float half_bits(unsigned short h)\n"
	"{ float f; asm(\"cvt.f32.f16 %0, %1;\" : \"=f\"(f) : \"h\"(h)); return f; }\n"
	"/* Two E4M3 values (low byte first) to floats. */\n"
	"__device__ __forceinline__ void e4m3x2(unsigned short v, float &lo, float &hi)\n"
	"{\n"
	"  unsigned int h2;\n"
	"  asm(\"cvt.rn.f16x2.e4m3x2 %0, %1;\" : \"=r\"(h2) : \"h\"(v));\n"
	"  lo = half_bits((unsigned short) (h2 & 0xffffu)); hi = half_bits((unsigned short) (h2 >> 16));\n"
	"}\n"
	"/* Two E2M1 values (low nibble first) to floats. */\n"
	"__device__ __forceinline__ void e2m1x2(unsigned int v, float &lo, float &hi)\n"
	"{\n"
	"  unsigned int h2; unsigned short w = (unsigned short) (v & 0xffu);\n"
	"  asm(\"{ .reg .b8 b; cvt.u8.u16 b, %1; cvt.rn.f16x2.e2m1x2 %0, b; }\" : \"=r\"(h2) : \"h\"(w));\n"
	"  lo = half_bits((unsigned short) (h2 & 0xffffu)); hi = half_bits((unsigned short) (h2 >> 16));\n"
	"}\n"
	"#define GROUP 8 /* threads per row; each takes 16-dimension chunks */\n"
	"__device__ __forceinline__ float group_sum(float s)\n"
	"{\n"
	"  for (int o = GROUP / 2; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o, GROUP);\n"
	"  return s;\n"
	"}\n"
	"/* Accumulate (x * scale - q)^2 over 2 decoded values at dimension d. */\n"
	"#define ACC(x0, x1, d) { float t0 = (x0) * scale - query[d], t1 = (x1) * scale - query[(d) + 1]; s += t0 * t0 + t1 * t1; }\n"
	"",
	"/*\n"
	" * Squared distance of one row, computed by a group of GROUP threads (t is\n"
	" * the thread's place in it) and returned to all of them; +inf for no row.\n"
	" * Every thread of the warp must call it: the group sum shuffles.\n"
	" */\n"
	"__device__ __forceinline__ float score4(const unsigned char *data, const unsigned char *scales,\n"
	"    const float *rowScales, unsigned rowBytes, unsigned blocks, unsigned row, const float *query,\n"
	"    int dims, int t)\n"
	"{\n"
	"  float s = 0.0f;\n"
	"  if (row != 0xffffffffu)\n"
	"  {\n"
	"    const uint2 *v = (const uint2 *) (data + (unsigned long long) row * rowBytes);\n"
	"    const unsigned char *sc = scales + (unsigned long long) row * blocks;\n"
	"    float rs = rowScales[row];\n"
	"    for (unsigned c = t; c < blocks; c += GROUP)\n"
	"    {\n"
	"      uint2 w = v[c];\n"
	"      float scale, unused, x0, x1;\n"
	"      e4m3x2((unsigned short) sc[c], scale, unused);\n"
	"      scale *= rs;\n"
	"      int d = c * 16;\n"
	"      if (d + 16 <= dims)\n"
	"      {\n"
	"        for (int b = 0; b < 4; b++) { e2m1x2(w.x >> (8 * b), x0, x1); ACC(x0, x1, d + 2 * b); }\n"
	"        for (int b = 0; b < 4; b++) { e2m1x2(w.y >> (8 * b), x0, x1); ACC(x0, x1, d + 8 + 2 * b); }\n"
	"      }\n"
	"      else\n"
	"        for (int b = 0; b < 8; b++)\n"
	"        {\n"
	"          e2m1x2((b < 4 ? w.x : w.y) >> (8 * (b & 3)), x0, x1);\n"
	"          int e = d + 2 * b;\n"
	"          if (e < dims) { float t0 = x0 * scale - query[e]; s += t0 * t0; }\n"
	"          if (e + 1 < dims) { float t1 = x1 * scale - query[e + 1]; s += t1 * t1; }\n"
	"        }\n"
	"    }\n"
	"  }\n"
	"  s = group_sum(s);\n"
	"  return row == 0xffffffffu ? __int_as_float(0x7f800000) : s;\n"
	"}\n",
	"__device__ __forceinline__ float score8(const unsigned char *data, const float *rowScales, unsigned rowBytes,\n"
	"    unsigned row, const float *query, int dims, int t)\n"
	"{\n"
	"  float s = 0.0f;\n"
	"  if (row != 0xffffffffu)\n"
	"  {\n"
	"    const uint4 *v = (const uint4 *) (data + (unsigned long long) row * rowBytes);\n"
	"    float scale = rowScales[row];\n"
	"    for (unsigned c = t; c < rowBytes / 16; c += GROUP)\n"
	"    {\n"
	"      uint4 w = v[c];\n"
	"      unsigned int words[4] = {w.x, w.y, w.z, w.w};\n"
	"      int d = c * 16;\n"
	"      float x0, x1;\n"
	"      if (d + 16 <= dims)\n"
	"        for (int b = 0; b < 8; b++) { e4m3x2((unsigned short) (words[b >> 1] >> (16 * (b & 1))), x0, x1); ACC(x0, x1, d + 2 * b); }\n"
	"      else\n"
	"        for (int b = 0; b < 8; b++)\n"
	"        {\n"
	"          e4m3x2((unsigned short) (words[b >> 1] >> (16 * (b & 1))), x0, x1);\n"
	"          int e = d + 2 * b;\n"
	"          if (e < dims) { float t0 = x0 * scale - query[e]; s += t0 * t0; }\n"
	"          if (e + 1 < dims) { float t1 = x1 * scale - query[e + 1]; s += t1 * t1; }\n"
	"        }\n"
	"    }\n"
	"  }\n"
	"  s = group_sum(s);\n"
	"  return row == 0xffffffffu ? __int_as_float(0x7f800000) : s;\n"
	"}\n"
	"/* Score every listed row (UINT32_MAX scores +inf); GROUP threads a row, no early exit. */\n"
	"extern \"C\" __global__ void score_nvfp4(const unsigned char *data, const unsigned char *scales,\n"
	"    const float *rowScales, unsigned rowBytes, unsigned blocks, const unsigned *rows, unsigned n,\n"
	"    const float *query, int dims, float *out)\n"
	"{\n"
	"  unsigned i = (blockIdx.x * blockDim.x + threadIdx.x) / GROUP;\n"
	"  float s = score4(data, scales, rowScales, rowBytes, blocks, i < n ? rows[i] : 0xffffffffu, query, dims, threadIdx.x % GROUP);\n"
	"  if (threadIdx.x % GROUP == 0 && i < n) out[i] = s;\n"
	"}\n"
	"/*\n"
	" * The same over several queries' row lists laid end to end: query q owns\n"
	" * rows offsets[q] .. offsets[q + 1] - 1 (mixed-filter groups).\n"
	" */\n"
	"extern \"C\" __global__ void score_nvfp4_seg(const unsigned char *data, const unsigned char *scales,\n"
	"    const float *rowScales, unsigned rowBytes, unsigned blocks, const unsigned *rows, unsigned n,\n"
	"    const unsigned *offsets, int nq, const float *queries, int dims, float *out)\n"
	"{\n"
	"  unsigned i = (blockIdx.x * blockDim.x + threadIdx.x) / GROUP;\n"
	"  int lo = 0, hi = nq;\n"
	"  if (i < n) while (hi - lo > 1) { int mid = (lo + hi) / 2; if (offsets[mid] <= i) lo = mid; else hi = mid; }\n"
	"  float s = score4(data, scales, rowScales, rowBytes, blocks, i < n ? rows[i] : 0xffffffffu,\n"
	"                   queries + (unsigned long long) lo * dims, dims, threadIdx.x % GROUP);\n"
	"  if (threadIdx.x % GROUP == 0 && i < n) out[i] = s;\n"
	"}\n"
	"extern \"C\" __global__ void score_fp8(const unsigned char *data, const float *rowScales,\n"
	"    unsigned rowBytes, const unsigned *rows, unsigned n, const float *query, int dims, float *out)\n"
	"{\n"
	"  unsigned i = (blockIdx.x * blockDim.x + threadIdx.x) / GROUP;\n"
	"  float s = score8(data, rowScales, rowBytes, i < n ? rows[i] : 0xffffffffu, query, dims, threadIdx.x % GROUP);\n"
	"  if (threadIdx.x % GROUP == 0 && i < n) out[i] = s;\n"
	"}\n",
	"",
	"/* Sorted (ascending) 32 smallest of a warp's best list and new candidates. */\n"
	"__device__ __forceinline__ void bitonic_sort(float &d, unsigned &r, int lane)\n"
	"{\n"
	"  for (int k = 2; k <= 32; k <<= 1)\n"
	"    for (int j = k >> 1; j > 0; j >>= 1)\n"
	"    {\n"
	"      float od = __shfl_xor_sync(0xffffffffu, d, j); unsigned orr = __shfl_xor_sync(0xffffffffu, r, j);\n"
	"      bool keepMin = ((lane & j) == 0) == ((lane & k) == 0);\n"
	"      if (keepMin ? od < d : od > d) { d = od; r = orr; }\n"
	"    }\n"
	"}\n"
	"/* best and fresh are ascending across lanes; best becomes the 32 smallest of both. */\n"
	"__device__ __forceinline__ void merge_best(float &bd, unsigned &br, float fd, unsigned fr, int lane)\n"
	"{\n"
	"  float rd = __shfl_sync(0xffffffffu, fd, 31 - lane); unsigned rr = __shfl_sync(0xffffffffu, fr, 31 - lane);\n"
	"  if (rd < bd) { bd = rd; br = rr; }\n"
	"  for (int j = 16; j > 0; j >>= 1)\n"
	"  {\n"
	"    float od = __shfl_xor_sync(0xffffffffu, bd, j); unsigned orr = __shfl_xor_sync(0xffffffffu, br, j);\n"
	"    if ((lane & j) == 0 ? od < bd : od > bd) { bd = od; br = orr; }\n"
	"  }\n"
	"}\n"
	"",
	"/*\n"
	" * Fused scoring and top-32 selection of one query over a list of rows.\n"
	" * Each warp scores four rows a step (8 threads a row) and keeps its own\n"
	" * best 32 in registers, merging new rows only when one beats both its own\n"
	" * 32nd best and the smallest 32nd best any warp of the block has published\n"
	" * (float bits in shared memory; non-negative floats order as integers), so\n"
	" * the scan has no block-wide barrier. Warps then merge per block;\n"
	" * the last block to finish merges every block's list, its warps each\n"
	" * taking a share before warp 0 merges theirs. Output holds positions in\n"
	" * the row list, ascending by distance.\n"
	" */\n"
	"template <int QUANT>\n"
	"__device__ void search(const unsigned char *data, const unsigned char *scales, const float *rowScales,\n"
	"    unsigned rowBytes, unsigned blocks, const unsigned *rows, unsigned n, unsigned perBlock,\n"
	"    const float *query, int dims, float *partD, unsigned *partR, unsigned *counter,\n"
	"    float *outD, unsigned *outR)\n"
	"{\n"
	"  __shared__ float sd[32 * 32]; __shared__ unsigned sr[32 * 32]; __shared__ bool last; __shared__ unsigned bound;\n"
	"  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;\n"
	"  int group = lane / GROUP, t = lane % GROUP, groups = 32 / GROUP;\n"
	"  const float inf = __int_as_float(0x7f800000);\n"
	"  float bd = inf, threshold = inf; unsigned br = 0xffffffffu;\n"
	"  unsigned begin = blockIdx.x * perBlock, end = min(n, begin + perBlock);\n"
	"  if (threadIdx.x == 0) bound = 0x7f800000u; /* block-wide bound, float bits */\n"
	"  __syncthreads();\n"
	"  volatile unsigned *shared = &bound;\n"
	"  for (unsigned base = begin + warp * groups; base < end; base += warps * groups)\n"
	"  {\n"
	"    /* Any warp's 32nd best bounds the block's 32nd best: rows past it can go. */\n"
	"    threshold = fminf(threshold, __uint_as_float(*shared));\n"
	"    unsigned i = base + group;\n"
	"    unsigned row = i < end ? rows[i] : 0xffffffffu;\n"
	"    float s = QUANT == 1 ? score4(data, scales, rowScales, rowBytes, blocks, row, query, dims, t)\n"
	"                         : score8(data, rowScales, rowBytes, row, query, dims, t);\n"
	"    bool pass = t == 0 && row != 0xffffffffu && s < threshold;\n"
	"    if (__ballot_sync(0xffffffffu, pass))\n"
	"    {\n"
	"      float fd = pass ? s : inf; unsigned fr = pass ? i : 0xffffffffu;\n"
	"      bitonic_sort(fd, fr, lane);\n"
	"      merge_best(bd, br, fd, fr, lane);\n"
	"      threshold = __shfl_sync(0xffffffffu, bd, 31);\n"
	"      if (lane == 0 && threshold < inf) atomicMin(&bound, __float_as_uint(threshold));\n"
	"    }\n"
	"  }\n"
	"  sd[warp * 32 + lane] = bd; sr[warp * 32 + lane] = br;\n"
	"  __syncthreads();\n"
	"  if (warp == 0)\n"
	"  {\n"
	"    for (int w = 1; w < warps; w++) merge_best(bd, br, sd[w * 32 + lane], sr[w * 32 + lane], lane);\n"
	"    partD[blockIdx.x * 32 + lane] = bd; partR[blockIdx.x * 32 + lane] = br;\n"
	"    __threadfence();\n"
	"    if (lane == 0) last = atomicAdd(counter, 1) == gridDim.x - 1;\n"
	"  }\n"
	"  __syncthreads();\n"
	"  if (!last) return;\n"
	"  __threadfence();\n"
	"  bd = inf; br = 0xffffffffu;\n"
	"  for (unsigned b = warp; b < gridDim.x; b += warps)\n"
	"    merge_best(bd, br, ((volatile float *) partD)[b * 32 + lane], ((volatile unsigned *) partR)[b * 32 + lane], lane);\n"
	"  sd[warp * 32 + lane] = bd; sr[warp * 32 + lane] = br;\n"
	"  __syncthreads();\n"
	"  if (warp != 0) return;\n"
	"  for (int w = 1; w < warps; w++) merge_best(bd, br, sd[w * 32 + lane], sr[w * 32 + lane], lane);\n"
	"  outD[lane] = bd; outR[lane] = br;\n"
	"  if (lane == 0) { counter[0] = 0; counter[1] = 0x7f800000u; }\n"
	"}\n",
	"extern \"C\" __global__ void search_nvfp4(const unsigned char *data, const unsigned char *scales,\n"
	"    const float *rowScales, unsigned rowBytes, unsigned blocks, const unsigned *rows, unsigned n,\n"
	"    unsigned perBlock, const float *query, int dims, float *partD, unsigned *partR, unsigned *counter,\n"
	"    float *outD, unsigned *outR)\n"
	"{ search<1>(data, scales, rowScales, rowBytes, blocks, rows, n, perBlock, query, dims, partD, partR, counter, outD, outR); }\n"
	"extern \"C\" __global__ void search_fp8(const unsigned char *data, const float *rowScales, unsigned rowBytes,\n"
	"    const unsigned *rows, unsigned n, unsigned perBlock, const float *query, int dims, float *partD,\n"
	"    unsigned *partR, unsigned *counter, float *outD, unsigned *outR)\n"
	"{ search<2>(data, 0, rowScales, rowBytes, 0, rows, n, perBlock, query, dims, partD, partR, counter, outD, outR); }\n",
	"",
	"/* Packed half2 helpers through PTX, so no CUDA headers are needed. */\n"
	"__device__ __forceinline__ unsigned hmul2(unsigned a, unsigned b)\n"
	"{ unsigned r; asm(\"mul.rn.f16x2 %0, %1, %2;\" : \"=r\"(r) : \"r\"(a), \"r\"(b)); return r; }\n"
	"__device__ __forceinline__ unsigned e2m1pair(unsigned byte)\n"
	"{\n"
	"  unsigned h2; unsigned short w = (unsigned short) (byte & 0xffu);\n"
	"  asm(\"{ .reg .b8 b; cvt.u8.u16 b, %1; cvt.rn.f16x2.e2m1x2 %0, b; }\" : \"=r\"(h2) : \"h\"(w));\n"
	"  return h2;\n"
	"}\n"
	"__device__ __forceinline__ unsigned e4m3pair(unsigned word)\n"
	"{ unsigned h2; asm(\"cvt.rn.f16x2.e4m3x2 %0, %1;\" : \"=r\"(h2) : \"h\"((unsigned short) (word & 0xffffu))); return h2; }\n"
	"/* An E4M3 block scale as a half2 holding it twice. */\n"
	"__device__ __forceinline__ unsigned scale2(unsigned byte)\n"
	"{ unsigned h2 = e4m3pair(byte & 0xffu); return (h2 & 0xffffu) | (h2 << 16); }\n"
	"__device__ __forceinline__ unsigned f32x2(float lo, float hi)\n"
	"{ unsigned r; asm(\"cvt.rn.f16x2.f32 %0, %1, %2;\" : \"=r\"(r) : \"f\"(hi), \"f\"(lo)); return r; }\n"
	"__device__ __forceinline__ float sumsq2(unsigned h2)\n"
	"{ float lo = half_bits((unsigned short) (h2 & 0xffffu)), hi = half_bits((unsigned short) (h2 >> 16)); return lo * lo + hi * hi; }\n"
	"/* D += A (16 rows x 16 k, FP16) * B (16 k x 8 queries, FP16), FP32 accumulate. */\n"
	"__device__ __forceinline__ void mma16816(float *c, unsigned a0, unsigned a1, unsigned a2, unsigned a3,\n"
	"    unsigned b0, unsigned b1)\n"
	"{\n"
	"  asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
	"      : \"+f\"(c[0]), \"+f\"(c[1]), \"+f\"(c[2]), \"+f\"(c[3]) : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(b0), \"r\"(b1));\n"
	"}\n",
	"/*\n"
	" * Distances of up to 8 * NT queries to every listed row on the tensor cores,\n"
	" * written as a [queries][rows] matrix for a batched top-k selection. Each\n"
	" * warp takes 16 rows at a time, loads their quantized bytes and block\n"
	" * scales up front, and multiplies them against every query with\n"
	" * mma.sync (FP16 in, FP32 out): the FP4 and FP8 conversion instructions\n"
	" * yield the A fragments directly (a half2 is two neighbouring dimensions\n"
	" * of one row), block scales are folded into them, and the B fragments\n"
	" * are two 32-bit loads of the half queries in shared memory, per tile of\n"
	" * 8 queries: a row's loads and conversions are shared by NT tiles (16,\n"
	" * 32 or 64 queries). Distances\n"
	" * follow from |x|^2 + |q|^2 - 2 x.q with |x|^2 summed from the same half\n"
	" * values. Rows absent from the index (UINT32_MAX) score +inf.\n"
	" */\n"
	"#define KSEG 8 /* k-steps loaded per round: 128 dimensions */\n"
	"/*\n"
	" * FUSED: instead of the whole matrix, append (distance, position) to query\n"
	" * q's candidate list (out and candR, n slots per query, counts[q] filled)\n"
	" * only when the distance is at most bounds[q * 32 + 31], the 32nd best\n"
	" * of a sample of the rows: no row of the true best 32 is above it.\n"
	" */\n"
	"template <int QUANT, int NT, int FUSED, int CD = 0>\n"
	"/* ANN_TUNE (ANN_FUSE_TUNE, fused scan only): bit 1 tests hits as fma(-2 rs, x.q, rs^2 |x|^2) against\n"
	" * cutoff - |q|^2 (slightly loose), voting per output so only warps with a candidate take the\n"
	" * exact test and stage append; bit 2 a 128-dimension NVFP4 kernel with constant loop bounds;\n"
	" * bit 4 B fragments by ldmatrix.x4; bit 8 block scales broadcast with one prmt; bit 32 (with 16)\n"
	" * keeps output j's x.q in list slot j (one predicated store, no list pointer); bit 64 (check, slow)\n"
	" * prints any hit the loose test of bit 16 would miss. */\n"
	"#ifndef ANN_TUNE\n"
	"#define ANN_TUNE 0\n"
	"#endif\n"
	"/* Halves past each query's row in shared memory (QueryPad): 8 moves the\n"
	" * eight lanes reading one B fragment off a single bank. */\n"
	"#ifndef ANN_QPAD\n"
	"#define ANN_QPAD 0\n"
	"#endif\n"
	"/* ANN_FUSE_ABLATE (timing only; results are wrong): 1 no appends, 2 no MMA,\n"
	" * 4 rows from a hot 1,024-row set, 8 constant B fragments. */\n"
	"#ifndef ANN_ABLATE\n"
	"#define ANN_ABLATE 0\n"
	"#endif\n"
	"/* ANN_STAGE (default 32; ANN_FUSE_APPEND=global sets 0): the fused scan stages each block's hits per query in\n"
	" * shared memory (ANN_STAGE of them) and appends them with one global atomic\n"
	" * per query per block; a hit past the stage takes the global path. */\n"
	"/* ANN_FUSE_LAYOUT=1: the fused NVFP4 scan reads the lane-contiguous copy. */\n"
	"#ifndef ANN_LAYOUT\n"
	"#define ANN_LAYOUT 0\n"
	"#endif\n"
	"#ifndef ANN_STAGE\n"
	"#define ANN_STAGE 0\n"
	"#endif\n",
	"/* ANN_ROWNORM (ANN_FUSE_ROWNORM): 1 the fused NVFP4 scan loads each row's (scale, |x|^2)\n"
	" * from rowNorm, made once per shard by row_norm_nvfp4 (FUSED 2) with the same code;\n"
	" * 2 computes as before and reports rows whose stored pair differs. */\n"
	"#ifndef ANN_ROWNORM\n"
	"#define ANN_ROWNORM 0\n"
	"#endif\n"
	"/* ANN_APPEND_MODE (ANN_FUSE_APPEND_MODE) bit 1: the fused scan reads each query's norm and cutoff\n"
	" * as one shared float2 (cutoff -inf past nq, so no nn < nq or i < n test). */\n"
	"#ifndef ANN_APPEND_MODE\n"
	"#define ANN_APPEND_MODE 0\n"
	"#endif\n"
	"/* ANN_PREFETCH (ANN_FUSE_PREFETCH, fused scan only): 2 loads row indices two tiles ahead and\n"
	" * L2-prefetches the next tile's rows and block scales; 16 adds the next tile's row norm\n"
	" * pair (ANN_ROWNORM) or row scale. */\n"
	"#ifndef ANN_PREFETCH\n"
	"#define ANN_PREFETCH 0\n"
	"#endif\n"
	"#define PFON(b) (FUSED == 1 && (ANN_PREFETCH & (b)))\n"
	"/* ANN_APPEND_MODE bit 4: each lane keeps the tile's hits in its own ANN_BCAP-entry shared list\n"
	" * (2 stores, no atomics in the row loop; a hit past it appends directly; assumes 256 threads)\n"
	" * and stages them per query after the tile. */\n"
	"#ifndef ANN_QVEC\n"
	"#define ANN_QVEC 0\n"
	"#endif\n"
	"#ifndef ANN_BCAP\n"
	"#define ANN_BCAP 32\n"
	"#endif\n"
	"__device__ void score_mma(const unsigned char *data, const unsigned char *scales,\n"
	"    const float *rowScales, unsigned rowBytes, unsigned blocks_, const unsigned *rows, unsigned n,\n"
	"    const float *queries, int nq, int dims_, float *out, const float *bounds, unsigned *candR,\n"
	"    unsigned *counts, unsigned cap, const float2 *rowNorm)\n"
	"{\n"
	"  const unsigned blocks = CD ? CD / 16 : blocks_; const int dims = CD ? CD : dims_;\n"
	"  constexpr bool PRE = FUSED == 1 && (ANN_ROWNORM == 1 || ANN_ROWNORM == 3) && QUANT == 1;\n"
	"  constexpr bool EARLY = PRE && ANN_ROWNORM == 3; /* issue the pair load before the row's k loop */\n"
	"  constexpr bool PK = FUSED == 1 && (ANN_APPEND_MODE & 1);\n"
	"  extern __shared__ unsigned short qh[];\n"
	"  __shared__ float qn[NT * 8];\n"
	"  __shared__ float th[NT * 8];\n"
	"  __shared__ float2 qt[PK ? NT * 8 : 1];\n"
	"  constexpr bool TV = FUSED == 1 && (ANN_TUNE & 1) && PK, TL = FUSED == 1 && (ANN_TUNE & 4) && !(ANN_ABLATE & 8);\n"
	"  constexpr bool PB = FUSED == 1 && (ANN_APPEND_MODE & 4) && !TV;\n"
	"  /* bit 16 (with the per-lane lists): the loose test above; a hit stores x.q alone and sets\n"
	"   * bit j of hm, and the flush recomputes the exact distance and test. */\n"
	"  constexpr bool TC = PB && PK && (ANN_TUNE & 16) && ANN_BCAP >= NT * 4;\n"
	"  __shared__ __align__(8) float qc[TV || TC ? NT * 8 : 1];\n"
	"  __shared__ float bD[PB ? ANN_BCAP * 256 : 1];\n"
	"  __shared__ unsigned char bJ[PB && !TC ? ANN_BCAP * 256 : 1];\n"
	"  unsigned bcnt = 0;\n"
	"  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;\n"
	"  int g = lane >> 2, t = lane & 3;\n"
	"  int qstride = ((dims + 15) & ~15) + ANN_QPAD;\n"
	"  float sink = 0.0f;\n"
	"  const float inf = __int_as_float(0x7f800000);\n"
	"  __shared__ float sD[FUSED && ANN_STAGE ? NT * 8 * ANN_STAGE : 1];\n"
	"  __shared__ unsigned sR[FUSED && ANN_STAGE ? NT * 8 * ANN_STAGE : 1];\n"
	"  __shared__ unsigned sCount[NT * 8];\n"
	"  if (threadIdx.x < NT * 8) sCount[threadIdx.x] = 0;\n"
	"  auto append = [&](unsigned q, float d, unsigned r) {\n"
	"    unsigned at = ANN_STAGE ? atomicAdd(sCount + q, 1u) : ANN_STAGE;\n"
	"    if (ANN_STAGE && at < ANN_STAGE) { sD[q * ANN_STAGE + at] = d; sR[q * ANN_STAGE + at] = r; }\n"
	"    else\n"
	"    {\n"
	"      at = atomicAdd(counts + q, 1u);\n"
	"      if (at < cap) { out[(unsigned long long) q * cap + at] = d; candR[(unsigned long long) q * cap + at] = r; }\n"
	"    }\n"
	"  };\n",
	"  /* ANN_QVEC (ANN_FUSE_QVEC, default 3; 0 turns it off) bit 1: score matrices (FUSED 0), bit 2: the fused scan stage their\n"
	"   * half queries four floats per load, the loads of a thread unrolled, instead of one. */\n"
	"  if ((ANN_QVEC & (FUSED == 0 ? 1 : FUSED == 1 ? 2 : 0)) && (dims & 3) == 0 && (qstride & 3) == 0)\n"
	"  {\n"
	"    int q4 = qstride >> 2;\n"
	"#pragma unroll 4\n"
	"    for (int i = threadIdx.x; i < NT * 8 * q4; i += blockDim.x)\n"
	"    {\n"
	"      int q = i / q4, d = (i - q * q4) << 2;\n"
	"      float4 v = (q < nq && d < dims) ? *(const float4 *) (queries + q * dims + d) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);\n"
	"      unsigned short h0, h1, h2, h3;\n"
	"      asm(\"cvt.rn.f16.f32 %0, %1;\" : \"=h\"(h0) : \"f\"(v.x)); asm(\"cvt.rn.f16.f32 %0, %1;\" : \"=h\"(h1) : \"f\"(v.y));\n"
	"      asm(\"cvt.rn.f16.f32 %0, %1;\" : \"=h\"(h2) : \"f\"(v.z)); asm(\"cvt.rn.f16.f32 %0, %1;\" : \"=h\"(h3) : \"f\"(v.w));\n"
	"      *(uint2 *) (qh + 4 * i) = make_uint2((unsigned) h0 | ((unsigned) h1 << 16), (unsigned) h2 | ((unsigned) h3 << 16));\n"
	"    }\n"
	"  }\n"
	"  else\n",
	"  for (int i = threadIdx.x; i < NT * 8 * qstride; i += blockDim.x)\n"
	"  {\n"
	"    int q = i / qstride, d = i - q * qstride;\n"
	"    float v = (q < nq && d < dims) ? queries[q * dims + d] : 0.0f;\n"
	"    unsigned short h; asm(\"cvt.rn.f16.f32 %0, %1;\" : \"=h\"(h) : \"f\"(v)); qh[i] = h;\n"
	"  }\n"
	"  __syncthreads();\n"
	"  if (threadIdx.x < NT * 8)\n"
	"  {\n"
	"    float s = 0.0f;\n"
	"    for (int d = 0; d < dims; d++) { float v = half_bits(qh[threadIdx.x * qstride + d]); s += v * v; }\n"
	"    qn[threadIdx.x] = s;\n"
	"    if (FUSED) th[threadIdx.x] = threadIdx.x < nq ? bounds[threadIdx.x * 32 + 31] : 0.0f;\n"
	"    if (PK) qt[threadIdx.x] = make_float2(s, threadIdx.x < nq ? bounds[threadIdx.x * 32 + 31] : -inf);\n"
	"    if (TV || TC)\n"
	"    {\n"
	"      float c = threadIdx.x < nq ? bounds[threadIdx.x * 32 + 31] : -inf;\n"
	"      qc[threadIdx.x] = c == -inf ? -inf : c - s + 1e-5f * (fabsf(c) + s);\n"
	"    }\n"
	"  }\n"
	"  __syncthreads();\n"
	"  unsigned stride = gridDim.x * warps * 16, first = (blockIdx.x * warps + warp) * 16;\n"
	"  unsigned pf0 = 0xffffffffu, pf1 = 0xffffffffu, pq0 = 0xffffffffu, pq1 = 0xffffffffu;\n"
	"  if (PFON(2))\n"
	"  {\n"
	"    pf0 = first + g < n ? rows[first + g] : 0xffffffffu; pf1 = first + g + 8 < n ? rows[first + g + 8] : 0xffffffffu;\n"
	"    unsigned b = first + stride; pq0 = b + g < n ? rows[b + g] : 0xffffffffu; pq1 = b + g + 8 < n ? rows[b + g + 8] : 0xffffffffu;\n"
	"  }\n"
	"  for (unsigned base = first; base < n; base += stride)\n",
	"  {\n"
	"    unsigned i0 = base + g, i1 = base + g + 8;\n"
	"    unsigned row0, row1;\n"
	"    if (PFON(2))\n"
	"    {\n"
	"      unsigned b = base + 2 * stride;\n"
	"      row0 = pf0; row1 = pf1; pf0 = pq0; pf1 = pq1;\n"
	"      pq0 = b + g < n ? rows[b + g] : 0xffffffffu; pq1 = b + g + 8 < n ? rows[b + g + 8] : 0xffffffffu;\n"
	"      unsigned pr = (t & 1) ? pf1 : pf0; pr = pr == 0xffffffffu ? 0u : pr;\n"
	"      const unsigned char *pp = (t < 2) ? data + (unsigned long long) pr * rowBytes : scales + (unsigned long long) pr * blocks;\n"
	"      if (QUANT == 1) asm volatile(\"prefetch.global.L2 [%0];\" :: \"l\"(pp));\n"
	"      if (PFON(16) && t < 2 && (PRE || rowScales))\n"
	"        asm volatile(\"prefetch.global.L2 [%0];\" :: \"l\"(PRE ? (const void *) (rowNorm + pr) : (const void *) (rowScales + pr)));\n"
	"    }\n"
	"    else { row0 = i0 < n ? (FUSED == 2 ? i0 : rows[i0]) : 0xffffffffu; row1 = i1 < n ? (FUSED == 2 ? i1 : rows[i1]) : 0xffffffffu; }\n"
	"    bool v0 = row0 != 0xffffffffu, v1 = row1 != 0xffffffffu;\n"
	"    unsigned r0 = v0 ? row0 : 0u, r1 = v1 ? row1 : 0u;\n"
	"    if (ANN_ABLATE & 4) { r0 = i0 & 1023u; r1 = i1 & 1023u; }\n"
	"    float acc[NT * 4]; float nx0 = 0.0f, nx1 = 0.0f;\n"
	"    float2 pn0 = {0.0f, 0.0f}, pn1 = {0.0f, 0.0f};\n"
	"    if (EARLY) { pn0 = rowNorm[r0]; pn1 = rowNorm[r1]; }\n"
	"#pragma unroll\n"
	"    for (int j = 0; j < NT * 4; j++) acc[j] = 0.0f;\n"
	"    for (unsigned seg = 0; seg < blocks; seg += KSEG)\n",
	"    {\n"
	"      uint2 w0[KSEG], w1[KSEG]; uint2 e0 = {0, 0}, e1 = {0, 0};\n"
	"      if (QUANT == 3)\n"
	"      {\n"
	"        /* FP16 dataset: this lane's two pairs of a k-step are 32-bit words at k = 2t and 2t + 8. */\n"
	"        const unsigned short *p0 = (const unsigned short *) (data + (unsigned long long) r0 * rowBytes) + seg * 16 + 2 * t;\n"
	"        const unsigned short *p1 = (const unsigned short *) (data + (unsigned long long) r1 * rowBytes) + seg * 16 + 2 * t;\n"
	"#pragma unroll\n"
	"        for (int k = 0; k < KSEG; k++)\n"
	"          if (seg + k < blocks)\n"
	"          {\n"
	"            w0[k].x = *(const unsigned *) (p0 + k * 16); w0[k].y = *(const unsigned *) (p0 + k * 16 + 8);\n"
	"            w1[k].x = *(const unsigned *) (p1 + k * 16); w1[k].y = *(const unsigned *) (p1 + k * 16 + 8);\n"
	"          }\n"
	"      }\n"
	"      else if (QUANT == 4)\n"
	"      {\n"
	"        /* FP32 dataset: float pairs, converted to half2 at use. */\n"
	"        const float2 *p0 = (const float2 *) (data + (unsigned long long) r0 * rowBytes) + seg * 8 + t;\n"
	"        const float2 *p1 = (const float2 *) (data + (unsigned long long) r1 * rowBytes) + seg * 8 + t;\n"
	"#pragma unroll\n"
	"        for (int k = 0; k < KSEG; k++)\n"
	"          if (seg + k < blocks)\n"
	"          {\n"
	"            float2 a = p0[k * 8], b = p0[k * 8 + 4], c = p1[k * 8], d = p1[k * 8 + 4];\n"
	"            w0[k].x = f32x2(a.x, a.y); w0[k].y = f32x2(b.x, b.y); w1[k].x = f32x2(c.x, c.y); w1[k].y = f32x2(d.x, d.y);\n"
	"          }\n"
	"      }\n"
	"      else if (QUANT == 1)\n",
	"      {\n"
	"        const uint2 *p0 = (const uint2 *) (data + (unsigned long long) r0 * rowBytes) + seg;\n"
	"        const uint2 *p1 = (const uint2 *) (data + (unsigned long long) r1 * rowBytes) + seg;\n"
	"#pragma unroll\n"
	"        if ((ANN_LAYOUT & 1) && FUSED == 1)\n"
	"        {\n"
	"          /* Lane-contiguous copy (FuseLayout): in each 64-byte chunk of 8 blocks lane t's 16 bytes are\n"
	"           * contiguous, byte 2k block k's .x byte t and 2k + 1 its .y byte t. */\n"
	"          uint4 q0 = ((const uint4 *) p0)[t], q1 = ((const uint4 *) p1)[t];\n"
	"          unsigned c0[4] = {q0.x, q0.y, q0.z, q0.w}, c1[4] = {q1.x, q1.y, q1.z, q1.w};\n"
	"#pragma unroll\n"
	"          for (int k = 0; k < KSEG; k++)\n"
	"          { w0[k].x = c0[k >> 1] >> (16 * (k & 1)); w0[k].y = w0[k].x >> 8; w1[k].x = c1[k >> 1] >> (16 * (k & 1)); w1[k].y = w1[k].x >> 8; }\n"
	"        }\n"
	"        else\n"
	"#pragma unroll\n"
	"        for (int k = 0; k < KSEG; k++) if (seg + k < blocks) { w0[k] = p0[k]; w1[k] = p1[k]; }\n"
	"        /* 8 block scales at once when rows hold whole multiples of 8 (aligned, in bounds); else bytes. */\n"
	"        if ((blocks & 7) == 0)\n"
	"        {\n"
	"          e0 = *(const uint2 *) (scales + (unsigned long long) r0 * blocks + seg);\n"
	"          e1 = *(const uint2 *) (scales + (unsigned long long) r1 * blocks + seg);\n"
	"        }\n"
	"        else\n"
	"        {\n"
	"          const unsigned char *s0 = scales + (unsigned long long) r0 * blocks + seg;\n"
	"          const unsigned char *s1 = scales + (unsigned long long) r1 * blocks + seg;\n"
	"#pragma unroll\n"
	"          for (int k = 0; k < KSEG; k++)\n"
	"            if (seg + k < blocks)\n"
	"            {\n"
	"              unsigned b0 = s0[k], b1 = s1[k];\n"
	"              if (k < 4) { e0.x |= b0 << (8 * k); e1.x |= b1 << (8 * k); } else { e0.y |= b0 << (8 * (k - 4)); e1.y |= b1 << (8 * (k - 4)); }\n"
	"            }\n"
	"        }\n"
	"      }\n"
	"      else\n"
	"      {\n"
	"        const uint4 *p0 = (const uint4 *) (data + (unsigned long long) r0 * rowBytes) + seg;\n"
	"        const uint4 *p1 = (const uint4 *) (data + (unsigned long long) r1 * rowBytes) + seg;\n"
	"#pragma unroll\n"
	"        for (int k = 0; k < KSEG; k++)\n"
	"          if (seg + k < blocks)\n"
	"          {\n"
	"            uint4 x0 = p0[k], x1 = p1[k];\n"
	"            w0[k].x = t < 2 ? x0.x : x0.y; w0[k].y = t < 2 ? x0.z : x0.w;\n"
	"            w1[k].x = t < 2 ? x1.x : x1.y; w1[k].y = t < 2 ? x1.z : x1.w;\n"
	"          }\n"
	"      }\n"
	"#pragma unroll\n"
	"      for (int k = 0; k < KSEG; k++)\n"
	"      {\n"
	"        if (seg + k >= blocks) break;\n"
	"        unsigned a0, a1, a2, a3;\n"
	"        if (QUANT == 1)\n"
	"        {\n"
	"          unsigned sc0 = scale2((k < 4 ? e0.x : e0.y) >> (8 * (k & 3))), sc1 = scale2((k < 4 ? e1.x : e1.y) >> (8 * (k & 3)));\n"
	"          if (FUSED == 1 && (ANN_TUNE & 8))\n"
	"          {\n"
	"            unsigned h0 = e4m3pair((k < 4 ? e0.x : e0.y) >> (8 * (k & 3))), h1 = e4m3pair((k < 4 ? e1.x : e1.y) >> (8 * (k & 3)));\n"
	"            asm(\"prmt.b32 %0, %1, %1, 0x1010;\" : \"=r\"(sc0) : \"r\"(h0)); asm(\"prmt.b32 %0, %1, %1, 0x1010;\" : \"=r\"(sc1) : \"r\"(h1));\n"
	"          }\n"
	"          int sh = ((ANN_LAYOUT & 1) && FUSED == 1) ? 0 : 8 * t;\n"
	"          a0 = hmul2(e2m1pair(w0[k].x >> sh), sc0); a2 = hmul2(e2m1pair(w0[k].y >> sh), sc0);\n"
	"          a1 = hmul2(e2m1pair(w1[k].x >> sh), sc1); a3 = hmul2(e2m1pair(w1[k].y >> sh), sc1);\n"
	"        }\n"
	"        else if (QUANT == 2)\n",
	"        {\n"
	"          a0 = e4m3pair(w0[k].x >> (16 * (t & 1))); a2 = e4m3pair(w0[k].y >> (16 * (t & 1)));\n"
	"          a1 = e4m3pair(w1[k].x >> (16 * (t & 1))); a3 = e4m3pair(w1[k].y >> (16 * (t & 1)));\n"
	"        }\n"
	"        else\n"
	"        { a0 = w0[k].x; a2 = w0[k].y; a1 = w1[k].x; a3 = w1[k].y; }\n"
	"        /* Dimensions past dims (a dataset's padding) count for nothing. */\n"
	"        if ((seg + k) * 16 + 16 > dims)\n"
	"        {\n"
	"          int d0 = (seg + k) * 16 + 2 * t, d2 = d0 + 8;\n"
	"          unsigned m0 = d0 + 1 < dims ? 0xffffffffu : d0 < dims ? 0x0000ffffu : 0u;\n"
	"          unsigned m2 = d2 + 1 < dims ? 0xffffffffu : d2 < dims ? 0x0000ffffu : 0u;\n"
	"          a0 &= m0; a1 &= m0; a2 &= m2; a3 &= m2;\n"
	"        }\n"
	"        if (!PRE) { nx0 += sumsq2(a0) + sumsq2(a2); nx1 += sumsq2(a1) + sumsq2(a3); }\n"
	"        const unsigned short *qk = qh + (seg + k) * 16 + 2 * t;\n"
	"        if (TL)\n"
	"        {\n"
	"          /* x4: matrices (tile np, k 0-7), (np, k 8-15), (np + 1, 0-7), (np + 1, 8-15); lane l\n"
	"           * addresses row l & 7 of matrix l >> 3. */\n"
	"          const unsigned short *ql = qh + (seg + k) * 16 + ((lane >> 4) * 8 + (lane & 7)) * qstride + ((lane >> 3) & 1) * 8;\n"
	"          unsigned qa; asm(\"{ .reg .u64 a; cvta.to.shared.u64 a, %1; cvt.u32.u64 %0, a; }\" : \"=r\"(qa) : \"l\"(ql));\n"
	"#pragma unroll\n"
	"          for (int np = 0; np < NT; np += 2)\n"
	"          {\n"
	"            unsigned b0, b1, b2, b3;\n"
	"            asm volatile(\"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\"\n"
	"                : \"=r\"(b0), \"=r\"(b1), \"=r\"(b2), \"=r\"(b3) : \"r\"(qa + np * 16 * qstride));\n"
	"            mma16816(acc + np * 4, a0, a1, a2, a3, b0, b1);\n"
	"            if (np + 1 < NT) mma16816(acc + np * 4 + 4, a0, a1, a2, a3, b2, b3);\n"
	"          }\n"
	"          continue;\n"
	"        }\n"
	"#pragma unroll\n"
	"        for (int nt = 0; nt < (FUSED == 2 ? 0 : NT); nt++)\n"
	"        {\n"
	"          const unsigned short *qq = qk + (nt * 8 + g) * qstride;\n"
	"          unsigned b0 = (ANN_ABLATE & 8) ? lane * 0x3c003c00u : *(const unsigned *) qq;\n"
	"          unsigned b1 = (ANN_ABLATE & 8) ? b0 ^ (unsigned) k : *(const unsigned *) (qq + 8);\n"
	"          if (ANN_ABLATE & 2) acc[nt * 4] += __int_as_float((a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1) & 0x3fffffffu);\n"
	"          else mma16816(acc + nt * 4, a0, a1, a2, a3, b0, b1);\n"
	"        }\n"
	"      }\n"
	"    }\n",
	"    float rs0, rs1;\n"
	"    if (PRE) { float2 p0 = EARLY ? pn0 : rowNorm[r0], p1 = EARLY ? pn1 : rowNorm[r1]; rs0 = p0.x; nx0 = p0.y; rs1 = p1.x; nx1 = p1.y; }\n"
	"    else\n"
	"    {\n"
	"      nx0 += __shfl_xor_sync(0xffffffffu, nx0, 1); nx0 += __shfl_xor_sync(0xffffffffu, nx0, 2);\n"
	"      nx1 += __shfl_xor_sync(0xffffffffu, nx1, 1); nx1 += __shfl_xor_sync(0xffffffffu, nx1, 2);\n"
	"      rs0 = rowScales ? rowScales[r0] : 1.0f; rs1 = rowScales ? rowScales[r1] : 1.0f;\n"
	"    }\n"
	"    if (FUSED == 2)\n"
	"    {\n"
	"      float2 *o = (float2 *) out;\n"
	"      if (t == 0 && v0) o[r0] = make_float2(rs0, nx0);\n"
	"      if (t == 0 && v1) o[r1] = make_float2(rs1, nx1);\n"
	"      continue;\n"
	"    }\n"
	"    if (FUSED == 1 && ANN_ROWNORM == 2 && QUANT == 1 && rowNorm && t == 0)\n"
	"    {\n"
	"      float2 p0 = rowNorm[r0], p1 = rowNorm[r1];\n"
	"      if ((v0 && (p0.x != rs0 || p0.y != nx0)) || (v1 && (p1.x != rs1 || p1.y != nx1)))\n"
	"        printf(\"rownorm mismatch rows %u %u: %.9g/%.9g %.9g/%.9g\\n\", r0, r1, p0.y, nx0, p1.y, nx1);\n"
	"    }\n",
	"    const float nan = __int_as_float(0x7fffffff);\n"
	"    float A0 = v0 ? rs0 * rs0 * nx0 * 0.99999f : nan, A1 = v1 ? rs1 * rs1 * nx1 * 0.99999f : nan;\n"
	"    float m0 = -2.0f * rs0, m1 = -2.0f * rs1;\n"
	"    unsigned hm = 0, sp = threadIdx.x;\n"
	"    /* acc[nt * 4 + c]: c0 (row g, query nt*8 + 2t), c1 (g, +1), c2 (row g + 8, 2t), c3 (g + 8, +1) */\n"
	"#pragma unroll\n"
	"    for (int j = 0; j < NT * 4; j++)\n"
	"    {\n"
	"      int nn = (j >> 2) * 8 + 2 * t + (j & 1);\n"
	"      unsigned i = (j & 2) ? i1 : i0; bool vr = (j & 2) ? v1 : v0;\n"
	"      float nx = (j & 2) ? nx1 : nx0; float rs = (j & 2) ? rs1 : rs0;\n",
	"      if (TV)\n"
	"      {\n"
	"        float2 cc = *(const float2 *) (qc + (j >> 2) * 8 + 2 * t);\n"
	"        bool pre = fmaf((j & 2) ? m1 : m0, acc[j], (j & 2) ? A1 : A0) <= ((j & 1) ? cc.y : cc.x);\n"
	"        if (__any_sync(0xffffffffu, pre) && pre)\n"
	"        {\n"
	"          float2 qv2 = qt[nn];\n"
	"          float dist = fmaxf(rs * rs * nx + qv2.x - 2.0f * rs * acc[j], 0.0f);\n"
	"          if (dist <= qv2.y) append(nn, dist, i);\n"
	"        }\n"
	"        continue;\n"
	"      }\n"
	"      if (TC)\n"
	"      {\n"
	"        float2 cc = *(const float2 *) (qc + (j >> 2) * 8 + 2 * t);\n"
	"        if (ANN_TUNE & 64)\n"
	"        {\n"
	"          if (blockIdx.x == 0 && threadIdx.x == 0 && j == 0 && base == first) printf(\"tc alive n %u\\n\", n);\n"
	"          float2 qd = qt[nn]; float dd = fmaxf(rs * rs * nx + qd.x - 2.0f * rs * acc[j], 0.0f);\n"
	"          float pp = fmaf((j & 2) ? m1 : m0, acc[j], (j & 2) ? A1 : A0), pc = (j & 1) ? cc.y : cc.x;\n"
	"          if (vr && dd <= qd.y && !(pp <= pc)) printf(\"tc miss q %d i %u d %.9g c %.9g p %.9g pc %.9g qn %.9g\\n\", nn, i, dd, qd.y, pp, pc, qd.x);\n"
	"        }\n"
	"        if (fmaf((j & 2) ? m1 : m0, acc[j], (j & 2) ? A1 : A0) <= ((j & 1) ? cc.y : cc.x))\n"
	"        { if (ANN_TUNE & 32) bD[j * 256 + threadIdx.x] = acc[j]; else { bD[sp] = acc[j]; sp += 256; } hm |= 1u << j; }\n"
	"        continue;\n"
	"      }\n"
	"      float2 qv = PK ? qt[nn] : make_float2(qn[nn], 0.0f);\n"
	"      float dist = fmaxf(rs * rs * nx + qv.x - 2.0f * rs * acc[j], 0.0f);\n"
	"      if (FUSED && (ANN_ABLATE & 1))\n"
	"        sink += dist;\n"
	"      else if (FUSED)\n"
	"      {\n"
	"        bool hit = PK ? (vr && dist <= qv.y) : (nn < nq && i < n && vr && dist <= th[nn]);\n"
	"        if (PB && hit && (ANN_BCAP >= NT * 4 || bcnt < ANN_BCAP))\n"
	"        { bD[bcnt * 256 + threadIdx.x] = dist; bJ[bcnt * 256 + threadIdx.x] = (unsigned char) j; bcnt++; hit = false; }\n"
	"        if (PB && hit) { append(nn, dist, i); hit = false; }\n"
	"        if (hit)\n"
	"        {\n"
	"          unsigned at = ANN_STAGE ? atomicAdd(sCount + nn, 1u) : ANN_STAGE;\n"
	"          if (ANN_STAGE && at < ANN_STAGE) { sD[nn * ANN_STAGE + at] = dist; sR[nn * ANN_STAGE + at] = i; }\n"
	"          else\n"
	"          {\n"
	"            at = atomicAdd(counts + nn, 1u);\n"
	"            if (at < cap) { out[(unsigned long long) nn * cap + at] = dist; candR[(unsigned long long) nn * cap + at] = i; }\n"
	"          }\n"
	"        }\n"
	"      }\n"
	"      else if (nn < nq && i < n) out[(unsigned long long) nn * n + i] = vr ? dist : inf;\n"
	"    }\n"
	"    if (TC)\n"
	"      for (unsigned e = 0; hm; e++)\n"
	"      {\n"
	"        int jj = __ffs(hm) - 1; hm &= hm - 1;\n"
	"        float a = bD[((ANN_TUNE & 32) ? jj : e) * 256 + threadIdx.x], rs = (jj & 2) ? rs1 : rs0, nx = (jj & 2) ? nx1 : nx0;\n"
	"        unsigned nn = (jj >> 2) * 8 + 2 * t + (jj & 1); float2 qv = qt[nn];\n"
	"        /* The main loop's rounding exactly: no contraction of rs^2 |x|^2 + |q|^2. */\n"
	"        float ax, tx, dist;\n"
	"        asm(\"mul.rn.f32 %0, %1, %1;\" : \"=f\"(ax) : \"f\"(rs)); asm(\"mul.rn.f32 %0, %0, %1;\" : \"+f\"(ax) : \"f\"(nx));\n"
	"        asm(\"add.rn.f32 %0, %1, %2;\" : \"=f\"(tx) : \"f\"(ax), \"f\"(qv.x));\n"
	"        asm(\"fma.rn.f32 %0, %1, %2, %3;\" : \"=f\"(dist) : \"f\"(-2.0f * rs), \"f\"(a), \"f\"(tx));\n"
	"        dist = fmaxf(dist, 0.0f);\n"
	"        if (dist <= qv.y) append(nn, dist, base + g + (jj & 2) * 4);\n"
	"      }\n"
	"    else if (PB)\n"
	"    {\n"
	"      for (unsigned e = 0; e < bcnt; e++)\n"
	"      {\n"
	"        unsigned jj = bJ[e * 256 + threadIdx.x];\n"
	"        append((jj >> 2) * 8 + 2 * t + (jj & 1), bD[e * 256 + threadIdx.x], base + g + (jj & 2) * 4);\n"
	"      }\n"
	"      bcnt = 0;\n"
	"    }\n"
	"  }\n"
	"  if (FUSED == 1 && (ANN_ABLATE & 1) && sink == 1234.5f) counts[0] = 0;\n"
	"  if (FUSED == 1 && ANN_STAGE)\n"
	"  {\n"
	"    __syncthreads();\n"
	"    for (int q = threadIdx.x; q < NT * 8; q += blockDim.x)\n"
	"    {\n"
	"      unsigned c = sCount[q] < ANN_STAGE ? sCount[q] : ANN_STAGE;\n",
	"      if (c == 0 || q >= nq) continue;\n",
	"      unsigned base = atomicAdd(counts + q, c);\n"
	"      for (unsigned j = 0; j < c && base + j < cap; j++)\n"
	"      { out[(unsigned long long) q * cap + base + j] = sD[q * ANN_STAGE + j]; candR[(unsigned long long) q * cap + base + j] = sR[q * ANN_STAGE + j]; }\n"
	"    }\n"
	"  }\n"
	"}\n",
	"#define MMA(NAME, QUANT, NT) extern \"C\" __global__ void NAME(const unsigned char *data, \\\n"
	"    const unsigned char *scales, const float *rowScales, unsigned rowBytes, unsigned blocks, \\\n"
	"    const unsigned *rows, unsigned n, const float *queries, int nq, int dims, float *out) \\\n"
	"{ score_mma<QUANT, NT, 0>(data, scales, rowScales, rowBytes, blocks, rows, n, queries, nq, dims, out, 0, 0, 0, 0, 0); }\n"
	"/* ANN_FUSE_LAYOUT 2 or 3 bound the fused NVFP4 scan to 3 blocks of 256 a multiprocessor, 4 or 5 to 4. */\n"
	"#if ANN_LAYOUT >= 4\n"
	"#define FUSE_BOUNDS __launch_bounds__(256, 4)\n"
	"#elif ANN_LAYOUT >= 2\n"
	"#define FUSE_BOUNDS __launch_bounds__(256, 3)\n"
	"#else\n"
	"#define FUSE_BOUNDS\n"
	"#endif\n"
	"#if ANN_ROWNORM\n"
	"#define ROWNORM_PARAM , const float2 *rowNorm\n"
	"#define ROWNORM_ARG rowNorm\n"
	"#else\n"
	"#define ROWNORM_PARAM\n"
	"#define ROWNORM_ARG 0\n"
	"#endif\n"
	"#define FUSE(NAME, QUANT, NT, BOUNDS) extern \"C\" __global__ void BOUNDS NAME(const unsigned char *data, \\\n"
	"    const unsigned char *scales, const float *rowScales, unsigned rowBytes, unsigned blocks, \\\n"
	"    const unsigned *rows, unsigned n, const float *queries, int nq, int dims, const float *bounds, \\\n"
	"    float *candD, unsigned *candR, unsigned *counts, unsigned cap ROWNORM_PARAM) \\\n"
	"{ score_mma<QUANT, NT, 1>(data, scales, rowScales, rowBytes, blocks, rows, n, queries, nq, dims, candD, bounds, \\\n"
	"    candR, counts, cap, ROWNORM_ARG); }\n"
	"MMA(score_mma_nvfp4, 1, 2) MMA(score_mma_nvfp4_32, 1, 4) MMA(score_mma_nvfp4_64, 1, 8)\n"
	"MMA(score_mma_fp8, 2, 2) MMA(score_mma_fp8_32, 2, 4) MMA(score_mma_fp8_64, 2, 8)\n"
	"MMA(score_mma_half, 3, 2) MMA(score_mma_half_32, 3, 4) MMA(score_mma_half_64, 3, 8)\n"
	"MMA(score_mma_float, 4, 2) MMA(score_mma_float_32, 4, 4) MMA(score_mma_float_64, 4, 8)\n"
	"FUSE(fuse_mma_nvfp4, 1, 2, FUSE_BOUNDS) FUSE(fuse_mma_nvfp4_32, 1, 4, FUSE_BOUNDS)\n"
	"FUSE(fuse_mma_nvfp4_64, 1, 8, FUSE_BOUNDS)\n"
	"#if ANN_TUNE & 2\n"
	"extern \"C\" __global__ void FUSE_BOUNDS fuse_mma_nvfp4_64_d128(const unsigned char *data,\n"
	"    const unsigned char *scales, const float *rowScales, unsigned rowBytes, unsigned blocks,\n"
	"    const unsigned *rows, unsigned n, const float *queries, int nq, int dims, const float *bounds,\n"
	"    float *candD, unsigned *candR, unsigned *counts, unsigned cap ROWNORM_PARAM)\n"
	"{ score_mma<1, 8, 1, 128>(data, scales, rowScales, rowBytes, blocks, rows, n, queries, nq, dims, candD, bounds,\n"
	"    candR, counts, cap, ROWNORM_ARG); }\n"
	"#endif\n"
	"FUSE(fuse_mma_fp8, 2, 2, ) FUSE(fuse_mma_fp8_32, 2, 4, ) FUSE(fuse_mma_fp8_64, 2, 8, )\n"
	"FUSE(fuse_mma_half, 3, 2, ) FUSE(fuse_mma_half_32, 3, 4, ) FUSE(fuse_mma_half_64, 3, 8, )\n"
	"FUSE(fuse_mma_float, 4, 2, ) FUSE(fuse_mma_float_32, 4, 4, ) FUSE(fuse_mma_float_64, 4, 8, )\n"
	"#if ANN_ROWNORM\n"
	"extern \"C\" __global__ void row_norm_nvfp4(const unsigned char *data, const unsigned char *scales,\n"
	"    const float *rowScales, unsigned rowBytes, unsigned blocks, unsigned n, int dims, float2 *norms)\n"
	"{ score_mma<1, 1, 2>(data, scales, rowScales, rowBytes, blocks, 0, n, 0, 0, dims, (float *) norms, 0, 0, 0, 0, 0); }\n"
	"#endif\n",
	"/*\n"
	" * Batched top-32 over a [queries][rows] score matrix. Block (slice, q)\n"
	" * scans its slice of query q's scores: each warp takes 32 scores a step,\n"
	" * one per lane, and keeps a sorted best-32 list in registers, merging a\n"
	" * step's scores only when one beats both its 32nd best and the block's\n"
	" * bound (so, with long slices, almost never). Warps merge per block into\n"
	" * one partial list per (q, slice); merge32 then folds a query's slices.\n"
	" */\n"
	"__device__ void select_slice(const float *row, const unsigned *pos, unsigned begin, unsigned end, float *partD,\n"
	"    unsigned *partR)\n"
	"{\n"
	"  __shared__ float sd[8 * 32]; __shared__ unsigned sr[8 * 32]; __shared__ unsigned bound;\n"
	"  int lane = threadIdx.x & 31, warp = __shfl_sync(0xffffffffu, (int) (threadIdx.x >> 5), 0);\n"
	"  int warps = blockDim.x >> 5;\n"
	"  const float inf = __int_as_float(0x7f800000);\n"
	"  float bd = inf; unsigned br = 0xffffffffu;\n"
	"  if (threadIdx.x == 0) bound = 0x7f800000u;\n"
	"  __syncthreads();\n"
	"  for (unsigned base = begin + warp * 32; base < end; base += warps * 32)\n"
	"  {\n"
	"    unsigned i = base + lane;\n"
	"    float v = i < end ? row[i] : inf;\n"
	"    float thr = fminf(__shfl_sync(0xffffffffu, bd, 31), __uint_as_float(*(volatile unsigned *) &bound));\n"
	"    if (__ballot_sync(0xffffffffu, v < thr) != 0u)\n"
	"    {\n"
	"      float fd = v < thr ? v : inf; unsigned fr = v < thr ? (pos ? pos[i] : i) : 0xffffffffu;\n"
	"      bitonic_sort(fd, fr, lane);\n"
	"      merge_best(bd, br, fd, fr, lane);\n"
	"      float worst = __shfl_sync(0xffffffffu, bd, 31);\n"
	"      if (lane == 0 && worst < inf) atomicMin(&bound, __float_as_uint(worst));\n"
	"    }\n"
	"  }\n"
	"  sd[warp * 32 + lane] = bd; sr[warp * 32 + lane] = br;\n"
	"  __syncthreads();\n"
	"  if (warp == 0)\n"
	"  {\n"
	"    for (int w = 1; w < warps; w++) merge_best(bd, br, sd[w * 32 + lane], sr[w * 32 + lane], lane);\n"
	"    partD[((unsigned long long) blockIdx.y * gridDim.x + blockIdx.x) * 32 + lane] = bd;\n"
	"    partR[((unsigned long long) blockIdx.y * gridDim.x + blockIdx.x) * 32 + lane] = br;\n"
	"  }\n"
	"}\n"
	"extern \"C\" __global__ void select32(const float *scores, unsigned n, unsigned perBlock, float *partD,\n"
	"    unsigned *partR)\n"
	"{\n"
	"  unsigned begin = blockIdx.x * perBlock;\n"
	"  select_slice(scores + (unsigned long long) blockIdx.y * n, 0, begin, min(n, begin + perBlock), partD, partR);\n"
	"}\n"
	"/* Every (n / s)-th row of a row list: a sample that does not follow the list's (key) order. */\n"
	"extern \"C\" __global__ void stride_rows(const unsigned *rows, unsigned n, unsigned s, unsigned *out)\n"
	"{\n"
	"  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n"
	"  if (i < s) out[i] = rows[(unsigned long long) i * n / s];\n"
	"}\n"
	"/* select32 over each query's candidate list: counts[q] entries, stride n; positions from candR. */\n"
	"extern \"C\" __global__ void select32c(const float *candD, const unsigned *candR, const unsigned *counts,\n"
	"    unsigned n, float *partD, unsigned *partR)\n"
	"{\n"
	"  unsigned count = min(counts[blockIdx.y], n), per = (count + gridDim.x - 1) / gridDim.x;\n"
	"  unsigned begin = blockIdx.x * per;\n"
	"  unsigned long long base = (unsigned long long) blockIdx.y * n;\n"
	"  select_slice(candD + base, candR + base, begin, min(count, begin + per), partD, partR);\n"
	"}\n"
	"/* One warp per query merges its slices' lists into the final 32, ascending. */\n"
	"extern \"C\" __global__ void merge32(const float *partD, const unsigned *partR, unsigned slices, float *outD,\n"
	"    unsigned *outR)\n"
	"{\n"
	"  int lane = threadIdx.x & 31;\n"
	"  unsigned q = blockIdx.x;\n"
	"  float bd = partD[((unsigned long long) q * slices) * 32 + lane]; unsigned br = partR[((unsigned long long) q * slices) * 32 + lane];\n"
	"  for (unsigned b = 1; b < slices; b++)\n"
	"    merge_best(bd, br, partD[((unsigned long long) q * slices + b) * 32 + lane], partR[((unsigned long long) q * slices + b) * 32 + lane], lane);\n"
	"  outD[q * 32 + lane] = bd; outR[q * 32 + lane] = br;\n"
	"}\n",
NULL};

typedef struct BlackwellModule
{
	atomic_int	state;			/* 0 untried/compiling, 1 ready, -1
								 * unavailable */
	CUcontext	context;
	CUmodule	module;
	CUfunction	nvfp4;
	CUfunction	nvfp4Segments;	/* score_nvfp4_seg */
	CUfunction	fp8;
	CUfunction	searchNvfp4;
	CUfunction	searchFp8;
	CUfunction	mma[4][3];		/* [format - 1][tiles]: score matrix of up to
								 * 16, 32 or 64 queries from NVFP4, FP8, FP16
								 * or FP32 rows */
	CUfunction	fuse128;		/* fuse_mma_nvfp4_64 at 128 dimensions
								 * (ANN_FUSE_TUNE bit 2) */
	CUfunction	fuse[4][3];		/* the same, appending only rows under a bound
								 * to candidate lists */
	CUfunction	select32;		/* batched top-32 over a score matrix */
	CUfunction	select32c;		/* the same over candidate lists */
	CUfunction	strideRows;		/* a spread sample of a row list */
	CUfunction	merge32;
	int			sms;			/* streaming multiprocessors */
	CUfunction	rowNorm;		/* row_norm_nvfp4 (ANN_FUSE_ROWNORM) */
} BlackwellModule;

/*
 * Per caller thread, device and slot (one per shard a thread searches on
 * that device): block partial lists for up to ANN_BLACKWELL_GROUP_MAX
 * queries, a finish counter, the final lists, and pinned host copies.
 */
typedef struct SearchSlot
{
	unsigned	maxBlocks;
	float	   *scores;			/* [queries][rows] for grouped searches */
	size_t		scoreCapacity;
	float	   *partD;
	unsigned   *partR;
	unsigned   *counter;
	unsigned   *counts;			/* candidates per query (fused scans) */
	unsigned   *sampleRows;		/* ANN_BLACKWELL_SAMPLE rows of a list */
	float	   *outD;
	unsigned   *outR;
	float	   *hostD;
	unsigned   *hostR;
	/* Large-k candidates (AnnBlackwellCandidates): sampled rows, their scores, bounded lists. */
	unsigned   *largeSample;
	size_t		largeSampleCapacity;
	float	   *largeSampleScores;
	size_t		largeScoreCapacity;
	float	   *candD;
	unsigned   *candR;
	size_t		candCapacity;
	float	   *hostCandD;		/* pinned */
	unsigned   *hostCandR;		/* pinned */
	size_t		hostCandCapacity;
	unsigned   *hostCounts;		/* pinned, ANN_BLACKWELL_GROUP_MAX */
} SearchSlot;

static _Thread_local SearchSlot *slots[ANN_BLACKWELL_MAX_DEVICES][ANN_INDEX_MAX_SHARDS];

static BlackwellModule modules[ANN_BLACKWELL_MAX_DEVICES];
static pthread_mutex_t moduleMutex = PTHREAD_MUTEX_INITIALIZER;

/*
 * Each query's FP16 row in the scan's shared memory is padded by 8 halves
 * (ANN_FUSE_QPAD=0 removes the pad, for measurements). Unpadded, a 128-
 * dimension row is 256 bytes, so the eight lanes that load one B fragment
 * all read one bank: the fused scan of 64 queries over 1M rows took 358.5
 * us, 209.7 us with the pad.
 */
static unsigned
QueryPad(void)
{
	static int	pad = -1;

	if (pad < 0)
	{
		const char *value = getenv("ANN_FUSE_QPAD");

		pad = value && atoi(value) == 0 ? 0 : 8;
	}
	return (unsigned) pad;
}

/*
 * ANN_FUSE_TUNE (default 42; 0 restores the generic scan): bits of
 * fused-scan variants, see ANN_TUNE in the kernel source. 42 is bits 2, 8
 * and 32: a kernel compiled for 128 dimensions, one-prmt scale broadcast and
 * hits stored at their output's list slot. Bit 16 (58) adds a looser hit
 * test rechecked at flush, about 4% faster, but its rounding moves a few
 * candidates at the cutoff (recall@500 0.9998, lowest 0.994, against 1.0 and
 * 0.998), so it is off by default.
 */
static int
FuseTune(void)
{
	static int	tune = -1;

	if (tune < 0)
	{
		const char *value = getenv("ANN_FUSE_TUNE");

		tune = value ? atoi(value) & 0xffff : 42;
	}
	return tune;
}

/*
 * ANN_FUSE_LAYOUT (default 3; 0 turns it off): bit 1 has the fused NVFP4
 * scan read a second, lane-contiguous copy of each shard's rows, built at
 * upload: rows padded to whole 64-byte chunks of 8 blocks, and in each chunk
 * the 16 bytes lane t of a quad decodes (byte t of each block's two words)
 * stored together, so a lane loads one 16-byte word per row where it loaded
 * eight 8-byte ones. That takes the kernel from 126 to 96 registers, and
 * value 3 adds __launch_bounds__(256, 3): 80 registers without spilling, so
 * 3 blocks fit a multiprocessor (4 or 5 bound it to 4, which spills). It
 * costs a second copy of the rows (64 bytes a row at 128 dimensions).
 */

static int
FuseLayout(void)
{
	static int	layout = -1;

	if (layout < 0)
	{
		const char *value = getenv("ANN_FUSE_LAYOUT");

		layout = value == NULL ? 3 : atoi(value) > 0 && atoi(value) <= 5 ? atoi(value) : 0;
	}
	return layout;
}

#define FUSE_PACKED_MAX 256
static struct
{
	const uint8_t *data;
	uint8_t    *packed;
}			packedCopies[FUSE_PACKED_MAX];
static pthread_mutex_t packedMutex = PTHREAD_MUTEX_INITIALIZER;

static uint8_t *
PackedCopy(const uint8_t *data)
{
	uint8_t    *packed = NULL;

	pthread_mutex_lock(&packedMutex);
	for (int i = 0; i < FUSE_PACKED_MAX; i++)
		if (packedCopies[i].data == data)
		{
			packed = packedCopies[i].packed;
			break;
		}
	pthread_mutex_unlock(&packedMutex);
	return packed;
}

/* Halves a query takes in the scan's shared memory. */
static unsigned
QueryStride(int dims)
{
	return (((unsigned) dims + 15) & ~15u) + QueryPad();
}

/*
 * ANN_FUSE_ROWNORM (default 1; 0 or off turns it off): 1 (or on) gives every NVFP4 shard a (row scale, |x|^2)
 * pair per row, made once at upload by the scan's own code (from the
 * original row layout, also under ANN_FUSE_LAYOUT), and the fused scan
 * loads it instead of summing |x|^2; 2 (or check) keeps summing and prints
 * rows whose stored pair differs.
 */
static int
RowNormMode(void)
{
	static int	mode = -1;

	if (mode < 0)
	{
		const char *value = getenv("ANN_FUSE_ROWNORM");

		mode = value == NULL ? 1 : strcmp(value, "check") == 0 ? 2 :
			strcmp(value, "on") == 0 ? 1 : atoi(value) >= 1 && atoi(value) <= 3 ? atoi(value) : 0;
	}
	return mode;
}

typedef struct RowNormEntry
{
	const void *data;			/* the shard's device rows */
	float	   *norms;			/* [rows] (scale, |x|^2) */
} RowNormEntry;

static RowNormEntry rowNorms[64];
static pthread_mutex_t rowNormMutex = PTHREAD_MUTEX_INITIALIZER;

static float *
ShardRowNorms(const AnnBlackwellShard *shard)
{
	float	   *norms = NULL;

	if (RowNormMode() == 0 || shard->data == NULL)
		return NULL;
	(void) pthread_mutex_lock(&rowNormMutex);
	for (int i = 0; i < 64; i++)
		if (rowNorms[i].data == shard->data)
			norms = rowNorms[i].norms;
	(void) pthread_mutex_unlock(&rowNormMutex);
	return norms;
}

/*
 * The fused scan's rows, row stride and row norms for a shard: the
 * lane-contiguous copy and the (scale, |x|^2) pairs for NVFP4 when those
 * modes are on. Returns 0 when a copy the compiled kernel reads is missing.
 */
static int
FuseInputs(const AnnBlackwellShard *shard, uint8_t **data, unsigned *rowBytes, float **norms)
{
	int			nvfp4 = shard->quant == ANN_QUANT_NVFP4;

	*data = nvfp4 && (FuseLayout() & 1) ? PackedCopy(shard->data) : shard->data;
	*rowBytes = nvfp4 && (FuseLayout() & 1) ? (shard->blocks + 7) / 8 * 64 : shard->rowBytes;
	*norms = nvfp4 ? ShardRowNorms(shard) : NULL;
	return *data != NULL && !(nvfp4 && (RowNormMode() & 1) && *norms == NULL);
}

static char *
JoinSource(void)
{
	size_t		length = 1024;
	char	   *source;

	for (int i = 0; BlackwellSource[i]; i++)
		length += strlen(BlackwellSource[i]);
	if ((source = malloc(length)) == NULL)
		return NULL;
	{
		const char *ablate = getenv("ANN_FUSE_ABLATE");
		const char *append = getenv("ANN_FUSE_APPEND");

		/*
		 * ANN_FUSE_APPEND=global puts every fused hit straight into its
		 * global list (about 250,000 atomics a 64-query batch on 64
		 * counters); the default stages them per block in shared memory
		 * first: 210.4 -> 185.2 us a batch.
		 */
		snprintf(source, length, "#define ANN_QPAD %u\n#define ANN_ABLATE %d\n#define ANN_STAGE %d\n#define ANN_LAYOUT %d\n"
				 "#define ANN_ROWNORM %d\n#define ANN_APPEND_MODE %d\n#define ANN_PREFETCH %d\n#define ANN_BCAP %d\n"
				 "#define ANN_QVEC %d\n",
				 QueryPad(), ablate ? atoi(ablate) : 0, append && strcmp(append, "global") == 0 ? 0 :
				 getenv("ANN_FUSE_STAGE") ? atoi(getenv("ANN_FUSE_STAGE")) : 32, FuseLayout(),
				 RowNormMode(), getenv("ANN_FUSE_APPEND_MODE") ? atoi(getenv("ANN_FUSE_APPEND_MODE")) : 5,
				 getenv("ANN_FUSE_PREFETCH") ? atoi(getenv("ANN_FUSE_PREFETCH")) : 0,
				 getenv("ANN_FUSE_BCAP") && atoi(getenv("ANN_FUSE_BCAP")) > 0 ? atoi(getenv("ANN_FUSE_BCAP")) : 32,
				 getenv("ANN_FUSE_QVEC") ? atoi(getenv("ANN_FUSE_QVEC")) : 3);
		snprintf(source + strlen(source), length - strlen(source), "#define ANN_TUNE %d\n", FuseTune());
	}
	for (int i = 0; BlackwellSource[i]; i++)
		strcat(source, BlackwellSource[i]);
	return source;
}

/*
 * The tensor-core scan in three widths per format: 2, 4 or 8 tiles of 8
 * queries. The wider ones keep their queries in up to 128 KB of shared
 * memory, past the 48 KB a launch gets without opting in.
 */
static int
LoadScoreFunctions(BlackwellModule *module, size_t optin)
{
	static const char *const formats[] = {"nvfp4", "fp8", "half", "float"};
	static const char *const widths[] = {"", "_32", "_64"};

	for (int f = 0; f < 4; f++)
		for (int w = 0; w < 3; w++)
		{
			char		name[64];
			CUfunction *functions[] = {&module->mma[f][w], &module->fuse[f][w]};

			for (int kind = 0; kind < 2; kind++)
			{
				int			fixed = 0;
				size_t		dynamic = ANN_BLACKWELL_SHARED_MAX;

				snprintf(name, sizeof(name), "%s_%s%s", kind ? "fuse_mma" : "score_mma", formats[f], widths[w]);
				if (cuModuleGetFunction(functions[kind], module->module, name) != CUDA_SUCCESS)
					return 0;
				if (w == 0)
					continue;
				/* Static and dynamic shared memory share the device's opt-in limit. */
				if (cuFuncGetAttribute(&fixed, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, *functions[kind]) != CUDA_SUCCESS)
					return 0;
				if (optin > 0 && optin - (size_t) fixed < dynamic)
					dynamic = optin - (size_t) fixed;
				if (cuFuncSetAttribute(*functions[kind], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
									   (int) dynamic) != CUDA_SUCCESS)
					return 0;
				if (dynamic < sharedMax)
					sharedMax = dynamic;
			}
		}
	return 1;
}

int
AnnBlackwellReady(int device)
{
	BlackwellModule *module;
	CUdevice	handle;
	int			major = 0,
				minor = 0,
				optin = 0;
	char		arch[16];
	const char *failed = NULL;
	CUresult	result = CUDA_SUCCESS;
	char	   *image = NULL;
	char	   *source = NULL;
	size_t		size = 0;

	if (device < 0 || device >= ANN_BLACKWELL_MAX_DEVICES)
		return 0;
	module = &modules[device];
	if (atomic_load_explicit(&module->state, memory_order_acquire) != 0)
		return atomic_load_explicit(&module->state, memory_order_acquire) > 0;
	(void) pthread_mutex_lock(&moduleMutex);
	if (atomic_load_explicit(&module->state, memory_order_relaxed) != 0)
		goto unlock;
	if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
		cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess ||
		(major != 10 && major != 12))
	{
		fprintf(stderr, "Blackwell module: device %d has compute capability %d.x, not 10.x or 12.x; "
				"quantized scans are off there\n", device, major);
		goto unlock;
	}
	snprintf(arch, sizeof(arch), "sm_%d%da", major, minor);
	if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) != cudaSuccess)
		optin = 0;
	if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&handle, device) != CUDA_SUCCESS ||
		cuDevicePrimaryCtxRetain(&module->context, handle) != CUDA_SUCCESS ||
		cuCtxSetCurrent(module->context) != CUDA_SUCCESS)
		failed = "context";
	else if ((source = JoinSource()) == NULL ||
			 (image = AnnNvrtcCompile(source, "Blackwell module", arch, &size)) == NULL)
		failed = "NVRTC compile";
	else if ((result = cuModuleLoadData(&module->module, image)) != CUDA_SUCCESS)
		failed = "module load";
	else if (cuModuleGetFunction(&module->nvfp4, module->module, "score_nvfp4") != CUDA_SUCCESS ||
			 cuModuleGetFunction(&module->nvfp4Segments, module->module, "score_nvfp4_seg") != CUDA_SUCCESS ||
			 cuModuleGetFunction(&module->fp8, module->module, "score_fp8") != CUDA_SUCCESS ||
			 cuModuleGetFunction(&module->searchNvfp4, module->module, "search_nvfp4") != CUDA_SUCCESS ||
			 cuModuleGetFunction(&module->searchFp8, module->module, "search_fp8") != CUDA_SUCCESS)
		failed = "scan kernels";
	else if (!LoadScoreFunctions(module, (size_t) optin))
		failed = "grouped kernels";
	else if (cuModuleGetFunction(&module->select32, module->module, "select32") != CUDA_SUCCESS ||
			 cuModuleGetFunction(&module->select32c, module->module, "select32c") != CUDA_SUCCESS ||
			 cuModuleGetFunction(&module->strideRows, module->module, "stride_rows") != CUDA_SUCCESS ||
			 cuModuleGetFunction(&module->merge32, module->module, "merge32") != CUDA_SUCCESS ||
			 cudaDeviceGetAttribute(&module->sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess)
		failed = "selection kernels";
	else if (RowNormMode() && cuModuleGetFunction(&module->rowNorm, module->module, "row_norm_nvfp4") != CUDA_SUCCESS)
		failed = "row norm kernel";
	else if (FuseTune() & 2)
	{
		int			fixed = 0;

		if (cuModuleGetFunction(&module->fuse128, module->module, "fuse_mma_nvfp4_64_d128") != CUDA_SUCCESS ||
			cuFuncGetAttribute(&fixed, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, module->fuse128) != CUDA_SUCCESS ||
			cuFuncSetAttribute(module->fuse128, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
							   (int) (optin > fixed ? optin - fixed : 0)) != CUDA_SUCCESS)
			failed = "128-dimension fused kernel";
	}
	if (failed)
	{
		fprintf(stderr, "Blackwell module: unavailable on device %d (%s failed for %s, CUDA error %d, "
				"%zu bytes of shared memory)\n", device, failed, arch, (int) result, sharedMax);
		goto unlock;
	}
	atomic_store_explicit(&module->state, 1, memory_order_release);
	{
		/* Register arrays indexed by an unrolled loop must not spill. */
		const char *names[] = {"search_nvfp4", "score_mma_nvfp4", "score_mma_fp8", "score_mma_nvfp4_64",
		"fuse_mma_nvfp4_64", "fuse_mma_nvfp4_64_d128"};
		CUfunction	functions[] = {module->searchNvfp4, module->mma[0][0], module->mma[1][0], module->mma[0][2],
		module->fuse[0][2], module->fuse128};

		for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++)
		{
			if (functions[i] == NULL)
				continue;
			int			registers = 0,
						local = 0;

			if (cuFuncGetAttribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, functions[i]) == CUDA_SUCCESS &&
				cuFuncGetAttribute(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, functions[i]) == CUDA_SUCCESS &&
				(local > 0 || getenv("ANN_BLACKWELL_VERBOSE")))
				fprintf(stderr, "Blackwell module: %s uses %d registers and %d bytes of local memory%s\n",
						names[i], registers, local, local > 0 ? " (spilling)" : "");
		}
	}
unlock:
	if (atomic_load_explicit(&module->state, memory_order_relaxed) == 0)
		atomic_store_explicit(&module->state, -1, memory_order_release);
	free(source);
	free(image);
	(void) pthread_mutex_unlock(&moduleMutex);
	return atomic_load_explicit(&module->state, memory_order_acquire) > 0;
}

void
AnnBlackwellFree(AnnBlackwellShard *shard)
{
	if (shard->data && RowNormMode())
	{
		(void) pthread_mutex_lock(&rowNormMutex);
		for (int i = 0; i < 64; i++)
			if (rowNorms[i].data == shard->data)
			{
				(void) cudaFree(rowNorms[i].norms);
				rowNorms[i].data = NULL;
				rowNorms[i].norms = NULL;
			}
		(void) pthread_mutex_unlock(&rowNormMutex);
	}
	if (shard->owned && shard->data)
	{
		pthread_mutex_lock(&packedMutex);
		for (int i = 0; i < FUSE_PACKED_MAX; i++)
			if (packedCopies[i].data == shard->data)
			{
				(void) cudaFree(packedCopies[i].packed);
				packedCopies[i].data = NULL;
				packedCopies[i].packed = NULL;
			}
		pthread_mutex_unlock(&packedMutex);
	}
	if (shard->owned)
	{
		(void) cudaFree(shard->data);
		(void) cudaFree(shard->scales);
		(void) cudaFree(shard->rowScales);
	}
	memset(shard, 0, sizeof(*shard));
}

int
AnnBlackwellView(const void *dataset, int64_t stride, int half, uint32_t dims, int device, AnnBlackwellShard *out)
{
	memset(out, 0, sizeof(*out));
	if (dataset == NULL || stride < (int64_t) dims || !AnnBlackwellReady(device))
		return 0;
	out->quant = half ? ANN_BLACKWELL_HALF : ANN_BLACKWELL_FLOAT;
	out->data = (uint8_t *) (uintptr_t) dataset;
	out->rowBytes = (uint32_t) (stride * (half ? 2 : 4));
	out->blocks = (dims + 15) / 16;
	return 1;
}

/* E4M3 (OCP FN: no infinities, 448 the largest) nearest to a value. */
static uint8_t
E4m3(float value)
{
	uint8_t		sign = value < 0 ? 0x80 : 0;
	float		magnitude = fabsf(value);
	int			exponent;
	float		fraction;

	if (!(magnitude > 0))
		return sign;
	if (magnitude >= 448.0f)
		return sign | 0x7e;
	fraction = frexpf(magnitude, &exponent);	/* magnitude = fraction * 2^exponent, fraction in [0.5, 1) */
	if (exponent + 6 >= 1)
	{
		int			biased = exponent + 6;
		int			mantissa = (int) nearbyintf((2.0f * fraction - 1.0f) * 8.0f);

		if (mantissa == 8)
		{
			mantissa = 0;
			biased++;
		}
		if (biased > 15 || (biased == 15 && mantissa == 7))
			return sign | 0x7e;
		return sign | (uint8_t) (biased << 3 | mantissa);
	}
	{
		/* Subnormals are multiples of 2^-9; eight of them make the smallest normal. */
		int			units = (int) nearbyintf(magnitude * 512.0f);

		return sign | (uint8_t) (units >= 8 ? 0x08 : units);
	}
}

int
AnnBlackwellUploadFp8(const float *vectors, uint64_t rows, uint32_t dims, int device, AnnBlackwellShard *out)
{
	uint32_t	rowBytes = (dims + 15) / 16 * 16;
	const uint64_t chunk = 65536;
	uint8_t    *codes = NULL;
	float	   *scales = NULL;
	int			ok;

	memset(out, 0, sizeof(*out));
	if (rows == 0 || !AnnBlackwellReady(device) || cudaSetDevice(device) != cudaSuccess)
		return 0;
	codes = malloc(chunk * rowBytes);
	scales = malloc(chunk * sizeof(float));
	ok = codes != NULL && scales != NULL &&
		cudaMalloc((void **) &out->data, rows * rowBytes) == cudaSuccess &&
		cudaMalloc((void **) &out->rowScales, rows * sizeof(float)) == cudaSuccess;
	for (uint64_t first = 0; ok && first < rows; first += chunk)
	{
		uint64_t	count = rows - first < chunk ? rows - first : chunk;

		/* As build_cuvs_index.py --quantize fp8 writes them: a row scale of max |x| / 448. */
		for (uint64_t r = 0; r < count; r++)
		{
			const float *x = vectors + (first + r) * dims;
			float		amax = 0;
			float		scale;

			for (uint32_t d = 0; d < dims; d++)
				amax = fabsf(x[d]) > amax ? fabsf(x[d]) : amax;
			scale = amax > 0 ? amax / 448.0f : 1.0f;
			scales[r] = scale;
			for (uint32_t d = 0; d < rowBytes; d++)
				codes[r * rowBytes + d] = d < dims ? E4m3(x[d] / scale) : 0;
		}
		ok = cudaMemcpy(out->data + first * rowBytes, codes, count * rowBytes, cudaMemcpyHostToDevice) == cudaSuccess &&
			cudaMemcpy(out->rowScales + first, scales, count * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
	}
	free(codes);
	free(scales);
	if (!ok)
	{
		fprintf(stderr, "Blackwell module: cannot make the FP8 re-scoring copy on device %d\n", device);
		out->owned = 1;			/* free what was allocated */
		AnnBlackwellFree(out);
		return 0;
	}
	out->quant = ANN_QUANT_FP8;
	out->owned = 1;
	out->rowBytes = rowBytes;
	out->blocks = rowBytes / 16;
	out->bytes = rows * (rowBytes + sizeof(float));
	return 1;
}

int
AnnBlackwellUpload(const AnnIndexFile *file, uint32_t s, int device, AnnBlackwellShard *out)
{
	uint64_t	first = file->shards[s].first;
	size_t		rows = file->shards[s].rows;
	size_t		dataBytes = rows * file->quantRowBytes;
	size_t		scaleBytes = file->quant == ANN_QUANT_NVFP4 ? rows * file->quantBlocks : 0;

	memset(out, 0, sizeof(*out));
	if (file->quant == ANN_QUANT_NONE || !AnnBlackwellReady(device) || cudaSetDevice(device) != cudaSuccess)
		return 0;
	if (cudaMalloc((void **) &out->data, dataBytes) != cudaSuccess ||
		cudaMemcpy(out->data, file->quantData + first * file->quantRowBytes, dataBytes,
				   cudaMemcpyHostToDevice) != cudaSuccess ||
		(scaleBytes &&
		 (cudaMalloc((void **) &out->scales, scaleBytes) != cudaSuccess ||
		  cudaMemcpy(out->scales, file->quantScales + first * file->quantBlocks, scaleBytes,
					 cudaMemcpyHostToDevice) != cudaSuccess)) ||
		cudaMalloc((void **) &out->rowScales, rows * sizeof(float)) != cudaSuccess ||
		cudaMemcpy(out->rowScales, file->rowScales + first, rows * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		fprintf(stderr, "Blackwell module: cannot copy shard %u's %s rows to device %d\n", s,
				AnnIndexQuantName(file->quant), device);
		AnnBlackwellFree(out);
		return 0;
	}
	out->quant = file->quant;
	out->owned = 1;
	out->rowBytes = file->quantRowBytes;
	out->blocks = file->quantBlocks;
	out->bytes = dataBytes + scaleBytes + rows * sizeof(float);
	if (RowNormMode() && out->quant == ANN_QUANT_NVFP4)
	{
		BlackwellModule *module = &modules[device];
		float	   *norms = NULL;
		unsigned	n = (unsigned) rows,
					rowBytes = out->rowBytes,
					blocks = out->blocks;
		int			dims = (int) file->dimensions;
		void	   *args[] = {(void *) &out->data, (void *) &out->scales, (void *) &out->rowScales, &rowBytes,
		&blocks, &n, &dims, &norms};
		int			slot = -1;

		(void) pthread_mutex_lock(&rowNormMutex);
		for (int i = 0; i < 64 && slot < 0; i++)
			if (rowNorms[i].data == NULL)
				slot = i;
		if (slot < 0 || cudaMalloc((void **) &norms, rows * 2 * sizeof(float)) != cudaSuccess ||
			cuCtxSetCurrent(module->context) != CUDA_SUCCESS ||
			cuLaunchKernel(module->rowNorm, (n + 127) / 128, 1, 1, ANN_BLACKWELL_BLOCK, 1, 1,
						   8 * QueryStride(dims) * sizeof(uint16_t), NULL, args, NULL) != CUDA_SUCCESS ||
			cuCtxSynchronize() != CUDA_SUCCESS)
		{
			fprintf(stderr, "Blackwell module: cannot make shard %u's row norms on device %d\n", s, device);
			if (norms)
				(void) cudaFree(norms);
		}
		else
		{
			rowNorms[slot].data = out->data;
			rowNorms[slot].norms = norms;
			if (getenv("ANN_BLACKWELL_VERBOSE"))
				fprintf(stderr, "Blackwell module: shard %u row norms (%zu rows) on device %d\n", s, rows, device);
			out->bytes += rows * 2 * sizeof(float);
		}
		(void) pthread_mutex_unlock(&rowNormMutex);
	}
	if ((FuseLayout() & 1) && file->quant == ANN_QUANT_NVFP4)
	{
		size_t		packedRow = (size_t) (file->quantBlocks + 7) / 8 * 64;
		size_t		chunkRows = 1 << 20;
		uint8_t    *host = malloc(chunkRows * packedRow);
		uint8_t    *packed = NULL;
		int			ok = host != NULL && cudaMalloc((void **) &packed, rows * packedRow) == cudaSuccess;

		for (size_t r0 = 0; ok && r0 < rows; r0 += chunkRows)
		{
			size_t		count = rows - r0 < chunkRows ? rows - r0 : chunkRows;

			memset(host, 0, count * packedRow);
			for (size_t r = 0; r < count; r++)
			{
				const uint8_t *in = file->quantData + (first + r0 + r) * file->quantRowBytes;
				uint8_t    *o = host + r * packedRow;

				for (uint32_t b = 0; b < file->quantBlocks; b++)
					for (int t = 0; t < 4; t++)
					{
						size_t		at = (size_t) (b / 8) * 64 + 16 * t + 2 * (b % 8);

						o[at] = in[8 * b + t];
						o[at + 1] = in[8 * b + 4 + t];
					}
			}
			ok = cudaMemcpy(packed + r0 * packedRow, host, count * packedRow, cudaMemcpyHostToDevice) == cudaSuccess;
		}
		free(host);
		if (ok)
		{
			ok = 0;
			pthread_mutex_lock(&packedMutex);
			for (int i = 0; i < FUSE_PACKED_MAX && !ok; i++)
				if (packedCopies[i].data == NULL)
				{
					packedCopies[i].data = out->data;
					packedCopies[i].packed = packed;
					ok = 1;
				}
			pthread_mutex_unlock(&packedMutex);
		}
		if (!ok)
		{
			fprintf(stderr, "Blackwell module: cannot build shard %u's lane-contiguous copy on device %d\n", s, device);
			(void) cudaFree(packed);
			AnnBlackwellFree(out);
			return 0;
		}
		out->bytes += rows * packedRow;
	}
	return 1;
}

int
AnnBlackwellScore(const AnnBlackwellShard *shard, int device, const uint32_t *rows, unsigned n,
				  const float *query, int dims, float *out, void *stream)
{
	BlackwellModule *module = &modules[device];
	unsigned	grid = (unsigned) (((uint64_t) n * ANN_BLACKWELL_GROUP + ANN_BLACKWELL_BLOCK - 1) / ANN_BLACKWELL_BLOCK);
	unsigned	rowBytes = shard->rowBytes;
	unsigned	blocks = shard->blocks;

	if (n == 0)
		return 1;
	if (shard->quant == ANN_QUANT_NONE || shard->quant >= ANN_BLACKWELL_HALF || atomic_load_explicit(&module->state, memory_order_acquire) <= 0 ||
		cuCtxSetCurrent(module->context) != CUDA_SUCCESS)
		return 0;
	if (shard->quant == ANN_QUANT_NVFP4)
	{
		void	   *args[] = {(void *) &shard->data, (void *) &shard->scales, (void *) &shard->rowScales,
		&rowBytes, &blocks, (void *) &rows, &n, (void *) &query, &dims, &out};

		return cuLaunchKernel(module->nvfp4, grid, 1, 1, ANN_BLACKWELL_BLOCK, 1, 1, 0,
							  (CUstream) stream, args, NULL) == CUDA_SUCCESS;
	}
	{
		void	   *args[] = {(void *) &shard->data, (void *) &shard->rowScales, &rowBytes,
		(void *) &rows, &n, (void *) &query, &dims, &out};

		return cuLaunchKernel(module->fp8, grid, 1, 1, ANN_BLACKWELL_BLOCK, 1, 1, 0,
							  (CUstream) stream, args, NULL) == CUDA_SUCCESS;
	}
}

int
AnnBlackwellScoreSegments(const AnnBlackwellShard *shard, int device, const uint32_t *rows, unsigned n,
						  const uint32_t *offsets, int nq, const float *queries, int dims, float *out, void *stream)
{
	BlackwellModule *module = &modules[device];
	unsigned	grid = (unsigned) (((uint64_t) n * ANN_BLACKWELL_GROUP + ANN_BLACKWELL_BLOCK - 1) / ANN_BLACKWELL_BLOCK);
	unsigned	rowBytes = shard->rowBytes;
	unsigned	blocks = shard->blocks;
	void	   *args[] = {(void *) &shard->data, (void *) &shard->scales, (void *) &shard->rowScales,
	&rowBytes, &blocks, (void *) &rows, &n, (void *) &offsets, &nq, (void *) &queries, &dims, &out};

	if (n == 0)
		return 1;
	if (shard->quant != ANN_QUANT_NVFP4 || atomic_load_explicit(&module->state, memory_order_acquire) <= 0 ||
		cuCtxSetCurrent(module->context) != CUDA_SUCCESS)
		return 0;
	return cuLaunchKernel(module->nvfp4Segments, grid, 1, 1, ANN_BLACKWELL_BLOCK, 1, 1, 0,
						  (CUstream) stream, args, NULL) == CUDA_SUCCESS;
}

/* A thread's slot on the current device, sized for maxBlocks; allocated on first use. */
static SearchSlot *
GetSlot(int device, unsigned slot, unsigned maxBlocks)
{
	SearchSlot *sc = slots[device][slot];
	size_t		lists = (size_t) maxBlocks * ANN_BLACKWELL_GROUP_MAX * ANN_BLACKWELL_TOP;
	unsigned	zero[2] = {0, 0x7f800000u};

	if (sc && sc->maxBlocks >= maxBlocks)
		return sc;
	if (sc)
	{
		(void) cudaFree(sc->scores);
		(void) cudaFree(sc->partD);
		(void) cudaFree(sc->partR);
		(void) cudaFree(sc->counter);
		(void) cudaFree(sc->counts);
		(void) cudaFree(sc->sampleRows);
		(void) cudaFree(sc->outD);
		(void) cudaFree(sc->outR);
		(void) cudaFreeHost(sc->hostD);
		(void) cudaFreeHost(sc->hostR);
		(void) cudaFree(sc->largeSample);
		(void) cudaFree(sc->largeSampleScores);
		(void) cudaFree(sc->candD);
		(void) cudaFree(sc->candR);
		(void) cudaFreeHost(sc->hostCandD);
		(void) cudaFreeHost(sc->hostCandR);
		(void) cudaFreeHost(sc->hostCounts);
		memset(sc, 0, sizeof(*sc));
	}
	else if ((sc = slots[device][slot] = calloc(1, sizeof(*sc))) == NULL)
		return NULL;
	if (cudaMalloc((void **) &sc->partD, lists * sizeof(float)) != cudaSuccess ||
		cudaMalloc((void **) &sc->partR, lists * sizeof(unsigned)) != cudaSuccess ||
		cudaMalloc((void **) &sc->counter, sizeof(zero)) != cudaSuccess ||
		cudaMemcpy(sc->counter, zero, sizeof(zero), cudaMemcpyHostToDevice) != cudaSuccess ||
		cudaMalloc((void **) &sc->counts, ANN_BLACKWELL_GROUP_MAX * sizeof(unsigned)) != cudaSuccess ||
		cudaMalloc((void **) &sc->sampleRows, ANN_BLACKWELL_SAMPLE * sizeof(unsigned)) != cudaSuccess ||
		cudaMalloc((void **) &sc->outD, ANN_BLACKWELL_GROUP_MAX * ANN_BLACKWELL_TOP * sizeof(float)) != cudaSuccess ||
		cudaMalloc((void **) &sc->outR, ANN_BLACKWELL_GROUP_MAX * ANN_BLACKWELL_TOP * sizeof(unsigned)) != cudaSuccess ||
		cudaMallocHost((void **) &sc->hostD, ANN_BLACKWELL_GROUP_MAX * ANN_BLACKWELL_TOP * sizeof(float)) != cudaSuccess ||
		cudaMallocHost((void **) &sc->hostR, ANN_BLACKWELL_GROUP_MAX * ANN_BLACKWELL_TOP * sizeof(unsigned)) != cudaSuccess)
	{
		fprintf(stderr, "Blackwell module: cannot allocate search scratch on device %d\n", device);
		return NULL;
	}
	sc->maxBlocks = maxBlocks;
	return sc;
}

/* Fused scans unless ANN_BLACKWELL_FUSE=off (for measurements). */
static int
FuseEnabled(void)
{
	static int	enabled = -1;

	if (enabled < 0)
	{
		const char *value = getenv("ANN_BLACKWELL_FUSE");

		enabled = value == NULL || strcmp(value, "off") != 0;
	}
	return enabled;
}

/*
 * The most queries one grouped scan takes: 64, 32 or 16 as their FP16 copies
 * fit in shared memory (ANN_BLACKWELL_GROUP caps it, for measurements), 1
 * when not even 16 do.
 */
int
AnnBlackwellGroupLimit(int dims)
{
	static int	cap = 0;
	size_t		qstride = QueryStride(dims);

	if (cap == 0)
	{
		const char *value = getenv("ANN_BLACKWELL_GROUP");

		cap = value ? atoi(value) : ANN_BLACKWELL_GROUP_MAX;
		if (cap < 16 || cap > ANN_BLACKWELL_GROUP_MAX)
			cap = ANN_BLACKWELL_GROUP_MAX;
	}
	for (int queries = ANN_BLACKWELL_GROUP_MAX; queries > 16; queries /= 2)
		if (queries <= cap && queries * qstride * 2 <= sharedMax)
			return queries;
	return 16 * qstride * 2 <= 45 * 1024 ? 16 : 1;
}

int
AnnBlackwellScoreMatrix(const AnnBlackwellShard *shard, int device, const uint32_t *rows,
						unsigned n, const float *queries, unsigned nq, int dims,
						float *out, void *stream)
{
	BlackwellModule *module = &modules[device];
	unsigned	rowBytes = shard->rowBytes;
	unsigned	blocks = shard->blocks;
	int			count = (int) nq;
	int			width = nq <= 16 ? 0 : nq <= 32 ? 1 : 2;
	unsigned	shared = (16u << width) * QueryStride(dims) * sizeof(uint16_t);
	unsigned	grid = (n + 127) / 128;
	void	   *args[] = {(void *) &shard->data, (void *) &shard->scales, (void *) &shard->rowScales,
	&rowBytes, &blocks, (void *) &rows, &n, (void *) &queries, &count, &dims, &out};

	if (n == 0 || nq == 0 || nq > (unsigned) AnnBlackwellGroupLimit(dims) ||
		shard->quant == ANN_QUANT_NONE || atomic_load_explicit(&module->state, memory_order_acquire) <= 0 ||
		cuCtxSetCurrent(module->context) != CUDA_SUCCESS)
		return 0;
	if (grid > (unsigned) module->sms * 4)
		grid = (unsigned) module->sms * 4;
	if (shard->quant == ANN_QUANT_FP8)
		blocks = rowBytes / 16;
	else if (shard->quant >= ANN_BLACKWELL_HALF)
		blocks = ((unsigned) dims + 15) / 16;
	return cuLaunchKernel(module->mma[shard->quant - 1][width], grid, 1, 1,
						  ANN_BLACKWELL_BLOCK, 1, 1, shared, (CUstream) stream, args, NULL) == CUDA_SUCCESS;
}

/*
 * Blocks per SM of the fused scan's grid: ANN_FUSE_GRID, default 3 for the
 * bounded lane-contiguous NVFP4 scan (ANN_FUSE_LAYOUT 2 or 3) and 2 otherwise. With
 * hits staged per block, fewer, longer blocks flush less: 64 queries over
 * 1M rows took 175.4 us at 2, 185.2 at 4, 243.4 at 16, 281.4 at 1, and two
 * loaded GPUs served 5% more at 2 than at 4.
 */
static unsigned
FuseGridPerSm(int quant)
{
	static int	blocks = -1;

	if (blocks < 0)
	{
		const char *value = getenv("ANN_FUSE_GRID");
		int			parsed = value ? atoi(value) : 0;

		blocks = parsed >= 1 && parsed <= 64 ? parsed : 0;
	}
	if (blocks > 0)
		return (unsigned) blocks;
	if (quant == ANN_QUANT_NVFP4 && FuseLayout() >= 2)
		return FuseLayout() >= 4 ? 4 : 3;
	return 2;
}

int
AnnBlackwellScoreBelow(const AnnBlackwellShard *shard, int device, const uint32_t *rows, unsigned n,
					   const float *queries, unsigned nq, int dims, const float *bounds, float *candD,
					   uint32_t *candR, uint32_t *counts, unsigned cap, void *stream)
{
	BlackwellModule *module = &modules[device];
	unsigned	rowBytes = shard->rowBytes;
	unsigned	blocks = shard->blocks;
	int			count = (int) nq;
	int			width = nq <= 16 ? 0 : nq <= 32 ? 1 : 2;
	unsigned	shared = (16u << width) * QueryStride(dims) * sizeof(uint16_t);
	unsigned	grid = (n + 127) / 128;
	uint8_t    *fuseData = NULL;
	unsigned	fuseRowBytes = 0;
	float	   *norms = NULL;
	int			ready = FuseInputs(shard, &fuseData, &fuseRowBytes, &norms);
	void	   *args[] = {(void *) &fuseData, (void *) &shard->scales, (void *) &shard->rowScales,
		&fuseRowBytes, &blocks, (void *) &rows, &n, (void *) &queries, &count, &dims, (void *) &bounds, &candD,
	&candR, &counts, &cap, &norms};

	if (!ready || n == 0 || nq == 0 || nq > (unsigned) AnnBlackwellGroupLimit(dims) ||
		shard->quant == ANN_QUANT_NONE || atomic_load_explicit(&module->state, memory_order_acquire) <= 0 ||
		module->fuse[shard->quant - 1][width] == NULL || cuCtxSetCurrent(module->context) != CUDA_SUCCESS)
		return 0;
	if (grid > (unsigned) module->sms * FuseGridPerSm(shard->quant))
		grid = (unsigned) module->sms * FuseGridPerSm(shard->quant);
	if (shard->quant == ANN_QUANT_FP8)
		blocks = rowBytes / 16;
	else if (shard->quant >= ANN_BLACKWELL_HALF)
		blocks = ((unsigned) dims + 15) / 16;
	return cuLaunchKernel(module->fuse128 && shard->quant == ANN_QUANT_NVFP4 && width == 2 && dims == 128 &&
						  blocks == 8 ? module->fuse128 : module->fuse[shard->quant - 1][width], grid, 1, 1,
						  ANN_BLACKWELL_BLOCK, 1, 1, shared, (CUstream) stream, args, NULL) == CUDA_SUCCESS;
}

int
AnnBlackwellStrideRows(int device, const uint32_t *rows, unsigned n, unsigned s, uint32_t *out, void *stream)
{
	BlackwellModule *module = &modules[device];
	void	   *args[] = {(void *) &rows, &n, &s, &out};

	if (atomic_load_explicit(&module->state, memory_order_acquire) <= 0 || s == 0 ||
		cuCtxSetCurrent(module->context) != CUDA_SUCCESS)
		return 0;
	return cuLaunchKernel(module->strideRows, (s + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream) stream, args,
						  NULL) == CUDA_SUCCESS;
}

int
AnnBlackwellSearch(const AnnBlackwellShard *shard, int device, unsigned slot, const uint32_t *rows, unsigned n,
				   const float *queries, unsigned nq, int dims, void *stream)
{
	BlackwellModule *module = &modules[device];
	unsigned	maxBlocks,
				grid,
				perBlock;
	unsigned	rowBytes = shard->rowBytes,
				blocks = shard->blocks;
	int			dimensions = dims,
				count = (int) nq;
	SearchSlot *sc;
	CUfunction	function;

	if (shard->quant == ANN_QUANT_NONE || atomic_load_explicit(&module->state, memory_order_acquire) <= 0 || n == 0 || nq == 0 ||
		nq > (unsigned) AnnBlackwellGroupLimit(dims) || slot >= ANN_INDEX_MAX_SHARDS)
		return 0;

	/*
	 * Small lists get 128-row blocks, so enough warps are in flight to hide
	 * scattered row loads; larger ones 512-row blocks, up to four per SM, so
	 * the last block's merge of every block's list stays short. Measured on
	 * B200 (10M x 128, NVFP4): 10k rows 26.7 -> 15.4 us with 128-row blocks;
	 * 100k and 1M rows fastest with 512 (40.8 and 156 us).
	 */
	maxBlocks = (unsigned) module->sms * 4;
	grid = n <= 32768 ? (n + 127) / 128 : (n + 511) / 512;
	if (grid > maxBlocks)
		grid = maxBlocks;
	perBlock = (n + grid - 1) / grid;
	if ((sc = GetSlot(device, slot, maxBlocks)) == NULL || cuCtxSetCurrent(module->context) != CUDA_SUCCESS)
		return 0;
	if (nq == 1 && shard->quant < ANN_BLACKWELL_HALF)
	{
		function = shard->quant == ANN_QUANT_NVFP4 ? module->searchNvfp4 : module->searchFp8;
		if (shard->quant == ANN_QUANT_NVFP4)
		{
			void	   *args[] = {(void *) &shard->data, (void *) &shard->scales, (void *) &shard->rowScales,
				&rowBytes, &blocks, (void *) &rows, &n, &perBlock, (void *) &queries, &dimensions, &sc->partD,
			&sc->partR, &sc->counter, &sc->outD, &sc->outR};

			if (cuLaunchKernel(function, grid, 1, 1, ANN_BLACKWELL_BLOCK, 1, 1, 0, (CUstream) stream, args, NULL) != CUDA_SUCCESS)
				return 0;
		}
		else
		{
			void	   *args[] = {(void *) &shard->data, (void *) &shard->rowScales, &rowBytes, (void *) &rows,
				&n, &perBlock, (void *) &queries, &dimensions, &sc->partD, &sc->partR, &sc->counter, &sc->outD,
			&sc->outR};

			if (cuLaunchKernel(function, grid, 1, 1, ANN_BLACKWELL_BLOCK, 1, 1, 0, (CUstream) stream, args, NULL) != CUDA_SUCCESS)
				return 0;
		}
	}
	else
	{
		/*
		 * Several queries (or a lone one over a dataset read in place): the
		 * tensor-core kernel writes every distance to a [queries][rows]
		 * matrix, select32 keeps each query's best 32 per slice of rows, and
		 * merge32 folds the slices. About a quarter of the SMs' worth of
		 * slices was fastest on B200 (1M rows x 16 queries: 113 us against
		 * 165 with one slice per SM): long slices let a warp's list fill
		 * early and then rarely change, and the merge stays short.
		 */
		uint8_t    *fuseData = NULL;
		unsigned	fuseRowBytes = 0;
		float	   *norms = NULL;
		int			fused = n >= ANN_BLACKWELL_FUSE_ROWS && dims <= ANN_BLACKWELL_FUSE_DIMS && FuseEnabled() &&
			FuseInputs(shard, &fuseData, &fuseRowBytes, &norms);
		size_t		needed = (size_t) nq * n * (fused ? 2 : 1) + (fused ? (size_t) nq * ANN_BLACKWELL_SAMPLE : 0);
		unsigned	sgrid = (n + 127) / 128 < maxBlocks ? (n + 127) / 128 : maxBlocks;
		unsigned	cap = (unsigned) module->sms / 4 > 1 ? (unsigned) module->sms / 4 : 1;
		unsigned	slices = (n + 8191) / 8192 < cap ? (n + 8191) / 8192 : cap;
		unsigned	perSlice;
		int			width = nq <= 16 ? 0 : nq <= 32 ? 1 : 2;
		unsigned	shared = (unsigned) ((16u << width) * QueryStride(dims) * sizeof(uint16_t));

		if (slices < 1)
			slices = 1;
		perSlice = (n + slices - 1) / slices;
		if (shard->quant == ANN_QUANT_FP8)
			blocks = rowBytes / 16;
		else if (shard->quant >= ANN_BLACKWELL_HALF)
			blocks = ((unsigned) dims + 15) / 16;
		if (needed > sc->scoreCapacity)
		{
			(void) cudaFree(sc->scores);
			sc->scores = NULL;
			sc->scoreCapacity = 0;
			if (cudaMalloc((void **) &sc->scores, needed * 2 * sizeof(float)) != cudaSuccess)
				return 0;
			sc->scoreCapacity = needed * 2;
		}
		if (fused)
		{
			/*
			 * Fused: the 32nd best of a spread sample bounds each query's
			 * best 32, the scan keeps only rows at or under it (candidate
			 * lists with room for every row, so none can overflow), and the
			 * selection runs over the candidates. No score matrix is written
			 * or read: at 128 dimensions and 1M rows it took half the time.
			 */
			unsigned	sample = ANN_BLACKWELL_SAMPLE;
			unsigned	one = 1;
			unsigned	candidateSlices = 8;
			float	   *candD = sc->scores;
			unsigned   *candR = (unsigned *) (sc->scores + (size_t) nq * n);
			float	   *sampleScores = sc->scores + (size_t) nq * n * 2;
			unsigned	sampleGrid = (sample + 127) / 128 < maxBlocks ? (sample + 127) / 128 : maxBlocks;
			void	   *strideArgs[] = {(void *) &rows, &n, &sample, &sc->sampleRows};
			void	   *sampleArgs[] = {(void *) &shard->data, (void *) &shard->scales, (void *) &shard->rowScales,
			&rowBytes, &blocks, &sc->sampleRows, &sample, (void *) &queries, &count, &dimensions, &sampleScores};
			void	   *sampleSelect[] = {&sampleScores, &sample, &sample, &sc->partD, &sc->partR};
			void	   *sampleMerge[] = {&sc->partD, &sc->partR, &one, &sc->outD, &sc->outR};
			void	   *fuseArgs[] = {(void *) &fuseData, (void *) &shard->scales, (void *) &shard->rowScales,
				&fuseRowBytes, &blocks, (void *) &rows, &n, (void *) &queries, &count, &dimensions, &sc->outD, &candD,
			&candR, &sc->counts, &n, &norms};
			void	   *selectArgs[] = {&candD, &candR, &sc->counts, &n, &sc->partD, &sc->partR};
			void	   *mergeArgs[] = {&sc->partD, &sc->partR, &candidateSlices, &sc->outD, &sc->outR};

			if (cuLaunchKernel(module->strideRows, (sample + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream) stream,
							   strideArgs, NULL) != CUDA_SUCCESS ||
				cuLaunchKernel(module->mma[shard->quant - 1][width], sampleGrid, 1, 1, ANN_BLACKWELL_BLOCK, 1, 1,
							   shared, (CUstream) stream, sampleArgs, NULL) != CUDA_SUCCESS ||
				cuLaunchKernel(module->select32, 1, nq, 1, ANN_BLACKWELL_BLOCK, 1, 1, 0, (CUstream) stream,
							   sampleSelect, NULL) != CUDA_SUCCESS ||
				cuLaunchKernel(module->merge32, nq, 1, 1, 32, 1, 1, 0, (CUstream) stream, sampleMerge,
							   NULL) != CUDA_SUCCESS ||
				cudaMemsetAsync(sc->counts, 0, nq * sizeof(unsigned), (cudaStream_t) stream) != cudaSuccess ||
				cuLaunchKernel(module->fuse[shard->quant - 1][width], sgrid, 1, 1, ANN_BLACKWELL_BLOCK, 1, 1, shared,
							   (CUstream) stream, fuseArgs, NULL) != CUDA_SUCCESS ||
				cuLaunchKernel(module->select32c, candidateSlices, nq, 1, ANN_BLACKWELL_BLOCK, 1, 1, 0,
							   (CUstream) stream, selectArgs, NULL) != CUDA_SUCCESS ||
				cuLaunchKernel(module->merge32, nq, 1, 1, 32, 1, 1, 0, (CUstream) stream, mergeArgs,
							   NULL) != CUDA_SUCCESS)
				return 0;
		}
		else
		{
			void	   *args[] = {(void *) &shard->data, (void *) &shard->scales, (void *) &shard->rowScales,
			&rowBytes, &blocks, (void *) &rows, &n, (void *) &queries, &count, &dimensions, &sc->scores};
			void	   *selectArgs[] = {&sc->scores, &n, &perSlice, &sc->partD, &sc->partR};
			void	   *mergeArgs[] = {&sc->partD, &sc->partR, &slices, &sc->outD, &sc->outR};

			if (cuLaunchKernel(module->mma[shard->quant - 1][width], sgrid, 1, 1, ANN_BLACKWELL_BLOCK, 1, 1, shared,
							   (CUstream) stream, args, NULL) != CUDA_SUCCESS ||
				cuLaunchKernel(module->select32, slices, nq, 1, ANN_BLACKWELL_BLOCK, 1, 1, 0, (CUstream) stream,
							   selectArgs, NULL) != CUDA_SUCCESS ||
				cuLaunchKernel(module->merge32, nq, 1, 1, 32, 1, 1, 0, (CUstream) stream, mergeArgs, NULL) != CUDA_SUCCESS)
				return 0;
		}
	}
	return cudaMemcpyAsync(sc->hostD, sc->outD, (size_t) nq * ANN_BLACKWELL_TOP * sizeof(float),
						   cudaMemcpyDeviceToHost, (cudaStream_t) stream) == cudaSuccess &&
		cudaMemcpyAsync(sc->hostR, sc->outR, (size_t) nq * ANN_BLACKWELL_TOP * sizeof(unsigned),
						cudaMemcpyDeviceToHost, (cudaStream_t) stream) == cudaSuccess;
}

/* Device memory of count units grown to at least that; 0 when it cannot be. */
static int
GrowScratch(void **pointer, size_t *capacity, size_t count, size_t unit)
{
	if (count <= *capacity)
		return 1;
	(void) cudaFree(*pointer);
	*pointer = NULL;
	*capacity = 0;
	if (cudaMalloc(pointer, count * unit) != cudaSuccess)
	{
		*pointer = NULL;
		return 0;
	}
	*capacity = count;
	return 1;
}

static int
GrowPinnedScratch(void **pointer, size_t *capacity, size_t count, size_t unit)
{
	if (count <= *capacity)
		return 1;
	(void) cudaFreeHost(*pointer);
	*pointer = NULL;
	*capacity = 0;
	if (cudaMallocHost(pointer, count * unit) != cudaSuccess)
	{
		*pointer = NULL;
		return 0;
	}
	*capacity = count;
	return 1;
}

unsigned
AnnBlackwellCandidateCap(unsigned n, unsigned wanted)
{
	uint64_t	sample;

	if (wanted == 0)
		return 0;
	sample = ((uint64_t) 32 * n + 4 * (uint64_t) wanted - 1) / (4 * (uint64_t) wanted);
	if (sample < ANN_BLACKWELL_TOP || sample * 4 > n)
		return 0;
	return (uint64_t) wanted * 16 < n ? wanted * 16 : n;
}

unsigned
AnnBlackwellCandidates(const AnnBlackwellShard *shard, int device, unsigned slot, const uint32_t *rows, unsigned n,
					   const float *queries, unsigned nq, int dims, unsigned wanted, void *stream)
{
	BlackwellModule *module = &modules[device];
	unsigned	cap = AnnBlackwellCandidateCap(n, wanted);
	unsigned	sample = cap ? (unsigned) (((uint64_t) 32 * n + 4 * (uint64_t) wanted - 1) / (4 * (uint64_t) wanted)) : 0;
	unsigned	maxBlocks = (unsigned) module->sms * 4;
	unsigned	rowBytes = shard->rowBytes,
				blocks = shard->blocks;
	int			dimensions = dims,
				count = (int) nq;
	int			width = nq <= 16 ? 0 : nq <= 32 ? 1 : 2;
	unsigned	shared = (unsigned) ((16u << width) * QueryStride(dims) * sizeof(uint16_t));
	unsigned	sampleGrid,
				grid,
				slices,
				perSlice,
				cutoff = (unsigned) module->sms / 4 > 1 ? (unsigned) module->sms / 4 : 1;
	SearchSlot *sc;

	if (cap == 0 || shard->quant == ANN_QUANT_NONE ||
		atomic_load_explicit(&module->state, memory_order_acquire) <= 0 || nq == 0 ||
		nq > (unsigned) AnnBlackwellGroupLimit(dims) || slot >= ANN_INDEX_MAX_SHARDS ||
		(sc = GetSlot(device, slot, maxBlocks)) == NULL || cuCtxSetCurrent(module->context) != CUDA_SUCCESS ||
		!GrowScratch((void **) &sc->largeSample, &sc->largeSampleCapacity, sample, sizeof(unsigned)) ||
		!GrowScratch((void **) &sc->largeSampleScores, &sc->largeScoreCapacity, (size_t) nq * sample, sizeof(float)) ||
		!GrowScratch((void **) &sc->candD, &sc->candCapacity, (size_t) nq * cap, sizeof(float)))
		return 0;
	if (sc->candR == NULL || sc->candCapacity > sc->hostCandCapacity)
	{
		/* candR follows candD's capacity; the host copies too. */
		(void) cudaFree(sc->candR);
		sc->candR = NULL;
		if (cudaMalloc((void **) &sc->candR, sc->candCapacity * sizeof(unsigned)) != cudaSuccess ||
			!GrowPinnedScratch((void **) &sc->hostCandD, &sc->hostCandCapacity, sc->candCapacity, sizeof(float)))
			return 0;
		(void) cudaFreeHost(sc->hostCandR);
		if (cudaMallocHost((void **) &sc->hostCandR, sc->hostCandCapacity * sizeof(unsigned)) != cudaSuccess)
		{
			sc->hostCandR = NULL;
			sc->hostCandCapacity = 0;
			return 0;
		}
	}
	if (sc->hostCounts == NULL &&
		cudaMallocHost((void **) &sc->hostCounts, ANN_BLACKWELL_GROUP_MAX * sizeof(unsigned)) != cudaSuccess)
	{
		sc->hostCounts = NULL;
		return 0;
	}
	if (shard->quant == ANN_QUANT_FP8)
		blocks = rowBytes / 16;
	else if (shard->quant >= ANN_BLACKWELL_HALF)
		blocks = ((unsigned) dims + 15) / 16;
	sampleGrid = (sample + 127) / 128 < maxBlocks ? (sample + 127) / 128 : maxBlocks;
	grid = (n + 127) / 128 < maxBlocks ? (n + 127) / 128 : maxBlocks;
	slices = (sample + 8191) / 8192 < cutoff ? (sample + 8191) / 8192 : cutoff;
	if (slices < 1)
		slices = 1;
	perSlice = (sample + slices - 1) / slices;
	{
		uint8_t    *fuseData = NULL;
		unsigned	fuseRowBytes = 0;
		float	   *norms = NULL;
		int			ready = FuseInputs(shard, &fuseData, &fuseRowBytes, &norms);
		void	   *strideArgs[] = {(void *) &rows, &n, &sample, &sc->largeSample};
		void	   *sampleArgs[] = {(void *) &shard->data, (void *) &shard->scales, (void *) &shard->rowScales,
			&rowBytes, &blocks, &sc->largeSample, &sample, (void *) &queries, &count, &dimensions,
		&sc->largeSampleScores};
		void	   *selectArgs[] = {&sc->largeSampleScores, &sample, &perSlice, &sc->partD, &sc->partR};
		void	   *mergeArgs[] = {&sc->partD, &sc->partR, &slices, &sc->outD, &sc->outR};
		void	   *fuseArgs[] = {(void *) &fuseData, (void *) &shard->scales, (void *) &shard->rowScales,
			&fuseRowBytes, &blocks, (void *) &rows, &n, (void *) &queries, &count, &dimensions, &sc->outD, &sc->candD,
		&sc->candR, &sc->counts, &cap, &norms};

		if (!ready)
			return 0;

		/*
		 * The 32nd best of the sampled rows bounds each query's candidates,
		 * about 32 n / sample = 4 * wanted of them: the scan keeps only rows
		 * at or under it, up to cap per query, and counts them all.
		 */
		if (cuLaunchKernel(module->strideRows, (sample + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream) stream,
						   strideArgs, NULL) != CUDA_SUCCESS ||
			cuLaunchKernel(module->mma[shard->quant - 1][width], sampleGrid, 1, 1, ANN_BLACKWELL_BLOCK, 1, 1, shared,
						   (CUstream) stream, sampleArgs, NULL) != CUDA_SUCCESS ||
			cuLaunchKernel(module->select32, slices, nq, 1, ANN_BLACKWELL_BLOCK, 1, 1, 0, (CUstream) stream,
						   selectArgs, NULL) != CUDA_SUCCESS ||
			cuLaunchKernel(module->merge32, nq, 1, 1, 32, 1, 1, 0, (CUstream) stream, mergeArgs, NULL) != CUDA_SUCCESS ||
			cudaMemsetAsync(sc->counts, 0, nq * sizeof(unsigned), (cudaStream_t) stream) != cudaSuccess ||
			cuLaunchKernel(module->fuse[shard->quant - 1][width], grid, 1, 1, ANN_BLACKWELL_BLOCK, 1, 1, shared,
						   (CUstream) stream, fuseArgs, NULL) != CUDA_SUCCESS ||
			cudaMemcpyAsync(sc->hostCounts, sc->counts, nq * sizeof(unsigned), cudaMemcpyDeviceToHost,
							(cudaStream_t) stream) != cudaSuccess)
			return 0;
	}
	return cap;
}

const unsigned *
AnnBlackwellCandidateCounts(int device, unsigned slot)
{
	return slots[device][slot]->hostCounts;
}

int
AnnBlackwellCopyCandidates(int device, unsigned slot, unsigned nq, unsigned cap, const unsigned *take, void *stream)
{
	SearchSlot *sc = slots[device][slot];

	for (unsigned q = 0; q < nq; q++)
		if (take[q] &&
			(cudaMemcpyAsync(sc->hostCandD + (size_t) q * cap, sc->candD + (size_t) q * cap, take[q] * sizeof(float),
							 cudaMemcpyDeviceToHost, (cudaStream_t) stream) != cudaSuccess ||
			 cudaMemcpyAsync(sc->hostCandR + (size_t) q * cap, sc->candR + (size_t) q * cap, take[q] * sizeof(unsigned),
							 cudaMemcpyDeviceToHost, (cudaStream_t) stream) != cudaSuccess))
			return 0;
	return 1;
}

void
AnnBlackwellCandidateRows(int device, unsigned slot, unsigned q, unsigned cap, float **distances, unsigned **rows)
{
	SearchSlot *sc = slots[device][slot];

	*distances = sc->hostCandD + (size_t) q * cap;
	*rows = sc->hostCandR + (size_t) q * cap;
}

void
AnnBlackwellResults(int device, unsigned slot, unsigned q, const uint32_t **positions, const float **distances)
{
	SearchSlot *sc = slots[device][slot];

	*positions = sc->hostR + (size_t) q * ANN_BLACKWELL_TOP;
	*distances = sc->hostD + (size_t) q * ANN_BLACKWELL_TOP;
}
