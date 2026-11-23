/*
 * GPU filter evaluation from postings kept in device memory: a compiled predicate (ann/server/udf.h steps)
 * becomes a bitset over the shard's rows by scattering the rows of each leaf's values, then combining
 * leaves word by word, instead of testing every row (the worker's GPU evaluator) or evaluating on the host.
 * A prototype for cagra_twohop_udf_probe: a leaf costs time in the rows it passes, a combination one pass
 * over the bitset's words.
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

extern "C" {
#include "udf.h"
}

/* One column's postings: distinct values ascending, value i owns postings[starts[i] .. starts[i + 1]). */
struct PostingsColumn {
  bool multi = false;
  std::vector<int32_t> distinct;   // host copy, for planning
  std::vector<uint64_t> starts;    // host copy, distinct + 1
  uint32_t* postings     = nullptr;  // device, rows ascending within a value
  uint32_t* null_words   = nullptr;  // device bitset: NULL rows (NULL arrays for a multi column)
  uint64_t null_count    = 0;
};

/* Builds postings for a column from its host data (multi: elements + rows + 1 offsets). */
PostingsColumn postings_build(const int32_t* data, const uint32_t* offsets, uint64_t rows);

/* Per-thread device scratch: a stack of bitsets and a range list. */
struct PostingsScratch {
  uint64_t rows = 0;
  size_t nwords = 0;
  std::vector<uint32_t*> stack;            // device bitsets, ANN_FILTER_MAX_DEPTH of them, lazily
  uint64_t* ranges        = nullptr;       // device: (first << 32 | count) pairs as two u64 arrays
  size_t range_capacity   = 0;
  std::vector<uint64_t> host_ranges;
};

void postings_scratch_init(PostingsScratch& s, uint64_t rows);

/*
 * Evaluates steps[0 .. nsteps) (values: the compiled values array) into out (nwords device words) on stream.
 * columns[c] is the step column c's postings. Returns the device bitset holding the result (one of the
 * scratch's or out).
 */
void postings_eval(const std::vector<PostingsColumn*>& columns, const AnnUdfStep* steps, uint32_t nsteps,
                   const int32_t* values, PostingsScratch& s, uint32_t* out, cudaStream_t stream);
