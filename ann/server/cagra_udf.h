#ifndef ANN_CAGRA_UDF_H
#define ANN_CAGRA_UDF_H

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "udf.h"

/*
 * Filtered FP32 CAGRA search whose eligibility is a JIT-LTO user-defined
 * filter (cuVS filtering::udf_filter) instead of a bitset: the predicate
 * interpreter of udf.h runs on the device for each (query, row) the
 * search would emit. params points to an AnnUdfParams in device memory. The
 * interpreter's source never changes, so cuVS compiles it once per process.
 *
 * filtering_rate is the estimated fraction of rows rejected (cuVS sizes its
 * search with it), or negative when unknown. Requires cuVS 26.10 and an
 * index loaded through the C API with a device padded FP32 dataset.
 */
cuvsError_t AnnCagraUdfSearch(cuvsResources_t resources, cuvsCagraSearchParams_t params,
							  cuvsCagraIndex_t index, const float *queries, int64_t m, int64_t dims,
							  uint32_t *neighbors, float *distances, int64_t k,
							  const AnnUdfParams *deviceParams, float filtering_rate);

/* The interpreter's CUDA source, as passed to cuVS (for tests and logs). */
const char *AnnCagraUdfSource(void);

#ifdef __cplusplus
}
#endif
#endif
