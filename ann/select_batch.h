#ifndef ANN_SELECT_BATCH_H
#define ANN_SELECT_BATCH_H

#include <cuvs/core/c_api.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Matrix top-k, optionally carrying row IDs rather than column positions. */
cuvsError_t AnnSelectBatch(cuvsResources_t resources, const float *scores, const int64_t *ids,
                          float *values, int64_t *positions, int64_t queries, int64_t rows,
                          int64_t k, int sorted);

#ifdef __cplusplus
}
#endif
#endif
