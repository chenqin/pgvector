#include "select_batch.h"

#include <cuvs/selection/select_k.hpp>
#include <exception>
#include <optional>
#include <stdexcept>

extern "C" cuvsError_t
AnnSelectBatch(cuvsResources_t resources, const float *scores, const int64_t *ids,
               float *values, int64_t *positions, int64_t queries, int64_t rows,
               int64_t k, int sorted)
{
    try {
        if (!resources || !scores || !values || !positions || queries < 1 || rows < 1 || k < 1 || k > rows)
            throw std::invalid_argument("invalid ANN matrix selection dimensions or buffers");
        auto input = raft::make_device_matrix_view<const float, int64_t>(scores, queries, rows);
        auto output = raft::make_device_matrix_view<float, int64_t>(values, queries, k);
        auto indices = raft::make_device_matrix_view<int64_t, int64_t>(positions, queries, k);
        std::optional<raft::device_matrix_view<const int64_t, int64_t, raft::row_major>> payload;
        if (ids)
            payload = raft::make_device_matrix_view<const int64_t, int64_t>(ids, queries, rows);
        cuvs::selection::select_k(*reinterpret_cast<raft::resources *>(resources), input,
                                 payload, output, indices, true, sorted != 0,
                                 cuvs::selection::SelectAlgo::kAuto);
        return CUVS_SUCCESS;
    } catch (const std::exception &error) {
        cuvsSetLastErrorText(error.what());
        return CUVS_ERROR;
    } catch (...) {
        cuvsSetLastErrorText("unknown ANN matrix selection failure");
        return CUVS_ERROR;
    }
}
