#include "cagra_udf.h"

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>
#include <exception>
#include <stdexcept>

/*
 * The device twin of AnnUdfMatch (ann/udf.c). cuVS wraps this source in its
 * JIT-LTO sample filter fragment and keys its compiled-fragment cache by the
 * whole source, so a constant source compiles once per process; queries vary
 * only the AnnUdfParams payload. Opcodes are ann/filter.h's.
 */
static const char interpreterSource[] = R"cuda(
struct ann_udf_step {
  unsigned char op, column, negate, pad;
  int low, high;
  unsigned int first, count;
};
struct ann_udf_params {
  const int* columns[8];
  const ann_udf_step* steps;
  const int* values;
  const unsigned int* starts;
  unsigned long long row_base;
};

__device__ bool ann_udf_leaf(const ann_udf_step& s, const int* values, int v)
{
  const int null_value = (int)0x80000000;
  bool match;
  if (s.op == 3) {
    match = v == null_value;
  } else if (v == null_value) {
    return false;
  } else if (s.op == 2) {
    match = s.low <= v && v <= s.high;
  } else {
    unsigned int lo = s.first, hi = s.first + s.count;
    while (lo < hi) {
      unsigned int mid = lo + (hi - lo) / 2;
      if (values[mid] < v) lo = mid + 1; else hi = mid;
    }
    match = lo < s.first + s.count && values[lo] == v;
  }
  return s.negate ? !match : match;
}

__device__ bool ann_predicate_udf(uint32_t query_id, source_index_t source_id, void* filter_data)
{
  const ann_udf_params* p = static_cast<const ann_udf_params*>(filter_data);
  const unsigned long long row = p->row_base + source_id;
  const unsigned int end = p->starts[query_id + 1];
  unsigned long long stack = 0;
  int depth = 0;
  for (unsigned int i = p->starts[query_id]; i < end; i++) {
    const ann_udf_step s = p->steps[i];
    unsigned long long bit;
    if (s.op == 5 || s.op == 6) {
      unsigned long long b = (stack >> (depth - 1)) & 1ull;
      unsigned long long a = (stack >> (depth - 2)) & 1ull;
      depth -= 2;
      bit = s.op == 5 ? (a & b) : (a | b);
    } else if (s.op == 7) {
      bit = ((stack >> (depth - 1)) & 1ull) ^ 1ull;
      depth--;
    } else if (s.op == 8) {
      bit = 1;
    } else if (s.op == 9) {
      bit = 0;
    } else {
      bit = ann_udf_leaf(s, p->values, p->columns[s.column][row]) ? 1ull : 0ull;
    }
    stack = (stack & ~(1ull << depth)) | (bit << depth);
    depth++;
  }
  return depth == 1 && (stack & 1ull);
}
)cuda";

/*
 * cuvsCagraIndex::addr points at cuVS 26.10's private index box, whose first
 * member is the cagra::index pointer and second its dataset layout (see
 * cagra_c_api_index_ptr in cuVS' c/src/neighbors/cagra.cpp). The worker
 * refuses other cuVS versions at load, so this mirror stays in step.
 */
namespace {
struct IndexBox {
	void	   *index;
	uint8_t		layout;			/* 0: device padded */
};
}

extern "C" const char *
AnnCagraUdfSource(void)
{
	return interpreterSource;
}

extern "C" cuvsError_t
AnnCagraUdfSearch(cuvsResources_t resources, cuvsCagraSearchParams_t params,
				  cuvsCagraIndex_t index, const float *queries, int64_t m, int64_t dims,
				  uint32_t *neighbors, float *distances, int64_t k,
				  const AnnUdfParams *deviceParams, float filtering_rate)
{
	try
	{
		namespace cagra = cuvs::neighbors::cagra;
		const IndexBox *box;

		if (!resources || !params || !index || !index->addr || !queries || !neighbors || !distances ||
			!deviceParams || m < 1 || dims < 1 || k < 1)
			throw std::invalid_argument("invalid CAGRA UDF search arguments");
		if (index->dtype.code != kDLFloat || index->dtype.bits != 32)
			throw std::invalid_argument("CAGRA UDF search needs an FP32 index");
		box = reinterpret_cast<const IndexBox *>(index->addr);
		if (box->index == nullptr || box->layout != 0)
			throw std::invalid_argument("CAGRA UDF search needs a device padded dataset");

		auto	   &graph = *reinterpret_cast<cagra::device_padded_index<float, uint32_t> *>(box->index);

		if (graph.dim() != dims)
			throw std::invalid_argument("CAGRA UDF search: query dimensions differ from the index");

		cagra::search_params search;

		/* As cuVS' convert_c_search_params. */
		search.max_queries = params->max_queries;
		search.itopk_size = params->itopk_size;
		search.max_iterations = params->max_iterations;
		search.algo = static_cast<cagra::search_algo>(params->algo);
		search.team_size = params->team_size;
		search.search_width = params->search_width;
		search.min_iterations = params->min_iterations;
		search.thread_block_size = params->thread_block_size;
		search.hashmap_mode = static_cast<cagra::hash_mode>(params->hashmap_mode);
		search.hashmap_min_bitlen = params->hashmap_min_bitlen;
		search.hashmap_max_fill_rate = params->hashmap_max_fill_rate;
		search.num_random_samplings = params->num_random_samplings;
		search.rand_xor_mask = params->rand_xor_mask;
		search.persistent = params->persistent;
		search.persistent_lifetime = params->persistent_lifetime;
		search.persistent_device_usage = params->persistent_device_usage;

		cuvs::neighbors::filtering::udf_filter filter(
			interpreterSource, const_cast<AnnUdfParams *>(deviceParams),
			filtering_rate >= 0.0f && filtering_rate < 1.0f ? filtering_rate : -1.0f,
			"ann_predicate_udf");

		cagra::search(*reinterpret_cast<raft::resources *>(resources), search, graph,
					  raft::make_device_matrix_view<const float, int64_t>(queries, m, dims),
					  raft::make_device_matrix_view<uint32_t, int64_t>(neighbors, m, k),
					  raft::make_device_matrix_view<float, int64_t>(distances, m, k), filter);
		return CUVS_SUCCESS;
	}
	catch (const std::exception &error)
	{
		cuvsSetLastErrorText(error.what());
		return CUVS_ERROR;
	}
	catch (...)
	{
		cuvsSetLastErrorText("unknown CAGRA UDF search failure");
		return CUVS_ERROR;
	}
}
