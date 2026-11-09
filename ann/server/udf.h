#ifndef ANN_UDF_H
#define ANN_UDF_H

#include <stddef.h>
#include <stdint.h>

/*
 * Predicate programs (ann/common/filter.h) compiled for evaluation row by row, as a
 * CAGRA filter UDF does on the GPU, instead of into a bitset over every row.
 * A compiled program is a postfix list of steps over int32 attribute
 * columns; a batch concatenates several programs, one per query, with
 * starts[q] .. starts[q + 1] - 1 the steps of query q. Leaves follow
 * AnnFilterEvaluate: VALUES and RANGE never match NULL, a negated leaf
 * matches the non-NULL rows the leaf does not, and NOT complements over all
 * rows. BITMAP leaves (stored filters) and OVERLAP leaves (multi-valued
 * attributes) are not compiled.
 *
 * The device interpreter (AnnUdfDeviceSource) declares these structures
 * again; their layout is fixed: 20-byte steps, 8-byte pointers.
 */
#define ANN_UDF_MAX_COLUMNS 8

typedef struct AnnUdfStep
{
	uint8_t		op;				/* ANN_FILTER_* */
	uint8_t		column;			/* leaves: index into columns */
	uint8_t		negate;
	uint8_t		pad;
	int32_t		low;			/* RANGE bounds, inclusive */
	int32_t		high;
	uint32_t	first;			/* VALUES: sorted unique values[first..] */
	uint32_t	count;
} AnnUdfStep;

typedef struct AnnUdfParams
{
	const int32_t *columns[ANN_UDF_MAX_COLUMNS];	/* by global ordinal */
	const AnnUdfStep *steps;
	const int32_t *values;
	const uint32_t *starts;		/* queries + 1 offsets into steps */
	uint64_t	rowBase;		/* global ordinal of the shard's row 0 */
} AnnUdfParams;

/* AnnUdfCompile results besides the ANN_FILTER_* codes. */
#define ANN_UDF_UNSUPPORTED (-10)	/* a BITMAP or OVERLAP leaf */
#define ANN_UDF_TOO_LARGE (-11)

/*
 * Append a checked program's steps and values (capacity: steps of at most
 * size, values of at most size / 4 more). column maps an attribute name to
 * its index in the columns array, or -1.
 */
int			AnnUdfCompile(const unsigned char *program, size_t size,
						  int (*column) (void *arg, const char *name), void *arg,
						  AnnUdfStep *steps, uint32_t maxSteps, uint32_t *nsteps,
						  int32_t *values, uint32_t maxValues, uint32_t *nvalues);

/* Whether row matches the steps; the host twin of the device interpreter. */
int			AnnUdfMatch(const AnnUdfStep *steps, uint32_t nsteps, const int32_t *values,
						const int32_t *const *columns, uint64_t row);

/*
 * Estimated matching rows: exact for one leaf, assuming independent leaves
 * otherwise. rows(arg, column, low, high) counts rows whose value is in
 * [low, high]; total is the row count.
 */
double		AnnUdfEstimate(const AnnUdfStep *steps, uint32_t nsteps, const int32_t *values,
						   uint64_t (*rows) (void *arg, int column, int32_t low, int32_t high),
						   void *arg, uint64_t total);

/*
 * CUDA source of the device interpreter, ann_predicate_udf(query_id,
 * source_id, filter_data) over an ann_udf_params (AnnUdfParams); the
 * includer defines uint32_t and source_index_t.
 */
const char *AnnUdfDeviceSource(void);

#endif
