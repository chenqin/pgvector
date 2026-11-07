#ifndef ANN_CANONICAL_H
#define ANN_CANONICAL_H

#include "filter.h"

#include <stddef.h>

/*
 * A predicate program (ann/filter.h) rewritten into one form per row set
 * this source can tell apart, so equivalent programs share a cache entry.
 * On a single-valued column every row has exactly one value (NULL
 * included), so each leaf there selects the rows whose value lies in a set
 * of the column's distinct values: values the column lacks drop out, ranges
 * snap to the values they hold, negation and NOT complement the set, and
 * AND/OR of leaves on one column merge into one. Empty and full sets fold to
 * NONE and ALL, AND/OR operands are flattened, deduplicated and sorted, and
 * double NOTs cancel. Leaves on multi-valued columns are kept as written.
 *
 * Returns 1 and a malloc'd program that AnnFilterEvaluate maps to the same
 * rows, or 0 (use the original): programs with BITMAP leaves, leaves the
 * source would reject, and results past the program limits.
 */
int			AnnFilterCanonical(const AnnFilterSource *source, const unsigned char *program, size_t size,
							   unsigned char **out, size_t *outSize);

/*
 * A canonical AND split into a base and exclusions: operands on one
 * single-valued column that keep all but a few rows (such as "tenant <> 7")
 * become exclusions, given as runs of the column's distinct-value indices
 * whose rows are removed; the other operands form the base program. The
 * program's rows are the base's rows minus every exclusion's rows. Returns 1
 * with a malloc'd base when at least one exclusion and one base operand
 * exist and the exclusions hold at most maxExcluded rows; otherwise 0.
 */
#define ANN_FILTER_MAX_EXCLUSIONS 4
#define ANN_FILTER_EXCLUSION_RUNS 16

typedef struct AnnFilterExclusion
{
	int			column;			/* the source's column index */
	uint32_t	nruns;
	uint64_t	lo[ANN_FILTER_EXCLUSION_RUNS];	/* value indices, half-open */
	uint64_t	hi[ANN_FILTER_EXCLUSION_RUNS];
} AnnFilterExclusion;

int			AnnFilterCanonicalSplit(const AnnFilterSource *source, const unsigned char *program, size_t size,
									uint64_t maxExcluded, unsigned char **base, size_t *baseSize,
									AnnFilterExclusion *exclusions, uint32_t *nexclusions);

#endif
