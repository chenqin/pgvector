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

#endif
