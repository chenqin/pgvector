#ifndef ANN_FILTER_H
#define ANN_FILTER_H

#include <stddef.h>
#include <stdint.h>

/*
 * Predicate programs over an index's int32 attribute columns, evaluated the
 * same way by PostgreSQL (from frozen index pages) and by the worker (from
 * its load-time postings), so both select the same rows of the shared
 * ordinal space. A program is postfix, little-endian:
 *
 *   VALUES  u8 op, u8 name length, name, u8 negate, u32 n, i32 values[n]
 *   RANGE   u8 op, u8 name length, name, u8 negate, i32 low, i32 high
 *   NULL    u8 op, u8 name length, name, u8 negate
 *   BITMAP  u8 op, u64 stored filter id
 *   AND, OR, NOT, ALL, NONE   u8 op
 *
 * VALUES matches rows whose column is one of the values, RANGE those with
 * low <= value <= high, NULL those whose column is NULL (stored as
 * INT32_MIN). A negated leaf matches the non-NULL rows the leaf does not,
 * which is SQL's NOT for comparisons, so translators push NOT down to the
 * leaves; the NOT instruction complements over all rows. The result is a
 * bitset over ordinals, bit i of word i / 32.
 */
#define ANN_FILTER_VALUES 1
#define ANN_FILTER_RANGE 2
#define ANN_FILTER_NULL 3
#define ANN_FILTER_BITMAP 4
#define ANN_FILTER_AND 5
#define ANN_FILTER_OR 6
#define ANN_FILTER_NOT 7
#define ANN_FILTER_ALL 8
#define ANN_FILTER_NONE 9

#define ANN_FILTER_NULL_VALUE INT32_MIN
#define ANN_FILTER_MAX_DEPTH 64
#define ANN_FILTER_MAX_STEPS 1024
#define ANN_FILTER_MAX_VALUES 65536
#define ANN_FILTER_MAX_PROGRAM (1024 * 1024)

/* Evaluation results. */
#define ANN_FILTER_OK 0
#define ANN_FILTER_INVALID (-1) /* malformed program */
#define ANN_FILTER_UNKNOWN_COLUMN (-2)
#define ANN_FILTER_NO_MEMORY (-3)
#define ANN_FILTER_UNKNOWN_BITMAP (-4)
#define ANN_FILTER_SOURCE_ERROR (-5)

/*
 * Where the rows come from. A column's distinct values are sorted; value i
 * owns postings start(i) .. start(i + 1) - 1, ascending ordinals. bitset and
 * bitmap may be NULL; alloc and release default to malloc and free.
 */
typedef struct AnnFilterSource
{
	void	   *arg;
	uint64_t	rows;
	int			(*column) (void *arg, const char *name);	/* -1 if unknown */
	uint64_t	(*count) (void *arg, int column);
	int32_t		(*value) (void *arg, int column, uint64_t i);
	uint64_t	(*start) (void *arg, int column, uint64_t i);
	int			(*postings) (void *arg, int column, uint64_t first, uint64_t n, uint32_t *out);
	int			(*bitset) (void *arg, int column, uint64_t i, uint32_t *words);
	int			(*bitmap) (void *arg, uint64_t id, uint32_t *words);
	void	   *(*alloc) (void *arg, size_t size);
	void		(*release) (void *arg, void *pointer);
} AnnFilterSource;

/* Whether a program is well formed: known opcodes, sizes, one result. */
int			AnnFilterCheck(const unsigned char *program, size_t size);

/* Evaluate into words ((rows + 31) / 32 of them); ANN_FILTER_OK or an error. */
int			AnnFilterEvaluate(const AnnFilterSource *source, const unsigned char *program, size_t size,
							  uint32_t *words, uint64_t *count);

/* Cache id of a program: the same text hashes the same on every host. */
uint64_t	AnnFilterHash(const unsigned char *program, size_t size);

#endif
