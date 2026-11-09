/*
 * The row-by-row predicate interpreter (ann/server/udf.c, twin of the CAGRA filter
 * UDF) must select exactly the rows AnnFilterEvaluate puts in its bitset:
 * random columns with NULLs, random programs with negated leaves, NOT,
 * AND/OR, ALL and NONE.
 */
#include "filter.h"
#include "udf.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROWS 5003
#define COLUMNS 3

static int32_t data[COLUMNS][ROWS];
static const int32_t *columns[COLUMNS];
static int32_t distinct[COLUMNS][ROWS];
static uint64_t ndistinct[COLUMNS];
static uint64_t starts[COLUMNS][ROWS + 1];
static uint32_t postings[COLUMNS][ROWS];
static const char *names[COLUMNS] = {"a", "b", "c"};
static int	sortColumn;

static int
ByValue(const void *left, const void *right)
{
	int32_t		x = data[sortColumn][*(const uint32_t *) left];
	int32_t		y = data[sortColumn][*(const uint32_t *) right];
	uint32_t	l = *(const uint32_t *) left,
				r = *(const uint32_t *) right;

	if (x != y)
		return (x > y) - (x < y);
	return (l > r) - (l < r);
}

static int
Column(void *arg, const char *name)
{
	(void) arg;
	for (int c = 0; c < COLUMNS; c++)
		if (strcmp(name, names[c]) == 0)
			return c;
	return -1;
}

static uint64_t
Count(void *arg, int c)
{
	(void) arg;
	return ndistinct[c];
}

static int32_t
Value(void *arg, int c, uint64_t i)
{
	(void) arg;
	return distinct[c][i];
}

static uint64_t
Start(void *arg, int c, uint64_t i)
{
	(void) arg;
	return starts[c][i];
}

static int
Postings(void *arg, int c, uint64_t first, uint64_t n, uint32_t *out)
{
	(void) arg;
	memcpy(out, postings[c] + first, n * sizeof(uint32_t));
	return 1;
}

static uint64_t
Rows(void *arg, int c, int32_t low, int32_t high)
{
	uint64_t	n = 0;

	(void) arg;
	for (uint32_t r = 0; r < ROWS; r++)
		n += data[c][r] >= low && data[c][r] <= high;
	return n;
}

typedef struct
{
	unsigned char b[8192];
	size_t		n;
} Program;

static void
Put(Program *p, uint64_t v, int bytes)
{
	for (int i = 0; i < bytes; i++)
		p->b[p->n++] = (unsigned char) (v >> (8 * i));
}

static int32_t
RandomValue(void)
{
	return rand() % 8 == 0 ? ANN_FILTER_NULL_VALUE : rand() % 40 - 20;
}

/* A random well-formed expression of at most depth levels. */
static void
Expression(Program *p, int depth)
{
	int			kind = depth == 0 ? rand() % 5 : rand() % 9;

	if (kind < 3)
	{
		int			op = kind == 0 ? ANN_FILTER_VALUES : kind == 1 ? ANN_FILTER_RANGE : ANN_FILTER_NULL;
		const char *name = names[rand() % COLUMNS];

		Put(p, op, 1);
		Put(p, 1, 1);
		Put(p, (unsigned char) name[0], 1);
		Put(p, rand() % 3 == 0, 1);
		if (op == ANN_FILTER_RANGE)
		{
			int32_t		lo = rand() % 10 == 0 ? INT32_MIN : rand() % 50 - 25;

			Put(p, (uint32_t) lo, 4);
			Put(p, (uint32_t) (rand() % 10 == 0 ? INT32_MAX : lo == INT32_MIN ? 0 : lo + rand() % 20 - 3), 4);
		}
		else if (op == ANN_FILTER_VALUES)
		{
			uint32_t	n = rand() % 6;

			Put(p, n, 4);
			for (uint32_t i = 0; i < n; i++)
				Put(p, (uint32_t) RandomValue(), 4);
		}
	}
	else if (kind == 3)
		Put(p, ANN_FILTER_ALL, 1);
	else if (kind == 4)
		Put(p, ANN_FILTER_NONE, 1);
	else if (kind == 5)
	{
		Expression(p, depth - 1);
		Put(p, ANN_FILTER_NOT, 1);
	}
	else
	{
		Expression(p, depth - 1);
		Expression(p, depth - 1);
		Put(p, kind % 2 ? ANN_FILTER_AND : ANN_FILTER_OR, 1);
	}
}

int
main(void)
{
	AnnFilterSource source;
	uint32_t   *words = malloc(((ROWS + 31) / 32) * sizeof(uint32_t));
	AnnUdfStep *steps = malloc(sizeof(AnnUdfStep) * 8192);
	int32_t    *values = malloc(sizeof(int32_t) * 8192);
	int			programs = 0;

	srand(7);
	for (int c = 0; c < COLUMNS; c++)
	{
		uint32_t	order[ROWS];

		for (uint32_t r = 0; r < ROWS; r++)
		{
			data[c][r] = RandomValue();
			order[r] = r;
		}
		columns[c] = data[c];
		sortColumn = c;
		qsort(order, ROWS, sizeof(uint32_t), ByValue);
		for (uint32_t r = 0; r < ROWS; r++)
		{
			if (r == 0 || data[c][order[r]] != data[c][order[r - 1]])
			{
				distinct[c][ndistinct[c]] = data[c][order[r]];
				starts[c][ndistinct[c]++] = r;
			}
			postings[c][r] = order[r];
		}
		starts[c][ndistinct[c]] = ROWS;
	}
	memset(&source, 0, sizeof(source));
	source.rows = ROWS;
	source.column = Column;
	source.count = Count;
	source.value = Value;
	source.start = Start;
	source.postings = Postings;

	for (int t = 0; t < 3000; t++)
	{
		Program		p = {.n = 0};
		uint64_t	count;
		uint32_t	nsteps = 0,
					nvalues = 0;
		uint64_t	matched = 0;
		double		estimate;

		Expression(&p, 1 + rand() % 5);
		assert(AnnFilterCheck(p.b, p.n));
		assert(AnnFilterEvaluate(&source, p.b, p.n, words, &count) == ANN_FILTER_OK);
		assert(AnnUdfCompile(p.b, p.n, Column, NULL, steps, 8192, &nsteps, values, 8192, &nvalues) ==
			   ANN_FILTER_OK);
		for (uint32_t r = 0; r < ROWS; r++)
		{
			int			expected = (words[r / 32] >> (r % 32)) & 1;
			int			got = AnnUdfMatch(steps, nsteps, values, columns, r);

			if (expected != got)
			{
				fprintf(stderr, "program %d row %u: bitset %d, udf %d\n", t, r, expected, got);
				return 1;
			}
			matched += (uint64_t) got;
		}
		assert(matched == count);
		estimate = AnnUdfEstimate(steps, nsteps, values, Rows, NULL, ROWS);
		assert(estimate >= 0 && estimate <= ROWS + 1e-6);
		/* A single leaf's estimate is exact. */
		if (nsteps == 1)
			assert((uint64_t) (estimate + 0.5) == count);
		programs++;
	}

	/* BITMAP leaves are not compiled; unknown columns are refused. */
	{
		Program		p = {.n = 0};
		uint32_t	nsteps = 0,
					nvalues = 0;

		Put(&p, ANN_FILTER_BITMAP, 1);
		Put(&p, 42, 8);
		assert(AnnUdfCompile(p.b, p.n, Column, NULL, steps, 8192, &nsteps, values, 8192, &nvalues) ==
			   ANN_UDF_UNSUPPORTED);
		p.n = 0;
		Put(&p, ANN_FILTER_NULL, 1);
		Put(&p, 1, 1);
		Put(&p, 'z', 1);
		Put(&p, 0, 1);
		assert(AnnUdfCompile(p.b, p.n, Column, NULL, steps, 8192, &nsteps, values, 8192, &nvalues) ==
			   ANN_FILTER_UNKNOWN_COLUMN);
		p.n = 0;
		Put(&p, ANN_FILTER_ALL, 1);
		assert(AnnUdfCompile(p.b, p.n, Column, NULL, steps, 0, &nsteps, values, 8192, &nvalues) ==
			   ANN_UDF_TOO_LARGE);
	}
	printf("udf: %d programs match AnnFilterEvaluate\n", programs);
	free(words);
	free(steps);
	free(values);
	return 0;
}
