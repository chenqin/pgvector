/*
 * The row-by-row predicate interpreter (ann/server/udf.c, twin of the CAGRA filter
 * UDF) must select exactly the rows AnnFilterEvaluate puts in its bitset:
 * random columns with NULLs, a multi-valued column with NULL and empty
 * arrays and repeated elements, random programs with negated leaves,
 * OVERLAP, NOT, AND/OR, ALL and NONE.
 */
#include "filter.h"
#include "udf.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROWS 5003
#define COLUMNS 3				/* single-valued "a", "b", "c" */
#define MULTI COLUMNS			/* the multi-valued "m", after them */
#define MAX_ELEMENTS (ROWS * 5)

static int32_t data[COLUMNS][ROWS];
static const int32_t *columns[COLUMNS];
static int32_t distinct[COLUMNS][ROWS];
static uint64_t ndistinct[COLUMNS];
static uint64_t starts[COLUMNS][ROWS + 1];
static uint32_t postings[COLUMNS][ROWS];
static const char *names[COLUMNS] = {"a", "b", "c"};
static int	sortColumn;

/* "m": row r's array is elements[offsets[r] .. offsets[r + 1]); a NULL array is one INT32_MIN. */
static int32_t elements[MAX_ELEMENTS];
static uint32_t offsets[ROWS + 1];
static int32_t mDistinct[MAX_ELEMENTS];
static uint64_t mNdistinct;
static uint64_t mStarts[MAX_ELEMENTS + 1];
static uint32_t mPostings[MAX_ELEMENTS];

typedef struct
{
	int32_t		value;
	uint32_t	row;
} Posting;

static int
ByPosting(const void *left, const void *right)
{
	const Posting *x = left,
			   *y = right;

	if (x->value != y->value)
		return (x->value > y->value) - (x->value < y->value);
	return (x->row > y->row) - (x->row < y->row);
}

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

/* AnnFilterEvaluate's column map: "m" is column MULTI. */
static int
Column(void *arg, const char *name)
{
	(void) arg;
	for (int c = 0; c < COLUMNS; c++)
		if (strcmp(name, names[c]) == 0)
			return c;
	return strcmp(name, "m") == 0 ? MULTI : -1;
}

/* AnnUdfCompile's: the same, with "m" marked multi-valued. */
static int
UdfColumn(void *arg, const char *name)
{
	int			c = Column(arg, name);

	return c == MULTI ? (MULTI | ANN_UDF_MULTI) : c;
}

/* A caller that knows no multi-valued columns: the map before ANN_UDF_MULTI. */
static int
SingleColumn(void *arg, const char *name)
{
	int			c = Column(arg, name);

	return c == MULTI ? -1 : c;
}

static int
Multi(void *arg, int c)
{
	(void) arg;
	return c == MULTI;
}

static uint64_t
Count(void *arg, int c)
{
	(void) arg;
	return c == MULTI ? mNdistinct : ndistinct[c];
}

static int32_t
Value(void *arg, int c, uint64_t i)
{
	(void) arg;
	return c == MULTI ? mDistinct[i] : distinct[c][i];
}

static uint64_t
Start(void *arg, int c, uint64_t i)
{
	(void) arg;
	return c == MULTI ? mStarts[i] : starts[c][i];
}

static int
Postings(void *arg, int c, uint64_t first, uint64_t n, uint32_t *out)
{
	(void) arg;
	memcpy(out, (c == MULTI ? mPostings : postings[c]) + first, n * sizeof(uint32_t));
	return 1;
}

/* Rows with a value in [low, high]; for "m", rows holding such an element (NULL: NULL arrays). */
static uint64_t
Rows(void *arg, int c, int32_t low, int32_t high)
{
	uint64_t	n = 0;

	(void) arg;
	for (uint32_t r = 0; r < ROWS; r++)
	{
		int			hit = 0;

		if (c != MULTI)
			hit = data[c][r] >= low && data[c][r] <= high;
		else
			for (uint32_t i = offsets[r]; i < offsets[r + 1] && !hit; i++)
				hit = elements[i] >= low && elements[i] <= high;
		n += (uint64_t) hit;
	}
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
		int			column = rand() % (COLUMNS + 1);
		const char *name = column == MULTI ? "m" : names[column];

		/* "m" takes OVERLAP and NULL leaves only. */
		if (column == MULTI && op != ANN_FILTER_NULL)
			op = ANN_FILTER_OVERLAP;
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
		else if (op == ANN_FILTER_VALUES || op == ANN_FILTER_OVERLAP)
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
	int			multiPrograms = 0;
	const int32_t *all[COLUMNS + 1];
	const uint32_t *allOffsets[COLUMNS + 1] = {NULL, NULL, NULL, offsets};

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
		all[c] = data[c];
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
	/* "m": NULL arrays, empty arrays, and 1-5 elements that may repeat. */
	for (uint32_t r = 0, at = 0; r < ROWS; r++)
	{
		int			kind = rand() % 10;
		int			n = kind == 0 ? 0 : 1 + rand() % 5;

		offsets[r] = at;
		if (kind == 1)
			elements[at++] = ANN_FILTER_NULL_VALUE;
		else
			for (int i = 0; i < n; i++)
				elements[at++] = rand() % 30 - 15;
		offsets[r + 1] = at;
	}
	{
		static Posting all[MAX_ELEMENTS];
		uint64_t	n = 0;

		for (uint32_t r = 0; r < ROWS; r++)
			for (uint32_t i = offsets[r]; i < offsets[r + 1]; i++)
			{
				all[n].value = elements[i];
				all[n++].row = r;
			}
		qsort(all, n, sizeof(Posting), ByPosting);
		for (uint64_t i = 0, kept = 0; i < n; i++)
		{
			/* A row appears once under each distinct element. */
			if (i > 0 && all[i].value == all[i - 1].value && all[i].row == all[i - 1].row)
				continue;
			if (kept == 0 || all[i].value != mDistinct[mNdistinct - 1])
			{
				mDistinct[mNdistinct] = all[i].value;
				mStarts[mNdistinct++] = kept;
			}
			mPostings[kept++] = all[i].row;
			mStarts[mNdistinct] = kept;
		}
	}
	all[MULTI] = elements;
	memset(&source, 0, sizeof(source));
	source.rows = ROWS;
	source.column = Column;
	source.count = Count;
	source.value = Value;
	source.start = Start;
	source.postings = Postings;
	source.multi = Multi;

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
		assert(AnnUdfCompile(p.b, p.n, UdfColumn, NULL, steps, 8192, &nsteps, values, 8192, &nvalues) ==
			   ANN_FILTER_OK);
		for (uint32_t r = 0; r < ROWS; r++)
		{
			int			expected = (words[r / 32] >> (r % 32)) & 1;
			int			got = AnnUdfMatchMulti(steps, nsteps, values, all, allOffsets, r);

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

		/*
		 * A single leaf's estimate is exact, except OVERLAP's (an upper bound
		 * unless negated).
		 */
		if (nsteps == 1 && !steps[0].multi)
			assert((uint64_t) (estimate + 0.5) == count);
		if (nsteps == 1 && steps[0].multi && steps[0].op == ANN_FILTER_OVERLAP && !steps[0].negate)
			assert(estimate + 1e-6 >= (double) count);
		multiPrograms += memchr(p.b, 'm', p.n) != NULL;
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

		/*
		 * OVERLAP needs a multi-valued column the caller marks; VALUES/RANGE
		 * refuse one.
		 */
		p.n = 0;
		Put(&p, ANN_FILTER_OVERLAP, 1);
		Put(&p, 1, 1);
		Put(&p, 'm', 1);
		Put(&p, 0, 1);
		Put(&p, 1, 4);
		Put(&p, 3, 4);
		assert(AnnUdfCompile(p.b, p.n, SingleColumn, NULL, steps, 8192, &nsteps, values, 8192, &nvalues) ==
			   ANN_UDF_UNSUPPORTED);
		assert(AnnUdfCompile(p.b, p.n, Column, NULL, steps, 8192, &nsteps, values, 8192, &nvalues) ==
			   ANN_UDF_UNSUPPORTED);
		nsteps = nvalues = 0;
		assert(AnnUdfCompile(p.b, p.n, UdfColumn, NULL, steps, 8192, &nsteps, values, 8192, &nvalues) ==
			   ANN_FILTER_OK);
		p.b[0] = ANN_FILTER_VALUES;
		assert(AnnUdfCompile(p.b, p.n, UdfColumn, NULL, steps, 8192, &nsteps, values, 8192, &nvalues) ==
			   ANN_FILTER_COLUMN_KIND);
	}
	assert(multiPrograms > 500);
	printf("udf: %d programs (%d on the multi-valued column) match AnnFilterEvaluate\n", programs, multiPrograms);
	free(words);
	free(steps);
	free(values);
	return 0;
}
