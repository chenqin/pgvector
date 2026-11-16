/*
 * The predicate evaluator (ann/common/filter.c) against a row-by-row
 * reference: random programs of OVERLAP leaves on a multi-valued column and
 * VALUES leaves on a single-valued one, negated or not, under AND, OR and
 * NOT. Leaves are sized so that both the sorted-list path (lists of several
 * posting runs, which must be sorted and deduplicated) and the bitset path
 * run. Over 100,003 rows, and over 4,194,319, where ordinals of 2^22 and
 * above (the radix sort's third digit) are drawn into those lists. Each
 * program is evaluated again with the sort's scratch allocation failing:
 * its fallback must give the same rows.
 */
#include "filter.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ELEMENTS 6
#define OUTSIDE 50				/* leaf values past the domain: no rows */
#define HIGH_ORDINAL (UINT32_C(1) << 22)

/* The run's shape. */
static uint32_t nrows;
static int	domain;				/* multi-valued elements: 0 .. domain - 1 */
static int	singleDomain;		/* single values: 0 .. singleDomain - 1 */

/* Column 0, "m": multi-valued; column 1, "s": single-valued. */
static int32_t *elements;		/* MAX_ELEMENTS per row */
static signed char *nelements;	/* -1: a NULL array */
static int32_t *single;

/* Values held by rows of ordinal 2^22 and above, which leaves favor. */
static int32_t highValues[2][64];
static int	nhigh[2];

static int32_t *distinct[2];
static uint64_t ndistinct[2];
static uint64_t *starts[2];
static uint32_t *postings[2];
static AnnFilterSource source;

/* --- the scratch-failing allocator --- */
static size_t lastSize;
static uint64_t refused;		/* sort scratch requests refused */

/*
 * FromRuns allocates a list of n ordinals and then SortOrdinals asks for n
 * plus three digit tables: refuse that one.
 */
static void *
RefuseScratch(void *arg, size_t size)
{
	(void) arg;
	if (lastSize >= 64 * sizeof(uint32_t) && size == lastSize + 3 * 2048 * sizeof(uint32_t))
	{
		refused++;
		lastSize = 0;
		return NULL;
	}
	lastSize = size;
	return malloc(size);
}

static int
Column(void *arg, const char *name)
{
	(void) arg;
	return strcmp(name, "m") == 0 ? 0 : strcmp(name, "s") == 0 ? 1 : -1;
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

static int
Multi(void *arg, int c)
{
	(void) arg;
	return c == 0;
}

/*
 * Postings of a column from (value, row) pairs: values ascending, each
 * value's rows ascending, a row once per distinct value.
 */
typedef struct Pair
{
	int32_t		value;
	uint32_t	row;
} Pair;

static int
PairOrder(const void *left, const void *right)
{
	const Pair *x = left,
			   *y = right;

	if (x->value != y->value)
		return (x->value > y->value) - (x->value < y->value);
	return (x->row > y->row) - (x->row < y->row);
}

static void
AddPair(Pair *pairs, uint64_t *npairs, int32_t value, uint32_t row)
{
	pairs[*npairs].value = value;
	pairs[*npairs].row = row;
	(*npairs)++;
}

static void
Invert(int c, Pair *pairs, uint64_t n)
{
	uint64_t	kept = 0;

	qsort(pairs, n, sizeof(Pair), PairOrder);
	postings[c] = malloc(sizeof(uint32_t) * (n ? n : 1));
	for (uint64_t i = 0; i < n; i++)
	{
		if (i > 0 && pairs[i].value == pairs[i - 1].value && pairs[i].row == pairs[i - 1].row)
			continue;
		if (kept == 0 || pairs[i].value != distinct[c][ndistinct[c] - 1])
		{
			distinct[c][ndistinct[c]] = pairs[i].value;
			starts[c][ndistinct[c]++] = kept;
		}
		postings[c][kept++] = pairs[i].row;
	}
	starts[c][ndistinct[c]] = kept;
}

typedef struct
{
	unsigned char *b;
	size_t		n;
} Program;

static void
Put(Program *p, uint64_t v, int bytes)
{
	for (int i = 0; i < bytes; i++)
		p->b[p->n++] = (unsigned char) (v >> (8 * i));
}

/* A random element: skewed, so a few values have long posting runs. */
static int32_t
Element(void)
{
	int			x = rand() % domain;

	return (int32_t) ((int64_t) x * x / domain);
}

/* Whether the leaf's values hold v (values are never NULL). */
static int
Holds(const unsigned char *want, int32_t v)
{
	return v >= 0 && v < domain + OUTSIDE && want[v];
}

/*
 * Emit a random subtree and compute its rows by the definition. A negated
 * leaf matches the non-NULL rows the leaf does not match.
 */
static void
Emit(Program *p, int depth, unsigned char *rows)
{
	int			kind = depth <= 0 ? rand() % 2 : rand() % 5;

	if (kind < 2)
	{
		int			multi = kind == 0;
		int			negate = rand() % 4 == 0;
		uint32_t	n = 1 + (rand() % 3 == 0 ? rand() % 40 : rand() % 3);
		int32_t		values[64];
		unsigned char *want = calloc(domain + OUTSIDE, 1);

		for (uint32_t i = 0; i < n; i++)
		{
			if (nhigh[!multi] && rand() % 3 == 0)
				values[i] = highValues[!multi][rand() % nhigh[!multi]];
			else
				values[i] = rand() % 10 == 0 ? domain + rand() % OUTSIDE : multi ? Element() :
					rand() % singleDomain;
			want[values[i]] = 1;
		}
		Put(p, multi ? ANN_FILTER_OVERLAP : ANN_FILTER_VALUES, 1);
		Put(p, 1, 1);
		Put(p, multi ? 'm' : 's', 1);
		Put(p, negate, 1);
		Put(p, n, 4);
		for (uint32_t i = 0; i < n; i++)
			Put(p, (uint32_t) values[i], 4);
		for (uint32_t r = 0; r < nrows; r++)
		{
			int			isnull = multi ? nelements[r] < 0 : single[r] == ANN_FILTER_NULL_VALUE;
			int			match = 0;

			if (!isnull && !multi)
				match = Holds(want, single[r]);
			else if (!isnull)
				for (int k = 0; k < nelements[r] && !match; k++)
					match = Holds(want, elements[(size_t) r * MAX_ELEMENTS + k]);
			rows[r] = negate ? !isnull && !match : match;
		}
		free(want);
		return;
	}
	if (kind == 2)
	{
		Emit(p, depth - 1, rows);
		Put(p, ANN_FILTER_NOT, 1);
		for (uint32_t r = 0; r < nrows; r++)
			rows[r] = !rows[r];
		return;
	}
	{
		unsigned char *other = malloc(nrows);

		Emit(p, depth - 1, rows);
		Emit(p, depth - 1, other);
		Put(p, kind == 3 ? ANN_FILTER_AND : ANN_FILTER_OR, 1);
		for (uint32_t r = 0; r < nrows; r++)
			rows[r] = kind == 3 ? rows[r] && other[r] : rows[r] || other[r];
		free(other);
	}
}

/* Remember the values rows of ordinal 2^22 and above hold, once each. */
static void
NoteHigh(int c, int32_t value)
{
	for (int i = 0; i < nhigh[c]; i++)
		if (highValues[c][i] == value)
			return;
	if (nhigh[c] < 64)
		highValues[c][nhigh[c]++] = value;
}

/*
 * One run: rows, value domains, programs and the seed. Returns 0 when every
 * program's rows and count equal the definition, also with the sort's
 * scratch refused.
 */
static int
Run			(uint32_t rows_, int domain_, int singleDomain_, int nprograms, unsigned seed)
{
	Pair	   *pairs;
	uint64_t	npairs = 0;
	size_t		nwords;
	uint32_t   *words;
	uint32_t   *again;
	unsigned char *rows;
	int			programs = 0;
	uint64_t	refusedBefore = refused;

	nrows = rows_;
	domain = domain_;
	singleDomain = singleDomain_;
	nwords = (nrows + 31) / 32;
	pairs = malloc(sizeof(Pair) * nrows * MAX_ELEMENTS);
	words = malloc(nwords * sizeof(uint32_t));
	again = malloc(nwords * sizeof(uint32_t));
	rows = malloc(nrows);
	elements = malloc(sizeof(int32_t) * nrows * MAX_ELEMENTS);
	nelements = malloc(nrows);
	single = malloc(sizeof(int32_t) * nrows);
	for (int c = 0; c < 2; c++)
	{
		distinct[c] = malloc(sizeof(int32_t) * (domain + OUTSIDE + 2));
		starts[c] = malloc(sizeof(uint64_t) * (domain + OUTSIDE + 3));
		ndistinct[c] = 0;
		nhigh[c] = 0;
	}
	assert(pairs && words && again && rows && elements && nelements && single);

	srand(seed);
	for (uint32_t r = 0; r < nrows; r++)
	{
		nelements[r] = (signed char) (rand() % 12 == 0 ? -1 : rand() % (MAX_ELEMENTS + 1));
		if (nelements[r] < 0)
			AddPair(pairs, &npairs, ANN_FILTER_NULL_VALUE, r);
		for (int k = 0; k < nelements[r]; k++)
		{
			elements[(size_t) r * MAX_ELEMENTS + k] = Element();
			AddPair(pairs, &npairs, elements[(size_t) r * MAX_ELEMENTS + k], r);
		}
	}
	Invert(0, pairs, npairs);
	npairs = 0;
	for (uint32_t r = 0; r < nrows; r++)
	{
		single[r] = rand() % 10 == 0 ? ANN_FILTER_NULL_VALUE : rand() % singleDomain;
		AddPair(pairs, &npairs, single[r], r);
	}
	Invert(1, pairs, npairs);
	free(pairs);
	for (uint32_t r = HIGH_ORDINAL; r < nrows; r++)
	{
		for (int k = 0; k < nelements[r]; k++)
			NoteHigh(0, elements[(size_t) r * MAX_ELEMENTS + k]);
		if (single[r] != ANN_FILTER_NULL_VALUE)
			NoteHigh(1, single[r]);
	}

	memset(&source, 0, sizeof(source));
	source.rows = nrows;
	source.column = Column;
	source.count = Count;
	source.value = Value;
	source.start = Start;
	source.postings = Postings;
	source.multi = Multi;

	for (int t = 0; t < nprograms; t++)
	{
		Program		p = {.b = malloc(1 << 16),.n = 0};
		uint64_t	count = 0,
					countAgain = 0,
					expected = 0;
		int			code;

		Emit(&p, 1 + rand() % 4, rows);
		code = AnnFilterEvaluate(&source, p.b, p.n, words, &count);
		if (code != ANN_FILTER_OK)
		{
			fprintf(stderr, "%u rows, program %d: evaluation failed (%d)\n", nrows, t, code);
			return 1;
		}
		for (uint32_t r = 0; r < nrows; r++)
		{
			int			bit = (words[r / 32] >> (r % 32)) & 1;

			if (bit != rows[r])
			{
				fprintf(stderr, "%u rows, program %d: row %u is %d, expected %d\n", nrows, t, r, bit, rows[r]);
				return 1;
			}
			expected += rows[r];
		}
		if (count != expected)
		{
			fprintf(stderr, "%u rows, program %d: count %llu, expected %llu\n", nrows, t,
					(unsigned long long) count, (unsigned long long) expected);
			return 1;
		}

		/*
		 * Again with the sort's scratch refused: the same code, words and
		 * count.
		 */
		source.alloc = RefuseScratch;
		lastSize = 0;
		code = AnnFilterEvaluate(&source, p.b, p.n, again, &countAgain);
		source.alloc = NULL;
		if (code != ANN_FILTER_OK || countAgain != count || memcmp(again, words, nwords * sizeof(uint32_t)) != 0)
		{
			fprintf(stderr, "%u rows, program %d: with the sort scratch refused, code %d, count %llu\n", nrows,
					t, code, (unsigned long long) countAgain);
			return 1;
		}
		free(p.b);
		programs++;
	}
	if (refused == refusedBefore)
	{
		fprintf(stderr, "%u rows: no sort scratch was requested\n", nrows);
		return 1;
	}
	printf("filter evaluator: %d programs over %u rows equal to the row-by-row definition "
		   "(%llu sort scratch requests refused)\n", programs, nrows,
		   (unsigned long long) (refused - refusedBefore));
	free(words);
	free(again);
	free(rows);
	free(elements);
	free(nelements);
	free(single);
	for (int c = 0; c < 2; c++)
	{
		free(postings[c]);
		free(distinct[c]);
		free(starts[c]);
	}
	return 0;
}

int
main(void)
{
	/*
	 * Past 2^22 rows the third digit varies: values are spread wider there,
	 * so lists of several runs stay under rows / 64 and hold those rows.
	 */
	if (Run (100003, 500, 100, 400, 7) != 0 ||
		Run (4194319, 20000, 4000, 100, 11) != 0)
		return 1;
	return 0;
}
