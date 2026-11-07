/*
 * Canonical predicate programs (ann/canonical.c) must select exactly the rows
 * the original selects, be fixed points, and coincide for programs written
 * differently: operands swapped, value lists reordered or repeated, ranges
 * against the value lists they hold, values the column lacks. Columns a and
 * b have NULLs and gaps; c has no NULLs and only the values 0..4.
 */
#include "canonical.h"
#include "filter.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROWS 5003
#define COLUMNS 3

static int32_t data[COLUMNS][ROWS];
static int32_t distinct[COLUMNS][ROWS];
static uint64_t ndistinct[COLUMNS];
static uint64_t starts[COLUMNS][ROWS + 1];
static uint32_t postings[COLUMNS][ROWS];
static const char *names[COLUMNS] = {"a", "b", "c"};
static int	sortColumn;
static AnnFilterSource source;
static uint32_t words[(ROWS + 31) / 32];
static uint32_t other[(ROWS + 31) / 32];

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

typedef struct
{
	unsigned char b[16384];
	size_t		n;
} Program;

static void
Put(Program *p, uint64_t v, int bytes)
{
	for (int i = 0; i < bytes; i++)
		p->b[p->n++] = (unsigned char) (v >> (8 * i));
}

static int32_t
RandomValue(int column)
{
	if (column == 2)
		return rand() % 5;
	return rand() % 8 == 0 ? ANN_FILTER_NULL_VALUE : rand() % 40 - 20;
}

/* A random expression tree, emitted possibly with operands reordered. */
typedef struct Tree
{
	int			kind;			/* 0 leaf, 1 ALL, 2 NONE, 3 NOT, 4 AND, 5 OR */
	int			op;
	int			column;
	int			negate;
	int32_t		low,
				high;
	int32_t		values[8];
	uint32_t	nvalues;
	struct Tree *a,
			   *b;
} Tree;

static Tree *
Build(int depth)
{
	Tree	   *t = calloc(1, sizeof(Tree));
	int			kind = depth == 0 ? rand() % 5 : rand() % 9;

	if (kind < 3)
	{
		t->kind = 0;
		t->op = kind == 0 ? ANN_FILTER_VALUES : kind == 1 ? ANN_FILTER_RANGE : ANN_FILTER_NULL;
		t->column = rand() % COLUMNS;
		t->negate = rand() % 3 == 0;
		t->low = rand() % 10 == 0 ? INT32_MIN : rand() % 50 - 25;
		t->high = rand() % 10 == 0 ? INT32_MAX : t->low == INT32_MIN ? 0 : t->low + rand() % 20 - 3;
		t->nvalues = rand() % 6;
		for (uint32_t i = 0; i < t->nvalues; i++)
			t->values[i] = rand() % 4 == 0 ? 99 + rand() % 3 : RandomValue(t->column);
	}
	else if (kind < 5)
		t->kind = kind - 2;
	else if (kind == 5)
	{
		t->kind = 3;
		t->a = Build(depth - 1);
	}
	else
	{
		t->kind = kind % 2 ? 4 : 5;
		t->a = Build(depth - 1);
		t->b = Build(depth - 1);
	}
	return t;
}

static void
Emit(const Tree *t, Program *p, int shuffle)
{
	switch (t->kind)
	{
		case 0:
			Put(p, t->op, 1);
			Put(p, 1, 1);
			Put(p, (unsigned char) names[t->column][0], 1);
			Put(p, t->negate, 1);
			if (t->op == ANN_FILTER_RANGE)
			{
				Put(p, (uint32_t) t->low, 4);
				Put(p, (uint32_t) t->high, 4);
			}
			else if (t->op == ANN_FILTER_VALUES)
			{
				int32_t		values[16];
				uint32_t	n = t->nvalues;

				memcpy(values, t->values, n * sizeof(int32_t));
				if (shuffle && n)
				{
					values[n] = values[rand() % n]; /* a repeat */
					n++;
					for (uint32_t i = n - 1; i > 0; i--)
					{
						uint32_t	j = (uint32_t) rand() % (i + 1);
						int32_t		x = values[i];

						values[i] = values[j];
						values[j] = x;
					}
				}
				Put(p, n, 4);
				for (uint32_t i = 0; i < n; i++)
					Put(p, (uint32_t) values[i], 4);
			}
			break;
		case 1:
			Put(p, ANN_FILTER_ALL, 1);
			break;
		case 2:
			Put(p, ANN_FILTER_NONE, 1);
			break;
		case 3:
			Emit(t->a, p, shuffle);
			Put(p, ANN_FILTER_NOT, 1);
			break;
		default:
			if (shuffle && rand() % 2)
			{
				Emit(t->b, p, shuffle);
				Emit(t->a, p, shuffle);
			}
			else
			{
				Emit(t->a, p, shuffle);
				Emit(t->b, p, shuffle);
			}
			Put(p, t->kind == 4 ? ANN_FILTER_AND : ANN_FILTER_OR, 1);
			break;
	}
}

static void
FreeTree(Tree *t)
{
	if (t == NULL)
		return;
	FreeTree(t->a);
	FreeTree(t->b);
	free(t);
}

/* The canonical form; the original's rows must be the canonical form's. */
static unsigned char *
Canonical(const Program *p, size_t *size)
{
	unsigned char *out;
	uint64_t	count,
				canonicalCount;

	assert(AnnFilterCanonical(&source, p->b, p->n, &out, size));
	assert(AnnFilterCheck(out, *size));
	assert(AnnFilterEvaluate(&source, p->b, p->n, words, &count) == ANN_FILTER_OK);
	assert(AnnFilterEvaluate(&source, out, *size, other, &canonicalCount) == ANN_FILTER_OK);
	assert(count == canonicalCount);
	assert(memcmp(words, other, sizeof(words)) == 0);
	return out;
}

static int
Same(const Program *x, const Program *y)
{
	size_t		nx,
				ny;
	unsigned char *cx = Canonical(x, &nx);
	unsigned char *cy = Canonical(y, &ny);
	int			same = nx == ny && memcmp(cx, cy, nx) == 0;

	free(cx);
	free(cy);
	return same;
}

static void
Leaf(Program *p, int op, char column, int negate)
{
	Put(p, op, 1);
	Put(p, 1, 1);
	Put(p, (unsigned char) column, 1);
	Put(p, negate, 1);
}

static void
RangeLeaf(Program *p, char column, int negate, int32_t low, int32_t high)
{
	Leaf(p, ANN_FILTER_RANGE, column, negate);
	Put(p, (uint32_t) low, 4);
	Put(p, (uint32_t) high, 4);
}

static void
ValuesLeaf(Program *p, char column, int negate, const int32_t *values, uint32_t n)
{
	Leaf(p, ANN_FILTER_VALUES, column, negate);
	Put(p, n, 4);
	for (uint32_t i = 0; i < n; i++)
		Put(p, (uint32_t) values[i], 4);
}

int
main(void)
{
	int			programs = 0,
				equal = 0;

	srand(11);
	for (int c = 0; c < COLUMNS; c++)
	{
		uint32_t	order[ROWS];

		for (uint32_t r = 0; r < ROWS; r++)
		{
			data[c][r] = RandomValue(c);
			order[r] = r;
		}
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
	source.rows = ROWS;
	source.column = Column;
	source.count = Count;
	source.value = Value;
	source.start = Start;
	source.postings = Postings;

	for (int t = 0; t < 5000; t++)
	{
		Tree	   *tree = Build(1 + rand() % 5);
		Program		p = {.n = 0},
					q = {.n = 0},
					again = {.n = 0};
		unsigned char *out;
		size_t		size;

		Emit(tree, &p, 0);
		Emit(tree, &q, 1);
		/* Same rows, a fixed point, and one form for both spellings. */
		out = Canonical(&p, &size);
		memcpy(again.b, out, size);
		again.n = size;
		free(out);
		out = Canonical(&again, &size);
		assert(size == again.n && memcmp(out, again.b, size) == 0);
		free(out);
		if (!Same(&p, &q))
		{
			fprintf(stderr, "program %d: reordered spelling has another canonical form\n", t);
			return 1;
		}
		FreeTree(tree);
		programs++;
	}

	{
		Program		x = {.n = 0},
					y = {.n = 0};
		int32_t		all[] = {4, 0, 3, 1, 2};
		int32_t		some[] = {3, 1, 2};
		int32_t		absent[] = {999};

		/* A range and the value list it holds. */
		RangeLeaf(&x, 'c', 0, 1, 3);
		ValuesLeaf(&y, 'c', 0, some, 3);
		assert(Same(&x, &y));
		/* Every value of a column without NULLs is every row. */
		x.n = y.n = 0;
		ValuesLeaf(&x, 'c', 0, all, 5);
		Put(&y, ANN_FILTER_ALL, 1);
		assert(Same(&x, &y));

		/*
		 * A value the column lacks matches nothing; its negation, every
		 * non-NULL row.
		 */
		x.n = y.n = 0;
		ValuesLeaf(&x, 'a', 0, absent, 1);
		Put(&y, ANN_FILTER_NONE, 1);
		assert(Same(&x, &y));
		x.n = y.n = 0;
		ValuesLeaf(&x, 'a', 1, absent, 1);
		Leaf(&y, ANN_FILTER_NULL, 'a', 1);
		assert(Same(&x, &y));
		/* The fresh10 shape: a range AND "c <> a value c lacks" is the range. */
		x.n = y.n = 0;
		RangeLeaf(&x, 'b', 0, -5, 4);
		ValuesLeaf(&x, 'c', 1, absent, 1);
		Put(&x, ANN_FILTER_AND, 1);
		RangeLeaf(&y, 'b', 0, -5, 4);
		assert(Same(&x, &y));
		/* Two ranges on one column ANDed are their intersection. */
		x.n = y.n = 0;
		RangeLeaf(&x, 'b', 0, -10, 4);
		RangeLeaf(&x, 'b', 0, -5, 30);
		Put(&x, ANN_FILTER_AND, 1);
		RangeLeaf(&y, 'b', 0, -5, 4);
		assert(Same(&x, &y));
		/* NOT NOT and x AND x. */
		x.n = y.n = 0;
		RangeLeaf(&x, 'a', 0, 0, 7);
		ValuesLeaf(&x, 'b', 0, some, 3);
		Put(&x, ANN_FILTER_OR, 1);
		Put(&x, ANN_FILTER_NOT, 1);
		Put(&x, ANN_FILTER_NOT, 1);
		ValuesLeaf(&y, 'b', 0, some, 3);
		RangeLeaf(&y, 'a', 0, 0, 7);
		Put(&y, ANN_FILTER_OR, 1);
		ValuesLeaf(&y, 'b', 0, some, 3);
		RangeLeaf(&y, 'a', 0, 0, 7);
		Put(&y, ANN_FILTER_OR, 1);
		Put(&y, ANN_FILTER_AND, 1);
		assert(Same(&x, &y));
		equal = 7;
		/* Stored filters and unknown columns are left as written. */
		x.n = 0;
		Put(&x, ANN_FILTER_BITMAP, 1);
		Put(&x, 42, 8);
		{
			unsigned char *out;
			size_t		size;

			assert(!AnnFilterCanonical(&source, x.b, x.n, &out, &size));
			x.n = 0;
			Leaf(&x, ANN_FILTER_NULL, 'z', 0);
			assert(!AnnFilterCanonical(&source, x.b, x.n, &out, &size));
		}
	}
	/*
	 * Splits: the base's rows minus every exclusion's rows are the program's
	 * rows, for random programs and for the shape "range AND c <> v".
	 */
	{
		int			splits = 0;
		static uint32_t base[(ROWS + 31) / 32];

		srand(23);
		for (int t = 0; t < 20000; t++)
		{
			Tree	   *tree = Build(1 + rand() % 4);
			Program		p = {.n = 0};
			unsigned char *out;
			size_t		size;
			AnnFilterExclusion exclusions[ANN_FILTER_MAX_EXCLUSIONS];
			uint32_t	nexclusions;
			uint64_t	count,
						baseCount;

			Emit(tree, &p, 0);
			if (rand() % 2)
			{
				/* Often AND in an exclusion so that splits happen. */
				int32_t		v = rand() % 5;

				ValuesLeaf(&p, 'c', 1, &v, 1);
				Put(&p, ANN_FILTER_AND, 1);
			}
			FreeTree(tree);
			if (!AnnFilterCanonicalSplit(&source, p.b, p.n, 2500, &out, &size, exclusions, &nexclusions))
				continue;
			assert(nexclusions > 0 && nexclusions <= ANN_FILTER_MAX_EXCLUSIONS);
			assert(AnnFilterCheck(out, size));
			assert(AnnFilterEvaluate(&source, p.b, p.n, words, &count) == ANN_FILTER_OK);
			assert(AnnFilterEvaluate(&source, out, size, base, &baseCount) == ANN_FILTER_OK);
			free(out);
			for (uint32_t e = 0; e < nexclusions; e++)
				for (uint32_t r = 0; r < exclusions[e].nruns; r++)
					for (uint64_t at = starts[exclusions[e].column][exclusions[e].lo[r]];
						 at < starts[exclusions[e].column][exclusions[e].hi[r]]; at++)
					{
						uint32_t	row = postings[exclusions[e].column][at];

						base[row / 32] &= ~(UINT32_C(1) << (row % 32));
					}
			if (memcmp(words, base, sizeof(base)) != 0)
			{
				fprintf(stderr, "split %d: base minus exclusions differs from the program\n", t);
				return 1;
			}
			splits++;
		}
		assert(splits > 1000);
		printf("canonical: %d splits keep their rows\n", splits);
	}
	{
		Program		x = {.n = 0};
		int32_t		v = 3;
		unsigned char *out;
		size_t		size;
		AnnFilterExclusion exclusions[ANN_FILTER_MAX_EXCLUSIONS];
		uint32_t	nexclusions;

		/* b in -5..4 AND c <> 3: base is the range, one exclusion on c. */
		RangeLeaf(&x, 'b', 0, -5, 4);
		ValuesLeaf(&x, 'c', 1, &v, 1);
		Put(&x, ANN_FILTER_AND, 1);
		assert(AnnFilterCanonicalSplit(&source, x.b, x.n, 2500, &out, &size, exclusions, &nexclusions));
		assert(nexclusions == 1 && exclusions[0].column == 2 && exclusions[0].nruns == 1);
		assert(distinct[2][exclusions[0].lo[0]] == 3 && exclusions[0].hi[0] == exclusions[0].lo[0] + 1);
		free(out);
		/* Past the row bound, or with nothing left for the base: no split. */
		assert(!AnnFilterCanonicalSplit(&source, x.b, x.n, 10, &out, &size, exclusions, &nexclusions));
		x.n = 0;
		ValuesLeaf(&x, 'c', 1, &v, 1);
		assert(!AnnFilterCanonicalSplit(&source, x.b, x.n, 2500, &out, &size, exclusions, &nexclusions));
	}
	printf("canonical: %d programs keep their rows and one form per spelling; %d equivalences\n",
		   programs, equal);
	return 0;
}
