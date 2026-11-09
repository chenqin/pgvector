#include "filter.h"

#include <stdlib.h>
#include <string.h>

/*
 * A set of ordinals: a sorted list while it is small, else a bitset. Lists
 * stay below rows / 64 entries, where merging lists costs less than words.
 */
typedef struct Set
{
	uint32_t   *list;
	uint64_t	n;
	uint32_t   *words;			/* bitset form when not NULL */
} Set;

typedef struct Evaluator
{
	const AnnFilterSource *source;
	uint64_t	rows;
	size_t		nwords;
	uint64_t	listMax;
	int			error;
} Evaluator;

static uint64_t
Read(const unsigned char *data, int bytes)
{
	uint64_t	value = 0;

	for (int i = bytes - 1; i >= 0; i--)
		value = (value << 8) | data[i];
	return value;
}

static void *
Alloc(Evaluator *e, size_t size)
{
	void	   *pointer;

	if (size == 0)
		size = 1;
	pointer = e->source->alloc ? e->source->alloc(e->source->arg, size) : malloc(size);
	if (pointer == NULL)
		e->error = ANN_FILTER_NO_MEMORY;
	return pointer;
}

static void
Release(Evaluator *e, void *pointer)
{
	if (pointer == NULL)
		return;
	if (e->source->release)
		e->source->release(e->source->arg, pointer);
	else
		free(pointer);
}

static void
FreeSet(Evaluator *e, Set *set)
{
	Release(e, set->list);
	Release(e, set->words);
	set->list = NULL;
	set->words = NULL;
	set->n = 0;
}

static void
MaskTail(Evaluator *e, uint32_t *words)
{
	if (e->rows % 32)
		words[e->nwords - 1] &= (UINT32_C(1) << (e->rows % 32)) - 1;
}

static uint32_t *
NewWords(Evaluator *e, int fill)
{
	uint32_t   *words = Alloc(e, e->nwords * sizeof(uint32_t));

	if (words)
	{
		memset(words, fill ? 0xFF : 0, e->nwords * sizeof(uint32_t));
		if (fill)
			MaskTail(e, words);
	}
	return words;
}

/* Convert to the bitset form in place. */
static int
ToWords(Evaluator *e, Set *set)
{
	uint32_t   *words;

	if (set->words)
		return 1;
	if ((words = NewWords(e, 0)) == NULL)
		return 0;
	for (uint64_t i = 0; i < set->n; i++)
		words[set->list[i] / 32] |= UINT32_C(1) << (set->list[i] % 32);
	Release(e, set->list);
	set->list = NULL;
	set->words = words;
	return 1;
}

static int
Complement(Evaluator *e, Set *set)
{
	if (!ToWords(e, set))
		return 0;
	for (size_t w = 0; w < e->nwords; w++)
		set->words[w] = ~set->words[w];
	MaskTail(e, set->words);
	return 1;
}

static int
OrdinalCompare(const void *a, const void *b)
{
	uint32_t	x = *(const uint32_t *) a;
	uint32_t	y = *(const uint32_t *) b;

	return (x > y) - (x < y);
}

static int
ValueCompare(const void *a, const void *b)
{
	int32_t		x = *(const int32_t *) a;
	int32_t		y = *(const int32_t *) b;

	return (x > y) - (x < y);
}

/* First distinct value >= target. */
static uint64_t
LowerBound(Evaluator *e, int column, uint64_t count, int64_t target)
{
	uint64_t	low = 0,
				high = count;

	while (low < high)
	{
		uint64_t	middle = low + (high - low) / 2;

		if ((int64_t) e->source->value(e->source->arg, column, middle) < target)
			low = middle + 1;
		else
			high = middle;
	}
	return low;
}

/* Postings runs [first, end) of a leaf; each value's run is ascending. */
typedef struct Run
{
	uint64_t	first;
	uint64_t	end;
	uint64_t	code;			/* the value's position; UINT64_MAX for
								 * several */
} Run;

/*
 * Union of postings runs: a sorted list when small, else bits. Runs of a
 * multi-valued column can share rows, so its lists are deduplicated.
 */
static int
FromRuns(Evaluator *e, int column, const Run *runs, uint32_t nruns, int sorted, int multi, Set *out)
{
	uint64_t	total = 0;

	memset(out, 0, sizeof(*out));
	for (uint32_t r = 0; r < nruns; r++)
		total += runs[r].end - runs[r].first;
	if (total <= e->listMax)
	{
		uint64_t	at = 0;

		if ((out->list = Alloc(e, total * sizeof(uint32_t))) == NULL)
			return 0;
		for (uint32_t r = 0; r < nruns; r++)
		{
			uint64_t	n = runs[r].end - runs[r].first;

			if (n && !e->source->postings(e->source->arg, column, runs[r].first, n, out->list + at))
			{
				e->error = ANN_FILTER_SOURCE_ERROR;
				return 0;
			}
			at += n;
		}
		out->n = total;
		if (nruns > 1 || !sorted)
		{
			qsort(out->list, total, sizeof(uint32_t), OrdinalCompare);
			if (multi && total > 1)
			{
				out->n = 1;
				for (uint64_t i = 1; i < total; i++)
					if (out->list[i] != out->list[out->n - 1])
						out->list[out->n++] = out->list[i];
			}
		}
		return 1;
	}
	if ((out->words = NewWords(e, 0)) == NULL)
		return 0;
	for (uint32_t r = 0; r < nruns; r++)
	{
		uint32_t	chunk[4096];

		/* A prebuilt bitset covers one whole value. */
		if (e->source->bitset && runs[r].code != UINT64_MAX && runs[r].end - runs[r].first > e->listMax)
		{
			uint32_t   *words = NewWords(e, 0);

			if (words == NULL)
				return 0;
			if (e->source->bitset(e->source->arg, column, runs[r].code, words))
			{
				for (size_t w = 0; w < e->nwords; w++)
					out->words[w] |= words[w];
				Release(e, words);
				continue;
			}
			Release(e, words);
		}
		for (uint64_t first = runs[r].first; first < runs[r].end; first += 4096)
		{
			uint64_t	n = runs[r].end - first < 4096 ? runs[r].end - first : 4096;

			if (!e->source->postings(e->source->arg, column, first, n, chunk))
			{
				e->error = ANN_FILTER_SOURCE_ERROR;
				return 0;
			}
			for (uint64_t i = 0; i < n; i++)
				if (chunk[i] < e->rows)
					out->words[chunk[i] / 32] |= UINT32_C(1) << (chunk[i] % 32);
		}
	}
	return 1;
}

/* The run of one distinct value, if present. */
static int
ValueRun(Evaluator *e, int column, uint64_t count, int32_t value, Run *run)
{
	uint64_t	i = LowerBound(e, column, count, value);

	if (i == count || e->source->value(e->source->arg, column, i) != value)
		return 0;
	run->code = i;
	run->first = e->source->start(e->source->arg, column, i);
	run->end = e->source->start(e->source->arg, column, i + 1);
	return 1;
}

/*
 * A leaf: VALUES, RANGE, OVERLAP or NULL on one column. Negated leaves
 * complement the union of the match and the NULL rows. OVERLAP is VALUES
 * over a multi-valued column's elements.
 */
static int
Leaf(Evaluator *e, int op, const unsigned char *at, Set *out)
{
	char		name[256];
	uint32_t	nameLength = at[1];
	int			negate;
	int			column;
	int			multi;
	uint64_t	count;
	Run		   *runs = NULL;
	uint32_t	nruns = 0;
	int			ok;

	memcpy(name, at + 2, nameLength);
	name[nameLength] = '\0';
	negate = at[2 + nameLength];
	at += 3 + nameLength;
	if ((column = e->source->column(e->source->arg, name)) < 0)
	{
		e->error = ANN_FILTER_UNKNOWN_COLUMN;
		return 0;
	}
	multi = e->source->multi && e->source->multi(e->source->arg, column);
	if (multi ? op == ANN_FILTER_VALUES || op == ANN_FILTER_RANGE : op == ANN_FILTER_OVERLAP)
	{
		e->error = ANN_FILTER_COLUMN_KIND;
		return 0;
	}
	count = e->source->count(e->source->arg, column);
	if (op == ANN_FILTER_VALUES || op == ANN_FILTER_OVERLAP)
	{
		uint32_t	n = (uint32_t) Read(at, 4);
		int32_t    *values = Alloc(e, (size_t) n * sizeof(int32_t));

		if (values == NULL)
			return 0;
		for (uint32_t i = 0; i < n; i++)
			values[i] = (int32_t) (uint32_t) Read(at + 4 + (size_t) i * 4, 4);
		qsort(values, n, sizeof(int32_t), ValueCompare);
		if ((runs = Alloc(e, (size_t) (n + 1) * sizeof(Run))) == NULL)
		{
			Release(e, values);
			return 0;
		}
		for (uint32_t i = 0; i < n; i++)
			if ((i == 0 || values[i] != values[i - 1]) && values[i] != ANN_FILTER_NULL_VALUE &&
				ValueRun(e, column, count, values[i], &runs[nruns]))
				nruns++;
		Release(e, values);
	}
	else if (op == ANN_FILTER_RANGE)
	{
		int64_t		low = (int32_t) (uint32_t) Read(at, 4);
		int64_t		high = (int32_t) (uint32_t) Read(at + 4, 4);

		if ((runs = Alloc(e, 2 * sizeof(Run))) == NULL)
			return 0;
		if (low <= ANN_FILTER_NULL_VALUE)
			low = (int64_t) ANN_FILTER_NULL_VALUE + 1;
		if (low <= high)
		{
			uint64_t	i0 = LowerBound(e, column, count, low);
			uint64_t	i1 = LowerBound(e, column, count, high + 1);

			if (i0 < i1)
			{
				runs[0].code = i1 - i0 == 1 ? i0 : UINT64_MAX;
				runs[0].first = e->source->start(e->source->arg, column, i0);
				runs[0].end = e->source->start(e->source->arg, column, i1);
				nruns = 1;
			}
		}
	}
	else if ((runs = Alloc(e, 2 * sizeof(Run))) == NULL)
		return 0;
	else if (ValueRun(e, column, count, ANN_FILTER_NULL_VALUE, &runs[0]))
		nruns = 1;

	if (negate && op != ANN_FILTER_NULL &&
		ValueRun(e, column, count, ANN_FILTER_NULL_VALUE, &runs[nruns]))
		nruns++;
	/* A range's run spans several values, so it is not in ordinal order. */
	ok = FromRuns(e, column, runs, nruns, op != ANN_FILTER_RANGE, multi, out) && (!negate || Complement(e, out));
	Release(e, runs);
	return ok;
}

static int
And(Evaluator *e, Set *a, Set *b, Set *out)
{
	memset(out, 0, sizeof(*out));
	if (a->words && b->words)
	{
		for (size_t w = 0; w < e->nwords; w++)
			a->words[w] &= b->words[w];
		*out = *a;
		memset(a, 0, sizeof(*a));
		return 1;
	}
	if (a->words)
	{
		Set		   *t = a;

		a = b;
		b = t;
	}
	/* a is a list */
	if ((out->list = Alloc(e, a->n * sizeof(uint32_t))) == NULL)
		return 0;
	if (b->words)
	{
		for (uint64_t i = 0; i < a->n; i++)
			if (b->words[a->list[i] / 32] & (UINT32_C(1) << (a->list[i] % 32)))
				out->list[out->n++] = a->list[i];
	}
	else
	{
		uint64_t	i = 0,
					j = 0;

		while (i < a->n && j < b->n)
		{
			if (a->list[i] < b->list[j])
				i++;
			else if (a->list[i] > b->list[j])
				j++;
			else
			{
				out->list[out->n++] = a->list[i];
				i++;
				j++;
			}
		}
	}
	return 1;
}

static int
Or(Evaluator *e, Set *a, Set *b, Set *out)
{
	memset(out, 0, sizeof(*out));
	if (!a->words && !b->words && a->n + b->n <= e->listMax)
	{
		uint64_t	i = 0,
					j = 0;

		if ((out->list = Alloc(e, (a->n + b->n) * sizeof(uint32_t))) == NULL)
			return 0;
		while (i < a->n || j < b->n)
		{
			uint32_t	next;

			if (j == b->n || (i < a->n && a->list[i] < b->list[j]))
				next = a->list[i++];
			else if (i == a->n || b->list[j] < a->list[i])
				next = b->list[j++];
			else
			{
				next = a->list[i++];
				j++;
			}
			out->list[out->n++] = next;
		}
		return 1;
	}
	if (!ToWords(e, a) || !ToWords(e, b))
		return 0;
	for (size_t w = 0; w < e->nwords; w++)
		a->words[w] |= b->words[w];
	*out = *a;
	memset(a, 0, sizeof(*a));
	return 1;
}

/* Size of the instruction at program[offset], or 0 if malformed. */
static size_t
StepSize(const unsigned char *program, size_t size, size_t offset)
{
	int			op = program[offset];
	size_t		nameLength;

	switch (op)
	{
		case ANN_FILTER_AND:
		case ANN_FILTER_OR:
		case ANN_FILTER_NOT:
		case ANN_FILTER_ALL:
		case ANN_FILTER_NONE:
			return 1;
		case ANN_FILTER_BITMAP:
			return size - offset >= 9 ? 9 : 0;
		case ANN_FILTER_VALUES:
		case ANN_FILTER_RANGE:
		case ANN_FILTER_NULL:
		case ANN_FILTER_OVERLAP:
			if (size - offset < 3)
				return 0;
			nameLength = program[offset + 1];
			if (nameLength == 0 || nameLength > 63 || size - offset < 3 + nameLength ||
				memchr(program + offset + 2, '\0', nameLength) || program[offset + 2 + nameLength] > 1)
				return 0;
			if (op == ANN_FILTER_NULL)
				return 3 + nameLength;
			if (op == ANN_FILTER_RANGE)
				return size - offset >= 11 + nameLength ? 11 + nameLength : 0;
			if (size - offset < 7 + nameLength)
				return 0;
			{
				uint64_t	n = Read(program + offset + 3 + nameLength, 4);

				if (n > ANN_FILTER_MAX_VALUES || size - offset < 7 + nameLength + n * 4)
					return 0;
				return 7 + nameLength + (size_t) n * 4;
			}
		default:
			return 0;
	}
}

int
AnnFilterCheck(const unsigned char *program, size_t size)
{
	size_t		offset = 0;
	int			depth = 0;
	int			steps = 0;

	if (size == 0 || size > ANN_FILTER_MAX_PROGRAM)
		return 0;
	while (offset < size)
	{
		size_t		step = StepSize(program, size, offset);
		int			op = program[offset];

		if (step == 0 || ++steps > ANN_FILTER_MAX_STEPS)
			return 0;
		if (op == ANN_FILTER_AND || op == ANN_FILTER_OR)
		{
			if (depth < 2)
				return 0;
			depth--;
		}
		else if (op == ANN_FILTER_NOT)
		{
			if (depth < 1)
				return 0;
		}
		else if (++depth > ANN_FILTER_MAX_DEPTH)
			return 0;
		offset += step;
	}
	return depth == 1;
}

int
AnnFilterEvaluate(const AnnFilterSource *source, const unsigned char *program, size_t size,
				  uint32_t *words, uint64_t *count)
{
	Evaluator	e;
	Set			stack[ANN_FILTER_MAX_DEPTH];
	int			depth = 0;
	size_t		offset = 0;
	int			ok = 1;

	if (!AnnFilterCheck(program, size))
		return ANN_FILTER_INVALID;
	memset(&e, 0, sizeof(e));
	e.source = source;
	e.rows = source->rows;
	e.nwords = (size_t) ((source->rows + 31) / 32);
	e.listMax = source->rows / 64 ? source->rows / 64 : 1;
	e.error = ANN_FILTER_OK;
	while (ok && offset < size)
	{
		int			op = program[offset];
		Set			result;

		memset(&result, 0, sizeof(result));
		switch (op)
		{
			case ANN_FILTER_VALUES:
			case ANN_FILTER_RANGE:
			case ANN_FILTER_NULL:
			case ANN_FILTER_OVERLAP:
				ok = Leaf(&e, op, program + offset, &result);
				break;
			case ANN_FILTER_BITMAP:
				if (source->bitmap == NULL)
				{
					e.error = ANN_FILTER_UNKNOWN_BITMAP;
					ok = 0;
				}
				else if ((result.words = NewWords(&e, 0)) == NULL)
					ok = 0;
				else if (!source->bitmap(source->arg, Read(program + offset + 1, 8), result.words))
				{
					e.error = ANN_FILTER_UNKNOWN_BITMAP;
					ok = 0;
				}
				else
					MaskTail(&e, result.words);
				break;
			case ANN_FILTER_ALL:
				ok = (result.words = NewWords(&e, 1)) != NULL;
				break;
			case ANN_FILTER_NONE:
				ok = (result.list = Alloc(&e, sizeof(uint32_t))) != NULL;
				break;
			case ANN_FILTER_NOT:
				result = stack[--depth];
				ok = Complement(&e, &result);
				break;
			default:
				{
					Set			b = stack[--depth];
					Set			a = stack[--depth];

					ok = op == ANN_FILTER_AND ? And(&e, &a, &b, &result) : Or(&e, &a, &b, &result);
					FreeSet(&e, &a);
					FreeSet(&e, &b);
				}
				break;
		}
		if (!ok)
		{
			FreeSet(&e, &result);
			break;
		}
		stack[depth++] = result;
		offset += StepSize(program, size, offset);
	}
	if (ok)
	{
		Set		   *final = &stack[0];

		memset(words, 0, e.nwords * sizeof(uint32_t));
		*count = 0;
		if (final->words)
			memcpy(words, final->words, e.nwords * sizeof(uint32_t));
		else
			for (uint64_t i = 0; i < final->n; i++)
				words[final->list[i] / 32] |= UINT32_C(1) << (final->list[i] % 32);
		for (size_t w = 0; w < e.nwords; w++)
			*count += (uint64_t) __builtin_popcount(words[w]);
	}
	while (depth > 0)
		FreeSet(&e, &stack[--depth]);
	return ok ? ANN_FILTER_OK : (e.error ? e.error : ANN_FILTER_SOURCE_ERROR);
}

/* FNV-1a, seeded apart from PostgreSQL's filter-text ids. */
uint64_t
AnnFilterHash(const unsigned char *program, size_t size)
{
	uint64_t	hash = UINT64_C(0xcbf29ce484222325) ^ UINT64_C(0x70726f6772616d31);

	for (size_t i = 0; i < size; i++)
	{
		hash ^= program[i];
		hash *= UINT64_C(0x100000001b3);
	}
	return hash;
}
