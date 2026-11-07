#include "canonical.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * See canonical.h. A program parses into a tree whose single-valued leaves
 * are sets of distinct-value indices, kept as sorted, disjoint, non-adjacent
 * half-open runs over 0 .. count - 1 (index 0 is NULL when the column has
 * NULL rows: INT32_MIN sorts first). Every node simplifies as it is built;
 * the result is serialized with operands in byte order.
 */

enum
{
	K_ALL,
	K_NONE,
	K_SET,						/* single-valued leaf: runs over value indices */
	K_LEAF,						/* multi-valued leaf, kept verbatim */
	K_NOT,
	K_AND,
	K_OR
};

typedef struct Run
{
	uint64_t	lo;
	uint64_t	hi;
} Run;

typedef struct Node
{
	int			kind;
	int			column;
	const unsigned char *name;	/* K_SET: the column name as written */
	uint8_t		nameLength;
	Run		   *runs;
	uint32_t	nruns;
	const unsigned char *leaf;	/* K_LEAF */
	size_t		leafSize;
	struct Node **children;
	uint32_t	nchildren;
	unsigned char *bytes;		/* serialized, once */
	size_t		nbytes;
} Node;

typedef struct Canon
{
	const AnnFilterSource *source;
	void	  **owned;
	size_t		nowned;
	size_t		capacity;
	int			failed;
} Canon;

static void *
Take(Canon *c, size_t size)
{
	void	   *pointer;

	if (c->failed)
		return NULL;
	if (c->nowned == c->capacity)
	{
		size_t		capacity = c->capacity ? c->capacity * 2 : 64;
		void	  **owned = realloc(c->owned, capacity * sizeof(void *));

		if (owned == NULL)
		{
			c->failed = 1;
			return NULL;
		}
		c->owned = owned;
		c->capacity = capacity;
	}
	if ((pointer = calloc(1, size ? size : 1)) == NULL)
	{
		c->failed = 1;
		return NULL;
	}
	c->owned[c->nowned++] = pointer;
	return pointer;
}

static uint32_t
Read32(const unsigned char *at)
{
	return (uint32_t) at[0] | (uint32_t) at[1] << 8 | (uint32_t) at[2] << 16 | (uint32_t) at[3] << 24;
}

static void
Write32(unsigned char *at, uint32_t value)
{
	for (int i = 0; i < 4; i++)
		at[i] = (unsigned char) (value >> (8 * i));
}

static uint64_t
Count(Canon *c, int column)
{
	return c->source->count(c->source->arg, column);
}

static int32_t
Value(Canon *c, int column, uint64_t i)
{
	return c->source->value(c->source->arg, column, i);
}

/* The first value index whose value is at least target. */
static uint64_t
Lower(Canon *c, int column, int64_t target)
{
	uint64_t	lo = 0,
				hi = Count(c, column);

	while (lo < hi)
	{
		uint64_t	mid = lo + (hi - lo) / 2;

		if ((int64_t) Value(c, column, mid) < target)
			lo = mid + 1;
		else
			hi = mid;
	}
	return lo;
}

static uint64_t
NullRows(Canon *c, int column)
{
	return Count(c, column) > 0 && Value(c, column, 0) == ANN_FILTER_NULL_VALUE ? 1 : 0;
}

static void
SetRun(Run *run, uint64_t lo, uint64_t hi)
{
	run->lo = lo;
	run->hi = hi;
}

static int
RunCompare(const void *left, const void *right)
{
	const Run  *a = left;
	const Run  *b = right;

	return (a->lo > b->lo) - (a->lo < b->lo);
}

/* Sort and merge runs in place; returns the new count. */
static uint32_t
Normalize(Run *runs, uint32_t n)
{
	uint32_t	kept = 0;

	qsort(runs, n, sizeof(Run), RunCompare);
	for (uint32_t i = 0; i < n; i++)
	{
		if (runs[i].lo >= runs[i].hi)
			continue;
		if (kept && runs[i].lo <= runs[kept - 1].hi)
		{
			if (runs[i].hi > runs[kept - 1].hi)
				runs[kept - 1].hi = runs[i].hi;
		}
		else
			runs[kept++] = runs[i];
	}
	return kept;
}

/* The runs of [0, count) outside a's. */
static Run *
Complement(Canon *c, const Run *a, uint32_t n, uint64_t count, uint32_t *out)
{
	Run		   *runs = Take(c, sizeof(Run) * (n + 1));
	uint64_t	at = 0;

	*out = 0;
	if (runs == NULL)
		return NULL;
	for (uint32_t i = 0; i < n; i++)
	{
		if (a[i].lo > at)
			SetRun(&runs[(*out)++], at, a[i].lo);
		at = a[i].hi;
	}
	if (at < count)
		SetRun(&runs[(*out)++], at, count);
	return runs;
}

static Run *
Union(Canon *c, const Run *a, uint32_t na, const Run *b, uint32_t nb, uint32_t *out)
{
	Run		   *runs = Take(c, sizeof(Run) * (na + nb));

	*out = 0;
	if (runs == NULL)
		return NULL;
	memcpy(runs, a, sizeof(Run) * na);
	memcpy(runs + na, b, sizeof(Run) * nb);
	*out = Normalize(runs, na + nb);
	return runs;
}

static Run *
Intersect(Canon *c, const Run *a, uint32_t na, const Run *b, uint32_t nb, uint32_t *out)
{
	Run		   *runs = Take(c, sizeof(Run) * (na + nb));
	uint32_t	i = 0,
				j = 0;

	*out = 0;
	if (runs == NULL)
		return NULL;
	while (i < na && j < nb)
	{
		uint64_t	lo = a[i].lo > b[j].lo ? a[i].lo : b[j].lo;
		uint64_t	hi = a[i].hi < b[j].hi ? a[i].hi : b[j].hi;

		if (lo < hi)
			SetRun(&runs[(*out)++], lo, hi);
		if (a[i].hi < b[j].hi)
			i++;
		else
			j++;
	}
	return runs;
}

static Node *
NewNode(Canon *c, int kind)
{
	Node	   *node = Take(c, sizeof(Node));

	if (node)
		node->kind = kind;
	return node;
}

/* A set node, or ALL / NONE when the set is full or empty. */
static Node *
SetNode(Canon *c, int column, const unsigned char *name, uint8_t nameLength, Run *runs, uint32_t n)
{
	uint64_t	count = Count(c, column);
	Node	   *node;

	if (c->failed)
		return NULL;
	if (n == 0)
		return NewNode(c, K_NONE);
	if (n == 1 && runs[0].lo == 0 && runs[0].hi == count)
		return NewNode(c, K_ALL);
	if ((node = NewNode(c, K_SET)) == NULL)
		return NULL;
	node->column = column;
	node->name = name;
	node->nameLength = nameLength;
	node->runs = runs;
	node->nruns = n;
	return node;
}

/* A leaf step at program[offset] (size bytes long), as a node. */
static Node *
Leaf(Canon *c, const unsigned char *at, size_t size)
{
	int			op = at[0];
	uint8_t		nameLength = at[1];
	int			negate = at[2 + nameLength];
	const unsigned char *payload = at + 3 + nameLength;
	char		name[256];
	int			column;
	int			multi;
	uint64_t	count;
	uint64_t	nulls;
	Run		   *runs;
	uint32_t	n = 0;

	memcpy(name, at + 2, nameLength);
	name[nameLength] = '\0';
	if ((column = c->source->column(c->source->arg, name)) < 0)
	{
		c->failed = 1;			/* the evaluator reports it */
		return NULL;
	}
	multi = c->source->multi && c->source->multi(c->source->arg, column);
	if (multi ? op == ANN_FILTER_VALUES || op == ANN_FILTER_RANGE : op == ANN_FILTER_OVERLAP)
	{
		c->failed = 1;
		return NULL;
	}
	if (multi)
	{
		Node	   *node = NewNode(c, K_LEAF);

		if (node)
		{
			node->leaf = at;
			node->leafSize = size;
		}
		return node;
	}
	count = Count(c, column);
	nulls = NullRows(c, column);
	if (op == ANN_FILTER_VALUES)
	{
		uint32_t	values = Read32(payload);

		if ((runs = Take(c, sizeof(Run) * (values + 2))) == NULL)
			return NULL;
		for (uint32_t i = 0; i < values; i++)
		{
			int32_t		v = (int32_t) Read32(payload + 4 + (size_t) i * 4);
			uint64_t	index;

			if (v == ANN_FILTER_NULL_VALUE)
				continue;
			index = Lower(c, column, v);
			if (index < count && Value(c, column, index) == v)
				SetRun(&runs[n++], index, index + 1);
		}
		n = Normalize(runs, n);
	}
	else if (op == ANN_FILTER_RANGE)
	{
		int64_t		low = (int32_t) Read32(payload);
		int64_t		high = (int32_t) Read32(payload + 4);

		if ((runs = Take(c, sizeof(Run) * 2)) == NULL)
			return NULL;
		if (low <= ANN_FILTER_NULL_VALUE)
			low = (int64_t) ANN_FILTER_NULL_VALUE + 1;
		if (low <= high)
		{
			uint64_t	i0 = Lower(c, column, low);
			uint64_t	i1 = Lower(c, column, high + 1);

			if (i0 < i1)
				SetRun(&runs[n++], i0, i1);
		}
	}
	else
	{
		if ((runs = Take(c, sizeof(Run) * 2)) == NULL)
			return NULL;
		if (nulls)
			SetRun(&runs[n++], 0, 1);
	}
	if (negate)
	{
		/* Rows neither matching nor NULL: complement (set and NULL). */
		Run		   *with;
		uint32_t	nwith;

		if (nulls && op != ANN_FILTER_NULL)
		{
			Run			null = {0, 1};

			with = Union(c, runs, n, &null, 1, &nwith);
		}
		else
		{
			with = runs;
			nwith = n;
		}
		if (with == NULL)
			return NULL;
		runs = Complement(c, with, nwith, count, &n);
	}
	return SetNode(c, column, at + 2, nameLength, runs, n);
}

static Node *
Not(Canon *c, Node *a)
{
	Node	   *node;

	if (a == NULL || c->failed)
		return NULL;
	switch (a->kind)
	{
		case K_ALL:
			return NewNode(c, K_NONE);
		case K_NONE:
			return NewNode(c, K_ALL);
		case K_NOT:
			return a->children[0];
		case K_SET:
			{
				uint32_t	n;
				Run		   *runs = Complement(c, a->runs, a->nruns, Count(c, a->column), &n);

				return runs ? SetNode(c, a->column, a->name, a->nameLength, runs, n) : NULL;
			}
		default:
			if ((node = NewNode(c, K_NOT)) == NULL || (node->children = Take(c, sizeof(Node *))) == NULL)
				return NULL;
			node->children[0] = a;
			node->nchildren = 1;
			return node;
	}
}

static int	Serialize(Canon *c, Node *node);

static int
BytesCompare(const void *left, const void *right)
{
	const Node *a = *(Node *const *) left;
	const Node *b = *(Node *const *) right;
	size_t		n = a->nbytes < b->nbytes ? a->nbytes : b->nbytes;
	int			order = memcmp(a->bytes, b->bytes, n);

	if (order)
		return order;
	return (a->nbytes > b->nbytes) - (a->nbytes < b->nbytes);
}

/* AND or OR of a and b: flattened, folded, same-column sets merged, sorted. */
static Node *
Combine(Canon *c, int op, Node *a, Node *b)
{
	int			absorbing = op == K_AND ? K_NONE : K_ALL;
	int			identity = op == K_AND ? K_ALL : K_NONE;
	Node	   *inputs[2] = {a, b};
	Node	  **list;
	uint32_t	total = 0,
				n = 0,
				kept = 0;
	Node	   *node;

	if (a == NULL || b == NULL || c->failed)
		return NULL;
	for (int i = 0; i < 2; i++)
		total += inputs[i]->kind == op ? inputs[i]->nchildren : 1;
	if ((list = Take(c, sizeof(Node *) * total)) == NULL)
		return NULL;
	for (int i = 0; i < 2; i++)
	{
		uint32_t	count = inputs[i]->kind == op ? inputs[i]->nchildren : 1;
		Node	  **from = inputs[i]->kind == op ? inputs[i]->children : &inputs[i];

		for (uint32_t j = 0; j < count; j++)
		{
			Node	   *x = from[j];
			uint32_t	same = n;

			if (x->kind == absorbing)
				return NewNode(c, absorbing);
			if (x->kind == identity)
				continue;
			if (x->kind == K_SET)
				for (uint32_t k = 0; k < n && same == n; k++)
					if (list[k]->kind == K_SET && list[k]->column == x->column)
						same = k;
			if (same < n)
			{
				uint32_t	merged;
				Run		   *runs = op == K_AND ?
					Intersect(c, list[same]->runs, list[same]->nruns, x->runs, x->nruns, &merged) :
					Union(c, list[same]->runs, list[same]->nruns, x->runs, x->nruns, &merged);
				Node	   *set = runs ? SetNode(c, x->column, x->name, x->nameLength, runs, merged) : NULL;

				if (set == NULL)
					return NULL;
				if (set->kind == absorbing)
					return set;
				if (set->kind == identity)
					list[same] = list[--n];
				else
					list[same] = set;
			}
			else
				list[n++] = x;
		}
	}
	for (uint32_t i = 0; i < n; i++)
		if (!Serialize(c, list[i]))
			return NULL;
	qsort(list, n, sizeof(Node *), BytesCompare);
	for (uint32_t i = 0; i < n; i++)
		if (kept == 0 || BytesCompare(&list[kept - 1], &list[i]) != 0)
			list[kept++] = list[i];
	if (kept == 0)
		return NewNode(c, identity);
	if (kept == 1)
		return list[0];
	if ((node = NewNode(c, op)) == NULL)
		return NULL;
	node->children = list;
	node->nchildren = kept;
	return node;
}

typedef struct ByteBuffer
{
	unsigned char *data;
	size_t		size;
	size_t		capacity;
	int			failed;
} ByteBuffer;

static void
AddBytes(ByteBuffer *b, const void *data, size_t size)
{
	if (b->failed)
		return;
	if (b->size + size > b->capacity)
	{
		size_t		capacity = b->capacity ? b->capacity : 64;
		unsigned char *grown;

		while (capacity < b->size + size)
			capacity *= 2;
		if (capacity > (size_t) ANN_FILTER_MAX_PROGRAM * 2 || (grown = realloc(b->data, capacity)) == NULL)
		{
			b->failed = 1;
			return;
		}
		b->data = grown;
		b->capacity = capacity;
	}
	memcpy(b->data + b->size, data, size);
	b->size += size;
}

/* A part of a set's leaves: one serialized step. */
typedef struct Part
{
	unsigned char *bytes;
	size_t		nbytes;
} Part;

static int
PartCompare(const void *left, const void *right)
{
	const Part *a = left;
	const Part *b = right;
	size_t		n = a->nbytes < b->nbytes ? a->nbytes : b->nbytes;
	int			order = memcmp(a->bytes, b->bytes, n);

	if (order)
		return order;
	return (a->nbytes > b->nbytes) - (a->nbytes < b->nbytes);
}

/*
 * The rows of a set of value indices as leaves ORed together: NULL, one
 * VALUES leaf of the isolated values, a RANGE leaf per longer run.
 */
static int
Positive(Canon *c, const Node *set, const Run *runs, uint32_t n, ByteBuffer *out)
{
	Part	   *parts = Take(c, sizeof(Part) * (n + 2));
	uint32_t	nparts = 0;
	uint32_t	singles = 0;
	unsigned char head[2 + 64 + 1];
	size_t		headSize = 2 + (size_t) set->nameLength + 1;
	ByteBuffer	values = {0};

	if (parts == NULL)
		return 0;
	head[1] = set->nameLength;
	memcpy(head + 2, set->name, set->nameLength);
	head[2 + set->nameLength] = 0;
	for (uint32_t i = 0; i < n; i++)
	{
		uint64_t	lo = runs[i].lo;

		if (lo == 0 && NullRows(c, set->column))
		{
			Part	   *part = &parts[nparts++];

			head[0] = ANN_FILTER_NULL;
			if ((part->bytes = Take(c, headSize)) == NULL)
				return 0;
			memcpy(part->bytes, head, headSize);
			part->nbytes = headSize;
			lo = 1;
		}
		if (runs[i].hi - lo == 1)
		{
			unsigned char v[4];

			Write32(v, (uint32_t) Value(c, set->column, lo));
			AddBytes(&values, v, 4);
			singles++;
		}
		else if (runs[i].hi - lo > 1)
		{
			Part	   *part = &parts[nparts++];

			head[0] = ANN_FILTER_RANGE;
			if ((part->bytes = Take(c, headSize + 8)) == NULL)
				return 0;
			memcpy(part->bytes, head, headSize);
			Write32(part->bytes + headSize, (uint32_t) Value(c, set->column, lo));
			Write32(part->bytes + headSize + 4, (uint32_t) Value(c, set->column, runs[i].hi - 1));
			part->nbytes = headSize + 8;
		}
	}
	if (values.failed || singles > ANN_FILTER_MAX_VALUES)
	{
		free(values.data);
		c->failed = 1;
		return 0;
	}
	if (singles)
	{
		Part	   *part = &parts[nparts++];

		head[0] = ANN_FILTER_VALUES;
		if ((part->bytes = Take(c, headSize + 4 + values.size)) == NULL)
		{
			free(values.data);
			return 0;
		}
		memcpy(part->bytes, head, headSize);
		Write32(part->bytes + headSize, singles);
		memcpy(part->bytes + headSize + 4, values.data, values.size);
		part->nbytes = headSize + 4 + values.size;
	}
	free(values.data);
	qsort(parts, nparts, sizeof(Part), PartCompare);
	for (uint32_t i = 0; i < nparts; i++)
	{
		unsigned char orStep = ANN_FILTER_OR;

		AddBytes(out, parts[i].bytes, parts[i].nbytes);
		if (i > 0)
			AddBytes(out, &orStep, 1);
	}
	return !out->failed;
}

static int
Serialize(Canon *c, Node *node)
{
	ByteBuffer	out = {0};
	unsigned char op;

	if (node->bytes || c->failed)
		return !c->failed;
	switch (node->kind)
	{
		case K_ALL:
		case K_NONE:
			op = node->kind == K_ALL ? ANN_FILTER_ALL : ANN_FILTER_NONE;
			AddBytes(&out, &op, 1);
			break;
		case K_LEAF:
			AddBytes(&out, node->leaf, node->leafSize);
			break;
		case K_NOT:
			if (!Serialize(c, node->children[0]))
				return 0;
			op = ANN_FILTER_NOT;
			AddBytes(&out, node->children[0]->bytes, node->children[0]->nbytes);
			AddBytes(&out, &op, 1);
			break;
		case K_AND:
		case K_OR:
			op = node->kind == K_AND ? ANN_FILTER_AND : ANN_FILTER_OR;
			for (uint32_t i = 0; i < node->nchildren; i++)
			{
				if (!Serialize(c, node->children[i]))
				{
					free(out.data);
					return 0;
				}
				AddBytes(&out, node->children[i]->bytes, node->children[i]->nbytes);
				if (i > 0)
					AddBytes(&out, &op, 1);
			}
			break;
		case K_SET:
			{
				/* The set, or NOT of its complement, whichever is shorter. */
				ByteBuffer	other = {0};
				uint32_t	n;
				Run		   *runs = Complement(c, node->runs, node->nruns, Count(c, node->column), &n);
				unsigned char notStep = ANN_FILTER_NOT;

				if (runs == NULL || !Positive(c, node, node->runs, node->nruns, &out) ||
					!Positive(c, node, runs, n, &other))
				{
					free(out.data);
					free(other.data);
					c->failed = 1;
					return 0;
				}
				AddBytes(&other, &notStep, 1);
				if (!other.failed && other.size < out.size)
				{
					free(out.data);
					out = other;
				}
				else
					free(other.data);
			}
			break;
	}
	if (out.failed || (node->bytes = Take(c, out.size)) == NULL)
	{
		free(out.data);
		c->failed = 1;
		return 0;
	}
	memcpy(node->bytes, out.data, out.size);
	node->nbytes = out.size;
	free(out.data);
	return 1;
}

int
AnnFilterCanonical(const AnnFilterSource *source, const unsigned char *program, size_t size,
				   unsigned char **out, size_t *outSize)
{
	Canon		c = {source, NULL, 0, 0, 0};
	Node	   *stack[ANN_FILTER_MAX_DEPTH + 1];
	int			depth = 0;
	size_t		offset = 0;
	int			ok = 0;

	*out = NULL;
	*outSize = 0;
	if (!AnnFilterCheck(program, size) || source->column == NULL || source->count == NULL ||
		source->value == NULL)
		return 0;
	while (offset < size && !c.failed)
	{
		int			op = program[offset];
		size_t		step = 1;
		Node	   *node = NULL;

		switch (op)
		{
			case ANN_FILTER_AND:
			case ANN_FILTER_OR:
				node = Combine(&c, op == ANN_FILTER_AND ? K_AND : K_OR, stack[depth - 2], stack[depth - 1]);
				depth -= 2;
				break;
			case ANN_FILTER_NOT:
				node = Not(&c, stack[--depth]);
				break;
			case ANN_FILTER_ALL:
			case ANN_FILTER_NONE:
				node = NewNode(&c, op == ANN_FILTER_ALL ? K_ALL : K_NONE);
				break;
			case ANN_FILTER_VALUES:
			case ANN_FILTER_OVERLAP:
				step = 7 + (size_t) program[offset + 1] + (size_t) Read32(program + offset + 3 + program[offset + 1]) * 4;
				node = Leaf(&c, program + offset, step);
				break;
			case ANN_FILTER_RANGE:
				step = 11 + (size_t) program[offset + 1];
				node = Leaf(&c, program + offset, step);
				break;
			case ANN_FILTER_NULL:
				step = 3 + (size_t) program[offset + 1];
				node = Leaf(&c, program + offset, step);
				break;
			default:			/* BITMAP: stored filters stay as written */
				c.failed = 1;
				break;
		}
		if (node == NULL)
			c.failed = 1;
		else
			stack[depth++] = node;
		offset += step;
	}
	if (!c.failed && depth == 1 && Serialize(&c, stack[0]) && stack[0]->nbytes <= ANN_FILTER_MAX_PROGRAM &&
		AnnFilterCheck(stack[0]->bytes, stack[0]->nbytes) && (*out = malloc(stack[0]->nbytes)) != NULL)
	{
		memcpy(*out, stack[0]->bytes, stack[0]->nbytes);
		*outSize = stack[0]->nbytes;
		ok = 1;
	}
	for (size_t i = 0; i < c.nowned; i++)
		free(c.owned[i]);
	free(c.owned);
	return ok;
}
