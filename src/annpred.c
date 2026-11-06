#include "postgres.h"

#include "access/stratnum.h"
#include "ann.h"
#include "ann/filter.h"
#include "catalog/pg_am_d.h"
#include "catalog/pg_operator_d.h"
#include "catalog/pg_opfamily_d.h"
#include "commands/defrem.h"
#include "catalog/pg_type_d.h"
#include "executor/executor.h"
#include "executor/spi.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/clauses.h"
#include "optimizer/optimizer.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/plancache.h"
#include "utils/rel.h"
#include "utils/varlena.h"

/*
 * Predicates over a frozen index's INCLUDE attributes become ann/filter.h
 * programs: =, <>, <, <=, >, >=, BETWEEN, IN, NOT IN, IS [NOT] NULL, AND,
 * OR and NOT, with integer operands that are constants, parameters or
 * stable expressions, and && of an integer[] attribute with an integer[]
 * operand. NOT is pushed down to the leaves, where a negated comparison
 * keeps SQL's meaning: rows whose attribute is NULL never match.
 */
#define STRATEGY_NE 6

typedef struct Context
{
	Index		varno;
	const AnnPredicateColumns *columns;
	StringInfo	program;		/* NULL when only checking */
	PlanState  *parent;
	ExprContext *econtext;
	int			steps;			/* program instructions, when checking */
	int			nesting;		/* current and deepest AND/OR nesting */
	int			maxNesting;
	size_t		bytes;			/* constant IN lists' values */
} Context;

void
AnnPredicateColumnsInit(Relation index, AnnPredicateColumns *columns)
{
	/*
	 * Computed once per relcache entry (GetDefaultOpClass scans pg_opclass),
	 * in local memory: catalog lookups can process invalidations that free
	 * the cache, which is only written once they are done.
	 */
	if (!AnnFrozenCachedColumns(index, columns))
	{
		AnnPredicateColumns local;

		memset(&local, 0, sizeof(local));
		local.ncolumns = RelationGetDescr(index)->natts - 2;
		for (int c = 0; c < local.ncolumns; c++)
		{
			Oid			opclass;

			local.attnos[c] = index->rd_index->indkey.values[2 + c];
			local.types[c] = TupleDescAttr(RelationGetDescr(index), 2 + c)->atttypid;
			local.collations[c] = AnnFrozenColumnCollation(index, 2 + c);
			/* integer[] takes && only: no comparison strategies */
			if (AnnFrozenMultiType(local.types[c]))
			{
				local.families[c] = InvalidOid;
				continue;
			}
			/* varchar compares with text's operators */
			opclass = GetDefaultOpClass(AnnFrozenTextType(local.types[c]) ? TEXTOID : local.types[c],
										BTREE_AM_OID);
			local.families[c] = OidIsValid(opclass) ? get_opclass_family(opclass) : InvalidOid;
		}
		AnnFrozenStoreColumns(index, &local);
		if (!AnnFrozenCachedColumns(index, columns))
			elog(ERROR, "could not cache the predicate columns of \"%s\"", RelationGetRelationName(index));
	}
	columns->index = index;
	columns->meta = AnnFrozenMeta(index);
}

/*
 * The predicate columns of a table's attribute columns, by name, for
 * programs only GPU nodes evaluate (no frozen index on this host): integer,
 * boolean, date and text columns. Text columns translate through
 * dictionaries[] (the collection's), which the caller attaches.
 */
void
AnnPredicateColumnsForTable(Relation rel, int count, char **names, AnnPredicateColumns *columns)
{
	TupleDesc	desc = RelationGetDescr(rel);

	if (count > INDEX_MAX_KEYS)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("an ann collection has at most %d attributes", INDEX_MAX_KEYS)));
	memset(columns, 0, sizeof(*columns));
	for (int c = 0; c < count; c++)
	{
		AttrNumber	attno = InvalidAttrNumber;
		Oid			opclass;

		for (int i = 0; i < desc->natts && attno == InvalidAttrNumber; i++)
			if (!TupleDescAttr(desc, i)->attisdropped &&
				strcmp(NameStr(TupleDescAttr(desc, i)->attname), names[c]) == 0)
				attno = i + 1;
		if (attno == InvalidAttrNumber)
			ereport(ERROR,
					(errcode(ERRCODE_UNDEFINED_COLUMN),
					 errmsg("column \"%s\" of an ann collection is not a column of \"%s\"",
							names[c], RelationGetRelationName(rel))));
		columns->attnos[c] = attno;
		columns->types[c] = TupleDescAttr(desc, attno - 1)->atttypid;
		columns->collations[c] = TupleDescAttr(desc, attno - 1)->attcollation;
		columns->names[c] = names[c];
		if (!AnnFrozenAttributeType(columns->types[c]))
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("ann collection attribute \"%s\" must be smallint, integer, bigint, boolean, date, text or varchar",
							names[c])));
		opclass = GetDefaultOpClass(AnnFrozenTextType(columns->types[c]) ? TEXTOID : columns->types[c],
									BTREE_AM_OID);
		columns->families[c] = OidIsValid(opclass) ? get_opclass_family(opclass) : InvalidOid;
	}
	columns->ncolumns = count;
}

/* Most strings a text range over a collection attribute may match. */
int			ann_collection_text_range_limit = ANN_FILTER_MAX_VALUES;

/* Bumped when a setting changes how filters translate: cached programs go. */
uint64		ann_collection_program_epoch = 1;

void
AnnCollectionTextRangeLimitAssign(int value, void *extra)
{
	ann_collection_program_epoch++;
}

static void
PutByte(StringInfo program, int value)
{
	appendStringInfoChar(program, (char) value);
}

static void
Put32(StringInfo program, uint32 value)
{
	char		bytes[4];

	for (int i = 0; i < 4; i++)
		bytes[i] = (char) (value >> (8 * i));
	appendBinaryStringInfo(program, bytes, 4);
}

static void
PutLeaf(Context *cxt, int op, int column, bool negate)
{
	const char *name = cxt->columns->names[column];

	PutByte(cxt->program, op);
	PutByte(cxt->program, (int) strlen(name));
	appendBinaryStringInfo(cxt->program, name, strlen(name));
	PutByte(cxt->program, negate ? 1 : 0);
}

static void
PutRange(Context *cxt, int column, int64 low, int64 high)
{
	low = Max(low, (int64) PG_INT32_MIN + 1);
	high = Min(high, (int64) PG_INT32_MAX);
	if (low > high)
	{
		PutByte(cxt->program, ANN_FILTER_NONE);
		return;
	}
	PutLeaf(cxt, ANN_FILTER_RANGE, column, false);
	Put32(cxt->program, (uint32) (int32) low);
	Put32(cxt->program, (uint32) (int32) high);
}

static bool
InRange(int64 value)
{
	return value > PG_INT32_MIN && value <= PG_INT32_MAX;
}

/* Attribute index of a plain column reference of the scanned table, or -1. */
static int
ColumnOf(Context *cxt, Node *node)
{
	Var		   *var;

	while (node && IsA(node, RelabelType))
		node = (Node *) ((RelabelType *) node)->arg;
	if (node == NULL || !IsA(node, Var))
		return -1;
	var = (Var *) node;
	if (var->varno != cxt->varno || var->varlevelsup != 0)
		return -1;
	for (int c = 0; c < cxt->columns->ncolumns; c++)
		if (cxt->columns->attnos[c] == var->varattno)
			return c;
	return -1;
}

/* Operand types an attribute of this type is compared with. */
static bool
OperandType(Oid column, Oid type, bool array)
{
	if (column == BOOLOID)
		return type == (array ? BOOLARRAYOID : BOOLOID);
	if (column == DATEOID)
		return type == (array ? DATEARRAYOID : DATEOID);
	if (AnnFrozenTextType(column))
		return type == (array ? TEXTARRAYOID : TEXTOID);
	if (array)
		return type == INT2ARRAYOID || type == INT4ARRAYOID || type == INT8ARRAYOID;
	return type == INT2OID || type == INT4OID || type == INT8OID;
}

/* A value known before the scan: no columns, nothing volatile. */
static bool
OperandOk(Context *cxt, int column, Node *node, bool array)
{
	if (!OperandType(cxt->columns->types[column], exprType(node), array))
		return false;
	return !contain_var_clause(node) && !contain_volatile_functions(node) && !contain_subplans(node);
}

static Datum
Evaluate(Context *cxt, Node *node, bool *isnull)
{
	ExprState  *state;

	if (IsA(node, Const))
	{
		*isnull = ((Const *) node)->constisnull;
		return ((Const *) node)->constvalue;
	}
	state = ExecInitExpr((Expr *) node, cxt->parent);
	return ExecEvalExprSwitchContext(state, cxt->econtext, isnull);
}

/* The int32 domain value of an operand: integers, 0/1, date days. */
static int64
IntegerValue(Datum value, Oid type)
{
	switch (type)
	{
		case INT2OID:
			return DatumGetInt16(value);
		case INT8OID:
			return DatumGetInt64(value);
		case BOOLOID:
			return DatumGetBool(value) ? 1 : 0;
		default:
			return DatumGetInt32(value);	/* integer, date */
	}
}

/* Strategy of a comparison in the column's btree family, STRATEGY_NE for <>, else 0. */
static int
Strategy(Oid opno, Oid family)
{
	int			strategy;
	Oid			negator;

	if (!OidIsValid(family))
		return 0;
	if ((strategy = get_op_opfamily_strategy(opno, family)) != 0)
		return strategy;
	negator = get_negator(opno);
	if (OidIsValid(negator) && get_op_opfamily_strategy(negator, family) == BTEqualStrategyNumber)
		return STRATEGY_NE;
	return 0;
}

static int
Commute(int strategy)
{
	switch (strategy)
	{
		case BTLessStrategyNumber:
			return BTGreaterStrategyNumber;
		case BTLessEqualStrategyNumber:
			return BTGreaterEqualStrategyNumber;
		case BTGreaterEqualStrategyNumber:
			return BTLessEqualStrategyNumber;
		case BTGreaterStrategyNumber:
			return BTLessStrategyNumber;
		default:
			return strategy;
	}
}

static int
Negate(int strategy)
{
	switch (strategy)
	{
		case BTLessStrategyNumber:
			return BTGreaterEqualStrategyNumber;
		case BTLessEqualStrategyNumber:
			return BTGreaterStrategyNumber;
		case BTEqualStrategyNumber:
			return STRATEGY_NE;
		case BTGreaterEqualStrategyNumber:
			return BTLessStrategyNumber;
		case BTGreaterStrategyNumber:
			return BTLessEqualStrategyNumber;
		default:
			return BTEqualStrategyNumber;
	}
}

static void
PutComparison(Context *cxt, int column, int strategy, int64 value)
{
	switch (strategy)
	{
		case BTEqualStrategyNumber:
			if (!InRange(value))
				PutByte(cxt->program, ANN_FILTER_NONE);
			else
			{
				PutLeaf(cxt, ANN_FILTER_VALUES, column, false);
				Put32(cxt->program, 1);
				Put32(cxt->program, (uint32) (int32) value);
			}
			break;
		case STRATEGY_NE:
			if (!InRange(value))
				PutLeaf(cxt, ANN_FILTER_NULL, column, true);
			else
			{
				PutLeaf(cxt, ANN_FILTER_VALUES, column, true);
				Put32(cxt->program, 1);
				Put32(cxt->program, (uint32) (int32) value);
			}
			break;
		case BTLessStrategyNumber:
			if (value == PG_INT64_MIN)
				PutByte(cxt->program, ANN_FILTER_NONE);
			else
				PutRange(cxt, column, PG_INT64_MIN, value - 1);
			break;
		case BTLessEqualStrategyNumber:
			PutRange(cxt, column, PG_INT64_MIN, value);
			break;
		case BTGreaterStrategyNumber:
			if (value == PG_INT64_MAX)
				PutByte(cxt->program, ANN_FILTER_NONE);
			else
				PutRange(cxt, column, value + 1, PG_INT64_MAX);
			break;
		default:
			PutRange(cxt, column, value, PG_INT64_MAX);
			break;
	}
}

/* Text comparisons must use the column's collation, which orders its codes. */
static bool
SameCollation(Context *cxt, int column, Oid collation)
{
	return !AnnFrozenTextType(cxt->columns->types[column]) || collation == cxt->columns->collations[column];
}

/*
 * Position of the first string at or after the value (after it when upper)
 * in the column's collation order: a frozen index's dictionary, or a
 * collection's sorted strings.
 */
static int64
TextPosition(Context *cxt, int column, text *value, bool upper)
{
	AnnTextDictionary *dictionary = cxt->columns->dictionaries[column];
	int64		low = 0;
	int64		high;

	if (dictionary == NULL)
		return AnnFrozenTextSearch(cxt->columns->index, cxt->columns->meta, column,
								   cxt->columns->collations[column], value, upper);
	high = dictionary->count;
	while (low < high)
	{
		int64		middle = low + (high - low) / 2;
		text	   *probe = dictionary->sorted[middle];
		int			cmp = varstr_cmp(VARDATA_ANY(probe), VARSIZE_ANY_EXHDR(probe), VARDATA_ANY(value),
									 VARSIZE_ANY_EXHDR(value), cxt->columns->collations[column]);

		if (cmp < 0 || (upper && cmp == 0))
			low = middle + 1;
		else
			high = middle;
	}
	return low;
}

/* The code of the string at a position of the collation order. */
static int64
TextCode(Context *cxt, int column, int64 position)
{
	AnnTextDictionary *dictionary = cxt->columns->dictionaries[column];

	return dictionary ? dictionary->codes[position] : position;
}

static int
Int32Order(const void *a, const void *b)
{
	int32		x = *(const int32 *) a;
	int32		y = *(const int32 *) b;

	return (x > y) - (x < y);
}

/*
 * The strings at positions first .. last - 1 of a collection's collation
 * order, as the set of their codes (which don't follow the collation).
 */
static void
PutCodeSet(Context *cxt, int column, int64 first, int64 last)
{
	AnnTextDictionary *dictionary = cxt->columns->dictionaries[column];
	int32	   *codes;
	int64		n = last - first;

	if (n <= 0)
	{
		PutByte(cxt->program, ANN_FILTER_NONE);
		return;
	}
	if (n > ann_collection_text_range_limit ||
		cxt->program->len + n * 4 + (n / ANN_FILTER_MAX_VALUES + 1) * (16 + NAMEDATALEN) > ANN_FILTER_MAX_PROGRAM)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("a text range over ann collection attribute \"%s\" matches %lld strings, more than %d",
						cxt->columns->names[column], (long long) n,
						Min(ann_collection_text_range_limit, ANN_TEXT_RANGE_MAX)),
				 errhint("Raise ann.collection_text_range_limit (at most %d), or narrow the range.",
						 ANN_TEXT_RANGE_MAX)));
	codes = palloc(sizeof(int32) * n);
	memcpy(codes, dictionary->codes + first, sizeof(int32) * n);
	qsort(codes, n, sizeof(int32), Int32Order);

	/* A leaf holds at most ANN_FILTER_MAX_VALUES values: OR the chunks. */
	for (int64 start = 0; start < n; start += ANN_FILTER_MAX_VALUES)
	{
		int64		count = Min(n - start, ANN_FILTER_MAX_VALUES);

		PutLeaf(cxt, ANN_FILTER_VALUES, column, false);
		Put32(cxt->program, (uint32) count);
		for (int64 i = start; i < start + count; i++)
			Put32(cxt->program, (uint32) codes[i]);
		if (start > 0)
			PutByte(cxt->program, ANN_FILTER_OR);
	}
	pfree(codes);
}

/*
 * A text comparison over the dictionary's codes. A frozen index's codes
 * follow the collation, so the strings below, at or above a value are code
 * ranges; a collection's are sets of codes.
 */
static void
PutTextComparison(Context *cxt, int column, int strategy, text *value)
{
	int64		lower = TextPosition(cxt, column, value, false);
	int64		upper = TextPosition(cxt, column, value, true);
	bool		found = upper > lower;
	AnnTextDictionary *dictionary = cxt->columns->dictionaries[column];

	if (strategy == BTEqualStrategyNumber || strategy == STRATEGY_NE)
	{
		if (found)
			PutComparison(cxt, column, strategy, TextCode(cxt, column, lower));
		else if (strategy == BTEqualStrategyNumber)
			PutByte(cxt->program, ANN_FILTER_NONE);
		else
			PutLeaf(cxt, ANN_FILTER_NULL, column, true);
		return;
	}
	if (dictionary)
	{
		switch (strategy)
		{
			case BTLessStrategyNumber:
				PutCodeSet(cxt, column, 0, lower);
				break;
			case BTLessEqualStrategyNumber:
				PutCodeSet(cxt, column, 0, upper);
				break;
			case BTGreaterStrategyNumber:
				PutCodeSet(cxt, column, upper, dictionary->count);
				break;
			default:
				PutCodeSet(cxt, column, lower, dictionary->count);
				break;
		}
		return;
	}
	switch (strategy)
	{
		case BTLessStrategyNumber:
			PutComparison(cxt, column, BTLessStrategyNumber, lower);
			break;
		case BTLessEqualStrategyNumber:
			PutComparison(cxt, column, BTLessStrategyNumber, upper);
			break;
		case BTGreaterStrategyNumber:
			PutComparison(cxt, column, BTGreaterEqualStrategyNumber, upper);
			break;
		default:
			PutComparison(cxt, column, BTGreaterEqualStrategyNumber, lower);
			break;
	}
}

static bool Translate(Context *cxt, Node *node, bool negate);

/*
 * column && array, either way round, for an integer[] attribute: one
 * OVERLAP leaf of the array's distinct elements. NULL elements never
 * overlap, nor does the value standing for NULL, which arrays never hold.
 */
static bool
TranslateOverlap(Context *cxt, OpExpr *op, bool negate)
{
	int			column;
	Node	   *operand;
	Datum		value;
	bool		isnull;
	ArrayType  *array;
	Datum	   *elements;
	bool	   *nulls;
	int			n;
	int32	   *values;
	int			nvalues = 0;

	if ((column = ColumnOf(cxt, linitial(op->args))) >= 0)
		operand = lsecond(op->args);
	else if ((column = ColumnOf(cxt, lsecond(op->args))) >= 0)
		operand = linitial(op->args);
	else
		return false;
	if (!AnnFrozenMultiType(cxt->columns->types[column]) || exprType(operand) != INT4ARRAYOID ||
		contain_var_clause(operand) || contain_volatile_functions(operand) || contain_subplans(operand))
		return false;
	if (cxt->program == NULL)
		return true;
	value = Evaluate(cxt, operand, &isnull);
	if (isnull)
	{
		PutByte(cxt->program, ANN_FILTER_NONE);
		return true;
	}
	array = DatumGetArrayTypeP(value);
	deconstruct_array(array, INT4OID, sizeof(int32), true, TYPALIGN_INT, &elements, &nulls, &n);
	values = palloc(sizeof(int32) * Max(n, 1));
	for (int i = 0; i < n; i++)
		if (!nulls[i] && DatumGetInt32(elements[i]) != ANN_FROZEN_NULL)
			values[nvalues++] = DatumGetInt32(elements[i]);
	if (nvalues > 1)
	{
		int			unique = 1;

		qsort(values, nvalues, sizeof(int32), Int32Order);
		for (int i = 1; i < nvalues; i++)
			if (values[i] != values[unique - 1])
				values[unique++] = values[i];
		nvalues = unique;
	}
	if (nvalues > ANN_FILTER_MAX_VALUES)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("ann predicates take at most %d values per && array", ANN_FILTER_MAX_VALUES)));
	if (nvalues == 0)
	{
		/* nothing overlaps; NOT of that holds for every array but NULL */
		if (negate)
			PutLeaf(cxt, ANN_FILTER_NULL, column, true);
		else
			PutByte(cxt->program, ANN_FILTER_NONE);
	}
	else
	{
		PutLeaf(cxt, ANN_FILTER_OVERLAP, column, negate);
		Put32(cxt->program, (uint32) nvalues);
		for (int i = 0; i < nvalues; i++)
			Put32(cxt->program, (uint32) values[i]);
	}
	pfree(values);
	return true;
}

static bool
TranslateOp(Context *cxt, OpExpr *op, bool negate)
{
	Node	   *left;
	Node	   *right;
	int			column;
	int			strategy;
	Node	   *operand;
	Datum		value;
	bool		isnull;

	if (list_length(op->args) != 2)
		return false;
	if (op->opno == OID_ARRAY_OVERLAP_OP)
		return TranslateOverlap(cxt, op, negate);
	left = linitial(op->args);
	right = lsecond(op->args);
	if ((column = ColumnOf(cxt, left)) >= 0)
	{
		operand = right;
		strategy = Strategy(op->opno, cxt->columns->families[column]);
	}
	else if ((column = ColumnOf(cxt, right)) >= 0)
	{
		operand = left;
		strategy = Commute(Strategy(op->opno, cxt->columns->families[column]));
	}
	else
		return false;
	if (strategy == 0 || !OperandOk(cxt, column, operand, false) || !SameCollation(cxt, column, op->inputcollid))
		return false;
	if (cxt->program == NULL)
		return true;
	if (negate)
		strategy = Negate(strategy);
	value = Evaluate(cxt, operand, &isnull);
	if (isnull)
		PutByte(cxt->program, ANN_FILTER_NONE);
	else if (AnnFrozenTextType(cxt->columns->types[column]))
		PutTextComparison(cxt, column, strategy, DatumGetTextPP(value));
	else
		PutComparison(cxt, column, strategy, IntegerValue(value, exprType(operand)));
	return true;
}

static bool
TranslateArray(Context *cxt, ScalarArrayOpExpr *op, bool negate)
{
	int			column;
	int			strategy;
	bool		in;
	Node	   *operand;
	Datum		value;
	bool		isnull;
	ArrayType  *array;
	Oid			type;
	int16		typlen;
	bool		typbyval;
	char		typalign;
	Datum	   *elements;
	bool	   *nulls;
	int			n;
	int			nvalues = 0;
	bool		anyNull = false;
	StringInfoData values;

	if (list_length(op->args) != 2 || (column = ColumnOf(cxt, linitial(op->args))) < 0)
		return false;
	operand = lsecond(op->args);
	strategy = Strategy(op->opno, cxt->columns->families[column]);
	if (op->useOr && strategy == BTEqualStrategyNumber)
		in = true;
	else if (!op->useOr && strategy == STRATEGY_NE)
		in = false;
	else
		return false;
	if (!OperandOk(cxt, column, operand, true) || !SameCollation(cxt, column, op->inputcollid))
		return false;
	if (cxt->program == NULL)
		return true;
	if (negate)
		in = !in;
	value = Evaluate(cxt, operand, &isnull);
	if (isnull)
	{
		PutByte(cxt->program, ANN_FILTER_NONE);
		return true;
	}
	array = DatumGetArrayTypeP(value);
	type = ARR_ELEMTYPE(array);
	get_typlenbyvalalign(type, &typlen, &typbyval, &typalign);
	deconstruct_array(array, type, typlen, typbyval, typalign, &elements, &nulls, &n);
	initStringInfo(&values);
	for (int i = 0; i < n; i++)
	{
		int64		v;

		if (nulls[i])
		{
			anyNull = true;
			continue;
		}
		if (AnnFrozenTextType(cxt->columns->types[column]))
		{
			/* a string's code, or none when it is not in the dictionary */
			text	   *string = DatumGetTextPP(elements[i]);
			int64		lower = TextPosition(cxt, column, string, false);

			if (TextPosition(cxt, column, string, true) == lower)
				continue;
			v = TextCode(cxt, column, lower);
		}
		else
			v = IntegerValue(elements[i], type);
		if (!InRange(v))
			continue;
		Put32(&values, (uint32) (int32) v);
		nvalues++;
	}
	if (nvalues > ANN_FILTER_MAX_VALUES)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("ann predicates take at most %d values per IN list", ANN_FILTER_MAX_VALUES)));
	if (in && nvalues == 0)
		PutByte(cxt->program, ANN_FILTER_NONE);
	else if (!in && anyNull)
		PutByte(cxt->program, ANN_FILTER_NONE); /* x <> NULL is never true */
	else if (!in && n == 0)
		PutByte(cxt->program, ANN_FILTER_ALL);	/* <> ALL of nothing holds */
	else if (!in && nvalues == 0)
		PutLeaf(cxt, ANN_FILTER_NULL, column, true);
	else
	{
		PutLeaf(cxt, ANN_FILTER_VALUES, column, !in);
		Put32(cxt->program, (uint32) nvalues);
		appendBinaryStringInfo(cxt->program, values.data, values.len);
	}
	return true;
}

static bool
Translate(Context *cxt, Node *node, bool negate)
{
	if (node == NULL)
		return false;
	if (IsA(node, RestrictInfo))
		node = (Node *) ((RestrictInfo *) node)->clause;
	if (IsA(node, BoolExpr))
	{
		BoolExpr   *expr = (BoolExpr *) node;
		bool		conjunction = (expr->boolop == AND_EXPR) != negate;
		int			i = 0;
		ListCell   *lc;

		if (expr->boolop == NOT_EXPR)
			return Translate(cxt, linitial(expr->args), !negate);
		cxt->maxNesting = Max(cxt->maxNesting, ++cxt->nesting);
		foreach(lc, expr->args)
		{
			if (!Translate(cxt, lfirst(lc), negate))
				return false;
			if (i++ > 0)
			{
				cxt->steps++;
				if (cxt->program)
					PutByte(cxt->program, conjunction ? ANN_FILTER_AND : ANN_FILTER_OR);
			}
		}
		cxt->nesting--;
		return true;
	}

	/*
	 * a leaf: at most three instructions (a NOT TRUE test is two leaves and
	 * OR)
	 */
	cxt->steps += 3;
	if (IsA(node, OpExpr))
	{
		OpExpr	   *op = (OpExpr *) node;

		/* constant && arrays count against the program's size, like lists */
		if (op->opno == OID_ARRAY_OVERLAP_OP && list_length(op->args) == 2)
		{
			for (int side = 0; side < 2; side++)
			{
				Node	   *operand = list_nth(op->args, side);

				if (IsA(operand, Const) && !((Const *) operand)->constisnull &&
					((Const *) operand)->consttype == INT4ARRAYOID)
				{
					ArrayType  *array = DatumGetArrayTypeP(((Const *) operand)->constvalue);
					int			n = ArrayGetNItems(ARR_NDIM(array), ARR_DIMS(array));

					if (n > ANN_FILTER_MAX_VALUES)
						return false;
					cxt->bytes += (size_t) n * 4 + 80;
				}
			}
		}
		return TranslateOp(cxt, op, negate);
	}
	if (IsA(node, ScalarArrayOpExpr))
	{
		Node	   *operand = list_length(((ScalarArrayOpExpr *) node)->args) == 2 ?
			lsecond(((ScalarArrayOpExpr *) node)->args) : NULL;

		/* constant lists count against the program's size */
		if (operand && IsA(operand, Const) && !((Const *) operand)->constisnull)
		{
			ArrayType  *array = DatumGetArrayTypeP(((Const *) operand)->constvalue);
			int			n = ArrayGetNItems(ARR_NDIM(array), ARR_DIMS(array));

			if (n > ANN_FILTER_MAX_VALUES)
				return false;
			cxt->bytes += (size_t) n * 4 + 80;
		}
		return TranslateArray(cxt, (ScalarArrayOpExpr *) node, negate);
	}
	if (ColumnOf(cxt, node) >= 0 && cxt->columns->types[ColumnOf(cxt, node)] == BOOLOID)
	{
		/*
		 * "flag" is flag = true; NOT flag is flag = false (NULL never
		 * matches).
		 */
		if (cxt->program)
			PutComparison(cxt, ColumnOf(cxt, node), BTEqualStrategyNumber, negate ? 0 : 1);
		return true;
	}
	if (IsA(node, BooleanTest))
	{
		BooleanTest *test = (BooleanTest *) node;
		int			column = ColumnOf(cxt, (Node *) test->arg);
		BoolTestType type = test->booltesttype;

		if (column < 0 || cxt->columns->types[column] != BOOLOID)
			return false;
		if (cxt->program == NULL)
			return true;
		/* Boolean tests are never NULL, so NOT is the complementary test. */
		if (negate)
			type = type == IS_TRUE ? IS_NOT_TRUE : type == IS_NOT_TRUE ? IS_TRUE :
				type == IS_FALSE ? IS_NOT_FALSE : type == IS_NOT_FALSE ? IS_FALSE :
				type == IS_UNKNOWN ? IS_NOT_UNKNOWN : IS_UNKNOWN;
		if (type == IS_UNKNOWN || type == IS_NOT_UNKNOWN)
			PutLeaf(cxt, ANN_FILTER_NULL, column, type == IS_NOT_UNKNOWN);
		else
		{
			PutComparison(cxt, column, BTEqualStrategyNumber, type == IS_TRUE || type == IS_NOT_FALSE ? 1 : 0);
			if (type == IS_NOT_TRUE || type == IS_NOT_FALSE)
			{
				PutLeaf(cxt, ANN_FILTER_NULL, column, false);
				PutByte(cxt->program, ANN_FILTER_OR);
			}
		}
		return true;
	}
	if (IsA(node, NullTest))
	{
		NullTest   *test = (NullTest *) node;
		int			column = ColumnOf(cxt, (Node *) test->arg);

		if (column < 0 || test->argisrow)
			return false;
		if (cxt->program)
			PutLeaf(cxt, ANN_FILTER_NULL, column, (test->nulltesttype == IS_NULL) == negate);
		return true;
	}
	if (IsA(node, Const) && ((Const *) node)->consttype == BOOLOID)
	{
		Const	   *constant = (Const *) node;

		if (cxt->program)
			PutByte(cxt->program, !constant->constisnull && (DatumGetBool(constant->constvalue) != negate) ?
					ANN_FILTER_ALL : ANN_FILTER_NONE);
		return true;
	}
	return false;
}

/* Whether every clause can be evaluated from the index's attribute columns. */
bool
AnnPredicateSupported(List *clauses, Index varno, const AnnPredicateColumns *columns)
{
	Context		cxt = {varno, columns, NULL, NULL, NULL};
	ListCell   *lc;

	foreach(lc, clauses)
	{
		if (!Translate(&cxt, lfirst(lc), false))
			return false;
		cxt.steps++;
	}

	/*
	 * Programs the evaluator would refuse keep ordinary plans. Stack depth is
	 * at most the AND/OR nesting plus two; parameter lists are checked when
	 * the program is built.
	 */
	return cxt.steps <= ANN_FILTER_MAX_STEPS && cxt.maxNesting + 2 <= ANN_FILTER_MAX_DEPTH &&
		cxt.bytes + (size_t) cxt.steps * 80 <= ANN_FILTER_MAX_PROGRAM;
}

static int
ClauseCompare(const void *a, const void *b)
{
	const StringInfoData *x = *(const StringInfoData *const *) a;
	const StringInfoData *y = *(const StringInfoData *const *) b;
	int			c = memcmp(x->data, y->data, Min(x->len, y->len));

	return c != 0 ? c : (x->len > y->len) - (x->len < y->len);
}

/*
 * The AND of clauses, in an order of their own: each clause's instructions
 * are sorted by their bytes, so a conjunction has one program however the
 * query spelled or the planner ordered it, and every route caches its rows
 * under the same hash (here and on the worker).
 */
static void
PutConjunction(Context *cxt, List *clauses)
{
	StringInfo	program = cxt->program;
	int			n = list_length(clauses);
	StringInfoData **parts;
	int			i = 0;
	ListCell   *lc;

	if (n == 0)
	{
		PutByte(program, ANN_FILTER_ALL);
		return;
	}
	parts = palloc(sizeof(StringInfoData *) * n);
	foreach(lc, clauses)
	{
		parts[i] = palloc(sizeof(StringInfoData));
		initStringInfo(parts[i]);
		cxt->program = parts[i];
		if (!Translate(cxt, lfirst(lc), false))
			elog(ERROR, "unsupported ann predicate");
		i++;
	}
	cxt->program = program;
	qsort(parts, n, sizeof(StringInfoData *), ClauseCompare);
	for (i = 0; i < n; i++)
	{
		appendBinaryStringInfo(program, parts[i]->data, parts[i]->len);
		if (i > 0)
			PutByte(program, ANN_FILTER_AND);
		pfree(parts[i]->data);
		pfree(parts[i]);
	}
	pfree(parts);
}

/* The program of supported clauses (AND of them; ALL when there are none). */
void
AnnPredicateBuild(List *clauses, Index varno, const AnnPredicateColumns *columns,
				  PlanState *parent, ExprContext *econtext, StringInfo program)
{
	Context		cxt = {varno, columns, program, parent, econtext};

	PutConjunction(&cxt, clauses);
}

/*
 * The program of a SQL filter text over the table, parsed and analyzed
 * (never run) as SELECT ... WHERE filter. False when it is not a supported
 * predicate over the attribute columns.
 */
bool
AnnPredicateFromText(Oid relid, const char *filter, const AnnPredicateColumns *columns, StringInfo program)
{
	SPIPlanPtr	plan;
	List	   *sources;
	CachedPlanSource *source;
	Query	   *query;
	Node	   *quals;
	List	   *clauses;
	bool		supported = false;

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");
	plan = SPI_prepare(psprintf("SELECT 1 FROM %s WHERE %s",
								quote_qualified_identifier(get_namespace_name(get_rel_namespace(relid)),
														   get_rel_name(relid)), filter), 0, NULL);
	if (plan == NULL)
		elog(ERROR, "could not parse ann filter \"%s\": %s", filter, SPI_result_code_string(SPI_result));
	sources = SPI_plan_get_plan_sources(plan);
	if (list_length(sources) == 1 &&
		list_length((source = linitial(sources))->query_list) == 1 &&
		(query = linitial(source->query_list))->commandType == CMD_SELECT &&
		list_length(query->rtable) == 1 && query->jointree && query->jointree->quals &&
		!query->hasSubLinks)
	{
		ExprContext *econtext = CreateStandaloneExprContext();
		Context		cxt = {1, columns, NULL, NULL, econtext};

		quals = eval_const_expressions(NULL, copyObject(query->jointree->quals));
		clauses = make_ands_implicit((Expr *) quals);
		supported = AnnPredicateSupported(clauses, 1, columns);
		if (supported)
		{
			cxt.program = program;
			PutConjunction(&cxt, clauses);
		}
		FreeExprContext(econtext, true);
	}
	SPI_finish();
	return supported;
}
