#include "postgres.h"

#include "access/stratnum.h"
#include "ann.h"
#include "ann/filter.h"
#include "catalog/pg_opfamily_d.h"
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

/*
 * Predicates over a frozen index's INCLUDE attributes become ann/filter.h
 * programs: =, <>, <, <=, >, >=, BETWEEN, IN, NOT IN, IS [NOT] NULL, AND,
 * OR and NOT, with integer operands that are constants, parameters or
 * stable expressions. NOT is pushed down to the leaves, where a negated
 * comparison keeps SQL's meaning: rows whose attribute is NULL never match.
 */
#define STRATEGY_NE 6

typedef struct Context
{
	Index		varno;
	const AnnPredicateColumns *columns;
	StringInfo	program;		/* NULL when only checking */
	PlanState  *parent;
	ExprContext *econtext;
} Context;

void
AnnPredicateColumnsInit(Relation index, AnnPredicateColumns *columns)
{
	columns->ncolumns = RelationGetDescr(index)->natts - 2;
	for (int c = 0; c < columns->ncolumns; c++)
	{
		columns->attnos[c] = index->rd_index->indkey.values[2 + c];
		columns->names[c] = pstrdup(AnnFrozenColumnName(index, 3 + c));
	}
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

static bool
IntegerType(Oid type)
{
	return type == INT2OID || type == INT4OID || type == INT8OID;
}

/* A value known before the scan: no columns, nothing volatile. */
static bool
OperandOk(Node *node, bool array)
{
	Oid			type = exprType(node);

	if (array ? !(type == INT2ARRAYOID || type == INT4ARRAYOID || type == INT8ARRAYOID) : !IntegerType(type))
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

static int64
IntegerValue(Datum value, Oid type)
{
	if (type == INT2OID)
		return DatumGetInt16(value);
	if (type == INT4OID)
		return DatumGetInt32(value);
	return DatumGetInt64(value);
}

/* Strategy of an integer comparison, STRATEGY_NE for <>, else 0. */
static int
Strategy(Oid opno)
{
	int			strategy = get_op_opfamily_strategy(opno, INTEGER_BTREE_FAM_OID);
	Oid			negator;

	if (strategy)
		return strategy;
	negator = get_negator(opno);
	if (OidIsValid(negator) &&
		get_op_opfamily_strategy(negator, INTEGER_BTREE_FAM_OID) == BTEqualStrategyNumber)
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

static bool Translate(Context *cxt, Node *node, bool negate);

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
	left = linitial(op->args);
	right = lsecond(op->args);
	strategy = Strategy(op->opno);
	if (strategy == 0)
		return false;
	if ((column = ColumnOf(cxt, left)) >= 0)
		operand = right;
	else if ((column = ColumnOf(cxt, right)) >= 0)
	{
		operand = left;
		strategy = Commute(strategy);
	}
	else
		return false;
	if (!OperandOk(operand, false))
		return false;
	if (cxt->program == NULL)
		return true;
	if (negate)
		strategy = Negate(strategy);
	value = Evaluate(cxt, operand, &isnull);
	if (isnull)
		PutByte(cxt->program, ANN_FILTER_NONE);
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
	strategy = Strategy(op->opno);
	if (op->useOr && strategy == BTEqualStrategyNumber)
		in = true;
	else if (!op->useOr && strategy == STRATEGY_NE)
		in = false;
	else
		return false;
	if (!OperandOk(operand, true))
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
		PutByte(cxt->program, ANN_FILTER_NONE);	/* x <> NULL is never true */
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
		foreach(lc, expr->args)
		{
			if (!Translate(cxt, lfirst(lc), negate))
				return false;
			if (cxt->program && i++ > 0)
				PutByte(cxt->program, conjunction ? ANN_FILTER_AND : ANN_FILTER_OR);
		}
		return true;
	}
	if (IsA(node, OpExpr))
		return TranslateOp(cxt, (OpExpr *) node, negate);
	if (IsA(node, ScalarArrayOpExpr))
		return TranslateArray(cxt, (ScalarArrayOpExpr *) node, negate);
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
		if (!Translate(&cxt, lfirst(lc), false))
			return false;
	return true;
}

/* The program of supported clauses (AND of them; ALL when there are none). */
void
AnnPredicateBuild(List *clauses, Index varno, const AnnPredicateColumns *columns,
				  PlanState *parent, ExprContext *econtext, StringInfo program)
{
	Context		cxt = {varno, columns, program, parent, econtext};
	int			i = 0;
	ListCell   *lc;

	foreach(lc, clauses)
	{
		if (!Translate(&cxt, lfirst(lc), false))
			elog(ERROR, "unsupported ann predicate");
		if (i++ > 0)
			PutByte(program, ANN_FILTER_AND);
	}
	if (i == 0)
		PutByte(program, ANN_FILTER_ALL);
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
			ListCell   *lc;
			int			i = 0;

			cxt.program = program;
			foreach(lc, clauses)
			{
				(void) Translate(&cxt, lfirst(lc), false);
				if (i++ > 0)
					PutByte(program, ANN_FILTER_AND);
			}
			if (i == 0)
				PutByte(program, ANN_FILTER_ALL);
		}
		FreeExprContext(econtext, true);
	}
	SPI_finish();
	return supported;
}
