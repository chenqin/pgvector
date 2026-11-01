#include "postgres.h"

#include <math.h>

#include "access/relscan.h"
#include "access/tableam.h"
#include "access/visibilitymap.h"
#include "ann.h"
#include "ann/filter.h"
#include "catalog/pg_type_d.h"
#include "commands/defrem.h"
#include "executor/executor.h"
#include "miscadmin.h"
#include "nodes/extensible.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/cost.h"
#include "optimizer/clauses.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/restrictinfo.h"
#include "port/pg_bitutils.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rel.h"

/*
 * Ordered scans of a table with a frozen ann index, answered from the index
 * and the GPU worker: "WHERE <predicate over INCLUDE attributes> ORDER BY
 * vector <-> $q". The predicate becomes an ann/filter.h program; the worker
 * serving the index's generation evaluates it, caches its rows and searches
 * them (operation 22, ordinals in the reply); without such a worker, the
 * program is evaluated from the frozen pages and candidates are scored on
 * the CPU. Rows come from the index pages when every column the query uses
 * is in the index and the heap page is all-visible, else from the heap.
 *
 * The path sits at the scan level with the query's distance ordering, so
 * partitions and UNION ALL branches (a frozen base and a mutable delta)
 * merge in PostgreSQL's MergeAppend under one LIMIT.
 */
#define ANN_FROZEN_WORKER_MAX_K 10000
#define ANN_FROZEN_DEFAULT_K 100

bool		ann_enable_frozen_search = true;
static set_rel_pathlist_hook_type previous_pathlist_hook = NULL;

typedef struct FrozenScanState
{
	CustomScanState css;
	Oid			indexOid;
	bool		covering;		/* every used column is in the index */
	int			initialK;
	Relation	index;
	AnnFrozenMetaData *meta;
	AnnPredicateColumns columns;
	ExprState  *queryState;
	List	   *quals;			/* clauses the program is built from */
	FmgrInfo   *distance;
	char	   *workerName;		/* loaded export of this generation, or NULL */
	bool		workerFailed;
	IndexFetchTableData *fetch;
	TupleTableSlot *heapSlot;	/* heap fetches, copied to the scan slot */
	Buffer		vmbuf;
	AnnArrayReader tids;
	AnnArrayReader keys;
	AnnArrayReader vectors;
	AnnArrayReader *values;
	Vector	   *vector;			/* the returned row's vector */
	MemoryContext resultCtx;

	/* One execution's results, in distance order. */
	bool		started;
	StringInfoData program;
	Datum		query;
	bool		queryIsNull;
	uint64	   *ordinals;
	double	   *distances;
	uint64		nresults;
	uint64		next;
	uint64		requested;
	bool		exhausted;
	uint64	   *returned;		/* ordinals already returned, for refills */
	uint64		nreturned;

	/* EXPLAIN ANALYZE counters */
	uint64		eligible;
	uint64		heapFetches;
	uint64		workerCalls;
	uint64		localRuns;
	int			workerRoute;
} FrozenScanState;

static Plan *PlanFrozenPath(PlannerInfo *root, RelOptInfo *rel, CustomPath *path,
							List *tlist, List *clauses, List *custom_plans);
static Node *CreateFrozenScanState(CustomScan *scan);
static void BeginFrozenScan(CustomScanState *node, EState *estate, int eflags);
static TupleTableSlot *ExecFrozenScan(CustomScanState *node);
static void EndFrozenScan(CustomScanState *node);
static void ReScanFrozenScan(CustomScanState *node);
static void ExplainFrozenScan(CustomScanState *node, List *ancestors, ExplainState *es);

static const CustomPathMethods FrozenPathMethods = {
	.CustomName = "AnnFrozenSearch",
	.PlanCustomPath = PlanFrozenPath,
};

static const CustomScanMethods FrozenScanMethods = {
	.CustomName = "AnnFrozenSearch",
	.CreateCustomScanState = CreateFrozenScanState,
};

static const CustomExecMethods FrozenExecMethods = {
	.CustomName = "AnnFrozenSearch",
	.BeginCustomScan = BeginFrozenScan,
	.ExecCustomScan = ExecFrozenScan,
	.EndCustomScan = EndFrozenScan,
	.ReScanCustomScan = ReScanFrozenScan,
	.ExplainCustomScan = ExplainFrozenScan,
};

/* The operand of "column <-> operand" ordering this rel, if the column is indexed. */
static Node *
DistanceOperand(PlannerInfo *root, RelOptInfo *rel, PathKey *pathkey, IndexOptInfo **found)
{
	Oid			amoid = get_am_oid("ann", true);
	ListCell   *lc;

	if (!OidIsValid(amoid))
		return NULL;
	foreach(lc, pathkey->pk_eclass->ec_members)
	{
		EquivalenceMember *em = lfirst(lc);
		Node	   *expr = (Node *) em->em_expr;
		OpExpr	   *op;
		Var		   *var;
		Node	   *operand;
		ListCell   *ic;

		if (em->em_is_const || !bms_equal(em->em_relids, rel->relids))
			continue;
		while (IsA(expr, RelabelType))
			expr = (Node *) ((RelabelType *) expr)->arg;
		if (!IsA(expr, OpExpr) || list_length(((OpExpr *) expr)->args) != 2)
			continue;
		op = (OpExpr *) expr;
		var = linitial(op->args);
		operand = lsecond(op->args);
		if (!IsA(var, Var) || var->varno != rel->relid || var->varlevelsup != 0 ||
			contain_var_clause(operand) || contain_volatile_functions(operand) || contain_subplans(operand))
			continue;
		foreach(ic, rel->indexlist)
		{
			IndexOptInfo *index = lfirst(ic);
			Oid			proc;

			if (index->relam != amoid || index->nkeycolumns != 1 || index->indexkeys[0] != var->varattno ||
				index->indpred || index->indexprs || !index->canreturn[0])
				continue;
			proc = get_opfamily_proc(index->opfamily[0], index->opcintype[0], index->opcintype[0],
									 ANN_DISTANCE_PROC);
			if (OidIsValid(proc) && proc == get_opcode(op->opno))
			{
				*found = index;
				return operand;
			}
		}
	}
	return NULL;
}

static void
AnnSetRelPathlist(PlannerInfo *root, RelOptInfo *rel, Index rti, RangeTblEntry *rte)
{
	PathKey    *pathkey;
	IndexOptInfo *index = NULL;
	Node	   *operand;
	List	   *quals;
	Relation	indexRel;
	AnnPredicateColumns columns;
	bool		supported;
	Bitmapset  *used = NULL;
	bool		covering = true;
	int			k;
	CustomPath *path;
	int			member;

	if (previous_pathlist_hook)
		previous_pathlist_hook(root, rel, rti, rte);
	if (!ann_enable_frozen_search || rte->rtekind != RTE_RELATION || rte->inh ||
		(rel->reloptkind != RELOPT_BASEREL && rel->reloptkind != RELOPT_OTHER_MEMBER_REL) ||
		root->query_pathkeys == NIL || rel->indexlist == NIL)
		return;
	pathkey = linitial(root->query_pathkeys);
#if PG_VERSION_NUM >= 180000
	if (pathkey->pk_cmptype != COMPARE_LT || pathkey->pk_nulls_first)
		return;
#else
	if (pathkey->pk_strategy != BTLessStrategyNumber || pathkey->pk_nulls_first)
		return;
#endif
	if ((operand = DistanceOperand(root, rel, pathkey, &index)) == NULL)
		return;
	quals = extract_actual_clauses(rel->baserestrictinfo, false);
	indexRel = index_open(index->indexoid, NoLock);
	AnnPredicateColumnsInit(indexRel, &columns);
	supported = AnnPredicateSupported(quals, rel->relid, &columns);

	/* Columns the query reads: covered if all are the index's. */
	pull_varattnos((Node *) rel->reltarget->exprs, rel->relid, &used);
	pull_varattnos((Node *) quals, rel->relid, &used);
	member = -1;
	while ((member = bms_next_member(used, member)) >= 0)
	{
		AttrNumber	attno = member + FirstLowInvalidHeapAttributeNumber;
		bool		in = false;

		for (int i = 0; i < RelationGetDescr(indexRel)->natts; i++)
			in |= indexRel->rd_index->indkey.values[i] == attno;
		covering &= in;
	}
	index_close(indexRel, NoLock);
	if (!supported)
		return;

	k = root->limit_tuples > 0 ? (int) Min(root->limit_tuples, ANN_FROZEN_WORKER_MAX_K) : ANN_FROZEN_DEFAULT_K;
	path = makeNode(CustomPath);
	path->path.pathtype = T_CustomScan;
	path->path.parent = rel;
	path->path.pathtarget = rel->reltarget;
	path->path.rows = rel->rows;
	path->path.startup_cost = 10.0;
	path->path.total_cost = 10.0 + rel->rows * cpu_tuple_cost;
	path->path.pathkeys = list_make1(pathkey);
	path->methods = &FrozenPathMethods;
	path->custom_private = list_make3(list_make1_oid(index->indexoid), list_make2_int(covering, k),
									  copyObject(operand));
	add_path(rel, &path->path);
}

static Plan *
PlanFrozenPath(PlannerInfo *root, RelOptInfo *rel, CustomPath *path, List *tlist, List *clauses,
			   List *custom_plans)
{
	CustomScan *scan = makeNode(CustomScan);
	List	   *quals = extract_actual_clauses(clauses, false);

	scan->scan.plan.targetlist = tlist;
	scan->scan.plan.qual = quals;
	scan->scan.scanrelid = rel->relid;
	scan->methods = &FrozenScanMethods;
	scan->custom_private = list_make2(linitial(path->custom_private), lsecond(path->custom_private));
	scan->custom_exprs = lcons(lthird(path->custom_private), list_copy(quals));
	return &scan->scan.plan;
}

static Node *
CreateFrozenScanState(CustomScan *scan)
{
	FrozenScanState *state = palloc0(sizeof(FrozenScanState));

	NodeSetTag(state, T_CustomScanState);
	state->css.methods = &FrozenExecMethods;
	/* Rows built from index pages are virtual; heap rows are copied. */
	state->css.slotOps = &TTSOpsVirtual;
	state->indexOid = linitial_oid(linitial(scan->custom_private));
	state->covering = linitial_int(lsecond(scan->custom_private)) != 0;
	state->initialK = lsecond_int(lsecond(scan->custom_private));
	return (Node *) state;
}

static void
BeginFrozenScan(CustomScanState *node, EState *estate, int eflags)
{
	FrozenScanState *state = (FrozenScanState *) node;
	CustomScan *scan = (CustomScan *) node->ss.ps.plan;

	state->index = index_open(state->indexOid, AccessShareLock);
	state->meta = AnnFrozenGetMeta(state->index, false);
	AnnPredicateColumnsInit(state->index, &state->columns);
	state->queryState = ExecInitExpr(linitial(scan->custom_exprs), &node->ss.ps);
	state->quals = list_copy_tail(scan->custom_exprs, 1);
	state->distance = index_getprocinfo(state->index, 1, ANN_DISTANCE_PROC);
	state->vmbuf = InvalidBuffer;
	state->resultCtx = AllocSetContextCreate(estate->es_query_cxt, "Ann frozen search results",
											 ALLOCSET_DEFAULT_SIZES);
	AnnArrayReaderInit(&state->tids, state->index, &state->meta->tids, NULL);
	AnnArrayReaderInit(&state->keys, state->index, &state->meta->keys, NULL);
	AnnArrayReaderInit(&state->vectors, state->index, &state->meta->vectors, NULL);
	state->values = palloc(sizeof(AnnArrayReader) * Max(state->meta->ncolumns, 1));
	for (uint32 c = 0; c < state->meta->ncolumns; c++)
		AnnArrayReaderInit(&state->values[c], state->index, &state->meta->columns[c].values, NULL);
	state->vector = palloc(state->meta->vectors.width);
	state->workerRoute = -1;
	if (eflags & EXEC_FLAG_EXPLAIN_ONLY)
		return;
	state->fetch = table_index_fetch_begin(node->ss.ss_currentRelation);
	state->heapSlot = table_slot_create(node->ss.ss_currentRelation, NULL);
	if (AnnRemoteConfigured() || (ann_worker_socket && ann_worker_socket[0]))
		state->workerName = AnnCatalogNameFor(state->index->rd_index->indrelid, state->meta->generation);
}

static int
ResultCompare(const void *a, const void *b, void *arg)
{
	const FrozenScanState *state = arg;
	uint64		x = *(const uint64 *) a;
	uint64		y = *(const uint64 *) b;
	double		dx = state->distances[x];
	double		dy = state->distances[y];

	if (dx != dy)
		return dx < dy ? -1 : 1;
	return (state->ordinals[x] > state->ordinals[y]) - (state->ordinals[x] < state->ordinals[y]);
}

/*
 * Score every eligible row from the vector pages, in ordinal order, and sort
 * them all: the CPU route returns the complete ordering at once.
 */
static void
LocalSearch(FrozenScanState *state)
{
	AnnFrozenMetaData *meta = state->meta;
	size_t		nwords = (meta->rows + 31) / 32;
	uint32	   *words = MemoryContextAllocHuge(state->resultCtx, Max(nwords, 1) * sizeof(uint32));
	uint64		eligible;
	uint64		n = 0;
	uint64	   *order;
	uint64	   *ordinals;
	double	   *distances;
	BufferAccessStrategy strategy = GetAccessStrategy(BAS_BULKREAD);
	AnnArrayReader reader;
	BlockNumber page = InvalidBlockNumber;
	char	   *data = NULL;

	AnnFrozenEvaluateOrError(state->index, meta, state->program.data, state->program.len, words, &eligible);
	state->eligible = eligible;
	state->localRuns++;
	ordinals = MemoryContextAllocHuge(state->resultCtx, Max(eligible, 1) * sizeof(uint64));
	distances = MemoryContextAllocHuge(state->resultCtx, Max(eligible, 1) * sizeof(double));
	AnnArrayReaderInit(&reader, state->index, &meta->vectors, strategy);
	for (size_t w = 0; w < nwords; w++)
	{
		for (uint32 bits = words[w]; bits; bits &= bits - 1)
		{
			uint64		ordinal = (uint64) w * 32 + pg_rightmost_one_pos32(bits);
			BlockNumber p = ordinal / meta->vectors.perPage;

			if (p != page)
			{
				CHECK_FOR_INTERRUPTS();
				data = AnnArrayLockPage(&reader, p, NULL);
				page = p;
			}
			ordinals[n] = ordinal;
			distances[n] = state->queryIsNull ? 0 :
				DatumGetFloat8(FunctionCall2Coll(state->distance, state->index->rd_indcollation[0],
														 PointerGetDatum(data + (size_t) (ordinal % meta->vectors.perPage) *
																		 meta->vectors.width),
														 state->query));
			n++;
		}
	}
	AnnArrayReaderEnd(&reader);
	FreeAccessStrategy(strategy);
	state->ordinals = ordinals;
	state->distances = distances;
	order = MemoryContextAllocHuge(state->resultCtx, Max(n, 1) * sizeof(uint64));
	for (uint64 i = 0; i < n; i++)
		order[i] = i;
	qsort_arg(order, n, sizeof(uint64), ResultCompare, state);
	state->ordinals = MemoryContextAllocHuge(state->resultCtx, Max(n, 1) * sizeof(uint64));
	state->distances = MemoryContextAllocHuge(state->resultCtx, Max(n, 1) * sizeof(double));
	for (uint64 i = 0; i < n; i++)
	{
		state->ordinals[i] = ordinals[order[i]];
		state->distances[i] = distances[order[i]];
	}
	state->nresults = n;
	state->exhausted = true;
}

/* Operation 22: the top k in ordinals, or false to fall back to the CPU. */
static bool
WorkerSearch(FrozenScanState *state, uint64 k)
{
	Vector	   *query = DatumGetVector(state->query);
	StringInfoData payload;
	uint32		status = 0;
	char	   *reply = NULL;
	uint32		replyLength = 0;
	uint32		count;
	bool		ok;
	uint32		header[4];

	initStringInfo(&payload);
	header[0] = strlen(state->workerName);
	appendBinaryStringInfo(&payload, (char *) &header[0], 4);
	appendBinaryStringInfo(&payload, state->workerName, header[0]);
	header[0] = query->dim;
	header[1] = (uint32) k;
	header[2] = ANN_PREDICATE_ORDINALS;
	header[3] = state->program.len;
	appendBinaryStringInfo(&payload, (char *) header, sizeof(header));
	appendBinaryStringInfo(&payload, state->program.data, state->program.len);
	appendBinaryStringInfo(&payload, (char *) query->x, sizeof(float) * query->dim);
	ann_request_generation = state->meta->generation;
	PG_TRY();
	{
		ok = AnnIndexCall(ANN_OP_INDEX_SEARCH_PREDICATE, payload.data, payload.len, 0, &status, &reply,
						  &replyLength);
	}
	PG_FINALLY();
	{
		ann_request_generation = 0;
	}
	PG_END_TRY();
	state->workerCalls++;
	if (!ok || status != ANN_STATUS_OK || replyLength < 16 ||
		(count = *(uint32 *) reply) != (replyLength - 16) / 12 || (Size) count * 12 + 16 != replyLength)
	{
		/* No worker, another generation there, or no op 22: use the pages. */
		elog(DEBUG1, "ann frozen search falls back to index pages (worker status %u)", status);
		state->workerFailed = true;
		return false;
	}
	state->workerRoute = *(uint32 *) (reply + 4);
	memcpy(&state->eligible, reply + 8, sizeof(uint64));
	state->ordinals = MemoryContextAlloc(state->resultCtx, Max(count, 1) * sizeof(uint64));
	state->distances = MemoryContextAlloc(state->resultCtx, Max(count, 1) * sizeof(double));
	for (uint32 i = 0; i < count; i++)
	{
		int64		ordinal;
		float		distance;

		memcpy(&ordinal, reply + 16 + (size_t) i * 12, 8);
		memcpy(&distance, reply + 24 + (size_t) i * 12, 4);
		state->distances[i] = distance;
		state->ordinals[i] = (uint64) ordinal;
	}
	state->nresults = count;
	state->exhausted = count < k || k >= state->meta->rows;
	return true;
}

static int
OrdinalCompare(const void *a, const void *b)
{
	uint64		x = *(const uint64 *) a;
	uint64		y = *(const uint64 *) b;

	return (x > y) - (x < y);
}

/* (Re)fill results: first k, then four times as many as asked before. */
static void
Fill(FrozenScanState *state)
{
	ExprContext *econtext = state->css.ss.ps.ps_ExprContext;
	uint64		k;
	MemoryContext oldCtx;

	if (!state->started)
	{
		state->started = true;
		initStringInfo(&state->program);
		AnnPredicateBuild(state->quals, ((Scan *) state->css.ss.ps.plan)->scanrelid, &state->columns,
						  &state->css.ss.ps, econtext, &state->program);
		oldCtx = MemoryContextSwitchTo(state->resultCtx);
		state->query = ExecEvalExprSwitchContext(state->queryState, econtext, &state->queryIsNull);
		if (!state->queryIsNull)
			state->query = PointerGetDatum(PG_DETOAST_DATUM_COPY(state->query));
		MemoryContextSwitchTo(oldCtx);
	}
	k = state->requested == 0 ? (uint64) state->initialK : state->requested * 4;
	state->requested = k;

	/* Remember what was returned, so a refill skips it. */
	if (state->nresults > 0)
	{
		state->returned = state->returned ?
			repalloc_huge(state->returned, (state->nreturned + state->next) * sizeof(uint64)) :
			MemoryContextAllocHuge(state->resultCtx, Max(state->next, 1) * sizeof(uint64));
		memcpy(state->returned + state->nreturned, state->ordinals, state->next * sizeof(uint64));
		state->nreturned += state->next;
		qsort(state->returned, state->nreturned, sizeof(uint64), OrdinalCompare);
	}
	state->nresults = 0;
	state->next = 0;
	if (state->workerName && !state->workerFailed && !state->queryIsNull && k <= ANN_FROZEN_WORKER_MAX_K &&
		WorkerSearch(state, k))
		return;
	LocalSearch(state);
}

static bool
AlreadyReturned(FrozenScanState *state, uint64 ordinal)
{
	return state->nreturned > 0 &&
		bsearch(&ordinal, state->returned, state->nreturned, sizeof(uint64), OrdinalCompare) != NULL;
}

/* A row from the index pages: every column the query reads is there. */
static void
StoreFromPages(FrozenScanState *state, TupleTableSlot *slot, uint64 ordinal)
{
	Relation	index = state->index;
	TupleDesc	desc = slot->tts_tupleDescriptor;
	int64		key;

	ExecClearTuple(slot);
	memset(slot->tts_isnull, true, sizeof(bool) * desc->natts);
	AnnArrayRead(&state->vectors, ordinal, state->vector);
	AnnArrayRead(&state->keys, ordinal, &key);
	for (int i = 0; i < RelationGetDescr(index)->natts; i++)
	{
		AttrNumber	attno = index->rd_index->indkey.values[i];
		Oid			type = TupleDescAttr(desc, attno - 1)->atttypid;
		Datum		value;
		bool		isnull = false;

		if (i == 0)
			value = PointerGetDatum(state->vector);
		else if (i == 1)
			value = type == INT2OID ? Int16GetDatum((int16) key) :
				type == INT4OID ? Int32GetDatum((int32) key) : Int64GetDatum(key);
		else
		{
			int32		v;

			AnnArrayRead(&state->values[i - 2], ordinal, &v);
			isnull = v == ANN_FROZEN_NULL;
			value = type == INT2OID ? Int16GetDatum((int16) v) : Int32GetDatum(v);
		}
		slot->tts_values[attno - 1] = value;
		slot->tts_isnull[attno - 1] = isnull;
	}
	ExecStoreVirtualTuple(slot);
}

static TupleTableSlot *
FrozenNext(ScanState *node)
{
	FrozenScanState *state = (FrozenScanState *) node;
	TupleTableSlot *slot = node->ss_ScanTupleSlot;

	for (;;)
	{
		uint64		ordinal;
		ItemPointerData tid;

		if (!state->started || (state->next >= state->nresults && !state->exhausted))
		{
			Fill(state);
			continue;
		}
		if (state->next >= state->nresults)
			return ExecClearTuple(slot);
		ordinal = state->ordinals[state->next++];
		if (ordinal >= state->meta->rows || AlreadyReturned(state, ordinal))
			continue;
		AnnArrayRead(&state->tids, ordinal, &tid);
		if (!ItemPointerIsValid(&tid))
			continue;
		if (state->covering &&
			VM_ALL_VISIBLE(node->ss_currentRelation, ItemPointerGetBlockNumber(&tid), &state->vmbuf))
		{
			StoreFromPages(state, slot, ordinal);
			slot->tts_tid = tid;
			return slot;
		}
		{
			bool		callAgain = false;
			bool		allDead = false;

			state->heapFetches++;
			if (table_index_fetch_tuple(state->fetch, &tid, node->ps.state->es_snapshot, state->heapSlot,
										&callAgain, &allDead))
			{
				ExecCopySlot(slot, state->heapSlot);
				slot->tts_tid = tid;
				return slot;
			}
		}
	}
}

static bool
FrozenRecheck(ScanState *node, TupleTableSlot *slot)
{
	return true;
}

static TupleTableSlot *
ExecFrozenScan(CustomScanState *node)
{
	return ExecScan(&node->ss, FrozenNext, FrozenRecheck);
}

static void
ResetResults(FrozenScanState *state)
{
	MemoryContextReset(state->resultCtx);
	state->started = false;
	state->ordinals = NULL;
	state->distances = NULL;
	state->returned = NULL;
	state->nresults = state->next = state->requested = state->nreturned = 0;
	state->exhausted = false;
}

static void
ReScanFrozenScan(CustomScanState *node)
{
	ResetResults((FrozenScanState *) node);
	ExecScanReScan(&node->ss);
}

static void
EndFrozenScan(CustomScanState *node)
{
	FrozenScanState *state = (FrozenScanState *) node;

	AnnArrayReaderEnd(&state->tids);
	AnnArrayReaderEnd(&state->keys);
	AnnArrayReaderEnd(&state->vectors);
	for (uint32 c = 0; c < state->meta->ncolumns; c++)
		AnnArrayReaderEnd(&state->values[c]);
	if (state->fetch)
		table_index_fetch_end(state->fetch);
	if (state->heapSlot)
		ExecDropSingleTupleTableSlot(state->heapSlot);
	if (BufferIsValid(state->vmbuf))
		ReleaseBuffer(state->vmbuf);
	index_close(state->index, AccessShareLock);
	MemoryContextDelete(state->resultCtx);
}

static void
ExplainFrozenScan(CustomScanState *node, List *ancestors, ExplainState *es)
{
	FrozenScanState *state = (FrozenScanState *) node;

	ExplainPropertyText("Frozen Index", RelationGetRelationName(state->index), es);
	ExplainPropertyText("Search Route", state->workerName && !state->workerFailed ?
						psprintf("frozen index predicates on GPU worker \"%s\"", state->workerName) :
						es->analyze || state->workerName ? "frozen index predicates from index pages" :
						"frozen index predicates", es);
	ExplainPropertyText("Columns", state->covering ? "index pages when all-visible" : "heap", es);
	if (es->analyze)
	{
		ExplainPropertyInteger("Filter Rows", NULL, state->eligible, es);
		ExplainPropertyInteger("Heap Fetches", NULL, state->heapFetches, es);
		ExplainPropertyInteger("Worker Calls", NULL, state->workerCalls, es);
		ExplainPropertyInteger("Local Evaluations", NULL, state->localRuns, es);
	}
}

void
AnnFrozenScanInit(void)
{
	DefineCustomBoolVariable("ann.enable_frozen_search",
							 "Answer ordered searches of tables with frozen ann indexes from the index and the GPU worker.",
							 NULL, &ann_enable_frozen_search, true, PGC_USERSET, 0, NULL, NULL, NULL);
	RegisterCustomScanMethods(&FrozenScanMethods);
	previous_pathlist_hook = set_rel_pathlist_hook;
	set_rel_pathlist_hook = AnnSetRelPathlist;
}
