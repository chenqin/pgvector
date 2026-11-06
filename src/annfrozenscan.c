#include "postgres.h"

#include <float.h>
#include <math.h>

#include "access/relscan.h"
#include "access/tableam.h"
#include "access/visibilitymap.h"
#include "ann.h"
#include "ann/filter.h"
#include "ann/cost.h"
#include "catalog/pg_type_d.h"
#include "commands/defrem.h"
#include "commands/extension.h"
#include "executor/spi.h"
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
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/hsearch.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/timestamp.h"

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
#define ANN_FROZEN_VM_BUFFERS 32

bool		ann_enable_frozen_search = true;
static double ann_frozen_setup_cost = 1000.0;
static char *ann_frozen_cost_profile = NULL;
static const AnnCostProfile *frozenCostProfile = NULL;
static int	ann_frozen_prefetch_distance = 4;
static int	ann_frozen_materialize_batch = 64;
static set_rel_pathlist_hook_type previous_pathlist_hook = NULL;

typedef struct FrozenScanState
{
	CustomScanState css;
	Oid			indexOid;
	bool		covering;		/* every used column is in the index */
	int			vectorMode;		/* ANN_FROZEN_VECTORS_* */
	int			initialK;
	Relation	index;
	AnnFrozenMetaData *meta;
	AnnPredicateColumns columns;
	ExprState  *queryState;
	List	   *quals;			/* clauses the program is built from */
	FmgrInfo   *distance;
	char	   *workerName;		/* loaded export of this generation, or NULL */
	bool		workerFailed;
	bool		workerShort;	/* the worker returned fewer than asked */
	IndexFetchTableData *fetch;
	TupleTableSlot *heapSlot;	/* heap fetches, copied to the scan slot */
	Buffer		vmbufs[ANN_FROZEN_VM_BUFFERS];
	AnnArrayReader tids;
	AnnArrayReader keys;
	AnnArrayReader vectors;
	AnnArrayReader *values;
	Vector	   *vector;			/* the returned row's vector */
	MemoryContext resultCtx;
	MemoryContext batchCtx;
	bool	   *batchValid;
	ItemPointerData *batchTids;
	uint64		batchFirst;
	uint64		batchEnd;
	bool		batchUnavailable;

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
	const uint64 *tombstones;	/* deleted rows' ordinals, sorted */
	int			ntombstones;

	/* EXPLAIN ANALYZE counters */
	uint64		eligible;
	uint64		heapFetches;
	uint64		workerCalls;
	uint64		localRuns;
	uint64		materializedRows;
	uint64		mappedTidRows;
	int			workerRoute;
	List	   *costEstimate;
} FrozenScanState;

static char *WorkerName(Oid relid, uint64 generation, bool refresh);
static void LoadTombstones(FrozenScanState *state);
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

/*
 * The operand of "column <-> operand" ordering this rel, if the column has a
 * frozen index (whose key, the first INCLUDE column, index-only scans return;
 * the vector may not be returnable when the pages hold halves or none).
 */
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
				index->indpred || index->indexprs || index->ncolumns < 2 || !index->canreturn[1])
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

/* A trusted deployment profile is loaded at SET time, never by a planner RPC. */
static bool
CheckCostProfile(char **newval, void **extra, GucSource source)
{
	char		error[160];
	AnnCostProfile *profile;

	if ((*newval)[0] == '\0')
		return true;
	profile = AnnCostLoad(*newval, error, sizeof(error));
	if (profile == NULL)
	{
		GUC_check_errdetail("%s", error);
		return false;
	}
#if PG_VERSION_NUM >= 170000
	*extra = guc_malloc(ERROR, sizeof(*profile));
	memcpy(*extra, profile, sizeof(*profile));
	free(profile);
#else
	*extra = profile;
#endif
	return true;
}

static void
AssignCostProfile(const char *newval, void *extra)
{
	frozenCostProfile = extra;
}

/* Include the sort/limit that upper planning has not added to base paths yet. */
static Cost
LocalLimitCost(PlannerInfo *root, RelOptInfo *rel, int k)
{
	Cost		best = DBL_MAX;
	ListCell   *lc;
	double		fraction = Min(1.0, k / Max(rel->rows, 1.0));

	foreach(lc, rel->pathlist)
	{
		Path	   *candidate = lfirst(lc);
		Path		sorted;
		Cost		cost;

		if (candidate->param_info)
			continue;
		if (!pathkeys_contained_in(root->query_pathkeys, candidate->pathkeys))
		{
			cost_sort(&sorted, root, root->query_pathkeys, candidate->total_cost,
					  rel->rows, rel->reltarget->width, 0, work_mem, k);
			candidate = &sorted;
		}
		cost = candidate->startup_cost + fraction * (candidate->total_cost - candidate->startup_cost);
		if (cost > 0 && cost < best)
			best = cost;
	}
	return best;
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
	QualCost	recheck;
	AnnFrozenMetaData *meta;
	int			dimensions;
	List	   *estimate = NIL;
	double		latencies[ANN_COST_MODES];
	Cost		localCost;
	int			mode;

	if (previous_pathlist_hook)
		previous_pathlist_hook(root, rel, rti, rte);
	if (!ann_enable_frozen_search || rte->rtekind != RTE_RELATION || rte->inh || rte->tablesample ||
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
	meta = AnnFrozenGetMeta(indexRel, false);
	dimensions = TupleDescAttr(RelationGetDescr(indexRel), 0)->atttypmod;
	supported = AnnPredicateSupported(quals, rel->relid, &columns);

	/* Columns the query reads: covered if all are the index's. */
	pull_varattnos((Node *) rel->reltarget->exprs, rel->relid, &used);
	pull_varattnos((Node *) quals, rel->relid, &used);
	member = -1;
	while ((member = bms_next_member(used, member)) >= 0)
	{
		AttrNumber	attno = member + FirstLowInvalidHeapAttributeNumber;
		bool		in = false;

		/* Half or no vector pages: the vector itself comes from the heap. */
		for (int i = AnnFrozenVectorMode(indexRel, meta) ==
			 ANN_FROZEN_VECTORS_FLOAT ? 0 : 1; i < RelationGetDescr(indexRel)->natts; i++)
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

	/*
	 * Cost the route so the planner can choose it or an ordinary plan. The
	 * call (the GPU round trip, or evaluating the program from the pages)
	 * comes before the first row: a fixed setup cost plus scoring every row
	 * the predicate selects. Each row returned then costs its fetch (from the
	 * index pages when covering, otherwise a heap page) and the qual recheck.
	 * Keeping rows = rel->rows lets a LIMIT charge only the rows it takes.
	 */
	cost_qual_eval(&recheck, quals, root);
	path->path.startup_cost = ann_frozen_setup_cost + recheck.startup +
		rel->rows * cpu_operator_cost;
	path->path.total_cost = path->path.startup_cost +
		rel->rows * (cpu_tuple_cost + recheck.per_tuple +
					 (covering ? cpu_index_tuple_cost : random_page_cost));

	/*
	 * Warm measurements put worker time into this query's local cost units.
	 * Keep standard row costs, and refuse profiles outside their measured
	 * generation/dimensions/domain or for uncovered, joined/unknown-LIMIT
	 * SQL.
	 */
	if (frozenCostProfile && covering && root->limit_tuples > 0 &&
		bms_num_members(root->all_baserels) == 1 &&
		(AnnRemoteConfigured() || (ann_worker_socket && ann_worker_socket[0])) &&
		AnnCostEstimate(frozenCostProfile, meta->rows, dimensions, meta->generation,
						rel->rows, k, 1, latencies) && latencies[ANN_COST_LOCAL] > 0 &&
		(mode = AnnCostBest(latencies, 1, 1, 1)) >= 0 &&
		(localCost = LocalLimitCost(root, rel, k)) < DBL_MAX &&
		WorkerName(rte->relid, meta->generation, false) != NULL)
	{
		double		predicted = latencies[mode] + frozenCostProfile->setup_us +
			Min((double) k, rel->rows) * frozenCostProfile->tuple_us;

		path->path.startup_cost = recheck.startup + localCost * predicted / latencies[ANN_COST_LOCAL];
		path->path.total_cost = path->path.startup_cost +
			rel->rows * (cpu_tuple_cost + recheck.per_tuple + cpu_index_tuple_cost);
		estimate = list_make5(makeString(pstrdup(AnnCostName(mode))),
							  makeFloat(psprintf("%.17g", predicted)),
							  makeFloat(psprintf("%.17g", latencies[ANN_COST_LOCAL])),
							  makeFloat(psprintf("%.17g", rel->rows)),
							  makeFloat(psprintf("%.17g", frozenCostProfile->min_recall)));
	}
	path->path.pathkeys = list_make1(pathkey);
	path->methods = &FrozenPathMethods;
	path->custom_private = list_make4(list_make1_oid(index->indexoid), list_make2_int(covering, k),
									  copyObject(operand), estimate);
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
	scan->custom_private = list_make3(linitial(path->custom_private), lsecond(path->custom_private),
									  lfourth(path->custom_private));
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
	state->costEstimate = lthird(scan->custom_private);
	return (Node *) state;
}

static void
BeginFrozenScan(CustomScanState *node, EState *estate, int eflags)
{
	FrozenScanState *state = (FrozenScanState *) node;
	CustomScan *scan = (CustomScan *) node->ss.ps.plan;

	state->index = index_open(state->indexOid, AccessShareLock);
	if ((state->meta = AnnFrozenMeta(state->index)) == NULL)
		elog(ERROR, "\"%s\" is not a frozen ann index", RelationGetRelationName(state->index));
	AnnPredicateColumnsInit(state->index, &state->columns);
	state->queryState = ExecInitExpr(linitial(scan->custom_exprs), &node->ss.ps);
	state->quals = list_copy_tail(scan->custom_exprs, 1);
	state->distance = index_getprocinfo(state->index, 1, ANN_DISTANCE_PROC);
	for (int i = 0; i < ANN_FROZEN_VM_BUFFERS; i++)
		state->vmbufs[i] = InvalidBuffer;
	state->resultCtx = AllocSetContextCreate(estate->es_query_cxt, "Ann frozen search results",
											 ALLOCSET_DEFAULT_SIZES);
	state->batchCtx = AllocSetContextCreate(estate->es_query_cxt, "Ann frozen materialization batch",
											ALLOCSET_DEFAULT_SIZES);
	AnnArrayReaderInit(&state->tids, state->index, &state->meta->tids, NULL);
	AnnArrayReaderInit(&state->keys, state->index, &state->meta->keys, NULL);
	AnnArrayReaderInit(&state->vectors, state->index, &state->meta->vectors, NULL);
	state->values = palloc(sizeof(AnnArrayReader) * Max(state->meta->ncolumns, 1));
	for (uint32 c = 0; c < state->meta->ncolumns; c++)
		AnnArrayReaderInit(&state->values[c], state->index, &state->meta->columns[c].values, NULL);
	state->vector = palloc(state->meta->vectors.width);
	state->vectorMode = AnnFrozenVectorMode(state->index, state->meta);
	state->workerRoute = -1;
	if (eflags & EXEC_FLAG_EXPLAIN_ONLY)
		return;
	LoadTombstones(state);
	state->fetch = table_index_fetch_begin(node->ss.ss_currentRelation);
	state->heapSlot = table_slot_create(node->ss.ss_currentRelation, NULL);
	if (AnnRemoteConfigured() || (ann_worker_socket && ann_worker_socket[0]))
		state->workerName = WorkerName(state->index->rd_index->indrelid, state->meta->generation, false);
}

/*
 * Loaded exports by (table, generation), cached per backend: read from the
 * catalog on first use, again after a failed call, and, when none was
 * loaded, at most once a second.
 */
typedef struct WorkerEntry
{
	Oid			relid;
	uint64		generation;
	char		name[NAMEDATALEN];
	TimestampTz checked;
} WorkerEntry;

static HTAB *workerNames = NULL;

/*
 * Tombstones (ann_frozen_tombstone) of each frozen index's table, as this
 * generation's ordinals, cached per backend: read again after a second, or
 * at once after ann_frozen_delete in this backend.
 */
typedef struct TombstoneEntry
{
	Oid			indexOid;
	uint64		generation;
	uint64		epoch;
	TimestampTz checked;
	uint64	   *ordinals;
	int			count;
} TombstoneEntry;

static HTAB *tombstoneCache = NULL;
static uint64 tombstoneEpoch = 1;

void
AnnFrozenTombstonesChanged(void)
{
	tombstoneEpoch++;
}

static int	OrdinalCompare(const void *a, const void *b);

/* A cache refresh must not reclaim another active scan's tombstones. */
static void
CopyTombstones(FrozenScanState *state, const TombstoneEntry *entry)
{
	state->ntombstones = entry->count;
	if (entry->count > 0)
	{
		uint64	   *copy = MemoryContextAlloc(state->css.ss.ps.state->es_query_cxt,
											  sizeof(uint64) * entry->count);

		memcpy(copy, entry->ordinals, sizeof(uint64) * entry->count);
		state->tombstones = copy;
	}
}

static void
LoadTombstones(FrozenScanState *state)
{
	Oid			indexOid = RelationGetRelid(state->index);
	TombstoneEntry *entry;
	bool		found;
	TimestampTz now = GetCurrentTimestamp();
	Oid			extension = get_extension_oid("vector", true);
	Oid			schema = OidIsValid(extension) ? get_extension_schema(extension) : InvalidOid;
	Oid			types[1] = {OIDOID};
	Datum		values[1] = {ObjectIdGetDatum(state->index->rd_index->indrelid)};
	uint64	   *ordinals;
	int			count = 0;

	if (tombstoneCache == NULL)
	{
		HASHCTL		ctl;

		memset(&ctl, 0, sizeof(ctl));
		ctl.keysize = sizeof(Oid);
		ctl.entrysize = sizeof(TombstoneEntry);
		tombstoneCache = hash_create("ann frozen tombstones", 16, &ctl, HASH_ELEM | HASH_BLOBS);
	}
	entry = hash_search(tombstoneCache, &indexOid, HASH_ENTER, &found);
	if (!found)
	{
		entry->ordinals = NULL;
		entry->count = 0;
		entry->generation = 0;
		entry->epoch = 0;
		entry->checked = 0;
	}
	if (found && entry->generation == state->meta->generation && entry->epoch == tombstoneEpoch &&
		!TimestampDifferenceExceeds(entry->checked, now, 1000))
	{
		CopyTombstones(state, entry);
		return;
	}
	if (!OidIsValid(schema) || !OidIsValid(get_relname_relid("ann_frozen_tombstone", schema)))
		return;
	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");
	if (SPI_execute_with_args(psprintf("SELECT key FROM %s.ann_frozen_tombstone WHERE source = $1",
									   quote_identifier(get_namespace_name(schema))),
							  1, types, values, NULL, true, 0) != SPI_OK_SELECT)
		elog(ERROR, "could not read the ann tombstones");
	ordinals = MemoryContextAlloc(TopMemoryContext, sizeof(uint64) * Max(SPI_processed, 1));
	for (uint64 i = 0; i < SPI_processed; i++)
	{
		bool		isnull;
		int64		key = DatumGetInt64(SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1, &isnull));
		uint64		low = 0;
		uint64		high = state->meta->rows;

		/* the key's ordinal: keys are sorted */
		while (low < high)
		{
			uint64		middle = low + (high - low) / 2;
			int64		probe;

			AnnArrayRead(&state->keys, middle, &probe);
			if (probe < key)
				low = middle + 1;
			else
				high = middle;
		}
		if (low < state->meta->rows)
		{
			int64		probe;

			AnnArrayRead(&state->keys, low, &probe);
			if (probe == key)
				ordinals[count++] = low;
		}
	}
	SPI_finish();
	qsort(ordinals, count, sizeof(uint64), OrdinalCompare);
	/* Publish only a complete refresh; an ERROR must not cache an empty set. */
	if (entry->ordinals)
		pfree(entry->ordinals);
	entry->ordinals = ordinals;
	entry->count = count;
	entry->generation = state->meta->generation;
	entry->epoch = tombstoneEpoch;
	entry->checked = now;
	CopyTombstones(state, entry);
}

static char *
WorkerName(Oid relid, uint64 generation, bool refresh)
{
	struct
	{
		Oid			relid;
		uint64		generation;
	}			key;
	WorkerEntry *entry;
	bool		found;
	TimestampTz now = GetCurrentTimestamp();

	if (workerNames == NULL)
	{
		HASHCTL		ctl;

		ctl.keysize = sizeof(key);
		ctl.entrysize = sizeof(WorkerEntry);
		workerNames = hash_create("ann frozen worker names", 16, &ctl, HASH_ELEM | HASH_BLOBS);
	}
	memset(&key, 0, sizeof(key));
	key.relid = relid;
	key.generation = generation;
	entry = hash_search(workerNames, &key, HASH_ENTER, &found);
	if (!found || refresh ||
		(entry->name[0] == '\0' && TimestampDifferenceExceeds(entry->checked, now, 1000)))
	{
		char	   *name = AnnCatalogNameFor(relid, generation);

		strlcpy(entry->name, name ? name : "", sizeof(entry->name));
		entry->checked = now;
	}
	return entry->name[0] ? pstrdup(entry->name) : NULL;
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

	AnnFrozenEvaluateOrError(state->index, meta, state->program.data, state->program.len, words, &eligible,
							 false);
	state->eligible = eligible;
	state->localRuns++;
	ordinals = MemoryContextAllocHuge(state->resultCtx, Max(eligible, 1) * sizeof(uint64));
	distances = MemoryContextAllocHuge(state->resultCtx, Max(eligible, 1) * sizeof(double));
	if (state->vectorMode != ANN_FROZEN_VECTORS_FLOAT)
	{
		/* Half pages widened, or the heap's vectors by TID. */
		AnnFrozenVectorReader vr;

		AnnFrozenVectorReaderInit(&vr, state->index, meta, strategy, false);
		for (size_t w = 0; w < nwords; w++)
			for (uint32 bits = words[w]; bits; bits &= bits - 1)
			{
				uint64		ordinal = (uint64) w * 32 + pg_rightmost_one_pos32(bits);

				CHECK_FOR_INTERRUPTS();
				ordinals[n] = ordinal;
				distances[n] = state->queryIsNull ? 0 :
					DatumGetFloat8(FunctionCall2Coll(state->distance, state->index->rd_indcollation[0],
													 PointerGetDatum(AnnFrozenVectorRead(&vr, ordinal)),
													 state->query));
				n++;
			}
		AnnFrozenVectorReaderEnd(&vr);
		nwords = 0;				/* done: skip the page loop */
	}
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
		(void) WorkerName(state->index->rd_index->indrelid, state->meta->generation, true);
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

	/*
	 * Done once every eligible row came back. An approximate route can return
	 * fewer than k while more rows qualify: the rest then come from the
	 * pages, exactly, skipping what was returned.
	 */
	state->exhausted = count >= state->eligible;
	if (!state->exhausted && count < k)
		state->workerShort = true;
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

	/* Everything a fill allocates goes away on rescan (resultCtx is reset). */
	oldCtx = MemoryContextSwitchTo(state->resultCtx);
	if (!state->started)
	{
		state->started = true;
		initStringInfo(&state->program);
		AnnPredicateBuild(state->quals, ((Scan *) state->css.ss.ps.plan)->scanrelid, &state->columns,
						  &state->css.ss.ps, econtext, &state->program);
		if (!AnnFilterCheck((const unsigned char *) state->program.data, state->program.len))
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("the predicate is too large for a frozen ann search"),
					 errhint("SET ann.enable_frozen_search = off to plan this query without it.")));
		state->query = ExecEvalExprSwitchContext(state->queryState, econtext, &state->queryIsNull);
		if (!state->queryIsNull)
			state->query = PointerGetDatum(PG_DETOAST_DATUM_COPY(state->query));
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
	state->batchFirst = state->batchEnd = 0;
	if (!(state->workerName && !state->workerFailed && !state->workerShort && !state->queryIsNull &&
		  k <= ANN_FROZEN_WORKER_MAX_K && WorkerSearch(state, k)))
		LocalSearch(state);
	MemoryContextSwitchTo(oldCtx);
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
	if (state->vectorMode == ANN_FROZEN_VECTORS_FLOAT)
		AnnArrayRead(&state->vectors, ordinal, state->vector);
	AnnArrayRead(&state->keys, ordinal, &key);
	for (int i = 0; i < RelationGetDescr(index)->natts; i++)
	{
		AttrNumber	attno = index->rd_index->indkey.values[i];
		Oid			type = TupleDescAttr(desc, attno - 1)->atttypid;
		Datum		value;
		bool		isnull = false;

		if (i == 0)
		{
			/* not read by the query unless the pages hold whole vectors */
			if (state->vectorMode != ANN_FROZEN_VECTORS_FLOAT)
				continue;
			value = PointerGetDatum(state->vector);
		}
		else if (i == 1)
			value = type == INT2OID ? Int16GetDatum((int16) key) :
				type == INT4OID ? Int32GetDatum((int32) key) : Int64GetDatum(key);
		else
		{
			int32		v;

			AnnArrayRead(&state->values[i - 2], ordinal, &v);
			isnull = v == ANN_FROZEN_NULL;
			if (!AnnFrozenTextType(type))
				value = AnnFrozenAttributeDatum(v, type);
			else if (isnull)
				value = (Datum) 0;
			else
			{
				/* per-tuple memory: ExecScan resets it before the next row */
				MemoryContext oldCtx = MemoryContextSwitchTo(state->css.ss.ps.ps_ExprContext->ecxt_per_tuple_memory);

				value = AnnFrozenTextDatum(index, state->meta, i - 2, v);
				MemoryContextSwitchTo(oldCtx);
			}
		}
		slot->tts_values[attno - 1] = value;
		slot->tts_isnull[attno - 1] = isnull;
	}
	ExecStoreVirtualTuple(slot);
}

/* Issue cache hints in result order without reading mutable TIDs or VM bits. */
static void
PrefetchResult(FrozenScanState *state, uint64 position)
{
	uint64		ordinal = state->ordinals[position];

	if (ordinal >= state->meta->rows)
		return;
	if (state->vectorMode == ANN_FROZEN_VECTORS_FLOAT)
		AnnArrayPrefetch(&state->vectors, ordinal);
	AnnArrayPrefetch(&state->keys, ordinal);
	for (uint32 c = 0; c < state->meta->ncolumns; c++)
		AnnArrayPrefetch(&state->values[c], ordinal);
}

/*
 * Validate a bounded result window using unchanged mapped TIDs. Cache only
 * all-visible proofs; the statement snapshot protects these rows until emit.
 * On old or physically vacuumed indexes, use the ordinary live-buffer path.
 */
static void
MaterializeBatch(FrozenScanState *state)
{
	ScanState  *node = &state->css.ss;
	TupleTableSlot *slot = node->ss_ScanTupleSlot;
	MemoryContext old;
	uint64		n;

	ExecClearTuple(slot);
	MemoryContextReset(state->batchCtx);
	old = MemoryContextSwitchTo(state->batchCtx);
	state->batchFirst = state->next;
	n = Min((uint64) ann_frozen_materialize_batch, state->nresults - state->next);
	state->batchEnd = state->next + n;
	state->batchValid = palloc0(n * sizeof(bool));
	state->batchTids = palloc(n * sizeof(ItemPointerData));
	if (AnnFrozenVisibleTids(state->index, node->ss_currentRelation, state->meta,
							 state->ordinals + state->next, n, state->batchTids,
							 state->batchValid, state->vmbufs, ANN_FROZEN_VM_BUFFERS))
	{
		for (uint64 i = 0; i < n; i++)
		{
			uint64		ordinal = state->ordinals[state->next + i];

			if (AlreadyReturned(state, ordinal) ||
				(state->ntombstones && bsearch(&ordinal, state->tombstones, state->ntombstones,
											   sizeof(uint64), OrdinalCompare)))
				state->batchValid[i] = false;
			if (state->batchValid[i])
				state->mappedTidRows++;
		}
		MemoryContextSwitchTo(old);
		return;
	}
	state->batchUnavailable = true;
	MemoryContextSwitchTo(old);
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
		bool		visible = false;

		if (!state->started || (state->next >= state->nresults && !state->exhausted))
		{
			Fill(state);
			continue;
		}
		if (state->next >= state->nresults)
			return ExecClearTuple(slot);
		if (state->covering && ann_frozen_prefetch_distance > 0)
		{
			uint64		distance = (uint64) ann_frozen_prefetch_distance;

			if (state->next == 0)
				for (uint64 i = 0; i < Min(distance, state->nresults); i++)
					PrefetchResult(state, i);
			if (state->nresults - state->next > distance)
				PrefetchResult(state, state->next + distance);
		}
		if (state->covering && ann_frozen_materialize_batch > 0 && !state->batchUnavailable)
		{
			uint64		position;

			if (state->next < state->batchFirst || state->next >= state->batchEnd)
				MaterializeBatch(state);
			position = state->next - state->batchFirst;
			if (state->batchValid[position])
			{
				StoreFromPages(state, slot, state->ordinals[state->next]);
				state->materializedRows++;
				slot->tts_tid = state->batchTids[position];
				state->next++;
				return slot;
			}
		}
		ordinal = state->ordinals[state->next++];
		if (ordinal >= state->meta->rows || AlreadyReturned(state, ordinal))
			continue;
		if (state->ntombstones &&
			bsearch(&ordinal, state->tombstones, state->ntombstones, sizeof(uint64), OrdinalCompare))
			continue;			/* deleted (ann_frozen_delete) */
		if (state->covering)
			AnnFrozenReadTid(&state->tids, node->ss_currentRelation, ordinal, &tid, &visible,
							 state->vmbufs, ANN_FROZEN_VM_BUFFERS);
		else
			AnnArrayRead(&state->tids, ordinal, &tid);
		if (!ItemPointerIsValid(&tid))
			continue;
		if (visible)
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
	ExecClearTuple(state->css.ss.ss_ScanTupleSlot);
	MemoryContextReset(state->batchCtx);
	state->batchValid = NULL;
	state->batchTids = NULL;
	state->batchFirst = state->batchEnd = 0;
	MemoryContextReset(state->resultCtx);
	state->started = false;
	state->ordinals = NULL;
	state->distances = NULL;
	state->returned = NULL;
	state->nresults = state->next = state->requested = state->nreturned = 0;
	state->exhausted = false;
	state->workerShort = false;
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
	for (int i = 0; i < ANN_FROZEN_VM_BUFFERS; i++)
		if (BufferIsValid(state->vmbufs[i]))
			ReleaseBuffer(state->vmbufs[i]);
	index_close(state->index, AccessShareLock);
	MemoryContextDelete(state->resultCtx);
	MemoryContextDelete(state->batchCtx);
}

static void
ExplainFrozenScan(CustomScanState *node, List *ancestors, ExplainState *es)
{
	FrozenScanState *state = (FrozenScanState *) node;

	ExplainPropertyText("Frozen Index", RelationGetRelationName(state->index), es);
	if (state->costEstimate && es->costs)
	{
		ExplainPropertyText("Estimated GPU Path", strVal(linitial(state->costEstimate)), es);
		ExplainPropertyFloat("Estimated Search Time", "ms", floatVal(lsecond(state->costEstimate)) / 1000, 3, es);
		ExplainPropertyFloat("Estimated Local Time", "ms", floatVal(lthird(state->costEstimate)) / 1000, 3, es);
		ExplainPropertyFloat("Estimated Matching Rows", NULL, floatVal(lfourth(state->costEstimate)), 0, es);
		ExplainPropertyFloat("Calibration Recall Target", "%", floatVal(list_nth(state->costEstimate, 4)) * 100, 1, es);
	}

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
		ExplainPropertyInteger("Materialized Rows", NULL, state->materializedRows, es);
		ExplainPropertyInteger("Mapped TID Rows", NULL, state->mappedTidRows, es);
	}
}

void
AnnFrozenScanInit(void)
{
	DefineCustomStringVariable("ann.frozen_cost_profile",
							   "Measured worker and local SQL latencies for frozen planning.",
							   "Empty uses legacy costing; profiles are generation- and dimension-specific.",
							   &ann_frozen_cost_profile, "", PGC_SUSET, 0,
							   CheckCostProfile, AssignCostProfile, NULL);
	DefineCustomIntVariable("ann.frozen_materialize_batch",
							"Validate covering result visibility in batches (0 disables).",
							NULL, &ann_frozen_materialize_batch, 64, 0, 1024, PGC_USERSET,
							0, NULL, NULL, NULL);
	DefineCustomIntVariable("ann.frozen_prefetch_distance",
							"Result rows ahead to prefetch immutable mapped columns (0 disables).",
							NULL, &ann_frozen_prefetch_distance, 4, 0, 64, PGC_USERSET,
							0, NULL, NULL, NULL);
	DefineCustomBoolVariable("ann.enable_frozen_search",
							 "Answer ordered searches of tables with frozen ann indexes from the index and the GPU worker.",
							 NULL, &ann_enable_frozen_search, true, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("ann.frozen_setup_cost",
							 "Sets the planner's estimate of the cost of one frozen search call.",
							 "The GPU worker round trip or the evaluation from the pages, before the first row.",
							 &ann_frozen_setup_cost, 1000.0, 0.0, DBL_MAX, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("ann.frozen_mmap",
							 "Read frozen ann index pages through read-only mappings of the index files.",
							 NULL, &ann_frozen_mmap, true, PGC_USERSET, 0, NULL, NULL, NULL);
	RegisterCustomScanMethods(&FrozenScanMethods);
	previous_pathlist_hook = set_rel_pathlist_hook;
	set_rel_pathlist_hook = AnnSetRelPathlist;
}
