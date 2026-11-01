#include "postgres.h"

#include <math.h>

#include "access/amvalidate.h"
#include "access/reloptions.h"
#include "ann.h"
#include "catalog/pg_type_d.h"
#include "commands/vacuum.h"
#include "optimizer/optimizer.h"
#include "utils/float.h"
#include "utils/lsyscache.h"
#include "utils/selfuncs.h"

/*
 * The CPU staging implementation reads and sorts the complete index.
 */
static void
anncostestimate(PlannerInfo *root, IndexPath *path, double loop_count,
				Cost *indexStartupCost, Cost *indexTotalCost,
				Selectivity *indexSelectivity, double *indexCorrelation,
				double *indexPages)
{
	GenericCosts costs = {0};
	double		tuples = path->indexinfo->tuples;

	if (path->indexorderbys == NIL)
	{
		*indexStartupCost = *indexTotalCost = get_float8_infinity();
		*indexSelectivity = *indexCorrelation = *indexPages = 0;
		return;
	}
	costs.numIndexTuples = tuples;
	genericcostestimate(root, path, loop_count, &costs);
	*indexStartupCost = costs.indexTotalCost +
		tuples * cpu_operator_cost * (1 + (tuples > 1 ? log(tuples) / log(2.0) : 0));
	*indexTotalCost = *indexStartupCost;
	*indexSelectivity = costs.indexSelectivity;
	*indexCorrelation = 0;
	*indexPages = costs.numIndexPages;
}

static bool
annvalidate(Oid opclassoid)
{
	Oid			type = get_opclass_input_type(opclassoid);
	Oid			proc = get_opfamily_proc(get_opclass_family(opclassoid), type, type, ANN_DISTANCE_PROC);

	return type == AnnVectorTypeOid() && OidIsValid(proc) &&
		check_amproc_signature(proc, FLOAT8OID, true, 2, 2, type, type);
}

static bool
annproperty(Oid index_oid, int attno, IndexAMProperty prop, const char *propname,
			bool *res, bool *isnull)
{
	if (attno == 0 || prop != AMPROP_DISTANCE_ORDERABLE)
		return false;
	*res = true;
	*isnull = false;
	return true;
}

static relopt_kind ann_relopt_kind;

void
AnnInitOptions(void)
{
	ann_relopt_kind = add_reloption_kind();
	add_bool_reloption(ann_relopt_kind, "frozen",
					   "Build an immutable index with forward and inverted pages",
					   false, AccessExclusiveLock);
	add_int_reloption(ann_relopt_kind, "bitset_min",
					  "Values of an attribute with at least this many rows get bitset pages (-1: rows / 32)",
					  -1, -1, INT_MAX, AccessExclusiveLock);
}

static bytea *
annoptions(Datum reloptions, bool validate)
{
	static const relopt_parse_elt tab[] = {
		{"frozen", RELOPT_TYPE_BOOL, offsetof(AnnOptions, frozen)},
		{"bitset_min", RELOPT_TYPE_INT, offsetof(AnnOptions, bitsetMin)},
	};

	return (bytea *) build_reloptions(reloptions, validate, ann_relopt_kind,
									  sizeof(AnnOptions), tab, lengthof(tab));
}

/* The reloption decides how a build lays out pages; the metapage records it. */
bool
AnnFrozenOption(Relation index)
{
	AnnOptions *options = (AnnOptions *) index->rd_options;

	return options && options->frozen;
}

/* Frozen pages hold every column, so index-only scans can return them all. */
static bool
anncanreturn(Relation index, int attno)
{
	return AnnIsFrozen(index);
}

/*
 * Register only implemented access-method capabilities.
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(annhandler);
Datum
annhandler(PG_FUNCTION_ARGS)
{
	IndexAmRoutine *amroutine = makeNode(IndexAmRoutine);

	amroutine->amstrategies = 0;
	amroutine->amsupport = ANN_NPROCS;
	amroutine->amcanorderbyop = true;
	amroutine->amoptionalkey = true;
	amroutine->amcaninclude = true;
	amroutine->amparallelvacuumoptions = VACUUM_OPTION_PARALLEL_BULKDEL;
	amroutine->amkeytype = InvalidOid;
	amroutine->ambuild = annbuild;
	amroutine->ambuildempty = annbuildempty;
	amroutine->aminsert = anninsert;
	amroutine->ambulkdelete = annbulkdelete;
	amroutine->amvacuumcleanup = annvacuumcleanup;
	amroutine->amcanreturn = anncanreturn;
	amroutine->amcostestimate = anncostestimate;
	amroutine->amoptions = annoptions;
	amroutine->amproperty = annproperty;
	amroutine->amvalidate = annvalidate;
	amroutine->ambeginscan = annbeginscan;
	amroutine->amrescan = annrescan;
	amroutine->amgettuple = anngettuple;
	amroutine->amendscan = annendscan;

	PG_RETURN_POINTER(amroutine);
}
