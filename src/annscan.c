#include "postgres.h"

#include "access/htup_details.h"
#include "ann.h"
#include "catalog/pg_operator_d.h"
#include "catalog/pg_type_d.h"
#include "executor/tuptable.h"
#include "miscadmin.h"
#include "pgstat.h"
#include "utils/memutils.h"
#include "utils/snapmgr.h"
#include "utils/tuplesort.h"

typedef struct AnnScanOpaqueData
{
	bool		first;
	FmgrInfo   *procinfo;
	TupleDesc	tupdesc;
	TupleTableSlot *input;
	TupleTableSlot *output;
	Tuplesortstate *sortstate;
	MemoryContext tmpCtx;

	/* Frozen indexes: arrays by ordinal, and the tuple returned to IOS. */
	AnnFrozenMetaData *meta;
	AnnArrayReader tids;
	AnnArrayReader keys;
	AnnArrayReader vectors;
	AnnArrayReader *columns;
	AnnArrayReader *starts;		/* integer[] columns' row starts */
	Vector	   *vector;
	MemoryContext tupleCtx;
} AnnScanOpaqueData;

typedef AnnScanOpaqueData *AnnScanOpaque;

static void
GetScanItems(IndexScanDesc scan)
{
	AnnScanOpaque so = scan->opaque;
	Relation	index = scan->indexRelation;
	BlockNumber blocks;
	Buffer		buf;
	Datum		query = scan->orderByData[0].sk_argument;
	bool		queryIsNull = (scan->orderByData[0].sk_flags & SK_ISNULL) != 0;

	buf = ReadBuffer(index, ANN_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	AnnCheckMetaPage(BufferGetPage(buf));
	UnlockReleaseBuffer(buf);
	blocks = RelationGetNumberOfBlocks(index);
	for (BlockNumber blkno = 1; blkno < blocks; blkno++)
	{
		Page		page;
		OffsetNumber maxoff;

		CHECK_FOR_INTERRUPTS();
		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		/* An aborted extension can leave an unpublished, empty physical page. */
		if (PageIsNew(page))
		{
			UnlockReleaseBuffer(buf);
			continue;
		}
		AnnCheckPage(page);
		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno++)
		{
			ItemId		itemid = PageGetItemId(page, offno);
			IndexTuple	tuple;
			bool		isnull;
			Datum		value;
			MemoryContext oldCtx;

			if (!ItemIdIsNormal(itemid))
				continue;
			tuple = (IndexTuple) PageGetItem(page, itemid);
			oldCtx = MemoryContextSwitchTo(so->tmpCtx);
			value = index_getattr(tuple, 1, RelationGetDescr(index), &isnull);
			ExecClearTuple(so->input);
			so->input->tts_isnull[0] = isnull || queryIsNull;
			so->input->tts_values[0] = (isnull || queryIsNull) ? (Datum) 0 :
				FunctionCall2Coll(so->procinfo, index->rd_indcollation[0],
								  PointerGetDatum(DatumGetVector(value)), query);
			so->input->tts_isnull[1] = false;
			so->input->tts_values[1] = PointerGetDatum(&tuple->t_tid);
			so->input->tts_isnull[2] = false;
			so->input->tts_values[2] = Int64GetDatum(0);
			ExecStoreVirtualTuple(so->input);
			MemoryContextSwitchTo(oldCtx);
			tuplesort_puttupleslot(so->sortstate, so->input);
			ExecClearTuple(so->input);
			MemoryContextReset(so->tmpCtx);
		}
		UnlockReleaseBuffer(buf);
	}
	tuplesort_performsort(so->sortstate);
}

/*
 * Frozen indexes: score the vector pages in ordinal order. Rows keep their
 * ordinal through the sort; TIDs and returned columns are read afterwards.
 */
static void
GetFrozenScanItems(IndexScanDesc scan)
{
	AnnScanOpaque so = scan->opaque;
	Relation	index = scan->indexRelation;
	const AnnArray *array = &so->meta->vectors;
	Datum		query = scan->orderByData[0].sk_argument;
	bool		queryIsNull = (scan->orderByData[0].sk_flags & SK_ISNULL) != 0;
	BufferAccessStrategy strategy = GetAccessStrategy(BAS_BULKREAD);
	double	   *distances = palloc(sizeof(double) * array->perPage);
	ItemPointerData none;
	AnnArrayReader reader;

	ItemPointerSetInvalid(&none);
	if (AnnFrozenVectorMode(index, so->meta) != ANN_FROZEN_VECTORS_FLOAT)
	{
		/* Widened half pages, or the heap's vectors, ordinal by ordinal. */
		AnnFrozenVectorReader vr;

		AnnFrozenVectorReaderInit(&vr, index, so->meta, strategy, false);
		for (uint64 ordinal = 0; ordinal < so->meta->rows; ordinal++)
		{
			double		distance = 0;

			CHECK_FOR_INTERRUPTS();
			if (!queryIsNull)
			{
				MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);

				distance = DatumGetFloat8(FunctionCall2Coll(so->procinfo, index->rd_indcollation[0],
															PointerGetDatum(AnnFrozenVectorRead(&vr, ordinal)),
															query));
				MemoryContextSwitchTo(oldCtx);
				MemoryContextReset(so->tmpCtx);
			}
			ExecClearTuple(so->input);
			so->input->tts_isnull[0] = queryIsNull;
			so->input->tts_values[0] = queryIsNull ? (Datum) 0 : Float8GetDatum(distance);
			so->input->tts_isnull[1] = false;
			so->input->tts_values[1] = PointerGetDatum(&none);
			so->input->tts_isnull[2] = false;
			so->input->tts_values[2] = Int64GetDatum((int64) ordinal);
			ExecStoreVirtualTuple(so->input);
			tuplesort_puttupleslot(so->sortstate, so->input);
			ExecClearTuple(so->input);
		}
		AnnFrozenVectorReaderEnd(&vr);
		FreeAccessStrategy(strategy);
		pfree(distances);
		tuplesort_performsort(so->sortstate);
		return;
	}
	AnnArrayReaderInit(&reader, index, array, strategy);
	for (BlockNumber p = 0; p < array->pages; p++)
	{
		uint32		count;
		char	   *data;
		MemoryContext oldCtx;

		CHECK_FOR_INTERRUPTS();
		data = AnnArrayLockPage(&reader, p, &count);
		oldCtx = MemoryContextSwitchTo(so->tmpCtx);
		for (uint32 j = 0; j < count && !queryIsNull; j++)
			distances[j] = DatumGetFloat8(FunctionCall2Coll(so->procinfo, index->rd_indcollation[0],
															PointerGetDatum(data + (size_t) j * array->width),
															query));
		MemoryContextSwitchTo(oldCtx);
		AnnArrayUnlockPage(&reader);
		for (uint32 j = 0; j < count; j++)
		{
			ExecClearTuple(so->input);
			so->input->tts_isnull[0] = queryIsNull;
			so->input->tts_values[0] = queryIsNull ? (Datum) 0 : Float8GetDatum(distances[j]);
			so->input->tts_isnull[1] = false;
			so->input->tts_values[1] = PointerGetDatum(&none);
			so->input->tts_isnull[2] = false;
			so->input->tts_values[2] = Int64GetDatum((int64) p * array->perPage + j);
			ExecStoreVirtualTuple(so->input);
			tuplesort_puttupleslot(so->sortstate, so->input);
			ExecClearTuple(so->input);
		}
		MemoryContextReset(so->tmpCtx);
	}
	AnnArrayReaderEnd(&reader);
	FreeAccessStrategy(strategy);
	pfree(distances);
	tuplesort_performsort(so->sortstate);
}

/* Every index column of one row, as the heap tuple index-only scans read. */
static void
ReturnColumns(IndexScanDesc scan, uint64 ordinal)
{
	AnnScanOpaque so = scan->opaque;
	TupleDesc	desc = RelationGetDescr(scan->indexRelation);
	Datum		values[INDEX_MAX_KEYS];
	bool		isnull[INDEX_MAX_KEYS] = {0};
	Oid			type = TupleDescAttr(desc, 1)->atttypid;
	int64		key;
	MemoryContext oldCtx;

	MemoryContextReset(so->tupleCtx);
	oldCtx = MemoryContextSwitchTo(so->tupleCtx);
	if (AnnFrozenVectorMode(scan->indexRelation, so->meta) == ANN_FROZEN_VECTORS_FLOAT)
	{
		AnnArrayRead(&so->vectors, ordinal, so->vector);
		values[0] = PointerGetDatum(so->vector);
	}
	else
	{
		/* anncanreturn offers no vector from half or missing pages */
		values[0] = (Datum) 0;
		isnull[0] = true;
	}
	AnnArrayRead(&so->keys, ordinal, &key);
	values[1] = type == INT2OID ? Int16GetDatum((int16) key) :
		type == INT4OID ? Int32GetDatum((int32) key) : Int64GetDatum(key);
	for (uint32 c = 0; c < so->meta->ncolumns; c++)
	{
		int32		value;

		if (AnnFrozenMultiType(TupleDescAttr(desc, 2 + c)->atttypid))
		{
			values[2 + c] = AnnFrozenArrayDatum(&so->columns[c], &so->starts[c], ordinal, &isnull[2 + c]);
			continue;
		}
		AnnArrayRead(&so->columns[c], ordinal, &value);
		isnull[2 + c] = value == ANN_FROZEN_NULL;
		if (AnnFrozenTextType(TupleDescAttr(desc, 2 + c)->atttypid))
			values[2 + c] = isnull[2 + c] ? (Datum) 0 :
				AnnFrozenTextDatum(scan->indexRelation, so->meta, c, value);
		else
			values[2 + c] = AnnFrozenAttributeDatum(value, TupleDescAttr(desc, 2 + c)->atttypid);
	}
	scan->xs_hitup = heap_form_tuple(desc, values, isnull);
	scan->xs_hitupdesc = desc;
	MemoryContextSwitchTo(oldCtx);
}

IndexScanDesc
annbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan = RelationGetIndexScan(index, nkeys, norderbys);
	AnnScanOpaque so = palloc0_object(AnnScanOpaqueData);
	AttrNumber	attNums[] = {1};
	Oid			operators[] = {Float8LessOperator};
	Oid			collations[] = {InvalidOid};
	bool		nullsFirst[] = {false};

	so->first = true;
	so->procinfo = index_getprocinfo(index, 1, ANN_DISTANCE_PROC);
	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext, "Ann scan temporary context", ALLOCSET_DEFAULT_SIZES);
	so->tupdesc = CreateTemplateTupleDesc(3);
	TupleDescInitEntry(so->tupdesc, 1, "distance", FLOAT8OID, -1, 0);
	TupleDescInitEntry(so->tupdesc, 2, "heaptid", TIDOID, -1, 0);
	TupleDescInitEntry(so->tupdesc, 3, "ordinal", INT8OID, -1, 0);
#if PG_VERSION_NUM >= 190000
	TupleDescFinalize(so->tupdesc);
#endif
	so->input = MakeSingleTupleTableSlot(so->tupdesc, &TTSOpsVirtual);
	so->output = MakeSingleTupleTableSlot(so->tupdesc, &TTSOpsMinimalTuple);
	so->sortstate = tuplesort_begin_heap(so->tupdesc, 1, attNums, operators, collations,
										 nullsFirst, work_mem, NULL, false);
	so->meta = AnnFrozenGetMeta(index, true);
	if (so->meta)
	{
		AnnArrayReaderInit(&so->tids, index, &so->meta->tids, NULL);
		AnnArrayReaderInit(&so->keys, index, &so->meta->keys, NULL);
		AnnArrayReaderInit(&so->vectors, index, &so->meta->vectors, NULL);
		so->columns = palloc(sizeof(AnnArrayReader) * Max(so->meta->ncolumns, 1));
		so->starts = palloc0(sizeof(AnnArrayReader) * Max(so->meta->ncolumns, 1));
		for (uint32 c = 0; c < so->meta->ncolumns; c++)
		{
			AnnArrayReaderInit(&so->columns[c], index, &so->meta->columns[c].values, NULL);
			if (AnnFrozenMultiType(TupleDescAttr(RelationGetDescr(index), 2 + c)->atttypid))
				AnnArrayReaderInit(&so->starts[c], index, &AnnFrozenGetExtra(so->meta)[c].starts, NULL);
		}
		so->vector = palloc(so->meta->vectors.width);
		so->tupleCtx = AllocSetContextCreate(CurrentMemoryContext, "Ann scan tuple context",
											 ALLOCSET_DEFAULT_SIZES);
	}
	scan->opaque = so;
	return scan;
}

void
annrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	AnnScanOpaque so = scan->opaque;

	so->first = true;
	ExecClearTuple(so->output);
	tuplesort_reset(so->sortstate);
	MemoryContextReset(so->tmpCtx);
	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));
	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));
}

bool
anngettuple(IndexScanDesc scan, ScanDirection dir)
{
	AnnScanOpaque so = scan->opaque;
	Datum		value;
	bool		isnull;

	Assert(ScanDirectionIsForward(dir));
	if (so->first)
	{
		if (scan->numberOfOrderBys != 1 || !scan->orderByData)
			elog(ERROR, "cannot scan ann index without one distance order");
		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with ann");
		pgstat_count_index_scan(scan->indexRelation);
		if (so->meta)
			GetFrozenScanItems(scan);
		else
			GetScanItems(scan);
		so->first = false;
	}
	for (;;)
	{
		int64		ordinal;

		if (!tuplesort_gettupleslot(so->sortstate, true, false, so->output, NULL))
			return false;
		if (!so->meta)
		{
			value = slot_getattr(so->output, 2, &isnull);
			Assert(!isnull);
			scan->xs_heaptid = *DatumGetItemPointer(value);
			break;
		}

		/* VACUUM invalidates the TIDs of deleted rows; skip them. */
		ordinal = DatumGetInt64(slot_getattr(so->output, 3, &isnull));
		AnnArrayRead(&so->tids, ordinal, &scan->xs_heaptid);
		if (!ItemPointerIsValid(&scan->xs_heaptid))
			continue;
		if (scan->xs_want_itup)
			ReturnColumns(scan, ordinal);
		break;
	}
	scan->xs_recheck = false;
	scan->xs_recheckorderby = false;
	return true;
}

void
annendscan(IndexScanDesc scan)
{
	AnnScanOpaque so = scan->opaque;

	if (so->meta)
	{
		AnnArrayReaderEnd(&so->tids);
		AnnArrayReaderEnd(&so->keys);
		AnnArrayReaderEnd(&so->vectors);
		for (uint32 c = 0; c < so->meta->ncolumns; c++)
		{
			AnnArrayReaderEnd(&so->columns[c]);
			if (AnnFrozenMultiType(TupleDescAttr(RelationGetDescr(scan->indexRelation), 2 + c)->atttypid))
				AnnArrayReaderEnd(&so->starts[c]);
		}
		MemoryContextDelete(so->tupleCtx);
	}
	ExecDropSingleTupleTableSlot(so->input);
	ExecDropSingleTupleTableSlot(so->output);
	tuplesort_end(so->sortstate);
	FreeTupleDesc(so->tupdesc);
	MemoryContextDelete(so->tmpCtx);
	pfree(so);
	scan->opaque = NULL;
}
