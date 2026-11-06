#include "postgres.h"

#include "access/generic_xlog.h"
#include "ann.h"
#include "commands/vacuum.h"
#include "storage/freespace.h"

#if PG_VERSION_NUM >= 180000
#define vacuum_delay_point() vacuum_delay_point(false)
#endif

/*
 * Frozen indexes never move a row's ordinal. A deleted row (possible only if
 * someone disabled or dropped the guard trigger) keeps its ordinal with an
 * invalid TID, which scans skip, so VACUUM can safely reuse its heap slot.
 * The first invalidated TID also clears ANN_FROZEN_PRISTINE in the metapage,
 * so ann.frozen_immutable never applies to the table again. VACUUM calls this
 * only with dead rows to remove, which a table it applies to never has.
 */
IndexBulkDeleteResult *
AnnFrozenBulkDelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
					IndexBulkDeleteCallback callback, void *callback_state)
{
	Relation	index = info->index;
	AnnFrozenMetaData *meta;

	if (AnnFrozenIndexImmutable(index))
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("cannot remove rows from frozen ann index \"%s\" while ann.frozen_immutable applies to its table",
						RelationGetRelationName(index)),
				 errhint("The table changed after its frozen index was built: VACUUM it with ann.frozen_immutable off.")));
	meta = AnnFrozenGetMeta(index, false);
	if (!stats)
		stats = palloc0_object(IndexBulkDeleteResult);
	stats->num_index_tuples = 0;
	for (BlockNumber p = 0; p < meta->tids.pages; p++)
	{
		Buffer		buf;
		Page		page;
		GenericXLogState *state;
		ItemPointerData *tids;
		uint32		count = p + 1 < meta->tids.pages ? meta->tids.perPage :
			(uint32) (meta->tids.count - (uint64) p * meta->tids.perPage);
		uint32		dead[BLCKSZ / sizeof(ItemPointerData)];
		uint32		ndead = 0;

		vacuum_delay_point();
		buf = ReadBufferExtended(index, MAIN_FORKNUM, meta->tids.start + p, RBM_NORMAL, info->strategy);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		AnnCheckPage(BufferGetPage(buf));
		tids = (ItemPointerData *) PageGetContents(BufferGetPage(buf));
		for (uint32 i = 0; i < count; i++)
		{
			if (!ItemPointerIsValid(&tids[i]))
				continue;
			if (callback(&tids[i], callback_state))
				dead[ndead++] = i;
			else
				stats->num_index_tuples++;
		}
		LockBuffer(buf, BUFFER_LOCK_UNLOCK);
		if (ndead)
		{
			Buffer		metabuf = ReadBuffer(index, ANN_METAPAGE_BLKNO);

			/* No TID page lock while waiting for readers of mapped TIDs. */
			LockBuffer(metabuf, BUFFER_LOCK_EXCLUSIVE);
			if (AnnPageGetMeta(BufferGetPage(metabuf))->insertPage == ANN_FROZEN_PRISTINE)
			{
				state = GenericXLogStart(index);
				page = GenericXLogRegisterBuffer(state, metabuf, 0);
				AnnPageGetMeta(page)->insertPage = ANN_FROZEN_FLUSHED;
				GenericXLogFinish(state);
			}
			UnlockReleaseBuffer(metabuf);
			/* The WAL barrier precedes all physical TID invalidation. */
			LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
			state = GenericXLogStart(index);
			page = GenericXLogRegisterBuffer(state, buf, 0);
			tids = (ItemPointerData *) PageGetContents(page);
			for (uint32 i = 0; i < ndead; i++)
				if (ItemPointerIsValid(&tids[dead[i]]))
				{
					ItemPointerSetInvalid(&tids[dead[i]]);
					stats->tuples_removed++;
				}
			GenericXLogFinish(state);
			LockBuffer(buf, BUFFER_LOCK_UNLOCK);
		}
		ReleaseBuffer(buf);
	}
	pfree(meta);
	return stats;
}

IndexBulkDeleteResult *
annbulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
			  IndexBulkDeleteCallback callback, void *callback_state)
{
	Relation	index = info->index;
	BlockNumber blocks;

	if (AnnIsFrozen(index))
		return AnnFrozenBulkDelete(info, stats, callback, callback_state);
	blocks = RelationGetNumberOfBlocks(index);
	if (!stats)
		stats = palloc0_object(IndexBulkDeleteResult);
	/* A later bulk-delete cycle rescans survivors; do not count them twice. */
	stats->num_index_tuples = 0;
	for (BlockNumber blkno = 1; blkno < blocks; blkno++)
	{
		Buffer		buf;
		Page		page;
		GenericXLogState *state;
		OffsetNumber deletable[MaxOffsetNumber];
		OffsetNumber maxoff;
		int			count = 0;
		Size		freeSpace;

		vacuum_delay_point();
		buf = ReadBuffer(index, blkno);
		LockBufferForCleanup(buf);
		page = BufferGetPage(buf);
		if (PageIsNew(page))
		{
			UnlockReleaseBuffer(buf);
			RecordPageWithFreeSpace(index, blkno, BLCKSZ - SizeOfPageHeaderData -
									MAXALIGN(sizeof(AnnPageOpaqueData)) - sizeof(ItemIdData));
			continue;
		}
		AnnCheckPage(page);
		state = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(state, buf, 0);
		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno++)
		{
			IndexTuple	tuple = (IndexTuple) PageGetItem(page, PageGetItemId(page, offno));

			if (callback(&tuple->t_tid, callback_state))
			{
				deletable[count++] = offno;
				stats->tuples_removed++;
			}
			else
				stats->num_index_tuples++;
		}
		if (count)
		{
			PageIndexMultiDelete(page, deletable, count);
			freeSpace = PageGetFreeSpace(page);
			GenericXLogFinish(state);
		}
		else
		{
			freeSpace = PageGetFreeSpace(page);
			GenericXLogAbort(state);
		}
		UnlockReleaseBuffer(buf);
		RecordPageWithFreeSpace(index, blkno, freeSpace);
	}
	FreeSpaceMapVacuum(index);
	return stats;
}

IndexBulkDeleteResult *
annvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	if (info->analyze_only || !stats)
		return stats;
	stats->num_pages = RelationGetNumberOfBlocks(info->index);
	return stats;
}
