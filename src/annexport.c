#include "postgres.h"

#include <fcntl.h>
#include <unistd.h>

#include "ann.h"
#include "catalog/pg_authid_d.h"
#include "commands/defrem.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "storage/fd.h"
#include "utils/acl.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"

#define ANN_EXPORT_BUFFER (1024 * 1024)

/* Manifest values are space-separated words the worker reads with %Ns. */
static const char *
ManifestWord(const char *value, size_t limit, const char *what)
{
	if (strlen(value) == 0 || strlen(value) > limit || strpbrk(value, " \t\r\n/") != NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_NAME),
				 errmsg("%s \"%s\" cannot be written to an ann index manifest", what, value),
				 errdetail("Names must be at most %zu bytes, without spaces or slashes.", limit)));
	return value;
}

/* A new directory, or an existing empty one. */
static void
PrepareDirectory(const char *directory)
{
	DIR		   *dir;
	struct dirent *entry;

	if (!is_absolute_path(directory))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("ann index export directory must be an absolute path")));
	if (MakePGDirectory(directory) == 0)
		return;
	if (errno != EEXIST)
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not create directory \"%s\": %m", directory)));
	dir = AllocateDir(directory);
	while ((entry = ReadDir(dir, directory)) != NULL)
		if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0)
			ereport(ERROR,
					(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
					 errmsg("directory \"%s\" is not empty", directory),
					 errhint("Index directories are immutable: export into a new directory.")));
	FreeDir(dir);
}

static void
WriteAll(int fd, const char *path, const char *data, size_t size)
{
	while (size > 0)
	{
		ssize_t		written;

		errno = 0;
		written = write(fd, data, size);
		if (written <= 0)
		{
			if (errno == 0)
				errno = ENOSPC;
			ereport(ERROR,
					(errcode_for_file_access(),
					 errmsg("could not write file \"%s\": %m", path)));
		}
		data += written;
		size -= written;
	}
}

static int
CreateFile(const char *path)
{
	int			fd = OpenTransientFile(path, O_WRONLY | O_CREAT | O_EXCL | PG_BINARY);

	if (fd < 0)
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not create file \"%s\": %m", path)));
	return fd;
}

static void
CloseFile(int fd, const char *path)
{
	if (pg_fsync(fd) != 0)
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not fsync file \"%s\": %m", path)));
	if (CloseTransientFile(fd) != 0)
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not close file \"%s\": %m", path)));
}

/* Write bytes [skip, skip + bytes) of every element of an array, in order. */
static void
WriteArrayFile(Relation index, const AnnArray *array, const char *directory, const char *name,
			   uint32 skip, uint32 bytes)
{
	char	   *path = psprintf("%s/%s", directory, name);
	int			fd = CreateFile(path);
	BufferAccessStrategy strategy = GetAccessStrategy(BAS_BULKREAD);
	char	   *buffer = palloc(ANN_EXPORT_BUFFER);
	size_t		used = 0;
	AnnArrayReader reader;

	AnnArrayReaderInit(&reader, index, array, strategy);
	for (BlockNumber p = 0; p < array->pages; p++)
	{
		uint32		count;
		char	   *data;

		CHECK_FOR_INTERRUPTS();
		data = AnnArrayLockPage(&reader, p, &count);
		for (uint32 i = 0; i < count; i++)
		{
			if (used + bytes > ANN_EXPORT_BUFFER)
			{
				AnnArrayUnlockPage(&reader);
				WriteAll(fd, path, buffer, used);
				used = 0;
				data = AnnArrayLockPage(&reader, p, NULL);
			}
			memcpy(buffer + used, data + (size_t) i * array->width + skip, bytes);
			used += bytes;
		}
		AnnArrayUnlockPage(&reader);
	}
	WriteAll(fd, path, buffer, used);
	AnnArrayReaderEnd(&reader);
	FreeAccessStrategy(strategy);
	CloseFile(fd, path);
	pfree(buffer);
}

/*
 * vectors.f32 of an index whose pages hold halves or no vectors: the heap's
 * exact vectors, by TID in key order, so every export is the same file.
 */
static void
WriteExactVectors(Relation index, const AnnFrozenMetaData *meta, const char *directory)
{
	char	   *path = psprintf("%s/vectors.f32", directory);
	int			fd = CreateFile(path);
	BufferAccessStrategy strategy = GetAccessStrategy(BAS_BULKREAD);
	char	   *buffer = palloc(ANN_EXPORT_BUFFER);
	size_t		used = 0;
	AnnFrozenVectorReader reader;
	size_t		bytes;

	AnnFrozenVectorReaderInit(&reader, index, meta, strategy, true);
	bytes = sizeof(float) * reader.dims;
	for (uint64 ordinal = 0; ordinal < meta->rows; ordinal++)
	{
		Vector	   *vector;

		CHECK_FOR_INTERRUPTS();
		vector = AnnFrozenVectorRead(&reader, ordinal);
		if (used + bytes > ANN_EXPORT_BUFFER)
		{
			WriteAll(fd, path, buffer, used);
			used = 0;
		}
		memcpy(buffer + used, vector->x, bytes);
		used += bytes;
	}
	WriteAll(fd, path, buffer, used);
	AnnFrozenVectorReaderEnd(&reader);
	FreeAccessStrategy(strategy);
	CloseFile(fd, path);
	pfree(buffer);
}

/*
 * ann_index_export(index, directory): write a frozen index's rows in the
 * worker's directory format: keys.i64, vectors.f32, one attr-COLUMN.i32 per
 * INCLUDE column after the key (for text, codes of the index's dictionary,
 * written as attr-COLUMN.strings and attr-COLUMN.starts; for integer[],
 * every row's elements, with attr-COLUMN.offsets, a u32 start per row plus
 * the end, on a "multivalued" line), and a manifest with the index's
 * generation and no CAGRA shards. scripts/ann/build_cuvs_index.py --from-export adds
 * the shards; ann_index_load checks the generation against the table.
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_index_export);
Datum
ann_index_export(PG_FUNCTION_ARGS)
{
	Oid			relid = PG_GETARG_OID(0);
	char	   *directory = text_to_cstring(PG_GETARG_TEXT_PP(1));
	Relation	index;
	AnnFrozenMetaData *meta;
	Oid			heapOid;
	AclResult	aclresult;
	StringInfoData manifest;
	char	   *source;
	char	   *path;
	char	   *temporary;
	int			fd;
	int64		rows;

	if (!superuser() && !has_privs_of_role(GetUserId(), ROLE_PG_WRITE_SERVER_FILES))
		ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_PRIVILEGE),
				 errmsg("must be superuser or a member of pg_write_server_files to export an ann index")));
	index = index_open(relid, AccessShareLock);
	if (index->rd_rel->relam != get_am_oid("ann", false))
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not an ann index", RelationGetRelationName(index))));
	heapOid = index->rd_index->indrelid;
	aclresult = pg_class_aclcheck(heapOid, GetUserId(), ACL_SELECT);
	if (aclresult != ACLCHECK_OK)
		aclcheck_error(aclresult, OBJECT_TABLE, get_rel_name(heapOid));
	meta = AnnFrozenGetMeta(index, false);
	if (meta->rows == 0)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("frozen ann index \"%s\" has no rows", RelationGetRelationName(index))));

	source = quote_qualified_identifier(get_namespace_name(get_rel_namespace(heapOid)), get_rel_name(heapOid));
	initStringInfo(&manifest);
	appendStringInfo(&manifest, "pgvector-ann-index 1\ndimensions %d\nrows " UINT64_FORMAT "\nmetric l2\n"
					 "dtype float32\nsource %s\nkey %s\nvector %s\nshards 0\n",
					 TupleDescAttr(RelationGetDescr(index), 0)->atttypmod, meta->rows,
					 ManifestWord(source, 255, "table"),
					 ManifestWord(AnnFrozenColumnName(index, 2), 63, "column"),
					 ManifestWord(AnnFrozenColumnName(index, 1), 63, "column"));
	for (uint32 c = 0; c < meta->ncolumns; c++)
	{
		const char *name = ManifestWord(NameStr(AnnFrozenGetNames(meta)[c]), 63, "column");

		appendStringInfo(&manifest, "attribute %s attr-%s.i32\n", name, name);
		if (AnnFrozenTextType(TupleDescAttr(RelationGetDescr(index), 2 + c)->atttypid))
			appendStringInfo(&manifest, "dictionary %s attr-%s.strings attr-%s.starts\n", name, name, name);
		if (AnnFrozenMultiType(TupleDescAttr(RelationGetDescr(index), 2 + c)->atttypid))
			appendStringInfo(&manifest, "multivalued %s attr-%s.offsets\n", name, name);
	}
	appendStringInfo(&manifest, "generation %016llx\n", (unsigned long long) meta->generation);

	PrepareDirectory(directory);
	WriteArrayFile(index, &meta->keys, directory, "keys.i64", 0, sizeof(int64));
	if (AnnFrozenVectorMode(index, meta) == ANN_FROZEN_VECTORS_FLOAT)
		WriteArrayFile(index, &meta->vectors, directory, "vectors.f32", offsetof(Vector, x),
					   meta->vectors.width - offsetof(Vector, x));
	else
		WriteExactVectors(index, meta, directory);
	for (uint32 c = 0; c < meta->ncolumns; c++)
	{
		const char *name = NameStr(AnnFrozenGetNames(meta)[c]);

		WriteArrayFile(index, &meta->columns[c].values, directory, psprintf("attr-%s.i32", name), 0, sizeof(int32));
		if (AnnFrozenTextType(TupleDescAttr(RelationGetDescr(index), 2 + c)->atttypid))
		{
			/* the strings in code (collation) order, and their starts */
			WriteArrayFile(index, &AnnFrozenGetExtra(meta)[c].strings, directory,
						   psprintf("attr-%s.strings", name), 0, 1);
			WriteArrayFile(index, &AnnFrozenGetExtra(meta)[c].starts, directory,
						   psprintf("attr-%s.starts", name), 0, sizeof(uint32));
		}
		if (AnnFrozenMultiType(TupleDescAttr(RelationGetDescr(index), 2 + c)->atttypid))
			WriteArrayFile(index, &AnnFrozenGetExtra(meta)[c].starts, directory,
						   psprintf("attr-%s.offsets", name), 0, sizeof(uint32));
	}

	/* The manifest comes last, so a partial export is never loadable. */
	temporary = psprintf("%s/manifest.tmp", directory);
	path = psprintf("%s/manifest", directory);
	fd = CreateFile(temporary);
	WriteAll(fd, temporary, manifest.data, manifest.len);
	CloseFile(fd, temporary);
	durable_rename(temporary, path, ERROR);
	rows = (int64) meta->rows;
	index_close(index, AccessShareLock);
	PG_RETURN_INT64(rows);
}
