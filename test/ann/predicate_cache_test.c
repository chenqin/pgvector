/*
 * Cached predicate results and deletes (indexfile.c), on a small index
 * written to a temporary directory: a result evaluated before a delete is
 * cached without the deleted rows, snapshot results stay cached across
 * deletes, results that read a pushed bitmap are dropped, and the others
 * are patched into what a fresh evaluation would cache. Fake device hooks
 * stand in for a backend, which the CPU worker lacks.
 */
#ifndef __APPLE__
#define _POSIX_C_SOURCE 200809L
#endif
#include "indexfile.h"
#include "filter.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#define ROWS 5003
#define DIMENSIONS 2
#define GROUPS 7
#define BUCKETS 100				/* attribute b: about 50 rows a value, under
								 * rows / 64 */

#define CHECK(condition) \
	do { \
		if (!(condition)) \
		{ \
			fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
			exit(1); \
		} \
	} while (0)

static char directory[] = "/tmp/ann-predicate-XXXXXX";
static AnnIndexFile *file;
static size_t nwords = (ROWS + 31) / 32;

/* Fake device hooks: count copies, and optionally delete a row during a build. */
static int	uploads;
static int	releases;
static int64_t deleteDuringBuild = -1;

static void
Upload(AnnIndexFile *index, AnnIndexBitmap *bitmap)
{
	if (deleteDuringBuild >= 0)
	{
		int64_t		key = deleteDuringBuild;
		uint64_t	added;

		/* Once: the re-masked build runs this hook again. */
		deleteDuringBuild = -1;
		CHECK(AnnIndexDelete(index, &key, 1, &added) == ANN_OK && added == 1);
	}
	bitmap->device = malloc(sizeof(void *));
	CHECK(bitmap->device != NULL);
	bitmap->deviceBytes = 64;
	uploads++;
}

static void
Release(AnnIndexFile *index, AnnIndexBitmap *bitmap)
{
	(void) index;
	free(bitmap->device);
	bitmap->device = NULL;
	releases++;
}

static int64_t
Key(uint64_t row)
{
	return (int64_t) row * 2 + 1;
}

static int32_t
Group		(uint64_t row)
{
	return (int32_t) (row % GROUPS);
}

static void
WriteFile(const char *name, const void *data, size_t size)
{
	char		path[256];
	FILE	   *out;

	snprintf(path, sizeof(path), "%s/%s", directory, name);
	CHECK((out = fopen(path, "wb")) != NULL);
	CHECK(fwrite(data, 1, size, out) == size);
	CHECK(fclose(out) == 0);
}

/* Three shards starting off 32-row boundaries, one int32 attribute a. */
static void
WriteIndex(void)
{
	static int64_t keys[ROWS];
	static float vectors[ROWS * DIMENSIONS];
	static int32_t groups[ROWS];
	static int32_t buckets[ROWS];
	char		manifest[1024];
	int			length;

	for (uint64_t i = 0; i < ROWS; i++)
	{
		keys[i] = Key(i);
		vectors[i * DIMENSIONS] = (float) i;
		vectors[i * DIMENSIONS + 1] = 1;
		groups[i] = Group (i);

		buckets[i] = (int32_t) (i % BUCKETS);
	}
	WriteFile("keys.i64", keys, sizeof(keys));
	WriteFile("vectors.f32", vectors, sizeof(vectors));
	WriteFile("attr-a.i32", groups, sizeof(groups));
	WriteFile("attr-b.i32", buckets, sizeof(buckets));
	length = snprintf(manifest, sizeof(manifest),
					  "pgvector-ann-index 1\ndimensions %d\nrows %d\nmetric l2\nshards 3\n"
					  "shard 0 0 1667 shard-0.cagra\nshard 1 1667 1668 shard-1.cagra\n"
					  "shard 2 3335 1668 shard-2.cagra\nattribute a attr-a.i32\nattribute b attr-b.i32\n",
					  DIMENSIONS, ROWS);
	WriteFile("manifest", manifest, (size_t) length);
}

static void
RemoveIndex(void)
{
	const char *names[] = {"keys.i64", "vectors.f32", "attr-a.i32", "attr-b.i32", "manifest"};
	char		path[256];

	for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++)
	{
		snprintf(path, sizeof(path), "%s/%s", directory, names[i]);
		(void) unlink(path);
	}
	(void) rmdir(directory);
}

/* VALUES column IN (values), or NOT IN with negate. */
static size_t
ColumnProgram(unsigned char *program, char column, int negate, const int32_t *values, uint32_t n)
{
	program[0] = ANN_FILTER_VALUES;
	program[1] = 1;
	program[2] = (unsigned char) column;
	program[3] = (unsigned char) negate;
	memcpy(program + 4, &n, 4);
	memcpy(program + 8, values, n * 4);
	return 8 + (size_t) n * 4;
}

/* VALUES a IN (value): the rows of one group. */
static size_t
ValuesProgram(unsigned char *program, int32_t value)
{
	return ColumnProgram(program, 'a', 0, &value, 1);
}

/* BITMAP id AND VALUES a IN (value). */
static size_t
BitmapProgram(unsigned char *program, uint64_t id, int32_t value)
{
	program[0] = ANN_FILTER_BITMAP;
	memcpy(program + 1, &id, 8);
	ValuesProgram(program + 9, value);
	program[21] = ANN_FILTER_AND;
	return 22;
}

static int
Holds(const AnnIndexBitmap *bitmap, uint64_t row)
{
	static uint32_t words[(ROWS + 31) / 32];

	AnnIndexBitmapWords(file, bitmap, words);
	return (words[row / 32] >> (row % 32)) & 1;
}

static void
Delete(uint64_t row)
{
	int64_t		key = Key(row);
	uint64_t	added;

	CHECK(AnnIndexDelete(file, &key, 1, &added) == ANN_OK && added == 1);
}

/*
 * d1: a result evaluated before a delete and cached after it is re-masked,
 * whether the delete is seen before the build (the index's first delete,
 * when nothing was masked at all) or only under the cache lock (a delete
 * during the build).
 */
static void
TestRacedPut(void)
{
	unsigned char program[32];
	size_t		size = ValuesProgram(program, 3);
	uint32_t   *words = malloc(nwords * 4);
	uint64_t	count;
	AnnIndexEvaluation how;
	AnnIndexBitmap *result;

	CHECK(words != NULL);
	CHECK(AnnIndexEvaluate(file, program, size, words, &count, 0, &how) == ANN_FILTER_OK);
	CHECK(how.columnsOnly && count == (ROWS + GROUPS - 1 - 3) / GROUPS);
	Delete(3);
	CHECK(AnnIndexPredicatePut(file, 1, ANN_BITMAP_PREDICATE, words, nwords, UINT64_MAX, how.generation,
							   how.remapGeneration, how.columnsOnly, &result) == ANN_OK);
	CHECK(result->count == count - 1 && !Holds(result, 3) && Holds(result, 10));
	CHECK(file->predicateRaced == 1);
	CHECK(AnnIndexBitmapGet(file, 1, ANN_BITMAP_PREDICATE) == result);
	AnnIndexBitmapReleaseCount(file, result, 2);

	/* The delete lands while the put builds its entry. */
	AnnIndexConfig.device_bitmaps = 1;
	AnnIndexConfig.gpu_predicates = 0;
	AnnIndexBitmapDevice.upload = Upload;
	AnnIndexBitmapDevice.release = Release;
	file->device = &file->device;
	uploads = releases = 0;
	CHECK(AnnIndexEvaluate(file, program, size, words, &count, 0, &how) == ANN_FILTER_OK);
	deleteDuringBuild = Key(10);
	CHECK(AnnIndexPredicatePut(file, 2, ANN_BITMAP_PREDICATE, words, nwords, UINT64_MAX, how.generation,
							   how.remapGeneration, how.columnsOnly, &result) == ANN_OK);
	CHECK(deleteDuringBuild < 0);
	CHECK(result->count == count - 1 && !Holds(result, 10) && Holds(result, 17));
	CHECK(file->predicateRaced == 2);
	CHECK(AnnIndexBitmapGet(file, 2, ANN_BITMAP_PREDICATE) == result);

	/*
	 * The discarded build's device copy was freed; the linked one has its
	 * own.
	 */
	CHECK(uploads == 2 && releases == 1 && result->device != NULL);
	AnnIndexBitmapReleaseCount(file, result, 2);

	/* With no delete in between, nothing is re-masked. */
	CHECK(AnnIndexEvaluate(file, program, size, words, &count, 0, &how) == ANN_FILTER_OK);
	CHECK(AnnIndexPredicatePut(file, 3, ANN_BITMAP_PREDICATE, words, nwords, UINT64_MAX, how.generation,
							   how.remapGeneration, how.columnsOnly, NULL) == ANN_OK);
	CHECK(file->predicateRaced == 2);
	free(words);
}

/*
 * A delete keeps snapshot results read from the columns alone (they keep
 * tombstoned rows anyway) and drops results that read a pushed bitmap,
 * snapshot or not.
 */
static void
TestDeleteKeepsSnapshots(void)
{
	unsigned char program[32];
	size_t		size;
	uint32_t   *words = malloc(nwords * 4);
	uint64_t	count;
	uint64_t	dropped = file->predicateDropped;
	AnnIndexEvaluation how;
	AnnIndexBitmap *snapshot;
	AnnIndexBitmap *entry;

	CHECK(words != NULL);
	size = ValuesProgram(program, 5);
	CHECK(AnnIndexEvaluate(file, program, size, words, &count, 1, &how) == ANN_FILTER_OK);
	CHECK(AnnIndexPredicatePut(file, 10, ANN_BITMAP_SNAPSHOT, words, nwords, UINT64_MAX, UINT64_MAX,
							   how.remapGeneration, how.columnsOnly, &snapshot) == ANN_OK);

	/* A pushed bitmap of every row, and results that read it. */
	memset(words, 0xff, nwords * 4);
	CHECK(AnnIndexBitmapPut(file, 77, ANN_BITMAP_PUSHED, words, nwords, UINT64_MAX, NULL) == ANN_OK);
	size = BitmapProgram(program, 77, 5);
	CHECK(AnnIndexEvaluate(file, program, size, words, &count, 0, &how) == ANN_FILTER_OK);
	CHECK(!how.columnsOnly);
	CHECK(AnnIndexPredicatePut(file, 11, ANN_BITMAP_PREDICATE, words, nwords, UINT64_MAX, how.generation,
							   how.remapGeneration, how.columnsOnly, NULL) == ANN_OK);
	CHECK(AnnIndexEvaluate(file, program, size, words, &count, 1, &how) == ANN_FILTER_OK);
	CHECK(AnnIndexPredicatePut(file, 12, ANN_BITMAP_SNAPSHOT, words, nwords, UINT64_MAX, UINT64_MAX,
							   how.remapGeneration, how.columnsOnly, NULL) == ANN_OK);

	Delete(5);
	CHECK(AnnIndexBitmapGet(file, 10, ANN_BITMAP_SNAPSHOT) == snapshot);
	CHECK(Holds(snapshot, 5));
	AnnIndexBitmapReleaseCount(file, snapshot, 2);
	CHECK(AnnIndexBitmapGet(file, 11, ANN_BITMAP_PREDICATE) == NULL);
	CHECK(AnnIndexBitmapGet(file, 12, ANN_BITMAP_SNAPSHOT) == NULL);
	CHECK((entry = AnnIndexBitmapGet(file, 77, ANN_BITMAP_PUSHED)) != NULL);
	AnnIndexBitmapRelease(file, entry);
	CHECK(file->predicateDropped >= dropped + 2);
	free(words);
}

/* Every form of two entries is the same, device copies aside. */
static void
CheckSame(const AnnIndexBitmap *a, const AnnIndexBitmap *b)
{
	CHECK(a->count == b->count && a->hostBytes == b->hostBytes && a->kind == b->kind);
	CHECK(!a->words == !b->words && (!a->words || memcmp(a->words, b->words, nwords * 4) == 0));
	CHECK(!a->ordinals == !b->ordinals &&
		  (!a->ordinals || memcmp(a->ordinals, b->ordinals, a->count * sizeof(uint64_t)) == 0));
	CHECK(!a->keys && !b->keys);
	for (uint32_t s = 0; s < ANN_INDEX_MAX_SHARDS; s++)
	{
		size_t		shardWords = s < file->nshards ? ((size_t) file->shards[s].rows + 31) / 32 : 0;

		CHECK(!a->bitsets[s] == !b->bitsets[s] && !a->owned[s] == !b->owned[s]);
		CHECK(!a->bitsets[s] || memcmp(a->bitsets[s], b->bitsets[s], shardWords * 4) == 0);
	}
}

/* Evaluate and cache a program as the server does; returns the entry, referenced. */
static AnnIndexBitmap *
Cache(uint64_t id, const unsigned char *program, size_t size, uint64_t expandLimit)
{
	uint32_t   *words = malloc(nwords * 4);
	uint64_t	count;
	AnnIndexEvaluation how;
	AnnIndexBitmap *entry;

	CHECK(words != NULL);
	CHECK(AnnIndexEvaluate(file, program, size, words, &count, 0, &how) == ANN_FILTER_OK);
	CHECK(AnnIndexPredicatePut(file, id, ANN_BITMAP_PREDICATE, words, nwords, expandLimit, how.generation,
							   how.remapGeneration, how.columnsOnly, &entry) == ANN_OK);
	CHECK(entry->count == count);
	free(words);
	return entry;
}

/* The cached entry under id equals a fresh evaluation built the same way. */
static void
CheckFresh(uint64_t id, const unsigned char *program, size_t size, uint64_t expandLimit)
{
	uint32_t   *words = malloc(nwords * 4);
	uint64_t	count;
	AnnIndexEvaluation how;
	AnnIndexBitmap *fresh;
	AnnIndexBitmap *entry = AnnIndexBitmapGet(file, id, ANN_BITMAP_PREDICATE);

	CHECK(words != NULL && entry != NULL);
	CHECK(AnnIndexEvaluate(file, program, size, words, &count, 0, &how) == ANN_FILTER_OK);
	CHECK(AnnIndexBitmapTransient(file, id, ANN_BITMAP_PREDICATE, words, nwords, expandLimit, &fresh) == ANN_OK);
	CheckSame(entry, fresh);
	AnnIndexBitmapRelease(file, fresh);
	AnnIndexBitmapRelease(file, entry);
	free(words);
}

/*
 * d2: patching an entry with device copies: the patch has none and uploads at
 * its next lookup; the old copies go with the old entry's last reference.
 */
static void
TestPatchDeviceCopies(void)
{
	unsigned char program[32];
	size_t		size = ValuesProgram(program, 2);
	AnnIndexBitmap *old = Cache(20, program, size, UINT64_MAX);
	AnnIndexBitmap *patched;
	uint64_t	deviceBytes = file->predicateDeviceBytes;
	uint64_t	patches = file->predicatePatched;

	CHECK(old->device != NULL && deviceBytes >= 64);
	uploads = releases = 0;
	Delete(9);
	CHECK(file->predicatePatched == patches + 1);
	CHECK((patched = AnnIndexBitmapGet(file, 20, ANN_BITMAP_PREDICATE)) != NULL && patched != old);
	CHECK(patched->device == NULL && patched->uploadDeferred == 1 && !Holds(patched, 9));
	CHECK(patched->count == old->count - 1 && file->predicateDeviceBytes == deviceBytes - 64);
	CHECK(releases == 0);
	AnnIndexBitmapRelease(file, old);	/* the search's: the last one */
	CHECK(releases == 1);
	AnnIndexBitmapUploadDeferred(file, patched);
	AnnIndexBitmapUploadDeferred(file, patched);
	CHECK(uploads == 1 && patched->device != NULL && file->predicateDeviceBytes == deviceBytes);
	AnnIndexBitmapRelease(file, patched);
	CheckFresh(20, program, size, UINT64_MAX);
}

/*
 * d3 and d4: after deletes, patched entries equal fresh evaluations: dense,
 * sparse, negated, one crossing the expand limit (it gains rows) and one
 * crossing rows / 64 (it keeps only rows).
 */
static void
TestPatchMatchesEvaluation(void)
{
	struct
	{
		uint64_t	id;
		char		column;
		int			negate;
		int32_t		values[2];
		uint32_t	n;
		uint64_t	expandLimit;
	}			cases[] = {
		{30, 'a', 0, {4, 0}, 1, UINT64_MAX},	/* dense */
		{31, 'b', 0, {41, 0}, 1, UINT64_MAX},	/* sparse */
		{32, 'a', 1, {1, 0}, 1, UINT64_MAX},	/* negated */
		{33, 'a', 0, {6, 0}, 1, 0}, /* expand limit set below */
		{34, 'b', 0, {7, 8}, 2, UINT64_MAX},	/* 100 rows, then under 78 */
	};
	unsigned char programs[5][32];
	size_t		sizes[5];
	uint64_t	patches = file->predicatePatched;
	uint64_t	dropped = file->predicateDropped;
	AnnIndexBitmap *entry;
	uint64_t	rows[64];
	uint32_t	nrows = 0;
	uint64_t	added;

	AnnIndexConfig.device_bitmaps = 0;
	for (int c = 0; c < 5; c++)
	{
		sizes[c] = ColumnProgram(programs[c], cases[c].column, cases[c].negate, cases[c].values, cases[c].n);
		if (c == 3)
		{
			entry = Cache(cases[c].id, programs[c], sizes[c], UINT64_MAX);
			cases[c].expandLimit = entry->count - 2;
			AnnIndexBitmapRelease(file, entry);
		}
		entry = Cache(cases[c].id, programs[c], sizes[c], cases[c].expandLimit);
		CHECK(c != 1 || entry->words == NULL);
		CHECK(c != 3 || entry->ordinals == NULL);
		CHECK(c != 4 || (entry->words != NULL && entry->count == 100));
		AnnIndexBitmapRelease(file, entry);
	}

	/*
	 * One delete hitting every case: 23 rows of b in (7, 8), two of b = 41
	 * (group 6), and three each of groups 4 and 6.
	 */
	for (uint64_t row = 0; nrows < 23; row++)
		if (row % BUCKETS == 7 || row % BUCKETS == 8)
			rows[nrows++] = row;
	rows[nrows++] = 41;
	rows[nrows++] = 141;
	for (uint64_t row = 2000, four = 0, six = 0; four < 3 || six < 3; row++)
		if (row % BUCKETS != 7 && row % BUCKETS != 8 &&
			((Group (row) == 4 && four++ < 3) ||(Group (row) == 6 && six++ < 3)))
			rows[nrows++] = row;
	{
		int64_t		keys[64];

		for (uint32_t i = 0; i < nrows; i++)
			keys[i] = Key(rows[nrows - 1 - i]);
		CHECK(AnnIndexDelete(file, keys, nrows, &added) == ANN_OK && added == nrows);
	}
	CHECK(file->predicateDropped == dropped);
	CHECK(file->predicatePatched >= patches + 5);
	for (int c = 0; c < 5; c++)
		CheckFresh(cases[c].id, programs[c], sizes[c], cases[c].expandLimit);
	CHECK((entry = AnnIndexBitmapGet(file, 33, ANN_BITMAP_PREDICATE)) != NULL && entry->ordinals != NULL);
	AnnIndexBitmapRelease(file, entry);
	CHECK((entry = AnnIndexBitmapGet(file, 34, ANN_BITMAP_PREDICATE)) != NULL && entry->words == NULL &&
		  entry->count == 77);
	AnnIndexBitmapRelease(file, entry);
}

/* Past --delete-patch-entries, the entries a delete hits are dropped; off drops them all. */
static void
TestPatchBounds(void)
{
	unsigned char program[32];
	int32_t		values[3] = {70, 71, 72};
	uint64_t	group1[3] = {470, 71, 372}; /* b = 70, 71, 72 in group 1,
											 * which no other cached entry
											 * reads */
	uint64_t	patches;
	uint64_t	dropped;
	int64_t		keys[3];
	uint64_t	added;

	for (int i = 0; i < 3; i++)
	{
		size_t		size = ColumnProgram(program, 'b', 0, &values[i], 1);

		AnnIndexBitmapRelease(file, Cache(40 + (uint64_t) i, program, size, UINT64_MAX));
		keys[i] = Key(group1[i]);
	}
	patches = file->predicatePatched;
	dropped = file->predicateDropped;
	AnnIndexConfig.delete_patch_entries = 1;
	CHECK(AnnIndexDelete(file, keys, 3, &added) == ANN_OK && added == 3);
	CHECK(file->predicatePatched == patches + 1 && file->predicateDropped == dropped + 2);
	AnnIndexConfig.delete_patch_entries = 64;

	AnnIndexConfig.delete_patch_off = 1;
	dropped = file->predicateDropped;
	patches = file->predicatePatched;
	Delete(4);
	CHECK(file->predicatePatched == patches && file->predicateDropped > dropped);
	CHECK(file->predicateEntries == 0 || AnnIndexBitmapGet(file, 30, ANN_BITMAP_PREDICATE) == NULL);
	AnnIndexConfig.delete_patch_off = 0;
}

/*
 * d5's rendezvous: each delete, once it holds the entries it patches, waits
 * up to half a second for the other. With deleteMutex the second cannot get
 * there before the first finishes, so the first goes on alone; without it,
 * both patch the same held entry.
 */
static pthread_mutex_t rendezvousMutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t rendezvousCondition = PTHREAD_COND_INITIALIZER;
static int	arrived;
static uint64_t heldBy[2];

static void
Rendezvous(AnnIndexFile *index, uint64_t nheld)
{
	struct timeval now;
	struct timespec deadline;

	(void) index;
	(void) gettimeofday(&now, NULL);
	deadline.tv_sec = now.tv_sec + (now.tv_usec >= 500000);
	deadline.tv_nsec = (long) ((now.tv_usec + 500000) % 1000000) * 1000;
	(void) pthread_mutex_lock(&rendezvousMutex);
	heldBy[arrived < 2 ? arrived : 1] = nheld;
	arrived++;
	(void) pthread_cond_broadcast(&rendezvousCondition);
	while (arrived < 2 &&
		   pthread_cond_timedwait(&rendezvousCondition, &rendezvousMutex, &deadline) == 0)
		;
	(void) pthread_mutex_unlock(&rendezvousMutex);
}

typedef struct DeleteRows
{
	int64_t		keys[2];
	uint64_t	added;
	AnnStatus	status;
} DeleteRows;

static void *
DeleteThread(void *argument)
{
	DeleteRows *rows = argument;

	rows->status = AnnIndexDelete(file, rows->keys, 2, &rows->added);
	return NULL;
}

/*
 * d5: two deletes of disjoint rows of one cached entry at once: the entry
 * linked afterwards has neither's rows, and equals a fresh evaluation.
 */
static void
TestConcurrentDeletes(void)
{
	unsigned char program[32];
	size_t		size = ValuesProgram(program, 0);
	uint64_t	rows[2][2] = {{3507, 3514}, {3521, 3528}};	/* group 0 */
	DeleteRows	deletes[2];
	pthread_t	threads[2];
	AnnIndexBitmap *entry;
	uint64_t	patches;

	for (int t = 0; t < 2; t++)
		for (int i = 0; i < 2; i++)
		{
			CHECK(Group (rows[t][i]) == 0 && !AnnIndexIsDeleted(file, rows[t][i]));
			deletes[t].keys[i] = Key(rows[t][i]);
		}
	entry = Cache(50, program, size, UINT64_MAX);
	CHECK(entry->words != NULL);
	for (int t = 0; t < 2; t++)
		for (int i = 0; i < 2; i++)
			CHECK(Holds(entry, rows[t][i]));
	AnnIndexBitmapRelease(file, entry);
	patches = file->predicatePatched;
	arrived = 0;
	AnnIndexDeleteHeldHook = Rendezvous;
	for (int t = 0; t < 2; t++)
		CHECK(pthread_create(&threads[t], NULL, DeleteThread, &deletes[t]) == 0);
	for (int t = 0; t < 2; t++)
	{
		CHECK(pthread_join(threads[t], NULL) == 0);
		CHECK(deletes[t].status == ANN_OK && deletes[t].added == 2);
	}
	AnnIndexDeleteHeldHook = NULL;
	CHECK(arrived == 2 && heldBy[0] > 0 && heldBy[1] > 0);
	CHECK((entry = AnnIndexBitmapGet(file, 50, ANN_BITMAP_PREDICATE)) != NULL);
	for (int t = 0; t < 2; t++)
		for (int i = 0; i < 2; i++)
			CHECK(!Holds(entry, rows[t][i]));
	CHECK(Holds(entry, 3500) && Holds(entry, 3535));
	AnnIndexBitmapRelease(file, entry);
	CHECK(file->predicatePatched >= patches + 2);
	CheckFresh(50, program, size, UINT64_MAX);
}

int
main(void)
{
	CHECK(mkdtemp(directory) != NULL);
	WriteIndex();
	CHECK(AnnIndexFileOpen("docs", directory, &file) == ANN_OK);
	TestRacedPut();
	TestDeleteKeepsSnapshots();
	TestPatchDeviceCopies();
	TestPatchMatchesEvaluation();
	TestPatchBounds();
	TestConcurrentDeletes();
	file->device = NULL;
	AnnIndexFileClose(file);
	RemoveIndex();
	printf("predicate cache: raced puts re-masked, snapshots kept and entries patched across deletes, "
		   "concurrent ones included\n");
	return 0;
}
