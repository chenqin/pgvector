/* getaddrinfo and TCP keepalive options are POSIX/GNU, not strict C11. */
#ifdef __linux__
#define _GNU_SOURCE
#endif

#include "server.h"
#include "indexfile.h"
#include "filter.h"
#include "rdma.h"

#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

typedef struct Connection
{
	int			fd;
	int			references;
	int			broken;
	int			remote;			/* accepted on the TCP listener */
	int			authenticated;
	const unsigned char *shared;	/* client-written read-only vectors */
	size_t		shared_size;
	AnnRdma    *rdma;			/* set once the client connected a queue pair */
	uint64_t	peer;
	pthread_mutex_t mutex;		/* serializes replies and guards references */
} Connection;

#define JOB_GATHERED 0
#define JOB_INDEX 1

typedef struct Job
{
	int			kind;
	AnnGatheredJob job;
	AnnIndexJob index;
	int64_t    *keys;			/* sorted filter keys owned by the job */
	uint64_t   *ordinals;		/* their ordinals, when owned by the job */
	uint32_t   *shardBitsets[ANN_INDEX_MAX_SHARDS]; /* bitmap route, per shard */
	uint32_t   *ownedBitsets[ANN_INDEX_MAX_SHARDS]; /* those this job
													 * allocated */
	AnnIndexFile *bitmapFile;	/* cached bitmap this job references */
	AnnIndexBitmap *bitmap;
	Connection *connection;
	int			rdma;			/* reply over RDMA, as requested */
	uint16_t	operation;
	uint32_t	flags;			/* operation 22 flags */
	uint32_t	wanted;			/* hits to return when over-fetching past
								 * tombstones, else 0 */
	uint64_t	request;
	unsigned char *payload;
	uint64_t	charge;
	struct Job *prev;			/* the queue, oldest first */
	struct Job *next;
	struct Job *fnext;			/* the same filter's jobs, or the plain jobs
								 * of this kind, oldest first */
	struct FilterEntry *filter; /* NULL for a job that cannot share a scan */
	uint32_t	compositePart;	/* query position for operation 30 */
	struct Composite *composite;	/* operation 25: the request this part
									 * answers into */
	uint64_t	arrived;		/* --trace: frame read (ns, realtime) */
	uint64_t	queued;			/* decoded and queued */
	uint64_t	started;		/* taken by a lane */
} Job;

/*
 * Operation 25: several loaded indexes (parts of a collection on this node)
 * searched with one program and query. Each part runs as an operation 22
 * job; each hands its hits here, and the last merges them and replies once.
 */
typedef struct BatchPart
{
	uint32_t	count;
	uint32_t	route;
	uint64_t	eligible;
} BatchPart;

typedef struct Composite
{
	pthread_mutex_t mutex;
	uint32_t	remaining;		/* parts not yet answered */
	uint32_t	parts;
	uint32_t	k;
	AnnStatus	status;
	BatchPart  *batch;			/* operation 30: separate results per query */
	uint32_t	flags;
	uint64_t	eligible;
	uint32_t	nhits;
	AnnIndexHit *hits;			/* up to k per part; ordinal: position in
								 * values */
	int			attributes;		/* ANN_PREDICATE_ATTRIBUTES was asked */
	int32_t		nattributes;	/* per hit; -1 until the first part answers */
	int32_t    *values;			/* nattributes per hit, as received */
	Connection *connection;		/* referenced until the reply */
	int			rdma;
	uint64_t	request;
} Composite;

/*
 * --trace: one line per sampled request, "W id rdma arrived queued started
 * done replied" in realtime nanoseconds, so a coordinator's trace of
 * the same requests on this host lines up (benchmarks/ann/fanout_trace.py).
 */
static FILE *traceFile = NULL;
static pthread_mutex_t traceMutex = PTHREAD_MUTEX_INITIALIZER;
static _Thread_local uint64_t frameArrived = 0;

/* Hits to return when SubmitBitmapJob's caller over-fetches (else 0). */
static _Thread_local uint32_t submittingWanted = 0;

/* Set while a reader or RDMA lane submits a composite's parts. */
static _Thread_local Composite *submittingComposite = NULL;
static _Thread_local uint32_t submittingPart = 0;

/*
 * A filter the worker already holds (a cached bitmap's rows, an attribute
 * value's rows), by its shared row arrays: its queued jobs, oldest first,
 * and the lanes working on it. An entry is freed once no queued or running
 * search uses it, so cache churn doesn't grow the table; while it exists,
 * its arrays are referenced by its jobs, so the pointers can't be reused.
 */
typedef struct FilterEntry
{
	const void *ordinals;
	const void *bitsets;
	uint32_t	nfilter;
	uint32_t	queued;
	uint32_t	active;
	Job		   *head;
	Job		   *tail;
	struct FilterEntry *next;	/* hash chain */
} FilterEntry;

#define ANN_FILTER_BUCKETS 4096

typedef struct Service
{
	const AnnBackend *backend;
	AnnServeOptions options;
	pthread_mutex_t mutex;
	pthread_cond_t condition;
	Job		   *head;
	Job		   *tail;
	uint64_t	pending_bytes;
	uint64_t	queued;
	uint32_t	idle;
	int			stopping;
	uint64_t	jobs;
	uint64_t	rejected;
	uint64_t	rejected_index; /* of those, over an index's share */
	uint64_t	failed;
	uint64_t	batches;
	uint64_t	max_batch;
	uint64_t	rows;
	uint64_t	groups;			/* batches that pulled same-filter jobs */
	uint64_t	grouped_jobs;	/* jobs in those groups */
	uint64_t	held;			/* pickups that found only jobs whose filter
								 * enough lanes were already on */
	uint64_t	predicate_hits; /* operation 22 programs found cached */
	uint64_t	predicate_evaluations;
	uint64_t	predicate_eval_us;	/* time spent evaluating them */
	uint32_t	blockedWaiters; /* lanes waiting for a filter to free up */
	uint32_t	activeLanes;	/* lanes working on cached filters */
	FilterEntry *filters[ANN_FILTER_BUCKETS];
	Job		   *plainHead[2];	/* jobs that cannot share a scan, by kind */
	Job		   *plainTail[2];
	AnnRdma    *rdma;

	/*
	 * The RDMA peer table has a lock of its own: every RDMA frame looks its
	 * peer up, and taking the scheduler's mutex for that as well as for
	 * queueing the job doubled the traffic on the busiest lock.
	 */
	pthread_mutex_t peersMutex;
	Connection **peers;			/* by RDMA peer id, under peersMutex */
	uint64_t	npeers;
	uint64_t	rdma_frames;	/* atomic */
} Service;

static volatile sig_atomic_t stopRequested;

void
AnnServeDefaultOptions(AnnServeOptions *options)
{
	long		cpus = sysconf(_SC_NPROCESSORS_ONLN);

	options->lanes = cpus > 0 ? (uint32_t) cpus : 2;
	options->max_batch_jobs = 64;
	options->filter_lanes = 2;
	options->max_batch_bytes = 32U * 1024U * 1024U;
	options->max_pending_bytes = 512U * 1024U * 1024U;
	options->delay_us = 0;
	options->index_pending_share = 100;
	options->trace = NULL;
	options->trace_sample = 64;
	options->tcp = NULL;
	options->token = NULL;
	options->rdma = NULL;
	options->rdma_lanes = 1;
	options->rdma_spin = 1;
}

static void
OnSignal(int signal)
{
	(void) signal;
	stopRequested = 1;
}

static uint64_t
Now(void)
{
	struct timespec ts;

	(void) clock_gettime(CLOCK_REALTIME, &ts);
	return (uint64_t) ts.tv_sec * UINT64_C(1000000000) + (uint64_t) ts.tv_nsec;
}

static int
LittleEndian(void)
{
	const uint16_t value = 1;

	return *(const unsigned char *) &value == 1;
}

static uint64_t
Read(const unsigned char *data, unsigned int width)
{
	uint64_t	value = 0;

	for (unsigned int i = 0; i < width; i++)
		value |= (uint64_t) data[i] << (8 * i);
	return value;
}

static void
Write(unsigned char *data, uint64_t value, unsigned int width)
{
	for (unsigned int i = 0; i < width; i++)
		data[i] = (unsigned char) (value >> (8 * i));
}

static int
ReadFull(int fd, unsigned char *data, size_t size)
{
	while (size)
	{
		ssize_t		n = read(fd, data, size);

		if (n < 0 && errno == EINTR)
			continue;
		if (n <= 0)
			return 0;
		data += n;
		size -= (size_t) n;
	}
	return 1;
}

static int
WriteFull(int fd, const unsigned char *data, size_t size)
{
	while (size)
	{
		ssize_t		n = write(fd, data, size);

		if (n < 0 && errno == EINTR)
			continue;
		if (n <= 0)
			return 0;
		data += n;
		size -= (size_t) n;
	}
	return 1;
}

static void
Release(Connection *connection)
{
	int			last;

	(void) pthread_mutex_lock(&connection->mutex);
	last = --connection->references == 0;
	(void) pthread_mutex_unlock(&connection->mutex);
	if (last)
	{
		/* No job or reader can send to the queue pair any more. */
		if (connection->rdma)
			AnnRdmaRemovePeer(connection->rdma, connection->peer);
		(void) close(connection->fd);
		if (connection->shared)
			(void) munmap((void *) connection->shared, connection->shared_size);
		(void) pthread_mutex_destroy(&connection->mutex);
		free(connection);
	}
}

/* Replies are written whole under the connection mutex, in completion order. */
static void
Reply(Connection *connection, uint16_t operation, uint64_t request,
	  AnnStatus status, const unsigned char *data, size_t size)
{
	unsigned char header[32] = {'P', 'V', 'A', 'N'};

	Write(header + 4, 1, 2);
	Write(header + 6, operation, 2);
	Write(header + 8, request, 8);
	Write(header + 16, 0, 8);
	Write(header + 24, size, 4);
	Write(header + 28, status, 4);
	(void) pthread_mutex_lock(&connection->mutex);
	if (!connection->broken &&
		(!WriteFull(connection->fd, header, sizeof(header)) ||
		 (size && !WriteFull(connection->fd, data, size))))
	{
		connection->broken = 1;
		(void) shutdown(connection->fd, SHUT_RDWR);
	}
	(void) pthread_mutex_unlock(&connection->mutex);
}

/* Reply on the transport the request used. */
static void
SendReply(Connection *connection, int rdma, uint16_t operation, uint64_t request,
		  AnnStatus status, const unsigned char *data, size_t size)
{
	unsigned char header[32] = {'P', 'V', 'A', 'N'};

	if (!rdma)
	{
		Reply(connection, operation, request, status, data, size);
		return;
	}
	Write(header + 4, 1, 2);
	Write(header + 6, operation, 2);
	Write(header + 8, request, 8);
	Write(header + 16, 0, 8);
	Write(header + 24, size, 4);
	Write(header + 28, status, 4);

	/* Replies larger than an RDMA frame fall back to the TCP connection. */
	if (!AnnRdmaSend(connection->rdma, connection->peer, header, sizeof(header), data, size))
		Reply(connection, operation, request, status, data, size);
}

static int
HitCompare(const void *a, const void *b)
{
	const AnnIndexHit *x = a;
	const AnnIndexHit *y = b;

	if (x->distance != y->distance)
		return x->distance < y->distance ? -1 : 1;
	return (x->key > y->key) - (x->key < y->key);
}

/*
 * ANN_PREDICATE_ATTRIBUTES: each hit's attribute values (int32, the
 * index's attribute order), read from the mapped columns by ordinal.
 */
static void
HitAttributes(const AnnIndexFile *file, const AnnIndexHit *hits, uint32_t nhits, int32_t *out)
{
	for (uint32_t i = 0; i < nhits; i++)
		for (uint32_t a = 0; a < file->nattributes; a++)
			out[(size_t) i * file->nattributes + a] = hits[i].ordinal < file->rows && file->attributes[a].column ?
				file->attributes[a].column[hits[i].ordinal] : INT32_MIN;
}

/* One reply for a matrix of queries, preserving query and distance order. */
static void
BatchAnswer(Composite *composite, uint32_t part, uint32_t route, AnnStatus status,
			const AnnIndexHit *hits, uint32_t nhits, uint64_t eligible)
{
	int			last;

	(void) pthread_mutex_lock(&composite->mutex);
	if (status == ANN_OK)
	{
		uint32_t	n = nhits < composite->k ? nhits : composite->k;

		if (n)
			memcpy(composite->hits + (size_t) part * composite->k, hits, sizeof(AnnIndexHit) * n);
		composite->batch[part].count = n;
		composite->batch[part].route = route;
		composite->batch[part].eligible = eligible;
	}
	else if (composite->status == ANN_OK)
		composite->status = status;
	last = --composite->remaining == 0;
	(void) pthread_mutex_unlock(&composite->mutex);
	if (last)
	{
		size_t		size = 4;
		unsigned char *data = NULL;
		AnnStatus	result = composite->status;

		for (uint32_t p = 0; p < composite->parts; p++)
			size += 16 + (size_t) composite->batch[p].count * 12;
		if (result == ANN_OK && (data = malloc(size)) == NULL)
			result = ANN_NOMEM;
		if (result == ANN_OK)
		{
			size_t		offset = 4;

			Write(data, composite->parts, 4);
			for (uint32_t p = 0; p < composite->parts; p++)
			{
				BatchPart  *partResult = &composite->batch[p];

				Write(data + offset, partResult->count, 4);
				Write(data + offset + 4, partResult->route, 4);
				Write(data + offset + 8, partResult->eligible, 8);
				offset += 16;
				for (uint32_t i = 0; i < partResult->count; i++)
				{
					AnnIndexHit *hit = &composite->hits[(size_t) p * composite->k + i];
					uint32_t	bits;

					memcpy(&bits, &hit->distance, sizeof(bits));
					Write(data + offset, (composite->flags & ANN_PREDICATE_ORDINALS) ?
						  hit->ordinal : (uint64_t) hit->key, 8);
					Write(data + offset + 8, bits, 4);
					offset += 12;
				}
			}
		}
		SendReply(composite->connection, composite->rdma, ANN_OP_INDEX_SEARCH_BATCH,
				  composite->request, result, data, result == ANN_OK ? size : 0);
		free(data);
		Release(composite->connection);
		(void) pthread_mutex_destroy(&composite->mutex);
		free(composite->hits);
		free(composite->batch);
		free(composite);
	}
}

/*
 * A part of a composite is answered (status, its hits and eligible rows);
 * the last one replies with the k nearest of all parts: u32 count, u32
 * route (0), u64 eligible rows over every part, then (i64 key, f32
 * distance) per hit, ascending.
 */
static void
CompositeAnswer(Composite *composite, uint32_t part, uint32_t route, AnnStatus status, const AnnIndexHit *hits, uint32_t nhits,
				uint64_t eligible, const AnnIndexFile *file)
{
	int			last;

	if (composite->batch)
	{
		BatchAnswer(composite, part, route, status, hits, nhits, eligible);
		return;
	}
	(void) pthread_mutex_lock(&composite->mutex);
	if (status == ANN_OK && composite->attributes && file &&
		composite->nattributes >= 0 && (uint32_t) composite->nattributes != file->nattributes)
		status = ANN_INVALID;	/* parts with different attributes */
	if (status == ANN_OK && composite->attributes && file && composite->values == NULL)
	{
		composite->nattributes = (int32_t) file->nattributes;
		composite->values = malloc(sizeof(int32_t) * ((size_t) composite->k * composite->parts *
													  (file->nattributes ? file->nattributes : 1)));
		if (composite->values == NULL)
			status = ANN_NOMEM;
	}
	if (status == ANN_OK)
	{
		uint32_t	n = nhits < composite->k ? nhits : composite->k;

		memcpy(composite->hits + composite->nhits, hits, sizeof(AnnIndexHit) * n);
		if (composite->attributes && file)
		{
			HitAttributes(file, hits, n, composite->values + (size_t) composite->nhits * file->nattributes);
			for (uint32_t i = 0; i < n; i++)
				composite->hits[composite->nhits + i].ordinal = composite->nhits + i;
		}
		composite->nhits += n;
		composite->eligible += eligible;
	}
	else if (composite->status == ANN_OK)
		composite->status = status;
	last = --composite->remaining == 0;
	(void) pthread_mutex_unlock(&composite->mutex);
	if (!last)
		return;
	{
		uint32_t	count = composite->nhits < composite->k ? composite->nhits : composite->k;
		uint32_t	nattributes = composite->nattributes > 0 ? (uint32_t) composite->nattributes : 0;
		size_t		size = 16 + (size_t) count * 12 +
			(composite->attributes ? 4 + (size_t) count * nattributes * 4 : 0);
		unsigned char *data = NULL;
		AnnStatus	result = composite->status;

		if (result == ANN_OK)
		{
			qsort(composite->hits, composite->nhits, sizeof(AnnIndexHit), HitCompare);
			if ((data = malloc(size)) == NULL)
				result = ANN_NOMEM;
			else
			{
				Write(data, count, 4);
				Write(data + 4, 0, 4);
				Write(data + 8, composite->eligible, 8);
				for (uint32_t i = 0; i < count; i++)
				{
					uint32_t	bits;

					memcpy(&bits, &composite->hits[i].distance, sizeof(bits));
					Write(data + 16 + (size_t) i * 12, (uint64_t) composite->hits[i].key, 8);
					Write(data + 24 + (size_t) i * 12, bits, 4);
				}
				if (composite->attributes)
				{
					unsigned char *at = data + 16 + (size_t) count * 12;

					Write(at, nattributes, 4);
					for (uint32_t i = 0; i < count; i++)
						for (uint32_t a = 0; a < nattributes; a++)
							Write(at + 4 + ((size_t) i * nattributes + a) * 4,
								  (uint32_t) composite->values[composite->hits[i].ordinal * nattributes + a], 4);
				}
			}
		}
		SendReply(composite->connection, composite->rdma, ANN_OP_COLLECTION_SEARCH, composite->request, result,
				  data, result == ANN_OK ? size : 0);
		free(data);
		Release(composite->connection);
		(void) pthread_mutex_destroy(&composite->mutex);
		free(composite->hits);
		free(composite->values);
		free(composite);
	}
}

static void
Finish(Service *service, Job *job)
{
	unsigned char *data = NULL;
	size_t		size = 0;
	uint64_t	finished = traceFile ? Now() : 0;

	/* Tombstoned rows never come back, whichever route found them. */
	if (job->kind == JOB_INDEX && job->bitmapFile && job->index.status == ANN_OK &&
		!(job->operation == ANN_OP_INDEX_SEARCH_BATCH && (job->flags & ANN_PREDICATE_SNAPSHOT)) &&
		__atomic_load_n(&job->bitmapFile->ndeleted, __ATOMIC_ACQUIRE) > 0)
	{
		uint32_t	kept = 0;

		for (uint32_t i = 0; i < job->index.nresults; i++)
			if (!AnnIndexIsDeleted(job->bitmapFile, job->index.results[i].ordinal))
				job->index.results[kept++] = job->index.results[i];
		job->index.nresults = kept;
	}
	if (job->kind == JOB_INDEX && job->wanted && job->index.nresults > job->wanted)
		job->index.nresults = job->wanted;

	if (job->composite)
	{
		job->job.status = job->index.status;
		CompositeAnswer(job->composite, job->compositePart, job->index.route, job->index.status, job->index.results, job->index.nresults,
						job->bitmap ? job->bitmap->count : job->bitmapFile ? job->bitmapFile->rows : 0,
						job->bitmapFile);
	}
	else if (job->kind == JOB_INDEX)
	{
		/*
		 * u32 count, u32 route, then (i64 key, f32 distance) per hit;
		 * operation 22 adds u64 eligible rows after the route, and with
		 * ANN_PREDICATE_ORDINALS returns ordinals instead of keys.
		 */
		size_t		head = job->operation == ANN_OP_INDEX_SEARCH_PREDICATE ? 16 : 8;

		uint32_t	nattributes = 0;

		if (job->operation == ANN_OP_INDEX_SEARCH_PREDICATE && (job->flags & ANN_PREDICATE_ATTRIBUTES) &&
			job->bitmapFile)
			nattributes = job->bitmapFile->nattributes;
		job->job.status = job->index.status;
		if (job->index.status == ANN_OK)
		{
			size = head + (size_t) job->index.nresults * 12;
			if (job->operation == ANN_OP_INDEX_SEARCH_PREDICATE && (job->flags & ANN_PREDICATE_ATTRIBUTES))
				size += 4 + (size_t) job->index.nresults * nattributes * 4;
			data = malloc(size);
			if (data == NULL)
				job->job.status = ANN_NOMEM;
			else
			{
				Write(data, job->index.nresults, 4);
				Write(data + 4, job->index.route, 4);
				if (head == 16)
					Write(data + 8, job->bitmap ? job->bitmap->count :
						  job->bitmapFile ? job->bitmapFile->rows : 0, 8);
				for (uint32_t i = 0; i < job->index.nresults; i++)
				{
					uint32_t	bits;
					uint64_t	id = (job->flags & ANN_PREDICATE_ORDINALS) ? job->index.results[i].ordinal :
						(uint64_t) job->index.results[i].key;

					memcpy(&bits, &job->index.results[i].distance, sizeof(bits));
					Write(data + head + (size_t) i * 12, id, 8);
					Write(data + head + 8 + (size_t) i * 12, bits, 4);
				}
				if (job->operation == ANN_OP_INDEX_SEARCH_PREDICATE && (job->flags & ANN_PREDICATE_ATTRIBUTES))
				{
					unsigned char *at = data + head + (size_t) job->index.nresults * 12;
					int32_t    *values = malloc(sizeof(int32_t) * ((size_t) job->index.nresults * nattributes + 1));

					Write(at, nattributes, 4);
					if (values == NULL)
					{
						free(data);
						data = NULL;
						job->job.status = ANN_NOMEM;
					}
					else
					{
						HitAttributes(job->bitmapFile, job->index.results, job->index.nresults, values);
						for (size_t i = 0; i < (size_t) job->index.nresults * nattributes; i++)
							Write(at + 4 + i * 4, (uint32_t) values[i], 4);
						free(values);
					}
				}
			}
		}
	}
	else if (job->job.status == ANN_OK)
	{
		size = 4 + (size_t) job->job.nresults * 12;
		data = malloc(size);
		if (data == NULL)
			job->job.status = ANN_NOMEM;
		else
		{
			Write(data, job->job.nresults, 4);
			for (uint32_t i = 0; i < job->job.nresults; i++)
			{
				uint64_t	bits;

				memcpy(&bits, &job->job.results[i].distance, sizeof(bits));
				Write(data + 4 + (size_t) i * 12, job->job.results[i].id, 4);
				Write(data + 8 + (size_t) i * 12, bits, 8);
			}
		}
	}
	if (!job->composite)
		SendReply(job->connection, job->rdma, job->operation, job->request, job->job.status,
				  data, job->job.status == ANN_OK ? size : 0);
	if (traceFile && job->request % service->options.trace_sample == 0)
	{
		(void) pthread_mutex_lock(&traceMutex);
		fprintf(traceFile, "W %llu %d %llu %llu %llu %llu %llu\n", (unsigned long long) job->request, job->rdma,
				(unsigned long long) job->arrived, (unsigned long long) job->queued,
				(unsigned long long) job->started, (unsigned long long) finished, (unsigned long long) Now());
		(void) pthread_mutex_unlock(&traceMutex);
	}
	free(data);
	(void) pthread_mutex_lock(&service->mutex);
	service->pending_bytes -= job->charge;
	if (job->kind == JOB_INDEX && job->bitmapFile)
		job->bitmapFile->pendingBytes -= job->charge;
	if (job->job.status != ANN_OK)
		service->failed++;
	(void) pthread_mutex_unlock(&service->mutex);
	Release(job->connection);
	free(job->job.results);
	free(job->index.results);
	free(job->keys);
	free(job->ordinals);
	for (uint32_t s = 0; s < ANN_INDEX_MAX_SHARDS; s++)
		free(job->ownedBitsets[s]);
	if (job->bitmap)
		AnnIndexBitmapRelease(job->bitmapFile, job->bitmap);
	if (job->bitmapFile)
		AnnIndexPut(job->bitmapFile);
	free(job->payload);
	free(job);
}

/*
 * Index searches over a filter the worker already holds (a cached bitmap's
 * rows, an attribute value's rows) can share one scan: their row lists are
 * the same arrays. Pointer equality is exact because every queued job holds
 * a reference to its entry, so no entry is reused within a batch.
 */
static int
Shareable(const Job *job)
{
	return job->kind == JOB_INDEX && job->index.filtered &&
		(job->index.ordinals != NULL || job->index.bitsets != NULL);
}

#define ANN_GROUP_SCAN 4096		/* queued jobs examined for an eligible one */

static uint64_t
JobBytes(const Job *job)
{
	return (uint64_t) job->job.rows * job->job.dimensions * 4;
}

/*
 * Bucket of a filter's arrays. Every pointer bit counts: arrays a thread
 * allocated under load come from arena chunks whose low bits barely differ,
 * and a hash of the low bits put hundreds of filters in a few buckets, whose
 * chains every enqueue then walked under service->mutex.
 */
static FilterEntry **
FilterBucket(Service *service, const void *ordinals, const void *bitsets, uint32_t nfilter)
{
	uint64_t	hash = (uint64_t) (uintptr_t) ordinals * UINT64_C(0x9E3779B97F4A7C15) ^
		(uint64_t) (uintptr_t) bitsets * UINT64_C(0xC2B2AE3D27D4EB4F) ^ nfilter;

	hash ^= hash >> 31;
	hash *= UINT64_C(0xBF58476D1CE4E5B9);
	hash ^= hash >> 29;
	return &service->filters[hash & (ANN_FILTER_BUCKETS - 1)];
}

/* Drop a filter's entry once no queued or running search uses it. */
static void
FilterDone(Service *service, FilterEntry *entry)
{
	FilterEntry **link;

	if (entry->queued || entry->active || entry->head)
		return;
	for (link = FilterBucket(service, entry->ordinals, entry->bitsets, entry->nfilter); *link;
		 link = &(*link)->next)
		if (*link == entry)
		{
			*link = entry->next;
			free(entry);
			return;
		}
}

/* The entry for a shareable job's filter, created on first sight. */
static FilterEntry *
FilterFor(Service *service, const Job *job)
{
	FilterEntry **bucket = FilterBucket(service, job->index.ordinals, job->index.bitsets, job->index.nfilter);
	FilterEntry *entry;

	for (entry = *bucket; entry; entry = entry->next)
		if (entry->ordinals == job->index.ordinals && entry->bitsets == job->index.bitsets &&
			entry->nfilter == job->index.nfilter)
			return entry;
	if ((entry = calloc(1, sizeof(*entry))) == NULL)
		return NULL;
	entry->ordinals = job->index.ordinals;
	entry->bitsets = job->index.bitsets;
	entry->nfilter = job->index.nfilter;
	entry->next = *bucket;
	*bucket = entry;
	return entry;
}

/* Take a queued job: out of the queue and out of its filter's or kind's list. */
static void
Take(Service *service, Job *job)
{
	Job		  **head = job->filter ? &job->filter->head : &service->plainHead[job->kind];
	Job		  **tail = job->filter ? &job->filter->tail : &service->plainTail[job->kind];

	if (job->prev)
		job->prev->next = job->next;
	else
		service->head = job->next;
	if (job->next)
		job->next->prev = job->prev;
	else
		service->tail = job->prev;
	service->queued--;
	if (traceFile)
		job->started = Now();
	if (job->filter)
		job->filter->queued--;
	if (*head == job)
	{
		*head = job->fnext;
		if (*head == NULL)
			*tail = NULL;
	}
	else
	{
		Job		   *before = *head;

		while (before && before->fnext != job)
			before = before->fnext;
		if (before)
		{
			before->fnext = job->fnext;
			if (*tail == job)
				*tail = before;
		}
	}
	job->prev = job->next = job->fnext = NULL;
}

/* Queue a job: last in the queue and in its filter's or kind's list. */
static void
Append		(Service *service, Job *job)
{
	job->prev = service->tail;
	job->next = NULL;
	job->fnext = NULL;
	if (service->tail)
		service->tail->next = job;
	else
		service->head = job;
	service->tail = job;
	service->queued++;
	job->filter = Shareable(job) ? FilterFor(service, job) : NULL;
	if (job->filter)
	{
		FilterEntry *entry = job->filter;

		entry->queued++;
		if (entry->tail)
			entry->tail->fnext = job;
		else
			entry->head = job;
		entry->tail = job;
	}
	else
	{
		if (service->plainTail[job->kind])
			service->plainTail[job->kind]->fnext = job;
		else
			service->plainHead[job->kind] = job;
		service->plainTail[job->kind] = job;
	}
}

/*
 * The oldest queued job a lane may start on. Once active_lanes lanes work on
 * cached filters, a filter takes no more than filter_lanes lanes per index
 * copy: searches over it arriving meanwhile wait and form one group behind
 * them, so the group size follows the load. Below that, lanes start on any
 * filter, so a few hot filters still keep the GPUs busy (on one GPU, 3 hot
 * filters ran 2x faster with 8 lanes each than with 2, while 10 filters
 * ran fastest with 2 each; 21 of 28 lanes active suits both).
 */
static Job *
Eligible(Service *service)
{
	uint32_t	limit = service->options.filter_lanes ? service->options.filter_lanes : 1;
	uint32_t	target = service->options.active_lanes ? service->options.active_lanes :
		service->options.lanes * 3 / 4 ? service->options.lanes * 3 / 4 : 1;
	uint32_t	scanned = 0;

	limit *= AnnIndexConfig.index_copies ? AnnIndexConfig.index_copies : 1;
	for (Job *job = service->head; job && scanned < ANN_GROUP_SCAN; job = job->next, scanned++)
		if (job->filter == NULL || job->filter->active < limit || service->activeLanes < target)
			return job;
	return NULL;
}

/*
 * A lane takes the oldest job and every queued job that fits the batch
 * limits, optionally waiting delay_us for more, then makes one submission.
 */
/*
 * A batch that failed in the backend may mean a device's context is lost:
 * then every later search on it fails too, and the process keeps its device
 * and pinned memory. Exit instead (without atexit handlers, which could call
 * into the broken context): the memory goes back to the driver, clients fail
 * over to other replicas on the closed connections, and a supervisor
 * restarts the worker.
 */
static void
DeviceLostCheck(Service *service, Job **batch, uint32_t count)
{
	const char *why;
	int			failed = 0;

	for (uint32_t i = 0; i < count && !failed; i++)
		failed = (batch[i]->kind == JOB_INDEX ? batch[i]->index.status : batch[i]->job.status) ==
			ANN_BACKEND_ERROR;
	if (!failed || service->backend->lost == NULL || (why = service->backend->lost()) == NULL)
		return;
	fprintf(stderr, "ann worker: GPU context lost (%s); exiting with status %d to release its memory\n",
			why, ANN_EXIT_DEVICE_LOST);
	fflush(stderr);
	_exit(ANN_EXIT_DEVICE_LOST);
}

static void *
Lane(void *argument)
{
	Service    *service = argument;
	Job		  **batch = calloc(service->options.max_batch_jobs, sizeof(*batch));
	AnnGatheredJob *jobs = calloc(service->options.max_batch_jobs, sizeof(*jobs));
	AnnIndexJob *indexJobs = calloc(service->options.max_batch_jobs, sizeof(*indexJobs));

	if (batch == NULL || jobs == NULL || indexJobs == NULL)
		abort();
	for (;;)
	{
		uint32_t	count = 0;
		uint32_t	limit;
		uint64_t	bytes = 0;
		uint64_t	deadline;
		Job		   *job = NULL;
		FilterEntry *owns = NULL;

		(void) pthread_mutex_lock(&service->mutex);
		service->idle++;
		while (!service->stopping && (job = Eligible(service)) == NULL)
		{
			if (service->head)
			{
				service->held++;
				service->blockedWaiters++;
				(void) pthread_cond_wait(&service->condition, &service->mutex);
				service->blockedWaiters--;
			}
			else
				(void) pthread_cond_wait(&service->condition, &service->mutex);
		}
		service->idle--;
		if (job == NULL)
		{
			(void) pthread_mutex_unlock(&service->mutex);
			break;
		}
		deadline = Now() + service->options.delay_us * 1000;

		/*
		 * Leave a share of the queue for idle lanes instead of serializing
		 * it.
		 */
		limit = (uint32_t) ((service->queued + service->idle) / (service->idle + 1));
		if (limit > service->options.max_batch_jobs)
			limit = service->options.max_batch_jobs;
		if (limit < 1)
			limit = 1;
		Take(service, job);
		batch[count++] = job;
		bytes += JobBytes(job);
		if (job->filter)
		{
			uint32_t	pulled = 0;

			/*
			 * This lane now works on the filter: it takes the filter's other
			 * queued jobs too, and the backend scores their rows once.
			 */
			owns = job->filter;
			owns->active++;
			service->activeLanes++;
			while (count < limit && owns->head)
			{
				Job		   *same = owns->head;

				Take(service, same);
				batch[count++] = same;
				pulled++;
			}
			if (pulled)
			{
				service->groups++;
				service->grouped_jobs += pulled + 1;
			}
		}

		/*
		 * Fill with jobs of the same kind that cannot share a filter; other
		 * filters stay queued for other lanes. Optionally hold a short batch
		 * while no lane is idle.
		 */
		for (;;)
		{
			Job		   *plain;

			while (count < limit && (plain = service->plainHead[batch[0]->kind]) != NULL &&
				   bytes + JobBytes(plain) <= service->options.max_batch_bytes)
			{
				Take(service, plain);
				batch[count++] = plain;
				bytes += JobBytes(plain);
			}
			if (count >= limit || service->idle || service->stopping || Now() >= deadline)
				break;
			{
				struct timespec ts = {(time_t) (deadline / 1000000000),
				(long) (deadline % 1000000000)};

				(void) pthread_cond_timedwait(&service->condition, &service->mutex, &ts);
			}
			limit = service->options.max_batch_jobs;
		}
		service->batches++;
		service->jobs += count;
		if (count > service->max_batch)
			service->max_batch = count;
		for (uint32_t i = 0; i < count; i++)
			service->rows += batch[i]->job.rows;
		(void) pthread_mutex_unlock(&service->mutex);
		if (batch[0]->kind == JOB_INDEX)
		{
			for (uint32_t i = 0; i < count; i++)
				indexJobs[i] = batch[i]->index;
			service->backend->search_index(indexJobs, count);
			for (uint32_t i = 0; i < count; i++)
				batch[i]->index = indexJobs[i];
		}
		else
		{
			for (uint32_t i = 0; i < count; i++)
				jobs[i] = batch[i]->job;
			service->backend->search_gathered(jobs, count);
			for (uint32_t i = 0; i < count; i++)
				batch[i]->job = jobs[i];
		}
		DeviceLostCheck(service, batch, count);
		for (uint32_t i = 0; i < count; i++)
			Finish(service, batch[i]);
		if (owns)
		{
			/* The filter is free again: one waiting lane may take its queue. */
			(void) pthread_mutex_lock(&service->mutex);
			owns->active--;
			service->activeLanes--;
			FilterDone(service, owns);
			if (service->blockedWaiters)
				(void) pthread_cond_signal(&service->condition);
			(void) pthread_mutex_unlock(&service->mutex);
		}
	}
	free(batch);
	free(jobs);
	free(indexJobs);
	return NULL;
}

/*
 * Inline payload: u32 rows, dimensions, k, metric; FP32 query; row-major FP32
 * vectors. Shared payload: the same four fields, u64 byte offset into the
 * attached segment, then the FP32 query; vectors stay in the segment.
 */
static AnnStatus
Submit(Service *service, Connection *connection, int rdma, uint16_t operation,
	   uint64_t request, unsigned char *payload, uint32_t size)
{
	Job		   *job;
	uint32_t	rows,
				dimensions;
	uint64_t	floats;
	uint64_t	charge;
	size_t		header = operation == ANN_OP_SHARED_SEARCH ? 24 : 16;
	float	   *values = (float *) (void *) (payload + header);
	const float *vectors;

	if (size < header)
		return ANN_INVALID;
	rows = (uint32_t) Read(payload, 4);
	dimensions = (uint32_t) Read(payload + 4, 4);
	floats = operation == ANN_OP_SHARED_SEARCH ? dimensions : ((uint64_t) rows + 1) * dimensions;
	if (rows == 0 || dimensions == 0 || dimensions > ANN_MAX_DIM ||
		Read(payload + 12, 4) > 2 || floats * 4 != (uint64_t) size - header)
		return ANN_INVALID;
	if (service->backend->search_gathered == NULL)
		return ANN_UNSUPPORTED;
	if (operation == ANN_OP_SHARED_SEARCH)
	{
		uint64_t	offset = Read(payload + 16, 8);
		uint64_t	bytes = (uint64_t) rows * dimensions * 4;

		/* The client must not rewrite this range until the reply arrives. */
		if (connection->shared == NULL || offset % 4 != 0 ||
			offset > connection->shared_size || bytes > connection->shared_size - offset)
			return ANN_INVALID;
		vectors = (const float *) (const void *) (connection->shared + offset);
	}
	else
		vectors = values + dimensions;

	/* Decode in place; the payload buffer is malloc-aligned. */
	for (uint64_t i = 0; !LittleEndian() && i < floats; i++)
	{
		uint32_t	bits = (uint32_t) Read(payload + header + i * 4, 4);

		memcpy(&values[i], &bits, sizeof(bits));
	}
	if (operation == ANN_OP_SHARED_SEARCH && !LittleEndian())
		return ANN_UNSUPPORTED;
	charge = (uint64_t) size + (uint64_t) rows * sizeof(AnnCandidate);
	job = calloc(1, sizeof(*job));
	if (job == NULL || (job->job.results = malloc((size_t) rows * sizeof(AnnCandidate))) == NULL)
	{
		free(job);
		return ANN_NOMEM;
	}
	job->job.rows = rows;
	job->job.dimensions = dimensions;
	job->job.k = (uint32_t) Read(payload + 8, 4);
	job->job.metric = (uint32_t) Read(payload + 12, 4);
	job->job.query = values;
	job->job.vectors = vectors;
	job->connection = connection;
	job->operation = operation;
	job->rdma = rdma;
	job->request = request;
	job->payload = payload;
	job->charge = charge;
	(void) pthread_mutex_lock(&service->mutex);
	if (service->pending_bytes + charge > service->options.max_pending_bytes)
	{
		service->rejected++;
		(void) pthread_mutex_unlock(&service->mutex);
		free(job->job.results);
		free(job);
		return ANN_OVERLOADED;
	}
	service->pending_bytes += charge;
	(void) pthread_mutex_lock(&connection->mutex);
	connection->references++;
	(void) pthread_mutex_unlock(&connection->mutex);
	Append		(service, job);

	(void) pthread_cond_signal(&service->condition);
	(void) pthread_mutex_unlock(&service->mutex);
	return ANN_PENDING;
}

/* Map a client's POSIX shared memory segment read-only, once per connection. */
static AnnStatus
Attach(Connection *connection, const unsigned char *name, uint32_t size)
{
	char		path[64];
	struct stat info;
	void	   *mapping;
	int			fd;

	if (connection->shared || size < 8 || size >= sizeof(path) ||
		memcmp(name, "/pgann.", 7) != 0 || memchr(name, '\0', size))
		return ANN_INVALID;
	memcpy(path, name, size);
	path[size] = '\0';
	if (strchr(path + 1, '/'))
		return ANN_INVALID;
	fd = shm_open(path, O_RDONLY, 0);
	if (fd < 0)
		return ANN_INVALID;
	if (fstat(fd, &info) != 0 || info.st_size <= 0)
	{
		(void) close(fd);
		return ANN_INVALID;
	}
	mapping = mmap(NULL, (size_t) info.st_size, PROT_READ, MAP_SHARED, fd, 0);
	(void) close(fd);
	if (mapping == MAP_FAILED)
		return ANN_NOMEM;
	connection->shared = mapping;
	connection->shared_size = (size_t) info.st_size;
	return ANN_OK;
}

/* Admit a prepared job and queue it; false when over the pending budget. */
static int
Enqueue(Service *service, Connection *connection, Job *job)
{
	AnnIndexFile *file = job->kind == JOB_INDEX ? job->bitmapFile : NULL;

	(void) pthread_mutex_lock(&service->mutex);
	if (service->pending_bytes + job->charge > service->options.max_pending_bytes)
	{
		service->rejected++;
		(void) pthread_mutex_unlock(&service->mutex);
		return 0;
	}

	/*
	 * --index-pending-share: one index's searches may hold only part of the
	 * budget, so a flood on one index leaves room for the others.
	 */
	if (file && service->options.index_pending_share < 100 &&
		file->pendingBytes + job->charge >
		service->options.max_pending_bytes / 100 * service->options.index_pending_share)
	{
		service->rejected++;
		service->rejected_index++;
		(void) pthread_mutex_unlock(&service->mutex);
		return 0;
	}
	if (file)
		file->pendingBytes += job->charge;
	service->pending_bytes += job->charge;
	(void) pthread_mutex_lock(&connection->mutex);
	connection->references++;
	(void) pthread_mutex_unlock(&connection->mutex);
	if (traceFile)
	{
		job->arrived = frameArrived;
		job->queued = Now();
	}
	Append		(service, job);

	(void) pthread_cond_signal(&service->condition);
	(void) pthread_mutex_unlock(&service->mutex);
	return 1;
}

static int
KeyCompare(const void *a, const void *b)
{
	int64_t		x = *(const int64_t *) a;
	int64_t		y = *(const int64_t *) b;

	return x < y ? -1 : x > y;
}

/*
 * Strictly increasing keys: a zigzag varint for the first, then varint
 * gaps. Dense filters cost one or two bytes per key. Returns 0 unless the
 * bytes hold exactly count keys.
 */
static int
DecodeKeys(const unsigned char *data, uint32_t bytes, uint64_t count, int64_t *keys)
{
	const unsigned char *end = data + bytes;
	uint64_t	previous = 0;

	for (uint64_t i = 0; i < count; i++)
	{
		uint64_t	value = 0;
		int			shift = 0;

		do
		{
			if (data == end || shift > 63)
				return 0;
			value |= (uint64_t) (*data & 0x7f) << shift;
			shift += 7;
		} while (*data++ & 0x80);
		if (i == 0)
			previous = (value >> 1) ^ (~(value & 1) + 1);
		else if (value == 0)
			return 0;
		else
			previous += value;
		keys[i] = (int64_t) previous;
	}
	return data == end;
}

/*
 * Index search payload: u32 name length, name, u32 dimensions, k, filter
 * count (UINT32_MAX for none); FP32 query; i64 filter keys in any order.
 */
static AnnStatus SubmitBitmapJob(Service *service, Connection *connection, int rdma, uint64_t request,
								 unsigned char *payload, uint32_t size, AnnIndexFile *file,
								 AnnIndexBitmap *bitmap, const char *name, uint32_t dimensions,
								 uint32_t k, size_t offset, uint16_t operation, uint32_t flags);
static AnnStatus PredicateBitmap(Service *service, AnnIndexFile *file, const unsigned char *program,
								 uint32_t size, AnnIndexBitmap **result, int snapshot);

static AnnStatus
SubmitIndex(Service *service, Connection *connection, int rdma, uint16_t operation, uint64_t request,
			unsigned char *payload, uint32_t size)
{
	Job		   *job;
	uint32_t	nameLength,
				dimensions,
				k,
				nfilter,
				packedBytes = 0;
	size_t		offset;
	uint64_t	keys;
	int			packed = operation == ANN_OP_INDEX_SEARCH_PACKED;

	if (service->backend->search_index == NULL)
		return ANN_UNSUPPORTED;
	if (size < 4 || (nameLength = (uint32_t) Read(payload, 4)) == 0 ||
		nameLength > ANN_INDEX_NAME_MAX || (uint64_t) size < 16 + (uint64_t) nameLength)
		return ANN_INVALID;
	offset = 4 + nameLength;
	dimensions = (uint32_t) Read(payload + offset, 4);
	k = (uint32_t) Read(payload + offset + 4, 4);
	nfilter = (uint32_t) Read(payload + offset + 8, 4);
	offset += 12;
	if (packed)
	{
		if ((uint64_t) size < offset + 4)
			return ANN_INVALID;
		packedBytes = (uint32_t) Read(payload + offset, 4);
		offset += 4;
	}
	keys = nfilter == UINT32_MAX ? 0 : nfilter;
	if (dimensions == 0 || dimensions > ANN_MAX_DIM || k == 0 || k > 10000 ||
		memchr(payload + 4, '\0', nameLength) ||
		(uint64_t) size != offset + (uint64_t) dimensions * 4 + (packed ? packedBytes : keys * 8))
		return ANN_INVALID;
	if (nfilter == UINT32_MAX)
	{
		char		name[ANN_INDEX_NAME_MAX + 1];
		AnnIndexFile *file;
		AnnIndexBitmap *bitmap = NULL;
		AnnStatus	status = ANN_OK;
		const unsigned char all = ANN_FILTER_ALL;

		memcpy(name, payload + 4, nameLength);
		name[nameLength] = '\0';
		if ((file = AnnIndexGet(name)) == NULL)
			return ANN_STALE;
		/* Legacy searches must honor the same tombstones as operation 22. */
		if (__atomic_load_n(&file->ndeleted, __ATOMIC_ACQUIRE) > 0)
			status = PredicateBitmap(service, file, &all, 1, &bitmap, 0);
		if (status == ANN_OK)
			status = SubmitBitmapJob(service, connection, rdma, request, payload, size, file, bitmap,
									 name, dimensions, k, offset, operation, 0);
		AnnIndexPut(file);
		return status;
	}
	job = calloc(1, sizeof(*job));
	if (job == NULL)
		return ANN_NOMEM;
	job->kind = JOB_INDEX;
	memcpy(job->index.name, payload + 4, nameLength);
	job->index.dimensions = dimensions;
	job->index.k = k;
	job->index.filtered = nfilter != UINT32_MAX;
	job->index.query = (const float *) (const void *) payload;	/* replaced below */
	job->index.results = malloc(sizeof(AnnIndexHit) * k);
	job->keys = malloc(sizeof(int64_t) * (keys ? keys : 1));
	if (job->index.results == NULL || job->keys == NULL)
	{
		free(job->index.results);
		free(job->keys);
		free(job);
		return ANN_NOMEM;
	}

	/* Decode the query in place at an aligned offset and copy the keys. */
	{
		float	   *query = (float *) (void *) payload;

		for (uint32_t d = 0; d < dimensions; d++)
		{
			uint32_t	bits = (uint32_t) Read(payload + offset + (size_t) d * 4, 4);

			memcpy(&query[d], &bits, sizeof(bits));
		}
		job->index.query = query;
	}
	offset += (size_t) dimensions * 4;
	if (packed)
	{
		if (!DecodeKeys(payload + offset, packedBytes, keys, job->keys))
		{
			free(job->index.results);
			free(job->keys);
			free(job);
			return ANN_INVALID;
		}
	}
	else if (LittleEndian())
		memcpy(job->keys, payload + offset, keys * 8);
	else
		for (uint64_t i = 0; i < keys; i++)
			job->keys[i] = (int64_t) Read(payload + offset + i * 8, 8);

	/* Clients normally send sorted keys; sort only when they did not. */
	for (uint64_t i = 1; i < keys; i++)
		if (job->keys[i] < job->keys[i - 1])
		{
			qsort(job->keys, keys, sizeof(int64_t), KeyCompare);
			break;
		}
	job->index.nfilter = 0;
	for (uint64_t i = 0; i < keys; i++)
		if (i == 0 || job->keys[i] != job->keys[i - 1])
			job->keys[job->index.nfilter++] = job->keys[i];
	job->index.filter = job->keys;
	job->connection = connection;
	job->operation = operation;
	job->rdma = rdma;
	job->request = request;
	job->payload = payload;
	job->charge = (uint64_t) size + keys * 8 + sizeof(AnnIndexHit) * k;
	if (!Enqueue(service, connection, job))
	{
		free(job->index.results);
		free(job->keys);
		free(job);
		return ANN_OVERLOADED;
	}
	return ANN_PENDING;
}

/*
 * Attribute search payload: u32 name length, name, u32 dimensions, k,
 * attribute name length, attribute name, value count; i32 values; FP32
 * query. The worker resolves "attribute IN (values)" from the index's own
 * inverted attribute columns, so the caller sends no keys.
 */
static AnnStatus
SubmitMatch(Service *service, Connection *connection, int rdma, uint64_t request,
			unsigned char *payload, uint32_t size)
{
	Job		   *job;
	AnnIndexFile *file;
	char		attribute[64];
	int32_t		values[ANN_INDEX_MAX_VALUES];
	uint32_t	nameLength,
				attributeLength,
				dimensions,
				k,
				nvalues;
	size_t		offset;
	const int64_t *keys;
	const uint64_t *ordinals;
	uint64_t	count;
	uint32_t   *const *bitsets;
	int			owned;
	AnnStatus	status;

	if (service->backend->search_index == NULL)
		return ANN_UNSUPPORTED;
	if (size < 4 || (nameLength = (uint32_t) Read(payload, 4)) == 0 ||
		nameLength > ANN_INDEX_NAME_MAX || (uint64_t) size < 16 + (uint64_t) nameLength ||
		memchr(payload + 4, '\0', nameLength))
		return ANN_INVALID;
	offset = 4 + nameLength;
	dimensions = (uint32_t) Read(payload + offset, 4);
	k = (uint32_t) Read(payload + offset + 4, 4);
	attributeLength = (uint32_t) Read(payload + offset + 8, 4);
	offset += 12;
	if (attributeLength == 0 || attributeLength >= sizeof(attribute) ||
		(uint64_t) size < offset + attributeLength + 4 || memchr(payload + offset, '\0', attributeLength))
		return ANN_INVALID;
	memcpy(attribute, payload + offset, attributeLength);
	attribute[attributeLength] = '\0';
	offset += attributeLength;
	nvalues = (uint32_t) Read(payload + offset, 4);
	offset += 4;
	if (dimensions == 0 || dimensions > ANN_MAX_DIM || k == 0 || k > 10000 ||
		nvalues == 0 || nvalues > ANN_INDEX_MAX_VALUES ||
		(uint64_t) size != offset + (uint64_t) nvalues * 4 + (uint64_t) dimensions * 4)
		return ANN_INVALID;
	for (uint32_t i = 0; i < nvalues; i++)
		values[i] = (int32_t) (uint32_t) Read(payload + offset + (size_t) i * 4, 4);
	offset += (size_t) nvalues * 4;

	/* Loaded indexes are never freed, so the slices outlive the job. */
	{
		char		name[ANN_INDEX_NAME_MAX + 1];

		memcpy(name, payload + 4, nameLength);
		name[nameLength] = '\0';
		if ((file = AnnIndexGet(name)) == NULL)
			return ANN_STALE;
	}
	status = AnnIndexMatch(file, attribute, values, nvalues, &keys, &ordinals, &count, &bitsets, &owned);
	AnnIndexPut(file);
	if (status != ANN_OK)
		return status;
	if (count > UINT32_MAX - 1)
	{
		if (owned)
		{
			free((void *) keys);
			free((void *) ordinals);
		}
		return ANN_INVALID;
	}
	job = calloc(1, sizeof(*job));
	if (job == NULL || (job->index.results = malloc(sizeof(AnnIndexHit) * k)) == NULL)
	{
		free(job);
		if (owned)
		{
			free((void *) keys);
			free((void *) ordinals);
		}
		return ANN_NOMEM;
	}
	job->kind = JOB_INDEX;
	memcpy(job->index.name, payload + 4, nameLength);
	job->index.dimensions = dimensions;
	job->index.k = k;
	job->index.filtered = 1;
	{
		float	   *query = (float *) (void *) payload;

		for (uint32_t d = 0; d < dimensions; d++)
		{
			uint32_t	bits = (uint32_t) Read(payload + offset + (size_t) d * 4, 4);

			memcpy(&query[d], &bits, sizeof(bits));
		}
		job->index.query = query;
	}
	job->index.filter = keys;
	job->index.nfilter = (uint32_t) count;
	job->index.ordinals = ordinals;
	job->index.bitsets = bitsets;
	if (owned)
	{
		job->keys = (int64_t *) keys;
		job->ordinals = (uint64_t *) ordinals;
	}
	job->connection = connection;
	job->operation = ANN_OP_INDEX_SEARCH_MATCH;
	job->rdma = rdma;
	job->request = request;
	job->payload = payload;
	job->charge = (uint64_t) size + sizeof(AnnIndexHit) * k + (owned ? count * 16 : 0);
	if (!Enqueue(service, connection, job))
	{
		free(job->index.results);
		free(job->keys);
		free(job->ordinals);
		free(job);
		return ANN_OVERLOADED;
	}
	return ANN_PENDING;
}

/*
 * Bitmap build payload: u32 name length, name, u32 key count, u32 packed
 * bytes, packed keys (as for packed index search). The reply is u64 eligible
 * rows, then one bit per index row in ordinal (key) order, as little-endian
 * u32 words: bit i of word w is ordinal 32 * w + i. Keys missing from the
 * index are dropped. PostgreSQL stores the reply in ann_filter_bitmap.
 */
static AnnStatus
BuildBitmap(const unsigned char *payload, uint32_t size, unsigned char **reply, size_t *replyLength)
{
	char		name[ANN_INDEX_NAME_MAX + 1];
	uint32_t	nameLength,
				nkeys,
				packedBytes;
	AnnIndexFile *file;
	int64_t    *keys;
	uint64_t   *ordinals;
	uint64_t	n;
	size_t		words;
	unsigned char *data;

	if (size < 12 || (nameLength = (uint32_t) Read(payload, 4)) == 0 ||
		nameLength > ANN_INDEX_NAME_MAX || (uint64_t) size < 12 + (uint64_t) nameLength ||
		memchr(payload + 4, '\0', nameLength))
		return ANN_INVALID;
	memcpy(name, payload + 4, nameLength);
	name[nameLength] = '\0';
	nkeys = (uint32_t) Read(payload + 4 + nameLength, 4);
	packedBytes = (uint32_t) Read(payload + 8 + nameLength, 4);
	if ((uint64_t) size != 12 + (uint64_t) nameLength + packedBytes)
		return ANN_INVALID;
	if ((file = AnnIndexGet(name)) == NULL)
		return ANN_STALE;
	words = (file->rows + 31) / 32;
	keys = malloc(sizeof(int64_t) * (nkeys ? nkeys : 1));
	ordinals = malloc(sizeof(uint64_t) * (nkeys ? nkeys : 1));
	data = calloc(8 + words * 4, 1);
	if (keys == NULL || ordinals == NULL || data == NULL ||
		!DecodeKeys(payload + 12 + nameLength, packedBytes, nkeys, keys))
	{
		AnnStatus	status = keys && ordinals && data ? ANN_INVALID : ANN_NOMEM;

		AnnIndexPut(file);
		free(keys);
		free(ordinals);
		free(data);
		return status;
	}
	n = AnnIndexOrdinals(file, keys, nkeys, ordinals);
	AnnIndexPut(file);
	Write(data, n, 8);
	for (uint64_t i = 0; i < n; i++)
		data[8 + ordinals[i] / 8] |= (unsigned char) (1U << (ordinals[i] % 8));
	free(keys);
	free(ordinals);
	*reply = data;
	*replyLength = 8 + words * 4;
	return ANN_OK;
}

/*
 * Bitmap search payload: u32 name length, name, u32 dimensions, k, bitmap
 * words; FP32 query; the bitmap as built by BuildBitmap, without its count.
 * Broad filters go straight to CAGRA with per-shard slices of the bitmap as
 * cuVS bitsets; filters within the bitset gather limit, and all filters
 * on indexes without device state (CPU backends, files without CAGRA
 * shards), are expanded to ordinals and keys and routed as usual.
 */
static AnnStatus
SubmitBitmap(Service *service, Connection *connection, int rdma, uint64_t request,
			 unsigned char *payload, uint32_t size)
{
	Job		   *job;
	AnnIndexFile *file;
	char		name[ANN_INDEX_NAME_MAX + 1];
	uint32_t	nameLength,
				dimensions,
				k,
				words;
	size_t		offset;
	uint32_t   *bitmap;
	uint64_t	n = 0;
	uint64_t	rows;

	if (service->backend->search_index == NULL)
		return ANN_UNSUPPORTED;
	if (size < 4 || (nameLength = (uint32_t) Read(payload, 4)) == 0 ||
		nameLength > ANN_INDEX_NAME_MAX || (uint64_t) size < 16 + (uint64_t) nameLength ||
		memchr(payload + 4, '\0', nameLength))
		return ANN_INVALID;
	memcpy(name, payload + 4, nameLength);
	name[nameLength] = '\0';
	offset = 4 + nameLength;
	dimensions = (uint32_t) Read(payload + offset, 4);
	k = (uint32_t) Read(payload + offset + 4, 4);
	words = (uint32_t) Read(payload + offset + 8, 4);
	offset += 12;
	if (dimensions == 0 || dimensions > ANN_MAX_DIM || k == 0 || k > 10000 ||
		(uint64_t) size != offset + (uint64_t) dimensions * 4 + (uint64_t) words * 4)
		return ANN_INVALID;
	if ((file = AnnIndexGet(name)) == NULL)
		return ANN_STALE;
	rows = file->rows;
	AnnIndexPut(file);
	if (words != (rows + 31) / 32)
		return ANN_INVALID;
	job = calloc(1, sizeof(*job));
	if (job == NULL || (job->index.results = malloc(sizeof(AnnIndexHit) * k)) == NULL)
	{
		free(job);
		return ANN_NOMEM;
	}

	/*
	 * Decode the query to the front of the payload and move the bitmap right
	 * behind it, both now aligned; clear bits past the last row.
	 */
	{
		float	   *query = (float *) (void *) payload;

		for (uint32_t d = 0; d < dimensions; d++)
		{
			uint32_t	bits = (uint32_t) Read(payload + offset + (size_t) d * 4, 4);

			memcpy(&query[d], &bits, sizeof(bits));
		}
		job->index.query = query;
	}
	offset += (size_t) dimensions * 4;
	bitmap = (uint32_t *) (void *) (payload + (size_t) dimensions * 4);
	memmove(bitmap, payload + offset, (size_t) words * 4);
	if (!LittleEndian())
		for (uint32_t w = 0; w < words; w++)
			bitmap[w] = (uint32_t) Read((unsigned char *) &bitmap[w], 4);
	if (rows % 32)
		bitmap[words - 1] &= (UINT32_C(1) << (rows % 32)) - 1;
	for (uint32_t w = 0; w < words; w++)
		n += (uint64_t) __builtin_popcount(bitmap[w]);

	job->kind = JOB_INDEX;
	strcpy(job->index.name, name);
	job->index.dimensions = dimensions;
	job->index.k = k;
	job->index.filtered = 1;
	job->index.nfilter = (uint32_t) (n < UINT32_MAX ? n : UINT32_MAX - 1);
	if (n <= AnnIndexConfig.bitset_gather_limit || file->device == NULL)
	{
		/* Expand to ordinals and keys; loaded indexes are never freed. */
		uint64_t	i = 0;

		job->keys = malloc(sizeof(int64_t) * (n ? n : 1));
		job->ordinals = malloc(sizeof(uint64_t) * (n ? n : 1));
		if (job->keys == NULL || job->ordinals == NULL)
			goto nomem;
		for (uint32_t w = 0; w < words; w++)
			for (uint32_t bits = bitmap[w]; bits; bits &= bits - 1)
			{
				uint64_t	ordinal = (uint64_t) w * 32 + (uint64_t) __builtin_ctz(bits);

				job->ordinals[i] = ordinal;
				job->keys[i++] = file->keys[ordinal];
			}
		job->index.filter = job->keys;
		job->index.ordinals = job->ordinals;
	}
	else
	{
		/* Each shard's slice: in place when word aligned, else shifted. */
		for (uint32_t s = 0; s < file->nshards; s++)
		{
			uint64_t	first = file->shards[s].first;
			size_t		shardWords = ((size_t) file->shards[s].rows + 31) / 32;
			uint32_t	shift = (uint32_t) (first % 32);

			if (shift == 0)
			{
				job->shardBitsets[s] = bitmap + first / 32;
				continue;
			}
			if ((job->ownedBitsets[s] = malloc(shardWords * 4)) == NULL)
				goto nomem;
			for (size_t w = 0; w < shardWords; w++)
			{
				size_t		at = first / 32 + w;
				uint32_t	low = bitmap[at] >> shift;
				uint32_t	high = at + 1 < words ? bitmap[at + 1] << (32 - shift) : 0;

				job->ownedBitsets[s][w] = low | high;
			}
			job->shardBitsets[s] = job->ownedBitsets[s];
		}
		job->index.bitsets = job->shardBitsets;
	}
	job->connection = connection;
	job->operation = ANN_OP_INDEX_SEARCH_BITMAP;
	job->rdma = rdma;
	job->request = request;
	job->payload = payload;
	job->charge = (uint64_t) size + sizeof(AnnIndexHit) * k + (job->keys ? n * 16 : 0);
	if (Enqueue(service, connection, job))
		return ANN_PENDING;
	job->payload = NULL;
	free(job->index.results);
	free(job->keys);
	free(job->ordinals);
	for (uint32_t s = 0; s < ANN_INDEX_MAX_SHARDS; s++)
		free(job->ownedBitsets[s]);
	free(job);
	return ANN_OVERLOADED;
nomem:
	free(job->index.results);
	free(job->keys);
	free(job->ordinals);
	for (uint32_t s = 0; s < ANN_INDEX_MAX_SHARDS; s++)
		free(job->ownedBitsets[s]);
	free(job);
	return ANN_NOMEM;
}

/*
 * Bitmap put payload: u32 name length, name, u64 filter id, u32 words, then
 * the bitmap as BuildBitmap returns it (without the count). The bitmap is
 * cached under the id until replaced or the worker exits.
 */
static AnnStatus
PutBitmap(const unsigned char *payload, uint32_t size)
{
	char		name[ANN_INDEX_NAME_MAX + 1];
	uint32_t	nameLength,
				words;
	uint64_t	id;
	AnnIndexFile *file;
	uint32_t   *bitmap;
	AnnStatus	status;

	if (size < 4 || (nameLength = (uint32_t) Read(payload, 4)) == 0 ||
		nameLength > ANN_INDEX_NAME_MAX || (uint64_t) size < 16 + (uint64_t) nameLength ||
		memchr(payload + 4, '\0', nameLength))
		return ANN_INVALID;
	memcpy(name, payload + 4, nameLength);
	name[nameLength] = '\0';
	id = Read(payload + 4 + nameLength, 8);
	words = (uint32_t) Read(payload + 12 + nameLength, 4);
	if ((uint64_t) size != 16 + (uint64_t) nameLength + (uint64_t) words * 4)
		return ANN_INVALID;
	if ((bitmap = malloc((size_t) words * 4 + 4)) == NULL)
		return ANN_NOMEM;
	for (uint32_t w = 0; w < words; w++)
		bitmap[w] = (uint32_t) Read(payload + 16 + nameLength + (size_t) w * 4, 4);
	if ((file = AnnIndexGet(name)) == NULL)
	{
		free(bitmap);
		return ANN_STALE;
	}

	/* Keep rows for every filter a scan may take; see SubmitCached. */
	status = AnnIndexBitmapPut(file, id, ANN_BITMAP_PUSHED, bitmap, words, file->device == NULL ? UINT64_MAX :
							   AnnIndexConfig.group_gather_limit > AnnIndexConfig.bitset_gather_limit ?
							   AnnIndexConfig.group_gather_limit : AnnIndexConfig.bitset_gather_limit, NULL);
	AnnIndexPut(file);
	free(bitmap);
	return status;
}

/*
 * Cached bitmap search payload: u32 name length, name, u32 dimensions, k,
 * u64 filter id; FP32 query. ANN_MISSING when the id is not cached here, so
 * the caller can push the bitmap and retry.
 */
static AnnStatus
SubmitCached(Service *service, Connection *connection, int rdma, uint64_t request,
			 unsigned char *payload, uint32_t size)
{
	AnnIndexFile *file;
	AnnIndexBitmap *bitmap;
	char		name[ANN_INDEX_NAME_MAX + 1];
	uint32_t	nameLength,
				dimensions,
				k;
	uint64_t	id;
	size_t		offset;

	if (service->backend->search_index == NULL)
		return ANN_UNSUPPORTED;
	if (size < 4 || (nameLength = (uint32_t) Read(payload, 4)) == 0 ||
		nameLength > ANN_INDEX_NAME_MAX || (uint64_t) size < 20 + (uint64_t) nameLength ||
		memchr(payload + 4, '\0', nameLength))
		return ANN_INVALID;
	memcpy(name, payload + 4, nameLength);
	name[nameLength] = '\0';
	offset = 4 + nameLength;
	dimensions = (uint32_t) Read(payload + offset, 4);
	k = (uint32_t) Read(payload + offset + 4, 4);
	id = Read(payload + offset + 8, 8);
	offset += 16;
	if (dimensions == 0 || dimensions > ANN_MAX_DIM || k == 0 || k > 10000 ||
		(uint64_t) size != offset + (uint64_t) dimensions * 4)
		return ANN_INVALID;
	if ((file = AnnIndexGet(name)) == NULL)
		return ANN_STALE;
	bitmap = AnnIndexBitmapGet(file, id, ANN_BITMAP_PUSHED);
	if (bitmap == NULL)
	{
		AnnIndexPut(file);
		return ANN_MISSING;
	}
	{
		/* The job takes its own reference before this one goes. */
		AnnStatus	status = SubmitBitmapJob(service, connection, rdma, request, payload, size, file, bitmap, name,
											 dimensions, k, offset, ANN_OP_INDEX_SEARCH_CACHED, 0);

		AnnIndexPut(file);
		return status;
	}
}

/*
 * Queue a search over a cached bitmap (a pushed filter or a predicate's
 * result), whose reference the job takes over. The query starts at offset.
 */
static AnnStatus
SubmitBitmapJob(Service *service, Connection *connection, int rdma, uint64_t request,
				unsigned char *payload, uint32_t size, AnnIndexFile *file, AnnIndexBitmap *bitmap,
				const char *name, uint32_t dimensions, uint32_t k, size_t offset, uint16_t operation,
				uint32_t flags)
{
	Job		   *job;

	job = calloc(1, sizeof(*job));
	if (job == NULL || (job->index.results = malloc(sizeof(AnnIndexHit) * k)) == NULL)
	{
		free(job);
		if (bitmap)
			AnnIndexBitmapRelease(file, bitmap);
		return ANN_NOMEM;
	}
	job->kind = JOB_INDEX;
	strcpy(job->index.name, name);
	job->index.dimensions = dimensions;
	job->index.k = k;
	job->index.filtered = bitmap != NULL;	/* NULL: every row, plain ANN */
	{
		float	   *query = (float *) (void *) payload;

		for (uint32_t d = 0; d < dimensions; d++)
		{
			uint32_t	bits = (uint32_t) Read(payload + offset + (size_t) d * 4, 4);

			memcpy(&query[d], &bits, sizeof(bits));
		}
		job->index.query = query;
	}

	/*
	 * The job carries every form the cache has: rows (ordinals and keys) for
	 * the scans, bitsets for filtered CAGRA. The backend routes by the
	 * filter's size and by how many queued jobs share it.
	 */
	if (bitmap)
		job->index.nfilter = (uint32_t) (bitmap->count < UINT32_MAX ? bitmap->count : UINT32_MAX - 1);
	if (bitmap && bitmap->keys)
	{
		job->index.filter = bitmap->keys;
		job->index.ordinals = bitmap->ordinals;
	}
	if (bitmap && file->nshards && bitmap->words)
	{
		job->index.bitsets = bitmap->bitsets;
		job->index.deviceBitsets = bitmap->device;
	}
	if (bitmap && file->nshards)
		job->index.deviceRows = bitmap->deviceRows;
	if (bitmap && job->index.filter == NULL && job->index.bitsets == NULL)
	{
		/* Neither form usable: the worker's limits changed under the entry. */
		free(job->index.results);
		free(job);
		AnnIndexBitmapRelease(file, bitmap);
		return ANN_MISSING;
	}
	job->bitmapFile = file;
	job->bitmap = bitmap;
	job->connection = connection;
	job->operation = operation;
	job->flags = flags;
	job->rdma = rdma;
	job->request = request;
	job->payload = payload;
	job->charge = (uint64_t) size + sizeof(AnnIndexHit) * k;
	job->composite = submittingComposite;
	job->compositePart = submittingPart;
	job->wanted = submittingWanted;
	if (file)
		AnnIndexHold(file);		/* until Finish: unloading waits for it */
	if (Enqueue(service, connection, job))
		return ANN_PENDING;
	if (file)
		AnnIndexPut(file);
	free(job->index.results);
	if (bitmap)
		AnnIndexBitmapRelease(file, bitmap);
	free(job);
	return ANN_OVERLOADED;
}

static uint64_t
MicrosecondsSince(const struct timespec *start)
{
	struct timespec now;

	(void) clock_gettime(CLOCK_MONOTONIC, &now);
	return (uint64_t) (now.tv_sec - start->tv_sec) * 1000000 +
		(uint64_t) ((now.tv_nsec - start->tv_nsec) / 1000);
}

/*
 * A predicate's rows as a cached bitmap: found under the program's hash, or
 * evaluated over the attribute columns and cached (dropping the least recently
 * used predicate results past --predicate-cache-mb, --predicate-device-mb or
 * --predicate-cache). Returns a referenced bitmap.
 */
static AnnStatus PredicateEvaluate(Service *service, AnnIndexFile *file, const unsigned char *program,
								   uint32_t size, uint64_t id, size_t nwords, AnnIndexBitmap **result, int snapshot);

/*
 * Programs being evaluated: a search that misses the cache while another
 * thread evaluates the same program waits for its result instead of
 * evaluating it again, so a burst of new predicates costs one evaluation
 * each.
 */
#define ANN_EVALUATING_MAX 256
static pthread_mutex_t evaluatingMutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t evaluatingDone = PTHREAD_COND_INITIALIZER;
static struct
{
	const AnnIndexFile *file;
	uint64_t	id;
}			evaluating[ANN_EVALUATING_MAX];

static int
Evaluating(const AnnIndexFile *file, uint64_t id)
{
	for (int i = 0; i < ANN_EVALUATING_MAX; i++)
		if (evaluating[i].file == file && evaluating[i].id == id)
			return i;
	return -1;
}

static AnnStatus
PredicateBitmap(Service *service, AnnIndexFile *file, const unsigned char *program, uint32_t size,
				AnnIndexBitmap **result, int snapshot)
{
	uint64_t	id = AnnFilterHash(program, size);
	size_t		nwords = ((size_t) file->rows + 31) / 32;
	int			slot = -1;
	AnnStatus	status;

	for (;;)
	{
		if ((*result = AnnIndexBitmapGet(file, id, snapshot ? ANN_BITMAP_SNAPSHOT : ANN_BITMAP_PREDICATE)) != NULL)
		{
			/* Not under service->mutex: every search would contend for it. */
			(void) __atomic_fetch_add(&service->predicate_hits, 1, __ATOMIC_RELAXED);
			return ANN_OK;
		}
		(void) pthread_mutex_lock(&evaluatingMutex);
		if (Evaluating(file, id) >= 0)
		{
			/* Someone else is on it: wait, then look in the cache again. */
			while (Evaluating(file, id) >= 0)
				(void) pthread_cond_wait(&evaluatingDone, &evaluatingMutex);
			(void) pthread_mutex_unlock(&evaluatingMutex);
			continue;
		}
		for (int i = 0; i < ANN_EVALUATING_MAX && slot < 0; i++)
			if (evaluating[i].file == NULL)
			{
				evaluating[i].file = file;
				evaluating[i].id = id;
				slot = i;
			}
		(void) pthread_mutex_unlock(&evaluatingMutex);
		break;					/* evaluate (unclaimed if every slot is busy) */
	}
	status = PredicateEvaluate(service, file, program, size, id, nwords, result, snapshot);
	if (slot >= 0)
	{
		(void) pthread_mutex_lock(&evaluatingMutex);
		evaluating[slot].file = NULL;
		(void) pthread_cond_broadcast(&evaluatingDone);
		(void) pthread_mutex_unlock(&evaluatingMutex);
	}
	return status;
}

static AnnStatus
PredicateEvaluate(Service *service, AnnIndexFile *file, const unsigned char *program, uint32_t size,
				  uint64_t id, size_t nwords, AnnIndexBitmap **result, int snapshot)
{
	uint32_t   *words;
	uint64_t	count;
	int			code;
	AnnStatus	status;
	struct timespec start;

	if ((words = malloc(nwords * 4 + 4)) == NULL)
		return ANN_NOMEM;
	(void) clock_gettime(CLOCK_MONOTONIC, &start);
	code = AnnIndexEvaluate(file, program, size, words, &count, snapshot);
	if (code != ANN_FILTER_OK)
	{
		free(words);
		return code == ANN_FILTER_NO_MEMORY ? ANN_NOMEM : code == ANN_FILTER_UNKNOWN_BITMAP ? ANN_MISSING :
			ANN_INVALID;
	}

	/* Keep rows for every filter a scan may take; see SubmitCached. */
	/* The entry comes back referenced: an eviction meanwhile can't lose it. */
	status = AnnIndexPredicatePut(file, id, snapshot ? ANN_BITMAP_SNAPSHOT : ANN_BITMAP_PREDICATE, words, nwords, file->device == NULL ? UINT64_MAX :
								  AnnIndexConfig.group_gather_limit > AnnIndexConfig.bitset_gather_limit ?
								  AnnIndexConfig.group_gather_limit : AnnIndexConfig.bitset_gather_limit, result);
	free(words);
	(void) __atomic_fetch_add(&service->predicate_evaluations, 1, __ATOMIC_RELAXED);
	(void) __atomic_fetch_add(&service->predicate_eval_us, MicrosecondsSince(&start), __ATOMIC_RELAXED);
	if (status != ANN_OK)
		return status;
	return ANN_OK;
}

/*
 * Predicate search payload: u32 name length, name, u32 dimensions, k,
 * flags, program length, the program (ann/filter.h); FP32 query. The rows
 * are cached under the program's hash, so repeated predicates share one
 * evaluation and are grouped like cached bitmaps.
 */
static AnnStatus
SubmitPredicate(Service *service, Connection *connection, int rdma, uint64_t request,
				unsigned char *payload, uint32_t size)
{
	AnnIndexFile *file;
	AnnIndexBitmap *bitmap;
	char		name[ANN_INDEX_NAME_MAX + 1];
	uint32_t	nameLength,
				dimensions,
				k,
				flags,
				programLength;
	size_t		offset;
	AnnStatus	status;

	if (service->backend->search_index == NULL)
		return ANN_UNSUPPORTED;
	if (size < 4 || (nameLength = (uint32_t) Read(payload, 4)) == 0 ||
		nameLength > ANN_INDEX_NAME_MAX || (uint64_t) size < 20 + (uint64_t) nameLength ||
		memchr(payload + 4, '\0', nameLength))
		return ANN_INVALID;
	memcpy(name, payload + 4, nameLength);
	name[nameLength] = '\0';
	offset = 4 + nameLength;
	dimensions = (uint32_t) Read(payload + offset, 4);
	k = (uint32_t) Read(payload + offset + 4, 4);
	flags = (uint32_t) Read(payload + offset + 8, 4);
	programLength = (uint32_t) Read(payload + offset + 12, 4);
	offset += 16;
	if (dimensions == 0 || dimensions > ANN_MAX_DIM || k == 0 || k > 10000 ||
		(flags & ~(ANN_PREDICATE_ORDINALS | ANN_PREDICATE_ATTRIBUTES | ANN_PREDICATE_COLLECTION_CODES)) ||
		(uint64_t) size != offset + (uint64_t) programLength + (uint64_t) dimensions * 4 ||
		!AnnFilterCheck(payload + offset, programLength))
		return ANN_INVALID;
	if ((file = AnnIndexGet(name)) == NULL)
		return ANN_STALE;
	if ((flags & ANN_PREDICATE_COLLECTION_CODES) && !AnnIndexRemapped(file))
	{
		/* Loaded but not mapped to the collection's dictionary (yet). */
		AnnIndexPut(file);
		return ANN_STALE;
	}

	/*
	 * A program matching every row is a plain ANN search. With tombstones, it
	 * asks for that many more hits and drops them (Finish), or, past the
	 * backend's k, runs as a filtered search whose rows leave them out.
	 */
	if (programLength == 1 && payload[offset] == ANN_FILTER_ALL &&
		(uint64_t) k + __atomic_load_n(&file->ndeleted, __ATOMIC_ACQUIRE) <=
		(service->backend->max_k < 10000 ? service->backend->max_k : 10000))
	{
		uint64_t	deleted = __atomic_load_n(&file->ndeleted, __ATOMIC_ACQUIRE);

		submittingWanted = deleted ? k : 0;
		status = SubmitBitmapJob(service, connection, rdma, request, payload, size, file, NULL, name,
								 dimensions, (uint32_t) (k + deleted), offset + programLength,
								 ANN_OP_INDEX_SEARCH_PREDICATE, flags);
		submittingWanted = 0;
	}
	else if ((status = PredicateBitmap(service, file, payload + offset, programLength, &bitmap, 0)) == ANN_OK)
		status = SubmitBitmapJob(service, connection, rdma, request, payload, size, file, bitmap, name,
								 dimensions, k, offset + programLength, ANN_OP_INDEX_SEARCH_PREDICATE, flags);
	/* The job took its own reference before this one goes. */
	AnnIndexPut(file);
	return status;
}

/*
 * Operation 25 payload: u32 parts, per part (u32 name length, name, u64
 * generation, 0 for any), then as operation 22 after its name: u32
 * dimensions, k, flags (no ANN_PREDICATE_ORDINALS: ordinals of different
 * indexes don't mix), program length, program, FP32 query. Each part is
 * submitted as an operation 22 job answering into one composite reply.
 * Always ANN_PENDING: the composite replies, errors included.
 */
static AnnStatus
SubmitCollection(Service *service, Connection *connection, int rdma, uint64_t request,
				 unsigned char *payload, uint32_t size)
{
	uint32_t	parts;
	size_t		offset = 4;
	size_t		common;
	uint32_t	k;
	Composite  *composite;

	if (size < 4 || (parts = (uint32_t) Read(payload, 4)) == 0 || parts > ANN_INDEX_MAX_SHARDS)
		return ANN_INVALID;
	for (uint32_t p = 0; p < parts; p++)
	{
		uint32_t	nameLength;

		if ((uint64_t) offset + 4 > size || (nameLength = (uint32_t) Read(payload + offset, 4)) == 0 ||
			nameLength > ANN_INDEX_NAME_MAX || (uint64_t) offset + 12 + nameLength > size)
			return ANN_INVALID;
		offset += 12 + nameLength;
	}
	common = offset;
	if ((uint64_t) common + 16 > size)
		return ANN_INVALID;
	k = (uint32_t) Read(payload + common + 4, 4);
	if (k == 0 || k > 10000 || (Read(payload + common + 8, 4) & ANN_PREDICATE_ORDINALS))
		return ANN_INVALID;
	composite = calloc(1, sizeof(*composite));
	if (composite == NULL || (composite->hits = malloc(sizeof(AnnIndexHit) * (size_t) k * parts)) == NULL)
	{
		free(composite);
		return ANN_NOMEM;
	}
	(void) pthread_mutex_init(&composite->mutex, NULL);
	composite->remaining = parts;
	composite->parts = parts;
	composite->k = k;
	composite->attributes = (Read(payload + common + 8, 4) & ANN_PREDICATE_ATTRIBUTES) != 0;
	composite->nattributes = -1;
	composite->status = ANN_OK;
	composite->connection = connection;
	composite->rdma = rdma;
	composite->request = request;
	(void) pthread_mutex_lock(&connection->mutex);
	connection->references++;	/* held until the composite replies */
	(void) pthread_mutex_unlock(&connection->mutex);

	offset = 4;
	for (uint32_t p = 0; p < parts; p++)
	{
		uint32_t	nameLength = (uint32_t) Read(payload + offset, 4);
		uint64_t	generation = Read(payload + offset + 4 + nameLength, 8);
		char		name[ANN_INDEX_NAME_MAX + 1];
		AnnIndexFile *file;
		AnnStatus	status = ANN_OK;
		unsigned char *part;
		size_t		partSize = 4 + nameLength + (size - common);

		memcpy(name, payload + offset + 4, nameLength);
		name[nameLength] = '\0';
		offset += 12 + nameLength;
		if (memchr(name, '\0', nameLength))
			status = ANN_INVALID;
		else if ((file = AnnIndexGet(name)) == NULL)
			status = ANN_STALE;
		else
		{
			if (generation != 0 && file->generationValue != generation)
				status = ANN_GENERATION;
			AnnIndexPut(file);
		}
		if (status == ANN_OK && (part = malloc(partSize)) == NULL)
			status = ANN_NOMEM;
		if (status == ANN_OK)
		{
			/* An operation 22 payload: the part's name, then the shared rest. */
			Write(part, nameLength, 4);
			memcpy(part + 4, name, nameLength);
			memcpy(part + 4 + nameLength, payload + common, size - common);
			submittingComposite = composite;
			status = SubmitPredicate(service, connection, rdma, request, part, (uint32_t) partSize);
			submittingComposite = NULL;
			if (status != ANN_PENDING)
				free(part);
		}
		if (status != ANN_PENDING)
			CompositeAnswer(composite, p, 0, status, NULL, 0, 0, NULL);
	}
	free(payload);				/* the parts own their copies */
	return ANN_PENDING;
}

/*
 * Operation 30: u32 name length, name, u32 dimensions, k, flags, program
 * length, query count, program, then a row-major FP32 query matrix.
 * A single predicate cache entry and index generation serve all queries.
 * Reply: u32 query count, then an operation 22 result per query (no
 * attribute trailer). Queries are scheduled together with other requests.
 * Up to 64 queries bound both admission overhead and reply allocation.
 */
static AnnStatus
SubmitBatch(Service *service, Connection *connection, int rdma, uint64_t request,
			unsigned char *payload, uint32_t size, uint64_t generation)
{
	uint32_t	nameLength,
				dimensions,
				k,
				flags,
				programLength,
				parts;
	size_t		common,
				queries;
	Composite  *composite;
	AnnIndexFile *file;
	AnnIndexBitmap *bitmap = NULL;
	AnnStatus	status;
	char		name[ANN_INDEX_NAME_MAX + 1];
	uint32_t	searchK;
	uint64_t	deleted;

	if (size < 4 || (nameLength = (uint32_t) Read(payload, 4)) == 0 ||
		nameLength > ANN_INDEX_NAME_MAX || (uint64_t) size < 24 + (uint64_t) nameLength ||
		memchr(payload + 4, '\0', nameLength))
		return ANN_INVALID;
	common = 4 + nameLength;
	dimensions = (uint32_t) Read(payload + common, 4);
	k = (uint32_t) Read(payload + common + 4, 4);
	flags = (uint32_t) Read(payload + common + 8, 4);
	programLength = (uint32_t) Read(payload + common + 12, 4);
	parts = (uint32_t) Read(payload + common + 16, 4);
	queries = common + 20 + (size_t) programLength;
	if (dimensions == 0 || dimensions > ANN_MAX_DIM || k == 0 || k > 10000 ||
		parts == 0 || parts > 64 || (flags & ~(ANN_PREDICATE_ORDINALS | ANN_PREDICATE_SNAPSHOT)) ||
		(uint64_t) size != (uint64_t) queries + (uint64_t) parts * dimensions * 4 ||
		!AnnFilterCheck(payload + common + 20, programLength))
		return ANN_INVALID;
	if (service->backend->search_index == NULL)
		return ANN_UNSUPPORTED;
	memcpy(name, payload + 4, nameLength);
	name[nameLength] = '\0';
	if ((file = AnnIndexGet(name)) == NULL)
		return ANN_STALE;

	/*
	 * Check the leased file too: another connection can reload after
	 * dispatch.
	 */
	if (generation && file->generationValue != generation)
	{
		AnnIndexPut(file);
		return ANN_GENERATION;
	}
	searchK = k;
	deleted = (flags & ANN_PREDICATE_SNAPSHOT) ? 0 :
		__atomic_load_n(&file->ndeleted, __ATOMIC_ACQUIRE);
	if (programLength == 1 && payload[common + 20] == ANN_FILTER_ALL &&
		(uint64_t) k + deleted <=
		(service->backend->max_k < 10000 ? service->backend->max_k : 10000))
		searchK += (uint32_t) deleted;
	else if ((status = PredicateBitmap(service, file, payload + common + 20,
									   programLength, &bitmap, (flags & ANN_PREDICATE_SNAPSHOT) != 0)) != ANN_OK)
	{
		AnnIndexPut(file);
		return status;
	}
	composite = calloc(1, sizeof(*composite));
	if (composite == NULL)
	{
		if (bitmap)
			AnnIndexBitmapRelease(file, bitmap);
		AnnIndexPut(file);
		return ANN_NOMEM;
	}
	composite->hits = malloc(sizeof(AnnIndexHit) * (size_t) k * parts);
	composite->batch = calloc(parts, sizeof(BatchPart));
	if (composite->hits == NULL || composite->batch == NULL)
	{
		free(composite->hits);
		free(composite->batch);
		free(composite);
		if (bitmap)
			AnnIndexBitmapRelease(file, bitmap);
		AnnIndexPut(file);
		return ANN_NOMEM;
	}
	(void) pthread_mutex_init(&composite->mutex, NULL);
	composite->remaining = composite->parts = parts;
	composite->k = k;
	composite->flags = flags;
	composite->status = ANN_OK;
	composite->connection = connection;
	composite->rdma = rdma;
	composite->request = request;
	(void) pthread_mutex_lock(&connection->mutex);
	connection->references++;
	(void) pthread_mutex_unlock(&connection->mutex);
	for (uint32_t p = 0; p < parts; p++)
	{
		size_t		partSize = (size_t) dimensions * 4;
		unsigned char *part = malloc(partSize);
		AnnStatus	partStatus = ANN_NOMEM;

		if (part)
		{
			memcpy(part, payload + queries + (size_t) p * dimensions * 4, partSize);
			if (bitmap)
				AnnIndexBitmapHold(file, bitmap);
			submittingWanted = searchK != k ? k : 0;
			submittingComposite = composite;
			submittingPart = p;
			partStatus = SubmitBitmapJob(service, connection, rdma, request, part,
										 (uint32_t) partSize, file, bitmap, name, dimensions, searchK,
										 0, ANN_OP_INDEX_SEARCH_BATCH, flags);
			submittingComposite = NULL;
			submittingPart = 0;
			submittingWanted = 0;
			if (partStatus != ANN_PENDING)
				free(part);
		}
		if (partStatus != ANN_PENDING)
			CompositeAnswer(composite, p, 0, partStatus, NULL, 0, 0, NULL);
	}
	if (bitmap)
		AnnIndexBitmapRelease(file, bitmap);
	AnnIndexPut(file);
	free(payload);
	return ANN_PENDING;
}

/*
 * Predicate evaluation payload: u32 name length, name, then the program.
 * Reply: u64 eligible rows, then the bitmap as u32 words. Comparing it with
 * PostgreSQL's evaluation from its frozen pages proves both agree.
 */
static AnnStatus
EvaluatePredicate(Service *service, const unsigned char *payload, uint32_t size,
				  unsigned char **reply, size_t *replyLength)
{
	char		name[ANN_INDEX_NAME_MAX + 1];
	uint32_t	nameLength;
	AnnIndexFile *file;
	AnnIndexBitmap *bitmap;
	AnnStatus	status;
	size_t		nwords;

	*reply = NULL;
	*replyLength = 0;
	if (size < 4 || (nameLength = (uint32_t) Read(payload, 4)) == 0 ||
		nameLength > ANN_INDEX_NAME_MAX || (uint64_t) size <= 4 + (uint64_t) nameLength ||
		memchr(payload + 4, '\0', nameLength) ||
		!AnnFilterCheck(payload + 4 + nameLength, size - 4 - nameLength))
		return ANN_INVALID;
	memcpy(name, payload + 4, nameLength);
	name[nameLength] = '\0';
	if ((file = AnnIndexGet(name)) == NULL)
		return ANN_STALE;
	status = PredicateBitmap(service, file, payload + 4 + nameLength, size - 4 - nameLength, &bitmap, 0);
	if (status == ANN_OK)
	{
		nwords = ((size_t) file->rows + 31) / 32;
		if ((*reply = malloc(8 + nwords * 4)) == NULL)
			status = ANN_NOMEM;
		else
		{
			uint32_t   *words = malloc(nwords * 4 + 4);

			if (words == NULL)
			{
				free(*reply);
				*reply = NULL;
				status = ANN_NOMEM;
			}
			else
			{
				AnnIndexBitmapWords(file, bitmap, words);
				Write(*reply, bitmap->count, 8);
				for (size_t w = 0; w < nwords; w++)
					Write(*reply + 8 + w * 4, words[w], 4);
				*replyLength = 8 + nwords * 4;
				free(words);
			}
		}
		AnnIndexBitmapRelease(file, bitmap);
	}
	AnnIndexPut(file);
	return status;
}

/*
 * Requests that carry a generation in their header (PostgreSQL sets it for
 * indexes exported from a frozen index) must name the loaded one.
 */
static AnnStatus
CheckGeneration(uint16_t operation, const unsigned char *payload, uint32_t size, uint64_t generation)
{
	char		name[ANN_INDEX_NAME_MAX + 1];
	uint32_t	nameLength;
	AnnIndexFile *file;
	AnnStatus	status = ANN_OK;

	if (generation == 0 ||
		(operation != ANN_OP_INDEX_SEARCH && operation != ANN_OP_INDEX_SEARCH_PACKED &&
		 operation != ANN_OP_INDEX_SEARCH_MATCH && operation != ANN_OP_INDEX_SEARCH_BITMAP &&
		 operation != ANN_OP_INDEX_SEARCH_CACHED && operation != ANN_OP_INDEX_SEARCH_PREDICATE &&
		 operation != ANN_OP_INDEX_PREDICATE && operation != ANN_OP_INDEX_SEARCH_BATCH &&
		 operation != ANN_OP_BITMAP_PUT) ||
		size < 4 || (nameLength = (uint32_t) Read(payload, 4)) == 0 ||
		nameLength > ANN_INDEX_NAME_MAX || (uint64_t) size < 4 + (uint64_t) nameLength)
		return ANN_OK;
	memcpy(name, payload + 4, nameLength);
	name[nameLength] = '\0';
	if ((file = AnnIndexGet(name)) == NULL)
		return ANN_OK;			/* the operation reports it missing */
	if (file->generationValue != generation)
		status = ANN_GENERATION;
	AnnIndexPut(file);
	return status;
}

/*
 * Load payload: u32 name length, name, then the directory path. The reply
 * is text: "rows N dimensions D shards S dtype T source NAME".
 */
static AnnStatus
LoadIndex(Service *service, const unsigned char *payload, uint32_t size,
		  char *reply, size_t replySize, size_t *replyLength)
{
	char		name[ANN_INDEX_NAME_MAX + 1];
	char		directory[1024];
	uint32_t	nameLength;
	AnnIndexFile *file;
	AnnStatus	status;
	int			written;

	if (service->backend->load_index == NULL)
		return ANN_UNSUPPORTED;
	if (size < 4 || (nameLength = (uint32_t) Read(payload, 4)) == 0 ||
		nameLength > ANN_INDEX_NAME_MAX || size - 4 <= nameLength ||
		size - 4 - nameLength >= sizeof(directory))
		return ANN_INVALID;
	memcpy(name, payload + 4, nameLength);
	name[nameLength] = '\0';
	memcpy(directory, payload + 4 + nameLength, size - 4 - nameLength);
	directory[size - 4 - nameLength] = '\0';
	if (strlen(name) != nameLength || strlen(directory) != size - 4 - nameLength)
		return ANN_INVALID;
	status = service->backend->load_index(name, directory);
	if (status != ANN_OK)
		return status;
	file = AnnIndexGet(name);
	if (file == NULL)
		return ANN_BACKEND_ERROR;
	written = snprintf(reply, replySize, "rows %llu dimensions %u shards %u dtype %s source %s key %s vector %s "
					   "quant %s quant_scan %s generation %s predicates on",
					   (unsigned long long) file->rows, file->dimensions, file->nshards,
					   file->dtype, file->source[0] ? file->source : "-",
					   file->key[0] ? file->key : "-", file->vector[0] ? file->vector : "-",
					   AnnIndexQuantName(file->quant), file->quantActive ? "on" : "off",
					   file->generation[0] ? file->generation : "-");
	AnnIndexPut(file);
	*replyLength = written > 0 && (size_t) written < replySize ? (size_t) written : 0;
	return ANN_OK;
}

/* Index info payload: the index name. Replies like LoadIndex. */
static AnnStatus
DescribeIndex(const unsigned char *payload, uint32_t size, char *reply, size_t replySize, size_t *replyLength)
{
	char		name[ANN_INDEX_NAME_MAX + 1];
	AnnIndexFile *file;
	int			written;

	if (size == 0 || size > ANN_INDEX_NAME_MAX || memchr(payload, '\0', size))
		return ANN_INVALID;
	memcpy(name, payload, size);
	name[size] = '\0';
	if ((file = AnnIndexGet(name)) == NULL)
		return ANN_STALE;
	(void) pthread_mutex_lock(&file->bitmapMutex);
	written = snprintf(reply, replySize, "rows %llu dimensions %u shards %u dtype %s source %s key %s vector %s "
					   "quant %s quant_scan %s bitmaps %llu bitmap_bytes %llu bitmap_device_bytes %llu generation %s "
					   "predicates on",
					   (unsigned long long) file->rows, file->dimensions, file->nshards,
					   file->dtype, file->source[0] ? file->source : "-",
					   file->key[0] ? file->key : "-", file->vector[0] ? file->vector : "-",
					   AnnIndexQuantName(file->quant), file->quantActive ? "on" : "off",
					   (unsigned long long) file->nbitmaps, (unsigned long long) file->bitmapBytes,
					   (unsigned long long) file->bitmapDeviceBytes,
					   file->generation[0] ? file->generation : "-");
	(void) pthread_mutex_unlock(&file->bitmapMutex);
	AnnIndexPut(file);
	*replyLength = written > 0 && (size_t) written < replySize ? (size_t) written : 0;
	return ANN_OK;
}

/* Constant-time token comparison; lengths are not secret. */
static int
TokenMatches(const char *token, const unsigned char *given, uint32_t size)
{
	size_t		length = strlen(token);
	unsigned char difference = 0;

	if (size != length)
		return 0;
	for (size_t i = 0; i < length; i++)
		difference |= (unsigned char) token[i] ^ given[i];
	return difference == 0;
}

/*
 * RDMA hello over an authenticated connection: the payload is the client's
 * queue pair address. The worker connects a queue pair to it, and frames on
 * that queue pair are then answered as this connection's. Reply: u32
 * largest frame, the worker's queue pair address, device name.
 */
static AnnStatus
Hello(Service *service, Connection *connection, const unsigned char *payload, uint32_t size,
	  unsigned char *reply, size_t replySize, size_t *replyLength)
{
	const char *device;
	uint64_t	peer;

	if (service->rdma == NULL)
		return ANN_UNSUPPORTED;
	if (connection->rdma || size != ANN_RDMA_ADDRESS || (connection->remote && !connection->authenticated))
		return ANN_INVALID;
	device = AnnRdmaDevice(service->rdma);
	if (4 + ANN_RDMA_ADDRESS + strlen(device) > replySize ||
		!AnnRdmaAddPeer(service->rdma, payload, reply + 4, &peer))
		return ANN_BACKEND_ERROR;
	(void) pthread_mutex_lock(&service->peersMutex);
	if (peer >= service->npeers)
	{
		uint64_t	count = peer + 64;
		Connection **peers = realloc(service->peers, sizeof(*peers) * count);

		if (peers == NULL)
		{
			(void) pthread_mutex_unlock(&service->peersMutex);
			AnnRdmaRemovePeer(service->rdma, peer);
			return ANN_NOMEM;
		}
		memset(peers + service->npeers, 0, sizeof(*peers) * (count - service->npeers));
		service->peers = peers;
		service->npeers = count;
	}
	service->peers[peer] = connection;
	(void) pthread_mutex_unlock(&service->peersMutex);
	(void) pthread_mutex_lock(&connection->mutex);
	connection->references++;	/* held by the peer table */
	connection->rdma = service->rdma;
	connection->peer = peer;
	(void) pthread_mutex_unlock(&connection->mutex);
	Write(reply, AnnRdmaMaxFrame(service->rdma), 4);
	memcpy(reply + 4 + ANN_RDMA_ADDRESS, device, strlen(device));
	*replyLength = 4 + ANN_RDMA_ADDRESS + strlen(device);
	return ANN_OK;
}

/*
 * A closing connection stops accepting RDMA frames. Its queue pair lives
 * until the connection's last job has replied; Release destroys it.
 */
static void
Goodbye(Service *service, Connection *connection)
{
	if (connection->rdma == NULL)
		return;
	(void) pthread_mutex_lock(&service->peersMutex);
	if (connection->peer < service->npeers && service->peers[connection->peer] == connection)
		service->peers[connection->peer] = NULL;
	(void) pthread_mutex_unlock(&service->peersMutex);
	Release(connection);
}

/*
 * A frame from the RDMA progress thread. Only searches and probes are
 * served here; loads, hellos, statistics and shared memory use TCP.
 */
static void
RdmaFrame(void *context, uint64_t peer, unsigned char *frame, size_t size)
{
	Service    *service = context;
	Connection *connection = NULL;
	uint16_t	operation;
	uint64_t	request;
	uint64_t	generation;
	uint32_t	length;
	unsigned char *payload;
	AnnStatus	status;

	if (traceFile)
		frameArrived = Now();
	if (size < 32 || memcmp(frame, "PVAN", 4) != 0 || Read(frame + 4, 2) != 1 ||
		(length = (uint32_t) Read(frame + 24, 4)) != size - 32)
	{
		free(frame);
		return;
	}
	(void) pthread_mutex_lock(&service->peersMutex);
	if (peer < service->npeers)
		connection = service->peers[peer];
	if (connection)
	{
		(void) pthread_mutex_lock(&connection->mutex);
		connection->references++;
		(void) pthread_mutex_unlock(&connection->mutex);
	}
	(void) pthread_mutex_unlock(&service->peersMutex);
	__atomic_fetch_add(&service->rdma_frames, 1, __ATOMIC_RELAXED);
	if (connection == NULL)
	{
		free(frame);
		return;
	}
	operation = (uint16_t) Read(frame + 6, 2);
	request = Read(frame + 8, 8);
	generation = Read(frame + 16, 8);
	payload = malloc(length ? length : 1);
	if (payload)
		memcpy(payload, frame + 32, length);
	free(frame);
	if (payload == NULL)
		status = ANN_NOMEM;
	else if ((status = CheckGeneration(operation, payload, length, generation)) != ANN_OK)
		;
	else if (operation == ANN_OP_INDEX_SEARCH_PREDICATE)
		status = SubmitPredicate(service, connection, 1, request, payload, length);
	else if (operation == ANN_OP_INDEX_SEARCH_BATCH)
		status = SubmitBatch(service, connection, 1, request, payload, length, generation);
	else if (operation == ANN_OP_COLLECTION_SEARCH)
		status = SubmitCollection(service, connection, 1, request, payload, length);
	else if (operation == ANN_OP_INDEX_SEARCH || operation == ANN_OP_INDEX_SEARCH_PACKED)
		status = SubmitIndex(service, connection, 1, operation, request, payload, length);
	else if (operation == ANN_OP_INDEX_SEARCH_MATCH)
		status = SubmitMatch(service, connection, 1, request, payload, length);
	else if (operation == ANN_OP_INDEX_SEARCH_BITMAP)
		status = SubmitBitmap(service, connection, 1, request, payload, length);
	else if (operation == ANN_OP_INDEX_SEARCH_CACHED)
		status = SubmitCached(service, connection, 1, request, payload, length);
	else if (operation == ANN_OP_GATHERED_SEARCH)
		status = Submit(service, connection, 1, operation, request, payload, length);
	else if (operation == 5)
		status = ANN_OK;
	else
		status = ANN_UNSUPPORTED;
	if (status != ANN_PENDING)
	{
		SendReply(connection, 1, operation, request, status, NULL, 0);
		free(payload);
	}
	Release(connection);
}

typedef struct ReaderArgs
{
	Service    *service;
	Connection *connection;
} ReaderArgs;

/*
 * Operation 27: u32 name length, name, u32 attribute length, attribute;
 * reply: the text attribute's dictionary (AnnIndexDictionary). Operation
 * 28: the same, then u32 count and count i32 collection codes, one per
 * dictionary string (AnnIndexRemap).
 */
static AnnStatus
DictionaryOperation(uint16_t operation, const unsigned char *payload, uint32_t length,
					unsigned char **reply, size_t *replyLength)
{
	char		name[ANN_INDEX_NAME_MAX + 1];
	char		attribute[64];
	uint32_t	nameLength,
				attributeLength,
				count = 0;
	size_t		offset;
	AnnIndexFile *file;
	AnnStatus	status;
	int32_t    *map = NULL;

	if (length < 4 || (nameLength = (uint32_t) Read(payload, 4)) == 0 || nameLength > ANN_INDEX_NAME_MAX ||
		(uint64_t) length < 8 + (uint64_t) nameLength ||
		(attributeLength = (uint32_t) Read(payload + 4 + nameLength, 4)) == 0 || attributeLength > 63 ||
		(uint64_t) length < 8 + (uint64_t) nameLength + attributeLength)
		return ANN_INVALID;
	memcpy(name, payload + 4, nameLength);
	name[nameLength] = '\0';
	memcpy(attribute, payload + 8 + nameLength, attributeLength);
	attribute[attributeLength] = '\0';
	if (strlen(name) != nameLength || strlen(attribute) != attributeLength)
		return ANN_INVALID;
	offset = 8 + (size_t) nameLength + attributeLength;
	if (operation == ANN_OP_REMAP)
	{
		if ((uint64_t) length < offset + 4 ||
			(uint64_t) length != offset + 4 + (uint64_t) (count = (uint32_t) Read(payload + offset, 4)) * 4)
			return ANN_INVALID;
		if ((map = malloc(sizeof(int32_t) * (count ? count : 1))) == NULL)
			return ANN_NOMEM;
		for (uint32_t i = 0; i < count; i++)
			map[i] = (int32_t) (uint32_t) Read(payload + offset + 4 + (size_t) i * 4, 4);
	}
	else if (length != offset)
		return ANN_INVALID;
	if ((file = AnnIndexGet(name)) == NULL)
	{
		free(map);
		return ANN_STALE;
	}
	status = operation == ANN_OP_DICTIONARY ? AnnIndexDictionary(file, attribute, reply, replyLength) :
		AnnIndexRemap(file, attribute, map, count);
	AnnIndexPut(file);
	free(map);
	return status;
}

static void *
Reader(void *argument)
{
	ReaderArgs *args = argument;
	Service    *service = args->service;
	Connection *connection = args->connection;

	free(args);
	for (;;)
	{
		unsigned char header[32];
		unsigned char *payload;
		uint16_t	operation;
		uint64_t	request;
		uint64_t	generation;
		uint32_t	length;
		AnnStatus	status;

		if (!ReadFull(connection->fd, header, sizeof(header)) ||
			memcmp(header, "PVAN", 4) != 0 || Read(header + 4, 2) != 1 ||
			Read(header + 28, 4) != 0 || (length = (uint32_t) Read(header + 24, 4)) > ANN_MAX_BYTES)
			break;
		if (traceFile)
			frameArrived = Now();
		operation = (uint16_t) Read(header + 6, 2);
		request = Read(header + 8, 8);
		generation = Read(header + 16, 8);
		payload = malloc(length ? length : 1);
		if (payload == NULL || !ReadFull(connection->fd, payload, length))
		{
			free(payload);
			break;
		}
		if (connection->remote && !connection->authenticated)
		{
			/*
			 * The first network frame must carry the token; anything else
			 * closes.
			 */
			int			ok = operation == ANN_OP_AUTH && service->options.token &&
				TokenMatches(service->options.token, payload, length);

			free(payload);
			if (!ok)
				break;
			connection->authenticated = 1;
			{
				struct timeval none = {0, 0};

				/* Authenticated clients may idle between queries. */
				(void) setsockopt(connection->fd, SOL_SOCKET, SO_RCVTIMEO, &none, sizeof(none));
			}
			Reply(connection, operation, request, ANN_OK, NULL, 0);
			continue;
		}
		if ((status = CheckGeneration(operation, payload, length, generation)) != ANN_OK)
		{
			free(payload);
			Reply(connection, operation, request, status, NULL, 0);
			continue;
		}
		if (connection->remote &&
			(operation == ANN_OP_ATTACH_SHARED || operation == ANN_OP_SHARED_SEARCH))
		{
			/* Shared memory only works on the same host. */
			free(payload);
			Reply(connection, operation, request, ANN_UNSUPPORTED, NULL, 0);
			continue;
		}
		if (operation == ANN_OP_GATHERED_SEARCH || operation == ANN_OP_SHARED_SEARCH)
		{
			status = Submit(service, connection, 0, operation, request, payload, length);
			if (status == ANN_PENDING)
				continue;		/* the lane owns the payload */
			Reply(connection, operation, request, status, NULL, 0);
		}
		else if (operation == 0 || operation == ANN_OP_STATS)
		{
			char		message[1024];
			int			size;
			uint64_t	predicateEntries = 0,
						predicateBytes = 0,
						predicateDeviceBytes = 0,
						predicateEvictions = 0;

			uint64_t	rdmaFrames = 0,
						rdmaFallbacks = 0;
			int			rdmaLanes = 0;

			if (service->rdma)
				AnnRdmaStats(service->rdma, &rdmaFrames, &rdmaFallbacks, &rdmaLanes);
			(void) rdmaFrames;
			if (operation == ANN_OP_STATS)
				AnnIndexPredicateStats(&predicateEntries, &predicateBytes, &predicateDeviceBytes,
									   &predicateEvictions);
			(void) pthread_mutex_lock(&service->mutex);
			if (operation == 0)
				size = snprintf(message, sizeof(message),
								"{\"backend\":\"%s\",\"mode\":\"server\",\"operations\":[0,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31],"
								"\"metrics_mask\":%u,\"exact\":%s,\"lanes\":%u,\"max_batch_jobs\":%u,"
								"\"max_frame_bytes\":%u}",
								service->backend->name, service->backend->metrics,
								service->backend->exact ? "true" : "false",
								service->options.lanes, service->options.max_batch_jobs, ANN_MAX_BYTES);
			else
				size = snprintf(message, sizeof(message),
								"{\"jobs\":%llu,\"batches\":%llu,\"max_batch\":%llu,\"rows\":%llu,"
								"\"rejected\":%llu,\"rejected_index\":%llu,\"failed\":%llu,\"pending_bytes\":%llu,\"device_jobs\":%llu,"
								"\"rdma_frames\":%llu,\"rdma_fallbacks\":%llu,\"rdma_lanes\":%d,"
								"\"groups\":%llu,\"grouped_jobs\":%llu,\"held\":%llu,\"predicate_hits\":%llu,"
								"\"predicate_evaluations\":%llu,\"predicate_eval_us\":%llu,"
								"\"predicate_entries\":%llu,\"predicate_bytes\":%llu,"
								"\"predicate_device_bytes\":%llu,\"predicate_evictions\":%llu}",
								(unsigned long long) service->jobs, (unsigned long long) service->batches,
								(unsigned long long) service->max_batch, (unsigned long long) service->rows,
								(unsigned long long) service->rejected, (unsigned long long) service->rejected_index,
								(unsigned long long) service->failed, (unsigned long long) service->pending_bytes,
								(unsigned long long) (service->backend->device_jobs ?
													  service->backend->device_jobs() : 0),
								(unsigned long long) __atomic_load_n(&service->rdma_frames, __ATOMIC_RELAXED),
								(unsigned long long) rdmaFallbacks,
								rdmaLanes, (unsigned long long) service->groups,
								(unsigned long long) service->grouped_jobs, (unsigned long long) service->held,
								(unsigned long long) service->predicate_hits,
								(unsigned long long) service->predicate_evaluations,
								(unsigned long long) service->predicate_eval_us,
								(unsigned long long) predicateEntries, (unsigned long long) predicateBytes,
								(unsigned long long) predicateDeviceBytes,
								(unsigned long long) predicateEvictions);
			(void) pthread_mutex_unlock(&service->mutex);
			Reply(connection, operation, request, ANN_OK, (unsigned char *) message,
				  size > 0 && (size_t) size < sizeof(message) ? (size_t) size : 0);
		}
		else if (operation == ANN_OP_INDEX_SEARCH || operation == ANN_OP_INDEX_SEARCH_PACKED)
		{
			status = SubmitIndex(service, connection, 0, operation, request, payload, length);
			if (status == ANN_PENDING)
				continue;
			Reply(connection, operation, request, status, NULL, 0);
		}
		else if (operation == ANN_OP_INDEX_SEARCH_MATCH)
		{
			status = SubmitMatch(service, connection, 0, request, payload, length);
			if (status == ANN_PENDING)
				continue;
			Reply(connection, operation, request, status, NULL, 0);
		}
		else if (operation == ANN_OP_INDEX_SEARCH_BITMAP)
		{
			status = SubmitBitmap(service, connection, 0, request, payload, length);
			if (status == ANN_PENDING)
				continue;
			Reply(connection, operation, request, status, NULL, 0);
		}
		else if (operation == ANN_OP_INDEX_SEARCH_CACHED)
		{
			status = SubmitCached(service, connection, 0, request, payload, length);
			if (status == ANN_PENDING)
				continue;
			Reply(connection, operation, request, status, NULL, 0);
		}
		else if (operation == ANN_OP_INDEX_SEARCH_PREDICATE)
		{
			status = SubmitPredicate(service, connection, 0, request, payload, length);
			if (status == ANN_PENDING)
				continue;
			Reply(connection, operation, request, status, NULL, 0);
		}
		else if (operation == ANN_OP_INDEX_SEARCH_BATCH)
		{
			status = SubmitBatch(service, connection, 0, request, payload, length, generation);
			if (status == ANN_PENDING)
				continue;
			Reply(connection, operation, request, status, NULL, 0);
		}
		else if (operation == ANN_OP_COLLECTION_SEARCH)
		{
			status = SubmitCollection(service, connection, 0, request, payload, length);
			if (status == ANN_PENDING)
				continue;
			Reply(connection, operation, request, status, NULL, 0);
		}
		else if (operation == ANN_OP_INDEX_ROUTING_STATS)
		{
			char		name[ANN_INDEX_NAME_MAX + 1];
			char		message[8192];
			size_t		size = 0;
			AnnIndexFile *file;

			status = ANN_INVALID;
			if (length > 0 && length <= ANN_INDEX_NAME_MAX && !memchr(payload, '\0', length))
			{
				memcpy(name, payload, length);
				name[length] = '\0';
				if ((file = AnnIndexGet(name)) == NULL)
					status = ANN_STALE;
				else
				{
					if (AnnIndexRoutingStats)
						size = AnnIndexRoutingStats(file, message, sizeof(message));
					status = size ? ANN_OK : ANN_UNSUPPORTED;
					AnnIndexPut(file);
				}
			}
			Reply(connection, operation, request, status, (unsigned char *) message, size);
		}
		else if (operation == ANN_OP_INDEX_PREDICATE)
		{
			unsigned char *bitmap = NULL;
			size_t		bitmapLength = 0;

			status = EvaluatePredicate(service, payload, length, &bitmap, &bitmapLength);
			Reply(connection, operation, request, status, bitmap, bitmapLength);
			free(bitmap);
		}
		else if (operation == ANN_OP_BITMAP_PUT)
			Reply(connection, operation, request, PutBitmap(payload, length), NULL, 0);
		else if (operation == ANN_OP_INDEX_BITMAP)
		{
			unsigned char *bitmap = NULL;
			size_t		bitmapLength = 0;

			status = BuildBitmap(payload, length, &bitmap, &bitmapLength);
			Reply(connection, operation, request, status, bitmap, bitmapLength);
			free(bitmap);
		}
		else if (operation == ANN_OP_LOAD_INDEX)
		{
			char		message[512];
			size_t		textLength = 0;

			/* Loading blocks only this connection's reader. */
			status = LoadIndex(service, payload, length, message, sizeof(message), &textLength);
			Reply(connection, operation, request, status, (unsigned char *) message,
				  status == ANN_OK ? textLength : 0);
		}
		else if (operation == ANN_OP_INDEX_INFO)
		{
			char		message[512];
			size_t		textLength = 0;

			status = DescribeIndex(payload, length, message, sizeof(message), &textLength);

			Reply(connection, operation, request, status, (unsigned char *) message,
				  status == ANN_OK ? textLength : 0);
		}
		else if (operation == ANN_OP_DELETE_KEYS)
		{
			/* u32 name length, name, u32 count, keys; reply: added, total */
			char		name[ANN_INDEX_NAME_MAX + 1];
			uint32_t	nameLength = length >= 4 ? (uint32_t) Read(payload, 4) : 0;
			uint32_t	count = 0;
			unsigned char answer[16];
			AnnIndexFile *file = NULL;
			uint64_t	added = 0;

			if (nameLength == 0 || nameLength > ANN_INDEX_NAME_MAX || (uint64_t) length < 8 + (uint64_t) nameLength ||
				(uint64_t) length != 8 + (uint64_t) nameLength +
				8 * (uint64_t) (count = (uint32_t) Read(payload + 4 + nameLength, 4)) ||
				memchr(payload + 4, '\0', nameLength))
				status = ANN_INVALID;
			else
			{
				int64_t    *keys = malloc(sizeof(int64_t) * (count ? count : 1));

				memcpy(name, payload + 4, nameLength);
				name[nameLength] = '\0';
				if (keys == NULL)
					status = ANN_NOMEM;
				else if ((file = AnnIndexGet(name)) == NULL)
					status = ANN_STALE;
				else
				{
					for (uint32_t i = 0; i < count; i++)
						keys[i] = (int64_t) Read(payload + 8 + nameLength + (size_t) i * 8, 8);
					status = AnnIndexDelete(file, keys, count, &added);
					Write(answer, added, 8);
					Write(answer + 8, __atomic_load_n(&file->ndeleted, __ATOMIC_ACQUIRE), 8);
					AnnIndexPut(file);
				}
				free(keys);
			}
			Reply(connection, operation, request, status, answer, status == ANN_OK ? 16 : 0);
		}
		else if (operation == ANN_OP_DICTIONARY || operation == ANN_OP_REMAP)
		{
			unsigned char *data = NULL;
			size_t		dataLength = 0;

			status = DictionaryOperation(operation, payload, length, &data, &dataLength);
			Reply(connection, operation, request, status, data, status == ANN_OK ? dataLength : 0);
			free(data);
		}
		else if (operation == ANN_OP_UNLOAD_INDEX)
		{
			/* u32 name length, name; freed after its searches */
			char		name[ANN_INDEX_NAME_MAX + 1];
			uint32_t	nameLength = length >= 4 ? (uint32_t) Read(payload, 4) : 0;

			if (nameLength == 0 || nameLength > ANN_INDEX_NAME_MAX || (uint64_t) length != 4 + (uint64_t) nameLength ||
				memchr(payload + 4, '\0', nameLength))
				status = ANN_INVALID;
			else
			{
				memcpy(name, payload + 4, nameLength);
				name[nameLength] = '\0';
				status = AnnIndexUnload(name);
			}
			Reply(connection, operation, request, status, NULL, 0);
		}
		else if (operation == ANN_OP_AUTH)
			Reply(connection, operation, request, ANN_OK, NULL, 0);
		else if (operation == ANN_OP_RDMA_HELLO)
		{
			unsigned char message[1400];
			size_t		messageLength = 0;

			status = Hello(service, connection, payload, length, message, sizeof(message), &messageLength);
			Reply(connection, operation, request, status, message, status == ANN_OK ? messageLength : 0);
		}
		else if (operation == ANN_OP_ATTACH_SHARED)
			Reply(connection, operation, request, Attach(connection, payload, length), NULL, 0);
		else if (operation == 5)
			Reply(connection, operation, request, ANN_OK, NULL, 0);
		else
			Reply(connection, operation, request, ANN_UNSUPPORTED, NULL, 0);
		free(payload);
	}
	(void) pthread_mutex_lock(&connection->mutex);
	connection->broken = 1;
	(void) shutdown(connection->fd, SHUT_RDWR);
	(void) pthread_mutex_unlock(&connection->mutex);
	Goodbye(service, connection);
	Release(connection);
	return NULL;
}

int
AnnServe(const char *path, const AnnBackend *backend, const AnnServeOptions *options)
{
	/*
	 * Process lifetime: detached readers and RDMA progress threads hold a
	 * pointer to it, and readers can outlive this function.
	 */
	static Service service;
	struct sockaddr_un address = {0};
	struct sigaction action = {0};
	pthread_t  *lanes;
	int			listener;
	int			network = -1;

	if (options->lanes == 0 || options->max_batch_jobs == 0 ||
		strlen(path) >= sizeof(address.sun_path))
	{
		fprintf(stderr, "invalid server options or socket path\n");
		return 2;
	}
	service.backend = backend;
	service.options = *options;
	if (options->trace)
	{
		if ((traceFile = fopen(options->trace, "a")) == NULL)
		{
			fprintf(stderr, "could not open trace file %s\n", options->trace);
			return 2;
		}
		(void) setvbuf(traceFile, NULL, _IOLBF, 0);
		if (service.options.trace_sample == 0)
			service.options.trace_sample = 1;
	}
	(void) pthread_mutex_init(&service.mutex, NULL);
	(void) pthread_mutex_init(&service.peersMutex, NULL);
	(void) pthread_cond_init(&service.condition, NULL);
	action.sa_handler = OnSignal;
	(void) sigaction(SIGINT, &action, NULL);
	(void) sigaction(SIGTERM, &action, NULL);
	(void) signal(SIGPIPE, SIG_IGN);

	address.sun_family = AF_UNIX;
	strcpy(address.sun_path, path);
	(void) unlink(path);
	listener = socket(AF_UNIX, SOCK_STREAM, 0);
	if (listener < 0 || bind(listener, (struct sockaddr *) &address, sizeof(address)) != 0 ||
		listen(listener, 128) != 0)
	{
		perror("ann-worker listen");
		return 1;
	}
	if (options->tcp)
	{
		char		host[256];
		const char *colon = strrchr(options->tcp, ':');
		struct addrinfo hints = {0};
		struct addrinfo *result;
		int			on = 1;

		if (colon == NULL || colon == options->tcp || (size_t) (colon - options->tcp) >= sizeof(host) ||
			options->token == NULL || strlen(options->token) < 16)
		{
			fprintf(stderr, "--listen-tcp needs HOST:PORT and an --auth-token-file of 16+ bytes\n");
			return 2;
		}
		memcpy(host, options->tcp, (size_t) (colon - options->tcp));
		host[colon - options->tcp] = '\0';
		hints.ai_family = AF_UNSPEC;
		hints.ai_socktype = SOCK_STREAM;
		hints.ai_flags = AI_PASSIVE;
		if (getaddrinfo(host, colon + 1, &hints, &result) != 0 ||
			(network = socket(result->ai_family, SOCK_STREAM, 0)) < 0 ||
			setsockopt(network, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) != 0 ||
			bind(network, result->ai_addr, result->ai_addrlen) != 0 || listen(network, 1024) != 0)
		{
			perror("ann-worker tcp listen");
			return 1;
		}
		freeaddrinfo(result);
	}
	lanes = calloc(options->lanes, sizeof(*lanes));
	if (lanes == NULL)
		return 1;
	for (uint32_t i = 0; i < options->lanes; i++)
		if (pthread_create(&lanes[i], NULL, Lane, &service) != 0)
			return 1;
	if (options->rdma)
	{
		char		error[512];

		service.rdma = AnnRdmaOpen(options->rdma, (int) options->rdma_lanes, options->rdma_spin, RdmaFrame, &service,
								   error, sizeof(error));
		if (service.rdma == NULL)
		{
			fprintf(stderr, "ann-worker: %s\n", error);
			return 1;
		}
		fprintf(stderr, "ann-worker: RDMA on %s, %u lane(s), frames up to %zu bytes, %s completions\n",
				AnnRdmaDevice(service.rdma), options->rdma_lanes, AnnRdmaMaxFrame(service.rdma),
				options->rdma_spin ? "spin-polled" : "interrupt-driven");
	}
	fprintf(stderr, "ann-worker: backend %s listening on %s%s%s\n", backend->name, path,
			options->tcp ? " and " : "", options->tcp ? options->tcp : "");
	while (!stopRequested)
	{
		struct pollfd pollers[2] = {{listener, POLLIN, 0}, {network, POLLIN, 0}};
		struct timeval timeout = {10, 0};
		int			buffer = 4 * 1024 * 1024;
		Connection *connection;
		ReaderArgs *args;
		pthread_t	thread;
		int			fd;

		int			remote;

		if (poll(pollers, network >= 0 ? 2 : 1, 200) <= 0)
			continue;
		remote = !(pollers[0].revents & POLLIN);
		fd = accept(remote ? network : listener, NULL, NULL);
		if (fd < 0)
			continue;
		if (remote)
		{
			int			on = 1;
			int			idle = 10,
						interval = 5,
						count = 3;
			struct timeval authDeadline = {10, 0};

			/* Small request/reply frames: disable Nagle batching. */
			(void) setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on));

			/*
			 * Keepalives drop connections from PostgreSQL hosts that crashed
			 * or vanished, releasing their reader and queued replies.
			 */
			(void) setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &on, sizeof(on));
#ifdef TCP_KEEPIDLE
			(void) setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));
#elif defined(TCP_KEEPALIVE)
			(void) setsockopt(fd, IPPROTO_TCP, TCP_KEEPALIVE, &idle, sizeof(idle));
#endif
#ifdef TCP_KEEPINTVL
			(void) setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &interval, sizeof(interval));
#endif
#ifdef TCP_KEEPCNT
			(void) setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &count, sizeof(count));
#endif
			/* Unauthenticated peers get a short deadline to send the token. */
			(void) setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &authDeadline, sizeof(authDeadline));
			(void) idle;
			(void) interval;
			(void) count;
		}
		/* A stalled client cannot block a lane indefinitely. */
		(void) setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
		(void) setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buffer, sizeof(buffer));
		(void) setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buffer, sizeof(buffer));
		connection = calloc(1, sizeof(*connection));
		args = malloc(sizeof(*args));
		if (connection == NULL || args == NULL)
		{
			free(connection);
			free(args);
			(void) close(fd);
			continue;
		}
		connection->fd = fd;
		connection->remote = remote;
		connection->references = 1;
		(void) pthread_mutex_init(&connection->mutex, NULL);
		args->service = &service;
		args->connection = connection;
		if (pthread_create(&thread, NULL, Reader, args) != 0)
		{
			free(args);
			Release(connection);
			continue;
		}
		(void) pthread_detach(thread);
	}
	(void) close(listener);
	if (network >= 0)
		(void) close(network);
	(void) unlink(path);
	(void) pthread_mutex_lock(&service.mutex);
	service.stopping = 1;
	(void) pthread_cond_broadcast(&service.condition);
	(void) pthread_mutex_unlock(&service.mutex);
	for (uint32_t i = 0; i < options->lanes; i++)
		(void) pthread_join(lanes[i], NULL);
	free(lanes);

	/*
	 * Stop RDMA progress threads so no new frames are handled; the rest of
	 * the RDMA state is left for process exit to release, because detached
	 * readers may still drop connections, and with them queue pairs.
	 */
	if (service.rdma)
		AnnRdmaStop(service.rdma);
	return 0;
}
