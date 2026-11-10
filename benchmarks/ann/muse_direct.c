/*
 * Muse-shaped load straight to an ANN server over TCP, without PostgreSQL.
 *
 *   muse_direct HOST PORT TOKEN_FILE FRAMES SECONDS CLIENTS [WARMUP]
 *               [--processes N] [--rate R] [--epoll]
 *
 * FRAMES is written by benchmarks/ann/muse_stress.py --emit-ann: u32
 * requests, u32 sub-requests per request, then per sub-request a u32 size
 * and a ready protocol frame (operation 22, search by predicate program).
 * Each client (a thread, unless --epoll) holds one authenticated connection
 * per sub-request, sends a request's sub-requests together and waits for
 * every reply, as a Muse SCORE_BLEND request runs its rankers in parallel;
 * the request's latency is its slowest sub-request's. Prints one JSON line.
 *
 * WARMUP (seconds, default 3) runs load before the SECONDS that are
 * measured. A request is counted when its last reply arrives inside the
 * measured window.
 *
 * --processes N (default 1) forks N processes that share the CLIENTS
 * (CLIENTS / N each, the remainder spread). Counters and histograms live in
 * shared memory, so the one JSON line has percentiles over every request of
 * every process. No process sends before every client of every process has
 * connected and authenticated; all start at the same instant.
 *
 * Connecting: at most 256 connection attempts are in flight across all
 * processes. A refused, reset or timed out attempt, or one whose
 * authentication reply never arrives, is retried with backoff; a connection
 * that cannot be made within 60 s ends the run with an error and no JSON
 * (the server closes a connection whose token is wrong without a reply, so
 * that too shows up after 60 s).
 * Above 8191 connections to a loopback address the sockets are bound to
 * several 127.0.0.x source addresses, because one source address has fewer
 * ephemeral ports (net.ipv4.ip_local_port_range) than that many connections
 * to one server port may need. The descriptor limit is raised to the hard
 * limit.
 *
 * --rate R switches from closed loop (each client sends its next request
 * when the previous one is answered) to open loop: R requests per second in
 * total, each client sending on a fixed schedule of its own, the clients'
 * schedules evenly staggered. Latency is then measured from the scheduled
 * send time, not the actual one: a client still waiting for its previous
 * request at the scheduled time sends late and the lateness counts, so a
 * server slower than R shows growing latency instead of a lower offered
 * load (no coordinated omission). "achieved_rate" is the completed
 * requests per second; "unsent" counts the scheduled sends of the measured
 * window that had not been sent when it ended.
 *
 * --epoll (Linux only; elsewhere the option is refused) replaces the thread
 * per client with one thread per process that drives all of the process's
 * clients through one level-triggered epoll set: non-blocking sockets, a
 * frame sent when its request starts (what the socket does not take at once
 * follows when it becomes writable), replies read as they arrive, the
 * --rate schedules kept in a heap behind a timerfd. Everything is measured
 * and reported as with threads; only the load generator's own cost differs
 * (one core per process at most, instead of a thread per client). The one
 * thread serves its clients in turn, so a process with more work than one
 * core can do adds its own queueing to the latencies: give it enough
 * --processes. Connections are still made with blocking calls, by up to 256
 * short-lived threads per process, before the run starts.
 *
 * JSON: "request_ms" percentiles and "max_ms" over the measured requests;
 * "subrequest_ms" by position of the sub-request within the request, from
 * the request's send (scheduled send with --rate) to that reply's arrival;
 * "requests_per_s_series" the requests completed in each measured second;
 * "connect_seconds" from the first attempt to the last authentication;
 * "connect_retries" the attempts that had to be repeated; "mode" is
 * "epoll" or "threads".
 *
 * Exit status: 0 with the JSON line; 1 bad arguments, or connecting or
 * authenticating failed; 2 a connection broke or stalled during the run.
 */
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE			/* MAP_ANONYMOUS is not POSIX (glibc) */
#define _DARWIN_C_SOURCE		/* likewise MAP_ANON (macOS) */
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/epoll.h>
#include <sys/timerfd.h>
#endif

#if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#define MAP_ANONYMOUS MAP_ANON
#endif

#define FINE_BINS 200000		/* 10 us bins below 2 s */
#define COARSE_BINS 598001		/* then 1 ms bins; the last holds >= 600 s */
#define BINS (FINE_BINS + COARSE_BINS)
#define MAX_SUBS 32
#define MAX_CONNECTING 256		/* connection attempts in flight, all
								 * processes */
#define CONNECT_SECONDS 60		/* per connection, retries included */
#define ATTEMPT_SECONDS 5		/* connect and authentication reply, each */
#define REPLY_SECONDS 60		/* a silent server ends the run */
#define PER_SOURCE 8192			/* connections per loopback source address */
#define STACK_BYTES (256 * 1024)
#define SECOND UINT64_C(1000000000)

/* Shared by every process of the run (anonymous shared mapping). */
typedef struct Shared
{
	pthread_mutex_t lock;		/* slots, go and the three times */
	pthread_cond_t slotFree;
	pthread_cond_t started;
	unsigned	slots;			/* connection attempts that may begin */
	int			go;
	uint64_t	start,
				measureFrom,
				deadline;
	atomic_uint connected;		/* clients with every connection ready */
	atomic_int	failed;			/* the first failure has been reported */
	atomic_uint_fast64_t requests,
				subrequests,
				errors,
				hits,
				notReady,
				unsent,
				retries,
				maxNs;

	/*
	 * BINS counts of request latency, then BINS per sub-request position,
	 * then the requests completed in each measured second.
	 */
	atomic_uint_fast64_t counts[];
} Shared;

/* One reply being read. */
typedef struct Inbox
{
	unsigned char *data;
	size_t		capacity;
	size_t		have;
} Inbox;

static const char *host,
		   *port;
static char token[256];
static size_t tokenSize;
static unsigned char **frames;	/* [request * subs + sub] */
static uint32_t *frameSizes;
static uint32_t nrequests,
			nsubs;
static struct sockaddr_storage server;
static socklen_t serverSize;
static int	serverFamily;
static unsigned sourceAddresses = 1;
static unsigned seconds,
			clients;
static double interval;			/* ns between one client's sends; 0: closed
								 * loop */
static Shared *shared;
static pid_t parent;
static pid_t *children;
static unsigned nchildren;

static uint64_t
Now(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return (uint64_t) t.tv_sec * SECOND + (uint64_t) t.tv_nsec;
}

static void
Sleep(uint64_t ns)
{
	struct timespec t;

	t.tv_sec = (time_t) (ns / SECOND);
	t.tv_nsec = (long) (ns % SECOND);
	(void) nanosleep(&t, NULL);
}

static void
SleepUntil(uint64_t when)
{
	for (;;)
	{
		uint64_t	now = Now();

		if (now >= when)
			return;
		Sleep(when - now);
	}
}

static uint32_t
Read32(const unsigned char *p)
{
	return (uint32_t) p[0] | (uint32_t) p[1] << 8 | (uint32_t) p[2] << 16 | (uint32_t) p[3] << 24;
}

/* Reports the run's first failure and ends this process; the parent ends the others. */
static void
Fatal(int code, const char *format,...)
{
	if (!atomic_exchange(&shared->failed, 1))
	{
		va_list		arguments;

		va_start(arguments, format);
		fprintf(stderr, "muse_direct: ");
		vfprintf(stderr, format, arguments);
		fprintf(stderr, "\n");
		va_end(arguments);
	}
	_exit(code);
}

static uint32_t
Bin(uint64_t ns)
{
	uint64_t	ms;

	if (ns < 2 * SECOND)
		return (uint32_t) (ns / 10000);
	ms = (ns - 2 * SECOND) / 1000000;
	return FINE_BINS + (uint32_t) (ms < COARSE_BINS - 1 ? ms : COARSE_BINS - 1);
}

static double
BinMs(uint32_t bin)
{
	return bin < FINE_BINS ? (bin + 0.5) / 100.0 : 2000.0 + (bin - FINE_BINS) + 0.5;
}

static double
Percentile(const atomic_uint_fast64_t *histogram, uint64_t total, double p)
{
	uint64_t	want = (uint64_t) (p * (double) total + 0.999999),
				seen = 0;

	for (uint32_t i = 0; i < BINS; i++)
		if ((seen += atomic_load(&histogram[i])) >= want && want)
			return BinMs(i);
	return 0;
}

/* errno is 0 after a failure when the peer closed the connection. */
static int
Transfer(int fd, unsigned char *data, size_t size, int sending)
{
	while (size)
	{
		ssize_t		n = sending ? send(fd, data, size, MSG_NOSIGNAL) : recv(fd, data, size, 0);

		if (n < 0 && errno == EINTR)
			continue;
		if (n == 0)
			errno = 0;
		if (n <= 0)
			return 0;
		data += n;
		size -= (size_t) n;
	}
	return 1;
}

/* Reads one reply frame, blocking; its status, or -1 with errno on a broken connection. */
static int
Reply(int fd)
{
	unsigned char header[32];
	unsigned char payload[256];
	uint32_t	size;

	if (!Transfer(fd, header, sizeof(header), 0))
		return -1;
	if (memcmp(header, "PVAN", 4) != 0)
	{
		errno = EPROTO;
		return -1;
	}
	for (size = Read32(header + 24); size;)
	{
		size_t		part = size < sizeof(payload) ? size : sizeof(payload);

		if (!Transfer(fd, payload, part, 0))
			return -1;
		size -= (uint32_t) part;
	}
	return (int) Read32(header + 28);
}

/*
 * Reads what has arrived of a reply: 1 when the frame is complete, 0 when
 * more is to come, -1 on a broken connection. One reply is outstanding per
 * connection, so everything that arrives belongs to it.
 */
static int
Receive(int fd, Inbox *inbox)
{
	ssize_t		n;
	size_t		need;

	do
		n = recv(fd, inbox->data + inbox->have, inbox->capacity - inbox->have, 0);
	while (n < 0 && errno == EINTR);
	if (n <= 0)
		return -1;
	inbox->have += (size_t) n;
	if (inbox->have < 32)
		return 0;
	if (memcmp(inbox->data, "PVAN", 4) != 0)
		return -1;
	need = 32 + (size_t) Read32(inbox->data + 24);
	if (inbox->have > need)
		return -1;
	if (need > inbox->capacity)
	{
		inbox->data = realloc(inbox->data, need);
		inbox->capacity = need;
		if (inbox->data == NULL)
			return -1;
	}
	return inbox->have == need;
}

static void
TakeSlot(void)
{
	pthread_mutex_lock(&shared->lock);
	while (shared->slots == 0)
		pthread_cond_wait(&shared->slotFree, &shared->lock);
	shared->slots--;
	pthread_mutex_unlock(&shared->lock);
}

static void
FreeSlot(void)
{
	pthread_mutex_lock(&shared->lock);
	shared->slots++;
	pthread_cond_signal(&shared->slotFree);
	pthread_mutex_unlock(&shared->lock);
}

/* Failures of a connection attempt that a busy or restarting server causes. */
static int
Retryable(int error)
{
	return error == 0 || error == ECONNREFUSED || error == ETIMEDOUT || error == ECONNRESET ||
		error == ECONNABORTED || error == EAGAIN || error == EWOULDBLOCK || error == EINPROGRESS ||
		error == EADDRNOTAVAIL || error == EPIPE || error == EINTR;
}

/*
 * One attempt at an authenticated connection: the descriptor, or -1 with
 * *error (0 when the server closed the connection before replying). Both
 * steps have a deadline: a server whose accept queue is full leaves the
 * client connected and waiting for a reply that never comes.
 */
static int
Attempt(unsigned connection, int *error)
{
	unsigned char header[32] = {'P', 'V', 'A', 'N', 1, 0, 13, 0};
	struct timeval limit = {ATTEMPT_SECONDS, 0},
				none = {0, 0};
	int			fd,
				on = 1,
				status;

	fd = socket(serverFamily, SOCK_STREAM, 0);
	if (fd < 0)
	{
		*error = errno;
		return -1;
	}
	(void) setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &limit, sizeof(limit));
	(void) setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &limit, sizeof(limit));
	if (sourceAddresses > 1)
	{
		struct sockaddr_in from;

		memset(&from, 0, sizeof(from));
		from.sin_family = AF_INET;
		from.sin_addr.s_addr = htonl(UINT32_C(0x7f000001) + connection % sourceAddresses);
#ifdef IP_BIND_ADDRESS_NO_PORT

		/*
		 * Let connect choose the port: it may then be one in use towards
		 * another address.
		 */
		(void) setsockopt(fd, IPPROTO_IP, IP_BIND_ADDRESS_NO_PORT, &on, sizeof(on));
#endif
		(void) bind(fd, (struct sockaddr *) &from, sizeof(from));
	}
	if (connect(fd, (struct sockaddr *) &server, serverSize) != 0)
	{
		*error = errno;
		close(fd);
		return -1;
	}
	(void) setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on));
	header[8] = 1;				/* request id */
	header[24] = (unsigned char) tokenSize;
	header[25] = (unsigned char) (tokenSize >> 8);
	if (!Transfer(fd, header, sizeof(header), 1) || !Transfer(fd, (unsigned char *) token, tokenSize, 1) ||
		(status = Reply(fd)) < 0)
	{
		*error = errno;
		close(fd);
		return -1;
	}
	if (status != 0)
		Fatal(1, "%s:%s rejected the token (status %d)", host, port, status);
	(void) setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &none, sizeof(none));
	(void) setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &none, sizeof(none));
	return fd;
}

/* An authenticated connection, retrying with backoff; does not return without one. */
static int
Connect(unsigned connection, uint32_t *seed)
{
	uint64_t	began = 0,
				backoff = 10000000;
	unsigned	attempts = 0;

	for (;;)
	{
		int			error = 0,
					fd;

		TakeSlot();
		if (began == 0)
			began = Now();
		fd = Attempt(connection, &error);
		FreeSlot();
		if (fd >= 0)
			return fd;
		attempts++;
		if (!Retryable(error))
			Fatal(1, "cannot connect to %s:%s: %s", host, port, strerror(error));
		if (Now() - began >= CONNECT_SECONDS * SECOND)
			Fatal(1, "no authenticated connection to %s:%s within %d s (%u attempts; last failure: %s)",
				  host, port, CONNECT_SECONDS, attempts,
				  error ? strerror(error) : "closed by the server before the authentication reply (wrong token?)");
		atomic_fetch_add(&shared->retries, 1);

		/*
		 * Between half and all of the backoff, so that retries do not arrive
		 * together.
		 */
		*seed = *seed * UINT32_C(1664525) + UINT32_C(1013904223);
		Sleep(backoff / 2 + backoff / 2 * (*seed >> 16) / 65536);
		if (backoff < 500000000)
			backoff *= 2;
	}
}

static void *
Client(void *arg)
{
	unsigned	client = (unsigned) (uintptr_t) arg;	/* among all processes */
	uint32_t	r = (uint32_t) ((uint64_t) client * nrequests / clients);
	uint32_t	seed = client * UINT32_C(2654435761) + 1;
	int			fds[MAX_SUBS];
	struct pollfd polls[MAX_SUBS];
	Inbox		inbox[MAX_SUBS];
	uint64_t	arrival[MAX_SUBS];
	uint32_t   *series = calloc(seconds, sizeof(*series));
	uint64_t	start,
				measureFrom,
				deadline,
				first,
				sends = 0;
	uint64_t	requests = 0,
				subrequests = 0,
				errors = 0,
				hits = 0,
				notReady = 0,
				unsent = 0,
				maxNs = 0,
				seen;

	if (series == NULL)
		Fatal(1, "out of memory");
	for (uint32_t s = 0; s < nsubs; s++)
	{
		fds[s] = Connect(client * nsubs + s, &seed);
		inbox[s].capacity = 16384;
		inbox[s].have = 0;
		if ((inbox[s].data = malloc(inbox[s].capacity)) == NULL)
			Fatal(1, "out of memory");
		polls[s].events = POLLIN;
	}
	atomic_fetch_add(&shared->connected, 1);

	pthread_mutex_lock(&shared->lock);
	while (!shared->go)
	{
		struct timespec until;

		clock_gettime(CLOCK_REALTIME, &until);
		until.tv_sec += 1;
		(void) pthread_cond_timedwait(&shared->started, &shared->lock, &until);
		if (getppid() != parent)
			_exit(3);			/* the parent is gone: no run will start */
	}
	start = shared->start;
	measureFrom = shared->measureFrom;
	deadline = shared->deadline;
	pthread_mutex_unlock(&shared->lock);

	/*
	 * Open loop: this client's schedule, staggered against the other
	 * clients'.
	 */
	first = start + (uint64_t) (interval * client / clients);
	SleepUntil(interval > 0 ? first : start);
	for (;;)
	{
		uint64_t	origin,
					sent,
					end = 0;
		uint32_t	pending = nsubs;
		int			failed = 0;

		if (interval > 0)
		{
			uint64_t	now = Now();

			origin = first + (uint64_t) (interval * (double) sends);
			if (origin >= deadline)
				break;
			if (now >= deadline)
			{
				/* Still behind at the end: these were due and never went out. */
				if (origin < measureFrom)
					origin = measureFrom;
				unsent = (uint64_t) ((double) (deadline - origin) / interval) + 1;
				break;
			}
			if (now < origin)
				SleepUntil(origin);
			sends++;
			sent = now < origin ? origin : now;
		}
		else
		{
			sent = origin = Now();
			if (origin >= deadline)
				break;
		}
		for (uint32_t s = 0; s < nsubs; s++)
		{
			if (!Transfer(fds[s], frames[(size_t) r * nsubs + s], frameSizes[(size_t) r * nsubs + s], 1))
				Fatal(2, "connection to %s:%s broke while sending: %s", host, port,
					  errno ? strerror(errno) : "closed by the server");
			polls[s].fd = fds[s];
			inbox[s].have = 0;
		}
		while (pending)
		{
			int			n = poll(polls, nsubs, 1000);

			if (n < 0 && errno != EINTR)
				Fatal(2, "poll: %s", strerror(errno));
			if (n <= 0)
			{
				if (Now() - sent >= REPLY_SECONDS * SECOND)
					Fatal(2, "no reply from %s:%s within %d s", host, port, REPLY_SECONDS);
				continue;
			}
			for (uint32_t s = 0; s < nsubs; s++)
			{
				int			done;

				if (polls[s].fd < 0 || polls[s].revents == 0)
					continue;
				done = Receive(fds[s], &inbox[s]);
				if (done < 0)
					Fatal(2, "connection to %s:%s broke while waiting for a reply", host, port);
				if (!done)
					continue;
				arrival[s] = end = Now();
				polls[s].fd = -1;	/* poll skips it */
				pending--;
			}
		}
		r = (r + 1) % nrequests;
		if (end < measureFrom || end >= deadline)
			continue;
		for (uint32_t s = 0; s < nsubs; s++)
		{
			uint32_t	status = Read32(inbox[s].data + 28);

			subrequests++;
			if (inbox[s].have >= 36)
				hits += Read32(inbox[s].data + 32);
			if (status != 0)
				failed = 1;
			if (status == 3 || status == 11)
				notReady++;
		}
		if (failed)
		{
			errors++;
			continue;
		}
		requests++;
		series[(end - measureFrom) / SECOND]++;
		if (end - origin > maxNs)
			maxNs = end - origin;
		atomic_fetch_add(&shared->counts[Bin(end - origin)], 1);
		for (uint32_t s = 0; s < nsubs; s++)
			atomic_fetch_add(&shared->counts[(size_t) (1 + s) * BINS + Bin(arrival[s] - origin)], 1);
	}
	for (uint32_t s = 0; s < nsubs; s++)
	{
		close(fds[s]);
		free(inbox[s].data);
	}
	atomic_fetch_add(&shared->requests, requests);
	atomic_fetch_add(&shared->subrequests, subrequests);
	atomic_fetch_add(&shared->errors, errors);
	atomic_fetch_add(&shared->hits, hits);
	atomic_fetch_add(&shared->notReady, notReady);
	atomic_fetch_add(&shared->unsent, unsent);
	for (unsigned i = 0; i < seconds; i++)
		if (series[i])
			atomic_fetch_add(&shared->counts[(size_t) (1 + nsubs) * BINS + i], series[i]);
	seen = atomic_load(&shared->maxNs);
	while (seen < maxNs && !atomic_compare_exchange_weak(&shared->maxNs, &seen, maxNs))
		;
	free(series);
	return NULL;
}

/* A forked process: clients [first, first + count) of the run. */
static void
Process(unsigned first, unsigned count)
{
	pthread_t  *threads = calloc(count, sizeof(*threads));
	pthread_attr_t attributes;

	if (threads == NULL || pthread_attr_init(&attributes) != 0 ||
		pthread_attr_setstacksize(&attributes, STACK_BYTES) != 0)
		Fatal(1, "cannot prepare %u threads", count);
	for (unsigned c = 0; c < count; c++)
	{
		int			error = pthread_create(&threads[c], &attributes, Client, (void *) (uintptr_t) (first + c));

		if (error != 0)
			Fatal(1, "pthread_create (thread %u of %u in one process): %s", c + 1, count, strerror(error));
	}
	for (unsigned c = 0; c < count; c++)
		pthread_join(threads[c], NULL);
	_exit(0);
}

#ifdef __linux__
/*
 * --epoll: one thread drives every client of the process. What is counted,
 * and when, follows Client() above line by line.
 */
#define EPOLL_EVENTS 1024
#define SCRATCH_BYTES 65536

/* One connection: a frame on its way out, then its reply on its way in. */
typedef struct EpollConnection
{
	int			fd;				/* -1 once its client has finished */
	uint32_t	client;			/* within the process */
	int			waiting;		/* a reply is outstanding */
	int			writing;		/* registered for EPOLLOUT too */
	const unsigned char *out;	/* what of the frame is still to send */
	size_t		outLeft;
	size_t		have;			/* bytes of the reply read */
	size_t		need;			/* its size; 0 until the header is read */
	uint64_t	arrival;
	unsigned char head[36];		/* the header and the hit count */
} EpollConnection;

typedef struct EpollClient
{
	uint32_t	r;				/* next request */
	uint32_t	pending;		/* replies outstanding */
	atomic_uint ready;			/* connections made */
	uint64_t	first,			/* open loop: the first scheduled send */
				sends,
				origin,			/* (scheduled) send of the current request */
				sent,
				end;
} EpollClient;

static int	epollSet,
			epollTimer;
static unsigned epollFirst,		/* this process's first client, among all */
			epollCount,
			epollLive;			/* clients that have not finished */
static EpollClient *epollClients;
static EpollConnection *epollConnections;	/* [client * nsubs + sub] */
static atomic_uint epollNext;	/* next connection to make */
static uint32_t *epollHeap;		/* idle clients, by origin (open loop) */
static unsigned epollHeaped;
static uint64_t epollArmed;		/* what the timer is set to */
static uint64_t epollMeasureFrom,
			epollDeadline;
static uint32_t *epollSeries;
static uint64_t epollRequests,
			epollSubrequests,
			epollErrors,
			epollHits,
			epollNotReady,
			epollUnsent,
			epollMaxNs;
static unsigned char epollScratch[SCRATCH_BYTES];

static void
EpollPush(uint32_t c)
{
	unsigned	i = epollHeaped++;

	while (i > 0)
	{
		unsigned	up = (i - 1) / 2;

		if (epollClients[epollHeap[up]].origin <= epollClients[c].origin)
			break;
		epollHeap[i] = epollHeap[up];
		i = up;
	}
	epollHeap[i] = c;
}

static uint32_t
EpollPop(void)
{
	uint32_t	top = epollHeap[0],
				last = epollHeap[--epollHeaped];
	unsigned	i = 0;

	for (;;)
	{
		unsigned	child = 2 * i + 1;

		if (child >= epollHeaped)
			break;
		if (child + 1 < epollHeaped &&
			epollClients[epollHeap[child + 1]].origin < epollClients[epollHeap[child]].origin)
			child++;
		if (epollClients[last].origin <= epollClients[epollHeap[child]].origin)
			break;
		epollHeap[i] = epollHeap[child];
		i = child;
	}
	epollHeap[i] = last;
	return top;
}

/* Sends what the socket takes now; the rest goes when it is writable again. */
static void
EpollFlush(EpollConnection *connection)
{
	int			writing;

	while (connection->outLeft)
	{
		ssize_t		n = send(connection->fd, connection->out, connection->outLeft, MSG_NOSIGNAL);

		if (n < 0 && errno == EINTR)
			continue;
		if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
			break;
		if (n <= 0)
			Fatal(2, "connection to %s:%s broke while sending: %s", host, port,
				  n < 0 ? strerror(errno) : "closed by the server");
		connection->out += n;
		connection->outLeft -= (size_t) n;
	}
	writing = connection->outLeft != 0;
	if (writing != connection->writing)
	{
		struct epoll_event event;

		memset(&event, 0, sizeof(event));
		event.events = EPOLLIN | (writing ? EPOLLOUT : 0);
		event.data.ptr = connection;
		if (epoll_ctl(epollSet, EPOLL_CTL_MOD, connection->fd, &event) != 0)
			Fatal(2, "epoll_ctl: %s", strerror(errno));
		connection->writing = writing;
	}
}

/* Sends the client's next request; its origin is set. */
static void
EpollLaunch(uint32_t c, uint64_t now)
{
	EpollClient *client = &epollClients[c];

	client->sent = now;
	client->pending = nsubs;
	for (uint32_t s = 0; s < nsubs; s++)
	{
		EpollConnection *connection = &epollConnections[(size_t) c * nsubs + s];

		connection->out = frames[(size_t) client->r * nsubs + s];
		connection->outLeft = frameSizes[(size_t) client->r * nsubs + s];
		connection->have = 0;
		connection->need = 0;
		connection->waiting = 1;
		EpollFlush(connection);
	}
}

/*
 * The client is idle: sends its next request, or queues it until its
 * scheduled time, or ends the client when the run is over.
 */
static void
EpollNext(uint32_t c)
{
	EpollClient *client = &epollClients[c];
	uint64_t	now = Now();

	if (interval > 0)
	{
		client->origin = client->first + (uint64_t) (interval * (double) client->sends);
		if (client->origin < epollDeadline && now < epollDeadline)
		{
			if (now < client->origin)
			{
				EpollPush(c);
				return;
			}
			client->sends++;
			EpollLaunch(c, now);
			return;
		}
		if (client->origin < epollDeadline)
		{
			/* Still behind at the end: these were due and never went out. */
			uint64_t	origin = client->origin < epollMeasureFrom ? epollMeasureFrom : client->origin;

			epollUnsent += (uint64_t) ((double) (epollDeadline - origin) / interval) + 1;
		}
	}
	else if (now < epollDeadline)
	{
		client->origin = now;
		EpollLaunch(c, now);
		return;
	}
	for (uint32_t s = 0; s < nsubs; s++)
	{
		EpollConnection *connection = &epollConnections[(size_t) c * nsubs + s];

		close(connection->fd);	/* which also takes it out of the set */
		connection->fd = -1;
	}
	epollLive--;
}

/* Every reply of the client's request has arrived. */
static void
EpollAnswered(uint32_t c)
{
	EpollClient *client = &epollClients[c];
	EpollConnection *connections = &epollConnections[(size_t) c * nsubs];
	uint64_t	origin = client->origin,
				end = client->end;
	int			failed = 0;

	client->r = (client->r + 1) % nrequests;
	if (end < epollMeasureFrom || end >= epollDeadline)
		return;
	for (uint32_t s = 0; s < nsubs; s++)
	{
		uint32_t	status = Read32(connections[s].head + 28);

		epollSubrequests++;
		if (connections[s].have >= 36)
			epollHits += Read32(connections[s].head + 32);
		if (status != 0)
			failed = 1;
		if (status == 3 || status == 11)
			epollNotReady++;
	}
	if (failed)
	{
		epollErrors++;
		return;
	}
	epollRequests++;
	epollSeries[(end - epollMeasureFrom) / SECOND]++;
	if (end - origin > epollMaxNs)
		epollMaxNs = end - origin;
	atomic_fetch_add(&shared->counts[Bin(end - origin)], 1);
	for (uint32_t s = 0; s < nsubs; s++)
		atomic_fetch_add(&shared->counts[(size_t) (1 + s) * BINS + Bin(connections[s].arrival - origin)], 1);
}

/*
 * Reads what has arrived on a connection. One reply is outstanding per
 * connection at most, so everything that arrives belongs to it; all but
 * the header and the hit count goes to the scratch buffer.
 */
static void
EpollRead(EpollConnection *connection)
{
	EpollClient *client = &epollClients[connection->client];

	if (!connection->waiting)
	{
		/*
		 * Nothing is expected: the server closed it, or sent what it should
		 * not.
		 */
		ssize_t		n = recv(connection->fd, epollScratch, sizeof(epollScratch), 0);

		if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR))
			return;
		Fatal(2, "connection to %s:%s broke between requests: %s", host, port,
			  n < 0 ? strerror(errno) : n == 0 ? "closed by the server" : "data without a request");
	}
	for (;;)
	{
		unsigned char *into;
		size_t		want;
		ssize_t		n;

		if (connection->have < sizeof(connection->head))
		{
			into = connection->head + connection->have;
			want = (connection->need && connection->need < sizeof(connection->head) ?
					connection->need : sizeof(connection->head)) - connection->have;
		}
		else
		{
			into = epollScratch;
			want = connection->need - connection->have;
			if (want > sizeof(epollScratch))
				want = sizeof(epollScratch);
		}
		n = recv(connection->fd, into, want, 0);
		if (n < 0 && errno == EINTR)
			continue;
		if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
			return;
		if (n <= 0)
			Fatal(2, "connection to %s:%s broke while waiting for a reply", host, port);
		connection->have += (size_t) n;
		if (connection->need == 0 && connection->have >= 32)
		{
			if (memcmp(connection->head, "PVAN", 4) != 0)
				Fatal(2, "connection to %s:%s broke while waiting for a reply", host, port);
			connection->need = 32 + (size_t) Read32(connection->head + 24);
			if (connection->have > connection->need)
				Fatal(2, "connection to %s:%s broke while waiting for a reply", host, port);
		}
		if (connection->need && connection->have == connection->need)
			break;
		if ((size_t) n < want)
			return;				/* the socket is empty; epoll tells of more */
	}
	connection->waiting = 0;
	connection->arrival = client->end = Now();
	if (--client->pending == 0)
	{
		EpollAnswered(connection->client);
		EpollNext(connection->client);
	}
}

/* Makes connections until none is left to make; several threads run this. */
static void *
EpollConnector(void *arg)
{
	(void) arg;
	for (;;)
	{
		unsigned	i = atomic_fetch_add(&epollNext, 1);
		EpollConnection *connection;
		struct epoll_event event;
		uint32_t	seed;
		int			flags;

		if (i >= epollCount * nsubs)
			return NULL;
		connection = &epollConnections[i];
		seed = (epollFirst * nsubs + i) * UINT32_C(2654435761) + 1;
		connection->fd = Connect(epollFirst * nsubs + i, &seed);
		connection->client = i / nsubs;
		flags = fcntl(connection->fd, F_GETFL, 0);
		if (flags < 0 || fcntl(connection->fd, F_SETFL, flags | O_NONBLOCK) != 0)
			Fatal(1, "fcntl: %s", strerror(errno));
		memset(&event, 0, sizeof(event));
		event.events = EPOLLIN;
		event.data.ptr = connection;
		if (epoll_ctl(epollSet, EPOLL_CTL_ADD, connection->fd, &event) != 0)
			Fatal(1, "epoll_ctl: %s", strerror(errno));
		/* The client counts as connected when nothing is left to set up. */
		if (atomic_fetch_add(&epollClients[i / nsubs].ready, 1) + 1 == nsubs)
			atomic_fetch_add(&shared->connected, 1);
	}
}

/* A forked process with --epoll: clients [first, first + count) of the run. */
static void
EpollProcess(unsigned first, unsigned count)
{
	unsigned	nconnectors = count * nsubs < MAX_CONNECTING ? count * nsubs : MAX_CONNECTING;
	pthread_t  *threads = calloc(nconnectors, sizeof(*threads));
	pthread_attr_t attributes;
	struct epoll_event *events = calloc(EPOLL_EVENTS, sizeof(*events));
	struct epoll_event event;
	uint64_t	start,
				scanned,
				seen;

	epollFirst = first;
	epollCount = epollLive = count;
	epollClients = calloc(count, sizeof(*epollClients));
	epollConnections = calloc((size_t) count * nsubs, sizeof(*epollConnections));
	epollHeap = calloc(count, sizeof(*epollHeap));
	epollSeries = calloc(seconds, sizeof(*epollSeries));
	if (threads == NULL || events == NULL || epollClients == NULL || epollConnections == NULL ||
		epollHeap == NULL || epollSeries == NULL)
		Fatal(1, "out of memory");
	for (unsigned c = 0; c < count; c++)
		atomic_init(&epollClients[c].ready, 0);
	if ((epollSet = epoll_create1(0)) < 0)
		Fatal(1, "epoll_create1: %s", strerror(errno));
	if ((epollTimer = timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK)) < 0)
		Fatal(1, "timerfd_create: %s", strerror(errno));
	memset(&event, 0, sizeof(event));
	event.events = EPOLLIN;
	event.data.ptr = NULL;		/* the timer; connections have theirs */
	if (epoll_ctl(epollSet, EPOLL_CTL_ADD, epollTimer, &event) != 0)
		Fatal(1, "epoll_ctl: %s", strerror(errno));

	/* Connect as the threads of Process() do, MAX_CONNECTING at a time. */
	if (pthread_attr_init(&attributes) != 0 || pthread_attr_setstacksize(&attributes, STACK_BYTES) != 0)
		Fatal(1, "cannot prepare %u threads", nconnectors);
	for (unsigned t = 0; t < nconnectors; t++)
	{
		int			error = pthread_create(&threads[t], &attributes, EpollConnector, NULL);

		if (error != 0)
			Fatal(1, "pthread_create (connecting thread %u of %u in one process): %s", t + 1, nconnectors, strerror(error));
	}
	for (unsigned t = 0; t < nconnectors; t++)
		pthread_join(threads[t], NULL);
	free(threads);

	pthread_mutex_lock(&shared->lock);
	while (!shared->go)
	{
		struct timespec until;

		clock_gettime(CLOCK_REALTIME, &until);
		until.tv_sec += 1;
		(void) pthread_cond_timedwait(&shared->started, &shared->lock, &until);
		if (getppid() != parent)
			_exit(3);			/* the parent is gone: no run will start */
	}
	start = shared->start;
	epollMeasureFrom = shared->measureFrom;
	epollDeadline = shared->deadline;
	pthread_mutex_unlock(&shared->lock);

	/*
	 * Open loop: every client's schedule, staggered against the other
	 * clients', waits in the heap. Closed loop: all send at the start.
	 */
	if (interval == 0)
		SleepUntil(start);
	for (unsigned c = 0; c < count; c++)
	{
		unsigned	client = first + c; /* among all processes */

		epollClients[c].r = (uint32_t) ((uint64_t) client * nrequests / clients);
		epollClients[c].first = start + (uint64_t) (interval * client / clients);
		EpollNext(c);
	}
	scanned = Now();
	while (epollLive)
	{
		uint64_t	now;
		int			n;

		if (epollHeaped && epollClients[epollHeap[0]].origin != epollArmed)
		{
			struct itimerspec when;

			memset(&when, 0, sizeof(when));
			epollArmed = epollClients[epollHeap[0]].origin;
			when.it_value.tv_sec = (time_t) (epollArmed / SECOND);
			when.it_value.tv_nsec = (long) (epollArmed % SECOND);
			if (timerfd_settime(epollTimer, TFD_TIMER_ABSTIME, &when, NULL) != 0)
				Fatal(2, "timerfd_settime: %s", strerror(errno));
		}
		n = epoll_wait(epollSet, events, EPOLL_EVENTS, 1000);
		if (n < 0 && errno != EINTR)
			Fatal(2, "epoll_wait: %s", strerror(errno));
		for (int i = 0; i < n; i++)
		{
			EpollConnection *connection = events[i].data.ptr;

			if (connection == NULL)
			{
				uint64_t	expirations;

				(void) !read(epollTimer, &expirations, sizeof(expirations));
				epollArmed = 0;
				continue;
			}
			/* An earlier event of this round may have ended the client. */
			if (connection->fd >= 0 && (events[i].events & EPOLLOUT))
				EpollFlush(connection);
			if (connection->fd >= 0 && (events[i].events & ~(uint32_t) EPOLLOUT))
				EpollRead(connection);
		}

		/* The scheduled sends that are due. */
		now = Now();
		while (epollHeaped && epollClients[epollHeap[0]].origin <= now)
		{
			uint32_t	c = EpollPop();

			epollClients[c].sends++;
			EpollLaunch(c, now);
		}
		if (now - scanned >= SECOND)
		{
			scanned = now;
			for (unsigned c = 0; c < count; c++)
				if (epollClients[c].pending && now > epollClients[c].sent &&
					now - epollClients[c].sent >= REPLY_SECONDS * SECOND)
					Fatal(2, "no reply from %s:%s within %d s", host, port, REPLY_SECONDS);
		}
	}
	atomic_fetch_add(&shared->requests, epollRequests);
	atomic_fetch_add(&shared->subrequests, epollSubrequests);
	atomic_fetch_add(&shared->errors, epollErrors);
	atomic_fetch_add(&shared->hits, epollHits);
	atomic_fetch_add(&shared->notReady, epollNotReady);
	atomic_fetch_add(&shared->unsent, epollUnsent);
	for (unsigned i = 0; i < seconds; i++)
		if (epollSeries[i])
			atomic_fetch_add(&shared->counts[(size_t) (1 + nsubs) * BINS + i], epollSeries[i]);
	seen = atomic_load(&shared->maxNs);
	while (seen < epollMaxNs && !atomic_compare_exchange_weak(&shared->maxNs, &seen, epollMaxNs))
		;
	_exit(0);
}
#endif							/* __linux__ */

static void
StopChildren(void)
{
	for (unsigned i = 0; i < nchildren; i++)
		if (children[i] > 0)
			(void) kill(children[i], SIGTERM);
}

static void
Interrupted(int signal)
{
	StopChildren();
	_exit(128 + signal);
}

/* A child ended: fine when it exited with 0 after the run began, else the run is void. */
static void
Reaped(pid_t pid, int status, int running)
{
	int			code = WIFEXITED(status) ? WEXITSTATUS(status) : 1;

	for (unsigned i = 0; i < nchildren; i++)
		if (children[i] == pid)
			children[i] = 0;
	if (WIFEXITED(status) && code == 0 && running)
		return;
	StopChildren();
	while (waitpid(-1, NULL, 0) > 0 || errno == EINTR)
		;
	if (WIFSIGNALED(status))
		fprintf(stderr, "muse_direct: process %ld was killed by signal %d; no result\n", (long) pid, WTERMSIG(status));
	else
		fprintf(stderr, "muse_direct: process %ld ended with status %d; no result\n", (long) pid, code);
	exit(code ? code : 1);
}

static int
Usage(const char *program)
{
	fprintf(stderr, "usage: %s HOST PORT TOKEN_FILE FRAMES SECONDS CLIENTS [WARMUP] [--processes N] [--rate R] [--epoll]\n", program);
	return 1;
}

int
main(int argc, char **argv)
{
	unsigned	warmup = 3,
				processes = 1;
	double		rate = 0,
				connectSeconds;
	FILE	   *file;
	struct addrinfo hints = {0}, *result;
	struct rlimit limit;
	struct sigaction action;
	pthread_mutexattr_t mutexAttributes;
	pthread_condattr_t condAttributes;
	size_t		sharedSize;
	uint64_t	connectFrom,
				reported;
	int			option = 7,
				useEpoll = 0;

	if (argc < 7)
		return Usage(argv[0]);
	host = argv[1];
	port = argv[2];
	seconds = (unsigned) atoi(argv[5]);
	clients = (unsigned) atoi(argv[6]);
	if (argc > 7 && strncmp(argv[7], "--", 2) != 0)
		warmup = (unsigned) atoi(argv[option++]);
	while (option < argc)
	{
		if (strcmp(argv[option], "--epoll") == 0)
		{
			useEpoll = 1;
			option += 1;
			continue;
		}
		if (option + 1 < argc && strcmp(argv[option], "--processes") == 0)
			processes = (unsigned) atoi(argv[option + 1]);
		else if (option + 1 < argc && strcmp(argv[option], "--rate") == 0)
			rate = atof(argv[option + 1]);
		else
			return Usage(argv[0]);
		option += 2;
	}
#ifndef __linux__
	if (useEpoll)
		return fprintf(stderr, "muse_direct: --epoll needs Linux (epoll and timerfd); "
					   "run without it for a thread per client\n"), 1;
#endif
	if (seconds == 0 || clients == 0 || processes == 0 || processes > clients || rate < 0)
		return fprintf(stderr, "muse_direct: SECONDS, CLIENTS and --processes must be positive, "
					   "--processes at most CLIENTS, --rate not negative\n"), 1;
	if (rate > 0)
		interval = (double) clients * 1e9 / rate;
	if ((file = fopen(argv[3], "r")) == NULL || fgets(token, sizeof(token), file) == NULL)
		return fprintf(stderr, "cannot read %s\n", argv[3]), 1;
	fclose(file);
	tokenSize = strcspn(token, "\r\n");
	if ((file = fopen(argv[4], "rb")) == NULL)
		return fprintf(stderr, "cannot open %s\n", argv[4]), 1;
	{
		unsigned char head[8];

		if (fread(head, 1, 8, file) != 8)
			return fprintf(stderr, "short %s\n", argv[4]), 1;
		nrequests = Read32(head);
		nsubs = Read32(head + 4);
		if (nrequests == 0 || nsubs == 0 || nsubs > MAX_SUBS)
			return fprintf(stderr, "bad header in %s\n", argv[4]), 1;
	}
	frames = calloc((size_t) nrequests * nsubs, sizeof(*frames));
	frameSizes = calloc((size_t) nrequests * nsubs, sizeof(*frameSizes));
	if (frames == NULL || frameSizes == NULL)
		return fprintf(stderr, "out of memory\n"), 1;
	for (size_t i = 0; i < (size_t) nrequests * nsubs; i++)
	{
		unsigned char size[4];

		if (fread(size, 1, 4, file) != 4)
			return fprintf(stderr, "short %s\n", argv[4]), 1;
		frameSizes[i] = Read32(size);
		frames[i] = malloc(frameSizes[i] ? frameSizes[i] : 1);
		if (frames[i] == NULL || fread(frames[i], 1, frameSizes[i], file) != frameSizes[i])
			return fprintf(stderr, "short %s\n", argv[4]), 1;
	}
	fclose(file);

	/* Resolve once; every connection of every process uses the address. */
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	if (getaddrinfo(host, port, &hints, &result) != 0)
		return fprintf(stderr, "muse_direct: cannot resolve %s:%s\n", host, port), 1;
	memcpy(&server, result->ai_addr, result->ai_addrlen);
	serverSize = result->ai_addrlen;
	serverFamily = result->ai_family;
	freeaddrinfo(result);
	if (serverFamily == AF_INET &&
		(ntohl(((struct sockaddr_in *) &server)->sin_addr.s_addr) >> 24) == 127)
		sourceAddresses = 1 + (unsigned) ((uint64_t) clients * nsubs / PER_SOURCE);

	/* Each process needs its clients' connections and a few descriptors more. */
	{
		uint64_t	need = (uint64_t) ((clients + processes - 1) / processes) * nsubs + 64;

		if (getrlimit(RLIMIT_NOFILE, &limit) == 0 && limit.rlim_cur < limit.rlim_max)
		{
			struct rlimit raised = limit;

			raised.rlim_cur = limit.rlim_max;
			if (setrlimit(RLIMIT_NOFILE, &raised) != 0)
			{
				/*
				 * Some systems refuse "unlimited" here; ask for what is
				 * needed.
				 */
				raised.rlim_cur = (rlim_t) need;
				(void) setrlimit(RLIMIT_NOFILE, &raised);
			}
		}
		if (getrlimit(RLIMIT_NOFILE, &limit) == 0 && limit.rlim_cur != RLIM_INFINITY && limit.rlim_cur < need)
			return fprintf(stderr, "muse_direct: %llu descriptors needed per process, the limit is %llu: "
						   "use more --processes or raise ulimit -n\n",
						   (unsigned long long) need, (unsigned long long) limit.rlim_cur), 1;
	}

	sharedSize = sizeof(Shared) + ((size_t) (1 + nsubs) * BINS + seconds) * sizeof(atomic_uint_fast64_t);
	shared = mmap(NULL, sharedSize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	if (shared == MAP_FAILED)
		return fprintf(stderr, "muse_direct: cannot map %zu shared bytes: %s\n", sharedSize, strerror(errno)), 1;
	if (pthread_mutexattr_init(&mutexAttributes) != 0 ||
		pthread_mutexattr_setpshared(&mutexAttributes, PTHREAD_PROCESS_SHARED) != 0 ||
		pthread_mutex_init(&shared->lock, &mutexAttributes) != 0 ||
		pthread_condattr_init(&condAttributes) != 0 ||
		pthread_condattr_setpshared(&condAttributes, PTHREAD_PROCESS_SHARED) != 0 ||
		pthread_cond_init(&shared->slotFree, &condAttributes) != 0 ||
		pthread_cond_init(&shared->started, &condAttributes) != 0)
		return fprintf(stderr, "muse_direct: cannot create process-shared locks\n"), 1;
	shared->slots = MAX_CONNECTING;

	/* Fork before any thread exists. The parent only coordinates. */
	parent = getpid();
	children = calloc(processes, sizeof(*children));
	if (children == NULL)
		return fprintf(stderr, "out of memory\n"), 1;
	memset(&action, 0, sizeof(action));
	action.sa_handler = Interrupted;
	(void) sigaction(SIGINT, &action, NULL);
	(void) sigaction(SIGTERM, &action, NULL);
	(void) sigaction(SIGHUP, &action, NULL);
	connectFrom = reported = Now();
	for (unsigned p = 0, first = 0; p < processes; p++)
	{
		unsigned	count = clients / processes + (p < clients % processes);
		pid_t		pid = fork();

		if (pid == 0)
		{
			action.sa_handler = SIG_DFL;
			(void) sigaction(SIGINT, &action, NULL);
			(void) sigaction(SIGTERM, &action, NULL);
			(void) sigaction(SIGHUP, &action, NULL);
#ifdef __linux__
			if (useEpoll)
				EpollProcess(first, count);
#endif
			Process(first, count);
		}
		if (pid < 0)
		{
			fprintf(stderr, "muse_direct: fork: %s\n", strerror(errno));
			StopChildren();
			return 1;
		}
		children[nchildren++] = pid;
		first += count;
	}

	/*
	 * The barrier: nothing is sent until every client everywhere is
	 * connected.
	 */
	while (atomic_load(&shared->connected) < clients)
	{
		int			status;
		pid_t		pid = waitpid(-1, &status, WNOHANG);

		if (pid > 0)
			Reaped(pid, status, 0);
		Sleep(1000000);
		if (Now() - reported >= 5 * SECOND)
		{
			reported = Now();
			fprintf(stderr, "muse_direct: %u of %u clients connected after %.0f s (%llu retries)\n",
					atomic_load(&shared->connected), clients, (double) (reported - connectFrom) / 1e9,
					(unsigned long long) atomic_load(&shared->retries));
		}
	}
	connectSeconds = (double) (Now() - connectFrom) / 1e9;
	pthread_mutex_lock(&shared->lock);
	/* Far enough ahead for every thread to have woken up by then. */
	shared->start = Now() + SECOND / 5;
	shared->measureFrom = shared->start + (uint64_t) warmup * SECOND;
	shared->deadline = shared->measureFrom + (uint64_t) seconds * SECOND;
	shared->go = 1;
	pthread_cond_broadcast(&shared->started);
	pthread_mutex_unlock(&shared->lock);
	for (unsigned left = nchildren; left;)
	{
		int			status;
		pid_t		pid = waitpid(-1, &status, 0);

		if (pid < 0 && errno == EINTR)
			continue;
		if (pid < 0)
			break;
		Reaped(pid, status, 1);
		left--;
	}

	{
		const atomic_uint_fast64_t *series = &shared->counts[(size_t) (1 + nsubs) * BINS];
		uint64_t	n = atomic_load(&shared->requests),
					subrequests = atomic_load(&shared->subrequests);

		printf("{\"clients\": %u, \"seconds\": %u, \"requests\": %llu, \"requests_per_s\": %.1f, "
			   "\"subrequests_per_s\": %.1f, \"hits_per_subrequest\": %.1f, \"errors\": %llu, \"not_ready\": %llu, "
			   "\"request_ms\": {\"50\": %.2f, \"95\": %.2f, \"99\": %.2f, \"99.9\": %.2f}, \"max_ms\": %.2f, "
			   "\"processes\": %u, \"mode\": \"%s\", \"connections\": %llu, \"connect_seconds\": %.3f, \"connect_retries\": %llu, ",
			   clients, seconds, (unsigned long long) n, (double) n / seconds,
			   (double) subrequests / seconds,
			   subrequests ? (double) atomic_load(&shared->hits) / (double) subrequests : 0.0,
			   (unsigned long long) atomic_load(&shared->errors), (unsigned long long) atomic_load(&shared->notReady),
			   Percentile(shared->counts, n, 0.50), Percentile(shared->counts, n, 0.95),
			   Percentile(shared->counts, n, 0.99), Percentile(shared->counts, n, 0.999),
			   (double) atomic_load(&shared->maxNs) / 1e6,
			   processes, useEpoll ? "epoll" : "threads", (unsigned long long) clients * nsubs, connectSeconds,
			   (unsigned long long) atomic_load(&shared->retries));
		if (rate > 0)
			printf("\"target_rate\": %.1f, \"achieved_rate\": %.1f, \"unsent\": %llu, ",
				   rate, (double) n / seconds, (unsigned long long) atomic_load(&shared->unsent));
		printf("\"subrequest_ms\": [");
		for (uint32_t s = 0; s < nsubs; s++)
			printf("%s{\"50\": %.2f, \"99\": %.2f}", s ? ", " : "",
				   Percentile(&shared->counts[(size_t) (1 + s) * BINS], n, 0.50),
				   Percentile(&shared->counts[(size_t) (1 + s) * BINS], n, 0.99));
		printf("], \"requests_per_s_series\": [");
		for (unsigned i = 0; i < seconds; i++)
			printf("%s%llu", i ? ", " : "", (unsigned long long) atomic_load(&series[i]));
		printf("]}\n");
	}
	return 0;
}
