/* Native operation-22 client: SOCKET REQUEST_FILE SECONDS CLIENTS K. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define BINS 100001				/* 10 us bins, final bin includes latencies >=
								 * 1 s */
typedef struct Client
{
	uint64_t	count;
	uint64_t	errors;
	uint32_t	seed;
} Client;

static const char *socketPath;
static unsigned char *requests;
static uint32_t nrequests,
			requestSize,
			wanted;
static uint64_t deadline;
static atomic_uint ready;
static atomic_int go;
static atomic_uint_fast64_t histogram[BINS];

static uint64_t
Now(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return (uint64_t) t.tv_sec * UINT64_C(1000000000) + t.tv_nsec;
}

static uint32_t
Read32(const unsigned char *p)
{
	return (uint32_t) p[0] | (uint32_t) p[1] << 8 | (uint32_t) p[2] << 16 | (uint32_t) p[3] << 24;
}

static int
Transfer(int fd, unsigned char *data, size_t size, int sending)
{
	while (size)
	{
		ssize_t		n = sending ? send(fd, data, size, 0) : recv(fd, data, size, 0);

		if (n < 0 && errno == EINTR)
			continue;
		if (n <= 0)
			return 0;
		data += n;
		size -= (size_t) n;
	}
	return 1;
}

static void *
Run(void *pointer)
{
	Client	   *client = pointer;
	struct sockaddr_un address = {0};
	struct timeval timeout = {30, 0};
	struct timespec pause = {0, 100000};
	unsigned char header[32];
	size_t		capacity = 16 + (size_t) wanted * 12;
	unsigned char *reply = malloc(capacity);
	int			fd = socket(AF_UNIX, SOCK_STREAM, 0);

	address.sun_family = AF_UNIX;
	strcpy(address.sun_path, socketPath);
	if (fd < 0 || reply == NULL || connect(fd, (struct sockaddr *) &address, sizeof(address)) != 0)
	{
		client->errors++;
		atomic_fetch_add(&ready, 1);
		goto done;
	}
	setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
	setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
	atomic_fetch_add(&ready, 1);
	while (!atomic_load(&go))
		nanosleep(&pause, NULL);
	while (Now() < deadline)
	{
		uint64_t	started = Now();
		uint64_t	finished;
		uint32_t	length;
		unsigned char *request;

		client->seed = client->seed * UINT32_C(1664525) + UINT32_C(1013904223);
		request = requests + (size_t) (client->seed % nrequests) * requestSize;
		if (!Transfer(fd, request, requestSize, 1) || !Transfer(fd, header, sizeof(header), 0))
		{
			client->errors++;
			break;
		}
		length = Read32(header + 24);
		if (memcmp(header, "PVAN", 4) != 0 || Read32(header + 28) != 0 ||
			length != capacity || !Transfer(fd, reply, length, 0) || Read32(reply) != wanted)
		{
			client->errors++;
			break;
		}
		finished = Now();
		if (finished > deadline)
			break;
		client->count++;
		if (client->count % 20 == 0)
		{
			uint64_t	bin = (finished - started) / 10000;

			atomic_fetch_add(&histogram[bin < BINS ? bin : BINS - 1], 1);
		}
	}
done:
	if (fd >= 0)
		close(fd);
	free(reply);
	return NULL;
}

static double
Percentile(unsigned percent, uint64_t samples)
{
	uint64_t	count = 0,
				target = (samples * percent + 99) / 100;

	for (unsigned i = 0; i < BINS; i++)
	{
		count += atomic_load(&histogram[i]);
		if (count >= target)
			return i / 100.0;
	}
	return 0;
}

int
main(int argc, char **argv)
{
	unsigned	seconds,
				clients;
	unsigned char header[8];
	uint64_t	count = 0,
				errors = 0,
				samples = 0;
	FILE	   *file;
	Client	   *state;
	pthread_t  *threads;
	struct timespec pause = {0, 1000000};
	uint64_t	setupDeadline;

	if (argc != 6)
		return fprintf(stderr, "usage: %s SOCKET REQUEST_FILE SECONDS CLIENTS K\n", argv[0]), 1;
	socketPath = argv[1];
	seconds = (unsigned) atoi(argv[3]);
	clients = (unsigned) atoi(argv[4]);
	wanted = (uint32_t) atoi(argv[5]);
	if (!seconds || seconds > 3600 || !clients || clients > 2048 || !wanted || wanted > 10000 ||
		strlen(socketPath) >= sizeof(((struct sockaddr_un *) 0)->sun_path))
		return 1;
	file = fopen(argv[2], "rb");
	if (!file || fread(header, sizeof(header), 1, file) != 1)
		return 1;
	nrequests = Read32(header);
	requestSize = Read32(header + 4);
	if (!nrequests || !requestSize || requestSize > 65536 || nrequests > SIZE_MAX / requestSize)
		return 1;
	requests = malloc((size_t) nrequests * requestSize);
	if (!requests || fread(requests, requestSize, nrequests, file) != nrequests)
		return 1;
	fclose(file);
	state = calloc(clients, sizeof(*state));
	threads = calloc(clients, sizeof(*threads));
	if (!state || !threads)
		return 1;
	signal(SIGPIPE, SIG_IGN);
	for (unsigned i = 0; i < clients; i++)
	{
		state[i].seed = i + 1;
		if (pthread_create(&threads[i], NULL, Run, &state[i]) != 0)
			return 1;
	}
	setupDeadline = Now() + UINT64_C(30000000000);
	while (atomic_load(&ready) < clients)
	{
		if (Now() > setupDeadline)
			return fprintf(stderr, "connection startup timed out\n"), 1;
		nanosleep(&pause, NULL);
	}
	deadline = Now() + (uint64_t) seconds * UINT64_C(1000000000);
	atomic_store(&go, 1);
	for (unsigned i = 0; i < clients; i++)
	{
		pthread_join(threads[i], NULL);
		count += state[i].count;
		errors += state[i].errors;
	}
	for (unsigned i = 0; i < BINS; i++)
		samples += atomic_load(&histogram[i]);
	printf("{\"clients\":%u,\"seconds\":%u,\"transactions\":%" PRIu64 ",\"qps\":%.3f,"
		   "\"errors\":%" PRIu64 ",\"latency_samples\":%" PRIu64 ",\"p50_ms\":%.2f,\"p95_ms\":%.2f,\"p99_ms\":%.2f}\n",
		   clients, seconds, count, (double) count / seconds, errors, samples,
		   Percentile(50, samples), Percentile(95, samples), Percentile(99, samples));
	free(threads);
	free(state);
	free(requests);
	return errors != 0;
}
