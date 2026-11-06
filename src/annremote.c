#include "postgres.h"

#include "ann.h"
#include "miscadmin.h"
#include "portability/instr_time.h"
#include "utils/guc.h"
#include "utils/memutils.h"

char	   *ann_worker_address = NULL;
char	   *ann_worker_auth_token = NULL;
int			ann_gpu_min_eligible = 1000;
int			ann_worker_heartbeat_ms = 5000;
int			ann_worker_retry_ms = 10000;
char	   *ann_worker_rdma = NULL;
int			ann_worker_rdma_spin_us = 50;
int			ann_worker_overload_wait_ms = 1000;
int			ann_worker_trace_sample = 0;

#if defined(WIN32) || defined(WORDS_BIGENDIAN)

bool
AnnRemoteConfigured(void)
{
	return false;
}

bool
AnnRemoteCall(uint16 operation, const char *payload, size_t length, int timeoutMs,
			  uint32 *status, char **reply, uint32 *replyLength)
{
	return false;
}

int
AnnRemoteBroadcast(uint16 operation, const char *payload, size_t length, int timeoutMs,
				   uint32 *status, char **reply, uint32 *replyLength)
{
	return -1;
}

int
AnnRemoteBroadcastEach(uint16 operation, const char *payload, size_t length, int timeoutMs,
					   int *failed, uint32 *status)
{
	*failed = 0;
	return 0;
}

int
AnnRemoteBroadcastTo(const char *nodes, uint16 operation, const char *payload, size_t length, int timeoutMs,
					 uint32 *status, char **reply, uint32 *replyLength)
{
	return -1;
}

int
AnnRemoteFanout(AnnRemoteRequest *requests, int count, int timeoutMs)
{
	return AnnRemoteFanoutWhile(requests, count, timeoutMs, NULL, NULL);
}

int
AnnRemoteFanoutWhile(AnnRemoteRequest *requests, int count, int timeoutMs, void (*whileWaiting) (void *),
					 void *arg)
{
	for (int i = 0; i < count; i++)
	{
		requests[i].ok = false;
		requests[i].done = true;
	}
	if (whileWaiting)
		whileWaiting(arg);
	return 0;
}

#else

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#define ANN_HEADER_SIZE 32
#define ANN_OP_PROBE 5
#define ANN_OP_AUTH 13
#define ANN_OP_INDEX_SEARCH 12
#define ANN_OP_INDEX_SEARCH_PACKED 14
#define ANN_OP_RDMA_HELLO 16
#define ANN_OP_INDEX_SEARCH_MATCH 17
#define ANN_OP_INDEX_SEARCH_BITMAP 19
#define ANN_OP_INDEX_SEARCH_CACHED 21
#define ANN_MAX_NODES 64

/*
 * One connection per backend to one GPU node, chosen from ann.worker_address
 * by process ID so backends spread across nodes. A failed node is skipped
 * and the next one tried. Requests are synchronous: one outstanding at most.
 */
/* Set around calls for indexes exported from a frozen index; see ann.h. */
uint64		ann_request_generation = 0;

static int	remoteFd = -1;
static int	remoteNode = -1;
static uint64 remoteRequest = 1;
static bool remoteBusy = false; /* an error interrupted an exchange */
static instr_time remoteUsed;	/* last successful exchange */
static instr_time downSince[ANN_MAX_NODES]; /* zero when believed healthy */

static bool Exchange(int fd, uint16 operation, const char *payload, size_t length, int timeoutMs,
					 uint32 *status, char **reply, uint32 *replyLength);
static double ElapsedMs(instr_time since);
static void PutHeader(char *header, uint16 operation, uint64 request, uint32 length);
static void PutHeaderGeneration(char *header, uint16 operation, uint64 request, uint32 length,
								uint64 generation);
static bool RecvAll(int fd, char *data, size_t size, instr_time start, int timeoutMs);

#ifdef ANN_USE_RDMA

#include <infiniband/verbs.h>

#define ANN_RDMA_ADDRESS 32

/*
 * Optional RDMA path to the connected node over InfiniBand/RoCE verbs: one
 * reliable connected queue pair per backend, connected through the HELLO
 * exchange on the authenticated TCP connection. A backend has one request
 * outstanding, so one receive and one send buffer suffice, registered once.
 * Completions are polled from user space, so a search costs no system call
 * or interrupt. TCP stays open for liveness, loads, and oversized frames.
 */
typedef struct RdmaClient
{
	struct ibv_context *context;
	struct ibv_pd *pd;
	struct ibv_comp_channel *channel;	/* wakes a backend that stopped
										 * spinning */
	struct ibv_cq *cq;
	struct ibv_qp *qp;
	struct ibv_mr *mr;
	char	   *region;			/* receive buffer, then send buffer */
	size_t		frame;
	bool		sending;		/* send completion not yet reaped */
	bool		shared;			/* context and pd belong to rdmaDevice */
} RdmaClient;

static RdmaClient *rdmaClient = NULL;

/*
 * The device context and protection domain, opened once per backend and
 * shared by its queue pairs: opening a device takes tens of milliseconds,
 * which a backend connecting to several nodes would otherwise pay per node.
 */
static struct
{
	char		name[64];
	struct ibv_context *context;
	struct ibv_pd *pd;
}			rdmaDevice;

static void
RdmaFree(RdmaClient *client)
{
	if (client == NULL)
		return;
	if (client->qp)
		(void) ibv_destroy_qp(client->qp);
	if (client->mr)
		(void) ibv_dereg_mr(client->mr);
	if (client->cq)
		(void) ibv_destroy_cq(client->cq);
	if (client->channel)
		(void) ibv_destroy_comp_channel(client->channel);
	if (client->pd && !client->shared)
		(void) ibv_dealloc_pd(client->pd);
	if (client->context && !client->shared)
		(void) ibv_close_device(client->context);
	free(client->region);
	free(client);
}

static void
CloseRdma(void)
{
	RdmaClient *client = rdmaClient;

	rdmaClient = NULL;
	RdmaFree(client);
}

static bool
PostRdmaReceive(RdmaClient *client)
{
	struct ibv_sge sge = {(uintptr_t) client->region, (uint32_t) client->frame, client->mr->lkey};
	struct ibv_recv_wr wr = {0};
	struct ibv_recv_wr *bad;

	wr.wr_id = 1;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	return ibv_post_recv(client->qp, &wr, &bad) == 0;
}

/*
 * Open the device, connect a queue pair with the node over the
 * authenticated TCP connection fd, and post the receive buffer. Returns NULL
 * on any failure, which leaves that connection on TCP only.
 */
static RdmaClient *
RdmaConnect(int fd, int timeoutMs)
{
	RdmaClient *client = calloc(1, sizeof(RdmaClient));
	struct ibv_device **list = NULL;
	struct ibv_port_attr port;
	struct ibv_qp_init_attr init = {0};
	struct ibv_qp_attr attr = {0};
	union ibv_gid gid;
	unsigned char address[ANN_RDMA_ADDRESS] = {0};
	char		device[64];
	const char *colon = strchr(ann_worker_rdma, ':');
	uint8		portNumber = colon ? (uint8) atoi(colon + 1) : 1;
	uint32		status;
	char	   *reply;
	uint32		replyLength;
	uint32		workerFrame;
	uint16		lid,
				mtu;
	uint32		qpn,
				psn,
				localPsn = (uint32) random() & 0xffffff;
	int			devices = 0;
	const char *step = "find device";

	if (client == NULL)
		goto fail;
	snprintf(device, sizeof(device), "%.*s",
			 (int) (colon ? (size_t) (colon - ann_worker_rdma) : strlen(ann_worker_rdma)), ann_worker_rdma);
	if (rdmaDevice.context && strcmp(rdmaDevice.name, device) == 0)
	{
		client->context = rdmaDevice.context;
		client->pd = rdmaDevice.pd;
		client->shared = true;
	}
	else
	{
		list = ibv_get_device_list(&devices);
		for (int i = 0; list && i < devices && client->context == NULL; i++)
			if (strcmp(ibv_get_device_name(list[i]), device) == 0)
				client->context = ibv_open_device(list[i]);
		if (list)
			ibv_free_device_list(list);
		if (client->context == NULL || (client->pd = ibv_alloc_pd(client->context)) == NULL)
			goto fail;
		if (rdmaDevice.context == NULL)
		{
			/* Keep it for the backend's later connections. */
			strlcpy(rdmaDevice.name, device, sizeof(rdmaDevice.name));
			rdmaDevice.context = client->context;
			rdmaDevice.pd = client->pd;
			client->shared = true;
		}
	}
	step = "query port";
	if (ibv_query_port(client->context, portNumber, &port) != 0 || port.state != IBV_PORT_ACTIVE ||
		ibv_query_gid(client->context, portNumber, 0, &gid) != 0)
		goto fail;
	step = "create queue pair";
	init.cap.max_send_wr = 4;
	init.cap.max_recv_wr = 4;
	init.cap.max_send_sge = 1;
	init.cap.max_recv_sge = 1;
	init.qp_type = IBV_QPT_RC;
	if ((client->channel = ibv_create_comp_channel(client->context)) == NULL ||
		fcntl(client->channel->fd, F_SETFL, fcntl(client->channel->fd, F_GETFL) | O_NONBLOCK) != 0 ||
		(client->cq = ibv_create_cq(client->context, 16, NULL, client->channel, 0)) == NULL)
		goto fail;
	init.send_cq = client->cq;
	init.recv_cq = client->cq;
	if ((client->qp = ibv_create_qp(client->pd, &init)) == NULL)
		goto fail;
	attr.qp_state = IBV_QPS_INIT;
	attr.port_num = portNumber;
	if (ibv_modify_qp(client->qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0)
		goto fail;

	/* Exchange queue pair addresses over the authenticated connection. */
	step = "hello";
	lid = port.lid;
	mtu = (uint16) port.active_mtu;
	qpn = client->qp->qp_num;
	memcpy(address, &lid, 2);
	memcpy(address + 2, &mtu, 2);
	memcpy(address + 4, &qpn, 4);
	memcpy(address + 8, &localPsn, 4);
	memcpy(address + 12, gid.raw, 16);
	if (!Exchange(fd, ANN_OP_RDMA_HELLO, (const char *) address, sizeof(address), timeoutMs,
				  &status, &reply, &replyLength) || status != 0 || replyLength < 4 + ANN_RDMA_ADDRESS)
		goto fail;
	memcpy(&workerFrame, reply, 4);
	memcpy(&lid, reply + 4, 2);
	memcpy(&mtu, reply + 6, 2);
	memcpy(&qpn, reply + 8, 4);
	memcpy(&psn, reply + 12, 4);
	memcpy(gid.raw, reply + 16, 16);

	/* Buffers before RTR, so the receive is posted before the node can send. */
	step = "register buffers";
	client->frame = Min((size_t) workerFrame, (size_t) 256 * 1024);
	client->region = malloc(client->frame * 2);
	if (client->region == NULL ||
		(client->mr = ibv_reg_mr(client->pd, client->region, client->frame * 2, IBV_ACCESS_LOCAL_WRITE)) == NULL ||
		!PostRdmaReceive(client))
		goto fail;
	step = "connect queue pair";
	memset(&attr, 0, sizeof(attr));
	attr.qp_state = IBV_QPS_RTR;
	attr.path_mtu = (enum ibv_mtu) mtu;
	attr.dest_qp_num = qpn;
	attr.rq_psn = psn;
	attr.max_dest_rd_atomic = 1;
	attr.min_rnr_timer = 12;
	attr.ah_attr.dlid = lid;
	attr.ah_attr.port_num = portNumber;
	if (port.link_layer == IBV_LINK_LAYER_ETHERNET)
	{
		attr.ah_attr.is_global = 1;
		attr.ah_attr.grh.dgid = gid;
		attr.ah_attr.grh.hop_limit = 1;
	}
	if (ibv_modify_qp(client->qp, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
					  IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0)
		goto fail;
	memset(&attr, 0, sizeof(attr));
	attr.qp_state = IBV_QPS_RTS;
	attr.timeout = 14;
	attr.retry_cnt = 7;
	attr.rnr_retry = 7;
	attr.sq_psn = localPsn;
	attr.max_rd_atomic = 1;
	if (ibv_modify_qp(client->qp, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
					  IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0)
		goto fail;
	elog(DEBUG1, "ann worker RDMA on %s port %u ready, frames up to %zu bytes", device, portNumber, client->frame);
	return client;

fail:
	elog(DEBUG1, "ann worker RDMA on %s unavailable: %s failed", ann_worker_rdma, step);
	RdmaFree(client);
	return NULL;
}

/*
 * Only searches travel over RDMA; the worker answers other operations
 * (loads, index info, statistics) on the TCP connection alone.
 */
static bool
RdmaOperation(uint16 operation)
{
	return operation == ANN_OP_INDEX_SEARCH || operation == ANN_OP_INDEX_SEARCH_PACKED ||
		operation == ANN_OP_INDEX_SEARCH_MATCH || operation == ANN_OP_INDEX_SEARCH_BITMAP ||
		operation == ANN_OP_INDEX_SEARCH_CACHED || operation == ANN_OP_INDEX_SEARCH_PREDICATE ||
		operation == ANN_OP_COLLECTION_SEARCH || operation == ANN_OP_INDEX_SEARCH_BATCH;
}

/* Reap the previous send's completion before reusing the send buffer. */
static bool
ReapRdmaSend(RdmaClient *client)
{
	struct ibv_wc wc;
	int			n;

	while (client->sending && (n = ibv_poll_cq(client->cq, 1, &wc)) >= 0)
		if (n == 1)
		{
			if (wc.status != IBV_WC_SUCCESS || wc.wr_id != 2)
				return false;
			client->sending = false;
		}
	return !client->sending;
}

static bool
CopyReply(const char *frame, size_t size, uint64 request, uint32 *status, char **reply, uint32 *replyLength)
{
	uint64		replyRequest;
	uint32		length;

	if (size < ANN_HEADER_SIZE || memcmp(frame, "PVAN", 4) != 0)
		return false;
	memcpy(&replyRequest, frame + 8, 8);
	memcpy(&length, frame + 24, 4);
	memcpy(status, frame + 28, 4);
	if (replyRequest != request || length != size - ANN_HEADER_SIZE)
		return false;
	*reply = palloc(length + 1);
	memcpy(*reply, frame + ANN_HEADER_SIZE, length);
	(*reply)[length] = '\0';
	*replyLength = length;
	return true;
}

/*
 * One request over RDMA. Completions are spin-polled in user space;
 * interrupts, the timeout, and the TCP socket are checked periodically,
 * because the node answers over TCP when a reply exceeds the RDMA frame.
 */
static bool
RdmaExchange(RdmaClient *client, int fd, uint16 operation, const char *payload, size_t length, int timeoutMs,
			 uint32 *status, char **reply, uint32 *replyLength)
{
	char	   *buffer = client->region + client->frame;
	uint64		request = remoteRequest++;
	struct ibv_sge sge;
	struct ibv_send_wr wr = {0};
	struct ibv_send_wr *bad;
	instr_time	start;
	instr_time	spinStart;
	bool		armed = false;

	INSTR_TIME_SET_CURRENT(start);
	spinStart = start;
	if (!ReapRdmaSend(client))
		return false;
	PutHeader(buffer, operation, request, (uint32) length);
	if (length)
		memcpy(buffer + ANN_HEADER_SIZE, payload, length);
	sge.addr = (uintptr_t) buffer;
	sge.length = (uint32) (ANN_HEADER_SIZE + length);
	sge.lkey = client->mr->lkey;
	wr.wr_id = 2;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.opcode = IBV_WR_SEND;
	wr.send_flags = IBV_SEND_SIGNALED;
	if (ibv_post_send(client->qp, &wr, &bad) != 0)
		return false;
	client->sending = true;
	for (;;)
	{
		struct ibv_wc wc;
		int			n = ibv_poll_cq(client->cq, 1, &wc);

		if (n < 0)
			return false;
		if (n == 1)
		{
			if (wc.status != IBV_WC_SUCCESS)
				return false;
			if (wc.wr_id == 2)
			{
				client->sending = false;
				continue;
			}
			if (!CopyReply(client->region, wc.byte_len, request, status, reply, replyLength) ||
				!PostRdmaReceive(client))
				return false;
			return ReapRdmaSend(client);
		}

		/*
		 * Spin briefly for latency, then sleep: hundreds of backends spinning
		 * on their own completion queues would starve the CPUs serving them.
		 * Arming the queue makes the adapter signal the channel; a completion
		 * that raced the arm is caught by polling the queue again first.
		 */
		if (!armed && ElapsedMs(spinStart) * 1000.0 < ann_worker_rdma_spin_us)
			continue;
		if (!armed)
		{
			if (ibv_req_notify_cq(client->cq, 0) != 0)
				return false;
			armed = true;
			continue;
		}
		{
			struct pollfd pollers[2] = {{client->channel->fd, POLLIN, 0}, {fd, POLLIN, 0}};

			CHECK_FOR_INTERRUPTS();
			if (ElapsedMs(start) > timeoutMs)
				return false;
			if (poll(pollers, 2, 10) < 0 && errno != EINTR)
				return false;
			if (pollers[0].revents & POLLIN)
			{
				struct ibv_cq *eventCq;
				void	   *ignored;

				if (ibv_get_cq_event(client->channel, &eventCq, &ignored) == 0)
					ibv_ack_cq_events(eventCq, 1);
				armed = false;
				INSTR_TIME_SET_CURRENT(spinStart);
				continue;
			}

			/* A reply that did not fit an RDMA send arrives on TCP instead. */
			if (pollers[1].revents & POLLIN)
			{
				char		header[ANN_HEADER_SIZE];
				uint64		replyRequest;
				uint32		size;

				if (!RecvAll(fd, header, sizeof(header), start, timeoutMs) ||
					memcmp(header, "PVAN", 4) != 0)
					return false;
				memcpy(&replyRequest, header + 8, 8);
				memcpy(&size, header + 24, 4);
				memcpy(status, header + 28, 4);
				if (replyRequest != request || size > 64 * 1024 * 1024)
					return false;
				*reply = palloc(size + 1);
				if (!RecvAll(fd, *reply, size, start, timeoutMs))
					return false;
				(*reply)[size] = '\0';
				*replyLength = size;
				return ReapRdmaSend(client);
			}
		}
	}
}

/*
 * Connect a queue pair and send one probe over it: a port the fabric does not
 * route accepts the connection but never delivers a frame, so without the
 * probe every request would wait for its timeout. A connection whose probe
 * goes unanswered stays on TCP.
 */
static RdmaClient *
RdmaConnectChecked(int fd, int timeoutMs)
{
	RdmaClient *client = RdmaConnect(fd, timeoutMs);
	uint32		status;
	char	   *reply = NULL;
	uint32		length;

	if (client && !RdmaExchange(client, fd, ANN_OP_PROBE, NULL, 0, Min(timeoutMs, 1000), &status, &reply,
								&length))
	{
		elog(DEBUG1, "ann worker RDMA on %s did not answer a probe; using TCP", ann_worker_rdma);
		RdmaFree(client);
		client = NULL;
	}
	return client;
}
#endif

static void
CloseRemote(void)
{
#ifdef ANN_USE_RDMA
	CloseRdma();
#endif
	if (remoteFd >= 0)
		close(remoteFd);
	remoteFd = -1;
}

static double
ElapsedMs(instr_time since)
{
	instr_time	now;

	INSTR_TIME_SET_CURRENT(now);
	INSTR_TIME_SUBTRACT(now, since);
	return INSTR_TIME_GET_MILLISEC(now);
}

/* A node that failed is skipped until ann.worker_retry_ms has passed. */
static void
MarkDown(int node)
{
	INSTR_TIME_SET_CURRENT(downSince[node]);
}

static bool
NodeAvailable(int node)
{
	if (INSTR_TIME_IS_ZERO(downSince[node]))
		return true;
	if (ElapsedMs(downSince[node]) < ann_worker_retry_ms)
		return false;
	INSTR_TIME_SET_ZERO(downSince[node]);
	return true;
}

/*
 * Kernel keepalives detect a node that vanished mid-request (host down or
 * partitioned), which a blocked receive would otherwise only notice at the
 * request timeout.
 */
static void
SetKeepalive(int fd)
{
	int			on = 1;
	int			idle = Max(1, ann_worker_heartbeat_ms / 1000);
	int			interval = Max(1, idle / 2);
	int			count = 3;

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
}

bool
AnnRemoteConfigured(void)
{
	return ann_worker_address != NULL && ann_worker_address[0] != '\0';
}

/* Split ann.worker_address into host and port strings; returns the count. */
static int
ParseNodes(const char *nodes, char **hosts, char **ports)
{
	char	   *list = pstrdup(nodes);
	char	   *token;
	char	   *save = NULL;
	int			count = 0;

	for (token = strtok_r(list, ", ", &save); token && count < ANN_MAX_NODES;
		 token = strtok_r(NULL, ", ", &save))
	{
		char	   *colon = strrchr(token, ':');

		if (colon == NULL || colon == token)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("invalid ann node address \"%s\"", token),
					 errhint("Use a comma-separated list of host:port.")));
		*colon = '\0';
		hosts[count] = token;
		ports[count] = colon + 1;
		count++;
	}
	return count;
}

static bool
WaitFor(int fd, short events, instr_time start, int timeoutMs)
{
	for (;;)
	{
		struct pollfd poller = {fd, events, 0};
		instr_time	now;
		int			result;

		CHECK_FOR_INTERRUPTS();
		INSTR_TIME_SET_CURRENT(now);
		INSTR_TIME_SUBTRACT(now, start);
		if (INSTR_TIME_GET_MILLISEC(now) > timeoutMs)
			return false;
		result = poll(&poller, 1, 100);
		if (result > 0)
			return (poller.revents & (POLLERR | POLLNVAL)) == 0 || (poller.revents & events);
		if (result < 0 && errno != EINTR)
			return false;
	}
}

static bool
SendAll(int fd, const char *data, size_t size, instr_time start, int timeoutMs)
{
	while (size > 0)
	{
#ifdef MSG_NOSIGNAL
		ssize_t		n = send(fd, data, size, MSG_NOSIGNAL);
#else
		ssize_t		n = send(fd, data, size, 0);
#endif

		if (n > 0)
		{
			data += n;
			size -= n;
		}
		else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR))
		{
			if (!WaitFor(fd, POLLOUT, start, timeoutMs))
				return false;
		}
		else
			return false;
	}
	return true;
}

static bool
RecvAll(int fd, char *data, size_t size, instr_time start, int timeoutMs)
{
	while (size > 0)
	{
		ssize_t		n = recv(fd, data, size, 0);

		if (n > 0)
		{
			data += n;
			size -= n;
		}
		else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR))
		{
			if (!WaitFor(fd, POLLIN, start, timeoutMs))
				return false;
		}
		else
			return false;
	}
	return true;
}

static void
PutHeaderGeneration(char *header, uint16 operation, uint64 request, uint32 length, uint64 generation)
{
	uint16		version = 1;
	uint32		status = 0;

	memcpy(header, "PVAN", 4);
	memcpy(header + 4, &version, 2);
	memcpy(header + 6, &operation, 2);
	memcpy(header + 8, &request, 8);
	memcpy(header + 16, &generation, 8);
	memcpy(header + 24, &length, 4);
	memcpy(header + 28, &status, 4);
}

static void
PutHeader(char *header, uint16 operation, uint64 request, uint32 length)
{
	PutHeaderGeneration(header, operation, request, length, ann_request_generation);
}

/*
 * Send one frame and read its reply on fd. The reply payload is palloc'd in
 * the current context and NUL-terminated.
 */
static bool
Exchange(int fd, uint16 operation, const char *payload, size_t length, int timeoutMs,
		 uint32 *status, char **reply, uint32 *replyLength)
{
	char		header[ANN_HEADER_SIZE];
	uint64		request = remoteRequest++;
	uint64		replyRequest;
	uint32		size;
	instr_time	start;

	INSTR_TIME_SET_CURRENT(start);
	PutHeader(header, operation, request, (uint32) length);
	if (!SendAll(fd, header, sizeof(header), start, timeoutMs) ||
		(length && !SendAll(fd, payload, length, start, timeoutMs)) ||
		!RecvAll(fd, header, sizeof(header), start, timeoutMs) ||
		memcmp(header, "PVAN", 4) != 0)
		return false;
	memcpy(&replyRequest, header + 8, 8);
	memcpy(&size, header + 24, 4);
	memcpy(status, header + 28, 4);
	if (replyRequest != request || size > 64 * 1024 * 1024)
		return false;
	*reply = palloc(size + 1);
	if (!RecvAll(fd, *reply, size, start, timeoutMs))
		return false;
	(*reply)[size] = '\0';
	*replyLength = size;
	return true;
}

/* Connect and authenticate to one node; returns the socket or -1. */
static int
ConnectNode(const char *host, const char *port, int timeoutMs)
{
	struct addrinfo hints = {0};
	struct addrinfo *addresses;
	int			fd = -1;
	instr_time	start;

	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	if (getaddrinfo(host, port, &hints, &addresses) != 0)
		return -1;
	INSTR_TIME_SET_CURRENT(start);
	for (struct addrinfo *a = addresses; a && fd < 0; a = a->ai_next)
	{
		int			on = 1;
		int			error = 0;
		socklen_t	errorLength = sizeof(error);

		fd = socket(a->ai_family, SOCK_STREAM, 0);
		if (fd < 0)
			continue;
		if (fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK) != 0 ||
			(connect(fd, a->ai_addr, a->ai_addrlen) != 0 &&
			 (errno != EINPROGRESS || !WaitFor(fd, POLLOUT, start, timeoutMs) ||
			  getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &errorLength) != 0 || error != 0)))
		{
			close(fd);
			fd = -1;
			continue;
		}
		(void) setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on));
		if (ann_worker_heartbeat_ms > 0)
			SetKeepalive(fd);
#ifdef SO_NOSIGPIPE
		(void) setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof(on));
#endif
	}
	freeaddrinfo(addresses);
	if (fd >= 0)
	{
		const char *token = ann_worker_auth_token ? ann_worker_auth_token : "";
		uint32		status;
		char	   *reply;
		uint32		replyLength;

		if (!Exchange(fd, ANN_OP_AUTH, token, strlen(token), timeoutMs, &status, &reply, &replyLength) ||
			status != 0)
		{
			elog(DEBUG1, "ann worker %s:%s rejected authentication", host, port);
			close(fd);
			fd = -1;
		}
	}
	return fd;
}

/*
 * Synchronous request to this backend's node. A transport failure moves to
 * the next node and retries once there; requests sent this way are
 * idempotent searches and lookups.
 */
bool
AnnRemoteCall(uint16 operation, const char *payload, size_t length, int timeoutMs,
			  uint32 *status, char **reply, uint32 *replyLength)
{
	char	   *hosts[ANN_MAX_NODES];
	char	   *ports[ANN_MAX_NODES];
	int			count = ParseNodes(ann_worker_address, hosts, ports);

	if (timeoutMs <= 0)
		timeoutMs = ann_worker_timeout_ms;
	if (count == 0)
		return false;

	/* A cancelled exchange may have left part of a reply unread. */
	if (remoteBusy)
		CloseRemote();

	/*
	 * Heartbeat: a connection idle longer than ann.worker_heartbeat_ms is
	 * probed first, so a node that died while idle costs one short probe
	 * rather than a full request timeout.
	 */
	if (remoteFd >= 0 && ann_worker_heartbeat_ms > 0 &&
		ElapsedMs(remoteUsed) > ann_worker_heartbeat_ms)
	{
		uint32		probeStatus;
		char	   *probeReply;
		uint32		probeLength;

		remoteBusy = true;
		if (!Exchange(remoteFd, ANN_OP_PROBE, NULL, 0, ann_worker_heartbeat_ms,
					  &probeStatus, &probeReply, &probeLength))
		{
			elog(DEBUG1, "ann worker %s:%s missed a heartbeat", hosts[remoteNode], ports[remoteNode]);
			MarkDown(remoteNode);
			CloseRemote();
		}
		remoteBusy = false;
	}
	for (int attempt = 0; attempt < count + 1; attempt++)
	{
		if (remoteFd < 0)
		{
			/*
			 * Prefer healthy nodes; try cooling-down ones only as a last
			 * resort.
			 */
			for (int pass = 0; pass < 2 && remoteFd < 0; pass++)
				for (int i = 0; i < count && remoteFd < 0; i++)
				{
					int			node = (remoteNode < 0 ? MyProcPid + i : remoteNode + 1 + i) % count;

					if (pass == 0 && !NodeAvailable(node))
						continue;
					remoteFd = ConnectNode(hosts[node], ports[node], Min(timeoutMs, Max(ann_worker_heartbeat_ms, 1000)));
					if (remoteFd >= 0)
					{
						remoteNode = node;
#ifdef ANN_USE_RDMA
						if (ann_worker_rdma && ann_worker_rdma[0] != '\0')
							rdmaClient = RdmaConnectChecked(remoteFd, timeoutMs);
#endif
					}
					else
						MarkDown(node);
				}
			if (remoteFd < 0)
				return false;
		}
		remoteBusy = true;
		if (
#ifdef ANN_USE_RDMA
			(rdmaClient && RdmaOperation(operation) && ANN_HEADER_SIZE + length <= rdmaClient->frame) ?
			RdmaExchange(rdmaClient, remoteFd, operation, payload, length, timeoutMs, status, reply, replyLength) :
#endif
			Exchange(remoteFd, operation, payload, length, timeoutMs, status, reply, replyLength))
		{
			remoteBusy = false;
			INSTR_TIME_SET_CURRENT(remoteUsed);
			return true;
		}
		remoteBusy = false;
		elog(DEBUG1, "ann worker %s:%s failed; trying the next node", hosts[remoteNode], ports[remoteNode]);
		MarkDown(remoteNode);
		CloseRemote();
	}
	return false;
}

/*
 * Send one request to every node, for loads. Returns the number of nodes on
 * success, or -1 after the first failure; *status holds that node's status.
 */
int
AnnRemoteBroadcast(uint16 operation, const char *payload, size_t length, int timeoutMs,
				   uint32 *status, char **reply, uint32 *replyLength)
{
	return AnnRemoteBroadcastTo(ann_worker_address, operation, payload, length, timeoutMs, status, reply,
								replyLength);
}

/* AnnRemoteBroadcast to the nodes of a host:port list instead. */
int
AnnRemoteBroadcastTo(const char *nodes, uint16 operation, const char *payload, size_t length, int timeoutMs,
					 uint32 *status, char **reply, uint32 *replyLength)
{
	char	   *hosts[ANN_MAX_NODES];
	char	   *ports[ANN_MAX_NODES];
	int			count = ParseNodes(nodes, hosts, ports);

	for (int i = 0; i < count; i++)
	{
		int			fd = ConnectNode(hosts[i], ports[i], timeoutMs);
		bool		ok;

		if (fd < 0)
		{
			*status = UINT32_MAX;
			return -1;
		}
		ok = Exchange(fd, operation, payload, length, timeoutMs, status, reply, replyLength);
		close(fd);
		if (!ok)
			*status = UINT32_MAX;
		if (!ok || *status != 0)
			return -1;
	}
	return count;
}

/*
 * Fan-out for collections: several requests in flight at once, each to one
 * replica of its own node list. A backend keeps one connection per node
 * address it has used (apart from the single-node connection above), with
 * at most one request outstanding on each, over RDMA when ann.worker_rdma is
 * set. Replicas are taken in rotation, so a backend spreads its requests
 * over them; a node that fails is skipped for ann.worker_retry_ms and the
 * request moves to the next replica while time remains.
 */
#define ANN_FANOUT_NODES 64

typedef struct FanoutNode
{
	char	   *address;		/* host:port, malloc'd */
	int			fd;
#ifdef ANN_USE_RDMA
	RdmaClient *rdma;
	bool		armed;			/* completion channel armed */
#endif
	bool		busy;			/* a request is outstanding */
	bool		viaRdma;		/* it was sent over RDMA */
	bool		readable;		/* poll reported the TCP socket readable */
	int			owner;			/* the request it serves */
	uint64		request;
	instr_time	used;
	instr_time	down;			/* zero when believed healthy */
	uint64		traceSent;		/* ann.worker_trace_sample: send time */
	char		header[ANN_HEADER_SIZE];	/* TCP reply being read */
	size_t		got;
	char	   *body;
} FanoutNode;

static FanoutNode fanoutNodes[ANN_FANOUT_NODES];
static int	fanoutCount = 0;
static uint64 fanoutRotation = 0;
static FILE *traceFile = NULL;

/* Realtime nanoseconds, comparable with a worker's --trace on this host. */
static uint64
TraceNow(void)
{
	struct timespec ts;

	(void) clock_gettime(CLOCK_REALTIME, &ts);
	return (uint64) ts.tv_sec * UINT64CONST(1000000000) + (uint64) ts.tv_nsec;
}

/*
 * ann.worker_trace_sample: "C pid id node rdma fanout sent replied" per
 * sampled request (ids that are multiples of the sample, as the worker
 * picks them), in /tmp/ann-fanout-PID.trace.
 */
static void
TraceReply(FanoutNode *node, uint64 fanoutStart)
{
	if (ann_worker_trace_sample <= 0 || node->request % (uint64) ann_worker_trace_sample != 0)
		return;
	if (traceFile == NULL)
	{
		char		path[MAXPGPATH];

		snprintf(path, sizeof(path), "/tmp/ann-fanout-%d.trace", MyProcPid);
		if ((traceFile = fopen(path, "a")) == NULL)
			return;
	}
	fprintf(traceFile, "C %d " UINT64_FORMAT " %s %d " UINT64_FORMAT " " UINT64_FORMAT " " UINT64_FORMAT "\n",
			MyProcPid, node->request, node->address, node->viaRdma ? 1 : 0, fanoutStart, node->traceSent,
			TraceNow());
}

static void
FanoutClose(FanoutNode *node)
{
#ifdef ANN_USE_RDMA
	RdmaFree(node->rdma);
	node->rdma = NULL;
	node->armed = false;
#endif
	if (node->fd >= 0)
		close(node->fd);
	node->fd = -1;
	node->busy = false;
	node->got = 0;
	node->body = NULL;
}

static FanoutNode *
FanoutNodeFor(const char *address, size_t length)
{
	FanoutNode *node;

	for (int i = 0; i < fanoutCount; i++)
		if (strlen(fanoutNodes[i].address) == length && strncmp(fanoutNodes[i].address, address, length) == 0)
			return &fanoutNodes[i];
	if (fanoutCount == ANN_FANOUT_NODES)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("more than %d ann nodes in use by one backend", ANN_FANOUT_NODES)));
	node = &fanoutNodes[fanoutCount];
	memset(node, 0, sizeof(*node));
	node->fd = -1;
	node->address = strndup(address, length);
	if (node->address == NULL)
		ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("out of memory")));
	fanoutCount++;
	return node;
}

static bool
FanoutAvailable(FanoutNode *node)
{
	if (INSTR_TIME_IS_ZERO(node->down))
		return true;
	if (ElapsedMs(node->down) < ann_worker_retry_ms)
		return false;
	INSTR_TIME_SET_ZERO(node->down);
	return true;
}

/* Connect (and probe an idle connection) so a request can be sent. */
static bool
FanoutReady(FanoutNode *node, int timeoutMs)
{
	if (node->fd >= 0 && ann_worker_heartbeat_ms > 0 && ElapsedMs(node->used) > ann_worker_heartbeat_ms)
	{
		uint32		status;
		char	   *reply;
		uint32		length;

		node->busy = true;
		if (!Exchange(node->fd, ANN_OP_PROBE, NULL, 0, ann_worker_heartbeat_ms, &status, &reply, &length))
		{
			elog(DEBUG1, "ann node %s missed a heartbeat", node->address);
			FanoutClose(node);
			INSTR_TIME_SET_CURRENT(node->down);
			return false;
		}
		node->busy = false;
		INSTR_TIME_SET_CURRENT(node->used);
	}
	if (node->fd < 0)
	{
		char	   *host = pstrdup(node->address);
		char	   *colon = strrchr(host, ':');

		if (colon == NULL || colon == host)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("invalid ann node address \"%s\"", node->address)));
		*colon = '\0';
		node->fd = ConnectNode(host, colon + 1, Max(timeoutMs, 1000));
		if (node->fd < 0)
		{
			INSTR_TIME_SET_CURRENT(node->down);
			return false;
		}
#ifdef ANN_USE_RDMA
		if (ann_worker_rdma && ann_worker_rdma[0] != '\0')
			node->rdma = RdmaConnectChecked(node->fd, Max(timeoutMs, 1000));
#endif
		INSTR_TIME_SET_CURRENT(node->used);
	}
	return true;
}

/* Send a request without waiting for its reply. */
static bool
FanoutSend(FanoutNode *node, AnnRemoteRequest *request, int owner, int timeoutMs)
{
	char		header[ANN_HEADER_SIZE];
	instr_time	start;

	/* Ids unique across backends, so traces of both ends join on them. */
	if (remoteRequest >> 32 != (uint64) MyProcPid)
		remoteRequest = ((uint64) MyProcPid << 32) + 1;
	node->request = remoteRequest++;
	node->traceSent = ann_worker_trace_sample > 0 ? TraceNow() : 0;
	node->owner = owner;
	node->got = 0;
	node->body = NULL;
	node->readable = false;
	node->viaRdma = false;
	node->busy = true;
#ifdef ANN_USE_RDMA
	if (node->rdma && RdmaOperation(request->operation) &&
		ANN_HEADER_SIZE + request->length <= node->rdma->frame)
	{
		RdmaClient *client = node->rdma;
		char	   *buffer = client->region + client->frame;
		struct ibv_sge sge;
		struct ibv_send_wr wr = {0};
		struct ibv_send_wr *bad;

		if (!ReapRdmaSend(client))
			return false;
		PutHeaderGeneration(buffer, request->operation, node->request, (uint32) request->length,
							request->generation);
		if (request->length)
			memcpy(buffer + ANN_HEADER_SIZE, request->payload, request->length);
		sge.addr = (uintptr_t) buffer;
		sge.length = (uint32) (ANN_HEADER_SIZE + request->length);
		sge.lkey = client->mr->lkey;
		wr.wr_id = 2;
		wr.sg_list = &sge;
		wr.num_sge = 1;
		wr.opcode = IBV_WR_SEND;
		wr.send_flags = IBV_SEND_SIGNALED;
		if (ibv_post_send(client->qp, &wr, &bad) != 0)
			return false;
		client->sending = true;
		node->viaRdma = true;
		return true;
	}
#endif
	INSTR_TIME_SET_CURRENT(start);
	PutHeaderGeneration(header, request->operation, node->request, (uint32) request->length,
						request->generation);
	return SendAll(node->fd, header, sizeof(header), start, timeoutMs) &&
		(request->length == 0 || SendAll(node->fd, request->payload, request->length, start, timeoutMs));
}

/*
 * Read what has arrived on a node's TCP socket without blocking. Returns 1
 * when the reply is complete, 0 when more is to come, -1 on failure.
 */
static int
FanoutReadTcp(FanoutNode *node, AnnRemoteRequest *request)
{
	for (;;)
	{
		ssize_t		n;
		uint32		size;

		if (node->got < ANN_HEADER_SIZE)
			n = recv(node->fd, node->header + node->got, ANN_HEADER_SIZE - node->got, 0);
		else
		{
			memcpy(&size, node->header + 24, 4);
			if (node->body == NULL)
			{
				uint64		replyRequest;

				memcpy(&replyRequest, node->header + 8, 8);
				if (memcmp(node->header, "PVAN", 4) != 0 || replyRequest != node->request ||
					size > 64 * 1024 * 1024)
					return -1;
				node->body = palloc(size + 1);
			}
			if (node->got == ANN_HEADER_SIZE + size)
			{
				node->body[size] = '\0';
				memcpy(&request->status, node->header + 28, 4);
				request->reply = node->body;
				request->replyLength = size;
				return 1;
			}
			n = recv(node->fd, node->body + (node->got - ANN_HEADER_SIZE),
					 ANN_HEADER_SIZE + size - node->got, 0);
		}
		if (n > 0)
			node->got += n;
		else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR))
			return 0;
		else
			return -1;
	}
}

#ifdef ANN_USE_RDMA
/* Reap a node's RDMA completions; 1 when its reply arrived, -1 on failure. */
static int
FanoutPollRdma(FanoutNode *node, AnnRemoteRequest *request)
{
	RdmaClient *client = node->rdma;
	struct ibv_wc wc;
	int			n;

	while ((n = ibv_poll_cq(client->cq, 1, &wc)) == 1)
	{
		if (wc.status != IBV_WC_SUCCESS)
			return -1;
		if (wc.wr_id == 2)
		{
			client->sending = false;
			continue;
		}
		if (!CopyReply(client->region, wc.byte_len, node->request, &request->status, &request->reply,
					   &request->replyLength) ||
			!PostRdmaReceive(client))
			return -1;
		return 1;
	}
	return n < 0 ? -1 : 0;
}
#endif

/*
 * The next replica of a request to try, or NULL when none can take it now;
 * *exhausted is then set when none ever will.
 */
static FanoutNode *
FanoutPick(AnnRemoteRequest *request, int timeoutMs, bool *exhausted)
{
	const char *list = request->nodes;
	const char *starts[ANN_MAX_NODES];
	size_t		lengths[ANN_MAX_NODES];
	int			count = 0;
	bool		waiting = false;

	while (*list && count < ANN_MAX_NODES)
	{
		size_t		length;

		while (*list == ',' || *list == ' ')
			list++;
		length = strcspn(list, ", ");
		if (length == 0)
			break;
		starts[count] = list;
		lengths[count++] = length;
		list += length;
	}
	for (int pass = 0; pass < 2; pass++)
		for (int i = 0; i < count; i++)
		{
			int			index = (int) ((request->rotation + i) % count);
			FanoutNode *node = FanoutNodeFor(starts[index], lengths[index]);

			if (request->tried & (UINT64_C(1) << index))
				continue;
			if (node->busy)
			{
				waiting = true; /* free once its current reply is in */
				continue;
			}
			if (pass == 0 && !FanoutAvailable(node))
				continue;
			request->tried |= UINT64_C(1) << index;
			if (FanoutReady(node, timeoutMs))
				return node;
		}
	*exhausted = !waiting;
	return NULL;
}

/*
 * Send every request and collect the replies within timeoutMs of sending
 * them; connecting to a node for the first time has a budget of its own (at
 * least a second), so a new backend's first search isn't cut short. Each
 * request's ok, status and reply are set; replies are palloc'd in the
 * current context. Returns the number of requests answered.
 */
int
AnnRemoteFanout(AnnRemoteRequest *requests, int count, int timeoutMs)
{
	return AnnRemoteFanoutWhile(requests, count, timeoutMs, NULL, NULL);
}

/*
 * AnnRemoteFanout, calling whileWaiting(arg) once the first requests are
 * sent and before any reply is read: work of the caller's own (an exact
 * query over local rows) overlaps the nodes'.
 */
int
AnnRemoteFanoutWhile(AnnRemoteRequest *requests, int count, int timeoutMs, void (*whileWaiting) (void *),
					 void *arg)
{
	FanoutNode *active[ANN_FANOUT_NODES];
	int			nactive = 0;
	int			pending = count;
	int			answered = 0;
	bool		started = false;
	instr_time	start;
	uint64		fanoutStart = ann_worker_trace_sample > 0 ? TraceNow() : 0;

	/*
	 * Over TCP, a backend rotates its requests over each part's replicas.
	 * Over RDMA it keeps to one replica per part (by process ID, the next one
	 * only when that fails): every node a backend uses costs a queue pair at
	 * both ends, and past about a thousand active queue pairs the adapters'
	 * message rate and latency fall (206k QPS at 256 clients against 172k at
	 * 1,024, rotating over four replicas).
	 */
	bool		sticky = ann_worker_rdma != NULL && ann_worker_rdma[0] != '\0';

	if (timeoutMs <= 0)
		timeoutMs = ann_worker_timeout_ms;
	INSTR_TIME_SET_CURRENT(start);

	/* An exchange an error interrupted may have left a reply unread. */
	for (int i = 0; i < fanoutCount; i++)
		if (fanoutNodes[i].busy)
			FanoutClose(&fanoutNodes[i]);
	for (int i = 0; i < count; i++)
	{
		requests[i].ok = false;
		requests[i].done = false;
		requests[i].status = UINT32_MAX;
		requests[i].reply = NULL;
		requests[i].replyLength = 0;
		requests[i].tried = 0;
		requests[i].rotation = sticky ? (uint64) MyProcPid : (uint64) MyProcPid + fanoutRotation++;
	}

	while (pending > 0)
	{
		instr_time	spinStart;

		/* Send what can be sent: each request to a free replica. */
		for (int i = 0; i < count; i++)
		{
			AnnRemoteRequest *request = &requests[i];
			bool		inflight = false;
			bool		exhausted = false;
			FanoutNode *node;

			if (request->done)
				continue;
			for (int a = 0; a < nactive && !inflight; a++)
				inflight = active[a]->owner == i;
			if (inflight)
				continue;
			while ((node = FanoutPick(request, timeoutMs, &exhausted)) != NULL)
			{
				if (FanoutSend(node, request, i, timeoutMs))
				{
					active[nactive++] = node;
					break;
				}
				elog(DEBUG1, "ann node %s failed a send; trying another replica", node->address);
				FanoutClose(node);
				INSTR_TIME_SET_CURRENT(node->down);
			}
			if (node == NULL && exhausted)
			{
				request->done = true;
				pending--;
			}
		}
		if (!started && whileWaiting)
		{
			/* Nothing is read until it returns; a reply waits in its socket. */
			whileWaiting(arg);
			whileWaiting = NULL;
		}
		if (pending == 0 || nactive == 0)
			break;
		if (!started)
		{
			/* The deadline counts from the first sends, after connecting. */
			INSTR_TIME_SET_CURRENT(start);
			started = true;
		}

		/* Collect replies until at least one request finishes. */
		INSTR_TIME_SET_CURRENT(spinStart);
		for (;;)
		{
			bool		finished = false;
			bool		rdmaWaiting = false;
			bool		newlyArmed = false;
			struct pollfd pollers[2 * ANN_FANOUT_NODES];
			int			npollers = 0;
			double		elapsed;

			for (int a = 0; a < nactive; a++)
			{
				FanoutNode *node = active[a];
				AnnRemoteRequest *request = &requests[node->owner];
				int			result = 0;

#ifdef ANN_USE_RDMA
				if (node->viaRdma)
				{
					result = FanoutPollRdma(node, request);
					rdmaWaiting |= result == 0;
				}
#endif
				/* A reply too large for RDMA comes over TCP. */
				if (result == 0 && (!node->viaRdma || node->readable || node->got > 0))
					result = FanoutReadTcp(node, request);
				node->readable = false;
				if (result == 0)
					continue;
				node->busy = false;
				if (result > 0)
				{
					INSTR_TIME_SET_CURRENT(node->used);
					if (ann_worker_trace_sample > 0)
						TraceReply(node, fanoutStart);
					request->ok = true;
					request->done = true;
					answered++;
					pending--;
				}
				else
				{
					elog(DEBUG1, "ann node %s failed; trying another replica", node->address);
					FanoutClose(node);
					INSTR_TIME_SET_CURRENT(node->down);
				}
				active[a--] = active[--nactive];
				finished = true;
			}
			if (finished || nactive == 0)
				break;
			elapsed = ElapsedMs(start);
			if (elapsed > timeoutMs)
			{
				/* Late replies would come out of order: close. */
				for (int a = 0; a < nactive; a++)
				{
					requests[active[a]->owner].done = true;
					FanoutClose(active[a]);
				}
				return answered;
			}
			if (rdmaWaiting && ElapsedMs(spinStart) * 1000.0 < ann_worker_rdma_spin_us)
				continue;
			CHECK_FOR_INTERRUPTS();
#ifdef ANN_USE_RDMA
			/* Arm, and poll again: a completion may race it. */
			for (int a = 0; a < nactive; a++)
				if (active[a]->viaRdma && !active[a]->armed &&
					ibv_req_notify_cq(active[a]->rdma->cq, 0) == 0)
				{
					active[a]->armed = true;
					newlyArmed = true;
				}
			if (newlyArmed)
				continue;
#endif
			for (int a = 0; a < nactive; a++)
			{
#ifdef ANN_USE_RDMA
				if (active[a]->viaRdma)
				{
					pollers[npollers].fd = active[a]->rdma->channel->fd;
					pollers[npollers].events = POLLIN;
					pollers[npollers++].revents = 0;
				}
#endif
				pollers[npollers].fd = active[a]->fd;
				pollers[npollers].events = POLLIN;
				pollers[npollers++].revents = 0;
			}
			if (poll(pollers, npollers, Max(1, Min(10, (int) (timeoutMs - elapsed)))) < 0 && errno != EINTR)
				break;
			for (int p = 0, a = 0; a < nactive; a++)
			{
				FanoutNode *node = active[a];

#ifdef ANN_USE_RDMA
				if (node->viaRdma)
				{
					if (pollers[p].revents & POLLIN)
					{
						struct ibv_cq *eventCq;
						void	   *ignored;

						if (ibv_get_cq_event(node->rdma->channel, &eventCq, &ignored) == 0)
							ibv_ack_cq_events(eventCq, 1);
						node->armed = false;
					}
					p++;
				}
#endif
				if (pollers[p].revents & (POLLIN | POLLERR | POLLHUP))
					node->readable = true;
				p++;
			}
			INSTR_TIME_SET_CURRENT(spinStart);
		}
	}
	if (whileWaiting)
		whileWaiting(arg);		/* no request was sent */
	return answered;
}

/*
 * Send one request to every node, continuing past failures, for caches that
 * nodes can refill later. Returns the number of nodes that accepted it;
 * *failed counts the rest and *status holds the last failure's status
 * (UINT32_MAX when a node was unreachable).
 */
int
AnnRemoteBroadcastEach(uint16 operation, const char *payload, size_t length, int timeoutMs,
					   int *failed, uint32 *status)
{
	char	   *hosts[ANN_MAX_NODES];
	char	   *ports[ANN_MAX_NODES];
	int			count = ParseNodes(ann_worker_address, hosts, ports);
	int			accepted = 0;

	*failed = 0;
	for (int i = 0; i < count; i++)
	{
		int			fd = ConnectNode(hosts[i], ports[i], timeoutMs);
		uint32		nodeStatus = UINT32_MAX;
		char	   *reply = NULL;
		uint32		replyLength = 0;

		if (fd >= 0)
		{
			if (!Exchange(fd, operation, payload, length, timeoutMs, &nodeStatus, &reply, &replyLength))
				nodeStatus = UINT32_MAX;
			close(fd);
		}
		if (nodeStatus == 0)
			accepted++;
		else
		{
			(*failed)++;
			*status = nodeStatus;
		}
	}
	return accepted;
}

#endif

void
AnnRemoteInit(void)
{
	DefineCustomStringVariable("ann.worker_address",
							   "Comma-separated host:port list of networked ann GPU workers.",
							   "When set, loaded-index functions use these nodes instead of ann.worker_socket.",
							   &ann_worker_address, "", PGC_SUSET, 0, NULL, NULL, NULL);
	DefineCustomStringVariable("ann.worker_auth_token",
							   "Shared secret sent to networked ann workers.",
							   NULL, &ann_worker_auth_token, "", PGC_SUSET,
							   GUC_SUPERUSER_ONLY | GUC_NO_SHOW_ALL, NULL, NULL, NULL);
	DefineCustomIntVariable("ann.worker_heartbeat_ms",
							"Idle time after which a GPU node connection is probed before reuse.",
							"Also sets TCP keepalive timing; 0 disables heartbeats.",
							&ann_worker_heartbeat_ms, 5000, 0, INT_MAX, PGC_USERSET, GUC_UNIT_MS,
							NULL, NULL, NULL);
	DefineCustomIntVariable("ann.worker_retry_ms",
							"Time a failed GPU node is skipped before being tried again.",
							NULL, &ann_worker_retry_ms, 10000, 0, INT_MAX, PGC_USERSET, GUC_UNIT_MS,
							NULL, NULL, NULL);
	DefineCustomStringVariable("ann.worker_rdma",
							   "InfiniBand/RoCE device[:port] for RDMA search frames to GPU nodes.",
							   "Empty uses TCP only. Requires a build with USE_RDMA=1.",
							   &ann_worker_rdma, "", PGC_SUSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("ann.worker_rdma_spin_us",
							"Time a backend busy-polls for an RDMA reply before sleeping.",
							"After this the backend waits on a completion channel and its TCP socket.",
							&ann_worker_rdma_spin_us, 50, 0, 1000000, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("ann.worker_trace_sample",
							"Trace every Nth collection request's send and reply times (0 for none).",
							"Lines go to /tmp/ann-fanout-PID.trace; workers started with --trace record the same requests.",
							&ann_worker_trace_sample, 0, 0, INT_MAX, PGC_SUSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("ann.collection_text_range_limit",
							"Most strings a text range in an ann collection filter may match.",
							"Ranges over text attributes become the set of the matching strings' codes; larger sets need bigger programs.",
							&ann_collection_text_range_limit, 65536, 1, ANN_TEXT_RANGE_MAX, PGC_SUSET, 0, NULL,
							AnnCollectionTextRangeLimitAssign, NULL);
	DefineCustomIntVariable("ann.worker_overload_wait_ms",
							"Time a loaded-index search retries while the GPU node is overloaded.",
							"Retries back off from 0.5 to 16 ms; 0 reports overload at once.",
							&ann_worker_overload_wait_ms, 1000, 0, INT_MAX, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("ann.gpu_min_eligible",
							"Filters with fewer eligible rows are scored exactly on this host.",
							"Filtered loaded-index searches below this count never leave the host.",
							&ann_gpu_min_eligible, 1000, 0, INT_MAX, PGC_USERSET, 0, NULL, NULL, NULL);
}
