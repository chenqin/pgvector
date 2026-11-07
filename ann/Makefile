# Standalone reference worker; independent of PGXS and GPU SDKs.
CC ?= cc
CFLAGS ?= -O2 -g
CFLAGS += -std=c11 -Wall -Wextra -Werror -pedantic
CUDA_HOME ?= /usr/local/cuda

# USE_CUVS=1 also builds the cuVS worker (requires CUVS_PREFIX); hosts
# without a GPU SDK keep the default CPU-only build.
USE_CUVS ?= 0

ifeq ($(USE_CUVS),1)
all: ann-worker libannbatch.a ann-worker-cuvs
else
all: ann-worker libannbatch.a
endif

batching.o: batching.c batching.h backend.h
	$(CC) $(CFLAGS) -pthread -c batching.c -o $@

cpu.o: cpu.c backend.h indexfile.h
	$(CC) $(CFLAGS) -c cpu.c -o $@

indexfile.o: indexfile.c indexfile.h backend.h filter.h canonical.h
	$(CC) $(CFLAGS) -pthread -c indexfile.c -o $@

canonical.o: canonical.c canonical.h filter.h
	$(CC) $(CFLAGS) -c canonical.c -o $@

filter.o: filter.c filter.h
	$(CC) $(CFLAGS) -c filter.c -o $@

libannbatch.a: batching.o cpu.o indexfile.o filter.o canonical.o
	$(AR) rcs $@ batching.o cpu.o indexfile.o filter.o canonical.o

ann-batch-tests: ../test/ann/batch_test.c batching.c cpu.c indexfile.c filter.c canonical.c batching.h backend.h
	$(CC) $(CFLAGS) -pthread -I. ../test/ann/batch_test.c batching.c cpu.c indexfile.c filter.c canonical.c -lm -o $@

ann-udf-tests: ../test/ann/udf_test.c udf.c udf.h filter.c filter.h
	$(CC) $(CFLAGS) -I. ../test/ann/udf_test.c udf.c filter.c -lm -o $@

ann-canonical-tests: ../test/ann/canonical_test.c canonical.c canonical.h filter.c filter.h
	$(CC) $(CFLAGS) -I. ../test/ann/canonical_test.c canonical.c filter.c -lm -o $@

ann-cost-tests: ../test/ann/cost_test.c cost.h
	$(CC) $(CFLAGS) -I. ../test/ann/cost_test.c -lm -o $@

ann-replica-tests: ../test/ann/replica_test.c replica.h
	$(CC) $(CFLAGS) -pthread -I. ../test/ann/replica_test.c -o $@

ann-batch-bench: ../benchmarks/ann/batch.c batching.c cpu.c indexfile.c filter.c canonical.c batching.h backend.h
	$(CC) $(CFLAGS) -pthread -I. ../benchmarks/ann/batch.c batching.c cpu.c indexfile.c filter.c canonical.c -lm -o $@

WORKER_SRCS = worker.c cpu.c mockgpu.c batching.c server.c indexfile.c filter.c canonical.c udf.c rdma.c
WORKER_DEPS = $(WORKER_SRCS) batching.h backend.h server.h indexfile.h filter.h canonical.h udf.h rdma.h cost.h

# USE_RDMA=1 enables RDMA search frames over InfiniBand/RoCE verbs
# (rdma-core); RDMA_PREFIX locates libibverbs if it is not in /usr.
USE_RDMA ?= 0
ifeq ($(USE_RDMA),1)
RDMA_PREFIX ?= /usr
CFLAGS += -DANN_USE_RDMA -I$(RDMA_PREFIX)/include
WORKER_LIBS = -L$(RDMA_PREFIX)/lib -libverbs
endif

ann-worker: $(WORKER_DEPS)
	$(CC) $(CFLAGS) -pthread $(WORKER_SRCS) $(WORKER_LIBS) -lm -o $@

# Monolithic SDKs keep these headers under CUVS_PREFIX/include; pip SDKs
# split them into sibling packages, with the CCCL RAPIDS built against under
# libraft/include/rapids. That CCCL must come before the CUDA toolkit's, whose
# older cuda/std would otherwise mix with it header by header; cupy's partial
# copy is a last resort. CUVS_CXXFLAGS can supply other layouts.
CUVS_CPP_DIRS = $(wildcard $(CUVS_PREFIX)/../libraft/include/rapids $(CUVS_PREFIX)/../libraft/include $(CUVS_PREFIX)/../librmm/include $(CUVS_PREFIX)/../rapids_logger/include)
CUVS_CPP_FALLBACK_DIRS = $(wildcard $(CUVS_PREFIX)/../cupy/_core/include/cupy/_cccl/libcudacxx $(CUVS_PREFIX)/../cupy/_core/include/cupy/_cccl/cub $(CUVS_PREFIX)/../cupy/_core/include/cupy/_cccl/thrust)

select_batch.o: select_batch.cpp select_batch.h
	@test -n "$(CUVS_PREFIX)" || { echo 'Set CUVS_PREFIX to the cuVS 26.10 installation' >&2; exit 1; }
	$(CXX) $(CXXFLAGS) -O2 -std=c++20 -I"$(CUVS_PREFIX)/include" $(addprefix -I,$(CUVS_CPP_DIRS)) -I"$(CUDA_HOME)/include" $(addprefix -I,$(CUVS_CPP_FALLBACK_DIRS)) $(CPPFLAGS) $(CUVS_CXXFLAGS) -c select_batch.cpp -o $@

# JIT-LTO filter UDF search for FP32 CAGRA (experimental; see
# docs/ann-cagra-filter-udf.md). pip SDKs need CUVS_CXXFLAGS with
# -I$(CUVS_PREFIX)/../librmm/include/rapids -DRAFT_SYSTEM_LITTLE_ENDIAN=1.
cagra_udf.o: cagra_udf.cpp cagra_udf.h udf.h
	@test -n "$(CUVS_PREFIX)" || { echo 'Set CUVS_PREFIX to the cuVS 26.10 installation' >&2; exit 1; }
	$(CXX) $(CXXFLAGS) -O2 -std=c++20 $(CUVS_CXXFLAGS) -I"$(CUVS_PREFIX)/include" -I"$(CUDA_HOME)/include" $(addprefix -I,$(CUVS_CPP_DIRS)) $(CPPFLAGS) -c cagra_udf.cpp -o $@

# Bitset versus filter UDF on one FP32 CAGRA shard: cagra-udf-probe INDEX_DIR [k itopk slack queries]
cagra-udf-probe: ../benchmarks/ann/cagra_udf_probe.c cagra_udf.o udf.c udf.h filter.c filter.h
	$(CC) -O3 -std=gnu11 -fopenmp -Wall -I. -I"$(CUVS_PREFIX)/include" -I"$(CUDA_HOME)/include" $(CPPFLAGS) ../benchmarks/ann/cagra_udf_probe.c udf.c filter.c cagra_udf.o -L"$(CUVS_PREFIX)/lib" -L"$(CUVS_PREFIX)/lib64" -L"$(CUDA_HOME)/lib64" $(LDFLAGS) -lcuvs_c -lcuvs -lstdc++ -lcudart -lm -o $@

# Requires cuVS 26.10 C/C++ SDKs, RAFT/RMM/CCCL, DLPack and CUDA runtime headers.
# A different executable keeps the dependency-free CPU build reproducible.
ann-worker-cuvs: $(WORKER_DEPS) cuvs.c replica.h nvrtc.c nvrtc.h blackwell.c blackwell.h select_batch.o select_batch.h
	@test -n "$(CUVS_PREFIX)" || { echo 'Set CUVS_PREFIX to the cuVS 26.10 installation' >&2; exit 1; }
	$(CC) $(CFLAGS) -pthread -DANN_USE_CUVS -I"$(CUVS_PREFIX)/include" -I"$(CUDA_HOME)/include" $(CPPFLAGS) $(WORKER_SRCS) cuvs.c nvrtc.c blackwell.c select_batch.o -L"$(CUVS_PREFIX)/lib" -L"$(CUVS_PREFIX)/lib64" -L"$(CUDA_HOME)/lib64" $(LDFLAGS) $(WORKER_LIBS) -lcuvs_c -lcuvs -lstdc++ -lcudart -lcuda -ldl -lm -o $@

check: ann-worker ann-batch-tests ann-replica-tests ann-cost-tests ann-udf-tests ann-canonical-tests
	./ann-cost-tests
	./ann-udf-tests
	./ann-canonical-tests
	./ann-replica-tests
	python3 ../test/ann/test_worker.py
	python3 ../test/ann/test_batching.py
	python3 ../test/ann/test_server.py
	python3 ../test/ann/test_index.py
	python3 ../test/ann/test_multivalued.py
	python3 ../test/ann/test_predicate_cache.py
	python3 ../test/ann/test_cuvs.py
	python3 ../test/ann/test_postgres.py
	python3 ../test/ann/test_native.py
	python3 ../test/ann/test_filtered.py
	python3 ../test/ann/test_pipeline.py
	python3 ../test/ann/test_frozen.py
	python3 ../test/ann/test_collection.py

check-cuvs: ann-worker ann-worker-cuvs
	ANN_CUVS_TEST=1 python3 ../test/ann/test_cuvs.py
	ANN_CUVS_TEST=1 python3 ../test/ann/test_cuvs_gathered.py
	ANN_CUVS_TEST=1 python3 ../test/ann/test_cuvs_index.py
	ANN_CUVS_TEST=1 python3 ../test/ann/test_frozen.py FrozenCuvsTests

check-postgres: ann-worker
	@test -n "$$ANN_PG_DSN" || { echo 'Set ANN_PG_DSN to a test database with pgvector installed' >&2; exit 1; }
	python3 ../test/ann/test_postgres.py
	python3 ../test/ann/test_native.py
	python3 ../test/ann/test_filtered.py
	python3 ../test/ann/test_pipeline.py
	python3 ../test/ann/test_index.py
	python3 ../test/ann/test_frozen.py

clean:
	rm -f ann-worker
	rm -f ann-replica-tests ann-cost-tests ann-udf-tests
	rm -rf ann-replica-tests.dSYM ann-cost-tests.dSYM ann-udf-tests.dSYM
	rm -f batching.o cpu.o indexfile.o filter.o canonical.o libannbatch.a ann-batch-tests
	rm -f ann-batch-bench
	rm -rf ann-batch-bench.dSYM
	rm -rf ann-batch-tests.dSYM
	rm -f ann-canonical-tests ann-worker-cuvs select_batch.o cagra_udf.o cagra-udf-probe
	rm -rf ann-worker.dSYM
	rm -rf ann-worker-cuvs.dSYM

.PHONY: all check check-cuvs check-postgres clean
