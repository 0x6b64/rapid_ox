#include <assert.h>
#include <vector>
#include <memory>
#include <iostream>

#include <cuda_runtime.h>
#include <mpi.h>

#include <boost/log/trivial.hpp>
#include "context.hpp"

#define CUDA_CHECK(stmt)                            \
  do {                                    \
    cudaError_t result = (stmt);                      \
    if (result != cudaSUCCESS) {                      \
      fprintf(stderr, "CUDA Error = %s at %s:%d\n",             \
          cudaGetErrorString(result), __FILE__, __LINE__);      \
      exit(EXIT_FAILURE);                         \
    }                                   \
  } while (0)

struct Address {
  // Filled before handshake
  union ibv_gid gid;
  uint32_t qpn;
  uint32_t qkey;
  // Filled after handshake
  struct ibv_ah *ah;
};

inline ibv_mr* registerMemory(Context *ctx, void *buff, size_t len) {
  int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ;
  return ibv_reg_mr(ctx->protectionDomain, buff, len, flags);
}

void runReader(Context *ctx, int peerRank, Address &peerAddr) {
  BOOST_LOG_TRIVIAL(trace) << "running reader";

  const size_t bufferSize = 128;
  const size_t devAllocSize = 64 * 1024;

  // Allocate host and device buffers
  void *hostBuffer = nullptr;
  CUDA_CHECK(cudaMallocHost(&hostBuffer, bufferSize));

  void *d_A = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, devAllocSize));
  char *buffer = static_cast<char*>(d_A);

  // Register device memory with RDMA
  ibv_mr *mr = registerMemory(ctx, buffer, bufferSize);
  assert(mr != nullptr);

  // Receive remote address and rkey via MPI
  uint64_t raddr = 0;
  uint32_t rkey = 0;
  MPI_Status status;
  MPI_Recv(&raddr, 1, MPI_UNSIGNED_LONG, peerRank, 0, MPI_COMM_WORLD, &status);
  MPI_Recv(&rkey, 1, MPI_UNSIGNED, peerRank, 0, MPI_COMM_WORLD, &status);

  // Setup SGE list for RDMA read
  struct ibv_sge list;
  list.addr = reinterpret_cast<uint64_t>(buffer);
  list.length = bufferSize / 2; // Read right half to demonstrate offset
  list.lkey = mr->lkey;

  // Build work request
  ibv_wr_start(ctx->queuePairEx);
  ctx->queuePairEx->wr_id = 5;
  ctx->queuePairEx->wr_flags |= IBV_SEND_SIGNALED;

  ibv_wr_rdma_read(ctx->queuePairEx, rkey, raddr + 64);
  ibv_wr_set_sge_list(ctx->queuePairEx, 1, &list);
  ibv_wr_set_ud_addr(ctx->queuePairEx, peerAddr.ah, peerAddr.qpn, peerAddr.qkey);

  int ret = ibv_wr_complete(ctx->queuePairEx);
  assert(ret == 0);

  // Poll for completion
  struct ibv_wc ibv_wc;
  int pollRet = 0;
  while (pollRet == 0) {
    pollRet = ibv_poll_cq(ctx->sendCQ, 1, &ibv_wc);
  }

  if (pollRet < 0 || ibv_wc.status != IBV_WC_SUCCESS) {
    fprintf(stderr, "Error: CQ poll failed! status=%d vendor_err=%d\n",
        ibv_wc.status, ibv_wc.vendor_err);
  }

  fprintf(stdout, "Read complete! wr_id: %lu\n", ibv_wc.wr_id);

  // Copy to host memory to verify
  CUDA_CHECK(cudaMemcpy(hostBuffer, d_A, bufferSize / 2, cudaMemcpyDeviceToHost));

  // Verify data
  char *intp = static_cast<char*>(hostBuffer);
  size_t numElements = (bufferSize / 2) / sizeof(char);
  for (size_t i = 0; i < numElements; i++) {
    assert(intp[i] == 'A');
  }
  std::cout << "Data verification passed successfully." << std::endl;

  // Cleanup local resources
  ibv_dereg_mr(mr);
  cudaFree(d_A);
  cudaFreeHost(hostBuffer);

  MPI_Barrier(MPI_COMM_WORLD);
}

void runSender(Context *ctx, Address &dest, int peerRank) {
  BOOST_LOG_TRIVIAL(trace) << "running worker";

  const size_t bufferSize = 128;
  const size_t devAllocSize = 64 * 1024;

  void *d_A = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, devAllocSize));
  CUDA_CHECK(cudaMemset(d_A, 'A', bufferSize));

  ibv_mr *mr = registerMemory(ctx, d_A, bufferSize);
  assert(mr != nullptr);

  // Send device buffer address and rkey to peer
  uint64_t sendAddr = reinterpret_cast<uint64_t>(d_A);
  MPI_Send(&sendAddr, 1, MPI_UNSIGNED_LONG, peerRank, 0, MPI_COMM_WORLD);

  uint32_t rkey = mr->rkey;
  MPI_Send(&rkey, 1, MPI_UNSIGNED, peerRank, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  // Cleanup
  ibv_dereg_mr(mr);
  cudaFree(d_A);

  fprintf(stdout, "Sender Done!\n");
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int worldSize, worldRank;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  Context ctx;
  ctx.init("efa_0");

  bool isWorker = (worldRank >= worldSize / 2);

  MPI_Comm subCluster;
  MPI_Comm_split(MPI_COMM_WORLD, isWorker, worldRank, &subCluster);

  Address myAddr;
  ibv_query_gid(ctx.context, ctx.ibPort, 0, &(myAddr.gid));
  myAddr.qpn = ctx.queuePair->qp_num;
  myAddr.qkey = ctx.MAGIC_QKEY;

  auto allAddress = std::make_shared<std::vector<Address>>(worldSize);
  MPI_Allgather(&myAddr, sizeof(myAddr), MPI_BYTE, allAddress->data(),
          sizeof(myAddr), MPI_BYTE, MPI_COMM_WORLD);

  int otherRank = !worldRank;
  struct ibv_ah_attr ahAttr;
  memset(&ahAttr, 0, sizeof(ahAttr));
  ahAttr.is_global = 1;
  ahAttr.port_num = 1;
  ahAttr.grh.dgid = allAddress->at(otherRank).gid;
  
  allAddress->at(otherRank).ah = ibv_create_ah(ctx.protectionDomain, &ahAttr);
  assert(allAddress->at(otherRank).ah != nullptr);

  CUDA_CHECK(cudaSetDevice(0));

  if (isWorker) {
    Address dest = allAddress->at(otherRank);
    runSender(&ctx, dest, otherRank);
  } else {
    Address peerAddr = allAddress->at(otherRank);
    runReader(&ctx, otherRank, peerAddr);
  }

  ibv_destroy_ah(allAddress->at(otherRank).ah);
  ctx.destroy();
  MPI_Finalize();
  return 0;
}