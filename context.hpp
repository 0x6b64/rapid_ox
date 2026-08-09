#pragma once

#include <infiniband/efadv.h>
#include <infiniband/verbs.h>
#include <iostream>
#include <string>

class Context {
public:
  int init(const std::string &deviceName);
  void destroy();

  struct ibv_context *context = nullptr;
  struct ibv_pd *protectionDomain = nullptr;
  struct ibv_cq *sendCQ = nullptr;
  struct ibv_cq *recvCQ = nullptr;
  struct ibv_qp *queuePair = nullptr;
  struct ibv_qp_ex *queuePairEx = nullptr;
  int maxMsgSize;

  // config
  int txBufferDepth = 128;
  int rxBufferDepth = 128;
  int ibPort = 1;
  static const int MAGIC_QKEY = 1234;

private:
  static inline int alignDownToPowerOfTwo(int x);
  struct ibv_context *createContext(const std::string &deviceName);
  struct ibv_qp *createQueuePair();
  int initQueuePair(struct ibv_qp *qp);
};