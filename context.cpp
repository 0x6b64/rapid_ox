#include "context.hpp"
#include <algorithm>

int Context::init(const std::string &deviceName) {
  context = createContext(deviceName);
  if (context == nullptr) {
    std::cerr << "Failed to create RDMA context" << std::endl;
    return -1;
  }

  // Protection domain
  protectionDomain = ibv_alloc_pd(context);
  if (protectionDomain == nullptr) {
    std::cerr << "Failed to create protection domain" << std::endl;
    destroy();
    return -1;
  }

  // Send completion queue
  sendCQ = ibv_create_cq(context, txBufferDepth, nullptr, nullptr, 0);
  if (sendCQ == nullptr) {
    std::cerr << "Failed to create send completion queue" << std::endl;
    destroy();
    return -1;
  }

  // Receive completion queue
  recvCQ = ibv_create_cq(context, rxBufferDepth, nullptr, nullptr, 0);
  if (recvCQ == nullptr) {
    std::cerr << "Failed to create recv completion queue" << std::endl;
    destroy();
    return -1;
  }

  // Queue pair
  queuePair = createQueuePair();
  if (queuePair == nullptr) {
    std::cerr << "Failed to create queue pair" << std::endl;
    destroy();
    return -1;
  }

  // Queue pair ex
  queuePairEx = ibv_qp_to_qp_ex(queuePair);
  if (queuePairEx == nullptr) {
    std::cerr << "Failed to convert queue pair to extended QP" << std::endl;
    destroy();
    return -1;
  }

  // Initialize queue pair
  int err = initQueuePair(queuePair);
  if (err != 0) {
    std::cerr << "Failed to initialize queue pair" << std::endl;
    destroy();
    return -1;
  }

  return 0;
}

void Context::destroy() {
  if (queuePair != nullptr) {
    ibv_destroy_qp(queuePair);
    queuePair = nullptr;
    queuePairEx = nullptr;
  }
  if (sendCQ != nullptr) {
    ibv_destroy_cq(sendCQ);
    sendCQ = nullptr;
  }
  if (recvCQ != nullptr) {
    ibv_destroy_cq(recvCQ);
    recvCQ = nullptr;
  }
  if (protectionDomain != nullptr) {
    ibv_dealloc_pd(protectionDomain);
    protectionDomain = nullptr;
  }
  if (context != nullptr) {
    ibv_close_device(context);
    context = nullptr;
  }
}

int Context::alignDownToPowerOfTwo(int x) {
  if (x <= 0) return 0;
  int n = x;
  while (n & (n - 1)) {
    n = n & (n - 1);
  }
  return n;
}

struct ibv_context *Context::createContext(const std::string &deviceName) {
  struct ibv_context *ctx = nullptr;
  int numDevices = 0;

  struct ibv_device **deviceList = ibv_get_device_list(&numDevices);
  if (deviceList == nullptr) {
    std::cerr << "Failed to get InfiniBand device list" << std::endl;
    return nullptr;
  }

  for (int i = 0; i < numDevices; i++) {
    if (deviceName == ibv_get_device_name(deviceList[i])) {
      ctx = ibv_open_device(deviceList[i]);
      break;
    }
  }

  ibv_free_device_list(deviceList);
  if (ctx == nullptr) {
    std::cerr << "Unable to find the device: " << deviceName << std::endl;
  }

  return ctx;
}

struct ibv_qp *Context::createQueuePair() {
  struct efadv_device_attr efadvAttr;
  memset(&efadvAttr, 0, sizeof(efadvAttr));
  int err = efadv_query_device(context, &efadvAttr, sizeof(efadvAttr));
  if (err != 0) {
    std::cerr << "efadv_query_device failed" << std::endl;
    return nullptr;
  }

  struct ibv_qp_init_attr_ex attrEx;
  memset(&attrEx, 0, sizeof(attrEx));
  attrEx.send_ops_flags = IBV_QP_EX_WITH_SEND | IBV_QP_EX_WITH_RDMA_READ;
  attrEx.pd = protectionDomain;
  attrEx.comp_mask |= IBV_QP_INIT_ATTR_SEND_OPS_FLAGS | IBV_QP_INIT_ATTR_PD;
  attrEx.send_cq = sendCQ;
  attrEx.recv_cq = recvCQ;
  attrEx.cap.max_send_wr = alignDownToPowerOfTwo(efadvAttr.max_sq_wr);
  attrEx.cap.max_send_sge = efadvAttr.max_sq_sge;
  attrEx.qp_type = IBV_QPT_DRIVER;
  attrEx.cap.max_inline_data = efadvAttr.inline_buf_size;
  attrEx.cap.max_recv_wr = alignDownToPowerOfTwo(efadvAttr.max_rq_wr / efadvAttr.max_rq_sge);
  attrEx.cap.max_recv_sge = efadvAttr.max_rq_sge;
  attrEx.qp_context = nullptr;
  attrEx.sq_sig_all = 1;

  struct efadv_qp_init_attr efaAttr = {};
  efaAttr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;

  return efadv_create_qp_ex(context, &attrEx, &efaAttr, sizeof(efaAttr));
}

int Context::initQueuePair(struct ibv_qp *qp) {
  int err;
  struct ibv_qp_attr qpAttr;

  // Step 1: RESET to INIT
  memset(&qpAttr, 0, sizeof(qpAttr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.port_num = 1;
  qpAttr.qkey = MAGIC_QKEY;
  err = ibv_modify_qp(qp, &qpAttr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY);
  if (err) {
    std::cerr << "Init QP step 1 (INIT) failed" << std::endl;
    return err;
  }

  // Step 2: INIT to RTR
  memset(&qpAttr, 0, sizeof(qpAttr));
  qpAttr.qp_state = IBV_QPS_RTR;
  err = ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE);
  if (err) {
    std::cerr << "Init QP step 2 (RTR) failed" << std::endl;
    return err;
  }

  // Step 3: RTR to RTS
  memset(&qpAttr, 0, sizeof(qpAttr));
  qpAttr.qp_state = IBV_QPS_RTS;
  err = ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_SQ_PSN);
  if (err) {
    std::cerr << "Init QP step 3 (RTS) failed" << std::endl;
    return err;
  }

  struct ibv_port_attr portAttr;
  memset(&portAttr, 0, sizeof(portAttr));
  err = ibv_query_port(context, 1, &portAttr);
  if (err) {
    std::cerr << "ibv_query_port failed" << std::endl;
    return err;
  }
  maxMsgSize = portAttr.max_msg_sz;

  return 0;
}