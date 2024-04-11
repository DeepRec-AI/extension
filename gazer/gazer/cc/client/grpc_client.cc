#include "gazer/cc/client/grpc_client.h"

#include <memory>
#include <chrono>

#include "gazer/cc/proto/scheduler.pb.h"
#include "gazer/cc/proto/scheduler.grpc.pb.h"
#include "grpcpp/support/channel_arguments.h"
#include "grpcpp/support/config.h"

namespace gazer {

using tensorflow::Status;
using tensorflow::error::Code;

inline std::string utc_time() {
  std::time_t now = std::time(0);
  std::tm* now_tm= std::gmtime(&now);
  char buf[42];
  std::strftime(buf, 42, "%Y%m%d %X", now_tm);
  return buf;
}

Status ReportMetricsClient::Initialize(const std::string& ai_master_addr)
{
  if (!initialized) {
   std::lock_guard<std::mutex> lock(mutexLock);
   initialized = true;
   grpc::ChannelArguments channel_args;
   channel_args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 1000);
   channel_args.SetInt(GRPC_ARG_ENABLE_RETRIES, 1);
   channel_ = grpc::CreateCustomChannel(
                "dns:///" + ai_master_addr, grpc::InsecureChannelCredentials(), channel_args);
   stub_ = ::deeprecmaster::ElasticTrainingService::NewStub(channel_);
   std::chrono::time_point<std::chrono::system_clock> deadline = 
      std::chrono::system_clock::now() + std::chrono::milliseconds(100);
   if (channel_->WaitForConnected(deadline)) {
    thread_.reset(new std::thread(
      [this]() {this->AsyncCompleteResponse();}));
    return Status::OK();
   } else {
    return Status(Code::DEADLINE_EXCEEDED, "AImaster is not ready.");
   }
  } else {
    return Status::OK();
  }
}
  
void ReportMetricsClient::AsyncReport(
    const std::string& job,
    int task_id, const JsonWriter& json_writer) {
  ::deeprecmaster::SendMetricsRequest job_monitor_request;
  double fractional_seconds_since_epoch
    = std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::system_clock::now().time_since_epoch()).count();
  job_monitor_request.set_timestamp(fractional_seconds_since_epoch);
  job_monitor_request.set_task_name(job);
  job_monitor_request.set_task_index(task_id);
  // {"MetricType": tag, "MetricValue": value}
  job_monitor_request.set_options(json_writer.GetString());

  VLOG(1) << "[DEBUG]: Request is: " << job_monitor_request.DebugString();
  AsyncClientCall* call  = new AsyncClientCall;
  //timeout 100ms
  std::chrono::time_point<std::chrono::system_clock> deadline = 
      std::chrono::system_clock::now() + std::chrono::milliseconds(100);
  call->context.set_deadline(deadline);
  call->response_reader =
        stub_->PrepareAsyncSendMetrics(&call->context, job_monitor_request, &cq_);
  call->response_reader->StartCall();
  call->response_reader->Finish(&call->response, &call->status, (void*)call);
}

void ReportMetricsClient::AsyncCompleteResponse() {
  
  void* got_tag;
  bool ok = false;
  // Block until the next result is available in the completion queue "cq".
  // The return value of Next should always be checked. This return value
  // tells us whether there is any kind of event or the cq_ is shutting down.
  while (cq_.Next(&got_tag, &ok)) {
    AsyncClientCall* call = static_cast<AsyncClientCall*>(got_tag);
    if (call->status.ok()) {
      VLOG(1) << "RPC OK.";
    } else {
      LOG(INFO) << "Report AIMaster RPC failed: "
                << call->response.msg();;
    }
    delete call;
  }

}

} // namespace gazer
