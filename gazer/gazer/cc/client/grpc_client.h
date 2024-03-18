#pragma once

#include "gazer/cc/proto/scheduler.grpc.pb.h"
#include "gazer/cc/util/json.h"
#include "tensorflow/core/lib/core/status.h"
#include <memory>
#include <thread>
#include <grpcpp/grpcpp.h>
#include <grpc/grpc.h>

namespace gazer {

class ReportMetricsClient {
  public:
    static ReportMetricsClient& GetInstance() {
      static ReportMetricsClient client;
      return client;
    }

    tensorflow::Status Initialize(const std::string& ai_master_addr);
    
    tensorflow::Status ConnectToAM(const std::string& job, int task_id);

    void AsyncReport(const std::string& job,
                     int task_id,
                     const JsonWriter& json_writer);

    void AsyncCompleteResponse();

    ReportMetricsClient() : initialized(false) {}
    ~ReportMetricsClient() = default;
    ReportMetricsClient(const ReportMetricsClient&) = default;
    ReportMetricsClient operator=(const ReportMetricsClient&) = delete;

  private:
    struct AsyncClientCall {
      // Container for the data we expect from the server.
      ::aimaster::JobMonitorResponse response;

      // Context for the client. It could be used to convey extra information to
      // the server and/or tweak certain RPC behaviors.
      grpc::ClientContext context;

      // Storage for the status of the RPC upon completion.
      grpc::Status status;

      std::unique_ptr<grpc::ClientAsyncResponseReader<
          ::aimaster::JobMonitorResponse>> response_reader;
    };

    std::unique_ptr<::aimaster::RpcScheduler::Stub> stub_;
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<std::thread> thread_;
    grpc::CompletionQueue cq_;
    bool initialized;
    std::mutex mutexLock;
};

} // namespace gazer
