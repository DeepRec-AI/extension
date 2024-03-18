/* Copyright 2024 The DeepRec Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=============================================================================*/
#include "deeprec_master/include/scheduler_service.h"

#include <memory>

#include <grpc/grpc.h>
#include <grpcpp/grpcpp.h>
#include "deeprec_master/include/errors.h"
#include "deeprec_master/include/logging.h"
#include "deeprec_master/include/metrics_mgr.h"
#include "deeprec_master/proto/elastic_training.grpc.pb.h"
#include "deeprec_master/proto/elastic_training.pb.h"

namespace deeprecmaster {

namespace {

class ElasticTrainingServiceImpl : public ElasticTrainingService::Service {
 public:
  ElasticTrainingServiceImpl(PsResourceAnalyzer* ps_res_analyzer,
                             MetricsMgr* metrics_mgr)
      : ps_res_analyzer_(ps_res_analyzer), metrics_mgr_(metrics_mgr) {}

  ~ElasticTrainingServiceImpl() {}

  grpc::Status IsReadyScaling(grpc::ServerContext* context,
                              const IsReadyScalingRequest* request,
                              IsReadyScalingResponse* response) {
    int ps_num = 0;
    ScalingAction scaling_action = ps_res_analyzer_->IsReadyScaling(ps_num);
    response->set_code(Code::OK);
    response->set_msg("ok");
    response->set_scaling_action(scaling_action);
    response->set_ps_num(ps_num);
    return grpc::Status::OK;
  }

  grpc::Status ReadyToUpdate(grpc::ServerContext* context,
                             const ReadyToUpdateRequest* request,
                             ReadyToUpdateResponse* response) {
    ps_res_analyzer_->ReadyToUpdate();
    return grpc::Status::OK;
  }

  grpc::Status GetTFConfig(grpc::ServerContext* context,
                           const MasterRequest* request,
                           MasterResponse* reply) override {
    (void)context;
    std::string tfconfig = ps_res_analyzer_->GetTFConfig();
    if (!tfconfig.empty()) {
      reply->set_code(deeprecmaster::Code::OK);
      reply->set_msg(tfconfig);
    } else {
      reply->set_code(deeprecmaster::Code::NOT_FOUND);
      reply->set_msg("tfconfig not found");
    }
    return grpc::Status::OK;
  }

  grpc::Status SetTFConfig(grpc::ServerContext* context,
                           const MasterRequest* request,
                           MasterResponse* reply) override {
    (void)context;
    ps_res_analyzer_->SetTFConfig(request->msg());
    reply->set_code(deeprecmaster::Code::OK);
    reply->set_msg("");
    return grpc::Status::OK;
  }

  grpc::Status EstimatePSResource(grpc::ServerContext* context,
                                  const PsResourceRequest* request,
                                  MasterResponse* reply) override {
    (void)context;
    Status st = ps_res_analyzer_->EstimatePSResource(
        request->has_ev(), request->ps_num(), request->ps_memory());
    if (st.ok()) {
      reply->set_code(deeprecmaster::Code(st.code()));
      reply->set_msg("ok");
    } else {
      reply->set_code(deeprecmaster::Code(st.code()));
      reply->set_msg(st.msg());
    }
    return grpc::Status::OK;
  }

  grpc::Status SendMetrics(grpc::ServerContext* context,
                           const SendMetricsRequest* request,
                           MasterResponse* reply) override {
    (void)context;
    int64_t timestamp = request->timestamp();
    std::string task_name = request->task_name();
    int32_t task_index = request->task_index();
    std::string options = request->options();
    Status st =
        metrics_mgr_->AddNewMetric({timestamp, task_name, task_index, options});
    if (st.ok()) {
      reply->set_code(deeprecmaster::Code(st.code()));
      reply->set_msg("ok");
    } else {
      reply->set_code(deeprecmaster::Code(st.code()));
      reply->set_msg(st.msg());
    }
    return grpc::Status::OK;
  }

 private:
  PsResourceAnalyzer* ps_res_analyzer_;
  MetricsMgr* metrics_mgr_;
};

class GrpcSchedulerService : public SchedulerService {
 public:
  explicit GrpcSchedulerService(const std::string& ip, int port)
      : ip_(ip), port_(port) {}
  std::string Start() override;
  void Join() override;

 private:
  std::string ip_;
  int port_;
  std::unique_ptr<ElasticTrainingServiceImpl> et_service_;
  std::unique_ptr<grpc::Server> grpc_server_;
};

std::string GrpcSchedulerService::Start() {
  et_service_.reset(new ElasticTrainingServiceImpl(
      PsResourceAnalyzer::GetInstance(), &MetricsMgr::GetInstance()));
  grpc::ServerBuilder builder;
  builder.AddListeningPort("0.0.0.0:" + std::to_string(port_),
                           grpc::InsecureServerCredentials(), &port_);
  builder.RegisterService(et_service_.get());
  grpc_server_ = builder.BuildAndStart();
  std::string my_addr = ip_ + ":" + std::to_string(port_);
  return my_addr;
}

void GrpcSchedulerService::Join() { grpc_server_->Wait(); }

}  // namespace

SchedulerService* NewSchedulerService(const std::string& ip, int port) {
  return new GrpcSchedulerService(ip, port);
}

}  // namespace deeprecmaster
