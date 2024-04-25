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

#include <cstdint>
#include <sstream>
#include <typeinfo>

#include "deeprec_master/include/data_distributor/grpc_data_manager.h"
#include "deeprec_master/include/metrics_mgr.h"
#include "deeprec_master/include/ps_resource.h"
#include "deeprec_master/include/ps_resource_analyzer.h"
#include "deeprec_master/include/scheduler_service.h"
#include "deeprec_master/include/status.h"
#include "deeprec_master/python/pybind.h"

using namespace deeprecmaster;
namespace py = pybind11;

PYBIND11_MODULE(pywrap_deeprecmaster, m) {
  m.doc() = "Python interface for deeprecmaster.";

  py::enum_<error::Code>(m, "ErrorCode", py::arithmetic())
      .value("OK", error::OK)
      .value("CANCELLED", error::CANCELLED)
      .value("UNKNOWN", error::UNKNOWN)
      .value("INVALID_ARGUMENT", error::INVALID_ARGUMENT)
      .value("DEADLINE_EXCEEDED", error::DEADLINE_EXCEEDED)
      .value("NOT_FOUND", error::NOT_FOUND)
      .value("ALREADY_EXISTS", error::ALREADY_EXISTS)
      .value("PERMISSION_DENIED", error::PERMISSION_DENIED)
      .value("RESOURCE_EXHAUSTED", error::RESOURCE_EXHAUSTED)
      .value("FAILED_PRECONDITION", error::FAILED_PRECONDITION)
      .value("ABORTED", error::ABORTED)
      .value("OUT_OF_RANGE", error::OUT_OF_RANGE)
      .value("UNIMPLEMENTED", error::UNIMPLEMENTED)
      .value("INTERNAL", error::INTERNAL)
      .value("UNAVAILABLE", error::UNAVAILABLE)
      .value("DATA_LOSS", error::DATA_LOSS)
      .value("UNAUTHENTICATED", error::UNAUTHENTICATED)
      .value("REQUEST_STOP", error::REQUEST_STOP)
      .value("RETRY_LATER", error::RETRY_LATER)
      .export_values();

  py::class_<Status>(m, "Status")
      .def(py::init<error::Code, const std::string&>())
      .def_static("OK", &Status::OK)
      .def("ok", &Status::ok)
      .def("code", &Status::code)
      .def("message", &Status::msg)
      .def("to_string", &Status::ToString);

  py::enum_<ElasticTrainingState>(m, "ElasticTrainingState", py::arithmetic())
      .value("INIT", INIT)
      .value("READY", READY)
      .value("SCALING", SCALING)
      .value("All_SESSION_CLOSED", All_SESSION_CLOSED)
      .export_values();

  py::class_<SchedulerService>(m, "SchedulerService")
      .def("start", &SchedulerService::Start)
      .def("join", &SchedulerService::Join);

  py::class_<PsResourceAnalyzer>(m, "PsResourceAnalyzer")
      .def("get_estimated_ps_resource",
           &PsResourceAnalyzer::GetEstimatedPSResource)
      .def("set_tfconfig", &PsResourceAnalyzer::SetTFConfig)
      .def("get_tfconfig", &PsResourceAnalyzer::GetTFConfig)
      .def("set_state", &PsResourceAnalyzer::SetState)
      .def("get_state", &PsResourceAnalyzer::GetState);

  py::class_<MetricsMgr>(m, "MetricsMgr")
      .def("get_all_actions", &MetricsMgr::GetAllActions);

  py::class_<PSResource>(m, "PSResource")
      .def("is_initialized", &PSResource::is_initialized)
      .def("has_ev", &PSResource::has_ev)
      .def("ps_num", &PSResource::ps_num)
      .def("ps_memory", &PSResource::ps_memory);
  
  m.def("get_metrics_mgr", &MetricsMgr::GetInstance,
        py::return_value_policy::take_ownership);

  m.def("get_ps_resource_analyzer", &PsResourceAnalyzer::GetInstance,
        py::return_value_policy::take_ownership);

  m.def("new_scheduler", &NewSchedulerService,
        py::return_value_policy::take_ownership, py::arg("ip"),
        py::arg("port"));

  py::class_<DataManagerQueryResult>(m, "DataManagerQueryResult")
    .def_readwrite("status", &DataManagerQueryResult::status)
    .def_readwrite("is_ready", &DataManagerQueryResult::is_ready)
    .def_readwrite("slice_tag", &DataManagerQueryResult::slice_tag)
    .def_readwrite("slice_prefix", &DataManagerQueryResult::slice_prefix)
    .def_readwrite("slice_size", &DataManagerQueryResult::slice_size)
    .def_readwrite("slice_data", &DataManagerQueryResult::slice_data)
    .def_readwrite("data_state", &DataManagerQueryResult::data_state);

  m.def("init_data_manager",
        &InitDataManager,
        py::return_value_policy::take_ownership,
        py::arg("master_addr"),
        py::arg("task_name"),
        py::arg("task_index"),
        py::arg("config"));

  m.def("is_ready_data_manager",
        &IsReadyDataManager,
        py::return_value_policy::take_ownership,
        py::arg("master_addr"),
        py::arg("task_name"),
        py::arg("task_index"));

  m.def("get_slice_from_data_manager",
        &GetSliceFromDataManager,
        py::return_value_policy::take_ownership,
        py::arg("master_addr"),
        py::arg("task_name"),
        py::arg("task_index"));

  m.def("get_data_state_from_data_manager",
        &GetDataStateFromDataManager,
        py::return_value_policy::take_ownership,
        py::arg("master_addr"),
        py::arg("task_name"),
        py::arg("task_index"));

  m.def("start_data_dispatch",
        &DataManagerStartDataDispatch,
        py::return_value_policy::take_ownership,
        py::arg("master_addr"),
        py::arg("task_name"),
        py::arg("task_index"));

  m.def("stop_data_dispatch_and_get_data_state",
        &DataManagerStopDataDispatchAndGetDataState,
        py::return_value_policy::take_ownership,
        py::arg("master_addr"),
        py::arg("task_name"),
        py::arg("task_index"));

}  // NOLINT [readability/fn_size]
