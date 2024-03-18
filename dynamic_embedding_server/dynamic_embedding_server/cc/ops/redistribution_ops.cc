/* Copyright 2023 The DeepRec Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/

#include "dynamic_embedding_server/include/utils/naming.h"

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP(::des::kEvExportOp)
    .Input("resource: resource")
    .Input("new_partition_nums: int32")
    .Attr("partition_id: int = 0")
    .Attr("partition_nums: int")
    .Output("keys: partition_nums * Tkeys")
    .Output("values: partition_nums * dtype")
    .Output("versions: partition_nums * int64")
    .Output("freqs: partition_nums * int64")
    .Attr("Tkeys: {int64, int32}")
    .Attr("dtype: type")
    .Doc(R"(
Input embedding variable and current partiton_id.
Filter out unneeded ids and values according to new partition num.
)");

REGISTER_OP(::des::kEvImportOp)
    .Input("resource_handle: resource")
    .Input("new_partition_nums: int32")
    .Input("keys: partition_nums * Tkeys")
    .Input("values: partition_nums * dtype")
    .Input("versions: partition_nums * int64")
    .Input("freqs: partition_nums * int64")
    .Attr("partition_id: int = 0")
    .Attr("partition_nums: int")
    .Attr("Tkeys: {int64, int32}")
    .Attr("dtype: type")
    .Doc(R"(
Input embedding variable its parition_id, import ids and values from other partion
    embedding variables.The first part is skipped. Load them according to new partition_num.
)");

REGISTER_OP(::des::kReAssign)
    .Input("old: Ref(T)")
    .Input("new: T")
    .Input("new_partition_nums: int32")
    .Output("output_ref: Ref(T)")
    .Attr("partition_id: int = 0")
    .Attr("device_id: int = 0")
    .Attr("partition_nums: int")
    .Attr("T: type")
    .Attr("use_locking: bool = true")
    .SetAllowsUninitializedInput()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    })
    .Doc(R"(
Input old partioned RefVariable and full Variables, Assign part of value according
    to  new_partition_nums and its partition_id.
)");

REGISTER_OP(::des::kReAssignRes)
    .Input("old_resource: resource")
    .Input("value: T")
    .Input("new_partition_nums: int32")
    .Attr("T: type")
    .Attr("partition_id: int = 0")
    .Attr("device_id: int = 0")
    .Attr("partition_nums: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return Status::OK();
    })
    .Doc(R"(
Input old ResourceVariable and full Variables, Assign part of value according
    to  new_partition_nums and its partition_id.
)");

}  // namespace tensorflow
