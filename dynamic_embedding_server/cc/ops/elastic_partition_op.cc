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

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP(::des::kElasticPartition)
    .Input("data: TKey")
    .Input("indices: int32")
    .Output("p_data: num_partitions * TKey")
    .Output("p_indices: num_partitions * int32")
    .Attr("num_partitions: int")
    .Attr("TKey: {int64, int32}")
    .Attr("partition_strategy: {'bucket', 'mod', 'div'}")
    .SetShapeFn([](InferenceContext* c) {
      int64 num_partitions;
      TF_RETURN_IF_ERROR(c->GetAttr("num_partitions", &num_partitions));

      ShapeHandle data_shape = c->input(0);

      // The partition shape is dynamic in the 0th dimension, and matches
      // data_shape in the remaining dimensions.
      ShapeHandle unknown_dim0 = c->MakeShape({c->UnknownDim()});

      const int64 rank = c->Rank(data_shape);
      ShapeHandle data_suffix_shape;
      TF_RETURN_IF_ERROR(c->Subshape(data_shape, rank, &data_suffix_shape));
      ShapeHandle result_shape;
      TF_RETURN_IF_ERROR(
          c->Concatenate(unknown_dim0, data_suffix_shape, &result_shape));

      for (int i = 0; i < num_partitions; ++i) {
        c->set_output(i, result_shape);
      }

      ShapeHandle indices_shape = c->input(1);
      const int64 id_rank = c->Rank(indices_shape);
      ShapeHandle indices_suffix_shape;
      TF_RETURN_IF_ERROR(
          c->Subshape(indices_shape, id_rank, &indices_suffix_shape));
      ShapeHandle indices_result_shape;
      TF_RETURN_IF_ERROR(c->Concatenate(unknown_dim0, indices_suffix_shape,
                                        &indices_result_shape));
      for (int i = 0; i < num_partitions; ++i) {
        c->set_output(i + num_partitions, indices_result_shape);
      }

      return Status::OK();
    });
}  // namespace tensorflow