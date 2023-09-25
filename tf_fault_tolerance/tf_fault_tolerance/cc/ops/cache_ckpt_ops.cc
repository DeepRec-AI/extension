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
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::ShapeHandle;
using shape_inference::InferenceContext;

REGISTER_OP("GenerateCacheCKPT")
    .Input("ckpt_prefix: string")
    .Input("cache_ckpt_path: string")
    .Input("shard: int32")
    .Input("num_shards: int32")
    .Output("cache_ckpt_key: string")
    .Output("cache_ckpt: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ShapeHandle unused;

      // Validate all input
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));

      // Set output shape
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());

      return Status::OK();
    });

REGISTER_OP("RecvRemoteCacheCKPT")
    .Input("cache_ckpt_key: string")
    .Input("cache_ckpt: string")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ShapeHandle unused;

      // validate all input
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));

      return Status::OK();
    });


} // End of namespace tensorflow
