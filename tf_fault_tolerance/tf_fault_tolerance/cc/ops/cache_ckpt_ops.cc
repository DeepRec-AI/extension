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

#include "tf_fault_tolerance/cc/ops/cache_ckpt_ops.h"

namespace tensorflow {

using shape_inference::ShapeHandle;
using shape_inference::InferenceContext;

REGISTER_OP("GenerateCacheCKPT")
    .Input("ckpt_path_prefix: string")
    .Input("cache_path: string")
    .Input("shard: int32")
    .Input("num_shards: int32")
    .Output("ckpt_key: string")
    .Output("ckpt_meta: string")
    .Output("ckpt_data: string")
    .Attr("is_merged_meta: bool = true")
    .Attr("ckpt_storage_type: string = ''")
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
      c->set_output(2, c->Scalar());

      return Status::OK();
    });

namespace CacheCKPTOp {
namespace GenerateCacheCKPTOp {
int MetaCKPTOutputIdx = 1;
int DataCKPTOutputIdx = 2;
} // End of namespace GenerateCacheCKPTOp
} // End of namespace CacheCKPTOp

REGISTER_OP("BackupRemoteCacheCKPT")
    .Input("cache_path: string")
    .Input("ckpt_key: string")
    .Input("ckpt_meta: string")
    .Input("ckpt_data: string")
    .Attr("is_merged_meta: bool = true")
    .Attr("ckpt_storage_type: string = ''")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ShapeHandle unused;

      // validate all input
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));

      return Status::OK();
    });

namespace CacheCKPTOp {
namespace BackupRemoteCacheCKPTOp {
int MetaCKPTInputIdx = 2;
int DataCKPTInputIdx = 3;
} // End of namespace BackupRemoteCacheCKPTOp
} // End of namespace CacheCKPTOp

REGISTER_OP("CheckLocalCacheCKPT")
    .Input("ckpt_path_prefix: string")
    .Input("shard: int32")
    .Input("num_shards: int32")
    .Output("exist_cache_ckpt: bool")
    .Output("ckpt: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = 'cache_ckpt'")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ShapeHandle unused;

      // Validate all input
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));

      // Set output shape
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());

      return Status::OK();
    });

REGISTER_OP("GetRemoteCacheCKPT")
    .Input("ckpt_path_prefix: string")
    .Input("cache_path: string")
    .Input("exist_cache_ckpt: bool")
    .Input("ckpt_key: string")
    .Input("ckpt_meta: string")
    .Input("ckpt_data: string")
    .Output("exist_cache_ckpt_out: bool")
    .Output("ckpt: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = 'cache_ckpt'")
    .Attr("is_merged_meta: bool = true")
    .Attr("ckpt_storage_type: string = ''")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ShapeHandle unused;

      // Validate all input
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));

      // Set output shape
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());

      return Status::OK();
    });

namespace CacheCKPTOp {
namespace GetRemoteCacheCKPTOp {
int MetaCKPTInputIdx = 4;
int DataCKPTInputIdx = 5;
} // End of namespace GetRemoteCacheCKPTOp
} // End of namespace CacheCKPTOp

REGISTER_OP("RepatriateRemoteCacheCKPT")
    .Input("ckpt_path_prefix: string")
    .Input("shard: int32")
    .Input("num_shards: int32")
    .Output("exist_cache_ckpt: bool")
    .Output("ckpt_key: string")
    .Output("cache_ckpt_meta: string")
    .Output("cache_ckpt_data: string")
    .Attr("output_is_path: bool = true")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ShapeHandle unused;

      // Validate all input
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));

      //Set output shape
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      c->set_output(3, c->Scalar());

      return Status::OK();
    });

namespace CacheCKPTOp {
namespace RepatriateRemoteCacheCKPTOp {
int MetaCKPTOutputIdx = 2;
int DataCKPTOutputIdx = 3;
} // End of namespace RepatriateRemoteCacheCKPTOp
} // End of namespace CacheCKPTOp

REGISTER_OP("LoadCKPTFromFilePath")
    .Input("ckpt_path_prefix: string")
    .Input("shard: int32")
    .Input("num_shards: int32")
    .Output("cache_ckpt: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = 'cache_ckpt'")
    .Attr("output_is_path: bool = true")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ShapeHandle unused;

      // Validate all input
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));

      //Set output shape
      c->set_output(0, c->Scalar());

      return Status::OK();
    });

REGISTER_OP("UnPackCacheCKPTResource")
    .Input("ckpt: resource")
    .Output("ckpt_meta: string")
    .Output("ckpt_data: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ShapeHandle unused;

      // Validate all input
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));

      //Set output shape
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());

      return Status::OK();
    });

} // End of namespace tensorflow
