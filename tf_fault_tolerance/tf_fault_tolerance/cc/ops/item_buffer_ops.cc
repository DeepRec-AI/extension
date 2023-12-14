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

#ifndef TF_FAULT_TOLERANCE_CC_OPS_ITEM_BUFFER_OPS_H_
#define TF_FAULT_TOLERANCE_CC_OPS_ITEM_BUFFER_OPS_H_

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("ItemBufferPut")
    .Input("record: dtype")
    .Attr("container: string = ''")
    .Attr("dtype: type")
    .Attr("shared_name: string = ''")
    .Attr("is_overwritable: bool = true")
    .Attr("timeout_millis: int >= 1 = 1000")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("ItemBufferTake")
    .Output("record: dtype")
    .Attr("container: string = ''")
    .Attr("dtype: type")
    .Attr("shared_name: string = ''")
    .Attr("is_overwritable: bool = true")
    .Attr("shared_threads: int >= 1 = 1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("ItemBufferSetState")
    .Attr("container: string = ''")
    .Attr("is_cancelled: bool = true")
    .Attr("shared_name: string = ''")
    .Attr("is_overwritable: bool = true")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

} // End of namespace tensorflow

#endif // End of TF_FAULT_TOLERANCE_CC_OPS_ITEM_BUFFER_OPS_H_
