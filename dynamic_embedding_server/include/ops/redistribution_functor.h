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

#ifndef DYNAMIC_EMBEDDING_SERVER_INCLUDE_OPS_REDISTRIBUTION_FUNCTOR_H_
#define DYNAMIC_EMBEDDING_SERVER_INCLUDE_OPS_REDISTRIBUTION_FUNCTOR_H_

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace des {

namespace functor {

template <typename Device, typename T>
struct CustomScale {
  void operator()(const Device& d, typename TTypes<T>::Flat output,
                  typename TTypes<T>::ConstFlat rhs, int partition_id,
                  int partition_num, int offset);
};

template <typename Device, typename T>
struct CustomDenseUpdate {
  void operator()(const Device& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::Flat update) {
    params.device(d) = update;
  }
};

}  // end namespace functor

} // end namespace des

}  // end namespace tensorflow

#endif  // DYNAMIC_EMBEDDING_SERVER_INCLUDE_OPS_REDISTRIBUTION_FUNCTOR_H_
