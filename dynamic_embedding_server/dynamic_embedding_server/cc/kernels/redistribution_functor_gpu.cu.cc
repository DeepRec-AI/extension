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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "dynamic_embedding_server/include/ops/redistribution_functor.h"

#include "tensorflow/core/framework/register_types.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

namespace des {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T>
struct CustomScale<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat output,
                  typename TTypes<T>::ConstFlat rhs, int partition_id,
                  int partition_num, int offset) {
    for (int i = 0; i < output.dimension(0); ++i) {
      output.data()[i] = rhs.data()[i + offset];
    }
  }
};

template <typename T>
struct CustomDenseUpdate<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) = update;
  }
};

}  // end namespace functor

#define DEFINE_GPU_KERNELS(T)                               \
  template struct functor::CustomDenseUpdate<GPUDevice, T>; \
  template struct functor::CustomScale<GPUDevice, T>;
TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);
TF_CALL_int32(DEFINE_GPU_KERNELS);
TF_CALL_int64(DEFINE_GPU_KERNELS);
TF_CALL_int8(DEFINE_GPU_KERNELS);
#undef DEFINE_GPU_KERNELS

}  // end namespace des

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
