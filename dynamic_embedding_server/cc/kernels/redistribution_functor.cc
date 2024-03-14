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

#define EIGEN_USE_THREADS

#include "dynamic_embedding_server/include/ops/redistribution_functor.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace des {

namespace functor {

template <>
struct CustomScale<CPUDevice, string> {
  void operator()(const CPUDevice& d, typename TTypes<tstring>::Flat output,
                  typename TTypes<tstring>::ConstFlat rhs, int partition_id,
                  int partition_num, int offset) {
    if (output.dimension(0) == 1) {
      output.data()->resize(rhs.data()->size());
      auto work = [&output, &rhs](int64 start, int64 end) {
        memmove(const_cast<char*>(output.data()->data()) + start,
                rhs.data()->data() + start, end - start);
      };
      d.parallelFor(rhs.data()->size(),
                    Eigen::TensorOpCost(.1,  // chosen to force large chunks
                                        .1, 0),
                    work);
    } else {
      auto work = [&output, &rhs](int64 start, int64 end) {
        for (int i = start; i < end; ++i) {
          output.data()[i].resize(rhs.data()[i].size());
          memmove(const_cast<char*>(output.data()[i].data()),
                  rhs.data()[i].data(), rhs.data()[i].size());
        }
      };
      int64 estimated_string_size;
      if (rhs.size() > 0) {
        // first element of the tensor seems as good a guess as any of the sizes
        // of the strings contained within...
        estimated_string_size = std::max(rhs.data()[0].size(), sizeof(tstring));
      } else {
        estimated_string_size = sizeof(tstring);
      }
      d.parallelFor(
          offset,
          Eigen::TensorOpCost(estimated_string_size, estimated_string_size, 0),
          work);

      // offset
      auto copy_work = [&output, &rhs, offset](int64 start, int64 end) {
        for (int i = start; i < end; ++i) {
          output.data()[offset + i].resize(rhs.data()[i].size());
          memmove(const_cast<char*>(output.data()[offset + i].data()),
                  rhs.data()[i].data(), rhs.data()[i].size());
        }
      };
      if (rhs.size() > 0) {
        // first element of the tensor seems as good a guess as any of the sizes
        // of the strings contained within...
        estimated_string_size = std::max(rhs.data()[0].size(), sizeof(tstring));
      } else {
        estimated_string_size = sizeof(tstring);
      }
      d.parallelFor(
          rhs.dimension(0),
          Eigen::TensorOpCost(estimated_string_size, estimated_string_size, 0),
          copy_work);
    }
  }
};

template <typename T>
struct CustomScale<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat output,
                  typename TTypes<T>::ConstFlat rhs, int partition_id,
                  int partition_num, int offset) {
    if (output.dimension(0) == 1) {
      auto work = [&output, &rhs](int64 start, int64 end) {
        for (int i = start; i < end; ++i) {
          output.data()[i] = rhs.data()[i];
        }
      };
      d.parallelFor(rhs.size(),
                    Eigen::TensorOpCost(.1,  // chosen to force large chunks
                                        .1, 0),
                    work);
    } else {
      auto size = output.dimension(0);

      auto work = [&output, &rhs, offset](int64 start, int64 end) {
        for (int i = start; i < end; ++i) {
          output.data()[i] = rhs.data()[i + offset];
        }
      };
      int64 estimated_string_size = sizeof(T);
      d.parallelFor(
          size,
          Eigen::TensorOpCost(estimated_string_size, estimated_string_size, 0),
          work);
    }
  }
};

template <typename T>
struct CustomDenseUpdate<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::Flat update) {
    LOG(INFO) << " ===== ";
    params.device(d) = update;
  }
};

}  // end namespace functor

#define DEFINE_CPU_KERNELS(T)                         \
  template struct functor::CustomScale<CPUDevice, T>; \
  template struct functor::CustomDenseUpdate<CPUDevice, T>;
TF_CALL_NUMBER_TYPES(DEFINE_CPU_KERNELS);
#undef DEFINE_CPU_KERNELS

}  // namespace des

}  // namespace tensorflow