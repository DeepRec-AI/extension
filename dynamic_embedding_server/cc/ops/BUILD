cc_library(
    name = "ops",
    deps = [
        ":elastic_partition_op",
        ":redistribution_op",
    ],
    alwayslink=1,
)

cc_library(
    name = "elastic_partition_op",
    srcs = ["elastic_partition_op.cc",],
    deps = [
        "@org_tensorflow//tensorflow/core/kernels:bounds_check",
        "@org_tensorflow//tensorflow/core/kernels:dense_update_functor",
        "@org_tensorflow//tensorflow/core/kernels:gather_functor",
        "@org_tensorflow//tensorflow/core/kernels:save_restore_tensor",
        "@org_tensorflow//tensorflow/core/kernels:scatter_functor",
        "@org_tensorflow//tensorflow/core/kernels:state",
        "@org_tensorflow//tensorflow/core/kernels:training_op_helpers",
        "@org_tensorflow//tensorflow/core/kernels:variable_ops",
        "@org_tensorflow//tensorflow/core/kernels:embedding_var_cu_cc",
        "@org_tensorflow//tensorflow/core:embedding_gpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:kv_variable_ops_op_lib",
        "@org_tensorflow//tensorflow/core/util/tensor_bundle",
        "@sparsehash_c11//:dense_hash_map",
        "@libcuckoo//:libcuckoo",
        "@com_github_google_leveldb//:leveldb",
        "@cuCollections//:cuco_hash_table"
    ],
    alwayslink=1,
)

cc_library(
    name = "redistribution_op",
    hdrs = ["redistribution_functor.h"],
    srcs = ["redistribution_ops.cc",
            "redistribution_functor.cc"],
    deps = [
        "@org_tensorflow//tensorflow/core/kernels:bounds_check",
        "@org_tensorflow//tensorflow/core/kernels:dense_update_functor",
        "@org_tensorflow//tensorflow/core/kernels:gather_functor",
        "@org_tensorflow//tensorflow/core/kernels:save_restore_tensor",
        "@org_tensorflow//tensorflow/core/kernels:scatter_functor",
        "@org_tensorflow//tensorflow/core/kernels:state",
        "@org_tensorflow//tensorflow/core/kernels:training_op_helpers",
        "@org_tensorflow//tensorflow/core/kernels:variable_ops",
        "@org_tensorflow//tensorflow/core/kernels:embedding_var_cu_cc",
        "@org_tensorflow//tensorflow/core:embedding_gpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:kv_variable_ops_op_lib",
        "@org_tensorflow//tensorflow/core/util/tensor_bundle",
        "@sparsehash_c11//:dense_hash_map",
        "@libcuckoo//:libcuckoo",
        "@com_github_google_leveldb//:leveldb",
        "@cuCollections//:cuco_hash_table"
    ],
    alwayslink=1,
)