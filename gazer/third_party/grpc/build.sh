mkdir -p build
install_dir=`pwd`/build

cd grpc

# switch grpc to the version
numeric_tf_version="$TF_VERSION"
if [ "$numeric_tf_version" -eq 112 ]; then
    echo "using grpc 1.13.0"
    git checkout v1.13.0
else
    echo "using grpc 1.19.1"
    git checkout v1.19.1
    
fi


export CXXFLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
option=-D_GLIBCXX_USE_CXX11_ABI=0
make all EXTRA_CXXFLAGS=${option} EXTRA_CPPFLAGS=${option} EXTRA_CFLAGS=${option} EXTRA_LDFLAGS=${option} PROTOBUF_CONFIG_OPTS=CXXFLAGS=${option} -j10

make install-headers prefix=${install_dir}
make install-static prefix=${install_dir}
make install-plugins prefix=${install_dir}
