mkdir -p build
install_dir=`pwd`/build
 
# switch grpc to the version
cd grpc
git submodule update --init
 
 
option=${CXX_CFLAGS}
eval "option=$option"
make all EXTRA_CXXFLAGS=${option} EXTRA_CPPFLAGS=${option} EXTRA_CFLAGS=${option} EXTRA_LDFLAGS=${option} PROTOBUF_CONFIG_OPTS=CXXFLAGS=${option} -j10
 
make install-headers prefix=${install_dir}
make install-static prefix=${install_dir}
make install-plugins prefix=${install_dir}
