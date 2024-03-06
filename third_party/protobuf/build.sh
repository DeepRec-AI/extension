mkdir -p build
prefix=`pwd`/build


CXXFLAGS+=" -fPIC"

cd protobuf/
git submodule update --init --recursive
./autogen.sh
./configure --prefix=${prefix} --disable-shared CXXFLAGS="${CXXFLAGS}"
make -j10
make install
