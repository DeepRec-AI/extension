mkdir -p build
prefix=`pwd`/build


CXXFLAGS="-fPIC -D_GLIBCXX_USE_CXX11_ABI=0"

pushd protobuf

numeric_tf_version="$TF_VERSION"
if [ "$numeric_tf_version" -eq 112 ]; then
    echo "using protobuf 3.6.0"
    git checkout v3.6.0
else
    echo "using protobuf 3.8.0"
    git checkout v3.8.0
fi

./autogen.sh
./configure --prefix=${prefix} --disable-shared CXXFLAGS="${CXXFLAGS}"
make -j10
make install
popd
