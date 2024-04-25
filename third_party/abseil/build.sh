set -o errexit

cd abseil
git submodule update --init
cd ..

mkdir -p build
prefix=`pwd`/build
cd build

cmake ../abseil \
  -DCMAKE_INSTALL_PREFIX=${prefix} \
  -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 -fPIC" \
  -DCMAKE_CXX_STANDARD=11
make -j10
make install

if [ -d lib64 ]; then
  mv lib64 lib
fi
find ./absl -name "*.o" |  xargs ar cru libabsl.a
mv libabsl.a lib