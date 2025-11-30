cd MNN
mkdir -p build_x86
cd build_x86

# 配置 CMake（包含转换工具和量化工具）
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/../../mnn-install-x86 \
  -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_SHARED_LIBS=OFF \
  -DMNN_BUILD_TRAIN=ON \
  -DMNN_BUILD_CONVERTER=ON \
  -DMNN_BUILD_QUANTOOLS=ON \
  -DMNN_BUILD_TOOLS=ON \
  -DMNN_BUILD_TEST=OFF \
  -DMNN_BUILD_BENCHMARK=OFF

# 编译
make -j$(nproc) && make install