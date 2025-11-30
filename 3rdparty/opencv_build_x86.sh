cd opencv && mkdir build_x86 && cd build_x86
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/../../opencv-install-x86 \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  \
  -DBUILD_opencv_apps=OFF \
  -DBUILD_opencv_highgui=OFF \
  -DBUILD_opencv_videoio=ON \
  -DBUILD_opencv_dnn=ON \
  \
  -DWITH_FFMPEG=OFF \
  -DWITH_V4L=OFF \
  -DWITH_OPENCL=OFF \
  -DWITH_GSTREAMER=OFF \
  -DWITH_OPENMP=OFF \
  -DWITH_OPENJPEG=OFF \
  -DWITH_TIFF=OFF \
  -DWITH_WEBP=OFF \
  \
  -DBUILD_ZLIB=ON \
  -DBUILD_JPEG=ON \
  -DBUILD_PNG=ON \
  \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_TESTS=OFF \
  -DCPU_BASELINE="NEON" \
  -DCPU_DISPATCH=""
make -j$(nproc) && make install