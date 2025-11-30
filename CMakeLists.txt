cmake_minimum_required(VERSION 3.16)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============== 平台识别逻辑 ===============
# 默认为本地 x86；若使用 toolchain（如 aarch64），CMAKE_SYSTEM_PROCESSOR 通常为 aarch64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(TARGET_PLATFORM "arm64")
else()
    set(TARGET_PLATFORM "x86_64")
endif()

message(STATUS "Target platform detected: ${TARGET_PLATFORM}")

# =============== 第三方库路径（平台自适应） ===============
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)

# 根据平台选择子目录
if(TARGET_PLATFORM STREQUAL "arm64")
    set(OPENCV_INSTALL_DIR ${THIRD_PARTY_DIR}/opencv-install-arm64)
    set(MNN_INSTALL_DIR ${THIRD_PARTY_DIR}/mnn-install-arm64)
else()
    set(OPENCV_INSTALL_DIR ${THIRD_PARTY_DIR}/opencv-install-x86)
    set(MNN_INSTALL_DIR ${THIRD_PARTY_DIR}/mnn-install-x86)
endif()

# =============== Eigen ===============
set(EIGEN3_INCLUDE_DIR ${THIRD_PARTY_DIR}/eigen)
add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE ${EIGEN3_INCLUDE_DIR})

# =============== OpenCV ===============
set(OpenCV_DIR ${OPENCV_INSTALL_DIR}/lib/cmake/opencv4)
find_package(OpenCV REQUIRED NO_MODULE)

# 检查 videoio 模块（可选）
if(NOT TARGET opencv_videoio)
    message(WARNING "OpenCV videoio module not found")
endif()

# =============== MNN ===============
add_library(mnn STATIC IMPORTED)
set_target_properties(mnn PROPERTIES
    IMPORTED_LOCATION ${MNN_INSTALL_DIR}/lib/libMNN.a
    INTERFACE_INCLUDE_DIRECTORIES ${MNN_INSTALL_DIR}/include
)

# =============== 收集源文件 ===============
file(GLOB_RECURSE SRC_FILES
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

# =============== 1. 内部核心库 ===============
add_library(_all_core STATIC ${SRC_FILES})
target_include_directories(_all_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(_all_core
    PUBLIC
        ${OpenCV_LIBS}
        mnn
        eigen
)

# =============== 2. 自包含静态库（平台自适应路径） ===============
set(OPENCV_STATIC_DIR ${OPENCV_INSTALL_DIR}/lib)
file(GLOB OPENCV_STATIC_LIBS ${OPENCV_STATIC_DIR}/libopencv_*.a)

if(OPENCV_STATIC_LIBS)
    message(STATUS "Found OpenCV static libs (${TARGET_PLATFORM}): ${OPENCV_STATIC_LIBS}")

    set(MYPROJECT_FULL_LIB "${CMAKE_BINARY_DIR}/libmyproject_core.a")

    add_custom_command(
        OUTPUT "${MYPROJECT_FULL_LIB}"
        COMMAND ${CMAKE_AR} rcs "${MYPROJECT_FULL_LIB}"
            $<TARGET_FILE:_all_core>
            ${OPENCV_STATIC_LIBS}
            ${MNN_INSTALL_DIR}/lib/libMNN.a
        DEPENDS _all_core
        COMMENT "Creating self-contained libmyproject_core.a for ${TARGET_PLATFORM}"
    )

    add_custom_target(myproject_core ALL
        DEPENDS "${MYPROJECT_FULL_LIB}"
    )
else()
    message(WARNING "OpenCV static libs not found at: ${OPENCV_STATIC_DIR}")
    add_custom_target(myproject_core)
endif()

# =============== 3. 安装规则 ===============
include(GNUInstallDirs)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/myproject
    FILES_MATCHING PATTERN "*.h*"
)

if(TARGET myproject_core AND EXISTS "${MYPROJECT_FULL_LIB}")
    install(FILES "${MYPROJECT_FULL_LIB}"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RENAME libmyproject_core.a
    )
endif()

# =============== 4. 测试可执行文件 ===============
function(add_test_executable TEST_SOURCE)
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE
        _all_core
        -static-libstdc++
        -static-libgcc
    )
endfunction()

file(GLOB TEST_SOURCES
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp
)

foreach(TEST_SRC ${TEST_SOURCES})
    add_test_executable(${TEST_SRC})
endforeach()