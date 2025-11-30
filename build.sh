#!/bin/bash

set -e  # 出错立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# === 配置区（请根据你的实际路径修改）===
TOOLCHAIN_ARM64="/project/arm64_build/toolchain-aarch64.cmake"

# 检查 ARM64 toolchain 是否存在
if [[ ! -f "$TOOLCHAIN_ARM64" ]]; then
    echo "⚠️  ARM64 toolchain not found at: $TOOLCHAIN_ARM64"
    echo "   Please adjust TOOLCHAIN_ARM64 in this script or create the file."
    exit 1
fi

# === 交互选择 ===
echo "请选择目标平台："
echo "1) x86_64 (本地开发)"
echo "2) aarch64 (ARM64 交叉编译)"
read -p "输入选项 [1/2]: " choice

case "$choice" in
    1)
        BUILD_DIR="build_x86"
        CMAKE_EXTRA_ARGS=""
        rm build_x86 -rf
        echo "🔧 正在配置 x86_64 构建..."
        ;;
    2)
        BUILD_DIR="build_arm64"
        CMAKE_EXTRA_ARGS="-DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_ARM64"
        rm build_arm64 -rf
        echo "🔧 正在配置 aarch64 (ARM64) 构建..."
        ;;
    *)
        echo "❌ 无效选项，请输入 1 或 2"
        exit 1
        ;;
esac

# === 执行构建 ===
BUILD_PATH="$PROJECT_ROOT/$BUILD_DIR"

# 创建/进入构建目录
mkdir -p "$BUILD_PATH"
cd "$BUILD_PATH"

# 调用 CMake
cmake "$PROJECT_ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    $CMAKE_EXTRA_ARGS

# 编译
make -j$(nproc)

echo "✅ 构建完成！输出目录：$BUILD_PATH"