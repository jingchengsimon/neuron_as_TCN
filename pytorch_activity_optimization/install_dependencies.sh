#!/bin/bash

echo "=== PyTorch Activity Optimization 依赖安装脚本 ==="
echo ""

# 检查Python版本
echo "检查Python版本..."
python_version=$(python3 --version 2>&1)
echo "Python版本: $python_version"

# 检查pip
echo ""
echo "检查pip..."
if command -v pip3 &> /dev/null; then
    echo "pip3 已安装"
    pip_cmd="pip3"
elif command -v pip &> /dev/null; then
    echo "pip 已安装"
    pip_cmd="pip"
else
    echo "错误: 未找到pip，请先安装pip"
    exit 1
fi

# 安装PyTorch
echo ""
echo "安装PyTorch..."
echo "注意: 这将安装CPU版本的PyTorch。如果需要GPU支持，请访问 https://pytorch.org/ 获取安装命令"

# 根据操作系统选择安装命令
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "检测到Linux系统，安装PyTorch..."
    $pip_cmd install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "检测到macOS系统，安装PyTorch..."
    $pip_cmd install torch torchvision torchaudio
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows
    echo "检测到Windows系统，安装PyTorch..."
    $pip_cmd install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "未知操作系统，尝试通用安装..."
    $pip_cmd install torch torchvision torchaudio
fi

# 安装其他依赖
echo ""
echo "安装其他依赖..."
$pip_cmd install numpy matplotlib

# 尝试安装psutil（可选）
echo ""
echo "安装psutil（可选，用于内存监控）..."
$pip_cmd install psutil

echo ""
echo "=== 依赖安装完成 ==="
echo ""
echo "现在可以运行以下命令来测试安装:"
echo "  python3 demo.py"
echo "  python3 run_tests.py"
echo ""
echo "如果遇到问题，请检查:"
echo "  1. Python版本是否为3.7+"
echo "  2. pip是否正确安装"
echo "  3. 网络连接是否正常"
echo ""
echo "对于GPU支持，请访问: https://pytorch.org/get-started/locally/"

