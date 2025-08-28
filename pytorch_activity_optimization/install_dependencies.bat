@echo off
echo === PyTorch Activity Optimization 依赖安装脚本 ===
echo.

REM 检查Python版本
echo 检查Python版本...
python --version
if %errorlevel% neq 0 (
    echo 错误: 未找到Python，请先安装Python 3.7+
    pause
    exit /b 1
)

REM 检查pip
echo.
echo 检查pip...
python -m pip --version
if %errorlevel% neq 0 (
    echo 错误: 未找到pip，请先安装pip
    pause
    exit /b 1
)

REM 安装PyTorch
echo.
echo 安装PyTorch...
echo 注意: 这将安装CPU版本的PyTorch。如果需要GPU支持，请访问 https://pytorch.org/ 获取安装命令

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

REM 安装其他依赖
echo.
echo 安装其他依赖...
python -m pip install numpy matplotlib

REM 尝试安装psutil（可选）
echo.
echo 安装psutil（可选，用于内存监控）...
python -m pip install psutil

echo.
echo === 依赖安装完成 ===
echo.
echo 现在可以运行以下命令来测试安装:
echo   python demo.py
echo   python run_tests.py
echo.
echo 如果遇到问题，请检查:
echo   1. Python版本是否为3.7+
echo   2. pip是否正确安装
echo   3. 网络连接是否正常
echo.
echo 对于GPU支持，请访问: https://pytorch.org/get-started/locally/
echo.
pause

