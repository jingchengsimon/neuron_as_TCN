# PyTorch Activity Optimization 快速启动指南

## 🚀 5分钟快速开始

### 1. 环境检查
```bash
# 检查Python版本（需要3.7+）
python3 --version

# 检查项目结构
python3 check_syntax.py
```

### 2. 安装依赖

#### Linux/macOS:
```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

#### Windows:
```cmd
install_dependencies.bat
```

#### 手动安装:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install numpy matplotlib psutil
```

### 3. 运行演示
```bash
python3 demo.py
```

### 4. 运行测试
```bash
# 运行所有测试
python3 run_tests.py

# 运行特定测试
python3 run_tests.py --unit
python3 run_tests.py --integration
python3 run_tests.py --performance
```

### 5. 运行主程序
```bash
python3 main.py
```

## 🔧 常见问题解决

### 问题1: 权限被拒绝
```bash
chmod +x install_dependencies.sh
```

### 问题2: PyTorch未安装
```bash
pip3 install torch torchvision torchaudio
```

### 问题3: 依赖缺失
```bash
pip3 install -r requirements.txt
```

### 问题4: 测试失败
```bash
# 先运行语法检查
python3 check_syntax.py

# 再运行测试
python3 run_tests.py --unit
```

## 📁 项目文件说明

| 文件 | 用途 | 状态 |
|------|------|------|
| `pytorch_activity_optimizer.py` | 核心优化器类 | ✅ 核心功能 |
| `main.py` | 主程序入口 | ✅ 完整实现 |
| `demo.py` | 功能演示 | ✅ 可运行 |
| `run_tests.py` | 测试运行器 | ✅ 完整测试 |
| `check_syntax.py` | 语法检查 | ✅ 验证通过 |
| `install_dependencies.sh` | Linux/macOS安装脚本 | ✅ 可执行 |
| `install_dependencies.bat` | Windows安装脚本 | ✅ 可执行 |

## 🎯 下一步

1. **熟悉代码结构**：阅读 `README.md` 和 `PROJECT_SUMMARY.md`
2. **运行演示**：执行 `python3 demo.py` 了解功能
3. **运行测试**：执行 `python3 run_tests.py` 验证功能
4. **修改配置**：根据需要调整 `main.py` 中的参数
5. **集成使用**：将优化器集成到你的项目中

## 📞 需要帮助？

- 查看 `README.md` 获取详细文档
- 查看 `PROJECT_SUMMARY.md` 了解项目详情
- 运行 `python3 check_syntax.py` 检查代码状态
- 运行 `python3 demo.py` 查看功能演示

---

**祝使用愉快！** 🎉

