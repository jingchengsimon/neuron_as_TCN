#!/usr/bin/env python3
"""
PyTorch Activity Optimization 安装和测试脚本
"""

import os
import sys
import subprocess
import importlib

def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    version = sys.version_info
    print(f"  Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("  ❌ Python版本过低，需要3.7+")
        return False
    else:
        print("  ✅ Python版本符合要求")
        return True

def check_dependencies():
    """检查依赖包"""
    print("\n检查依赖包...")
    
    dependencies = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'psutil': 'psutil',
        'tensorflow': 'TensorFlow (可选，用于模型转换)'
    }
    
    missing = []
    available = []
    
    for package, name in dependencies.items():
        try:
            if package == 'torch':
                import torch
                version = torch.__version__
                device = 'CUDA' if torch.cuda.is_available() else 'CPU'
                print(f"  ✅ {name}: {version} ({device})")
            elif package == 'tensorflow':
                import tensorflow as tf
                version = tf.__version__
                print(f"  ✅ {name}: {version}")
            else:
                module = importlib.import_module(package)
                print(f"  ✅ {name}: 已安装")
            available.append(package)
        except ImportError:
            print(f"  ❌ {name}: 未安装")
            if package != 'tensorflow':  # TensorFlow是可选的
                missing.append(package)
    
    return missing, available

def install_dependencies(missing_packages):
    """安装缺失的依赖"""
    if not missing_packages:
        print("\n所有必需依赖已安装！")
        return True
    
    print(f"\n需要安装以下依赖: {', '.join(missing_packages)}")
    
    try:
        for package in missing_packages:
            print(f"\n正在安装 {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} 安装成功")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n测试基本功能...")
    
    try:
        # 测试核心模块导入
        from pytorch_activity_optimizer import PytorchActivityOptimizer
        print("  ✅ 核心优化器导入成功")
        
        # 测试模型转换器导入
        try:
            from model_converter import TCNModelConverter
            print("  ✅ 模型转换器导入成功")
        except ImportError as e:
            print(f"  ⚠ 模型转换器导入失败: {e}")
            print("    这通常是因为TensorFlow未安装，但不影响基本功能")
        
        # 测试测试运行器
        from run_tests import run_unit_tests
        print("  ✅ 测试运行器导入成功")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ 基本功能测试失败: {e}")
        return False

def run_unit_tests():
    """运行单元测试"""
    print("\n运行单元测试...")
    
    try:
        from run_tests import run_unit_tests
        success = run_unit_tests()
        
        if success:
            print("  ✅ 单元测试通过")
            return True
        else:
            print("  ❌ 单元测试失败")
            return False
            
    except Exception as e:
        print(f"  ❌ 运行测试失败: {e}")
        return False

def run_demo():
    """运行演示"""
    print("\n运行功能演示...")
    
    try:
        from demo import run_demo
        run_demo()
        print("  ✅ 演示运行成功")
        return True
        
    except Exception as e:
        print(f"  ❌ 演示运行失败: {e}")
        return False

def check_model_conversion():
    """检查模型转换功能"""
    print("\n检查模型转换功能...")
    
    try:
        from model_converter import TCNModelConverter
        print("  ✅ 模型转换器可用")
        
        # 检查是否有TensorFlow
        try:
            import tensorflow as tf
            print(f"  ✅ TensorFlow可用: {tf.__version__}")
            print("  ✅ 可以转换.h5模型到PyTorch格式")
            return True
        except ImportError:
            print("  ⚠ TensorFlow未安装")
            print("  ⚠ 无法转换.h5模型，但可以使用测试模型")
            return False
            
    except ImportError:
        print("  ❌ 模型转换器不可用")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("PyTorch Activity Optimization 安装和测试")
    print("=" * 60)
    
    # 检查Python版本
    if not check_python_version():
        print("\n❌ Python版本不符合要求，请升级到Python 3.7+")
        return False
    
    # 检查依赖
    missing, available = check_dependencies()
    
    # 安装缺失的依赖
    if missing:
        if not install_dependencies(missing):
            print("\n❌ 依赖安装失败")
            return False
    
    # 测试基本功能
    if not test_basic_functionality():
        print("\n❌ 基本功能测试失败")
        return False
    
    # 运行单元测试
    if not run_unit_tests():
        print("\n❌ 单元测试失败")
        return False
    
    # 运行演示
    if not run_demo():
        print("\n❌ 演示运行失败")
        return False
    
    # 检查模型转换功能
    conversion_available = check_model_conversion()
    
    print("\n" + "=" * 60)
    print("安装和测试完成！")
    print("=" * 60)
    
    if conversion_available:
        print("✅ 所有功能正常，包括模型转换")
        print("\n🚀 现在可以：")
        print("  1. 运行主程序: python main.py")
        print("  2. 转换模型: python model_converter.py model.h5")
        print("  3. 批量转换: python quick_convert.py /path/to/models")
    else:
        print("✅ 基本功能正常，但模型转换不可用")
        print("\n🚀 现在可以：")
        print("  1. 运行主程序: python main.py (使用测试模型)")
        print("  2. 运行演示: python demo.py")
        print("  3. 运行测试: python run_tests.py")
        print("\n💡 如需模型转换功能，请安装TensorFlow:")
        print("  pip install tensorflow")
    
    print("\n📖 更多信息请查看:")
    print("  - README.md: 项目概述")
    print("  - QUICK_START.md: 快速开始")
    print("  - MODEL_CONVERSION_GUIDE.md: 模型转换指南")
    print("  - SOLUTION_SUMMARY.md: 完整解决方案")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

