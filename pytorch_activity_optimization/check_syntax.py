#!/usr/bin/env python3
"""
PyTorch Activity Optimization 语法检查脚本
"""

import ast
import os
import sys
from pathlib import Path

def check_python_file(file_path):
    """检查单个Python文件的语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 尝试解析AST
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    except Exception as e:
        return False, f"其他错误: {e}"

def check_all_python_files():
    """检查所有Python文件"""
    current_dir = Path(__file__).parent
    python_files = list(current_dir.rglob("*.py"))
    
    print("=" * 60)
    print("PyTorch Activity Optimization 语法检查")
    print("=" * 60)
    print(f"检查目录: {current_dir}")
    print(f"找到 {len(python_files)} 个Python文件")
    print()
    
    all_passed = True
    failed_files = []
    
    for file_path in python_files:
        # 跳过__pycache__目录
        if "__pycache__" in str(file_path):
            continue
            
        relative_path = file_path.relative_to(current_dir)
        success, error = check_python_file(file_path)
        
        if success:
            print(f"✓ {relative_path}")
        else:
            print(f"✗ {relative_path}: {error}")
            all_passed = False
            failed_files.append((relative_path, error))
    
    print()
    print("=" * 60)
    if all_passed:
        print("所有Python文件语法检查通过！✓")
    else:
        print(f"发现 {len(failed_files)} 个语法错误:")
        for file_path, error in failed_files:
            print(f"  - {file_path}: {error}")
    print("=" * 60)
    
    return all_passed

def check_imports():
    """检查导入语句"""
    print("\n检查导入语句...")
    
    # 检查核心模块的导入
    core_files = [
        "pytorch_activity_optimizer.py",
        "main.py",
        "demo.py",
        "run_tests.py"
    ]
    
    for filename in core_files:
        file_path = Path(filename)
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 提取导入语句
                tree = ast.parse(content)
                imports = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        for alias in node.names:
                            imports.append(f"{module}.{alias.name}")
                
                print(f"  {filename}: {', '.join(imports[:5])}{'...' if len(imports) > 5 else ''}")
                
            except Exception as e:
                print(f"  {filename}: 检查失败 - {e}")

def main():
    """主函数"""
    # 检查语法
    syntax_ok = check_all_python_files()
    
    # 检查导入
    check_imports()
    
    # 检查文件结构
    print("\n检查项目结构...")
    current_dir = Path(__file__).parent
    
    expected_files = [
        "pytorch_activity_optimizer.py",
        "main.py",
        "demo.py",
        "run_tests.py",
        "README.md",
        "requirements.txt",
        "install_dependencies.sh",
        "install_dependencies.bat"
    ]
    
    expected_dirs = [
        "tests",
        "tests/unit",
        "tests/integration", 
        "tests/performance"
    ]
    
    print("  核心文件:")
    for file in expected_files:
        if (current_dir / file).exists():
            print(f"    ✓ {file}")
        else:
            print(f"    ✗ {file} (缺失)")
    
    print("  目录结构:")
    for dir_path in expected_dirs:
        if (current_dir / dir_path).is_dir():
            print(f"    ✓ {dir_path}/")
        else:
            print(f"    ✗ {dir_path}/ (缺失)")
    
    print("\n项目结构检查完成！")
    
    return 0 if syntax_ok else 1

if __name__ == '__main__':
    sys.exit(main())

