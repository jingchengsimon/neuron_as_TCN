#!/usr/bin/env python3
"""
PyTorch Activity Optimization 测试运行器
"""

import os
import sys
import unittest
import argparse
from pathlib import Path

def run_unit_tests():
    """运行单元测试"""
    print("=" * 60)
    print("运行单元测试...")
    print("=" * 60)
    
    # 添加当前目录到路径
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # 发现并运行单元测试
    loader = unittest.TestLoader()
    start_dir = current_dir / 'tests' / 'unit'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_integration_tests():
    """运行集成测试"""
    print("=" * 60)
    print("运行集成测试...")
    print("=" * 60)
    
    # 添加当前目录到路径
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # 发现并运行集成测试
    loader = unittest.TestLoader()
    start_dir = current_dir / 'tests' / 'integration'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_performance_tests():
    """运行性能测试"""
    print("=" * 60)
    print("运行性能测试...")
    print("=" * 60)
    
    # 添加当前目录到路径
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # 发现并运行性能测试
    loader = unittest.TestLoader()
    start_dir = current_dir / 'tests' / 'performance'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行所有测试...")
    print("=" * 60)
    
    # 添加当前目录到路径
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # 发现并运行所有测试
    loader = unittest.TestLoader()
    start_dir = current_dir / 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_specific_test(test_path):
    """运行特定的测试文件"""
    print(f"=" * 60)
    print(f"运行特定测试: {test_path}")
    print("=" * 60)
    
    # 添加当前目录到路径
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # 运行特定测试
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_path)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PyTorch Activity Optimization 测试运行器')
    parser.add_argument('--unit', action='store_true', help='运行单元测试')
    parser.add_argument('--integration', action='store_true', help='运行集成测试')
    parser.add_argument('--performance', action='store_true', help='运行性能测试')
    parser.add_argument('--all', action='store_true', help='运行所有测试')
    parser.add_argument('--test', type=str, help='运行特定的测试文件（相对于tests目录的路径）')
    
    args = parser.parse_args()
    
    # 如果没有指定参数，默认运行所有测试
    if not any([args.unit, args.integration, args.performance, args.all, args.test]):
        args.all = True
    
    success = True
    
    try:
        if args.unit:
            success &= run_unit_tests()
        
        if args.integration:
            success &= run_integration_tests()
        
        if args.performance:
            success &= run_performance_tests()
        
        if args.all:
            success &= run_all_tests()
        
        if args.test:
            success &= run_specific_test(args.test)
    
    except Exception as e:
        print(f"测试运行出错: {e}")
        success = False
    
    # 输出结果
    print("=" * 60)
    if success:
        print("所有测试通过！✓")
    else:
        print("部分测试失败！✗")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())

