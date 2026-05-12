#!/usr/bin/env python3
"""
GPUMonitor类使用示例
演示如何使用GPUMonitor类进行GPU监控
"""

import time
import sys
import os

# 添加当前目录到路径，以便导入GPUMonitor类
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from train_and_analyze import GPUMonitor
except ImportError:
    print("Error: Cannot import GPUMonitor from train_and_analyze.py")
    sys.exit(1)

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 创建GPU监控器
    gpu_monitor = GPUMonitor()
    
    if not gpu_monitor.available:
        print("GPU监控不可用")
        return
    
    # 获取基本信息
    print("GPU基本信息:")
    gpu_monitor.print_status("  ")
    
    # 获取详细信息
    info = gpu_monitor.get_comprehensive_info()
    print(f"\n详细信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print()

def example_performance_benchmark():
    """性能基准测试示例"""
    print("=== 性能基准测试示例 ===")
    
    gpu_monitor = GPUMonitor()
    
    if not gpu_monitor.available:
        print("GPU监控不可用")
        return
    
    # 运行30秒的性能基准测试
    gpu_monitor.benchmark_performance(test_duration=30)
    print()

def example_continuous_monitoring():
    """持续监控示例"""
    print("=== 持续监控示例 ===")
    
    gpu_monitor = GPUMonitor()
    
    if not gpu_monitor.available:
        print("GPU监控不可用")
        return
    
    # 监控30秒，每3秒更新一次
    print("开始30秒监控（每3秒更新）...")
    gpu_monitor.monitor_continuously(duration_seconds=30, interval_seconds=3)
    print()

def example_multi_gpu():
    """多GPU监控示例"""
    print("=== 多GPU监控示例 ===")
    
    # 检查可用的GPU数量
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        print(f"检测到 {gpu_count} 个GPU设备")
        
        for i in range(gpu_count):
            try:
                gpu_monitor = GPUMonitor(gpu_index=i)
                if gpu_monitor.available:
                    print(f"\nGPU {i}:")
                    gpu_monitor.print_status("  ")
                else:
                    print(f"GPU {i}: 不可用")
            except Exception as e:
                print(f"GPU {i}: 初始化失败 - {e}")
                
    except Exception as e:
        print(f"多GPU监控失败: {e}")
    
    print()

def example_save_to_file():
    """保存监控数据到文件示例"""
    print("=== 保存监控数据示例 ===")
    
    gpu_monitor = GPUMonitor()
    
    if not gpu_monitor.available:
        print("GPU监控不可用")
        return
    
    output_file = "gpu_monitoring_example.csv"
    print(f"开始监控并保存到 {output_file}...")
    
    # 监控20秒，每2秒更新一次，保存到文件
    gpu_monitor.monitor_continuously(
        duration_seconds=20, 
        interval_seconds=2, 
        output_file=output_file
    )
    
    if os.path.exists(output_file):
        print(f"监控数据已保存到 {output_file}")
        # 显示文件前几行
        with open(output_file, 'r') as f:
            lines = f.readlines()
            print("文件内容预览:")
            for line in lines[:5]:  # 显示前5行
                print(f"  {line.strip()}")
    
    print()

def main():
    """主函数"""
    print("GPUMonitor类使用示例")
    print("=" * 50)
    
    # 检查GPU监控是否可用
    gpu_monitor = GPUMonitor()
    if not gpu_monitor.available:
        print("错误: GPU监控不可用")
        print("请确保:")
        print("1. 已安装 nvidia-ml-py: pip install nvidia-ml-py")
        print("2. 有NVIDIA GPU和驱动")
        return
    
    # 运行各种示例
    example_basic_usage()
    example_performance_benchmark()
    example_continuous_monitoring()
    example_multi_gpu()
    example_save_to_file()
    
    print("所有示例完成！")

if __name__ == "__main__":
    main() 