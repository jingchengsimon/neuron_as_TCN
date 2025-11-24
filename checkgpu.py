#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU状态检查脚本
类似checkcpu的形式，实时查看GPU占用率

使用方法:
    python checkgpu.py          # 查看GPU 0
    python checkgpu.py 1        # 查看GPU 1
    python checkgpu.py 0 2      # 查看GPU 0，采样间隔2秒
"""

import sys

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    print("错误: 请先安装 nvidia-ml-py")
    print("安装命令: pip install nvidia-ml-py")
    sys.exit(1)
except Exception as e:
    print(f"错误: GPU监控初始化失败 - {e}")
    sys.exit(1)


def checkgpu(gpu_index=0, interval=1):
    """
    实时查看GPU占用率，类似checkcpu的形式
    
    Args:
        gpu_index: GPU设备索引，默认为0
        interval: 采样间隔（秒），用于计算平均利用率，默认为1秒
    """
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        
        # 获取GPU名称
        gpu_name_bytes = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name_bytes, bytes):
            gpu_name = gpu_name_bytes.decode('utf-8')
        else:
            gpu_name = str(gpu_name_bytes)
        
        # 获取GPU利用率
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = utilization.gpu
        
        # 获取内存信息
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_used_gb = memory_info.used / 1024**3
        memory_total_gb = memory_info.total / 1024**3
        memory_available_gb = memory_info.free / 1024**3
        memory_percent = (memory_info.used / memory_info.total) * 100
        
        # 获取温度
        try:
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except:
            temperature = "N/A"
        
        # 获取功耗
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
        except:
            power = "N/A"
        
        # 打印信息（类似checkcpu的格式）
        print(f"GPU {gpu_index} ({gpu_name}):")
        print(f"  GPU 使用率: {gpu_util}%")
        print(f"  已用内存: {memory_used_gb:.2f} GB")
        print(f"  可用内存: {memory_available_gb:.2f} GB")
        print(f"  总内存: {memory_total_gb:.2f} GB ({memory_percent:.1f}%)")
        if temperature != "N/A":
            print(f"  温度: {temperature}°C")
        else:
            print(f"  温度: {temperature}")
        if power != "N/A":
            print(f"  功耗: {power:.1f}W")
        else:
            print(f"  功耗: {power}")
        
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # 解析命令行参数
    gpu_index = 0
    interval = 1
    
    if len(sys.argv) > 1:
        try:
            gpu_index = int(sys.argv[1])
        except ValueError:
            print(f"错误: 无效的GPU索引 '{sys.argv[1]}'")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        try:
            interval = float(sys.argv[2])
        except ValueError:
            print(f"错误: 无效的采样间隔 '{sys.argv[2]}'")
            sys.exit(1)
    
    # 检查GPU数量
    try:
        num_gpus = pynvml.nvmlDeviceGetCount()
        if gpu_index >= num_gpus:
            print(f"错误: GPU索引 {gpu_index} 超出范围，可用GPU数量: {num_gpus}")
            sys.exit(1)
    except Exception as e:
        print(f"错误: 无法获取GPU数量 - {e}")
        sys.exit(1)
    
    # 执行检查
    checkgpu(gpu_index, interval)

