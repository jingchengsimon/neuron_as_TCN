#!/usr/bin/env python3
"""
简化的GPU监控脚本
不依赖复杂的类，直接使用pynvml
"""

import time
import sys

def check_gpu_availability():
    """检查GPU是否可用"""
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        if gpu_count > 0:
            return True, gpu_count
        else:
            return False, 0
    except ImportError:
        print("错误: pynvml未安装，请运行: pip install nvidia-ml-py")
        return False, 0
    except Exception as e:
        print(f"错误: GPU检查失败 - {e}")
        return False, 0

def get_gpu_info(gpu_index=0):
    """获取GPU信息"""
    try:
        import pynvml
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        
        # 获取GPU名称
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            gpu_name = name.decode('utf-8')
        else:
            gpu_name = str(name)
        
        # 获取利用率
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        # 获取内存信息
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mb = memory_info.used / 1024 / 1024
        total_mb = memory_info.total / 1024 / 1024
        memory_percent = (memory_info.used / memory_info.total) * 100
        
        # 获取温度
        try:
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except:
            temperature = "N/A"
        
        # 获取功耗
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except:
            power = "N/A"
        
        return {
            'name': gpu_name,
            'utilization': utilization.gpu,
            'memory_used_mb': used_mb,
            'memory_total_mb': total_mb,
            'memory_percent': memory_percent,
            'temperature': temperature,
            'power': power
        }
        
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
        return None

def print_gpu_status(gpu_index=0):
    """打印GPU状态"""
    info = get_gpu_info(gpu_index)
    if info:
        print(f"GPU {gpu_index}: {info['name']}")
        print(f"  利用率: {info['utilization']}%")
        print(f"  内存: {info['memory_used_mb']:.0f}/{info['memory_total_mb']:.0f} MB ({info['memory_percent']:.1f}%)")
        print(f"  温度: {info['temperature']}°C")
        print(f"  功耗: {info['power']}W")
    else:
        print(f"GPU {gpu_index}: 无法获取信息")

def monitor_gpu(duration_minutes=60, interval_seconds=2, gpu_index=0):
    """监控GPU使用情况"""
    print(f"开始监控GPU {gpu_index}")
    print(f"持续时间: {duration_minutes}分钟, 间隔: {interval_seconds}秒")
    print("=" * 80)
    print("时间戳           | GPU利用率(%) | 内存使用(MB) | 内存使用率(%) | 温度(°C) | 功耗(W)")
    print("=" * 80)
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    try:
        while time.time() < end_time:
            info = get_gpu_info(gpu_index)
            if info:
                timestamp = time.strftime("%H:%M:%S")
                line = f"{timestamp} | {info['utilization']:11} | {info['memory_used_mb']:10.0f} | {info['memory_percent']:12.1f} | {info['temperature']:8} | {info['power']:6}"
                print(line)
            else:
                timestamp = time.strftime("%H:%M:%S")
                print(f"{timestamp} | 获取信息失败")
            
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\n监控已停止")
    
    print("=" * 80)
    print("监控完成")

def main():
    """主函数"""
    print("简化GPU监控工具")
    print("=" * 50)
    
    # 检查GPU可用性
    available, gpu_count = check_gpu_availability()
    if not available:
        print("GPU不可用，请检查:")
        print("1. 是否有NVIDIA GPU")
        print("2. 是否安装了NVIDIA驱动")
        print("3. 是否安装了pynvml: pip install nvidia-ml-py")
        return
    
    print(f"检测到 {gpu_count} 个GPU设备")
    
    # 显示所有GPU的状态
    for i in range(gpu_count):
        print(f"\nGPU {i} 状态:")
        print_gpu_status(i)
    
    # 如果只有一个GPU，询问是否开始监控
    if gpu_count == 1:
        print(f"\n是否开始监控GPU 0? (y/n): ", end="")
        response = input().lower().strip()
        if response in ['y', 'yes', '是']:
            monitor_gpu(5, 2, 0)  # 监控5分钟，每2秒更新
    else:
        # 如果有多个GPU，让用户选择
        print(f"\n请选择要监控的GPU (0-{gpu_count-1}): ", end="")
        try:
            gpu_index = int(input())
            if 0 <= gpu_index < gpu_count:
                monitor_gpu(5, 2, gpu_index)
            else:
                print("无效的GPU索引")
        except ValueError:
            print("请输入有效的数字")

if __name__ == "__main__":
    main() 