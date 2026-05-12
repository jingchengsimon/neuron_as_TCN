#!/usr/bin/env python3
"""
GPU诊断脚本
帮助诊断GPU监控问题
"""

import sys
import os

def check_tensorflow_gpu():
    """检查TensorFlow GPU支持"""
    print("=== TensorFlow GPU检查 ===")
    try:
        import tensorflow as tf
        print(f"TensorFlow版本: {tf.__version__}")
        print(f"CUDA可用: {tf.test.is_built_with_cuda()}")
        print(f"GPU设备数量: {len(tf.config.list_physical_devices('GPU'))}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print("检测到的GPU设备:")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
        else:
            print("未检测到GPU设备")
            
    except ImportError:
        print("TensorFlow未安装")
    except Exception as e:
        print(f"TensorFlow检查失败: {e}")
    print()

def check_nvidia_driver():
    """检查NVIDIA驱动"""
    print("=== NVIDIA驱动检查 ===")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("nvidia-smi命令可用")
            print("输出预览:")
            lines = result.stdout.split('\n')[:10]  # 显示前10行
            for line in lines:
                print(f"  {line}")
        else:
            print("nvidia-smi命令不可用")
            print(f"错误: {result.stderr}")
    except FileNotFoundError:
        print("nvidia-smi命令未找到")
    except Exception as e:
        print(f"NVIDIA驱动检查失败: {e}")
    print()

def check_pynvml():
    """检查pynvml库"""
    print("=== pynvml库检查 ===")
    try:
        import pynvml
        print("pynvml库已安装")
        
        # 初始化
        pynvml.nvmlInit()
        print("pynvml初始化成功")
        
        # 获取GPU数量
        gpu_count = pynvml.nvmlDeviceGetCount()
        print(f"检测到 {gpu_count} 个GPU设备")
        
        # 检查每个GPU
        for i in range(gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                
                # 处理不同版本的返回格式
                if isinstance(name, bytes):
                    gpu_name = name.decode('utf-8')
                else:
                    gpu_name = str(name)
                
                print(f"  GPU {i}: {gpu_name}")
                
                # 获取内存信息
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_mb = memory_info.total / 1024 / 1024
                print(f"    总内存: {total_mb:.0f} MB")
                
            except Exception as e:
                print(f"  GPU {i}: 获取信息失败 - {e}")
        
        pynvml.nvmlShutdown()
        
    except ImportError:
        print("pynvml库未安装")
        print("请运行: pip install nvidia-ml-py")
    except Exception as e:
        print(f"pynvml检查失败: {e}")
    print()

def check_gpu_monitor():
    """检查GPUMonitor类"""
    print("=== GPUMonitor类检查 ===")
    try:
        # 添加当前目录到路径
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from train_and_analyze import GPUMonitor
        
        gpu_monitor = GPUMonitor()
        if gpu_monitor.available:
            print("GPUMonitor类工作正常")
            gpu_monitor.print_status("  ")
        else:
            print("GPUMonitor类不可用")
            
    except ImportError as e:
        print(f"无法导入GPUMonitor: {e}")
    except Exception as e:
        print(f"GPUMonitor检查失败: {e}")
    print()

def main():
    """主函数"""
    print("GPU诊断工具")
    print("=" * 50)
    
    check_tensorflow_gpu()
    check_nvidia_driver()
    check_pynvml()
    check_gpu_monitor()
    
    print("诊断完成！")
    print("\n如果所有检查都通过，GPU监控应该可以正常工作。")
    print("如果有问题，请根据上述输出进行相应的修复。")

if __name__ == "__main__":
    main() 