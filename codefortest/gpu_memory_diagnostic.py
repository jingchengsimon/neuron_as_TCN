#!/usr/bin/env python3
"""
GPU内存诊断脚本
帮助诊断和解决GPU内存不足问题
"""

import os
import sys
import time
import numpy as np

def check_gpu_memory():
    """检查GPU内存使用情况"""
    try:
        import pynvml
        pynvml.nvmlInit()
        
        gpu_count = pynvml.nvmlDeviceGetCount()
        print(f"检测到 {gpu_count} 个GPU设备")
        
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # 获取GPU名称
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                gpu_name = name.decode('utf-8')
            else:
                gpu_name = str(name)
            
            # 获取内存信息
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mb = memory_info.total / 1024 / 1024
            used_mb = memory_info.used / 1024 / 1024
            free_mb = memory_info.free / 1024 / 1024
            memory_percent = (memory_info.used / memory_info.total) * 100
            
            print(f"\nGPU {i}: {gpu_name}")
            print(f"  总内存: {total_mb:.0f} MB")
            print(f"  已使用: {used_mb:.0f} MB")
            print(f"  可用: {free_mb:.0f} MB")
            print(f"  使用率: {memory_percent:.1f}%")
            
            # 内存使用建议
            if memory_percent > 90:
                print("  ⚠️  警告: GPU内存使用率过高!")
            elif memory_percent > 70:
                print("  ⚠️  注意: GPU内存使用率较高")
            else:
                print("  ✓ GPU内存使用正常")
        
        pynvml.nvmlShutdown()
        
    except ImportError:
        print("错误: pynvml未安装，请运行: pip install nvidia-ml-py")
    except Exception as e:
        print(f"GPU内存检查失败: {e}")

def check_tensorflow_gpu():
    """检查TensorFlow GPU配置"""
    try:
        import tensorflow as tf
        print(f"\nTensorFlow版本: {tf.__version__}")
        print(f"CUDA可用: {tf.test.is_built_with_cuda()}")
        
        gpus = tf.config.list_physical_devices('GPU')
        print(f"TensorFlow检测到的GPU数量: {len(gpus)}")
        
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
                
    except ImportError:
        print("TensorFlow未安装")
    except Exception as e:
        print(f"TensorFlow检查失败: {e}")

def estimate_memory_usage(network_depth, num_filters, input_window_size, batch_size):
    """估算模型内存使用量"""
    print(f"\n=== 内存使用估算 ===")
    print(f"网络深度: {network_depth}")
    print(f"每层滤波器数: {num_filters}")
    print(f"输入窗口大小: {input_window_size}")
    print(f"批次大小: {batch_size}")
    
    # 估算参数数量
    # 第一层卷积
    first_layer_params = 54 * 639 * num_filters  # filter_size * input_channels * num_filters
    # 后续层卷积
    subsequent_layers_params = (network_depth - 1) * 24 * num_filters * num_filters
    # 输出层
    output_layer_params = num_filters * (639 + 639)  # 兴奋性和抑制性输出
    
    total_params = first_layer_params + subsequent_layers_params + output_layer_params
    params_mb = total_params * 4 / (1024 * 1024)  # 假设float32
    
    print(f"估算参数数量: {total_params:,}")
    print(f"参数内存: {params_mb:.1f} MB")
    
    # 估算激活内存
    # 输入数据
    input_memory = batch_size * input_window_size * 639 * 4 / (1024 * 1024)
    # 中间激活（简化估算）
    activation_memory = batch_size * input_window_size * num_filters * network_depth * 4 / (1024 * 1024)
    
    print(f"输入数据内存: {input_memory:.1f} MB")
    print(f"激活内存: {activation_memory:.1f} MB")
    
    total_memory = params_mb + input_memory + activation_memory
    print(f"总估算内存: {total_memory:.1f} MB")
    
    return total_memory

def provide_memory_optimization_suggestions():
    """提供内存优化建议"""
    print(f"\n=== 内存优化建议 ===")
    print("1. 减少批次大小 (batch_size):")
    print("   - 当前: 32 → 建议: 16 或 8")
    print("   - 内存使用与批次大小成正比")
    
    print("\n2. 减少网络规模:")
    print("   - 网络深度: 7 → 建议: 3-5")
    print("   - 滤波器数量: 256 → 建议: 64-128")
    
    print("\n3. 减少数据加载:")
    print("   - train_file_load: 0.5 → 建议: 0.3")
    print("   - valid_file_load: 0.5 → 建议: 0.3")
    
    print("\n4. 关闭多进程:")
    print("   - use_multiprocessing: True → False")
    print("   - num_workers: 8 → 1")
    
    print("\n5. 设置GPU内存限制:")
    print("   - 在代码中添加内存限制配置")
    print("   - 根据GPU总内存设置合适的限制")
    
    print("\n6. 使用混合精度训练:")
    print("   - 使用tf.keras.mixed_precision")
    print("   - 可以减少约50%的内存使用")

def main():
    """主函数"""
    print("GPU内存诊断工具")
    print("=" * 50)
    
    # 检查当前GPU状态
    check_gpu_memory()
    check_tensorflow_gpu()
    
    # 提供优化建议
    provide_memory_optimization_suggestions()
    
    # 估算不同配置的内存使用
    print(f"\n=== 不同配置的内存估算 ===")
    
    configs = [
        (3, 64, 400, 16),
        (3, 128, 400, 16),
        (5, 64, 400, 16),
        (5, 128, 400, 16),
        (7, 256, 400, 32),  # 原始配置
    ]
    
    for depth, filters, window, batch in configs:
        memory = estimate_memory_usage(depth, filters, window, batch)
        if memory > 6000:  # 6GB
            print(f"  ⚠️  配置 {depth}x{filters}x{window}@{batch}: {memory:.1f}MB (可能超出6GB限制)")
        else:
            print(f"  ✓ 配置 {depth}x{filters}x{window}@{batch}: {memory:.1f}MB")
    
    print(f"\n=== 推荐配置 ===")
    print("基于6GB GPU内存限制，推荐使用:")
    print("- 网络深度: 3-5")
    print("- 滤波器数量: 64-128")
    print("- 批次大小: 16")
    print("- 输入窗口: 400")

if __name__ == "__main__":
    main() 