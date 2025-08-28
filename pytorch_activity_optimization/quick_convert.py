#!/usr/bin/env python3
"""
快速模型转换脚本
将TensorFlow .h5模型快速转换为PyTorch .pth模型
"""

import os
import sys
import glob
from pathlib import Path

def quick_convert_models(models_dir: str, output_dir: str = None):
    """
    快速转换目录中的所有模型
    
    Args:
        models_dir: 包含.h5模型的目录
        output_dir: 输出目录，默认为models_dir
    """
    if output_dir is None:
        output_dir = models_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有.h5文件
    h5_files = glob.glob(os.path.join(models_dir, "*.h5"))
    
    if not h5_files:
        print(f"在 {models_dir} 中未找到.h5文件")
        return
    
    print(f"找到 {len(h5_files)} 个.h5模型文件")
    
    # 尝试导入转换器
    try:
        from model_converter import TCNModelConverter
    except ImportError as e:
        print(f"错误: 无法导入模型转换器: {e}")
        print("请确保已安装TensorFlow: pip install tensorflow")
        return
    
    # 转换每个模型
    for h5_file in h5_files:
        try:
            print(f"\n正在转换: {os.path.basename(h5_file)}")
            
            # 查找对应的参数文件
            params_file = h5_file.replace('.h5', '.pickle')
            if not os.path.exists(params_file):
                print(f"  警告: 未找到参数文件: {os.path.basename(params_file)}")
                params_file = None
            
            # 创建转换器
            converter = TCNModelConverter(h5_file, params_file)
            
            # 生成输出路径
            output_name = Path(h5_file).stem + "_converted.pth"
            output_path = os.path.join(output_dir, output_name)
            
            # 执行转换
            pytorch_model = converter.convert(output_path)
            
            if pytorch_model is not None:
                print(f"  ✓ 转换成功: {output_name}")
                
                # 快速测试
                try:
                    converter.test_conversion(pytorch_model)
                except Exception as e:
                    print(f"  ⚠ 测试失败: {e}")
            else:
                print(f"  ✗ 转换失败")
                
        except Exception as e:
            print(f"  ✗ 转换失败: {e}")
    
    print(f"\n转换完成！输出目录: {output_dir}")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python quick_convert.py <models_dir> [output_dir]")
        print("示例: python quick_convert.py ./models")
        print("示例: python quick_convert.py ./models ./converted_models")
        return
    
    models_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(models_dir):
        print(f"错误: 目录不存在: {models_dir}")
        return
    
    print("=" * 60)
    print("快速模型转换工具")
    print("=" * 60)
    print(f"输入目录: {models_dir}")
    print(f"输出目录: {output_dir or models_dir}")
    print()
    
    quick_convert_models(models_dir, output_dir)

if __name__ == "__main__":
    main()

