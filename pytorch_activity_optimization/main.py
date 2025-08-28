#!/usr/bin/env python3
"""
PyTorch版本的Activity Optimization主程序
"""

import os
import glob
import pickle
from datetime import datetime
from pytorch_activity_optimizer import PytorchActivityOptimizer

# 尝试导入可视化模块，如果不存在则跳过
try:
    from utils.visualization_utils import visualize_firing_rates_trace, visualize_firing_rates_heatmap

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("警告: 可视化模块不可用，将跳过可视化步骤")

def find_best_model(models_dir: str) -> tuple:
    """
    找到最佳模型（基于验证损失）
    
    Args:
        models_dir: 模型目录
        
    Returns:
        best_model_path: 最佳模型的.pth文件路径
        best_params_path: 对应的.pickle文件路径
    """
    pickle_files = glob.glob(os.path.join(models_dir, '*.pickle'))
    
    if not pickle_files:
        raise ValueError(f"在{models_dir}中未找到模型文件")
    
    best_val_loss = float('inf')
    best_model_path = None
    best_params_path = None
    
    for pickle_path in pickle_files:
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            # 获取最小验证损失
            val_losses = data['training_history_dict']['val_spikes_loss']
            min_val_loss = min(val_losses)
            
            if min_val_loss < best_val_loss:
                best_val_loss = min_val_loss
                best_params_path = pickle_path
                # 尝试找到对应的PyTorch模型文件
                best_model_path = pickle_path.replace('.pickle', '.pth')
                
                # 如果.pth文件不存在，尝试.pt文件
                if not os.path.exists(best_model_path):
                    best_model_path = pickle_path.replace('.pickle', '.pt')
        
        except Exception as e:
            print(f"读取{pickle_path}时出错: {e}")
    
    if best_model_path is None:
        raise ValueError("未找到有效的模型文件")
    
    print(f"找到最佳模型: {best_model_path}")
    
    return best_model_path, best_params_path

def convert_tf_model_to_pytorch(tf_model_path: str, model_params_path: str) -> str:
    """
    将TensorFlow模型转换为PyTorch模型
    
    Args:
        tf_model_path: TensorFlow模型文件路径(.h5)
        model_params_path: 模型参数文件路径(.pickle)
        
    Returns:
        pytorch_model_path: 转换后的PyTorch模型路径
    """
    print(f"\n正在转换TensorFlow模型: {tf_model_path}")
    
    try:
        # 导入模型转换器
        from model_converter import TCNModelConverter
        
        # 创建转换器
        converter = TCNModelConverter(tf_model_path, model_params_path)
        
        # 执行转换
        output_path = tf_model_path.replace('.h5', '_converted.pth')
        pytorch_model = converter.convert(output_path)
        
        # 测试转换
        converter.test_conversion(pytorch_model)
        
        print(f"✓ 模型转换成功: {output_path}")
        return output_path
        
    except ImportError as e:
        print(f"⚠ 无法导入模型转换器: {e}")
        print("  请确保已安装TensorFlow: pip install tensorflow")
        return None
    except Exception as e:
        print(f"⚠ 模型转换失败: {e}")
        return None

def find_or_convert_model(models_dir: str) -> tuple:
    """
    查找或转换模型
    
    Args:
        models_dir: 模型目录
        
    Returns:
        model_path: 模型文件路径
        params_path: 参数文件路径
    """
    # 首先尝试找到最佳模型
    try:
        model_path, params_path = find_best_model(models_dir)
        print(f"选择的模型: {os.path.basename(model_path)}")
        
        # 检查PyTorch模型文件是否存在
        if os.path.exists(model_path):
            print("✓ PyTorch模型文件存在")
            return model_path, params_path
        else:
            print(f"⚠ PyTorch模型文件不存在: {model_path}")
            
            # 尝试找到对应的TensorFlow模型
            tf_model_path = model_path.replace('.pth', '.h5').replace('.pt', '.h5')
            if os.path.exists(tf_model_path):
                print(f"找到对应的TensorFlow模型: {tf_model_path}")
                
                # 尝试转换模型
                converted_path = convert_tf_model_to_pytorch(tf_model_path, params_path)
                if converted_path and os.path.exists(converted_path):
                    print("✓ 模型转换成功，使用转换后的模型")
                    return converted_path, params_path
                else:
                    print("⚠ 模型转换失败")
            else:
                print(f"未找到对应的TensorFlow模型: {tf_model_path}")
    
    except Exception as e:
        print(f"查找模型时出错: {e}")
    
    # 如果都失败了，返回None
    return None, None

def main():
    """
    主函数：运行PyTorch版本的activity optimization
    """
    print("=== PyTorch Activity Optimization ===")
    
    # 设置文件路径
    models_dir = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut_SJC_funcgroup2_var2/models/NMDA_fullStrategy_2/depth_7_filters_256_window_400/'
    init_firing_rates_path = './init_firing_rate_array.npy'  # 初始firing rates文件
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'./results/pytorch_activity_optimization_results/{timestamp}'
    
    # 检查初始firing rates文件是否存在
    if not os.path.exists(init_firing_rates_path):
        print(f"警告：初始firing rates文件不存在: {init_firing_rates_path}")
        print("将使用随机生成的初始firing rates")
        init_firing_rates_path = None
    
    # 查找或转换模型
    print("寻找或转换模型...")
    model_path, params_path = find_or_convert_model(models_dir)
    
    if model_path is None or params_path is None:
        print("错误: 无法找到或转换模型")
        print("将使用测试模型进行演示")
        # 创建一个虚拟的.pth文件路径
        model_path = "test_model.pth"
        # 创建一个虚拟的参数文件路径
        params_path = os.path.join(models_dir, "dummy_params.pickle")
    
    # 创建优化器
    optimizer = PytorchActivityOptimizer(
        model_path=model_path, 
        model_params_path=params_path, 
        init_firing_rates_path=init_firing_rates_path,
        time_duration_ms=300,
        device='auto'  # 自动选择设备
    )
    
    # 执行优化
    optimized_firing_rates, loss_history, fixed_exc_indices = optimizer.optimize_activity(
        num_iterations=10,
        learning_rate=0.001,
        batch_size=1,
        target_spike_prob=0.8,
        save_dir=save_dir,
        start_time_ms=0  # 从0ms开始
    )
    
    # 评估结果
    evaluation_results = optimizer.evaluate_optimized_activity(
        optimized_firing_rates, fixed_exc_indices, num_evaluations=20
    )
    
    print("\n=== 优化完成 ===")
    print(f"优化后的firing rates形状: {optimized_firing_rates.shape}")
    print(f"最终损失: {loss_history[-1]:.6f}")
    print(f"固定添加spikes的excitatory segments: {fixed_exc_indices}")
    
    print(f"\n结果已保存到: {save_dir}")

    # 可视化（如果可用）
    if VISUALIZATION_AVAILABLE:
        try:
            # 取第一个batch进行可视化
            optimized_sample = optimized_firing_rates[0]  # (num_segments, time_duration)
            
            print("\n生成优化后的firing rates可视化...")
            os.makedirs(save_dir, exist_ok=True)
            
            # 指定要可视化的segments：优先显示fixed_exc_indices，如果没有则使用默认采样
            specified_segments = None
            if fixed_exc_indices is not None and len(fixed_exc_indices) > 0:
                # 扩展fixed_exc_indices，包含周围的一些segments以便更好地观察
                extended_indices = []
                for idx in fixed_exc_indices:
                    # 为每个fixed index添加前后各2个segments
                    start_idx = max(0, idx - 1)
                    end_idx = min(optimized_sample.shape[0], idx + 1)
                    extended_indices.extend(range(start_idx, end_idx))
                
                # 去重并排序
                extended_indices = sorted(list(set(extended_indices)))
                specified_segments = extended_indices
                
                print(f"指定可视化segments: {specified_segments}")
                print(f"包含fixed_exc_indices: {fixed_exc_indices}")
            
            # Raster plot
            visualize_firing_rates_trace(
                firing_rates=optimized_sample,
                num_exc_segments=639,
                save_path=os.path.join(save_dir, 'optimized_firing_rates_raster.png'),
                title="Optimized Firing Rates - Raster Plot",
                max_segments_to_show=10,
                specified_segments=specified_segments
            )
            
            # Heatmap
            visualize_firing_rates_heatmap(
                firing_rates=optimized_sample,
                num_exc_segments=639,
                save_path=os.path.join(save_dir, 'optimized_firing_rates_heatmap.png'),
                title="Optimized Firing Rates - Heatmap",
                max_segments_to_show=10,
                specified_segments=specified_segments
            )
            
            print("优化后的firing rates可视化已保存")
            
        except Exception as e:
            print(f"生成可视化时出错: {e}")
    else:
        print("跳过可视化步骤（可视化模块不可用）")

if __name__ == "__main__":
    main()
