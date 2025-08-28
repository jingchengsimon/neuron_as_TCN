#!/usr/bin/env python3
"""
PyTorch Activity Optimization 演示脚本
"""

import numpy as np
import torch
import os
import tempfile
import shutil
from pytorch_activity_optimizer import PytorchActivityOptimizer

def create_demo_data():
    """创建演示数据"""
    print("创建演示数据...")
    
    # 创建临时目录
    demo_dir = tempfile.mkdtemp()
    
    # 创建模型参数
    model_params = {
        'architecture_dict': {'input_window_size': 400},
        'data_dict': {'train_files': []}
    }
    
    # 创建初始firing rates
    firing_rates = np.random.uniform(0.01, 0.05, (1279, 400)).astype(np.float32)
    
    # 保存数据
    params_file = os.path.join(demo_dir, 'demo_model_params.pickle')
    firing_rates_file = os.path.join(demo_dir, 'demo_firing_rates.npy')
    
    with open(params_file, 'wb') as f:
        import pickle
        pickle.dump(model_params, f)
    
    np.save(firing_rates_file, firing_rates)
    
    print(f"演示数据已创建在: {demo_dir}")
    return demo_dir, params_file, firing_rates_file

def run_demo():
    """运行演示"""
    print("=" * 60)
    print("PyTorch Activity Optimization 演示")
    print("=" * 60)
    
    # 创建演示数据
    demo_dir, params_file, firing_rates_file = create_demo_data()
    
    try:
        # 创建优化器
        print("\n1. 创建优化器...")
        optimizer = PytorchActivityOptimizer(
            model_path="demo_model.pth",  # 将使用测试模型
            model_params_path=params_file,
            init_firing_rates_path=firing_rates_file,
            time_duration_ms=300,
            device='auto'
        )
        
        print("✓ 优化器创建成功")
        print(f"  设备: {optimizer.device}")
        print(f"  输入窗口大小: {optimizer.input_window_size}ms")
        print(f"  Segments: {optimizer.num_segments_total} ({optimizer.num_segments_exc} exc + {optimizer.num_segments_inh} inh)")
        
        # 测试基本功能
        print("\n2. 测试基本功能...")
        
        # 测试数据准备
        test_firing_rates = np.random.uniform(0.01, 0.05, (1, 1279, 300)).astype(np.float32)
        prepared_rates = optimizer.prepare_firing_rates_for_optimization(
            test_firing_rates, batch_size=2, start_time_ms=0
        )
        print(f"✓ 数据准备测试通过，输出形状: {prepared_rates.shape}")
        
        # 测试spike生成
        fixed_exc_indices = np.array([100, 200, 300])
        spike_trains, returned_indices = optimizer.generate_spikes_with_modification(
            test_firing_rates, fixed_exc_indices, random_seed=42
        )
        print(f"✓ Spike生成测试通过，输出形状: {spike_trains.shape}")
        
        # 测试损失计算
        loss = optimizer.compute_loss_numpy(
            test_firing_rates, fixed_exc_indices, 0.8, random_seed=42
        )
        print(f"✓ 损失计算测试通过，损失值: {loss:.6f}")
        
        # 测试梯度计算
        print("\n3. 测试梯度计算...")
        gradient = optimizer.compute_numerical_gradient(
            test_firing_rates, fixed_exc_indices, 0.8, random_seed=42
        )
        print(f"✓ 梯度计算测试通过，梯度形状: {gradient.shape}")
        print(f"  梯度范数: {np.linalg.norm(gradient):.8f}")
        
        # 测试完整优化流程（少量迭代）
        print("\n4. 测试完整优化流程...")
        save_dir = os.path.join(demo_dir, 'demo_results')
        
        optimized_firing_rates, loss_history, fixed_exc_indices = optimizer.optimize_activity(
            num_iterations=5,  # 少量迭代用于演示
            learning_rate=0.001,
            batch_size=1,
            target_spike_prob=0.8,
            save_dir=save_dir,
            start_time_ms=0
        )
        
        print("✓ 优化流程测试通过")
        print(f"  优化后的firing rates形状: {optimized_firing_rates.shape}")
        print(f"  损失历史长度: {len(loss_history)}")
        print(f"  最终损失: {loss_history[-1]:.6f}")
        
        # 测试评估
        print("\n5. 测试评估功能...")
        evaluation_results = optimizer.evaluate_optimized_activity(
            optimized_firing_rates, fixed_exc_indices, num_evaluations=3
        )
        print("✓ 评估功能测试通过")
        print(f"  平均spike概率: {evaluation_results['mean_spike_probability']:.4f}")
        
        print("\n" + "=" * 60)
        print("所有演示测试通过！✓")
        print("=" * 60)
        
        # 显示结果文件
        if os.path.exists(save_dir):
            print(f"\n结果文件已保存到: {save_dir}")
            files = os.listdir(save_dir)
            for file in files:
                print(f"  - {file}")
        
    except Exception as e:
        print(f"\n演示过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理演示数据
        print(f"\n清理演示数据...")
        shutil.rmtree(demo_dir)
        print("演示数据已清理")

if __name__ == '__main__':
    run_demo()
