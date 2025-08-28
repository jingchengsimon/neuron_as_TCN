#!/usr/bin/env python3
"""
优化工作流的集成测试
"""

import unittest
import numpy as np
import torch
import sys
import os
import tempfile
import shutil

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pytorch_activity_optimizer import PytorchActivityOptimizer

class TestOptimizationWorkflow(unittest.TestCase):
    """测试完整的优化工作流"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.test_dir = tempfile.mkdtemp()
        
        # 创建测试用的模型参数
        self.test_model_params = {
            'architecture_dict': {'input_window_size': 400},
            'data_dict': {'train_files': []}
        }
        
        # 创建测试用的初始firing rates
        self.test_firing_rates = np.random.uniform(0.01, 0.05, (1279, 400)).astype(np.float32)
        
        # 保存测试数据
        test_data_dir = os.path.join(self.test_dir, 'test_data')
        os.makedirs(test_data_dir, exist_ok=True)
        
        np.save(os.path.join(test_data_dir, 'test_firing_rates.npy'), self.test_firing_rates)
        
        with open(os.path.join(test_data_dir, 'test_model_params.pickle'), 'wb') as f:
            import pickle
            pickle.dump(self.test_model_params, f)
        
        # 创建优化器
        self.optimizer = PytorchActivityOptimizer(
            model_path="test_model.pth",
            model_params_path=os.path.join(test_data_dir, 'test_model_params.pickle'),
            init_firing_rates_path=os.path.join(test_data_dir, 'test_firing_rates.npy'),
            time_duration_ms=300
        )
    
    def tearDown(self):
        """清理测试环境"""
        # 删除临时目录
        shutil.rmtree(self.test_dir)
    
    def test_complete_optimization_workflow(self):
        """测试完整的优化工作流"""
        print("\n=== 测试完整优化工作流 ===")
        
        # 设置优化参数
        num_iterations = 10  # 使用较少的迭代次数进行测试
        learning_rate = 0.001
        batch_size = 1
        target_spike_prob = 0.8
        start_time_ms = 0
        
        # 创建保存目录
        save_dir = os.path.join(self.test_dir, 'optimization_results')
        
        try:
            # 执行优化
            print("开始执行优化...")
            optimized_firing_rates, loss_history, fixed_exc_indices = self.optimizer.optimize_activity(
                num_iterations=num_iterations,
                learning_rate=learning_rate,
                batch_size=batch_size,
                target_spike_prob=target_spike_prob,
                save_dir=save_dir,
                start_time_ms=start_time_ms
            )
            
            # 检查输出
            self.assertIsNotNone(optimized_firing_rates)
            self.assertIsNotNone(loss_history)
            self.assertIsNotNone(fixed_exc_indices)
            
            # 检查形状
            expected_shape = (batch_size, 1279, 300)  # batch_size=1, segments=1279, time=300ms
            self.assertEqual(optimized_firing_rates.shape, expected_shape)
            
            # 检查损失历史长度
            self.assertEqual(len(loss_history), num_iterations)
            
            # 检查fixed_exc_indices
            self.assertEqual(len(fixed_exc_indices), 3)
            
            print("✓ 优化工作流执行成功")
            print(f"  优化后的firing rates形状: {optimized_firing_rates.shape}")
            print(f"  损失历史长度: {len(loss_history)}")
            print(f"  固定excitatory indices: {fixed_exc_indices}")
            
        except Exception as e:
            self.fail(f"优化工作流执行失败: {e}")
    
    def test_loss_decrease_trend(self):
        """测试损失是否呈下降趋势"""
        print("\n=== 测试损失下降趋势 ===")
        
        # 设置优化参数
        num_iterations = 20  # 使用更多迭代次数来观察趋势
        learning_rate = 0.001
        batch_size = 1
        target_spike_prob = 0.8
        start_time_ms = 0
        
        # 创建保存目录
        save_dir = os.path.join(self.test_dir, 'loss_trend_test')
        
        try:
            # 执行优化
            print("开始执行优化以观察损失趋势...")
            optimized_firing_rates, loss_history, fixed_exc_indices = self.optimizer.optimize_activity(
                num_iterations=num_iterations,
                learning_rate=learning_rate,
                batch_size=batch_size,
                target_spike_prob=target_spike_prob,
                save_dir=save_dir,
                start_time_ms=start_time_ms
            )
            
            # 检查损失历史
            self.assertEqual(len(loss_history), num_iterations)
            
            # 计算损失下降次数
            decrease_count = 0
            for i in range(1, len(loss_history)):
                if loss_history[i] < loss_history[i-1]:
                    decrease_count += 1
            
            # 检查是否有损失下降
            decrease_ratio = decrease_count / (len(loss_history) - 1)
            print(f"  损失下降比例: {decrease_ratio:.2f} ({decrease_count}/{len(loss_history)-1})")
            
            # 至少应该有30%的迭代出现损失下降
            self.assertGreater(decrease_ratio, 0.3, "损失下降比例过低")
            
            print("✓ 损失下降趋势测试通过")
            
        except Exception as e:
            self.fail(f"损失下降趋势测试失败: {e}")
    
    def test_firing_rates_constraints(self):
        """测试firing rates约束"""
        print("\n=== 测试Firing Rates约束 ===")
        
        # 设置优化参数
        num_iterations = 10
        learning_rate = 0.001
        batch_size = 1
        target_spike_prob = 0.8
        start_time_ms = 0
        
        # 创建保存目录
        save_dir = os.path.join(self.test_dir, 'constraints_test')
        
        try:
            # 执行优化
            print("开始执行优化以检查约束...")
            optimized_firing_rates, loss_history, fixed_exc_indices = self.optimizer.optimize_activity(
                num_iterations=num_iterations,
                learning_rate=learning_rate,
                batch_size=batch_size,
                target_spike_prob=target_spike_prob,
                save_dir=save_dir,
                start_time_ms=start_time_ms
            )
            
            # 检查firing rates是否在合理范围内 [0, 0.1]
            min_rate = np.min(optimized_firing_rates)
            max_rate = np.max(optimized_firing_rates)
            
            print(f"  Firing rates范围: [{min_rate:.6f}, {max_rate:.6f}]")
            
            # 检查约束
            self.assertGreaterEqual(min_rate, 0.0, "Firing rates应该大于等于0")
            self.assertLessEqual(max_rate, 0.1, "Firing rates应该小于等于0.1")
            
            print("✓ Firing rates约束测试通过")
            
        except Exception as e:
            self.fail(f"Firing rates约束测试失败: {e}")
    
    def test_evaluation_workflow(self):
        """测试评估工作流"""
        print("\n=== 测试评估工作流 ===")
        
        # 设置优化参数
        num_iterations = 5
        learning_rate = 0.001
        batch_size = 1
        target_spike_prob = 0.8
        start_time_ms = 0
        
        # 创建保存目录
        save_dir = os.path.join(self.test_dir, 'evaluation_test')
        
        try:
            # 执行优化
            print("开始执行优化...")
            optimized_firing_rates, loss_history, fixed_exc_indices = self.optimizer.optimize_activity(
                num_iterations=num_iterations,
                learning_rate=learning_rate,
                batch_size=batch_size,
                target_spike_prob=target_spike_prob,
                save_dir=save_dir,
                start_time_ms=start_time_ms
            )
            
            # 执行评估
            print("开始执行评估...")
            evaluation_results = self.optimizer.evaluate_optimized_activity(
                optimized_firing_rates, fixed_exc_indices, num_evaluations=5
            )
            
            # 检查评估结果
            required_keys = ['mean_spike_probability', 'std_spike_probability', 
                           'min_spike_probability', 'max_spike_probability', 'spike_probabilities']
            
            for key in required_keys:
                self.assertIn(key, evaluation_results, f"评估结果缺少键: {key}")
            
            # 检查spike概率值
            mean_prob = evaluation_results['mean_spike_probability']
            self.assertGreaterEqual(mean_prob, 0.0, "平均spike概率应该大于等于0")
            self.assertLessEqual(mean_prob, 1.0, "平均spike概率应该小于等于1")
            
            print("✓ 评估工作流测试通过")
            print(f"  平均spike概率: {mean_prob:.4f}")
            
        except Exception as e:
            self.fail(f"评估工作流测试失败: {e}")
    
    def test_save_and_load_results(self):
        """测试结果保存和加载"""
        print("\n=== 测试结果保存和加载 ===")
        
        # 设置优化参数
        num_iterations = 5
        learning_rate = 0.001
        batch_size = 1
        target_spike_prob = 0.8
        start_time_ms = 0
        
        # 创建保存目录
        save_dir = os.path.join(self.test_dir, 'save_load_test')
        
        try:
            # 执行优化
            print("开始执行优化...")
            optimized_firing_rates, loss_history, fixed_exc_indices = self.optimizer.optimize_activity(
                num_iterations=num_iterations,
                learning_rate=learning_rate,
                batch_size=batch_size,
                target_spike_prob=target_spike_prob,
                save_dir=save_dir,
                start_time_ms=start_time_ms
            )
            
            # 检查保存的文件
            expected_files = ['activity_optimization.pickle', 'loss_history.png', 'firing_rates_evolution.png']
            
            for filename in expected_files:
                file_path = os.path.join(save_dir, filename)
                self.assertTrue(os.path.exists(file_path), f"文件不存在: {filename}")
            
            # 尝试加载保存的结果
            result_file = os.path.join(save_dir, 'activity_optimization.pickle')
            with open(result_file, 'rb') as f:
                import pickle
                loaded_results = pickle.load(f)
            
            # 检查加载的数据
            self.assertIn('optimized_firing_rates', loaded_results)
            self.assertIn('loss_history', loaded_results)
            self.assertIn('fixed_exc_indices', loaded_results)
            
            # 检查数据一致性
            np.testing.assert_array_equal(loaded_results['optimized_firing_rates'], optimized_firing_rates)
            np.testing.assert_array_equal(loaded_results['fixed_exc_indices'], fixed_exc_indices)
            
            print("✓ 结果保存和加载测试通过")
            
        except Exception as e:
            self.fail(f"结果保存和加载测试失败: {e}")

if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)

