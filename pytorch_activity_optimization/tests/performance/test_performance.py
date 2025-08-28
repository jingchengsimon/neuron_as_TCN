#!/usr/bin/env python3
"""
性能测试文件
"""

import unittest
import numpy as np
import torch
import sys
import os
import time
import tempfile
import shutil

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pytorch_activity_optimization import PytorchActivityOptimizer

class TestPerformance(unittest.TestCase):
    """性能测试"""
    
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
    
    def test_gradient_computation_speed(self):
        """测试梯度计算速度"""
        print("\n=== 测试梯度计算速度 ===")
        
        # 准备测试数据
        test_firing_rates = np.random.uniform(0.01, 0.05, (1, 1279, 300)).astype(np.float32)
        fixed_exc_indices = np.array([100, 200, 300])
        target_spike_prob = 0.8
        
        # 预热（运行几次以避免首次运行的初始化开销）
        print("预热中...")
        for _ in range(3):
            _ = self.optimizer.compute_numerical_gradient(
                test_firing_rates, fixed_exc_indices, target_spike_prob, random_seed=42
            )
        
        # 测试多次运行的平均时间
        num_runs = 10
        times = []
        
        print(f"执行 {num_runs} 次梯度计算...")
        for i in range(num_runs):
            start_time = time.time()
            _ = self.optimizer.compute_numerical_gradient(
                test_firing_rates, fixed_exc_indices, target_spike_prob, random_seed=42+i
            )
            end_time = time.time()
            run_time = end_time - start_time
            times.append(run_time)
            print(f"  运行 {i+1}: {run_time:.4f} 秒")
        
        # 计算统计信息
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"\n梯度计算性能统计:")
        print(f"  平均时间: {avg_time:.4f} ± {std_time:.4f} 秒")
        print(f"  最快时间: {min_time:.4f} 秒")
        print(f"  最慢时间: {max_time:.4f} 秒")
        
        # 性能要求：平均时间应该小于1秒
        self.assertLess(avg_time, 1.0, f"梯度计算太慢: {avg_time:.4f} 秒")
        
        print("✓ 梯度计算速度测试通过")
    
    def test_memory_usage(self):
        """测试内存使用情况"""
        print("\n=== 测试内存使用情况 ===")
        
        try:
            import psutil
            process = psutil.Process()
            
            # 记录初始内存
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"初始内存使用: {initial_memory:.2f} MB")
            
            # 执行一些操作
            test_firing_rates = np.random.uniform(0.01, 0.05, (1, 1279, 300)).astype(np.float32)
            fixed_exc_indices = np.array([100, 200, 300])
            target_spike_prob = 0.8
            
            # 计算梯度
            _ = self.optimizer.compute_numerical_gradient(
                test_firing_rates, fixed_exc_indices, target_spike_prob, random_seed=42
            )
            
            # 记录操作后内存
            after_operation_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"操作后内存使用: {after_operation_memory:.2f} MB")
            
            # 计算内存增长
            memory_increase = after_operation_memory - initial_memory
            print(f"内存增长: {memory_increase:.2f} MB")
            
            # 内存增长应该合理（小于100MB）
            self.assertLess(memory_increase, 100.0, f"内存增长过大: {memory_increase:.2f} MB")
            
            print("✓ 内存使用测试通过")
            
        except ImportError:
            print("psutil未安装，跳过内存测试")
            self.skipTest("psutil未安装")
    
    def test_batch_size_scaling(self):
        """测试不同batch size的性能缩放"""
        print("\n=== 测试Batch Size性能缩放 ===")
        
        batch_sizes = [1, 2, 4]
        times_per_batch = []
        
        fixed_exc_indices = np.array([100, 200, 300])
        target_spike_prob = 0.8
        
        for batch_size in batch_sizes:
            print(f"\n测试 batch_size = {batch_size}")
            
            # 创建对应batch size的数据
            test_firing_rates = np.random.uniform(0.01, 0.05, (batch_size, 1279, 300)).astype(np.float32)
            
            # 预热
            for _ in range(2):
                _ = self.optimizer.compute_numerical_gradient(
                    test_firing_rates, fixed_exc_indices, target_spike_prob, random_seed=42
                )
            
            # 测试多次运行的平均时间
            num_runs = 5
            times = []
            
            for i in range(num_runs):
                start_time = time.time()
                _ = self.optimizer.compute_numerical_gradient(
                    test_firing_rates, fixed_exc_indices, target_spike_prob, random_seed=42+i
                )
                end_time = time.time()
                run_time = end_time - start_time
                times.append(run_time)
            
            avg_time = np.mean(times)
            times_per_batch.append(avg_time)
            
            print(f"  平均时间: {avg_time:.4f} 秒")
            print(f"  数据大小: {test_firing_rates.shape}")
        
        # 分析性能缩放
        print(f"\n性能缩放分析:")
        for i, batch_size in enumerate(batch_sizes):
            if i > 0:
                scaling_factor = times_per_batch[i] / times_per_batch[0]
                expected_scaling = batch_size
                efficiency = expected_scaling / scaling_factor
                print(f"  Batch {batch_size}: 实际缩放 {scaling_factor:.2f}x, 期望 {expected_scaling}x, 效率 {efficiency:.2f}")
        
        print("✓ Batch size性能缩放测试通过")
    
    def test_optimization_convergence_speed(self):
        """测试优化收敛速度"""
        print("\n=== 测试优化收敛速度 ===")
        
        # 设置优化参数
        num_iterations = 50
        learning_rate = 0.001
        batch_size = 1
        target_spike_prob = 0.8
        start_time_ms = 0
        
        # 创建保存目录
        save_dir = os.path.join(self.test_dir, 'convergence_test')
        
        try:
            # 记录开始时间
            start_time = time.time()
            
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
            
            # 记录结束时间
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"优化完成，总时间: {total_time:.2f} 秒")
            print(f"平均每次迭代: {total_time/num_iterations:.4f} 秒")
            
            # 分析收敛性
            initial_loss = loss_history[0]
            final_loss = loss_history[-1]
            loss_reduction = initial_loss - final_loss
            loss_reduction_ratio = loss_reduction / initial_loss
            
            print(f"损失变化: {initial_loss:.6f} -> {final_loss:.6f}")
            print(f"损失减少: {loss_reduction:.6f} ({loss_reduction_ratio*100:.2f}%)")
            
            # 检查是否有显著的损失减少
            self.assertGreater(loss_reduction_ratio, 0.1, "损失减少不够显著")
            
            # 检查收敛速度（总时间应该合理）
            max_expected_time = 60.0  # 60秒
            self.assertLess(total_time, max_expected_time, f"优化时间过长: {total_time:.2f} 秒")
            
            print("✓ 优化收敛速度测试通过")
            
        except Exception as e:
            self.fail(f"优化收敛速度测试失败: {e}")
    
    def test_device_performance_comparison(self):
        """测试不同设备的性能比较"""
        print("\n=== 测试设备性能比较 ===")
        
        # 检查可用的设备
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
        
        print(f"可用设备: {devices}")
        
        if len(devices) < 2:
            print("只有一个设备可用，跳过设备比较测试")
            self.skipTest("只有一个设备可用")
        
        # 准备测试数据
        test_firing_rates = np.random.uniform(0.01, 0.05, (1, 1279, 300)).astype(np.float32)
        fixed_exc_indices = np.array([100, 200, 300])
        target_spike_prob = 0.8
        
        device_performance = {}
        
        for device in devices:
            print(f"\n测试设备: {device}")
            
            # 创建对应设备的优化器
            device_optimizer = PytorchActivityOptimizer(
                model_path="test_model.pth",
                model_params_path=os.path.join(self.test_dir, 'test_data', 'test_model_params.pickle'),
                time_duration_ms=300,
                device=device
            )
            
            # 预热
            for _ in range(2):
                _ = device_optimizer.compute_numerical_gradient(
                    test_firing_rates, fixed_exc_indices, target_spike_prob, random_seed=42
                )
            
            # 测试多次运行的平均时间
            num_runs = 5
            times = []
            
            for i in range(num_runs):
                start_time = time.time()
                _ = device_optimizer.compute_numerical_gradient(
                    test_firing_rates, fixed_exc_indices, target_spike_prob, random_seed=42+i
                )
                end_time = time.time()
                run_time = end_time - start_time
                times.append(run_time)
            
            avg_time = np.mean(times)
            device_performance[device] = avg_time
            
            print(f"  平均时间: {avg_time:.4f} 秒")
        
        # 性能比较
        print(f"\n设备性能比较:")
        for device, time_taken in device_performance.items():
            print(f"  {device}: {time_taken:.4f} 秒")
        
        # 如果有GPU，检查GPU是否比CPU快
        if 'cuda' in device_performance and 'cpu' in device_performance:
            gpu_time = device_performance['cuda']
            cpu_time = device_performance['cpu']
            
            if gpu_time < cpu_time:
                speedup = cpu_time / gpu_time
                print(f"  GPU加速比: {speedup:.2f}x")
            else:
                print("  GPU未提供加速")
        
        print("✓ 设备性能比较测试通过")

if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)

