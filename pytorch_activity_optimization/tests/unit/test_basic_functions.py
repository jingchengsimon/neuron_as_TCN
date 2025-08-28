#!/usr/bin/env python3
"""
基本功能的单元测试
"""

import unittest
import numpy as np
import torch
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pytorch_activity_optimizer import PytorchActivityOptimizer

class TestBasicFunctions(unittest.TestCase):
    """测试基本功能"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建测试用的模型参数
        self.test_model_params = {
            'architecture_dict': {'input_window_size': 400},
            'data_dict': {'train_files': []}
        }
        
        # 创建测试用的初始firing rates
        self.test_firing_rates = np.random.uniform(0.01, 0.05, (1279, 400)).astype(np.float32)
        
        # 保存测试数据
        os.makedirs('./test_data', exist_ok=True)
        np.save('./test_data/test_firing_rates.npy', self.test_firing_rates)
        
        with open('./test_data/test_model_params.pickle', 'wb') as f:
            import pickle
            pickle.dump(self.test_model_params, f)
    
    def tearDown(self):
        """清理测试环境"""
        # 删除测试文件
        import shutil
        if os.path.exists('./test_data'):
            shutil.rmtree('./test_data')
    
    def test_optimizer_initialization(self):
        """测试优化器初始化"""
        try:
            optimizer = PytorchActivityOptimizer(
                model_path="test_model.pth",
                model_params_path="./test_data/test_model_params.pickle",
                init_firing_rates_path="./test_data/test_firing_rates.npy",
                time_duration_ms=300
            )
            
            # 检查基本属性
            self.assertEqual(optimizer.time_duration_ms, 300)
            self.assertEqual(optimizer.input_window_size, 400)
            self.assertEqual(optimizer.num_segments_exc, 639)
            self.assertEqual(optimizer.num_segments_inh, 640)
            self.assertEqual(optimizer.num_segments_total, 1279)
            
            # 检查初始firing rates是否正确加载
            self.assertIsNotNone(optimizer.init_firing_rates)
            self.assertEqual(optimizer.init_firing_rates.shape, (1279, 400))
            
            print("✓ 优化器初始化测试通过")
            
        except Exception as e:
            self.fail(f"优化器初始化失败: {e}")
    
    def test_load_init_firing_rates(self):
        """测试初始firing rates加载"""
        optimizer = PytorchActivityOptimizer(
            model_path="test_model.pth",
            model_params_path="./test_data/test_model_params.pickle",
            time_duration_ms=300
        )
        
        # 测试加载firing rates
        firing_rates = optimizer.load_init_firing_rates("./test_data/test_firing_rates.npy")
        
        self.assertIsNotNone(firing_rates)
        self.assertEqual(firing_rates.shape, (1279, 400))
        self.assertEqual(firing_rates.dtype, np.float32)
        
        print("✓ 初始firing rates加载测试通过")
    
    def test_prepare_firing_rates_for_optimization(self):
        """测试firing rates准备函数"""
        optimizer = PytorchActivityOptimizer(
            model_path="test_model.pth",
            model_params_path="./test_data/test_model_params.pickle",
            time_duration_ms=300
        )
        
        # 测试数据准备
        prepared_rates = optimizer.prepare_firing_rates_for_optimization(
            self.test_firing_rates, batch_size=2, start_time_ms=50
        )
        
        # 检查输出形状
        expected_shape = (2, 1279, 300)  # batch_size=2, segments=1279, time=300ms
        self.assertEqual(prepared_rates.shape, expected_shape)
        
        # 检查数据类型
        self.assertEqual(prepared_rates.dtype, np.float32)
        
        print("✓ Firing rates准备函数测试通过")
    
    def test_generate_spikes_with_modification(self):
        """测试spike生成函数"""
        optimizer = PytorchActivityOptimizer(
            model_path="test_model.pth",
            model_params_path="./test_data/test_model_params.pickle",
            time_duration_ms=300
        )
        
        # 准备测试数据
        test_firing_rates = np.random.uniform(0.01, 0.05, (1, 1279, 300)).astype(np.float32)
        fixed_exc_indices = np.array([100, 200, 300])
        
        # 测试spike生成
        spike_trains, returned_indices = optimizer.generate_spikes_with_modification(
            test_firing_rates, fixed_exc_indices, random_seed=42
        )
        
        # 检查输出
        self.assertEqual(spike_trains.shape, (2, 1279, 300))  # 2个batch
        self.assertEqual(returned_indices.shape, (3,))
        np.testing.assert_array_equal(returned_indices, fixed_exc_indices)
        
        # 检查spike值是否为0或1
        self.assertTrue(np.all(np.isin(spike_trains, [0, 1])))
        
        print("✓ Spike生成函数测试通过")
    
    def test_compute_loss_numpy(self):
        """测试损失计算函数"""
        optimizer = PytorchActivityOptimizer(
            model_path="test_model.pth",
            model_params_path="./test_data/test_model_params.pickle",
            time_duration_ms=300
        )
        
        # 准备测试数据
        test_firing_rates = np.random.uniform(0.01, 0.05, (1, 1279, 300)).astype(np.float32)
        fixed_exc_indices = np.array([100, 200, 300])
        target_spike_prob = 0.8
        
        # 测试损失计算
        loss = optimizer.compute_loss_numpy(
            test_firing_rates, fixed_exc_indices, target_spike_prob, random_seed=42
        )
        
        # 检查损失值
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)  # 损失应该为正数
        
        print("✓ 损失计算函数测试通过")

if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)


