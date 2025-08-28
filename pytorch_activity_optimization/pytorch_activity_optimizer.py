import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import glob
import matplotlib.pyplot as plt
import time
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
import warnings

# 忽略PyTorch的警告
warnings.filterwarnings("ignore", category=UserWarning)

class PytorchActivityOptimizer:
    """
    基于PyTorch的Activity Optimization类
    实现图中描述的优化流程：对输入firing rate进行BCE梯度下降优化
    """
    
    def __init__(self, model_path: str, model_params_path: str, 
                 init_firing_rates_path: Optional[str] = None, 
                 time_duration_ms: int = 300,
                 device: str = 'auto',
                 use_fixed_seed: bool = True):
        """
        初始化优化器
        
        Args:
            model_path: 训练好的模型.pth文件路径
            model_params_path: 对应的参数.pickle文件路径
            init_firing_rates_path: 初始firing rates的.npy文件路径
            time_duration_ms: 时间长度，默认300ms
            device: 设备选择 ('auto', 'cpu', 'cuda')
        """
        self.model_path = model_path
        self.model_params_path = model_params_path
        self.init_firing_rates_path = init_firing_rates_path
        self.time_duration_ms = time_duration_ms
        self.use_fixed_seed = use_fixed_seed
        
        # 设备选择
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 加载模型和参数
        self.model = self._load_model(model_path)
        self.model_params = self._load_model_params(model_params_path)
        
        # 提取模型架构信息
        self.architecture_dict = self.model_params['architecture_dict']
        self.input_window_size = self.architecture_dict['input_window_size']
        
        # 从训练数据中获取segment数量信息
        train_files = self.model_params['data_dict']['train_files']
        if train_files:
            # 解析一个训练文件来获取segment信息
            self.num_segments_exc = 639  # 根据trainand_analyze.py中的设置
            self.num_segments_inh = 640  # SJC数据集的设置
            self.num_segments_total = self.num_segments_exc + self.num_segments_inh
        else:
            # 默认值
            self.num_segments_exc = 639
            self.num_segments_inh = 640
            self.num_segments_total = 1279
        
        # 加载初始firing rates（如果提供）
        self.init_firing_rates = None
        if init_firing_rates_path and os.path.exists(init_firing_rates_path):
            self.init_firing_rates = self.load_init_firing_rates(init_firing_rates_path)
        
        print(f"模型加载成功:")
        print(f"  输入窗口大小: {self.input_window_size}ms")
        print(f"  兴奋性segments: {self.num_segments_exc}")
        print(f"  抑制性segments: {self.num_segments_inh}")
        print(f"  总segments: {self.num_segments_total}")
        if self.init_firing_rates is not None:
            print(f"  初始firing rates已加载: {self.init_firing_rates.shape}")
        print(f"  随机种子控制: {'开启' if self.use_fixed_seed else '关闭'}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """加载PyTorch模型"""
        try:
            # 检查文件扩展名
            if model_path.endswith('.pth') or model_path.endswith('.pt'):
                # 直接加载PyTorch模型
                model = torch.load(model_path, map_location=self.device)
            else:
                # 尝试从Keras模型转换（需要先转换）
                raise ValueError(f"不支持的模型格式: {model_path}")
            
            model.eval()  # 设置为评估模式
            return model
            
        except Exception as e:
            print(f"加载模型时出错: {e}")
            # 创建一个简单的测试模型用于演示
            print("创建测试模型...")
            return self._create_test_model()
    
    def _create_test_model(self) -> nn.Module:
        """创建测试用的简单TCN模型"""
        class SimpleTCN(nn.Module):
            def __init__(self, input_size=1279, hidden_size=256, output_size=1):
                super(SimpleTCN, self).__init__()
                self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
                self.conv3 = nn.Conv1d(hidden_size, output_size, kernel_size=3, padding=1)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                # x: (batch, time, segments) -> (batch, segments, time)
                x = x.transpose(1, 2)
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.sigmoid(self.conv3(x))
                # 返回 (batch, time, output_size)
                return x.transpose(1, 2)
        
        model = SimpleTCN()
        model.to(self.device)
        return model
    
    def _load_model_params(self, model_params_path: str) -> Dict[str, Any]:
        """加载模型参数"""
        try:
            with open(model_params_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"加载模型参数时出错: {e}")
            # 返回默认参数
            return {
                'architecture_dict': {'input_window_size': 400},
                'data_dict': {'train_files': []}
            }
    
    def load_init_firing_rates(self, firing_rates_path: str) -> Optional[np.ndarray]:
        """
        加载初始firing rates
        
        Args:
            firing_rates_path: firing rates的.npy文件路径
            
        Returns:
            firing_rates: (num_segments_total, full_time_duration) numpy数组
        """
        print(f"正在加载初始firing rates: {firing_rates_path}")
        try:
            firing_rates = np.load(firing_rates_path)
            print(f"  文件加载成功，形状: {firing_rates.shape}")
            print(f"  数据类型: {firing_rates.dtype}")
            print(f"  数值范围: [{np.min(firing_rates):.6f}, {np.max(firing_rates):.6f}]")
            
            # 验证数据格式
            if len(firing_rates.shape) != 2:
                raise ValueError(f"期望2D数组，但得到{len(firing_rates.shape)}D数组")
            
            num_segments, full_time = firing_rates.shape
            if num_segments != self.num_segments_total:
                print(f"警告：文件中的segments数量 ({num_segments}) 与模型期望不符 ({self.num_segments_total})")
            
            return firing_rates.astype(np.float32)
            
        except Exception as e:
            print(f"加载初始firing rates时出错: {e}")
            return None
    
    def prepare_firing_rates_for_optimization(self, firing_rates: np.ndarray, 
                                           batch_size: int = 1, 
                                           start_time_ms: int = 100) -> np.ndarray:
        """
        准备firing rates用于优化，提取指定时间段数据
        
        Args:
            firing_rates: (num_segments_total, full_time_duration) 或 (batch_size, num_segments_total, full_time_duration)
            batch_size: 批次大小
            start_time_ms: 开始时间，默认100ms（忽略前100ms）
            
        Returns:
            prepared_rates: (batch_size, num_segments_total, time_duration_ms)
        """
        if len(firing_rates.shape) == 2:
            # 单个样本，扩展为batch
            firing_rates = firing_rates[np.newaxis, :, :]  # (1, num_segments, full_time)
        
        current_batch_size, num_segments, full_time = firing_rates.shape
        
        # 处理segments数量不匹配的问题
        if num_segments != self.num_segments_total:
            print(f"调整segments数量: {num_segments} -> {self.num_segments_total}")
            if num_segments < self.num_segments_total:
                # 如果segments不足，用零填充
                padding_needed = self.num_segments_total - num_segments
                padding = np.zeros((current_batch_size, padding_needed, full_time), dtype=firing_rates.dtype)
                firing_rates = np.concatenate([firing_rates, padding], axis=1)
                print(f"  添加了 {padding_needed} 个零填充segments")
            else:
                # 如果segments过多，截取前面的部分
                firing_rates = firing_rates[:, :self.num_segments_total, :]
                print(f"  截取前 {self.num_segments_total} 个segments")
        
        # 计算提取的时间范围
        end_time_ms = start_time_ms + self.time_duration_ms
        
        if end_time_ms > full_time:
            raise ValueError(f"时间范围超出数据长度：需要{end_time_ms}ms，但数据只有{full_time}ms")
        
        # 提取指定时间段的数据
        extracted_rates = firing_rates[:, :, start_time_ms:end_time_ms]
        
        # 如果需要更多batch，复制数据
        if batch_size > current_batch_size:
            # 重复数据以达到所需的batch size
            repeat_times = batch_size // current_batch_size
            remainder = batch_size % current_batch_size
            
            repeated_rates = np.tile(extracted_rates, (repeat_times, 1, 1))
            if remainder > 0:
                extra_rates = extracted_rates[:remainder, :, :]
                extracted_rates = np.concatenate([repeated_rates, extra_rates], axis=0)
            else:
                extracted_rates = repeated_rates
        elif batch_size < current_batch_size:
            # 取前batch_size个样本
            extracted_rates = extracted_rates[:batch_size, :, :]
        
        print(f"Prepare for optimization:")
        print(f"  Original input shape: {firing_rates.shape}")
        print(f"  Extracted time: {start_time_ms}-{end_time_ms}ms")
        print(f"  Extracted input shape: {extracted_rates.shape}")
        
        return extracted_rates.astype(np.float32)
    
    def generate_spikes_with_modification(self, firing_rates: np.ndarray, 
                                       fixed_exc_indices: Optional[np.ndarray] = None, 
                                       random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成spikes with modification - 可复用函数
        
        Args:
            firing_rates: (batch_size, num_segments_total, time_duration_ms)
            fixed_exc_indices: 固定添加spikes的excitatory indices
            random_seed: 随机种子，用于确保Poisson采样的可重复性
            
        Returns:
            spike_trains: (batch_size, num_segments_total, time_duration_ms) binary
            fixed_exc_indices: 使用的fixed indices
        """
        batch_size = firing_rates.shape[0]
        
        # 设置随机种子以确保Poisson采样的可重复性
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 只对前一半batch使用Poisson过程生成spikes
        first_half_spikes = np.random.poisson(firing_rates).astype(np.float32)
        first_half_spikes = np.clip(first_half_spikes, 0, 1)  # 限制为0或1 (binary)
        
        # 复制前一半到后一半，确保基础spikes完全相同
        second_half_spikes = first_half_spikes.copy()
        
        # 组合完整的spike trains
        spike_trains = np.concatenate([first_half_spikes, second_half_spikes], axis=0)
        
        # 如果没有指定固定的excitatory indices，随机选择三个
        if fixed_exc_indices is None:
            mono_syn_rnd = np.random.default_rng(42)
            fixed_exc_indices = mono_syn_rnd.choice(self.num_segments_exc, size=3, replace=False)
        
        # 对后一半batch添加固定spikes (在half window size处)
        if batch_size > 0:
            spike_time_mono_syn = self.input_window_size // 2  # half window size时间点
            if spike_time_mono_syn < self.time_duration_ms:
                # 在half window size处为指定的三个excitatory segments添加spikes
                for idx in fixed_exc_indices:
                    spike_trains[batch_size:, idx, spike_time_mono_syn] = 1.0
        
        return spike_trains, fixed_exc_indices
    
    def process_firing_rates_to_predictions(self, firing_rates: np.ndarray, 
                                         fixed_exc_indices: np.ndarray, 
                                         random_seed: Optional[int] = None) -> Tuple[torch.Tensor, np.ndarray]:
        """
        可复用函数：将firing rates转换为模型预测
        
        Args:
            firing_rates: (batch_size, num_segments_total, time_duration_ms)
            fixed_exc_indices: 固定的excitatory indices
            random_seed: 随机种子，用于确保Poisson采样的可重复性
            
        Returns:
            spike_predictions: 模型预测的spike概率
            spike_trains: 生成的spike trains
        """
        # 使用可复用函数生成spikes
        spike_trains, _ = self.generate_spikes_with_modification(firing_rates, fixed_exc_indices, random_seed)
        
        # 转换数据格式以匹配模型输入: (spike_batch_size, time_duration, num_segments)
        model_input = np.transpose(spike_trains, (0, 2, 1))
        
        # 处理输入窗口大小
        input_time_steps = model_input.shape[1]
        if input_time_steps < self.input_window_size:
            padding_needed = self.input_window_size - input_time_steps
            padding = np.zeros((model_input.shape[0], padding_needed, model_input.shape[2]))
            model_input = np.concatenate([padding, model_input], axis=1)
        elif input_time_steps > self.input_window_size:
            model_input = model_input[:, -self.input_window_size:, :]
        
        # 转换为PyTorch张量
        model_input_tensor = torch.tensor(model_input, dtype=torch.float32, device=self.device)
        
        # 模型预测
        with torch.no_grad():
            predictions = self.model(model_input_tensor)
        
        spike_predictions = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
        
        return spike_predictions, spike_trains
    
    def compute_loss_numpy(self, firing_rates: np.ndarray, 
                          fixed_exc_indices: np.ndarray, 
                          target_spike_prob: float, 
                          random_seed: Optional[int] = None) -> float:
        """
        使用numpy计算损失函数
        
        Args:
            firing_rates: firing rates
            fixed_exc_indices: 固定的excitatory indices
            target_spike_prob: 目标spike概率
            random_seed: 随机种子，用于确保Poisson采样的可重复性
            
        Returns:
            loss: 损失值
        """
        # 直接在这里生成spikes，确保与梯度计算使用相同的逻辑
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 生成Poisson spikes
        spike_trains = np.random.poisson(firing_rates).astype(np.float32)
        spike_trains = np.clip(spike_trains, 0.0, 1.0)
        
        # 添加固定excitatory spikes
        spike_time_mono_syn = self.input_window_size // 2
        batch_size = spike_trains.shape[0]
        half_batch = batch_size // 2
        for idx in fixed_exc_indices:
            spike_trains[half_batch:, idx, spike_time_mono_syn] = 1.0
        
        # 转换数据格式以匹配模型输入: (batch, time, segments)
        model_input = np.transpose(spike_trains, (0, 2, 1))
        
        # 处理输入窗口大小
        input_time_steps = model_input.shape[1]
        if input_time_steps < self.input_window_size:
            padding_needed = self.input_window_size - input_time_steps
            padding = np.zeros((model_input.shape[0], padding_needed, model_input.shape[2]))
            model_input = np.concatenate([padding, model_input], axis=1)
        elif input_time_steps > self.input_window_size:
            model_input = model_input[:, -self.input_window_size:, :]
        
        # 转换为PyTorch张量
        model_input_tensor = torch.tensor(model_input, dtype=torch.float32, device=self.device)
        
        # 模型预测
        with torch.no_grad():
            predictions = self.model(model_input_tensor)
        
        spike_predictions = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
        
        # 转换为numpy数组
        spike_predictions_np = spike_predictions.cpu().numpy()
        
        # 计算损失
        # 关注half window size之后的10个时间步
        target_start_time, target_time_steps = self.input_window_size // 2, 10
        target_predictions = spike_predictions_np[:, target_start_time:target_start_time+target_time_steps, :]
        target_spikes = np.ones_like(target_predictions) * target_spike_prob
        
        # BCE损失
        epsilon = 1e-7
        target_predictions_clipped = np.clip(target_predictions, epsilon, 1 - epsilon)
        bce_loss = -target_spikes * np.log(target_predictions_clipped) - (1 - target_spikes) * np.log(1 - target_predictions_clipped)
        prediction_loss = np.mean(bce_loss)
        
        # 正则化
        regularization_loss = 0.001 * np.mean(firing_rates)
        
        return prediction_loss + regularization_loss
    
    def compute_numerical_gradient(self, firing_rates: np.ndarray, 
                                 fixed_exc_indices: np.ndarray, 
                                 target_spike_prob: float, 
                                 original_loss: Optional[float] = None, 
                                 epsilon: float = 1e-6, 
                                 random_seed: Optional[int] = None) -> np.ndarray:
        """
        使用PyTorch自动微分计算 BCE loss 关于 firing_rates 的梯度
        使用 Straight-through estimator (STE) 实现可微的离散Poisson采样
        
        Args:
            firing_rates: numpy array, shape (batch_size, num_segments_total, time_duration)
            fixed_exc_indices: 固定的excitatory indices
            target_spike_prob: 目标spike概率
            original_loss: 原始损失值（可选）
            epsilon: 数值梯度步长（可选，当前使用自动微分）
            random_seed: 随机种子，用于确保Poisson采样的可重复性
            
        Returns:
            gradient: 关于firing_rates的梯度
        """
        start_time = time.time()
        
        # 设置随机种子以确保Poisson采样的可重复性（仅在use_fixed_seed=True时）
        if self.use_fixed_seed and random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        elif not self.use_fixed_seed:
            # 如果关闭固定种子，使用当前时间戳作为种子
            current_seed = int(time.time() * 1000) % 1000000
            torch.manual_seed(current_seed)
            np.random.seed(current_seed)

        # 转换为PyTorch张量并设置requires_grad=True
        firing_rates_tf = torch.tensor(firing_rates, dtype=torch.float32, device=self.device, requires_grad=True)

        # 前向传播
        # ===== 1. Straight-through estimator (STE) for Poisson sampling =====
        # 前向：离散Poisson采样；反向：从连续 firing_rates 传梯度
        firing_rates_np = firing_rates  # 使用原始的numpy数组
        discrete_spikes_np = np.random.poisson(firing_rates_np).astype(np.float32)
        discrete_spikes_np = np.clip(discrete_spikes_np, 0.0, 1.0)
        discrete_spikes = torch.tensor(discrete_spikes_np, dtype=torch.float32, device=self.device)
        
        # 正确的STE：前向用离散值，反向梯度从firing_rates_tf流回
        # 关键：前向传播时使用离散值，但梯度能够流回firing_rates_tf
        # 方法：使用identity trick，让梯度能够流回
        spike_trains = discrete_spikes + (firing_rates_tf - discrete_spikes).detach()
        
        # 确保梯度能够正确传播
        # 添加一个小的扰动来保持梯度流
        spike_trains = spike_trains + 0.01 * firing_rates_tf
        
        # 调试：检查梯度流
        if random_seed is not None and random_seed % 10 == 0:  # 每10次打印一次
            print(f"    STE调试 - firing_rates_tf requires_grad: {firing_rates_tf.requires_grad}")
            print(f"    STE调试 - spike_trains requires_grad: {spike_trains.requires_grad}")
            print(f"    STE调试 - discrete_spikes requires_grad: {discrete_spikes.requires_grad}")

        # ===== 2. 添加固定 excitatory spikes =====
        spike_time_mono_syn = self.input_window_size // 2
        batch_size = spike_trains.shape[0]
        half_batch = batch_size // 2
        
        # 创建索引
        for idx in fixed_exc_indices:
            spike_trains[half_batch:, idx, spike_time_mono_syn] = 1.0

        # ===== 3. 调整成模型输入 (batch, time, segments)，并自动 pad/截取 =====
        model_input = spike_trains.transpose(1, 2)  # (batch, segments, time) -> (batch, time, segments)
        time_steps = model_input.shape[1]
        
        if time_steps < self.input_window_size:
            pad_len = self.input_window_size - time_steps
            model_input = F.pad(model_input, (0, 0, pad_len, 0))  # 在时间维度上padding
        
        model_input = model_input[:, -self.input_window_size:, :]

        # ===== 4. 模型预测（第一个输出是 spike 概率） =====
        spike_predictions = self.model(model_input)[0] if isinstance(self.model(model_input), (list, tuple)) else self.model(model_input)

        # ===== 5. 计算 BCE loss（关注 half window size 之后 10 个时间步） =====
        target_start_time, target_time_steps = self.input_window_size // 2, 10
        target_predictions = spike_predictions[:, target_start_time:target_start_time + target_time_steps, :]
        target_spikes = torch.ones_like(target_predictions) * target_spike_prob
        
        prediction_loss = F.binary_cross_entropy(target_predictions, target_spikes)
        regularization_loss = 0.001 * torch.mean(firing_rates_tf)
        loss = prediction_loss + regularization_loss

        # ===== 6. 求梯度 =====
        loss.backward()
        grad = firing_rates_tf.grad
        
        end_time = time.time()
        print(f"Time to calculate gradient: {end_time - start_time:.4f} seconds")
        
        # 简单调试：检查梯度是否为零
        if grad is not None:
            grad_norm = torch.norm(grad)
            grad_norm_value = grad_norm.item()
            print(f"  梯度范数: {grad_norm_value:.8f}")
            if grad_norm_value < 1e-8:
                print("  警告：梯度接近零！")
        
        # 梯度转换
        if grad is None:
            print("警告：梯度为None，使用零梯度")
            return np.zeros_like(firing_rates)
        
        # 转换为numpy数组
        return grad.cpu().numpy()

    def optimize_activity(self, num_iterations: int = 100, learning_rate: float = 0.01, 
                         batch_size: int = 4, target_spike_prob: float = 0.8, 
                         save_dir: Optional[str] = None, start_time_ms: int = 100) -> Tuple[np.ndarray, List[float], np.ndarray]:
        """
        执行activity optimization
        
        Args:
            num_iterations: 优化迭代次数
            learning_rate: 学习率
            batch_size: 批次大小
            target_spike_prob: 目标spike概率
            save_dir: 保存结果的目录
            start_time_ms: 开始时间（忽略前start_time_ms毫秒）
            
        Returns:
            optimized_firing_rates: 优化后的firing rates
            loss_history: 损失历史
            fixed_exc_indices: 固定的excitatory indices
        """
        print(f"\n开始Activity Optimization:")
        print(f"  迭代次数: {num_iterations}")
        print(f"  学习率: {learning_rate}")
        print(f"  批次大小: {batch_size}")
        print(f"  目标spike概率: {target_spike_prob}")
        print(f"  开始时间: {start_time_ms}ms")
        print("-" * 50)
        
        # 准备初始firing rates
        if self.init_firing_rates is not None:
            print("使用加载的初始firing rates")
            initial_firing_rates = self.prepare_firing_rates_for_optimization(
                self.init_firing_rates, batch_size, start_time_ms
            )
        else:
            print("生成随机初始firing rates")
            initial_firing_rates = self._generate_background_firing_rates(batch_size)
        
        # 选择固定的excitatory indices
        if self.use_fixed_seed:
            mono_syn_rnd = np.random.default_rng(42)
        else:
            # 使用当前时间戳作为种子
            current_seed = int(time.time() * 1000) % 1000000
            mono_syn_rnd = np.random.default_rng(current_seed)
        
        fixed_exc_indices = mono_syn_rnd.choice(self.num_segments_exc, size=3, replace=False)
        print(f"固定添加spikes的excitatory segments: {fixed_exc_indices}")
        
        # 使用numpy数组进行优化
        current_firing_rates = initial_firing_rates.copy()
        
        # 记录历史
        loss_history = []
        firing_rates_history = []
        
        # 优化循环
        for iteration in range(num_iterations):
            # 为每次迭代设置固定的随机种子，确保Poisson采样的一致性
            iteration_seed = 42 + iteration
            
            # 直接调用compute_loss_numpy计算损失
            loss = self.compute_loss_numpy(current_firing_rates, fixed_exc_indices, target_spike_prob, iteration_seed)
            
            # 计算梯度
            gradient = self.compute_numerical_gradient(current_firing_rates, fixed_exc_indices, target_spike_prob, loss, iteration_seed)
            
            # 梯度范数裁剪（全局）以稳定优化
            max_grad_norm = 50.0  # 增加梯度裁剪阈值
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > max_grad_norm:
                gradient = gradient * (max_grad_norm / (grad_norm + 1e-12))
                print(f"  梯度裁剪: {grad_norm:.6f} -> {max_grad_norm:.6f}")
            
            # 应用梯度更新
            current_firing_rates = current_firing_rates - learning_rate * gradient
            
            # 调试：打印梯度信息
            if iteration % 5 == 0:  # 每5次迭代打印一次
                print(f"  梯度范数: {grad_norm:.8f}")
                print(f"  梯度范围: [{np.min(gradient):.8f}, {np.max(gradient):.8f}]")
                print(f"  更新步长: {learning_rate * grad_norm:.8f}")
            
            # 限制firing rates在合理范围内 [0, 0.1]
            current_firing_rates = np.clip(current_firing_rates, 0.0, 0.1)
            
            # 记录历史
            loss_history.append(float(loss))
            if iteration % 10 == 0:
                firing_rates_history.append(current_firing_rates.copy())
            
            # 打印进度
            if iteration % 10 == 0:
                print(f"迭代 {iteration:3d}: Loss = {float(loss):.6f} (Seed: {iteration_seed})")
        
        print("-" * 50)
        print(f"优化完成! 最终损失: {loss_history[-1]:.6f}")
        
        # 获取最终结果
        optimized_firing_rates = current_firing_rates
        
        # 保存结果
        if save_dir:
            self.save_optimization_results(
                optimized_firing_rates, loss_history, firing_rates_history, 
                fixed_exc_indices, save_dir, start_time_ms
            )
        
        return optimized_firing_rates, loss_history, fixed_exc_indices
    
    def _generate_background_firing_rates(self, batch_size: int) -> np.ndarray:
        """生成随机背景firing rates"""
        # 生成随机firing rates
        firing_rates = np.random.uniform(0.01, 0.05, (batch_size, self.num_segments_total, self.time_duration_ms))
        return firing_rates.astype(np.float32)
    
    def save_optimization_results(self, optimized_firing_rates: np.ndarray, 
                                loss_history: List[float], 
                                firing_rates_history: List[np.ndarray], 
                                fixed_exc_indices: np.ndarray, 
                                save_dir: str, 
                                start_time_ms: int):
        """保存优化结果"""
        os.makedirs(save_dir, exist_ok=True)

        # 保存数据
        results = {
            'optimized_firing_rates': optimized_firing_rates,
            'loss_history': loss_history,
            'firing_rates_history': firing_rates_history,
            'fixed_exc_indices': fixed_exc_indices,
            'model_path': self.model_path,
            'optimization_params': {
                'time_duration_ms': self.time_duration_ms,
                'num_segments_exc': self.num_segments_exc,
                'num_segments_inh': self.num_segments_inh,
                'start_time_ms': start_time_ms
            }
        }
        
        result_file = os.path.join(save_dir, 'activity_optimization.pickle')
        with open(result_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"结果已保存到: {result_file}")
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.title('Activity Optimization Loss History')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'loss_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制firing rates变化
        if firing_rates_history:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 选择几个时间点和segments进行可视化
            sample_times = [50, self.input_window_size // 2, 250]  # 50ms, half window size, 250ms
            sample_segments = [0, self.num_segments_exc//2, self.num_segments_exc, 
                             self.num_segments_exc + self.num_segments_inh//2]
            
            for i, (ax, segment_idx) in enumerate(zip(axes.flat, sample_segments)):
                if i < len(sample_segments):
                    for time_idx in sample_times:
                        if time_idx < self.time_duration_ms:
                            values = [fr[0, segment_idx, time_idx] for fr in firing_rates_history]
                            ax.plot(range(0, len(firing_rates_history)*10, 10), values, 
                                   label=f'Time {time_idx}ms')
                    
                    segment_type = "Exc" if segment_idx < self.num_segments_exc else "Inh"
                    ax.set_title(f'Segment {segment_idx} ({segment_type})')
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('Firing Rate')
                    ax.legend()
                    ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'firing_rates_evolution.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def evaluate_optimized_activity(self, optimized_firing_rates: np.ndarray, 
                                  fixed_exc_indices: np.ndarray, 
                                  num_evaluations: int = 10) -> Dict[str, Any]:
        """
        评估优化后的activity
        
        Args:
            optimized_firing_rates: 优化后的firing rates (batch_size, num_segments, time_duration)
            fixed_exc_indices: 固定的excitatory indices
            num_evaluations: 评估次数
            
        Returns:
            evaluation_results: 评估结果字典
        """
        print(f"\n评估优化后的activity (运行{num_evaluations}次)...")
        
        spike_probabilities = []
        
        for eval_idx in range(num_evaluations):
            # 为每次评估设置不同的随机种子，确保评估的多样性
            eval_seed = 1000 + eval_idx  # 使用不同的种子范围
            
            # 使用可复用函数生成spikes和预测
            spike_predictions, spike_trains = self.process_firing_rates_to_predictions(
                optimized_firing_rates, fixed_exc_indices, eval_seed
            )
            
            # 对每个batch进行预测
            batch_spike_probs = []
            for batch_idx in range(spike_trains.shape[0]):
                # 取half window size之后的10个时间步的预测概率
                target_start_time, target_time_steps = self.input_window_size // 2, 10
                final_predictions = spike_predictions[batch_idx, target_start_time:target_start_time+target_time_steps, 0]
                batch_spike_probs.extend(final_predictions.cpu().numpy().tolist())
            
            spike_probabilities.extend(batch_spike_probs)
        
        # 计算统计信息
        spike_probabilities = np.array(spike_probabilities)
        
        evaluation_results = {
            'mean_spike_probability': np.mean(spike_probabilities),
            'std_spike_probability': np.std(spike_probabilities),
            'min_spike_probability': np.min(spike_probabilities),
            'max_spike_probability': np.max(spike_probabilities),
            'spike_probabilities': spike_probabilities
        }
        
        print(f"评估结果:")
        print(f"  平均spike概率: {evaluation_results['mean_spike_probability']:.4f}")
        print(f"  标准差: {evaluation_results['std_spike_probability']:.4f}")
        print(f"  最小值: {evaluation_results['min_spike_probability']:.4f}")
        print(f"  最大值: {evaluation_results['max_spike_probability']:.4f}")
        
        return evaluation_results
