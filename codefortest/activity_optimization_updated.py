import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
import pickle
import os
import glob
import matplotlib.pyplot as plt
import time
from datetime import datetime
from fit_CNN import parse_sim_experiment_file
from utils.visualization_utils import visualize_firing_rates_trace, visualize_firing_rates_heatmap


def _enable_gpu_memory_growth():
    """
    Enable TensorFlow GPU memory growth to reduce OOM caused by upfront allocation
    and fragmentation. Safe to call when no GPU is present.
    """
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                # Best-effort; ignore if not supported in the TF build
                pass
    except Exception:
        # No GPU or API not available; ignore
        pass

_enable_gpu_memory_growth()

class ActivityOptimizer:
    """
    基于已训练的TCN模型进行activity optimization的类
    实现图中描述的优化流程：对输入firing rate进行BCE梯度下降优化
    """
    
    def __init__(self, model_path, model_params_path, init_firing_rates_path=None, time_duration_ms=300, use_cpu_for_grad=True, use_fixed_seed=True):
        """
        初始化优化器
        
        Args:
            model_path: 训练好的模型.h5文件路径
            model_params_path: 对应的参数.pickle文件路径
            init_firing_rates_path: 初始firing rates的.npy文件路径
            time_duration_ms: 时间长度，默认300ms
        """
        self.model_path = model_path
        self.model_params_path = model_params_path
        self.init_firing_rates_path = init_firing_rates_path
        self.time_duration_ms = time_duration_ms # We want this to be identical to the input window size
        self.use_cpu_for_grad = False # use_cpu_for_grad
        self.use_fixed_seed = use_fixed_seed
        
        # 加载模型和参数
        self.model = load_model(model_path)
        
        # 强制将模型移动到CPU设备上，避免GPU OOM
        with tf.device('/CPU:0'):
            # 重新构建模型权重，确保在CPU上
            self.model = tf.keras.models.clone_model(self.model)
            self.model.set_weights(self.model.get_weights())
        
        with open(model_params_path, 'rb') as f:
            self.model_params = pickle.load(f)
        
        # 提取模型架构信息
        self.architecture_dict = self.model_params['architecture_dict']
        self.input_window_size = self.architecture_dict['input_window_size']
        
        # 从训练数据中获取segment数量信息
        train_files = self.model_params['data_dict']['train_files']
        if train_files:
            # 解析一个训练文件来获取segment信息
            X_sample, _, _ = parse_sim_experiment_file(train_files[0])
            self.num_segments_exc = 639  # 根据train_and_analyze.py中的设置
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
    
    def load_init_firing_rates(self, firing_rates_path):
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
    
    def prepare_firing_rates_for_optimization(self, firing_rates, batch_size=1, start_time_ms=100):
        """
        准备firing rates用于优化，提取后300ms数据（忽略前100ms）
        
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
    
    def generate_spikes_with_modification(self, firing_rates, fixed_exc_indices=None, random_seed=None):
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
        
        # 设置随机种子以确保Poisson采样的可重复性（仅在use_fixed_seed=True时）
        if self.use_fixed_seed and random_seed is not None:
            np.random.seed(random_seed)
        elif not self.use_fixed_seed:
            # 如果关闭固定种子，使用当前时间戳作为种子
            current_seed = int(time.time() * 1000) % 1000000
            np.random.seed(current_seed)
        
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
    
    def process_firing_rates_to_predictions(self, firing_rates, fixed_exc_indices, random_seed=None):
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
        
        # 模型预测（强制在CPU上执行以避免GPU OOM）
        with tf.device('/CPU:0'):
            # 使用 __call__ 方法而不是 predict 方法，确保设备上下文生效
            predictions = self.model(model_input, training=False)
        spike_predictions = predictions[0]  # 第一个输出是spike预测
        
        return spike_predictions, spike_trains
    
    def compute_loss_numpy(self, firing_rates, fixed_exc_indices, target_spike_prob, random_seed=None):
        """
        使用numpy计算损失函数（在CPU上执行以避免GPU OOM）
        
        Args:
            firing_rates: firing rates
            fixed_exc_indices: 固定的excitatory indices
            target_spike_prob: 目标spike概率
            random_seed: 随机种子，用于确保Poisson采样的可重复性
            
        Returns:
            loss: 损失值
        """
        # 直接在这里生成spikes，确保与梯度计算使用相同的逻辑
        # 设置随机种子（仅在use_fixed_seed=True时）
        if self.use_fixed_seed and random_seed is not None:
            np.random.seed(random_seed)
        elif not self.use_fixed_seed:
            # 如果关闭固定种子，使用当前时间戳作为种子
            current_seed = int(time.time() * 1000) % 1000000
            np.random.seed(current_seed)
        
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
        
        # 模型预测（在CPU上执行）
        with tf.device('/CPU:0'):
            predictions = self.model(model_input, training=False)
        spike_predictions = predictions[0]
        
        # 将TensorFlow张量转换为numpy数组
        if hasattr(spike_predictions, 'numpy'):
            spike_predictions = spike_predictions.numpy()
        elif hasattr(tf.keras.backend, 'eval'):
            spike_predictions = tf.keras.backend.eval(spike_predictions)
        else:
            spike_predictions = np.array(spike_predictions)
        
        # 计算损失
        # 关注half window size之后的10个时间步
        target_start_time, target_time_steps = self.input_window_size // 2, 10  # 关注mono synaptic spike之后的10个时间步
        target_predictions = spike_predictions[:, target_start_time:target_start_time+target_time_steps, :]
        target_spikes = np.ones_like(target_predictions) * target_spike_prob
        
        # BCE损失
        epsilon = 1e-7
        target_predictions_clipped = np.clip(target_predictions, epsilon, 1 - epsilon)
        bce_loss = -target_spikes * np.log(target_predictions_clipped) - (1 - target_spikes) * np.log(1 - target_predictions_clipped)
        prediction_loss = np.mean(bce_loss)
        
        # 正则化
        regularization_loss = 0.001 * np.mean(firing_rates)
        
        return prediction_loss + regularization_loss
    
    def compute_numerical_gradient(self, firing_rates, fixed_exc_indices, target_spike_prob, original_loss=None, epsilon=1e-6, random_seed=None):
        """
        使用自动微分计算 BCE loss 关于 firing_rates 的梯度
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
        # 固定模型参数（不更新，但梯度可回传到输入）
        for layer in self.model.layers:
            layer.trainable = False

        # 设置随机种子以确保Poisson采样的可重复性（仅在use_fixed_seed=True时）
        if self.use_fixed_seed and random_seed is not None:
            tf.random.set_seed(random_seed)
            np.random.seed(random_seed)
        elif not self.use_fixed_seed:
            # 如果关闭固定种子，使用当前时间戳作为种子
            current_seed = int(time.time() * 1000) % 1000000
            tf.random.set_seed(current_seed)
            np.random.seed(current_seed)

        firing_rates_tf = tf.convert_to_tensor(firing_rates, dtype=tf.float32)

        # Place the forward and backward on CPU if configured to avoid GPU OOM
        device_to_use = '/CPU:0' if self.use_cpu_for_grad else '/GPU:0'
        with tf.device(device_to_use):
            with tf.GradientTape() as tape:
                tape.watch(firing_rates_tf)

                # ===== 1. Straight-through estimator (STE) for Poisson sampling =====
                # 前向：离散Poisson采样；反向：从连续 firing_rates 传梯度
                firing_rates_np = firing_rates  # 使用原始的numpy数组
                discrete_spikes_np = np.random.poisson(firing_rates_np).astype(np.float32)
                discrete_spikes_np = np.clip(discrete_spikes_np, 0.0, 1.0)
                discrete_spikes = tf.convert_to_tensor(discrete_spikes_np, dtype=tf.float32)
                
                # 正确的STE：前向用离散值，反向梯度从firing_rates_tf流回
                # 关键：前向传播时使用离散值，但梯度能够流回firing_rates_tf
                # 方法：使用identity trick，让梯度能够流回
                # 注意：不能使用tf.stop_gradient，这会阻断梯度流
                spike_trains = discrete_spikes + (firing_rates_tf - discrete_spikes)
                
                # 确保梯度能够正确传播
                # 添加一个小的扰动来保持梯度流
                spike_trains = spike_trains + 0.01 * firing_rates_tf

                # ===== 2. 添加固定 excitatory spikes =====
                spike_time_mono_syn = self.input_window_size // 2
                batch_size = tf.shape(spike_trains)[0]
                half_batch = batch_size // 2
                batch_indices = tf.range(half_batch, batch_size, dtype=tf.int32)
                fixed_exc_tf = tf.constant(fixed_exc_indices, dtype=tf.int32)
                b_grid, idx_grid = tf.meshgrid(batch_indices, fixed_exc_tf, indexing='ij')
                b_coords = tf.reshape(b_grid, [-1])
                idx_coords = tf.reshape(idx_grid, [-1])
                time_coords = tf.fill([tf.shape(b_coords)[0]], spike_time_mono_syn)
                indices = tf.stack([b_coords, idx_coords, time_coords], axis=1)
                updates = tf.ones([tf.shape(indices)[0]], dtype=tf.float32)
                spike_trains = tf.tensor_scatter_nd_update(spike_trains, indices, updates)

                # ===== 3. 调整成模型输入 (batch, time, segments)，并自动 pad/截取 =====
                model_input = tf.transpose(spike_trains, perm=[0, 2, 1])
                time_steps = tf.shape(model_input)[1]
                pad_len = tf.maximum(self.input_window_size - time_steps, 0)
                model_input = tf.pad(model_input, [[0, 0], [pad_len, 0], [0, 0]])
                model_input = model_input[:, -self.input_window_size:, :]

                # ===== 4. 模型预测（第一个输出是 spike 概率） =====
                spike_predictions = self.model(model_input, training=False)[0]

                # ===== 5. 计算 BCE loss（关注 half window size 之后 10 个时间步） =====
                target_start_time, target_time_steps = self.input_window_size // 2, 10
                target_predictions = spike_predictions[:, target_start_time:target_start_time + target_time_steps, :]
                target_spikes = tf.ones_like(target_predictions) * target_spike_prob
                prediction_loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(target_spikes, target_predictions)
                )
                regularization_loss = 0.001 * tf.reduce_mean(firing_rates_tf)
                loss = prediction_loss + regularization_loss

        # ===== 6. 求梯度 =====
        grad = tape.gradient(loss, firing_rates_tf)
        end_time = time.time()
        print(f"Time to calculate gradient: {end_time - start_time:.4f} seconds")
        
        # 兼容不同TensorFlow版本的张量转换方法
        if grad is None:
            print("警告：梯度为None，使用零梯度")
            return np.zeros_like(firing_rates)
        
        # 尝试不同的转换方法
        try:
            # 方法1: 直接调用numpy()方法
            if hasattr(grad, 'numpy'):
                return grad.numpy()
            # 方法2: 使用tf.keras.backend.eval
            elif hasattr(tf.keras.backend, 'eval'):
                return tf.keras.backend.eval(grad)
            # 方法3: 使用tf.make_ndarray
            elif hasattr(tf, 'make_ndarray'):
                return tf.make_ndarray(grad)
            # 方法4: 最后备选方案
            else:
                return np.array(grad)
        except Exception as e:
            print(f"转换梯度到numpy时出错: {e}")
            print("尝试使用tf.keras.backend.eval作为最后备选方案...")
            try:
                return tf.keras.backend.eval(grad)
            except Exception as e2:
                print(f"所有转换方法都失败: {e2}")
                print("返回零梯度")
                return np.zeros_like(firing_rates)

    def optimize_activity(self, num_iterations=100, learning_rate=0.01, batch_size=4, 
                         target_spike_prob=0.8, save_dir=None, start_time_ms=100):
        """
        执行activity optimization - 使用纯numpy操作避免TensorFlow变量问题
        
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
        # else:
        #     print("生成随机初始firing rates")
        #     initial_firing_rates = self.generate_background_firing_rates(batch_size)
        
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
            # 为每次迭代设置随机种子（仅在use_fixed_seed=True时）
            if self.use_fixed_seed:
                iteration_seed = 42 + iteration
            else:
                # 使用当前时间戳作为种子
                iteration_seed = int(time.time() * 1000) % 1000000 + iteration
            
            # 直接调用compute_loss_numpy计算损失
            loss = self.compute_loss_numpy(current_firing_rates, fixed_exc_indices, target_spike_prob, iteration_seed)
            
            # 计算梯度（使用数值梯度）
            gradient = self.compute_numerical_gradient(current_firing_rates, fixed_exc_indices, target_spike_prob, loss, iteration_seed)
            
            # 梯度范数裁剪（全局）以稳定优化
            max_grad_norm = 5.0
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > max_grad_norm:
                gradient = gradient * (max_grad_norm / (grad_norm + 1e-12))
                print(f"  梯度裁剪: {grad_norm:.6f} -> {max_grad_norm:.6f}")
            
            # 应用梯度更新
            current_firing_rates = current_firing_rates - learning_rate * gradient
            
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
    
    def reset_random_seeds(self, base_seed=None):
        """
        重置随机种子，用于不同epoch之间的随机化
        
        Args:
            base_seed: 基础种子，如果为None则使用当前时间戳
        """
        if base_seed is None:
            base_seed = int(time.time()) % 10000
        
        # 重置numpy和tensorflow的随机种子
        np.random.seed(base_seed)
        tf.random.set_seed(base_seed)
        
        print(f"随机种子已重置为: {base_seed}")
        return base_seed
    
    def save_optimization_results(self, optimized_firing_rates, loss_history, 
                                firing_rates_history, fixed_exc_indices, save_dir, start_time_ms):
        """
        保存优化结果
        """
        os.makedirs(save_dir, exist_ok=True)

        # 保存数据（不在文件名中加入时间戳；时间戳用于上层子目录名）
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
    
    def evaluate_optimized_activity(self, optimized_firing_rates, fixed_exc_indices, num_evaluations=10):
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
            # 为每次评估设置随机种子（仅在use_fixed_seed=True时）
            if self.use_fixed_seed:
                eval_seed = 1000 + eval_idx  # 使用不同的种子范围
            else:
                # 使用当前时间戳作为种子
                eval_seed = int(time.time() * 1000) % 1000000 + eval_idx
            
            # 使用可复用函数生成spikes和预测
            spike_predictions, spike_trains = self.process_firing_rates_to_predictions(
                optimized_firing_rates, fixed_exc_indices, eval_seed
            )
            
            # 对每个batch进行预测
            batch_spike_probs = []
            for batch_idx in range(spike_trains.shape[0]):
                # 取half window size之后的10个时间步的预测概率
                target_start_time, target_time_steps = self.input_window_size // 2, 10  # 关注mono synaptic spike之后的10个时间步
                final_predictions = spike_predictions[batch_idx, target_start_time:target_start_time+target_time_steps, 0]
                batch_spike_probs.extend(final_predictions.tolist())
            
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

    def visualize_firing_rates(self, firing_rates, save_path=None, title="Firing Rates Visualization", 
                              figsize=(15, 10), time_step_ms=1, max_segments_to_show=200):
        """
        可视化firing rate array，类似raster plot但显示连续曲线
        
        Args:
            firing_rates: (batch_size, time_duration_ms, num_segments_total) 或 (time_duration_ms, num_segments_total)
            save_path: 保存图片的路径，如果为None则显示图片
            title: 图片标题
            figsize: 图片大小
            time_step_ms: 时间步长（毫秒）
            max_segments_to_show: 最大显示的segments数量（避免图片过于拥挤）
        """
        # 如果是3D数组，取第一个batch
        if len(firing_rates.shape) == 3:
            firing_rates_2d = firing_rates[0]  # (time_duration_ms, num_segments_total)
        else:
            firing_rates_2d = firing_rates  # (time_duration_ms, num_segments_total)
        
        time_duration_ms, num_segments_total = firing_rates_2d.shape
        
        # 如果segments太多，进行采样
        if num_segments_total > max_segments_to_show:
            # 保持兴奋性和抑制性的比例
            exc_ratio = self.num_segments_exc / num_segments_total
            exc_to_show = int(max_segments_to_show * exc_ratio)
            inh_to_show = max_segments_to_show - exc_to_show
            
            # 采样兴奋性segments
            exc_indices = np.linspace(0, self.num_segments_exc-1, exc_to_show, dtype=int)
            # 采样抑制性segments
            inh_indices = np.linspace(self.num_segments_exc, num_segments_total-1, inh_to_show, dtype=int)
            
            selected_indices = np.concatenate([exc_indices, inh_indices])
            firing_rates_2d = firing_rates_2d[:, selected_indices]
            
            # 更新segment数量信息
            num_segments_to_show = len(selected_indices)
            num_exc_to_show = len(exc_indices)
            num_inh_to_show = len(inh_indices)
        else:
            selected_indices = np.arange(num_segments_total)
            num_segments_to_show = num_segments_total
            num_exc_to_show = self.num_segments_exc
            num_inh_to_show = self.num_segments_inh
        
        # 创建时间轴
        time_axis = np.arange(time_duration_ms) * time_step_ms
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 为每个segment绘制firing rate曲线
        y_positions = np.arange(num_segments_to_show)
        
        # 设置颜色
        colors = []
        for i, seg_idx in enumerate(selected_indices):
            if seg_idx < self.num_segments_exc:
                colors.append('blue')  # 兴奋性
            else:
                colors.append('red')   # 抑制性
        
        # 绘制每个segment的firing rate
        for i in range(num_segments_to_show):
            # 获取该segment的firing rate
            segment_firing_rate = firing_rates_2d[:, i]
            
            # 将firing rate缩放到合适的显示范围（每个segment占用0.8个单位高度）
            scaled_firing_rate = segment_firing_rate * 0.4  # 缩放因子
            y_base = y_positions[i]
            y_values = y_base + scaled_firing_rate
            
            # 绘制曲线
            ax.plot(time_axis, y_values, color=colors[i], linewidth=0.5, alpha=0.7)
            
            # 绘制基线
            ax.axhline(y=y_base, color='gray', linewidth=0.2, alpha=0.3)
        
        # 设置坐标轴
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Segments', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 设置y轴刻度和标签
        y_tick_positions = []
        y_tick_labels = []
        
        # 添加一些代表性的刻度
        if num_exc_to_show > 0:
            y_tick_positions.extend([0, num_exc_to_show//2, num_exc_to_show-1])
            y_tick_labels.extend([f'Exc 0', f'Exc {num_exc_to_show//2}', f'Exc {num_exc_to_show-1}'])
        
        if num_inh_to_show > 0:
            inh_start = num_exc_to_show
            y_tick_positions.extend([inh_start, inh_start + num_inh_to_show//2, inh_start + num_inh_to_show-1])
            y_tick_labels.extend([f'Inh 0', f'Inh {num_inh_to_show//2}', f'Inh {num_inh_to_show-1}'])
        
        ax.set_yticks(y_tick_positions)
        ax.set_yticklabels(y_tick_labels, fontsize=10)
        
        # 设置网格
        ax.grid(True, alpha=0.3)
        
        # 添加分隔线（兴奋性和抑制性之间）
        if num_exc_to_show > 0 and num_inh_to_show > 0:
            ax.axhline(y=num_exc_to_show-0.5, color='black', linewidth=2, linestyle='--', alpha=0.5)
        
        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label=f'Excitatory ({num_exc_to_show} segments)'),
            Line2D([0], [0], color='red', lw=2, label=f'Inhibitory ({num_inh_to_show} segments)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # 添加统计信息文本
        stats_text = f"Time duration: {time_duration_ms} ms\n"
        stats_text += f"Total segments: {num_segments_total} ({self.num_segments_exc} exc + {self.num_segments_inh} inh)\n"
        if num_segments_total > max_segments_to_show:
            stats_text += f"Showing: {num_segments_to_show} segments (sampled)"
        else:
            stats_text += f"Showing: all {num_segments_to_show} segments"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Firing rates visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # 返回一些统计信息
        stats = {
            'mean_firing_rate_exc': np.mean(firing_rates_2d[:, :num_exc_to_show]) if num_exc_to_show > 0 else 0,
            'mean_firing_rate_inh': np.mean(firing_rates_2d[:, num_exc_to_show:]) if num_inh_to_show > 0 else 0,
            'max_firing_rate': np.max(firing_rates_2d),
            'min_firing_rate': np.min(firing_rates_2d),
            'segments_shown': num_segments_to_show,
            'total_segments': num_segments_total
        }
        
        return stats

    def visualize_firing_rates_heatmap(self, firing_rates, save_path=None, title="Firing Rates Heatmap",
                                      figsize=(15, 8), max_segments_to_show=300):
        """
        使用热图方式可视化firing rate array
        
        Args:
            firing_rates: (batch_size, time_duration_ms, num_segments_total) 或 (time_duration_ms, num_segments_total)
            save_path: 保存图片的路径
            title: 图片标题
            figsize: 图片大小
            max_segments_to_show: 最大显示的segments数量
        """
        # 如果是3D数组，取第一个batch
        if len(firing_rates.shape) == 3:
            firing_rates_2d = firing_rates[0]  # (time_duration_ms, num_segments_total)
        else:
            firing_rates_2d = firing_rates  # (time_duration_ms, num_segments_total)
        
        time_duration_ms, num_segments_total = firing_rates_2d.shape
        
        # 转置为 (num_segments_total, time_duration_ms) 以便显示
        firing_rates_2d = firing_rates_2d.T
        
        # 如果segments太多，进行采样
        if num_segments_total > max_segments_to_show:
            exc_ratio = self.num_segments_exc / num_segments_total
            exc_to_show = int(max_segments_to_show * exc_ratio)
            inh_to_show = max_segments_to_show - exc_to_show
            
            exc_indices = np.linspace(0, self.num_segments_exc-1, exc_to_show, dtype=int)
            inh_indices = np.linspace(self.num_segments_exc, num_segments_total-1, inh_to_show, dtype=int)
            
            selected_indices = np.concatenate([exc_indices, inh_indices])
            firing_rates_2d = firing_rates_2d[selected_indices, :]
            
            num_segments_to_show = len(selected_indices)
            num_exc_to_show = len(exc_indices)
        else:
            num_segments_to_show = num_segments_total
            num_exc_to_show = self.num_segments_exc
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 创建热图
        im = ax.imshow(firing_rates_2d, aspect='auto', cmap='viridis', origin='lower')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Firing Rate', rotation=270, labelpad=15)
        
        # 设置坐标轴标签
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Segments', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 添加分隔线（兴奋性和抑制性之间）
        if num_exc_to_show < num_segments_to_show:
            ax.axhline(y=num_exc_to_show-0.5, color='red', linewidth=2, linestyle='--', alpha=0.8)
        
        # 设置y轴刻度
        y_ticks = [0, num_exc_to_show-1, num_exc_to_show, num_segments_to_show-1]
        y_labels = ['Exc 0', f'Exc {num_exc_to_show-1}', 'Inh 0', f'Inh {num_segments_to_show-num_exc_to_show-1}']
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        
        # 添加统计信息
        stats_text = f"Time: {time_duration_ms} ms\nSegments: {num_segments_to_show}/{num_segments_total}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Firing rates heatmap saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()

def find_best_model(models_dir):
    """
    找到最佳模型（基于验证损失）
    
    Args:
        models_dir: 模型目录
        
    Returns:
        best_model_path: 最佳模型的.h5文件路径
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
                best_model_path = pickle_path.replace('.pickle', '.h5')
        
        except Exception as e:
            print(f"读取{pickle_path}时出错: {e}")
    
    if best_model_path is None:
        raise ValueError("未找到有效的模型文件")
    
    print(f"找到最佳模型: {best_model_path}")
    
    return best_model_path, best_params_path

def main():
    """
    主函数：运行activity optimization
    """
    print("=== Activity Optimization ===")
    
    # 设置文件路径
    models_dir = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut_SJC_funcgroup2_var2/models/NMDA_fullStrategy_2/depth_7_filters_256_window_400/'
    init_firing_rates_path = './init_firing_rate_array.npy'  # 初始firing rates文件
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = f'./results/activity_optimization_results/{timestamp}'
    
    # 检查初始firing rates文件是否存在
    if not os.path.exists(init_firing_rates_path):
        print(f"警告：初始firing rates文件不存在: {init_firing_rates_path}")
        print("将使用随机生成的初始firing rates")
        init_firing_rates_path = None
    
    # 找到最佳模型
    print("寻找最佳模型...")
    try:
        model_path, params_path = find_best_model(models_dir)
        print(f"选择的模型: {os.path.basename(model_path)}")
    except Exception as e:
        print(f"错误: {e}")
        return
    
    # 创建优化器
    optimizer = ActivityOptimizer(
        model_path=model_path, 
        model_params_path=params_path, 
        init_firing_rates_path=init_firing_rates_path,
        time_duration_ms=300,
        use_fixed_seed=True  # 设置为False以关闭固定随机种子，使用完全随机
    )
    
    # 执行优化（第一个epoch）
    optimized_firing_rates, loss_history, fixed_exc_indices = optimizer.optimize_activity(
        num_iterations=50,
        learning_rate=0.001,
        batch_size=1,
        target_spike_prob=0.8,
        save_dir=save_dir,
        start_time_ms= 0 #100  # 忽略前100ms
    )
    
    # 评估结果
    evaluation_results = optimizer.evaluate_optimized_activity(
        optimized_firing_rates, fixed_exc_indices, num_evaluations=20
    )
    
    print("\n=== 优化完成 ===")
    print(f"优化后的firing rates形状: {optimized_firing_rates.shape}")
    print(f"最终损失: {loss_history[-1]:.6f}")
    print(f"固定添加spikes的excitatory segments: {fixed_exc_indices}")
    
    # 可选：可视化优化后的firing rates
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
        
    except ImportError:
        print("可视化模块不可用，跳过可视化步骤")
    except Exception as e:
        print(f"生成可视化时出错: {e}")

if __name__ == "__main__":
    main() 