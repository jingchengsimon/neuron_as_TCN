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
from visualize_firing_rates import visualize_firing_rates_raster, visualize_firing_rates_heatmap

class ActivityOptimizer:
    """
    基于已训练的TCN模型进行activity optimization的类
    实现图中描述的优化流程：对输入firing rate进行BCE梯度下降优化
    """
    
    def __init__(self, model_path, model_params_path, init_firing_rates_path=None, time_duration_ms=300):
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
        
        # 加载模型和参数
        self.model = load_model(model_path)
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
    
    def generate_background_firing_rates(self, batch_size=1):
        """
        生成背景firing rates (泊松过程)
        
        Args:
            batch_size: 批次大小
            
        Returns:
            firing_rates: (batch_size, time_duration_ms, num_segments_total)
        """
        # 为兴奋性和抑制性segments设置不同的背景firing rate
        exc_rate = 0.01  # 兴奋性背景firing rate (1%)
        inh_rate = 0.02  # 抑制性背景firing rate (2%)
        
        # 生成firing rates
        firing_rates = np.zeros((batch_size, self.time_duration_ms, self.num_segments_total))
        
        # 兴奋性segments
        firing_rates[:, :, :self.num_segments_exc] = np.random.uniform(
            0.005, 0.02, (batch_size, self.time_duration_ms, self.num_segments_exc)
        )
        
        # 抑制性segments
        firing_rates[:, :, self.num_segments_exc:] = np.random.uniform(
            0.01, 0.03, (batch_size, self.time_duration_ms, self.num_segments_inh)
        )
        
        return firing_rates.astype(np.float32)
    
    def generate_spikes_with_modification(self, firing_rates, fixed_exc_indices=None):
        """
        生成spikes with modification - 可复用函数
        
        Args:
            firing_rates: (batch_size, num_segments_total, time_duration_ms)
            fixed_exc_indices: 固定添加spikes的excitatory indices
            
        Returns:
            spike_trains: (batch_size, num_segments_total, time_duration_ms) binary
            fixed_exc_indices: 使用的fixed indices
        """
        batch_size = firing_rates.shape[0]
        
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
    
    def process_firing_rates_to_predictions(self, firing_rates, fixed_exc_indices):
        """
        可复用函数：将firing rates转换为模型预测
        
        Args:
            firing_rates: (batch_size, num_segments_total, time_duration_ms)
            fixed_exc_indices: 固定的excitatory indices
            
        Returns:
            spike_predictions: 模型预测的spike概率
            spike_trains: 生成的spike trains
        """
        # 使用可复用函数生成spikes
        spike_trains, _ = self.generate_spikes_with_modification(firing_rates, fixed_exc_indices)
        
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
        
        # 模型预测
        predictions = self.model.predict(model_input, verbose=0)
        spike_predictions = predictions[0]  # 第一个输出是spike预测
        
        return spike_predictions, spike_trains
    
    def compute_loss_numpy(self, firing_rates, fixed_exc_indices, target_spike_prob):
        """
        使用numpy计算损失函数
        
        Args:
            firing_rates: firing rates
            fixed_exc_indices: 固定的excitatory indices
            target_spike_prob: 目标spike概率
            
        Returns:
            loss: 损失值
        """
        # 使用可复用函数生成spikes
        spike_predictions, spike_trains = self.process_firing_rates_to_predictions(firing_rates, fixed_exc_indices)
        
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

    def compute_numerical_gradient(self, firing_rates, fixed_exc_indices, target_spike_prob, original_loss=None, epsilon=1e-6):
        """
        使用自动微分计算 BCE loss 关于 firing_rates 的梯度
        firing_rates: numpy array, shape (batch_size, num_segments_total, time_duration)
        """
        start_time = time.time()
        # 固定模型参数（不更新，但梯度可回传到输入）
        for layer in self.model.layers:
            layer.trainable = False

        firing_rates_tf = tf.convert_to_tensor(firing_rates, dtype=tf.float32)

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

            # 前补零（如果时间不足）
            pad_len = tf.maximum(self.input_window_size - time_steps, 0)
            model_input = tf.pad(model_input, [[0, 0], [pad_len, 0], [0, 0]])

            # 截取最后 input_window_size 个时间步（多了就裁掉）
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

            # 正则化项
            regularization_loss = 0.001 * tf.reduce_mean(firing_rates_tf)

            loss = prediction_loss + regularization_loss

        # ===== 6. 求梯度 =====
        grad = tape.gradient(loss, firing_rates_tf)
        end_time = time.time()
        print(f"Time to calculate gradient: {end_time - start_time:.4f} seconds")
        
        # 调试：检查梯度是否为零
        if grad is not None:
            try:
                # 先转换为numpy数组，避免Tensor对象的问题
                grad_np = self._tensor_to_numpy(grad)
                grad_norm = np.linalg.norm(grad_np)
                # grad_min = np.min(grad_np)
                # grad_max = np.max(grad_np)
                
                # print(f"  梯度范数: {grad_norm:.8f}")
                if grad_norm < 1e-8:
                    print("  警告：梯度接近零！")
                # else:
                    # print(f"  梯度范围: [{grad_min:.8f}, {grad_max:.8f}]")
            except Exception as e:
                print(f"  计算梯度统计信息时出错: {e}")
        else:
            print("  警告：梯度为None！")
        
        # 兼容不同TensorFlow版本的张量转换方法
        try:
            # 方法1: 直接调用numpy()方法（TensorFlow 2.x）
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
        else:
            print("生成随机初始firing rates")
            initial_firing_rates = self.generate_background_firing_rates(batch_size)
        
        # 选择固定的excitatory indices
        mono_syn_rnd = np.random.default_rng(42)
        fixed_exc_indices = mono_syn_rnd.choice(self.num_segments_exc, size=3, replace=False)
        print(f"固定添加spikes的excitatory segments: {fixed_exc_indices}")
        
        # 使用numpy数组进行优化
        current_firing_rates = initial_firing_rates.copy()
        
        # 记录历史
        loss_history = []
        firing_rates_history = []
        
        # 优化循环
        for iteration in range(num_iterations):
            # 直接调用compute_loss_numpy计算损失
            loss = self.compute_loss_numpy(current_firing_rates, fixed_exc_indices, target_spike_prob)
            
            # 计算梯度（使用数值梯度）
            gradient = self.compute_numerical_gradient(current_firing_rates, fixed_exc_indices, target_spike_prob, loss)
            
            # 调试：打印梯度信息
            if iteration % 10 == 0:  # 每5次迭代打印一次
                grad_norm = np.linalg.norm(gradient)
                print(f"  梯度范数: {grad_norm:.8f}")
                # print(f"  梯度范围: [{np.min(gradient):.8f}, {np.max(gradient):.8f}]")
                # print(f"  更新步长: {learning_rate * grad_norm:.8f}")
            
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
                print(f"迭代 {iteration:3d}: Loss = {loss:.6f}")
        
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
    
    def save_optimization_results(self, optimized_firing_rates, loss_history, 
                                firing_rates_history, fixed_exc_indices, save_dir, start_time_ms):
        """
        保存优化结果
        """
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
        
        result_file = os.path.join(save_dir, f'activity_optimization.pickle')
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
        plt.savefig(os.path.join(save_dir, f'loss_history.png'), dpi=300, bbox_inches='tight')
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
            plt.savefig(os.path.join(save_dir, f'firing_rates_evolution.png'), 
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
            # 使用可复用函数生成spikes和预测
            spike_predictions, spike_trains = self.process_firing_rates_to_predictions(
                optimized_firing_rates, fixed_exc_indices
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

    def _tensor_to_numpy(self, tensor):
        """
        将TensorFlow张量转换为numpy数组的辅助方法
        
        Args:
            tensor: TensorFlow张量
            
        Returns:
            numpy数组
        """
        # 兼容不同TensorFlow版本的张量转换方法
        try:
            # 方法1: 直接调用numpy()方法（TensorFlow 2.x）
            if hasattr(tensor, 'numpy'):
                return tensor.numpy()
            # 方法2: 使用tf.keras.backend.eval
            elif hasattr(tf.keras.backend, 'eval'):
                return tf.keras.backend.eval(tensor)
            # 方法3: 使用tf.make_ndarray
            elif hasattr(tf, 'make_ndarray'):
                return tf.make_ndarray(tensor)
            # 方法4: 最后备选方案
            else:
                return np.array(tensor)
        except Exception as e:
            print(f"转换张量到numpy时出错: {e}")
            print("尝试使用tf.keras.backend.eval作为最后备选方案...")
            try:
                return tf.keras.backend.eval(tensor)
            except Exception as e2:
                print(f"所有转换方法都失败: {e2}")
                print("返回零数组")
                return np.zeros((1,))

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
    save_dir = f'./Results/activity_optimization_results/{timestamp}'
    
    # 检查初始firing rates文件是否存在
    if not os.path.exists(init_firing_rates_path):
        print(f"警告：初始firing rates文件不存在: {init_firing_rates_path}")
        print("将使用随机生成的初始firing rates")
        init_firing_rates_path = None
    
    # # 找到最佳模型
    # print("寻找最佳模型...")
    # try:
    #     model_path, params_path = find_best_model(models_dir)
    #     print(f"选择的模型: {os.path.basename(model_path)}")
    # except Exception as e:
    #     print(f"错误: {e}")
    #     return
    
    # # 创建优化器
    # optimizer = ActivityOptimizer(
    #     model_path=model_path, 
    #     model_params_path=params_path, 
    #     init_firing_rates_path=init_firing_rates_path,
    #     time_duration_ms=300
    # )
    
    # # 执行优化
    # optimized_firing_rates, loss_history, fixed_exc_indices = optimizer.optimize_activity(
    #     num_iterations=5000,
    #     learning_rate=0.001,
    #     batch_size=1,
    #     target_spike_prob=0.8,
    #     save_dir=save_dir,
    #     start_time_ms= 0 #100  # 忽略前100ms
    # )
    
    # # 评估结果
    # evaluation_results = optimizer.evaluate_optimized_activity(
    #     optimized_firing_rates, fixed_exc_indices, num_evaluations=20
    # )
    
    # print("\n=== 优化完成 ===")
    # print(f"优化后的firing rates形状: {optimized_firing_rates.shape}")
    # print(f"最终损失: {loss_history[-1]:.6f}")
    # print(f"固定添加spikes的excitatory segments: {fixed_exc_indices}")
    
    ## Get optimized firing rates from pickle file
    save_dir = './Results/activity_optimization_results/20250827_163803'
    with open(os.path.join(save_dir, 'activity_optimization.pickle'), 'rb') as f:
        data = pickle.load(f)
    optimized_firing_rates = data['optimized_firing_rates']
    fixed_exc_indices = data['fixed_exc_indices']
    
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
                start_idx = max(0, idx - 3)
                end_idx = min(optimized_sample.shape[0], idx + 4)
                extended_indices.extend(range(start_idx, end_idx))
            
            # 去重并排序
            extended_indices = sorted(list(set(extended_indices)))
            specified_segments = extended_indices
            
            print(f"指定可视化segments: {specified_segments}")
            print(f"包含fixed_exc_indices: {fixed_exc_indices}")
        
        # Raster plot
        visualize_firing_rates_raster(
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