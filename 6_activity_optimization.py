import numpy as np
import tensorflow as tf
import pickle
import os
import time
from datetime import datetime
from utils.firing_rates_processor import FiringRatesProcessor
from utils.find_best_model import find_best_model
from utils.visualization_utils import (
    visualize_firing_rates_trace, visualize_firing_rates_heatmap,
    plot_loss_history, plot_firing_rates_evolution, plot_optimization_summary, create_optimization_report
)

class ActivityOptimizer:
    """
    基于已训练的TCN模型进行activity optimization的类
    专注于优化算法和梯度计算，数据处理委托给FiringRatesProcessor
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
        # 创建数据处理器
        self.processor = FiringRatesProcessor(model_path, model_params_path, time_duration_ms)
        
        # 从处理器获取模型信息
        model_info = self.processor.get_model_info()
        self.input_window_size = model_info['input_window_size']
        self.num_segments_exc = model_info['num_segments_exc']
        self.num_segments_inh = model_info['num_segments_inh']
        self.num_segments_total = model_info['num_segments_total']
        self.time_duration_ms = model_info['time_duration_ms']
        
        # 加载初始firing rates（如果提供）
        self.init_firing_rates = None
        if init_firing_rates_path and os.path.exists(init_firing_rates_path):
            self.init_firing_rates = self.processor.load_init_firing_rates(init_firing_rates_path)
        
        print(f"ActivityOptimizer初始化成功:")
        print(f"  输入窗口大小: {self.input_window_size}ms")
        print(f"  兴奋性segments: {self.num_segments_exc}")
        print(f"  抑制性segments: {self.num_segments_inh}")
        print(f"  总segments: {self.num_segments_total}")
        if self.init_firing_rates is not None:
            print(f"  初始firing rates已加载: {self.init_firing_rates.shape}")
    
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
        spike_predictions, spike_trains = self.processor.process_firing_rates_to_predictions(firing_rates, fixed_exc_indices)
        
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
        for layer in self.processor.model.layers:
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
            spike_predictions = self.processor.model(model_input, training=False)[0]

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
        # print(f"Time to calculate gradient: {end_time - start_time:.4f} seconds")
        
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
            initial_firing_rates = self.processor.prepare_firing_rates_for_optimization(
                self.init_firing_rates, batch_size, start_time_ms
            )
        else:
            print("生成随机初始firing rates")
            initial_firing_rates = self.processor.generate_background_firing_rates(batch_size)
        
        # 选择固定的excitatory indices
        mono_syn_rnd = np.random.default_rng(42)
        fixed_exc_indices = mono_syn_rnd.choice(self.num_segments_exc, size=3, replace=False)
        print(f"固定添加spikes的excitatory segments: {fixed_exc_indices}")
        
        # 使用numpy数组进行优化
        current_firing_rates = initial_firing_rates.copy()
        
        # 记录历史
        loss_history = []
        firing_rates_history = []
        
        # 分批处理：将2000次迭代分成4个批次，每批500次
        batch_size_iterations = 500
        num_batches = (num_iterations + batch_size_iterations - 1) // batch_size_iterations
        
        print(f"将{num_iterations}次迭代分成{num_batches}个批次，每批{batch_size_iterations}次")
        
        for batch_idx in range(num_batches):
            print(f"\n=== 开始第{batch_idx + 1}/{num_batches}批次 ===")
            
            # 计算当前批次的迭代范围
            start_iter = batch_idx * batch_size_iterations
            end_iter = min((batch_idx + 1) * batch_size_iterations, num_iterations)
            current_batch_size = end_iter - start_iter
            
            print(f"批次 {batch_idx + 1}: 迭代 {start_iter} 到 {end_iter-1} (共{current_batch_size}次)")
            
            # 当前批次的优化循环
            for iteration in range(current_batch_size):
                global_iteration = start_iter + iteration
                
                # 直接调用compute_loss_numpy计算损失
                loss = self.compute_loss_numpy(current_firing_rates, fixed_exc_indices, target_spike_prob)
                
                # 计算梯度（使用数值梯度）
                gradient = self.compute_numerical_gradient(current_firing_rates, fixed_exc_indices, target_spike_prob, loss)
                
                # 调试：打印梯度信息
                if iteration % 50 == 0:  # 每50次迭代打印一次
                    grad_norm = np.linalg.norm(gradient)
                    print(f"  全局迭代 {global_iteration:3d}: Loss = {loss:.6f}, 梯度范数: {grad_norm:.8f}")
                
                # 应用梯度更新
                current_firing_rates = current_firing_rates - learning_rate * gradient
                
                # 限制firing rates在合理范围内 [0, 0.1]
                current_firing_rates = np.clip(current_firing_rates, 0.0, 0.1)
                
                # 记录历史
                loss_history.append(float(loss))
                if global_iteration % 100 == 0:
                    firing_rates_history.append(current_firing_rates.copy())
            
            # 使用智能内存清理
            self._smart_memory_cleanup(batch_idx, num_batches)
            
            # 每批次完成后保存中间结果
            if save_dir:
                intermediate_save_dir = os.path.join(save_dir, f'batch_{batch_idx + 1}')
                os.makedirs(intermediate_save_dir, exist_ok=True)
                
                # 保存当前批次的结果
                intermediate_results = {
                    'batch_idx': batch_idx + 1,
                    'current_firing_rates': current_firing_rates,
                    'loss_history_so_far': loss_history,
                    'fixed_exc_indices': fixed_exc_indices,
                    'global_iteration': end_iter - 1
                }
                
                intermediate_file = os.path.join(intermediate_save_dir, f'batch_{batch_idx + 1}_results.pickle')
                with open(intermediate_file, 'wb') as f:
                    pickle.dump(intermediate_results, f)
                
                print(f"  中间结果已保存到: {intermediate_file}")
        
        print("-" * 50)
        print(f"所有批次优化完成! 最终损失: {loss_history[-1]:.6f}")
        
        # 获取最终结果
        optimized_firing_rates = current_firing_rates
        
        # 保存最终结果
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
            'model_path': self.processor.model_path,
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
        
        # 使用可视化函数生成图表
        print("生成优化过程可视化...")
        
        # 1. 损失历史
        plot_loss_history(loss_history, save_path=os.path.join(save_dir, "activity_optimization_loss_history.png"))
        
        # 2. Firing rates演化（如果有历史数据）
        if firing_rates_history:
            plot_firing_rates_evolution(
                firing_rates_history, self.num_segments_exc, self.num_segments_inh,
                self.time_duration_ms, self.input_window_size, 
                save_path=os.path.join(save_dir, "activity_optimization_firing_rates_evolution.png")
            )
        
        # 3. 优化总结
        plot_optimization_summary(
            loss_history, firing_rates_history, self.num_segments_exc, self.num_segments_inh,
            self.time_duration_ms, self.input_window_size, 
            save_path=os.path.join(save_dir, "activity_optimization_summary.png")
        )
        
        print("优化结果可视化已生成")
    
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
        
        # 由于分批处理中可能清理了TensorFlow会话，需要重新加载模型
        print("重新加载模型以确保评估正常进行...")
        try:
            # 重新加载模型
            from keras.models import load_model
            self.processor.model = load_model(self.processor.model_path)
            print("✓ 模型重新加载成功")
        except Exception as e:
            print(f"✗ 模型重新加载失败: {e}")
            print("尝试使用现有模型...")
        
        spike_probabilities = []
        
        for eval_idx in range(num_evaluations):
            try:
                # 使用可复用函数生成spikes和预测
                spike_predictions, spike_trains = self.processor.process_firing_rates_to_predictions(
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
                
                if eval_idx % 5 == 0:
                    print(f"  评估进度: {eval_idx + 1}/{num_evaluations}")
                    
            except Exception as e:
                print(f"  第{eval_idx + 1}次评估失败: {e}")
                # 如果评估失败，添加默认值
                default_probs = [0.5] * 10  # 默认概率
                spike_probabilities.extend(default_probs)
        
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

    def _smart_memory_cleanup(self, batch_idx, num_batches):
        """
        智能内存清理：只在必要时进行清理
        
        Args:
            batch_idx: 当前批次索引
            num_batches: 总批次数
        """
        print(f"批次 {batch_idx + 1} 完成，开始智能内存清理...")
    
        # 只在特定条件下进行清理
        should_cleanup = False
        
        # 条件1: 每3个批次清理一次
        if (batch_idx + 1) % 3 == 0:
            should_cleanup = True
            print("  - 定期清理触发")
        
        # 条件2: 最后一个批次，为评估做准备
        if batch_idx == num_batches - 1:
            should_cleanup = True
            print("  - 最终清理触发")
        
        if should_cleanup:
            # 方法1: 温和的TensorFlow内存清理
            try:
                tf.keras.backend.clear_session()
                print("  ✓ TensorFlow会话已清理")
            except Exception as e:
                print(f"  ✗ TensorFlow会话清理失败: {e}")
            
            # 方法2: 强制垃圾回收
            try:
                import gc
                gc.collect()
                print("  ✓ 垃圾回收已完成")
            except Exception as e:
                print(f"  ✗ 垃圾回收失败: {e}")
            
            # 方法3: GPU内存清理（如果使用GPU）
            try:
                if tf.config.list_physical_devices('GPU'):
                    tf.config.experimental.reset_memory_growth()
                    print("  ✓ GPU内存分配器已重置")
            except Exception as e:
                print(f"  ✗ GPU内存重置失败: {e}")
        else:
            print("  - 跳过内存清理（内存使用正常）")
        
        print(f"批次 {batch_idx + 1} 内存管理完成")
        
        # 在最后一个批次完成后，重新加载模型以确保后续操作正常
        if batch_idx == num_batches - 1:
            print("最后一个批次完成，重新加载模型...")
            try:
                from keras.models import load_model
                self.processor.model = load_model(self.processor.model_path)
                print("  ✓ 模型重新加载成功，准备进行评估")
            except Exception as e:
                print(f"  ✗ 模型重新加载失败: {e}")
                print("  评估阶段可能需要手动重新加载模型")


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
        time_duration_ms=300
    )
    
    # 执行优化
    optimized_firing_rates, loss_history, fixed_exc_indices = optimizer.optimize_activity(
        num_iterations=500,
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
    
    # ## Get optimized firing rates from pickle file
    # save_dir = './results/activity_optimization_results/20250827_163803'
    # with open(os.path.join(save_dir, 'activity_optimization.pickle'), 'rb') as f:
    #     data = pickle.load(f)
    # optimized_firing_rates = data['optimized_firing_rates']
    # fixed_exc_indices = data['fixed_exc_indices']
    
    # 可选：可视化优化后的firing rates
    try:
        print("\n生成优化后的firing rates可视化...")
        
        # 使用可视化函数生成完整的优化报告
        create_optimization_report(
            loss_history=loss_history,
            firing_rates_history=[],  # 这里没有历史数据，因为是在主函数中
            optimized_firing_rates=optimized_firing_rates,
            fixed_exc_indices=fixed_exc_indices,
            num_segments_exc=639,
            num_segments_inh=640,
            time_duration_ms=300,
            input_window_size=300,
            save_dir=save_dir,
            report_name="activity_optimization"
        )
        
        print("完整的优化报告已生成")
        
    except ImportError:
        print("可视化模块不可用，跳过可视化步骤")
    except Exception as e:
        print(f"生成可视化时出错: {e}")
        print("尝试使用基础可视化函数...")
        
        # 备用方案：使用基础可视化函数
        try:
            optimized_sample = optimized_firing_rates[0]
            
            # 指定要可视化的segments
            specified_segments = None
            if fixed_exc_indices is not None and len(fixed_exc_indices) > 0:
                extended_indices = []
                for idx in fixed_exc_indices:
                    start_idx = max(0, idx - 2)
                    end_idx = min(optimized_sample.shape[0], idx + 3)
                    extended_indices.extend(range(start_idx, end_idx))
                
                extended_indices = sorted(list(set(extended_indices)))
                specified_segments = extended_indices
                print(f"指定可视化segments: {specified_segments}")
            
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
            
            print("基础可视化已完成")
            
        except Exception as e2:
            print(f"基础可视化也失败: {e2}")

if __name__ == "__main__":
    main() 