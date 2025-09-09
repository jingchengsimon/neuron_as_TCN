import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


import pickle
import os
from datetime import datetime
from utils.firing_rates_processor import FiringRatesProcessor
from utils.find_best_model import find_best_model
from utils.visualization_utils import (
    visualize_firing_rates_trace, visualize_firing_rates_heatmap,
    plot_loss_history, plot_firing_rates_evolution, plot_optimization_summary, create_optimization_report
)

class ActivityOptimizer:
    """
    Activity optimization class based on trained TCN model
    Focuses on optimization algorithms and gradient computation, delegates data processing to FiringRatesProcessor
    """
    
    def __init__(self, model_path, model_params_path, init_firing_rates_path=None, time_duration_ms=300):
        """
        Initialize optimizer
        
        Args:
            model_path: Path to trained model .h5 file
            model_params_path: Path to corresponding parameters .pickle file
            init_firing_rates_path: Path to initial firing rates .npy file
            time_duration_ms: Time duration, default 300ms
        """
        # Create data processor
        self.processor = FiringRatesProcessor(model_path, model_params_path, time_duration_ms)
        
        # Get model info from processor
        model_info = self.processor.get_model_info()
        self.input_window_size = model_info['input_window_size']
        self.num_segments_exc = model_info['num_segments_exc']
        self.num_segments_inh = model_info['num_segments_inh']
        self.num_segments_total = model_info['num_segments_total']
        self.time_duration_ms = model_info['time_duration_ms']
        
        # Load initial firing rates (if provided)
        self.init_firing_rates = None
        if init_firing_rates_path and os.path.exists(init_firing_rates_path):
            self.init_firing_rates = self.processor.load_init_firing_rates(init_firing_rates_path)
        
        print(f"ActivityOptimizer initialized successfully:")
        print(f"  Input window size: {self.input_window_size}ms")
        print(f"  Excitatory segments: {self.num_segments_exc}")
        print(f"  Inhibitory segments: {self.num_segments_inh}")
        print(f"  Total segments: {self.num_segments_total}")
        if self.init_firing_rates is not None:
            print(f"  Initial firing rates loaded: {self.init_firing_rates.shape}")
    
    def create_optimization_model(self, fixed_exc_indices, target_spike_prob):
        """
        创建一个基于NumPy/PyTorch评估的损失函数闭包，返回可调用对象
        """
        input_window = self.input_window_size
        model = self.processor.model  # PyTorch 模型
        model.eval()

        def evaluate_loss(firing_rates_np):
            # firing_rates_np: (batch, num_segments_total, time_duration)
            # 1) 生成spikes（与processor一致）
            spike_preds, spike_trains = self.processor.process_firing_rates_to_predictions(
                firing_rates_np, fixed_exc_indices
            )
            # 2) 取窗口中心后的10个时间步做BCE
            target_start_time, target_time_steps = input_window // 2, 10
            target_predictions = spike_preds[:, target_start_time:target_start_time+target_time_steps, :]
            B = target_predictions.shape[0]
            half = B // 2
            target_spikes = np.zeros_like(target_predictions)
            if half > 0:
                target_spikes[half:] = target_spike_prob
            eps = 1e-7
            clipped = np.clip(target_predictions, eps, 1 - eps)
            # 聚合10个time steps（保持与原实现一致，取max）
            clipped = np.max(clipped, axis=1)
            target_spikes_max = np.max(target_spikes, axis=1)
            bce = -target_spikes_max*np.log(clipped) - (1-target_spikes_max)*np.log(1-clipped)
            pred_loss = float(np.mean(bce))
            reg_loss = 0.001 * float(np.mean(firing_rates_np))
            return pred_loss + reg_loss

        return evaluate_loss

    def optimize_activity(self, num_iterations=100, learning_rate=0.01, batch_size=4, 
                         target_spike_prob=0.8, save_dir=None, start_time_ms=100):
        """
        Execute activity optimization using TensorFlow compile and fit methods
        
        Args:
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate
            batch_size: Batch size
            target_spike_prob: Target spike probability
            save_dir: Directory to save results
            start_time_ms: Start time (ignore first start_time_ms milliseconds)
            
        Returns:
            optimized_firing_rates: Optimized firing rates
            loss_history: Loss history
        """
        print(f"\nStarting Activity Optimization (NumPy/PyTorch method):")
        print(f"  Iterations: {num_iterations}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {batch_size}")
        print(f"  Target spike probability: {target_spike_prob}")
        print(f"  Start time: {start_time_ms}ms")
        print("-" * 50)
        
        # Prepare initial firing rates
        if self.init_firing_rates is not None:
            print("Using loaded initial firing rates")
            initial_firing_rates = self.processor.prepare_firing_rates_for_optimization(
                self.init_firing_rates, batch_size, start_time_ms
            )
        else:
            print("Generating random initial firing rates")
            initial_firing_rates = self.processor.generate_background_firing_rates(batch_size)
        
        # Select fixed excitatory indices
        mono_syn_rnd = np.random.default_rng(42)
        fixed_exc_indices = mono_syn_rnd.choice(self.num_segments_exc, size=3, replace=False)
        print(f"Fixed excitatory segments for adding spikes: {fixed_exc_indices}")
        
        loss_fn = self.create_optimization_model(fixed_exc_indices, target_spike_prob)
        
        firing_rates = initial_firing_rates.astype(np.float32)
        
        print("Ensuring TCN model parameters are frozen... (PyTorch eval mode)")
        # 简单的NumPy梯度下降（近似）：使用有限差分估计梯度
        loss_history = []
        firing_rates_history = []
        epsilon = 1e-3
        lr = float(learning_rate)
        optimized_firing_rates = firing_rates.copy()

        for epoch in range(num_iterations):
            # 计算当前loss
            current_loss = loss_fn(optimized_firing_rates)
            loss_history.append(current_loss)

            # 每100步记录一次状态
            if epoch % 50 == 0:
                firing_rates_history.append(optimized_firing_rates.copy())
                print(f"  Iter {epoch:4d}: Loss = {current_loss:.6f}")

            # 估计梯度（随机选择一个子块以减少计算量）
            grad = np.zeros_like(optimized_firing_rates)
            # 这里采用整体扰动的近似（SPSA风格）
            perturb = np.random.choice([-1.0, 1.0], size=optimized_firing_rates.shape).astype(np.float32)
            loss_pos = loss_fn(optimized_firing_rates + epsilon * perturb)
            loss_neg = loss_fn(optimized_firing_rates - epsilon * perturb)
            grad_est = (loss_pos - loss_neg) / (2 * epsilon) * perturb
            grad += grad_est

            # 梯度下降更新
            optimized_firing_rates -= lr * grad
            # 约束到[0, 0.1]
            optimized_firing_rates = np.clip(optimized_firing_rates, 0.0, 0.1)

        print("-" * 50)
        print(f"Optimization completed! Final loss: {loss_history[-1]:.6f}")
        
        # # Save final results
        # if save_dir:
        #     self.save_optimization_results(
        #         optimized_firing_rates, loss_history, firing_rates_history, 
        #         fixed_exc_indices, save_dir, start_time_ms
        #     )
        
        return optimized_firing_rates, loss_history, fixed_exc_indices
    
    # def save_optimization_results(self, optimized_firing_rates, loss_history, 
    #                             firing_rates_history, fixed_exc_indices, save_dir, start_time_ms):
    #     """
    #     保存优化结果
    #     """
    #     os.makedirs(save_dir, exist_ok=True)
        
    #     # 保存数据
    #     results = {
    #         'optimized_firing_rates': optimized_firing_rates,
    #         'loss_history': loss_history,
    #         'firing_rates_history': firing_rates_history,
    #         'fixed_exc_indices': fixed_exc_indices,
    #         'model_path': self.processor.model_path,
    #         'optimization_params': {
    #             'time_duration_ms': self.time_duration_ms,
    #             'num_segments_exc': self.num_segments_exc,
    #             'num_segments_inh': self.num_segments_inh,
    #             'start_time_ms': start_time_ms
    #         }
    #     }
        
    #     result_file = os.path.join(save_dir, f'activity_optimization.pickle')
    #     with open(result_file, 'wb') as f:
    #         pickle.dump(results, f)
        
    #     print(f"结果已保存到: {result_file}")
        
    #     # 使用可视化函数生成图表
    #     print("生成优化过程可视化...")
        
    #     # 1. 损失历史
    #     plot_loss_history(loss_history, save_path=os.path.join(save_dir, "loss_history.png"))
        
    #     # 2. Firing rates演化（如果有历史数据）
    #     if firing_rates_history:
    #         plot_firing_rates_evolution(
    #             firing_rates_history, self.num_segments_exc, self.num_segments_inh,
    #             self.time_duration_ms, self.input_window_size, 
    #             save_path=os.path.join(save_dir, "firing_rates_evolution.png")
    #         )
        
    #     # 3. 优化总结
    #     plot_optimization_summary(
    #         loss_history, firing_rates_history, self.num_segments_exc, self.num_segments_inh,
    #         self.time_duration_ms, self.input_window_size, 
    #         save_path=os.path.join(save_dir, "summary.png")
    #     )
        
    #     print("优化结果可视化已生成")
    
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
        
        print("评估使用现有的PyTorch模型（已在processor中构建）...")
        
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


def main():
    """
    主函数：运行activity optimization
    """
    print("=== Activity Optimization ===")
    models_dir = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut_SJC_funcgroup2_var2/models/NMDA/depth_7_filters_256_window_400/'
    init_firing_rates_path = './init_firing_rate_array.npy'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'./results/6_activity_optimization_results/{timestamp}'
    if not os.path.exists(init_firing_rates_path):
        print(f"警告：初始firing rates文件不存在: {init_firing_rates_path}")
        print("将使用随机生成的初始firing rates")
        init_firing_rates_path = None
    print("寻找最佳模型...")
    try:
        model_path, params_path = find_best_model(models_dir)
        print(f"选择的模型: {os.path.basename(model_path)}")
    except Exception as e:
        print(f"错误: {e}")
        return
    optimizer = ActivityOptimizer(
        model_path=model_path, 
        model_params_path=params_path, 
        init_firing_rates_path=init_firing_rates_path,
        time_duration_ms=400
    )
    optimized_firing_rates, loss_history, fixed_exc_indices = optimizer.optimize_activity(
        num_iterations=1000,
        learning_rate=0.001,
        batch_size=1,
        target_spike_prob=1,
        save_dir=save_dir,
        start_time_ms=0
    )
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
        create_optimization_report(
            loss_history=loss_history,
            firing_rates_history=[],
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
        try:
            optimized_sample = optimized_firing_rates[0]
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
            visualize_firing_rates_trace(
                firing_rates=optimized_sample,
                num_exc_segments=639,
                save_path=os.path.join(save_dir, 'optimized_firing_rates_raster.png'),
                title="Optimized Firing Rates - Raster Plot",
                max_segments_to_show=10,
                specified_segments=specified_segments
            )
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