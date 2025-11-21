import gc
import os
import numpy as np
import torch
import pickle
from utils.tcn_poisson_model import TCNPoissonModel
from utils.find_best_model import find_best_model
from utils.visualization_utils import create_optimization_report, plot_average_heatmap
from utils.simpleModelVer2_Aim2 import run_simulation_batch

def convert_to_preferred_type(data, use_torch=True, device=None, dtype=torch.float32):
    """
    Convert data to torch tensor or numpy array based on use_torch setting
    
    Args:
        data: numpy array or torch tensor
        use_torch: whether to use torch tensors
        device: target device (only for torch tensors)
        dtype: target data type (only for torch tensors)
        
    Returns:
        converted data (torch tensor or numpy array)
    """
    if use_torch:
        # Convert to torch tensor
        if isinstance(data, torch.Tensor):
            if device is not None:
                return data.to(device, dtype=dtype)
            return data.to(dtype=dtype)
        else:
            tensor = torch.from_numpy(data).to(dtype=dtype)
            if device is not None:
                return tensor.to(device)
            return tensor
    else:
        # Convert to numpy array
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data


class FiringRatesProcessor:
    """
    Class specialized for processing firing rates data
    Contains data loading, preprocessing, and utility functions
    """
    
    def __init__(self, time_duration_ms=300, return_torch=True, device=None):
        """
        Initialize data processor
        
        Args:
            time_duration_ms: Time duration, default 300ms
            return_torch: Whether to return torch tensors, default True
            device: Device for torch tensors
        """
        # Basic configuration
        self.time_duration_ms = time_duration_ms
        self.return_torch = return_torch
        self.device = device
        
        print(f"FiringRatesProcessor initialized successfully:")
        print(f"  Time duration: {self.time_duration_ms}ms")
        print(f"  Return type: {'torch tensors' if return_torch else 'numpy arrays'}")
        if device is not None:
            print(f"  Device: {device}")
    
    def load_init_firing_rates(self, firing_rates_path, num_segments_total=1279):
        """
        Load initial firing rates
        
        Args:
            firing_rates_path: Path to firing rates .npy file
            num_segments_total: Expected number of segments (default 1279)
            
        Returns:
            firing_rates: (num_segments_total, full_time_duration) numpy array or torch tensor
        """
        print(f"Loading initial firing rates: {firing_rates_path}")
        try:
            firing_rates = np.load(firing_rates_path)
            print(f"  File loaded successfully, shape: {firing_rates.shape}")
            print(f"  Data type: {firing_rates.dtype}")
            print(f"  Value range: [{np.min(firing_rates):.6f}, {np.max(firing_rates):.6f}]")
            
            # Validate data format
            if len(firing_rates.shape) != 2:
                raise ValueError(f"Expected 2D array, but got {len(firing_rates.shape)}D array")
            
            num_segments, _ = firing_rates.shape
            if num_segments != num_segments_total:
                print(f"Warning: Segments count in file ({num_segments}) does not match expected ({num_segments_total})")
            
            firing_rates = firing_rates.astype(np.float32)
            return convert_to_preferred_type(firing_rates, use_torch=self.return_torch)
            
        except Exception as e:
            print(f"Error loading initial firing rates: {e}")
            return None
    
    def prepare_firing_rates_for_optimization(self, firing_rates, batch_size=2, start_time_ms=100, num_segments_total=1279):
        """
        Prepare firing rates for optimization, extract specified time range data
        
        Args:
            firing_rates: (num_segments_total, full_time_duration) or (batch_size, num_segments_total, full_time_duration)
            batch_size: batch size
            start_time_ms: start time, default 100ms
            num_segments_total: Expected number of segments (default 1279)
            
        Returns:
            prepared_rates: (batch_size, num_segments_total, time_duration_ms)
        """
        # Handle input that might be torch tensor
        is_torch_input = isinstance(firing_rates, torch.Tensor)
        if is_torch_input:
            firing_rates_np = firing_rates.detach().cpu().numpy()
        else:
            firing_rates_np = firing_rates
            
        if len(firing_rates_np.shape) == 2:
            # Single sample, expand to batch
            firing_rates_np = firing_rates_np[np.newaxis, :, :]  # (1, num_segments, full_time)
        
        current_batch_size, num_segments, full_time = firing_rates_np.shape
        
        # Handle segments count mismatch
        if num_segments != num_segments_total:
            print(f"Adjusting segments count: {num_segments} -> {num_segments_total}")
            if num_segments < num_segments_total:
                # If segments insufficient, pad with zeros
                padding_needed = num_segments_total - num_segments
                padding = np.zeros((current_batch_size, padding_needed, full_time), dtype=firing_rates_np.dtype)
                firing_rates_np = np.concatenate([firing_rates_np, padding], axis=1)
                print(f"  Added {padding_needed} zero-padded segments")
            else:
                # If segments too many, take the first part
                firing_rates_np = firing_rates_np[:, :num_segments_total, :]
                print(f"  Took first {num_segments_total} segments")
        
        # Calculate extraction time range
        end_time_ms = start_time_ms + self.time_duration_ms
        
        if end_time_ms > full_time:
            raise ValueError(f"Time range exceeds data length: need {end_time_ms}ms, but data only has {full_time}ms")
        
        # Extract data for specified time period
        extracted_rates = firing_rates_np[:, :, start_time_ms:end_time_ms]
        
        # If more batches needed, copy data
        if batch_size > current_batch_size:
            # Repeat data to achieve required batch size
            repeat_times = batch_size // current_batch_size
            remainder = batch_size % current_batch_size
            
            repeated_rates = np.tile(extracted_rates, (repeat_times, 1, 1))
            if remainder > 0:
                extra_rates = extracted_rates[:remainder, :, :]
                extracted_rates = np.concatenate([repeated_rates, extra_rates], axis=0)
            else:
                extracted_rates = repeated_rates
        elif batch_size < current_batch_size:
            # Take first batch_size samples
            extracted_rates = extracted_rates[:batch_size, :, :]
        
        # print(f"Prepare for optimization:")
        # print(f"  Original input shape: {firing_rates_np.shape}")
        # print(f"  Extracted time: {start_time_ms}-{end_time_ms}ms")
        # print(f"  Extracted input shape: {extracted_rates.shape}")
        
        extracted_rates = extracted_rates.astype(np.float32)
        return convert_to_preferred_type(extracted_rates, use_torch=self.return_torch, device=self.device)
    
    def generate_background_firing_rates(self, batch_size=2, num_segments_exc=639, num_segments_inh=640):
        """
        Generate background firing rates (Poisson process)
        
        Args:
            batch_size: batch size
            num_segments_exc: number of excitatory segments (default 639)
            num_segments_inh: number of inhibitory segments (default 640)
            
        Returns:
            firing_rates: (batch_size, num_segments_total, time_duration_ms)
        """
        # Set different background firing rates for excitatory and inhibitory segments
        exc_rate = 0.01  # Excitatory background firing rate (1%)
        inh_rate = 0.02  # Inhibitory background firing rate (2%)
        
        num_segments_total = num_segments_exc + num_segments_inh
        
        # Generate firing rates with correct shape: (batch_size, num_segments_total, time_duration_ms)
        firing_rates = np.zeros((batch_size, num_segments_total, self.time_duration_ms))
        
        # Excitatory segments
        firing_rates[:, :num_segments_exc, :] = np.random.uniform(
            0.005, 0.02, (batch_size, num_segments_exc, self.time_duration_ms)
        )
        
        # Inhibitory segments
        firing_rates[:, num_segments_exc:, :] = np.random.uniform(
            0.01, 0.03, (batch_size, num_segments_inh, self.time_duration_ms)
        )
        
        firing_rates = firing_rates.astype(np.float32)
        return convert_to_preferred_type(firing_rates, use_torch=self.return_torch, device=self.device)
    

class ActivityOptimizer:
    """
    Activity optimization class based on trained TCN model
    Focuses on optimization algorithms and gradient computation, delegates data processing to FiringRatesProcessor
    """
    
    def __init__(self, model_path, model_params_path, init_firing_rates=None, time_duration_ms=300, monoconn_seg_indices=None):
        """
        Initialize optimizer
        
        Args:
            model_path: Path to trained model .h5 file
            model_params_path: Path to corresponding parameters .pickle file
            init_firing_rates: initial firing rates numpy array
            time_duration_ms: Time duration, default 300ms
            monoconn_seg_indices: fixed excitatory indices
        """
        # Load model parameters
        with open(model_params_path, 'rb') as f:
            model_params = pickle.load(f)
        
        # Extract model architecture information
        architecture_dict = model_params['architecture_dict']
        self.input_window_size = architecture_dict['input_window_size']
        self.num_segments_exc = 639
        self.num_segments_inh = 640
        self.num_segments_total = self.num_segments_exc + self.num_segments_inh
        self.time_duration_ms = time_duration_ms
        self.monoconn_seg_indices = monoconn_seg_indices
        
        # Create TCN Poisson model
        self.tcn_poisson_model = TCNPoissonModel(
            model_path=model_path,
            model_params=model_params,
            input_window_size=self.input_window_size,
            num_segments_exc=self.num_segments_exc,
            num_segments_inh=self.num_segments_inh,
            time_duration_ms=time_duration_ms
        )
        
        # Set device and ensure model is on the correct device
        self.device = next(self.tcn_poisson_model.parameters()).device
        
        # Create data processor for utility functions
        self.processor = FiringRatesProcessor(time_duration_ms=time_duration_ms, return_torch=True, device=self.device)
        
        # Load initial firing rates (if provided)
        # self.init_firing_rates = None
        # if init_firing_rates_path and os.path.exists(init_firing_rates_path):
        #     self.init_firing_rates = self.processor.load_init_firing_rates(init_firing_rates_path, self.num_segments_total)
        
        self.init_firing_rates = init_firing_rates
        
        print(f"Activity Optimizer initialized successfully.")
        # print(f"  Device: {self.device}")
        # print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        if self.init_firing_rates is not None:
            print(f"  Initial firing rates loaded: {self.init_firing_rates.shape}")
            # Move initial firing rates to device if they are torch tensors
            if isinstance(self.init_firing_rates, torch.Tensor):
                self.init_firing_rates = self.init_firing_rates.to(self.device)
        
        # Set default data type to torch
        self.use_torch = True
    
    def compute_loss(self, spike_preds, target_spike_prob):
        """
        Calculate optimization loss function
        
        Args:
            firing_rates: (batch, num_segments_total, time_duration) torch tensor
            target_spike_prob: target spike probability
            
        Returns:
            loss: scalar loss value
        """
    
        # Ensure spike_preds is on the same device as tcn_poisson_model
        spike_preds = spike_preds.to(self.device, dtype=torch.float32)
 
        # Take 10 time steps after window center for BCE
        target_start_time, target_time_steps = self.input_window_size // 2, 10
        end = min(spike_preds.shape[1], target_start_time + target_time_steps)
        spike_preds_trunc = spike_preds[:, target_start_time:end, :]
        spike_preds_max = torch.max(spike_preds_trunc, dim=1)[0]  # torch.max returns (values, indices)
        
        # Create target spikes with same shape as spike_preds_trunc_max
        spike_target_max = torch.full_like(spike_preds_max, fill_value=target_spike_prob)
        spike_target_max[:spike_target_max.shape[0]//2] = 0.0 # Set first half to 0, second half to target_spike_prob

        # Use PyTorch's built-in BCE loss function
        pred_loss = torch.nn.functional.binary_cross_entropy(spike_preds_max, spike_target_max)

        # Add penalty term: ensure second half >= first half (non-negative difference)
        batch_size = spike_preds_max.shape[0]
        if batch_size >= 2:
            first_half = spike_preds_max[:batch_size//2]  # Control group
            second_half = spike_preds_max[batch_size//2:]  # Stimulation group
            # Penalty for negative differences (second_half < first_half)
            diff_penalty = torch.mean(torch.relu(first_half - second_half))
        else:
            diff_penalty = torch.tensor(0.0, device=spike_preds_max.device)
        
        # Apply coefficient to make diff penalty contribute meaningfully to loss
        diff_penalty_weight = 20.0  # Adjust this coefficient based on training dynamics
        total_loss = pred_loss # + diff_penalty_weight * diff_penalty

        # reg_loss = learning_rate * torch.mean(firing_rates)
        return total_loss, spike_preds_max

    def optimize_activity(self, num_iterations=100, learning_rate=0.01, batch_size=4, 
                         target_spike_prob=0.8, start_time_ms=100, random_seed=42, freq_inh_hz=4.0):
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
        print("-" * 50)
        
        torch.manual_seed(random_seed)

        # Prepare initial firing rates
        if self.init_firing_rates is not None:
            print("Using loaded initial firing rates")
            initial_firing_rates = self.processor.prepare_firing_rates_for_optimization(
                self.init_firing_rates, batch_size, start_time_ms, self.num_segments_total
            )
        else:
            print("Generating random initial firing rates")
            initial_firing_rates = self.processor.generate_background_firing_rates(
                batch_size, self.num_segments_exc, self.num_segments_inh
            )
        
        # Convert firing_rates to torch nn.Parameter (default requires_grad=True)
        # Ensure initial_firing_rates is on the correct device
        if isinstance(initial_firing_rates, torch.Tensor):
            initial_firing_rates = initial_firing_rates.to(self.device)
        firing_rates = torch.nn.Parameter(initial_firing_rates)
        # Only allow gradients for excitatory segments (first 639), freeze inhibitory segments (last 640)
        firing_rates.register_hook(lambda grad: grad * torch.cat([
            torch.ones_like(grad[:, :self.num_segments_exc, :]),
            torch.zeros_like(grad[:, self.num_segments_exc:, :])
        ], dim=1))
        
        # Choose an optimizer, add learning rate scheduler
        optimizer = torch.optim.Adam([firing_rates], lr=float(learning_rate))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50
        )

        print("Ensuring TCN model parameters are frozen... (PyTorch eval mode)")
        
        firing_rates_history = [firing_rates.clone().detach().cpu().numpy()]
        loss_history = []
        spike_preds_history = []
        gradient_norm_history = []

        # 梯度累积参数
        accumulation_steps = 2  # 每2步累积一次梯度
        
        for epoch in range(num_iterations):
            # 只在累积步骤开始时清零梯度
            if epoch % accumulation_steps == 0:
                optimizer.zero_grad()
                
            spike_preds, _ = self.tcn_poisson_model(firing_rates, self.monoconn_seg_indices)
            loss, spike_preds_max = self.compute_loss(spike_preds, target_spike_prob)
            
            # 缩放损失以适应梯度累积
            loss = loss / accumulation_steps

            batch1_max = torch.mean(torch.max(spike_preds_max[:batch_size//2], dim=1)[0]).detach().cpu().numpy().item()      
            batch2_max = torch.mean(torch.max(spike_preds_max[batch_size//2:], dim=1)[0]).detach().cpu().numpy().item()

            # Backward propagation
            loss.backward()
            
            # 只在累积步骤结束时更新参数
            if (epoch + 1) % accumulation_steps == 0:
                # torch.nn.utils.clip_grad_norm_([firing_rates], max_norm=2.0) # Gradient clipping (optional)
                
                # Calculate gradient norm
                grad_norm = torch.norm(firing_rates.grad).item()
                gradient_norm_history.append(grad_norm)

                optimizer.step()
                scheduler.step(loss.item() * accumulation_steps)  # 恢复原始损失值用于调度器

            # After updating excitatory by gradient, update inhibitory deterministically from excitatory
            with torch.no_grad():
                time_duration_ms = firing_rates.shape[2]
                inh_delay = 4 # ms
                for batch_idx in range(firing_rates.shape[0]):
                    exc_firing_rate_tensor = firing_rates[batch_idx, :self.num_segments_exc, :] # (639, T)
                    inh_firing_rate_tensor_new = torch.zeros(self.num_segments_inh, time_duration_ms, device=firing_rates.device, dtype=firing_rates.dtype)

                    
                    inh_firing_rate_tensor_new[1:,inh_delay:] = (float(freq_inh_hz)/1000.0)*(exc_firing_rate_tensor[:,:time_duration_ms-inh_delay]/torch.mean(exc_firing_rate_tensor[:,:time_duration_ms-inh_delay]))
                    inh_firing_rate_tensor_new[0,inh_delay:] = (float(freq_inh_hz)/1000.0)*(exc_firing_rate_tensor[0,:time_duration_ms-inh_delay]/torch.mean(exc_firing_rate_tensor[:,:time_duration_ms-inh_delay]))
                    firing_rates[batch_idx, self.num_segments_exc:, :] = inh_firing_rate_tensor_new

            # 只在累积步骤结束时记录历史
            if (epoch + 1) % accumulation_steps == 0:
                loss_history.append(float(loss.detach().cpu().numpy() * accumulation_steps))  # 恢复原始损失值
                spike_preds_history.append([batch1_max, batch2_max])

            # 定期清理GPU缓存和垃圾回收
            if epoch % 5 == 0:  # 更频繁的清理
                firing_rates_history.append(firing_rates.clone().detach().cpu().numpy())
                torch.cuda.empty_cache()
                gc.collect()

            # Record spike predictions every 100 steps
            if epoch % 100 == 0:
                print(f"Iter {epoch:4d}: Pred_loss: {loss.item():.4f}, Spike_preds_max: [{batch1_max:.3f}, {batch2_max:.3f}], Diff = {batch2_max-batch1_max:.3f}")
                
            # 删除不再需要的中间变量以释放内存
            del spike_preds, spike_preds_max, loss

        print("-" * 50)
        print(f"Optimization completed! Final loss: {loss_history[-1]:.6f}")
        
        optimized_firing_rates = firing_rates.detach().cpu().numpy()

        return optimized_firing_rates, firing_rates_history, loss_history, spike_preds_history, gradient_norm_history
    
    def evaluate_optimized_activity(self, optimized_firing_rates, num_evaluations=10):
        """
        Evaluate optimized activity
        
        Args:
            optimized_firing_rates: optimized firing rates (batch_size, num_segments, time_duration)
            num_evaluations: number of evaluations
            
        Returns:
            evaluation_results: evaluation results dictionary
        """
        print(f"\nEvaluating optimized activity (running {num_evaluations} times)...")
        # print("Evaluation using existing PyTorch model (already built in processor)...")
        
        spike_probabilities = []
        
        for eval_idx in range(num_evaluations):
            try:
                # Use TCN Poisson model for evaluation (no gradients)
                spike_predictions, _ = self.tcn_poisson_model.predict_eval(
                    optimized_firing_rates, self.monoconn_seg_indices
                )
                
                # Ensure spike_predictions is torch tensor
                spike_predictions = convert_to_preferred_type(spike_predictions, use_torch=self.use_torch)
                
                # Predict for each batch
                batch_spike_probs = []
                for batch_idx in range(spike_predictions.shape[0]):
                    # Take prediction probabilities for 10 time steps after half window size
                    target_start_time, target_time_steps = self.input_window_size // 2, 10  # Focus on 10 time steps after mono synaptic spike
                    final_predictions = spike_predictions[batch_idx, target_start_time:target_start_time+target_time_steps, 0]
                    batch_spike_probs.extend(final_predictions.tolist())
                
                spike_probabilities.extend(batch_spike_probs)
                
                # 清理中间变量
                del spike_predictions, batch_spike_probs
                
                # 定期清理GPU缓存
                if eval_idx % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    print(f"  Evaluation progress: {eval_idx + 1}/{num_evaluations}")
                    
            except Exception as e:
                print(f"  Evaluation {eval_idx + 1} failed: {e}")
                # If evaluation fails, add default values
                default_probs = [0.5] * 10  # Default probability
                spike_probabilities.extend(default_probs)
        
        # Calculate statistics using torch
        spike_probabilities_tensor = torch.tensor(spike_probabilities, dtype=torch.float32)
        
        evaluation_results = {
            'mean_spike_probability': float(torch.mean(spike_probabilities_tensor)),
            'std_spike_probability': float(torch.std(spike_probabilities_tensor)),
            'min_spike_probability': float(torch.min(spike_probabilities_tensor)),
            'max_spike_probability': float(torch.max(spike_probabilities_tensor)),
            'spike_probabilities': spike_probabilities_tensor
        }
        
        print(f"Evaluation results:")
        print(f"  Mean spike probability: {evaluation_results['mean_spike_probability']:.4f}")
        print(f"  Standard deviation: {evaluation_results['std_spike_probability']:.4f}")
        print(f"  Minimum: {evaluation_results['min_spike_probability']:.4f}")
        print(f"  Maximum: {evaluation_results['max_spike_probability']:.4f}")
        
        return evaluation_results


def main():
    """
    Main function: run activity optimization
    """
    print("=== Activity Optimization ===")
    models_dir = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut_SJC_funcgroup2_var2/models/NMDA_torch/depth_7_filters_256_window_400/'
    # models_dir = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut_SJC_funcgroup2_var2_AMPA/models/AMPA_torch/depth_1_filters_128_window_400/'
    
    data_type = 'NMDA' if 'NMDA' in models_dir else 'AMPA'
    batch_size = 200

    # Select fixed excitatory indices
    range0_idx = [1, 2, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 
                  27, 30, 31, 34, 35, 36, 37, 40, 43, 44, 45, 46, 47, 48, 49, 52, 53, 54, 55, 
                  56, 57, 58, 61, 64, 65, 66, 69, 72, 73, 74, 75, 76, 79, 80, 81, 82, 83, 84]

    for save_dir_idx in range(2, 6):
        # random choose 3 indices
        np.random.seed(save_dir_idx)
        monoconn_seg_indices = np.random.choice(range0_idx, size=3, replace=False) # torch.randperm(self.num_segments_exc)[:3].numpy()
        print(f"Fixed excitatory segments for adding spikes: {monoconn_seg_indices}")
        
        # Store heatmap data for all seeds
        all_seeds_heatmap_data = []
            
        for random_seed in [42, 43, 44, 45, 46, 47]:
            save_dir = f'./results/6_activity_optimization_results/{data_type}_seed_{random_seed}_inhfixed_{save_dir_idx}'

            init_firing_rates = run_simulation_batch(num_runs=batch_size, epoch=random_seed, rebuild_cell=False)

            try:
                model_path, params_path = find_best_model(models_dir)
                # print(f"Selected model: {os.path.basename(model_path)}")
            except Exception as e:
                print(f"Error: {e}")
                
            optimizer = ActivityOptimizer(
                model_path=model_path, 
                model_params_path=params_path, 
                init_firing_rates=init_firing_rates,
                time_duration_ms=400,
                monoconn_seg_indices=monoconn_seg_indices
            )

            optimized_firing_rates, firing_rates_history, loss_history, spike_preds_history, gradient_norm_history = optimizer.optimize_activity(
                num_iterations=500,
                learning_rate=0.002,
                batch_size=batch_size,  # 增加batch_size，确保有对照组和刺激组
                target_spike_prob=1,
                start_time_ms=0,
                random_seed=random_seed
            )

            # evaluation_results = optimizer.evaluate_optimized_activity(
            #     optimized_firing_rates, num_evaluations=20
            # )
            print("\n===== Optimization Completed =====")
            
            try:
                heatmap_data = create_optimization_report(
                    loss_history=loss_history,
                    firing_rates_history=firing_rates_history,
                    spike_preds_history=spike_preds_history,
                    gradient_norm_history=gradient_norm_history,
                    monoconn_seg_indices=monoconn_seg_indices,
                    num_segments_exc=639, num_segments_inh=640,
                    time_duration_ms=400, input_window_size=400,
                    save_dir=save_dir, report_name="activity_optimization"
                )
                # Store heatmap data for this seed
                if heatmap_data is not None:
                    all_seeds_heatmap_data.append(heatmap_data)
                print("Complete optimization report generated")
            except ImportError:
                print("Visualization module not available, skipping visualization step")
        
        # After all seeds are processed, calculate average heatmap
        if len(all_seeds_heatmap_data) > 0:
            print(f"\n===== Calculating Average Heatmap for save_dir_idx {save_dir_idx} =====")
            
            # Calculate average for each heatmap component
            avg_initial_exc = np.mean([data['initial_exc_data'] for data in all_seeds_heatmap_data], axis=0)
            avg_optimized_exc = np.mean([data['optimized_exc_data'] for data in all_seeds_heatmap_data], axis=0)
            avg_initial_inh = np.mean([data['initial_inh_data'] for data in all_seeds_heatmap_data], axis=0)
            avg_optimized_inh = np.mean([data['optimized_inh_data'] for data in all_seeds_heatmap_data], axis=0)
            
            # Use indices from the first seed (should be the same for all seeds)
            first_seed_data = all_seeds_heatmap_data[0]
            
            average_heatmap_data = {
                'initial_exc_data': avg_initial_exc,
                'optimized_exc_data': avg_optimized_exc,
                'initial_inh_data': avg_initial_inh,
                'optimized_inh_data': avg_optimized_inh,
                'exc_indices': first_seed_data['exc_indices'],
                'opt_exc_indices': first_seed_data['opt_exc_indices'],
                'inh_indices': first_seed_data['inh_indices'],
                'opt_inh_indices': first_seed_data['opt_inh_indices']
            }
            
            # Create average heatmap visualization
            average_save_dir = f'./results/6_activity_optimization_results/{data_type}_average_inhfixed_{save_dir_idx}'
            os.makedirs(average_save_dir, exist_ok=True)
            
            average_heatmap_save_path = os.path.join(average_save_dir, 'average_combined_firing_rate_heatmap.png')
            plot_average_heatmap(
                average_heatmap_data, 
                monoconn_seg_indices,
                num_exc_segments=639, 
                num_inh_segments=640,
                save_path=average_heatmap_save_path,
                title_prefix=f"Average Heatmap (Seeds 42-46)"
            )
            
            print(f"Average heatmap saved to: {average_heatmap_save_path}")

if __name__ == "__main__":
    main() 