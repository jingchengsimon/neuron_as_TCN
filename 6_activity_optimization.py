import os
import torch
import pickle
from datetime import datetime
from utils.firing_rates_processor import FiringRatesProcessor
from utils.tcn_poisson_model import TCNPoissonModel
from utils.find_best_model import find_best_model
from utils.visualization_utils import (
    visualize_optimized_firing_rates, create_optimization_report
)


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
        
        # Create TCN Poisson model
        self.tcn_poisson_model = TCNPoissonModel(
            model_path=model_path,
            model_params=model_params,
            input_window_size=self.input_window_size,
            num_segments_exc=self.num_segments_exc,
            num_segments_inh=self.num_segments_inh,
            time_duration_ms=time_duration_ms
        )
        
        # Create data processor for utility functions
        self.processor = FiringRatesProcessor(time_duration_ms=time_duration_ms, return_torch=True)
        
        # Load initial firing rates (if provided)
        self.init_firing_rates = None
        if init_firing_rates_path and os.path.exists(init_firing_rates_path):
            self.init_firing_rates = self.processor.load_init_firing_rates(init_firing_rates_path, self.num_segments_total)
        
        print(f"Activity Optimizer initialized successfully.")
        if self.init_firing_rates is not None:
            print(f"  Initial firing rates loaded: {self.init_firing_rates.shape}")
        
        # Set default data type to torch
        self.use_torch = True
    
    def compute_loss(self, firing_rates, spike_preds, target_spike_prob, learning_rate):
        """
        Calculate optimization loss function
        
        Args:
            firing_rates: (batch, num_segments_total, time_duration) torch tensor
            fixed_exc_indices: fixed excitatory indices
            target_spike_prob: target spike probability
            
        Returns:
            loss: scalar loss value
        """
    
        # Ensure spike_preds is on the same device as firing_rates
        spike_preds = spike_preds.to(firing_rates.device, dtype=torch.float32)
 
        # 2) Take 10 time steps after window center for BCE
        target_start_time, target_time_steps = self.input_window_size // 2, 10
        target_predictions = spike_preds[:, target_start_time:target_start_time+target_time_steps, :]

        target_predictions_max = torch.max(target_predictions, dim=1)[0]  # torch.max returns (values, indices)
        
        # Create target spikes with same shape as target_predictions_max
        target_spikes_max = torch.full_like(target_predictions_max, fill_value=target_spike_prob)
        target_spikes_max[:target_spikes_max.shape[0]//2] = 0.0 # Set first half to 0, second half to target_spike_prob

        # Use PyTorch's built-in BCE loss function
        pred_loss = torch.nn.functional.binary_cross_entropy(target_predictions_max, target_spikes_max)
        # reg_loss = learning_rate * torch.mean(firing_rates)

        return pred_loss # + reg_loss

    def optimize_activity(self, num_iterations=100, learning_rate=0.01, batch_size=4, 
                         target_spike_prob=0.8, start_time_ms=100):
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
        
        torch.manual_seed(42)

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
        
        # Select fixed excitatory indices
        fixed_exc_indices = torch.randperm(self.num_segments_exc)[:3].numpy()
        print(f"Fixed excitatory segments for adding spikes: {fixed_exc_indices}")
        
        # Convert firing_rates to torch tensor and set as optimizable parameter
        firing_rates = convert_to_preferred_type(initial_firing_rates, use_torch=self.use_torch, dtype=torch.float32)
        firing_rates.requires_grad_(True)

        # Choose an optimizer, add learning rate scheduler
        optimizer = torch.optim.Adam([firing_rates], lr=float(learning_rate))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, verbose=True
        )

        print("Ensuring TCN model parameters are frozen... (PyTorch eval mode)")
        loss_history = []
        firing_rates_history = []

        for epoch in range(num_iterations):
            # Zero gradients
            optimizer.zero_grad()

            # 1) Generate spikes and predictions using TCN Poisson model
            spike_preds, _ = self.tcn_poisson_model(firing_rates, fixed_exc_indices)
            
            # 2) Compute loss
            loss = self.compute_loss(firing_rates, spike_preds, target_spike_prob, learning_rate)
            loss_history.append(float(loss.detach().cpu().numpy()))

            # Record status every 100 steps
            if epoch % 20 == 0:
                firing_rates_history.append(firing_rates.detach().cpu().numpy())
                print(f"  Iter {epoch:4d}: Loss = {loss:.6f}")
        
            # Backward propagation
            loss.backward()
            
            # Gradient clipping (optional)
            # torch.nn.utils.clip_grad_norm_([firing_rates], max_norm=1.0)
            
            # Update parameters
            optimizer.step()

            # Update learning rate
            scheduler.step(loss)

        print("-" * 50)
        print(f"Optimization completed! Final loss: {loss_history[-1]:.6f}")
        
        optimized_firing_rates = firing_rates.detach().numpy()

        return optimized_firing_rates, firing_rates_history, loss_history, fixed_exc_indices
    
    def evaluate_optimized_activity(self, optimized_firing_rates, fixed_exc_indices, num_evaluations=10):
        """
        Evaluate optimized activity
        
        Args:
            optimized_firing_rates: optimized firing rates (batch_size, num_segments, time_duration)
            fixed_exc_indices: fixed excitatory indices
            num_evaluations: number of evaluations
            
        Returns:
            evaluation_results: evaluation results dictionary
        """
        print(f"\nEvaluating optimized activity (running {num_evaluations} times)...")
        print("Evaluation using existing PyTorch model (already built in processor)...")
        
        spike_probabilities = []
        
        for eval_idx in range(num_evaluations):
            try:
                # Use TCN Poisson model for evaluation (no gradients)
                spike_predictions, _ = self.tcn_poisson_model.predict_eval(
                    optimized_firing_rates, fixed_exc_indices
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
                
                if eval_idx % 5 == 0:
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
    # models_dir = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut_SJC_funcgroup2_var2/models/NMDA_torch/depth_7_filters_256_window_400/'
    models_dir = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut_SJC_funcgroup2_var2_AMPA/models/AMPA_torch/depth_1_filters_128_window_400/'
    init_firing_rates_path = './init_firing_rate_array.npy'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'./results/6_activity_optimization_results/{timestamp}'
    if not os.path.exists(init_firing_rates_path):
        print(f"Warning: Initial firing rates file does not exist: {init_firing_rates_path}")
        print("Will use randomly generated initial firing rates")
        init_firing_rates_path = None
    print("Finding best model...")
    try:
        model_path, params_path = find_best_model(models_dir)
        print(f"Selected model: {os.path.basename(model_path)}")
    except Exception as e:
        print(f"Error: {e}")
        
    optimizer = ActivityOptimizer(
        model_path=model_path, 
        model_params_path=params_path, 
        init_firing_rates_path=init_firing_rates_path,
        time_duration_ms=400
    )

    optimized_firing_rates, firing_rates_history, loss_history, fixed_exc_indices = optimizer.optimize_activity(
        num_iterations=500,
        learning_rate=0.001,
        batch_size=1,  # 增加batch_size，确保有对照组和刺激组
        target_spike_prob=1,
        start_time_ms=0
    )

    evaluation_results = optimizer.evaluate_optimized_activity(
        optimized_firing_rates, fixed_exc_indices, num_evaluations=20
    )
    print("\n=== Optimization Completed ===")
    print(f"Optimized firing rates shape: {optimized_firing_rates.shape}")
    print(f"Final loss: {loss_history[-1]:.6f}")
    print(f"Fixed excitatory segments for adding spikes: {fixed_exc_indices}")
    
    # ## Get optimized firing rates from pickle file
    # save_dir = './results/activity_optimization_results/20250827_163803'
    # with open(os.path.join(save_dir, 'activity_optimization.pickle'), 'rb') as f:
    #     data = pickle.load(f)
    # optimized_firing_rates = data['optimized_firing_rates']
    # fixed_exc_indices = data['fixed_exc_indices']
    
    # Optional: visualize optimized firing rates
    try:
        print("\nGenerating visualization for optimized firing rates...")
        create_optimization_report(
            loss_history=loss_history,
            firing_rates_history=firing_rates_history,
            optimized_firing_rates=optimized_firing_rates,
            fixed_exc_indices=fixed_exc_indices,
            num_segments_exc=639,
            num_segments_inh=640,
            time_duration_ms=300,
            input_window_size=300,
            save_dir=save_dir,
            report_name="activity_optimization"
        )
        print("Complete optimization report generated")
    except ImportError:
        print("Visualization module not available, skipping visualization step")
    except Exception as e:
        print(f"Error generating visualization: {e}")
        print("Trying to use basic visualization functions...")
        try:
            visualize_optimized_firing_rates(optimized_firing_rates, fixed_exc_indices, 
                                   num_exc_segments=639, save_dir=None, 
                                   title_prefix="Optimized Firing Rates")
        except Exception as e2:
            print(f"Basic visualization also failed: {e2}")

if __name__ == "__main__":
    main() 