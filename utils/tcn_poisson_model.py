import torch
import torch.nn as nn
import numpy as np
from utils.fit_CNN_torch import TCNModel


class TCNPoissonModel(nn.Module):
    """
    TCN Model with Poisson spike generation using Straight-Through Estimator (STE)
    
    This class combines TCN model prediction with differentiable Poisson spike generation,
    designed for activity optimization tasks.
    """
    
    def __init__(self, model_path, model_params, input_window_size, 
                 num_segments_exc, num_segments_inh, time_duration_ms):
        """
        Initialize TCN Poisson Model
        
        Args:
            model_path: Path to the trained TCN model weights
            model_params: Model parameters dictionary containing architecture info
            input_window_size: Input window size in ms
            num_segments_exc: Number of excitatory segments
            num_segments_inh: Number of inhibitory segments
            time_duration_ms: Time duration in ms
        """
        super(TCNPoissonModel, self).__init__()
        
        self.input_window_size = input_window_size
        self.num_segments_exc = num_segments_exc
        self.num_segments_inh = num_segments_inh
        self.num_segments_total = num_segments_exc + num_segments_inh
        self.time_duration_ms = time_duration_ms
        
        # Load TCN model
        architecture_dict = model_params['architecture_dict']
        self.tcn_model = TCNModel(
            max_input_window_size=input_window_size,
            num_segments_exc=num_segments_exc,
            num_segments_inh=num_segments_inh,
            filter_sizes_per_layer=architecture_dict['filter_sizes_per_layer'],
            num_filters_per_layer=architecture_dict['num_filters_per_layer'],
            activation_function_per_layer=architecture_dict['activation_function_per_layer'],
            strides_per_layer=architecture_dict['strides_per_layer'],
            dilation_rates_per_layer=architecture_dict['dilation_rates_per_layer'],
            initializer_per_layer=architecture_dict['initializer_per_layer'],
            use_improved_initialization=False
        )
        
        # Load weights
        try:
            pt_path = model_path.replace('.h5', '.pt') if model_path.endswith('.h5') else model_path
            # Auto-detect device for loading weights
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            state = torch.load(pt_path, map_location=device, weights_only=False)
            self.tcn_model.load_state_dict(state, strict=False)
            # Move model to the detected device
            self.tcn_model = self.tcn_model.to(device)
            # print(f"TCN weights loaded successfully from {pt_path} on {device}")
        except Exception as e:
            print(f"Error: Unable to load TCN weights: {e}")
        
        # Set model to evaluation mode and freeze parameters
        self.tcn_model.eval()
        for param in self.tcn_model.parameters():
            param.requires_grad = False
    
    def forward(self, firing_rates, fixed_exc_indices=None):
        """
        Forward pass: firing rates -> Poisson spikes -> TCN prediction
        
        Args:
            firing_rates: (batch_size, num_segments_total, time_duration_ms) torch tensor
            fixed_exc_indices: Fixed excitatory indices for adding spikes
            
        Returns:
            spike_predictions: (batch_size*2, time_duration_ms, 1) torch tensor
            spike_trains: (batch_size*2, num_segments_total, time_duration_ms) torch tensor
        """
        # Ensure input is torch tensor
        if not isinstance(firing_rates, torch.Tensor):
            firing_rates = torch.from_numpy(firing_rates).float()
        
        batch_size = firing_rates.shape[0]
        
        # Safe handling: remove NaN/Inf and ensure non-negative for Poisson
        safe_firing_rates = torch.nan_to_num(firing_rates, nan=0.0, posinf=1.0, neginf=0.0)
        safe_firing_rates = torch.clamp(safe_firing_rates, min=0.0)
        
        # Use Poisson sampling with Straight-Through Estimator for differentiability
        with torch.no_grad():
            # Poisson sampling in forward pass
            poisson_samples = torch.poisson(safe_firing_rates)
        
        # Straight-Through Estimator: use Poisson result but gradient flows through safe_firing_rates
        first_half_spikes = poisson_samples + safe_firing_rates - safe_firing_rates.detach()
        
        # Copy first half to second half, ensure base spikes are identical
        second_half_spikes = first_half_spikes.clone()
        
        # Combine complete spike trains: (batch_size*2, num_segments, time_duration)
        spike_trains = torch.cat([first_half_spikes, second_half_spikes], dim=0)
        
        # If no fixed excitatory indices specified, randomly select three
        if fixed_exc_indices is None:
            torch.manual_seed(42)  # For reproducibility
            fixed_exc_indices = torch.randperm(self.num_segments_exc)[:3].cpu().numpy()
        
        # Add fixed spikes to second half batch (at half window size)
        if batch_size > 0:
            spike_time_mono_syn = self.input_window_size // 2  # half window size time point
            if spike_time_mono_syn < self.time_duration_ms:
                # Add spikes to specified three excitatory segments at half window size
                for idx in fixed_exc_indices:
                    spike_trains[batch_size:, idx, spike_time_mono_syn] = 1.0
                    spike_trains[:batch_size:, idx, spike_time_mono_syn] = 0.0
        
        # Prepare input for TCN model
        model_input = spike_trains.permute(0, 2, 1)
        
        # TCN prediction (gradients flow through)
        spike_predictions, _ = self.tcn_model(model_input)
        
        return spike_predictions, spike_trains
    
    def predict_eval(self, firing_rates, fixed_exc_indices=None):
        """
        Evaluation prediction (without gradients, using numpy Poisson)
        
        Args:
            firing_rates: (batch_size, num_segments_total, time_duration_ms)
            fixed_exc_indices: Fixed excitatory indices for adding spikes
            
        Returns:
            spike_predictions: model predicted spike probabilities
            spike_trains: generated spike trains
        """
        # Handle input that might be torch tensor
        if isinstance(firing_rates, torch.Tensor):
            firing_rates_np = firing_rates.detach().cpu().numpy()
        else:
            firing_rates_np = firing_rates
            
        batch_size = firing_rates_np.shape[0]
        
        # Use numpy-based Poisson process for evaluation (more realistic)
        safe_firing_rates = np.nan_to_num(firing_rates_np, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        safe_firing_rates = np.where(np.isfinite(safe_firing_rates), safe_firing_rates, 0.0).astype(np.float32)
        # Only apply lower bound to ensure non-negative (required for Poisson)
        safe_firing_rates = np.clip(safe_firing_rates, 0.0, None)
        first_half_spikes = np.random.poisson(safe_firing_rates).astype(np.float32)
        first_half_spikes = np.clip(first_half_spikes, 0, 1)  # Limit to 0 or 1 (binary)
        
        # Copy first half to second half, ensure base spikes are identical
        second_half_spikes = first_half_spikes.copy()
        
        # Combine complete spike trains
        spike_trains = np.concatenate([first_half_spikes, second_half_spikes], axis=0)
        
        # If no fixed excitatory indices specified, randomly select three
        if fixed_exc_indices is None:
            mono_syn_rnd = np.random.default_rng(42)
            fixed_exc_indices = mono_syn_rnd.choice(self.num_segments_exc, size=3, replace=False)
        
        # Add fixed spikes to second half batch (at half window size)
        if batch_size > 0:
            spike_time_mono_syn = self.input_window_size // 2  # half window size time point
            if spike_time_mono_syn < self.time_duration_ms:
                # Add spikes to specified three excitatory segments at half window size
                for idx in fixed_exc_indices:
                    spike_trains[batch_size:, idx, spike_time_mono_syn] = 1.0
        
        # Convert data format to match model input: (spike_batch_size, time_duration, num_segments)
        model_input = np.transpose(spike_trains, (0, 2, 1))
        
        # Handle input window size
        input_time_steps = model_input.shape[1]
        if input_time_steps < self.input_window_size:
            padding_needed = self.input_window_size - input_time_steps
            padding = np.zeros((model_input.shape[0], padding_needed, model_input.shape[2]))
            model_input = np.concatenate([padding, model_input], axis=1)
        elif input_time_steps > self.input_window_size:
            model_input = model_input[:, -self.input_window_size:, :]
        
        # Model prediction (PyTorch) without gradients
        model_input_t = torch.from_numpy(model_input.astype(np.float32))
        with torch.no_grad():
            pred_spike_t, _ = self.tcn_model(model_input_t)
        
        return pred_spike_t, torch.from_numpy(spike_trains)

    