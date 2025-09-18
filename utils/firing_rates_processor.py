import numpy as np
import torch

class FiringRatesProcessor:
    """
    Class specialized for processing firing rates data
    Contains data loading, preprocessing, and utility functions
    """
    
    def __init__(self, time_duration_ms=300, return_torch=True):
        """
        Initialize data processor
        
        Args:
            time_duration_ms: Time duration, default 300ms
            return_torch: Whether to return torch tensors, default True
        """
        # Basic configuration
        self.time_duration_ms = time_duration_ms
        self.return_torch = return_torch
        
        print(f"FiringRatesProcessor initialized successfully:")
        print(f"  Time duration: {self.time_duration_ms}ms")
        print(f"  Return type: {'torch tensors' if return_torch else 'numpy arrays'}")
    
    def _convert_to_return_type(self, data):
        """
        Convert data to correct return type based on self.return_torch setting
        
        Args:
            data: numpy array or torch tensor
            
        Returns:
            converted data
        """
        if self.return_torch:
            return data if isinstance(data, torch.Tensor) else torch.from_numpy(data)
        else:
            return data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data
    
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
            return self._convert_to_return_type(firing_rates)
            
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
        
        print(f"Prepare for optimization:")
        print(f"  Original input shape: {firing_rates_np.shape}")
        print(f"  Extracted time: {start_time_ms}-{end_time_ms}ms")
        print(f"  Extracted input shape: {extracted_rates.shape}")
        
        extracted_rates = extracted_rates.astype(np.float32)
        return self._convert_to_return_type(extracted_rates)
    
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
        return self._convert_to_return_type(firing_rates)
    