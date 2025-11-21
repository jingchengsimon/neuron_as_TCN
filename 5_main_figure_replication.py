import os
import glob
import time
import pickle
import numpy as np
import matplotlib
import torch
from sklearn.metrics import roc_curve, auc
from utils.find_best_model import find_best_model
from utils.fit_CNN_torch import parse_multiple_sim_experiment_files, TCNModel
from utils.visualization_utils import plot_summary_panels, plot_voltage_traces

# Set matplotlib backend
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

class DeviceSelector:
    """Device selection utilities with safe fallbacks"""
    def get_optimal_device(self):
        """Dynamically select the best available device (GPU if available and has memory, otherwise CPU)"""
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            return torch.device('cpu')
        try:
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            print(f"GPU memory available: {gpu_memory_gb:.2f} GB")
            # If GPU has less than 2GB free memory, use CPU
            if gpu_memory_gb < 2.0:
                print("GPU memory insufficient, using CPU")
                return torch.device('cpu')
            # Try to allocate a small tensor to test if GPU is actually usable
            test_tensor = torch.zeros(1000, 1000).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print("Using GPU for computation")
            return torch.device('cuda')
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"GPU allocation failed: {e}")
            print("Falling back to CPU")
            torch.cuda.empty_cache()
            return torch.device('cpu')

    def safe_model_to_device(self, model, device):
        """Safely move model to device with fallback to CPU if GPU fails"""
        try:
            model = model.to(device)
            # Test if model can actually run on this device
            if device.type == 'cuda':
                # Get the expected number of input channels from model
                if hasattr(model, 'tcn') and hasattr(model.tcn, 'in_channels'):
                    num_channels = model.tcn.in_channels
                else:
                    # Default fallback - based on checkpoint error: expects 1279 channels
                    num_channels = 1279  # 639 + 640
                test_input = torch.randn(1, 400, num_channels).to(device)  # (batch, time, channels)
                with torch.no_grad():
                    _ = model(test_input)
                del test_input
                torch.cuda.empty_cache()
            return model, device
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"Model failed on {device}: {e}")
            if device.type == 'cuda':
                print("Falling back to CPU")
                torch.cuda.empty_cache()
                cpu_device = torch.device('cpu')
                model = model.cpu()
                return model, cpu_device
            else:
                raise e

    def safe_tensor_to_device(self, tensor, device):
        """Safely move tensor to device with fallback to CPU if GPU fails"""
        try:
            return tensor.to(device)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"Tensor allocation failed on {device}: {e}")
            if device.type == 'cuda':
                print("Falling back to CPU")
                torch.cuda.empty_cache()
                return tensor.cpu()
            else:
                raise e

class MainFigureReplication:
    """Encapsulates main-figure replication utilities"""
    def __init__(self):
        self._device_selector = DeviceSelector()

    def setup_paths_and_files(self, models_dir, data_dir, model_string='NMDA', model_size='large', desired_fpr=0.002):
        """Setup paths and files"""
        # Set data directory
        test_data_dir = data_dir + f'L5PC_{model_string}_test/'
        # Set model directory
        if model_string == 'NMDA':
            model_dir = models_dir + ('depth_3_filters_256_window_400/' if model_size == 'small' 
                                     else 'depth_7_filters_256_window_400/')
        elif model_string == 'AMPA':
            model_dir = models_dir + ('depth_1_filters_128_window_400/' if model_size == 'small' 
                                     else 'depth_7_filters_256_window_400/')
        # Build output directory
        dataset_identifier = self._build_dataset_identifier(model_dir, model_size, desired_fpr)
        output_dir = f"./results/5_main_figure_replication/{dataset_identifier}"
        os.makedirs(output_dir, exist_ok=True)
        # Find test files
        test_files = sorted(glob.glob(test_data_dir + '*_128x6_*')) #[:10]
        if not test_files:
            test_files = sorted(glob.glob(os.path.join(test_data_dir, '*.p')))
        # Find best model
        model_filename, model_metadata_filename = find_best_model(model_dir)
        print(f'Model file: {model_filename.split("/")[-1]}')
        print(f'Metadata file: {model_metadata_filename.split("/")[-1]}')
        print(f'Test files count: {len(test_files)}')
        return test_files, model_filename, model_metadata_filename, output_dir

    def load_test_data(self, test_files):
        """Load test data"""
        print('Loading test data...')
        start_time = time.time()
        v_threshold = -55
        X_test, y_spike_test, y_soma_test = parse_multiple_sim_experiment_files(test_files)
        y_soma_test_transposed = y_soma_test.copy().T
        y_soma_test[y_soma_test > v_threshold] = v_threshold
        duration = (time.time() - start_time) / 60
        print(f'Data loading completed, took {duration:.3f} minutes')
        return X_test, y_spike_test, y_soma_test, y_soma_test_transposed

    def load_model_and_metadata(self, model_filename, model_metadata_filename, base_path=""):
        """Load model and metadata"""
        print(f'Loading model: {model_filename.split("/")[-1]}')
        start_time = time.time()
        model_metadata_dict = pickle.load(open(model_metadata_filename, "rb"), encoding='latin1')
        architecture_dict = model_metadata_dict['architecture_dict']
        device = self._device_selector.get_optimal_device()
        model_state = torch.load(model_filename, map_location=device)
        print(f"Loaded model type: {type(model_state)}")
        if isinstance(model_state, dict) and 'state_dict' in model_state:
            print("Loading from checkpoint with state_dict")
            temporal_conv_net = model_state['model']
            temporal_conv_net.load_state_dict(model_state['state_dict'])
            temporal_conv_net.eval()
            temporal_conv_net, device = self._device_selector.safe_model_to_device(temporal_conv_net, device)
        elif isinstance(model_state, dict):
            print("Loading state dict directly - reconstructing model architecture")
            print("Available keys:", list(model_state.keys())[:10])
            temporal_conv_net = self._reconstruct_model_from_architecture(architecture_dict, device, base_path)
            temporal_conv_net.load_state_dict(model_state, strict=False)
            temporal_conv_net.eval()
            temporal_conv_net, device = self._device_selector.safe_model_to_device(temporal_conv_net, device)
        else:
            print("Loading complete model object")
            temporal_conv_net = model_state
            temporal_conv_net.eval()
            temporal_conv_net, device = self._device_selector.safe_model_to_device(temporal_conv_net, device)
        total_params = sum(p.numel() for p in temporal_conv_net.parameters())
        trainable_params = sum(p.numel() for p in temporal_conv_net.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params:,}')
        print(f'Trainable parameters: {trainable_params:,}')
        if 'input_shape' in model_metadata_dict:
            input_window_size = model_metadata_dict['input_shape'][1]
        else:
            input_window_size = architecture_dict.get('input_window_size', 400)
        time_window_T = (np.array(architecture_dict['filter_sizes_per_layer']) - 1).sum() + 1
        overlap_size = min(max(time_window_T + 1, min(150, input_window_size - 50)), 250)
        print(f'Overlap size: {overlap_size}')
        print(f'Time window T: {time_window_T}')
        print(f'Input window size: {input_window_size}')
        duration = (time.time() - start_time) / 60
        print(f'Model loading completed, took {duration:.3f} minutes')
        return temporal_conv_net, overlap_size, input_window_size

    def predict_with_model(self, temporal_conv_net, X_test, y_spike_test, y_soma_test, 
                           input_window_size, overlap_size):
        """Predict using model"""
        print('Predicting using model...')
        start_time = time.time()
        y_train_soma_bias = -67.7
        
        X_test_for_TCN = np.transpose(X_test, axes=[2, 1, 0])
        y1_test_for_TCN = y_spike_test.T[:, :, np.newaxis]
        y2_test_for_TCN = y_soma_test.T[:, :, np.newaxis] - y_train_soma_bias
        y1_test_for_TCN_hat = np.zeros(y1_test_for_TCN.shape)
        y2_test_for_TCN_hat = np.zeros(y2_test_for_TCN.shape)
        num_test_splits = int(2 + (X_test_for_TCN.shape[1] - input_window_size) / 
                             (input_window_size - overlap_size))
        device = next(temporal_conv_net.parameters()).device
        print(f"Using device for prediction: {device}")
        for k in range(num_test_splits):
            start_time_ind = k * (input_window_size - overlap_size)
            end_time_ind = start_time_ind + input_window_size
            curr_X_test = X_test_for_TCN[:, start_time_ind:end_time_ind, :]
            if curr_X_test.shape[1] < input_window_size:
                padding_size = input_window_size - curr_X_test.shape[1]
                X_pad = np.zeros((curr_X_test.shape[0], padding_size, curr_X_test.shape[2]))
                curr_X_test = np.hstack((curr_X_test, X_pad))
            curr_X_test_tensor = torch.FloatTensor(curr_X_test)
            curr_X_test_tensor = self._device_selector.safe_tensor_to_device(curr_X_test_tensor, device)
            try:
                with torch.no_grad():
                    outputs = temporal_conv_net(curr_X_test_tensor)
                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        curr_y1, curr_y2 = outputs
                    else:
                        curr_y1 = outputs
                        curr_y2 = torch.zeros_like(curr_y1)
                    curr_y1 = curr_y1.cpu().numpy()
                    curr_y2 = curr_y2.cpu().numpy()
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                print(f"Prediction failed on {device}: {e}")
                if device.type == 'cuda':
                    print("Falling back to CPU for prediction")
                    torch.cuda.empty_cache()
                    temporal_conv_net = temporal_conv_net.cpu()
                    device = torch.device('cpu')
                    curr_X_test_tensor = curr_X_test_tensor.cpu()
                    with torch.no_grad():
                        outputs = temporal_conv_net(curr_X_test_tensor)
                        if isinstance(outputs, tuple) and len(outputs) == 2:
                            curr_y1, curr_y2 = outputs
                        else:
                            curr_y1 = outputs
                            curr_y2 = torch.zeros_like(curr_y1)
                        curr_y1 = curr_y1.cpu().numpy()
                        curr_y2 = curr_y2.cpu().numpy()
                else:
                    raise e
            actual_time_steps = curr_y1.shape[1]
            expected_time_steps = end_time_ind - start_time_ind
            if k == 0:
                actual_end = min(start_time_ind + actual_time_steps, y1_test_for_TCN_hat.shape[1])
                y1_test_for_TCN_hat[:, start_time_ind:actual_end, :] = curr_y1[:, :actual_end-start_time_ind, :]
                y2_test_for_TCN_hat[:, start_time_ind:actual_end, :] = curr_y2[:, :actual_end-start_time_ind, :]
            elif k == (num_test_splits - 1):
                t0 = start_time_ind + overlap_size
                duration_to_fill = min(actual_time_steps - overlap_size, y1_test_for_TCN_hat.shape[1] - t0)
                if duration_to_fill > 0:
                    y1_test_for_TCN_hat[:, t0:t0+duration_to_fill, :] = curr_y1[:, overlap_size:overlap_size+duration_to_fill, :]
                    y2_test_for_TCN_hat[:, t0:t0+duration_to_fill, :] = curr_y2[:, overlap_size:overlap_size+duration_to_fill, :]
            else:
                t0 = start_time_ind + overlap_size
                actual_end = min(t0 + actual_time_steps - overlap_size, y1_test_for_TCN_hat.shape[1])
                duration = actual_end - t0
                if duration > 0:
                    y1_test_for_TCN_hat[:, t0:actual_end, :] = curr_y1[:, overlap_size:overlap_size+duration, :]
                    y2_test_for_TCN_hat[:, t0:actual_end, :] = curr_y2[:, overlap_size:overlap_size+duration, :]
        s_dst, m_dst = y2_test_for_TCN.std(), y2_test_for_TCN.mean()
        s_src, m_src = y2_test_for_TCN_hat.std(), y2_test_for_TCN_hat.mean()
        y2_test_for_TCN_hat = (y2_test_for_TCN_hat - m_src) / s_src
        y2_test_for_TCN_hat = s_dst * y2_test_for_TCN_hat + m_dst
        y_spikes_GT = y1_test_for_TCN[:, :, 0]
        y_spikes_hat = y1_test_for_TCN_hat[:, :, 0]
        y_soma_GT = y2_test_for_TCN[:, :, 0]
        y_soma_hat = y2_test_for_TCN_hat[:, :, 0]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        duration = (time.time() - start_time) / 60
        print(f'Prediction completed, took {duration:.3f} minutes')
        return y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat

    def evaluate_and_visualize(self, y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat, 
                          y_soma_test_transposed, desired_fpr, model_string, output_dir):
        """Evaluate and visualize results"""
        # Filter data
        
        ignore_time_at_start_ms = 500
        time_points_to_eval = np.arange(y_spikes_GT.shape[1]) >= ignore_time_at_start_ms
        
        spike_counts = y_spikes_GT.sum(axis=1)
        max_spike_counts = max(spike_counts)
        num_spikes_per_sim = [0, max_spike_counts]
        simulations_to_eval = np.logical_and(
            spike_counts >= num_spikes_per_sim[0],
            spike_counts <= num_spikes_per_sim[1]
        )
        
        print(f'Total simulations: {y_spikes_GT.shape[0]}')
        print(f'Simulations kept: {100 * simulations_to_eval.mean():.2f}%')
        
        # Filtered data
        y_spikes_GT_eval = y_spikes_GT[simulations_to_eval, :][:, time_points_to_eval]
        y_spikes_hat_eval = y_spikes_hat[simulations_to_eval, :][:, time_points_to_eval]
        y_soma_GT_eval = y_soma_GT[simulations_to_eval, :][:, time_points_to_eval]
        y_soma_hat_eval = y_soma_hat[simulations_to_eval, :][:, time_points_to_eval]
        y_soma_original_eval = y_soma_test_transposed[simulations_to_eval, :][:, time_points_to_eval]
        
        # Calculate ROC metrics
        roc_metrics = self._calculate_roc_metrics(y_spikes_GT_eval, y_spikes_hat_eval, desired_fpr)
        
        print(f'Desired FPR: {desired_fpr:.4f}')
        print(f'Actual FPR: {roc_metrics["actual_fpr"]:.4f}')
        print(f'Threshold: {roc_metrics["threshold"]:.10f}')
        
        # Analyze FPR distribution
        _, _, perfect_tpr_samples = self._analyze_fpr_distribution(
            y_spikes_GT_eval, y_spikes_hat_eval, roc_metrics["threshold"], target_fpr=desired_fpr
        )
        
        # Pre-compute per-sample metrics for visualization
        per_sample_global_metrics = []
        per_sample_optimal_metrics = []
        for i in range(y_spikes_GT_eval.shape[0]):
            gm = self._calculate_sample_specific_fpr(y_spikes_GT_eval[i, :], y_spikes_hat_eval[i, :], roc_metrics["threshold"]) 
            opt = self._find_optimal_threshold_for_sample(y_spikes_GT_eval[i, :], y_spikes_hat_eval[i, :], target_fpr=desired_fpr)
            per_sample_global_metrics.append(gm)
            per_sample_optimal_metrics.append(opt['metrics'])
        
        # Plot summary panels
        y_train_soma_bias = -67.7
        
        plot_summary_panels(
            roc_metrics["fpr"], roc_metrics["tpr"], roc_metrics["desired_fp_ind"],
            y_spikes_GT_eval, y_spikes_hat_eval,
            y_soma_GT_eval + y_train_soma_bias, y_soma_hat_eval + y_train_soma_bias, 
            roc_metrics["threshold"], save_path=f'{output_dir}/summary_panels.pdf'
        )
        
        # Select display samples
        if model_string == 'NMDA':
            candidates = np.nonzero(np.logical_and(spike_counts >= 0, spike_counts <= max_spike_counts))[0]
        elif model_string == 'AMPA':
            candidates = np.nonzero(np.logical_and(spike_counts >= 0, spike_counts <= 15))[0]
        
        # Plot voltage traces
        if len(perfect_tpr_samples) > 0:
            selected_traces = candidates[perfect_tpr_samples[:min(15, len(perfect_tpr_samples))]]
            
            plot_voltage_traces(
                y_spikes_GT_eval, y_spikes_hat_eval,
                y_soma_original_eval, y_soma_hat_eval,
                roc_metrics["threshold"], selected_traces, output_dir,
                per_sample_global_metrics=per_sample_global_metrics,
                per_sample_optimal_metrics=per_sample_optimal_metrics
            )

    def _build_dataset_identifier(self, model_dir, model_size, desired_fpr):
        """Build dataset identifier"""
        path_parts = model_dir.split('/')
        # Extract InOut suffix
        inout_suffix = 'original'
        for part in path_parts:
            if 'SJC' in part:
                if 'AMPA' in part:
                    inout_suffix = 'SJC_AMPA'
                else:
                    inout_suffix = 'SJC_NMDA'
                break
        # Extract strategy part
        strategy_part = path_parts[-3] if len(path_parts) >= 3 else path_parts[-1]
        if '_' in strategy_part:
            strategy_part = strategy_part.split('_', 1)[1]
        base_identifier = f"{inout_suffix}_{strategy_part}"
        return f'{model_size}_fpr{desired_fpr}_{base_identifier}'
    
    def _reconstruct_model_from_architecture(self, architecture_dict, device, base_path=""):
        """Reconstruct PyTorch model from architecture dictionary using TCNModel from fit_CNN_torch"""
        num_filters_per_layer = architecture_dict.get('num_filters_per_layer', [256] * 7)
        filter_sizes_per_layer = architecture_dict.get('filter_sizes_per_layer', [54, 24, 24, 24, 24, 24, 24])
        dilation_rates_per_layer = architecture_dict.get('dilation_rates_per_layer', [1, 1, 1, 1, 1, 1, 1])
        strides_per_layer = architecture_dict.get('strides_per_layer', [1, 1, 1, 1, 1, 1, 1])
        activation_function_per_layer = architecture_dict.get('activation_function_per_layer', ['relu'] * 7)
        l2_regularization_per_layer = architecture_dict.get('l2_regularization_per_layer', [1e-06] * 7)
        initializer_per_layer = architecture_dict.get('initializer_per_layer', [0.002] * 7)
        input_window_size = architecture_dict.get('input_window_size', 400)
        if 'SJC' in base_path:
            num_segments_inh = 640
        else:
            num_segments_inh = 639
        model = TCNModel(
            max_input_window_size=input_window_size,
            num_segments_exc=639,
            num_segments_inh=num_segments_inh,
            filter_sizes_per_layer=filter_sizes_per_layer,
            num_filters_per_layer=num_filters_per_layer,
            activation_function_per_layer=activation_function_per_layer,
            strides_per_layer=strides_per_layer,
            dilation_rates_per_layer=dilation_rates_per_layer,
            initializer_per_layer=initializer_per_layer,
            use_improved_initialization=False
        )
        model = model.to(device)
        return model

    def _calculate_roc_metrics(self, y_test, y_test_hat, desired_fpr=0.002):
        """Calculate ROC metrics at desired FPR"""
        fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_test_hat.ravel())
        desired_fp_ind = max(1, np.argmin(abs(fpr - desired_fpr)))
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'desired_fp_ind': desired_fp_ind,
            'actual_fpr': fpr[desired_fp_ind],
            'threshold': thresholds[desired_fp_ind],
            'auc': auc(fpr, tpr)
        }

    def _calculate_sample_specific_fpr(self, y_spikes_GT_sample, y_spikes_hat_sample, threshold):
        """Calculate FPR and TPR for a single sample"""
        y_pred_binary = (y_spikes_hat_sample > threshold).astype(int)
        tp = np.sum((y_spikes_GT_sample == 1) & (y_pred_binary == 1))
        fp = np.sum((y_spikes_GT_sample == 0) & (y_pred_binary == 1))
        tn = np.sum((y_spikes_GT_sample == 0) & (y_pred_binary == 0))
        fn = np.sum((y_spikes_GT_sample == 1) & (y_pred_binary == 1))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        return {
            'fpr': fpr, 'tpr': tpr, 'precision': precision,
            'false_positives': fp, 'true_positives': tp,
            'false_negatives': fn, 'true_negatives': tn,
            'total_spikes_gt': np.sum(y_spikes_GT_sample),
            'total_spikes_pred': np.sum(y_pred_binary)
        }

    def _find_optimal_threshold_for_sample(self, y_spikes_GT_sample, y_spikes_hat_sample, target_fpr=0.002):
        """Find optimal threshold for a single sample"""
        thresholds = np.linspace(0.01, 0.99, 100)
        best_threshold = None
        best_fpr_diff = float('inf')
        best_metrics = None
        for threshold in thresholds:
            metrics = self._calculate_sample_specific_fpr(y_spikes_GT_sample, y_spikes_hat_sample, threshold)
            fpr_diff = abs(metrics['fpr'] - target_fpr)
            if fpr_diff < best_fpr_diff:
                best_fpr_diff = fpr_diff
                best_threshold = threshold
                best_metrics = metrics
        return {
            'optimal_threshold': best_threshold,
            'metrics': best_metrics,
            'fpr_difference': best_fpr_diff
        }

    def _analyze_fpr_distribution(self, y_spikes_GT, y_spikes_hat, global_threshold, target_fpr=0.002):
        """Analyze FPR distribution"""
        print(f"\n=== FPR Analysis ===")
        print(f"Global threshold: {global_threshold:.6f}")
        print(f"Target FPR: {target_fpr:.4f}")
        print(f"Total samples: {y_spikes_GT.shape[0]}")
        sample_fprs = []
        sample_metrics = []
        sample_tprs = []
        for i in range(y_spikes_GT.shape[0]):
            metrics = self._calculate_sample_specific_fpr(y_spikes_GT[i, :], y_spikes_hat[i, :], global_threshold)
            sample_fprs.append(metrics['fpr'])
            sample_metrics.append(metrics)
            tpr_value = metrics['true_positives'] / metrics['total_spikes_gt'] if metrics['total_spikes_gt'] > 0 else 0.0
            sample_tprs.append(tpr_value)
        sample_fprs = np.array(sample_fprs)
        sample_tprs = np.array(sample_tprs)
        print(f"\nFPR Statistics:")
        print(f"  Mean: {np.mean(sample_fprs):.4f}")
        print(f"  Median: {np.median(sample_fprs):.4f}")
        print(f"  Std: {np.std(sample_fprs):.4f}")
        print(f"  Range: [{np.min(sample_fprs):.4f}, {np.max(sample_fprs):.4f}]")
        mean_fpr = np.mean(sample_fprs)
        fpr_difference = abs(mean_fpr - target_fpr)
        print(f"\nTarget FPR: {target_fpr:.4f}")
        print(f"Actual mean FPR: {mean_fpr:.4f}")
        print(f"Relative error: {(fpr_difference/target_fpr)*100:.2f}%")
        desired_tpr = 1.0
        perfect_tpr_samples = np.where(abs(sample_tprs - desired_tpr) < 1e-6)[0].tolist()
        print(f"\nPerfect TPR samples: {len(perfect_tpr_samples)}")
        if len(perfect_tpr_samples) > 0:
            print(f"Sample IDs: {perfect_tpr_samples[:10]}")
        return sample_fprs, sample_metrics, perfect_tpr_samples
  
def main(models_dir, data_dir, model_string='NMDA', model_size='large', desired_fpr=0.002):
    """Main function"""
    try:
        mfr = MainFigureReplication()
        # Setup paths and files
        test_files, model_filename, model_metadata_filename, output_dir = mfr.setup_paths_and_files(
            models_dir, data_dir, model_string, model_size, desired_fpr
        )
        # Load data
        X_test, y_spike_test, y_soma_test, y_soma_test_transposed = mfr.load_test_data(test_files)
        # Load model
        temporal_conv_net, overlap_size, input_window_size = mfr.load_model_and_metadata(
            model_filename, model_metadata_filename, models_dir
        )
        # Predict
        y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat = mfr.predict_with_model(
            temporal_conv_net, X_test, y_spike_test, y_soma_test, input_window_size, overlap_size
        )
        # Evaluate and visualize
        mfr.evaluate_and_visualize(
            y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat,
            y_soma_test_transposed, desired_fpr, model_string, output_dir
        )
    except Exception as e:
        print(f"Error in main function: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        
if __name__ == "__main__":
    data_suffix = 'NMDA'
    base_path = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut_SJC_funcgroup2_var2/'
    models_dir = base_path + f'models/{data_suffix}_torch_2/'
    data_dir = base_path + 'data/'
    desired_fpr = 0.002
    
    main(models_dir, data_dir, data_suffix, 'large', desired_fpr)