import os
import glob
import time
import pickle
import argparse
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

# 配置PyTorch资源限制，防止系统过载（与TF版本一致的“严格模式”）
def configure_pytorch_resources(
    max_cpu_threads=None,
    gpu_memory_fraction=None,
    gpu_memory_limit_mb=None,):
    """
    配置PyTorch使用的CPU线程数和GPU内存
    
    Args:
        max_cpu_threads: 最大使用的CPU线程数，None 表示使用自动检测（总核心数的 1/4，更保守）
        gpu_memory_fraction: GPU 内存使用比例（0.0-1.0），None 表示不限制
        gpu_memory_limit_mb: 按 MB 限制 GPU 内存（会转换为 fraction），None 表示不限制
    """
    print('\n' + '='*60)
    print('Configuring PyTorch resource limits (STRICT MODE)')
    print('='*60)
    
    # -------- CPU 线程数限制（更保守：总核心数的 1/4） --------
    if max_cpu_threads is not None:
        try:
            torch.set_num_threads(max_cpu_threads)
            torch.set_num_interop_threads(max_cpu_threads)
            print(f'CPU threads limited to: {max_cpu_threads}')
        except Exception as e:
            print(f'Warning: Could not limit CPU threads: {e}')
    else:
        import multiprocessing
        total_cores = multiprocessing.cpu_count()
        safe_threads = max(1, total_cores // 4)  # 改为 1/4，更保守
        try:
            torch.set_num_threads(safe_threads)
            torch.set_num_interop_threads(safe_threads)
            print(f'Auto-limiting CPU threads to: {safe_threads} (out of {total_cores} total cores, 1/4 for safety)')
        except Exception as e:
            print(f'Warning: Could not auto-limit CPU threads: {e}')
    
    # -------- GPU 显存限制 --------
    if torch.cuda.is_available():
        try:
            gpu_count = torch.cuda.device_count()
            print(f'GPU devices available: {gpu_count}')
            
            # 如果指定了绝对 MB 限制，则转换为 fraction
            if gpu_memory_limit_mb is not None:
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    total_mb = props.total_memory / (1024**2)
                    frac = min(1.0, max(0.01, gpu_memory_limit_mb / total_mb))
                    torch.cuda.set_per_process_memory_fraction(frac, device=i)
                    print(f'GPU {i} memory limited to ~{gpu_memory_limit_mb}MB ({frac:.2%} of {total_mb:.0f}MB)')
            elif gpu_memory_fraction is not None:
                for i in range(gpu_count):
                    torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=i)
                    print(f'GPU {i} memory fraction limited to: {gpu_memory_fraction:.2%}')
            else:
                # 默认启用缓存清理，但不强行限制显存
                torch.cuda.empty_cache()
                print('GPU memory cache cleared')
            
            # 打印 GPU 信息
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f'GPU {i}: {gpu_name} ({gpu_memory_gb:.1f} GB)')
        except Exception as e:
            print(f'GPU configuration error: {e}')
    else:
        print('No GPU devices found, using CPU only')
    
    print('='*60 + '\n')

# 在导入后立即配置（在 main 函数之前）
# 默认限制为总核心数的 1/4，可以通过环境变量覆盖
max_cpu_threads_env = os.environ.get('TORCH_MAX_CPU_THREADS')
if max_cpu_threads_env:
    max_cpu_threads = int(max_cpu_threads_env)
else:
    max_cpu_threads = None  # 将使用自动检测（总核心数的 1/4，更保守）

# GPU 内存使用比例（可选，通过环境变量设置）
gpu_memory_fraction_env = os.environ.get('TORCH_GPU_MEMORY_FRACTION')
if gpu_memory_fraction_env:
    gpu_memory_fraction = float(gpu_memory_fraction_env)
else:
    gpu_memory_fraction = None

# GPU 内存 MB 限制（可选，通过环境变量设置）
gpu_memory_limit_mb_env = os.environ.get('TORCH_GPU_MEMORY_LIMIT_MB')
if gpu_memory_limit_mb_env:
    gpu_memory_limit_mb = int(gpu_memory_limit_mb_env)
else:
    gpu_memory_limit_mb = None

configure_pytorch_resources(
    max_cpu_threads=max_cpu_threads,
    gpu_memory_fraction=gpu_memory_fraction,
    gpu_memory_limit_mb=gpu_memory_limit_mb,
)

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
                if hasattr(model, 'tcn') and hasattr(model.tcn[0], 'conv'):
                    conv = model.tcn[0].conv 
                    num_channels = conv.in_channels
                else:
                    num_channels = 1279

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
        valid_data_dir = data_dir + f'L5PC_{model_string}_valid/'

        # Set model directory
        if 'IF_model' in models_dir:
            model_dir = models_dir + 'depth_1_filters_1_window_80/'
            test_data_dir = data_dir + f'IF_model_test/'
            valid_data_dir = data_dir + f'IF_model_valid/'
        elif model_string == 'NMDA':
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
        # Find validation files
        valid_files = sorted(glob.glob(valid_data_dir + '*_128x6_*'))
        if not valid_files:
            valid_files = sorted(glob.glob(os.path.join(valid_data_dir, '*.p')))
        # Find best model
        model_filename, model_metadata_filename = find_best_model(model_dir)
        print(f'Model file: {model_filename.split("/")[-1]}')
        print(f'Metadata file: {model_metadata_filename.split("/")[-1]}')
        print(f'Test files count: {len(test_files)}')
        print(f'Validation files count: {len(valid_files)}')
        return test_files, valid_files, model_filename, model_metadata_filename, output_dir

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
    
    def load_validation_data(self, valid_files):
        """Load validation data"""
        print('Loading validation data...')
        start_time = time.time()
        v_threshold = -55
        X_valid, y_spike_valid, y_soma_valid = parse_multiple_sim_experiment_files(valid_files)
        y_soma_valid_transposed = y_soma_valid.copy().T
        y_soma_valid[y_soma_valid > v_threshold] = v_threshold
        duration = (time.time() - start_time) / 60
        print(f'Validation data loading completed, took {duration:.3f} minutes')
        return X_valid, y_spike_valid, y_soma_valid, y_soma_valid_transposed

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
                          y_soma_test_transposed, threshold, desired_fpr, model_string, output_dir):
        """Evaluate and visualize results on test set using threshold determined from validation set"""
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
        
        # Calculate ROC metrics on test set (for visualization, but threshold is from validation)
        roc_metrics = self._calculate_roc_metrics(y_spikes_GT_eval, y_spikes_hat_eval, desired_fpr)
        # Override threshold with the one determined from validation set
        roc_metrics["threshold"] = threshold
        
        # Calculate actual FPR on test set using threshold from validation
        test_fpr_at_threshold = self._calculate_fpr_at_threshold(y_spikes_GT_eval, y_spikes_hat_eval, threshold)
        roc_metrics["actual_fpr"] = test_fpr_at_threshold
        
        print(f'Threshold (from validation set): {threshold:.10f}')
        print(f'Test set FPR at this threshold: {test_fpr_at_threshold:.4f}')
        print(f'Desired FPR: {desired_fpr:.4f}')
        
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

        if 'IF_model' in model_dir:
            inout_suffix = 'IF'
        else:
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

        return f'{base_identifier}/{model_size}/fpr{desired_fpr}'

    
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

        if 'IF_model' in base_path:
            num_segments_exc = 80
            num_segments_inh = 20
        else:
            num_segments_exc = 639
            if 'SJC' in base_path:
                num_segments_inh = 640
            else:
                num_segments_inh = 639

        model = TCNModel(
            max_input_window_size=input_window_size,
            num_segments_exc=num_segments_exc,
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
    
    def _calculate_fpr_at_threshold(self, y_test, y_test_hat, threshold):
        """Calculate FPR at a specific threshold"""
        y_pred_binary = (y_test_hat > threshold).astype(int)
        y_test_flat = y_test.ravel()
        y_pred_flat = y_pred_binary.ravel()
        fp = np.sum((y_test_flat == 0) & (y_pred_flat == 1))
        tn = np.sum((y_test_flat == 0) & (y_pred_flat == 0))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return fpr

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
        test_files, valid_files, model_filename, model_metadata_filename, output_dir = mfr.setup_paths_and_files(
            models_dir, data_dir, model_string, model_size, desired_fpr
        )
        # Load model
        temporal_conv_net, overlap_size, input_window_size = mfr.load_model_and_metadata(
            model_filename, model_metadata_filename, models_dir
        )
        
        # Step 1: Load validation data and determine threshold
        print('\n' + '='*60)
        print('Step 1: Determining threshold from validation set')
        print('='*60)
        X_valid, y_spike_valid, y_soma_valid, y_soma_valid_transposed = mfr.load_validation_data(valid_files)
        y_spikes_valid_GT, y_spikes_valid_hat, y_soma_valid_GT, y_soma_valid_hat = mfr.predict_with_model(
            temporal_conv_net, X_valid, y_spike_valid, y_soma_valid, input_window_size, overlap_size
        )
        
        # Filter validation data for threshold calculation
        ignore_time_at_start_ms = 500
        time_points_to_eval_valid = np.arange(y_spikes_valid_GT.shape[1]) >= ignore_time_at_start_ms
        spike_counts_valid = y_spikes_valid_GT.sum(axis=1)
        max_spike_counts_valid = max(spike_counts_valid)
        simulations_to_eval_valid = np.logical_and(
            spike_counts_valid >= 0,
            spike_counts_valid <= max_spike_counts_valid
        )
        y_spikes_valid_GT_eval = y_spikes_valid_GT[simulations_to_eval_valid, :][:, time_points_to_eval_valid]
        y_spikes_valid_hat_eval = y_spikes_valid_hat[simulations_to_eval_valid, :][:, time_points_to_eval_valid]
        
        # Calculate threshold on validation set
        valid_roc_metrics = mfr._calculate_roc_metrics(y_spikes_valid_GT_eval, y_spikes_valid_hat_eval, desired_fpr)
        threshold_from_validation = valid_roc_metrics["threshold"]
        print(f'\nThreshold determined from validation set: {threshold_from_validation:.10f}')
        print(f'Validation set FPR at this threshold: {valid_roc_metrics["actual_fpr"]:.4f}')
        
        # Step 2: Load test data and evaluate with threshold from validation
        print('\n' + '='*60)
        print('Step 2: Evaluating on test set using threshold from validation')
        print('='*60)
        X_test, y_spike_test, y_soma_test, y_soma_test_transposed = mfr.load_test_data(test_files)
        y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat = mfr.predict_with_model(
            temporal_conv_net, X_test, y_spike_test, y_soma_test, input_window_size, overlap_size
        )
        # Evaluate and visualize on test set
        mfr.evaluate_and_visualize(
            y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat,
            y_soma_test_transposed, threshold_from_validation, desired_fpr, model_string, output_dir
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
    # ========== Parse command line arguments ==========
    parser = argparse.ArgumentParser(description='Main figure replication (PyTorch version)')
    
    # Path and model configuration arguments
    parser.add_argument('--data_suffix', type=str, default='NMDA',
                        help='Data suffix for train/valid/test directories (default: NMDA)')
    parser.add_argument('--model_suffix', type=str, default='NMDA_torch_ratio0.6',
                        help='Model suffix for model directory (default: NMDA_torch_ratio0.6)')
    parser.add_argument('--desired_fpr', type=float, default=0.002,
                        help='Desired false positive rate for threshold determination (default: 0.002)')
    
    args = parser.parse_args()
    
    # ========== Configuration: Build paths from arguments ==========
    base_path = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut/'
    models_dir = base_path + f'models/{args.model_suffix}/'
    data_dir = base_path + 'data/'
    
    print(f"\n=== Configuration ===")
    print(f"Data suffix: {args.data_suffix}")
    print(f"Model suffix: {args.model_suffix}")
    print(f"Desired FPR: {args.desired_fpr}")
    print(f"Base path: {base_path}")
    print(f"Models directory: {models_dir}")
    print(f"Data directory: {data_dir}")
    print(f"==================\n")
    
    main(models_dir, data_dir, args.data_suffix, 'large', args.desired_fpr)