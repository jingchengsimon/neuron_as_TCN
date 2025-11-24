import os
import glob
import time
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE
from sklearn.metrics import explained_variance_score, roc_curve, auc
from utils.find_best_model import find_best_model
from utils.fit_CNN import parse_sim_experiment_file
from utils.visualization_utils import plot_summary_panels

# 设置matplotlib后端
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

## Helper Functions

def safe_save_fig(fig, save_path, dpi=300, bbox_inches='tight'):
    """Safely save matplotlib figure"""
    try:
        fig.canvas.draw()
        plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, 
                       facecolor='white', edgecolor='none')
    except Exception:
        try:
            plt.savefig(save_path, dpi=dpi, facecolor='white', edgecolor='none')
        except Exception:
                plt.savefig(save_path, dpi=150)

def bin2dict(bin_spikes_matrix):
    """Convert binary spike matrix to dictionary format"""
    spike_row_inds, spike_times = np.nonzero(bin_spikes_matrix)
    row_inds_spike_times_map = {}
    for row_ind, syn_time in zip(spike_row_inds, spike_times):
        if row_ind not in row_inds_spike_times_map:
            row_inds_spike_times_map[row_ind] = []
            row_inds_spike_times_map[row_ind].append(syn_time)
    return row_inds_spike_times_map


def parse_multiple_sim_experiment_files(sim_experiment_files):
    """Parse multiple simulation experiment files"""
    if not sim_experiment_files:
        raise ValueError('No test files found. Please check test_data_dir and glob pattern.')
    
    data_list = []
    for sim_experiment_file in sim_experiment_files:
        X_curr, y_spike_curr, y_soma_curr = parse_sim_experiment_file(sim_experiment_file)
        data_list.append((X_curr, y_spike_curr, y_soma_curr))
    
    if not data_list:
        raise ValueError('Failed to assemble test data. Parsed zero files.')
    
    # Merge all data
    X = np.dstack([data[0] for data in data_list])
    y_spike = np.hstack([data[1] for data in data_list])
    y_soma = np.hstack([data[2] for data in data_list])
    
    return X, y_spike, y_soma


def calc_AUC_at_desired_FP(y_test, y_test_hat, desired_fpr=0.01):
    """Calculate AUC at specified FPR"""
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_test_hat.ravel())
    linear_spaced_FPR = np.linspace(0, 1, num=20000)
    linear_spaced_TPR = np.interp(linear_spaced_FPR, fpr, tpr)
    desired_fp_ind = min(max(1, np.argmin(abs(linear_spaced_FPR - desired_fpr))), 
                        linear_spaced_TPR.shape[0] - 1)
    return linear_spaced_TPR[:desired_fp_ind].mean()


def calc_TP_at_desired_FP(y_test, y_test_hat, desired_fpr=0.0025):
    """Calculate TP at specified FPR"""
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_test_hat.ravel())
    desired_fp_ind = max(1, np.argmin(abs(fpr - desired_fpr)))
    return tpr[desired_fp_ind]


def extract_key_results(y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat, desired_FP_list=[0.0025, 0.0100]):
    """Extract key evaluation results"""
    print('Calculating key results...')
    start_time = time.time()
    
    results = {}
    
    # Calculate metrics at specified FPR
    for desired_FP in desired_FP_list:
        TP_at_FP = calc_TP_at_desired_FP(y_spikes_GT, y_spikes_hat, desired_fpr=desired_FP)
        AUC_at_FP = calc_AUC_at_desired_FP(y_spikes_GT, y_spikes_hat, desired_fpr=desired_FP)
        
        print(f'TP  at {desired_FP:.4f} FP rate = {TP_at_FP:.4f}')
        print(f'AUC at {desired_FP:.4f} FP rate = {AUC_at_FP:.4f}')
        
        results[f'TP @ {desired_FP:.4f} FP'] = TP_at_FP
        results[f'AUC @ {desired_FP:.4f} FP'] = AUC_at_FP
    
    # Calculate overall AUC
    fpr, tpr, _ = roc_curve(y_spikes_GT.ravel(), y_spikes_hat.ravel())
    AUC_score = auc(fpr, tpr)
    print(f'AUC = {AUC_score:.4f}')
    
    # Calculate soma voltage metrics
    soma_explained_var = 100.0 * explained_variance_score(y_soma_GT.ravel(), y_soma_hat.ravel())
    soma_RMSE = np.sqrt(MSE(y_soma_GT.ravel(), y_soma_hat.ravel()))
    soma_MAE = MAE(y_soma_GT.ravel(), y_soma_hat.ravel())
    
    print(f'soma explained_variance = {soma_explained_var:.2f}%')
    print(f'soma RMSE = {soma_RMSE:.3f} [mV]')
    print(f'soma MAE = {soma_MAE:.3f} [mV]')
    
    results.update({
        'AUC': AUC_score,
        'soma_explained_variance_percent': soma_explained_var,
        'soma_RMSE': soma_RMSE,
        'soma_MAE': soma_MAE
    })
    
    duration = (time.time() - start_time) / 60
    print(f'Evaluation completed, took {duration:.2f} minutes')
    
    return results


def filter_and_extract_key_results(y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat, 
                                  desired_FP_list=[0.0025, 0.0100], ignore_time_at_start_ms=500, 
                                  num_spikes_per_sim=[0, 24]):
    """Filter data and extract key results"""
    time_points_to_eval = np.arange(y_spikes_GT.shape[1]) >= ignore_time_at_start_ms
    spike_counts = y_spikes_GT.sum(axis=1)
    simulations_to_eval = np.logical_and(
        spike_counts >= num_spikes_per_sim[0],
        spike_counts <= num_spikes_per_sim[1]
    )
    
    print(f'Total simulations: {y_spikes_GT.shape[0]}')
    print(f'Simulations kept: {100 * simulations_to_eval.mean():.2f}%')
    
    # Filter data
    filtered_data = {
        'y_spikes_GT': y_spikes_GT[simulations_to_eval, :][:, time_points_to_eval],
        'y_spikes_hat': y_spikes_hat[simulations_to_eval, :][:, time_points_to_eval],
        'y_soma_GT': y_soma_GT[simulations_to_eval, :][:, time_points_to_eval],
        'y_soma_hat': y_soma_hat[simulations_to_eval, :][:, time_points_to_eval]
    }
    
    return extract_key_results(**filtered_data, desired_FP_list=desired_FP_list)

def calculate_sample_specific_fpr(y_spikes_GT_sample, y_spikes_hat_sample, threshold):
    """Calculate FPR and TPR for a single sample"""
    y_pred_binary = (y_spikes_hat_sample > threshold).astype(int)
    
    # Calculate confusion matrix
    tp = np.sum((y_spikes_GT_sample == 1) & (y_pred_binary == 1))
    fp = np.sum((y_spikes_GT_sample == 0) & (y_pred_binary == 1))
    tn = np.sum((y_spikes_GT_sample == 0) & (y_pred_binary == 0))
    fn = np.sum((y_spikes_GT_sample == 1) & (y_pred_binary == 0))
    
    # Calculate metrics
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

def find_optimal_threshold_for_sample(y_spikes_GT_sample, y_spikes_hat_sample, target_fpr=0.002):
    """Find optimal threshold for a single sample"""
    thresholds = np.linspace(0.01, 0.99, 100)
    
    best_threshold = None
    best_fpr_diff = float('inf')
    best_metrics = None
    
    for threshold in thresholds:
        metrics = calculate_sample_specific_fpr(y_spikes_GT_sample, y_spikes_hat_sample, threshold)
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

def plot_voltage_traces(y_spikes_GT, y_spikes_hat, y_voltage_GT, y_voltage_hat, 
                       global_threshold, selected_traces, output_dir):
    """Plot voltage traces"""
    num_subplots = len(selected_traces)
    fig, axes = plt.subplots(nrows=num_subplots, ncols=1, figsize=(7, 2*num_subplots), sharex=True)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.01, hspace=0.3)
    
    for k, selected_trace in enumerate(selected_traces):
        ax = axes[k] if num_subplots > 1 else axes
        
        y_train_soma_bias = -67.7
        
        # Extract data
        spike_GT = y_spikes_GT[selected_trace, :]
        spike_hat = y_spikes_hat[selected_trace, :]
        voltage_GT = y_voltage_GT[selected_trace, :]
        voltage_hat = y_voltage_hat[selected_trace, :] + y_train_soma_bias
        
        # Calculate metrics
        global_metrics = calculate_sample_specific_fpr(spike_GT, spike_hat, global_threshold)
        optimal_result = find_optimal_threshold_for_sample(spike_GT, spike_hat, target_fpr=0.002)
        optimal_metrics = optimal_result['metrics']
            
        spike_trace_pred_global = spike_hat > global_threshold
        output_spike_times_in_ms_pred_global = np.nonzero(spike_trace_pred_global)[0]
        voltage_hat[output_spike_times_in_ms_pred_global] = 0

        if 'IF_model' in output_dir:
            spike_trace_GT_global = spike_GT > 0.5
            output_spike_times_in_ms_pred_global = np.nonzero(spike_trace_GT_global)[0]
            voltage_GT[output_spike_times_in_ms_pred_global] = 0

        # Time axis
        sim_duration_ms = spike_GT.shape[0]
        time_in_sec = np.arange(sim_duration_ms) / 1000.0
            
        # Plot voltage traces
        ax.plot(time_in_sec[:1000], voltage_GT[:1000], c='c', linewidth=2, label='Ground Truth')
        ax.plot(time_in_sec[:1000], voltage_hat[:1000], c='m', linestyle=':', linewidth=1.5, label='Prediction')
        
        ax.set_ylim(-80, -60)
        
        # Set axes
        ax.set_xlim(0.02, 1000/1000) #sim_duration_ms / 1000.0)
        ax.set_ylim(-80, 5)
        ax.set_ylabel('$V_m$ (mV)', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
            
        # Add title
        title_text = (f'Sim {selected_trace}: Global FPR={global_metrics["fpr"]:.4f}, '
                        f'Optimal FPR={optimal_metrics["fpr"]:.4f}, '
                        f'FP={global_metrics["false_positives"]}, '
                        f'TP={global_metrics["true_positives"]}')
        ax.set_title(title_text, fontsize=10)
            
        # Set legend
        if k == 0:
            ax.legend(loc='upper right', fontsize=8)
            
        # Hide x-axis labels
        if k < num_subplots - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        
        # Set x-axis label
        if num_subplots > 1:
            axes[-1].set_xlabel('Time (s)', fontsize=12)
        else:
            axes.set_xlabel('Time (s)', fontsize=12)
            
    # Save figure
    save_path = f'{output_dir}/voltage_traces.pdf'
    safe_save_fig(fig, save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Voltage traces saved to: {save_path}')

def analyze_fpr_distribution(y_spikes_GT, y_spikes_hat, global_threshold, target_fpr=0.002):
    """Analyze FPR distribution"""
    print(f"\n=== FPR Analysis ===")
    print(f"Global threshold: {global_threshold:.6f}")
    print(f"Target FPR: {target_fpr:.4f}")
    print(f"Total samples: {y_spikes_GT.shape[0]}")
    
    sample_fprs = []
    sample_metrics = []
    sample_tprs = []
    
    # Calculate metrics for all samples
    for i in range(y_spikes_GT.shape[0]):
        metrics = calculate_sample_specific_fpr(y_spikes_GT[i, :], y_spikes_hat[i, :], global_threshold)
        sample_fprs.append(metrics['fpr'])
        sample_metrics.append(metrics)
        
        # Calculate TPR
        tpr_value = metrics['true_positives'] / metrics['total_spikes_gt'] if metrics['total_spikes_gt'] > 0 else 0.0
        sample_tprs.append(tpr_value)
    
    sample_fprs = np.array(sample_fprs)
    sample_tprs = np.array(sample_tprs)
    
    # Print statistics
    print(f"\nFPR Statistics:")
    print(f"  Mean: {np.mean(sample_fprs):.4f}")
    print(f"  Median: {np.median(sample_fprs):.4f}")
    print(f"  Std: {np.std(sample_fprs):.4f}")
    print(f"  Range: [{np.min(sample_fprs):.4f}, {np.max(sample_fprs):.4f}]")
    
    # Compare with target FPR
    mean_fpr = np.mean(sample_fprs)
    fpr_difference = abs(mean_fpr - target_fpr)
    print(f"\nTarget FPR: {target_fpr:.4f}")
    print(f"Actual mean FPR: {mean_fpr:.4f}")
    print(f"Relative error: {(fpr_difference/target_fpr)*100:.2f}%")
    
    # Find perfect TPR samples
    perfect_tpr_samples = np.where(abs(sample_tprs - 1.0) < 1e-6)[0].tolist()
    print(f"\nPerfect TPR samples: {len(perfect_tpr_samples)}")
    if len(perfect_tpr_samples) > 0:
        print(f"Sample IDs: {perfect_tpr_samples[:10]}")  # Show first 10
    
    return sample_fprs, sample_metrics, perfect_tpr_samples


def build_dataset_identifier(model_dir, model_size, desired_fpr):
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
    return f'{model_size}_{base_identifier}/fpr{desired_fpr}'

def setup_paths_and_files(models_dir, data_dir, model_string='NMDA', model_size='large', desired_fpr=0.002):
    """Setup paths and files"""
    # Set data directory
    test_data_dir = data_dir + f'L5PC_{model_string}_test/'
    
    # Set model directory
    if 'IF_model' in models_dir:
        model_dir = models_dir + 'depth_1_filters_1_window_80/'
        test_data_dir = data_dir + f'IF_model_test/'
    elif model_string == 'NMDA':
        model_dir = models_dir + ('depth_3_filters_256_window_400/' if model_size == 'small' 
                                 else 'depth_7_filters_256_window_400/')
    elif model_string == 'AMPA':
        model_dir = models_dir + ('depth_1_filters_128_window_400/' if model_size == 'small' 
                                 else 'depth_7_filters_256_window_400/')
    
    # Build output directory
    dataset_identifier = build_dataset_identifier(model_dir, model_size, desired_fpr)
    output_dir = f"./results/5_main_figure_replication/{dataset_identifier}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find test files
    test_files = sorted(glob.glob(test_data_dir + '*_128x6_*'))
    if not test_files:
        test_files = sorted(glob.glob(os.path.join(test_data_dir, '*.p')))

    # Find best model
    model_filename, model_metadata_filename = find_best_model(model_dir)

    print(f'Model file: {model_filename.split("/")[-1]}')
    print(f'Metadata file: {model_metadata_filename.split("/")[-1]}')
    print(f'Test files count: {len(test_files)}')
    
    return test_files, model_filename, model_metadata_filename, output_dir


def load_test_data(test_files):
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


def load_model_and_metadata(model_filename, model_metadata_filename):
    """Load model and metadata"""
    print(f'Loading model: {model_filename.split("/")[-1]}')
    start_time = time.time()
    
    # Load model
    temporal_conv_net = load_model(model_filename)
    temporal_conv_net.summary()

    # Load metadata
    model_metadata_dict = pickle.load(open(model_metadata_filename, "rb"), encoding='latin1')
    architecture_dict = model_metadata_dict['architecture_dict']
    
    # Calculate parameters
    input_window_size = temporal_conv_net.input_shape[1]
    time_window_T = (np.array(architecture_dict['filter_sizes_per_layer']) - 1).sum() + 1
    overlap_size = min(max(time_window_T + 1, min(150, input_window_size - 50)), 250)

    print(f'Overlap size: {overlap_size}')
    print(f'Time window T: {time_window_T}')
    print(f'Input shape: {temporal_conv_net.input_shape}')
    
    duration = (time.time() - start_time) / 60
    print(f'Model loading completed, took {duration:.3f} minutes')
    
    return temporal_conv_net, overlap_size, input_window_size


def evaluate_and_visualize(y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat, 
                          y_soma_test_transposed, desired_fpr, model_string, output_dir):
    """Evaluate and visualize results"""
    # Filter data
    num_spikes_per_sim = [0, 24]
    ignore_time_at_start_ms = 500
    time_points_to_eval = np.arange(y_spikes_GT.shape[1]) >= ignore_time_at_start_ms
    spike_counts = y_spikes_GT.sum(axis=1)
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
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_spikes_GT_eval.ravel(), y_spikes_hat_eval.ravel())
    desired_fp_ind = max(1, np.argmin(abs(fpr - desired_fpr)))
    desired_threshold = thresholds[desired_fp_ind]
    
    print(f'Desired FPR: {desired_fpr:.4f}')
    print(f'Actual FPR: {fpr[desired_fp_ind]:.4f}')
    print(f'Threshold: {desired_threshold:.10f}')
    
    # Analyze FPR distribution
    sample_fprs, sample_metrics, perfect_tpr_samples = analyze_fpr_distribution(
        y_spikes_GT_eval, y_spikes_hat_eval, desired_threshold, target_fpr=desired_fpr
    )
    
    # Plot summary panels
    plot_summary_panels(
        fpr, tpr, desired_fp_ind,
        y_spikes_GT_eval, y_spikes_hat_eval,
        y_soma_GT_eval, y_soma_hat_eval,
        desired_threshold, save_path=f'{output_dir}/summary_panels.pdf'
    )
    
    # Select display samples
    if model_string == 'NMDA':
        candidates = np.nonzero(np.logical_and(spike_counts >= 3, spike_counts <= 15))[0]
    elif model_string == 'AMPA':
        candidates = np.nonzero(np.logical_and(spike_counts >= 0, spike_counts <= 15))[0]
    
    # Plot voltage traces
    if len(perfect_tpr_samples) > 0:
        # Find intersection of candidates and perfect_tpr_samples
        intersection = np.intersect1d(candidates, perfect_tpr_samples)
        selected_traces = intersection[:min(5, len(intersection))] if len(intersection) > 0 else []
        
        if len(selected_traces) > 0:
            plot_voltage_traces(
                y_spikes_GT_eval, y_spikes_hat_eval,
                y_soma_original_eval, y_soma_hat_eval,
                desired_threshold, selected_traces, output_dir
            )


def predict_with_model(temporal_conv_net, X_test, y_spike_test, y_soma_test, 
                      input_window_size, overlap_size):
    """Predict using model"""
    print('Predicting using model...')
    start_time = time.time()

    y_train_soma_bias = -67.7

    # Prepare data
    X_test_for_TCN = np.transpose(X_test, axes=[2, 1, 0])
    y1_test_for_TCN = y_spike_test.T[:, :, np.newaxis]
    y2_test_for_TCN = y_soma_test.T[:, :, np.newaxis] - y_train_soma_bias
    
    # Initialize prediction results
    y1_test_for_TCN_hat = np.zeros(y1_test_for_TCN.shape)
    y2_test_for_TCN_hat = np.zeros(y2_test_for_TCN.shape)

    # Calculate number of splits
    num_test_splits = int(2 + (X_test_for_TCN.shape[1] - input_window_size) / 
                         (input_window_size - overlap_size))

    # Segment prediction
    for k in range(num_test_splits):
        start_time_ind = k * (input_window_size - overlap_size)
        end_time_ind = start_time_ind + input_window_size
        
        curr_X_test = X_test_for_TCN[:, start_time_ind:end_time_ind, :]
        
        # Pad insufficient windows
        if curr_X_test.shape[1] < input_window_size:
            padding_size = input_window_size - curr_X_test.shape[1]
            X_pad = np.zeros((curr_X_test.shape[0], padding_size, curr_X_test.shape[2]))
            curr_X_test = np.hstack((curr_X_test, X_pad))
        
        # Predict
        curr_y1, curr_y2 = temporal_conv_net.predict(curr_X_test)
        
        # Merge results
        if k == 0:
            y1_test_for_TCN_hat[:, :end_time_ind, :] = curr_y1
            y2_test_for_TCN_hat[:, :end_time_ind, :] = curr_y2
        elif k == (num_test_splits - 1):
            t0 = start_time_ind + overlap_size
            duration_to_fill = y1_test_for_TCN_hat.shape[1] - t0
            y1_test_for_TCN_hat[:, t0:, :] = curr_y1[:, overlap_size:(overlap_size + duration_to_fill), :]
            y2_test_for_TCN_hat[:, t0:, :] = curr_y2[:, overlap_size:(overlap_size + duration_to_fill), :]
        else:
            t0 = start_time_ind + overlap_size
            y1_test_for_TCN_hat[:, t0:end_time_ind, :] = curr_y1[:, overlap_size:, :]
            y2_test_for_TCN_hat[:, t0:end_time_ind, :] = curr_y2[:, overlap_size:, :]
    
    # Normalize prediction results
    s_dst, m_dst = y2_test_for_TCN.std(), y2_test_for_TCN.mean()
    s_src, m_src = y2_test_for_TCN_hat.std(), y2_test_for_TCN_hat.mean()

    y2_test_for_TCN_hat = (y2_test_for_TCN_hat - m_src) / s_src
    y2_test_for_TCN_hat = s_dst * y2_test_for_TCN_hat + m_dst

    # Convert to simple format
    y_spikes_GT = y1_test_for_TCN[:, :, 0]
    y_spikes_hat = y1_test_for_TCN_hat[:, :, 0]
    y_soma_GT = y2_test_for_TCN[:, :, 0]
    y_soma_hat = y2_test_for_TCN_hat[:, :, 0]
    
    duration = (time.time() - start_time) / 60
    print(f'Prediction completed, took {duration:.3f} minutes')
    
    return y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat

   
def main(models_dir, data_dir, model_string='NMDA', model_size='large', desired_fpr=0.002):
    """Main function"""
    # Setup paths and files
    test_files, model_filename, model_metadata_filename, output_dir = setup_paths_and_files(
        models_dir, data_dir, model_string, model_size, desired_fpr
    )
    
    # Load data
    X_test, y_spike_test, y_soma_test, y_soma_test_transposed = load_test_data(test_files)
    
    # Load model
    temporal_conv_net, overlap_size, input_window_size = load_model_and_metadata(
        model_filename, model_metadata_filename
    )
    
    # Predict
    y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat = predict_with_model(
        temporal_conv_net, X_test, y_spike_test, y_soma_test, input_window_size, overlap_size
    )
    
    # Evaluate and visualize
    evaluate_and_visualize(
        y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat,
        y_soma_test_transposed, desired_fpr, model_string, output_dir
    )

        
if __name__ == "__main__":
    models_dir = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut_SJC_funcgroup2_var2/models/NMDA_tensorflow/'
    data_dir = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut_SJC_funcgroup2_var2/data/'

    # models_dir = '/G/results/aim2_sjc/Models_TCN/IF_model_InOut/models/IF_model_tensorflow/'
    # data_dir = '/G/results/aim2_sjc/Models_TCN/IF_model_InOut/data/'
    desired_fpr = 0.002
    
    main(models_dir, data_dir, 'NMDA', 'large', desired_fpr)


