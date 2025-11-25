import os
import glob
import time
import pickle
import importlib.util
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE
from sklearn.metrics import explained_variance_score, roc_curve, auc
from utils.find_best_model import find_best_model
from utils.fit_CNN import parse_sim_experiment_file
from utils.fit_CNN_torch import parse_multiple_sim_experiment_files
from utils.visualization_utils import plot_summary_panels, plot_voltage_traces

_SPEC = importlib.util.spec_from_file_location(
    "torch_mfr", os.path.join(os.path.dirname(os.path.abspath(__file__)), "5_main_figure_replication.py")
)
_TORCH_MFR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TORCH_MFR)
BaseMainFigureReplication = _TORCH_MFR.MainFigureReplication

# 设置matplotlib后端
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

## Helper Functions
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


class MainFigureReplicationTF(BaseMainFigureReplication):
    """TensorFlow wrapper that reuses shared replication logic from the PyTorch implementation."""

    def load_model_and_metadata(self, model_filename, model_metadata_filename, base_path=""):
        print(f'Loading model: {model_filename.split("/")[-1]}')
        start_time = time.time()

        temporal_conv_net = load_model(model_filename)
        temporal_conv_net.summary()

        model_metadata_dict = pickle.load(open(model_metadata_filename, "rb"), encoding='latin1')
        architecture_dict = model_metadata_dict['architecture_dict']

        input_window_size = temporal_conv_net.input_shape[1]
        time_window_T = (np.array(architecture_dict['filter_sizes_per_layer']) - 1).sum() + 1
        overlap_size = min(max(time_window_T + 1, min(150, input_window_size - 50)), 250)

        print(f'Overlap size: {overlap_size}')
        print(f'Time window T: {time_window_T}')
        print(f'Input shape: {temporal_conv_net.input_shape}')

        duration = (time.time() - start_time) / 60
        print(f'Model loading completed, took {duration:.3f} minutes')

        return temporal_conv_net, overlap_size, input_window_size

    
    def predict_with_model(self, temporal_conv_net, X_data, y_spike_data, y_soma_data,
                           input_window_size, overlap_size):
        print('Predicting using model...')
        start_time = time.time()

        y_train_soma_bias = -67.7

        X_data_for_TCN = np.transpose(X_data, axes=[2, 1, 0])
        y1_data_for_TCN = y_spike_data.T[:, :, np.newaxis]
        y2_data_for_TCN = y_soma_data.T[:, :, np.newaxis] - y_train_soma_bias

        y1_hat = np.zeros(y1_data_for_TCN.shape)
        y2_hat = np.zeros(y2_data_for_TCN.shape)

        num_splits = int(2 + (X_data_for_TCN.shape[1] - input_window_size) /
                         (input_window_size - overlap_size))

        for k in range(num_splits):
            start_time_ind = k * (input_window_size - overlap_size)
            end_time_ind = start_time_ind + input_window_size

            curr_X = X_data_for_TCN[:, start_time_ind:end_time_ind, :]

            if curr_X.shape[1] < input_window_size:
                padding_size = input_window_size - curr_X.shape[1]
                X_pad = np.zeros((curr_X.shape[0], padding_size, curr_X.shape[2]))
                curr_X = np.hstack((curr_X, X_pad))

            curr_y1, curr_y2 = temporal_conv_net.predict(curr_X)

            if k == 0:
                y1_hat[:, :end_time_ind, :] = curr_y1
                y2_hat[:, :end_time_ind, :] = curr_y2
            elif k == (num_splits - 1):
                t0 = start_time_ind + overlap_size
                duration_to_fill = y1_hat.shape[1] - t0
                y1_hat[:, t0:, :] = curr_y1[:, overlap_size:(overlap_size + duration_to_fill), :]
                y2_hat[:, t0:, :] = curr_y2[:, overlap_size:(overlap_size + duration_to_fill), :]
            else:
                t0 = start_time_ind + overlap_size
                y1_hat[:, t0:end_time_ind, :] = curr_y1[:, overlap_size:, :]
                y2_hat[:, t0:end_time_ind, :] = curr_y2[:, overlap_size:, :]

        s_dst, m_dst = y2_data_for_TCN.std(), y2_data_for_TCN.mean()
        s_src, m_src = y2_hat.std(), y2_hat.mean()

        y2_hat = (y2_hat - m_src) / s_src
        y2_hat = s_dst * y2_hat + m_dst

        y_spikes_GT = y1_data_for_TCN[:, :, 0]
        y_spikes_hat = y1_hat[:, :, 0]
        y_soma_GT = y2_data_for_TCN[:, :, 0]
        y_soma_hat = y2_hat[:, :, 0]

        duration = (time.time() - start_time) / 60
        print(f'Prediction completed, took {duration:.3f} minutes')

        return y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat


def main(models_dir, data_dir, model_string='NMDA', model_size='large', desired_fpr=0.002):
    """Main function"""
    mfr = MainFigureReplicationTF()

    test_files, valid_files, model_filename, model_metadata_filename, output_dir = mfr.setup_paths_and_files(
        models_dir, data_dir, model_string, model_size, desired_fpr
    )

    temporal_conv_net, overlap_size, input_window_size = mfr.load_model_and_metadata(
        model_filename, model_metadata_filename, models_dir
    )

    print('\n' + '='*60)
    print('Step 1: Determining threshold from validation set')
    print('='*60)
    X_valid, y_spike_valid, y_soma_valid, y_soma_valid_transposed = mfr.load_validation_data(valid_files)
    y_spikes_valid_GT, y_spikes_valid_hat, y_soma_valid_GT, y_soma_valid_hat = mfr.predict_with_model(
        temporal_conv_net, X_valid, y_spike_valid, y_soma_valid, input_window_size, overlap_size
    )

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

    valid_roc_metrics = mfr._calculate_roc_metrics(y_spikes_valid_GT_eval, y_spikes_valid_hat_eval, desired_fpr)
    threshold_from_validation = valid_roc_metrics["threshold"]
    print(f'\nThreshold determined from validation set: {threshold_from_validation:.10f}')
    print(f'Validation set FPR at this threshold: {valid_roc_metrics["actual_fpr"]:.4f}')

    print('\n' + '='*60)
    print('Step 2: Evaluating on test set using threshold from validation')
    print('='*60)
    X_test, y_spike_test, y_soma_test, y_soma_test_transposed = mfr.load_test_data(test_files)
    y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat = mfr.predict_with_model(
        temporal_conv_net, X_test, y_spike_test, y_soma_test, input_window_size, overlap_size
    )

    mfr.evaluate_and_visualize(
        y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat,
        y_soma_test_transposed, threshold_from_validation, desired_fpr, model_string, output_dir
    )

        
if __name__ == "__main__":
    # models_dir = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut/models/NMDA_tensorflow/'
    # data_dir = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut/data/'

    models_dir = '/G/results/aim2_sjc/Models_TCN/IF_model_InOut/models/IF_model_tensorflow/'
    data_dir = '/G/results/aim2_sjc/Models_TCN/IF_model_InOut/data/'
    desired_fpr = 0.002
    
    main(models_dir, data_dir, 'NMDA', 'large', desired_fpr)


