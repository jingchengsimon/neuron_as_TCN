import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import glob
from skimage.transform import resize
import time
import pickle
import imageio
from scipy import signal
from keras.models import Model, load_model
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import explained_variance_score
from sklearn.metrics import roc_curve, auc
from utils.find_best_model import find_best_model

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

save_figures = False
all_file_endings_to_use = ['.png', '.pdf', '.svg']

# NOTE: during this project I've changed my coding style
# and was too lazy to edit the old code to match the new style
# so please ignore any style related wierdness
# thanks for not being petty about unimportant shit

# ALSO NOTE: prints are for logging purposes

## helper functions


def bin2dict(bin_spikes_matrix):
    spike_row_inds, spike_times = np.nonzero(bin_spikes_matrix)
    row_inds_spike_times_map = {}
    for row_ind, syn_time in zip(spike_row_inds,spike_times):
        if row_ind in row_inds_spike_times_map.keys():
            row_inds_spike_times_map[row_ind].append(syn_time)
        else:
            row_inds_spike_times_map[row_ind] = [syn_time]

    return row_inds_spike_times_map


# def dict2bin(row_inds_spike_times_map, num_segments, sim_duration_ms):
    
#     bin_spikes_matrix = np.zeros((num_segments, sim_duration_ms), dtype='bool')
#     for row_ind in row_inds_spike_times_map.keys():
#         for spike_time in row_inds_spike_times_map[row_ind]:
#             bin_spikes_matrix[row_ind,spike_time] = 1.0
    
#     return bin_spikes_matrix


# def parse_sim_experiment_file(sim_experiment_file):
    
#     print('-----------------------------------------------------------------')
#     print("loading file: '" + sim_experiment_file.split("\\")[-1] + "'")
#     loading_start_time = time.time()
#     experiment_dict = pickle.load(open(sim_experiment_file, "rb" ), encoding='latin1')
    
#     # gather params
#     num_simulations = len(experiment_dict['Results']['listOfSingleSimulationDicts'])
#     num_segments    = len(experiment_dict['Params']['allSegmentsType'])
#     sim_duration_ms = experiment_dict['Params']['totalSimDurationInSec'] * 1000
#     num_ex_synapses  = num_segments
#     num_inh_synapses = num_segments
#     num_synapses = num_ex_synapses + num_inh_synapses
    
#     # collect X, y_spike, y_soma
#     X = np.zeros((num_synapses,sim_duration_ms,num_simulations), dtype='bool')
#     y_spike = np.zeros((sim_duration_ms,num_simulations))
#     y_soma  = np.zeros((sim_duration_ms,num_simulations))
#     for k, sim_dict in enumerate(experiment_dict['Results']['listOfSingleSimulationDicts']):
#         X_ex  = dict2bin(sim_dict['exInputSpikeTimes'], num_segments, sim_duration_ms)
#         X_inh = dict2bin(sim_dict['inhInputSpikeTimes'], num_segments, sim_duration_ms)
#         X[:,:,k] = np.vstack((X_ex,X_inh))
#         spike_times = (sim_dict['outputSpikeTimes'].astype(float) - 0.5).astype(int)
#         y_spike[spike_times,k] = 1.0
#         y_soma[:,k] = sim_dict['somaVoltageLowRes']

#     loading_duration_sec = time.time() - loading_start_time
#     print('loading took %.3f seconds' %(loading_duration_sec))
#     print('-----------------------------------------------------------------')

#     return X, y_spike, y_soma


def dict2bin(row_inds_spike_times_map, num_segments, sim_duration_ms, syn_type):
    
    # 在循环开始前对字典的key进行批量操作
    # if syn_type == 'exc':
    if num_segments == 640:
        # 对兴奋性突触字典，将所有key减1（从1-639变为0-638）
        adjusted_dict = {}
        for key, value in row_inds_spike_times_map.items():
            adjusted_dict[key - 1] = value
        row_inds_spike_times_map = adjusted_dict
    # 对于inh类型，不需要修改key

    bin_spikes_matrix = np.zeros((num_segments, sim_duration_ms), dtype='bool')            
    for row_ind in row_inds_spike_times_map.keys():
        for spike_time in row_inds_spike_times_map[row_ind]:
            bin_spikes_matrix[row_ind,spike_time] = 1.0
    
    return bin_spikes_matrix


def parse_sim_experiment_file(sim_experiment_file):
    
    print('-----------------------------------------------------------------')
    print("loading file: '" + sim_experiment_file.split("\\")[-1] + "'")
    loading_start_time = time.time()
    experiment_dict = pickle.load(open(sim_experiment_file, "rb" ), encoding='latin1')
    
    # gather params（兼容original与SJC）
    num_simulations = len(experiment_dict['Results']['listOfSingleSimulationDicts'])

    # 优先用Params里的总时长；若无，则从第一条模拟的电压长度推断
    if 'totalSimDurationInSec' in experiment_dict.get('Params', {}):
        sim_duration_ms = int(experiment_dict['Params']['totalSimDurationInSec'] * 1000)
    elif 'STIM DURATION' in experiment_dict.get('Params', {}):
        sim_duration_ms = int(experiment_dict['Params']['STIM DURATION'] - 100)
    else:
        first_sim = experiment_dict['Results']['listOfSingleSimulationDicts'][0]
        sim_duration_ms = len(first_sim['somaVoltageLowRes'])
    
    # 提取段数：original可从allSegmentsType得到；SJC按ex/inh字典长度
    params = experiment_dict.get('Params', {})
    if 'allSegmentsType' in params:
        num_segments_exc = len(params['allSegmentsType'])
        num_segments_inh = len(params['allSegmentsType'])
    else:
        first_sim = experiment_dict['Results']['listOfSingleSimulationDicts'][0]
        num_segments_exc = len(first_sim['exInputSpikeTimes'])
        num_segments_inh = len(first_sim['inhInputSpikeTimes'])
    num_synapses = num_segments_exc + num_segments_inh
    
    # collect X, y_spike, y_soma
    X = np.zeros((num_synapses, sim_duration_ms, num_simulations), dtype='bool')
    y_spike = np.zeros((sim_duration_ms, num_simulations))
    y_soma  = np.zeros((sim_duration_ms, num_simulations))
    for k, sim_dict in enumerate(experiment_dict['Results']['listOfSingleSimulationDicts']):
        X_ex  = dict2bin(sim_dict['exInputSpikeTimes'],  num_segments_exc, sim_duration_ms, syn_type='exc')
        X_inh = dict2bin(sim_dict['inhInputSpikeTimes'], num_segments_inh, sim_duration_ms, syn_type='inh')
        X[:, :, k] = np.vstack((X_ex, X_inh))
        spike_times = (sim_dict['outputSpikeTimes'].astype(float) - 0.5).astype(int)
        spike_times = spike_times[(spike_times >= 0) & (spike_times < sim_duration_ms)]
        y_spike[spike_times, k] = 1.0
        y_soma[:, k] = sim_dict['somaVoltageLowRes'][:sim_duration_ms]

    loading_duration_sec = time.time() - loading_start_time
    print('loading took %.3f seconds' %(loading_duration_sec))
    print('-----------------------------------------------------------------')

    return X, y_spike, y_soma


def parse_multiple_sim_experiment_files(sim_experiment_files):
    
    if not sim_experiment_files or len(sim_experiment_files) == 0:
        raise ValueError('No test files found. Please check test_data_dir and glob pattern.')
    
    for k, sim_experiment_file in enumerate(sim_experiment_files):
        X_curr, y_spike_curr, y_soma_curr = parse_sim_experiment_file(sim_experiment_file)
        
        if k == 0:
            X       = X_curr
            y_spike = y_spike_curr
            y_soma  = y_soma_curr
        else:
            X       = np.dstack((X,X_curr))
            y_spike = np.hstack((y_spike,y_spike_curr))
            y_soma  = np.hstack((y_soma,y_soma_curr))

    # 确保已成功构建
    try:
        return X, y_spike, y_soma
    except UnboundLocalError:
        raise ValueError('Failed to assemble test data. Parsed zero files.')


def calc_AUC_at_desired_FP(y_test, y_test_hat, desired_false_positive_rate=0.01):
    fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_test_hat.ravel())

    linear_spaced_FPR = np.linspace(0,1, num=20000)
    linear_spaced_TPR = np.interp(linear_spaced_FPR, fpr, tpr)
    
    desired_fp_ind = min(max(1, np.argmin(abs(linear_spaced_FPR - desired_false_positive_rate))), linear_spaced_TPR.shape[0] - 1)
    
    return linear_spaced_TPR[:desired_fp_ind].mean()


def calc_TP_at_desired_FP(y_test, y_test_hat, desired_false_positive_rate=0.0025):
    fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_test_hat.ravel())
    
    desired_fp_ind = np.argmin(abs(fpr - desired_false_positive_rate))
    if desired_fp_ind == 0:
        desired_fp_ind = 1

    return tpr[desired_fp_ind]


def exctract_key_results(y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat, desired_FP_list=[0.0025,0.0100]):
    
    # evaluate the model and save the results
    print('----------------------------------------------------------------------------------------')
    print('calculating key results...')
    
    evaluation_start_time = time.time()
    
    # store results in the hyper param dict and return it
    evaluations_results_dict = {}
    
    for desired_FP in desired_FP_list:
        TP_at_desired_FP  = calc_TP_at_desired_FP(y_spikes_GT, y_spikes_hat, desired_false_positive_rate=desired_FP)
        AUC_at_desired_FP = calc_AUC_at_desired_FP(y_spikes_GT, y_spikes_hat, desired_false_positive_rate=desired_FP)
        print('-----------------------------------')
        print('TP  at %.4f FP rate = %.4f' %(desired_FP, TP_at_desired_FP))
        print('AUC at %.4f FP rate = %.4f' %(desired_FP, AUC_at_desired_FP))
        TP_key_string = 'TP @ %.4f FP' %(desired_FP)
        evaluations_results_dict[TP_key_string] = TP_at_desired_FP
    
        AUC_key_string = 'AUC @ %.4f FP' %(desired_FP)
        evaluations_results_dict[AUC_key_string] = AUC_at_desired_FP
    
    print('--------------------------------------------------')
    fpr, tpr, thresholds = roc_curve(y_spikes_GT.ravel(), y_spikes_hat.ravel())
    AUC_score = auc(fpr, tpr)
    print('AUC = %.4f' %(AUC_score))
    print('--------------------------------------------------')
    
    soma_explained_variance_percent = 100.0 * explained_variance_score(y_soma_GT.ravel(), y_soma_hat.ravel())
    soma_RMSE = np.sqrt(MSE(y_soma_GT.ravel(), y_soma_hat.ravel()))
    soma_MAE  = MAE(y_soma_GT.ravel(), y_soma_hat.ravel())
    
    print('--------------------------------------------------')
    print('soma explained_variance percent = %.2f%s' %(soma_explained_variance_percent, '%'))
    print('soma RMSE = %.3f [mV]' %(soma_RMSE))
    print('soma MAE = %.3f [mV]' %(soma_MAE))
    print('--------------------------------------------------')
    
    evaluations_results_dict['AUC'] = AUC_score
    evaluations_results_dict['soma_explained_variance_percent'] = soma_explained_variance_percent
    evaluations_results_dict['soma_RMSE'] = soma_RMSE
    evaluations_results_dict['soma_MAE'] = soma_MAE
    
    evaluation_duration_min = (time.time() - evaluation_start_time) / 60
    print('finished evaluation. time took to evaluate results is %.2f minutes' %(evaluation_duration_min))
    print('----------------------------------------------------------------------------------------')
    
    return evaluations_results_dict


def filter_and_exctract_key_results(y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat, desired_FP_list=[0.0025,0.0100],
                                    ignore_time_at_start_ms=500, num_spikes_per_sim=[0,24]):

    time_points_to_eval = np.arange(y_spikes_GT.shape[1]) >= ignore_time_at_start_ms
    simulations_to_eval = np.logical_and((y_spikes_GT.sum(axis=1) >= num_spikes_per_sim[0]),(y_spikes_GT.sum(axis=1) <= num_spikes_per_sim[1]))
    
    print('total amount of simualtions is %d' %(y_spikes_GT.shape[0]))
    print('percent of simulations kept = %.2f%s' %(100 * simulations_to_eval.mean(),'%'))
    
    y_spikes_GT_to_eval  = y_spikes_GT[simulations_to_eval,:][:,time_points_to_eval]
    y_spikes_hat_to_eval = y_spikes_hat[simulations_to_eval,:][:,time_points_to_eval]
    y_soma_GT_to_eval    = y_soma_GT[simulations_to_eval,:][:,time_points_to_eval]
    y_soma_hat_to_eval   = y_soma_hat[simulations_to_eval,:][:,time_points_to_eval]
    
    return exctract_key_results(y_spikes_GT_to_eval, y_spikes_hat_to_eval, y_soma_GT_to_eval, y_soma_hat_to_eval, desired_FP_list=desired_FP_list)


def plot_summary_panels(fpr, tpr, desired_fp_ind, y_spikes_GT_to_eval, y_spikes_hat_to_eval, y_soma_GT_to_eval, y_soma_hat_to_eval, desired_threshold, save_path='summary_panels.png'):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel D: ROC curve with chosen operating point
    ax_roc = axes[0]
    ax_roc.plot(fpr, tpr, 'k-', linewidth=2)
    ax_roc.set_xlabel('False alarm rate')
    ax_roc.set_ylabel('Hit rate')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.grid(False)
    # chosen point
    ax_roc.plot(fpr[desired_fp_ind], tpr[desired_fp_ind], 'o', color='red', markersize=6)
    # inset zoom near origin
    try:
        axins = inset_axes(ax_roc, width='45%', height='45%', loc='lower right')
        axins.plot(fpr, tpr, 'k-', linewidth=1.5)
        axins.plot(fpr[desired_fp_ind], tpr[desired_fp_ind], 'o', color='red', markersize=4)
        axins.set_xlim(0.0, max(0.04, float(fpr[min(len(fpr)-1, desired_fp_ind*3)])))
        axins.set_ylim(0.0, 1.0)
        # 显示坐标轴标签
        axins.set_xlabel('False alarm rate', fontsize=8)
        axins.set_ylabel('Hit rate', fontsize=8)
        # 设置刻度标签字体大小
        axins.tick_params(axis='both', which='major', labelsize=7)
    except Exception:
        pass

    # Panel E: Cross-correlation between GT and predicted spike trains
    ax_xcorr = axes[1]
    max_lag_ms = 50
    # Binarize predictions with desired threshold
    spikes_pred_bin = (y_spikes_hat_to_eval > desired_threshold).astype(np.float32)
    spikes_gt_bin = (y_spikes_GT_to_eval > 0.5).astype(np.float32)
    num_sims, T = spikes_gt_bin.shape
    mid = T - 1
    lags = np.arange(-max_lag_ms, max_lag_ms + 1)
    xcorr_sum = np.zeros(len(lags), dtype=np.float64)
    for i in range(num_sims):
        cc = np.correlate(spikes_gt_bin[i], spikes_pred_bin[i], mode='full')
        xcorr_sum += cc[mid - max_lag_ms: mid + max_lag_ms + 1]
    xcorr_avg = xcorr_sum / max(1, num_sims)
    # convert per-ms count to Hz (bins are 1 ms)
    xcorr_hz = xcorr_avg * 1000.0
    ax_xcorr.plot(lags, xcorr_hz, 'k-', linewidth=1.5)
    ax_xcorr.set_xlabel('Δt (ms)')
    ax_xcorr.set_ylabel('spike rate (Hz)')
    ax_xcorr.axvline(0, color='k', linestyle=':', alpha=0.5)
    ax_xcorr.grid(False)

    # Panel F: Scatter plot of subthreshold voltage (pred vs GT)
    ax_scatter = axes[2]
    x = y_soma_GT_to_eval.ravel()
    y = y_soma_hat_to_eval.ravel()
    # subsample for speed/clarity
    n = x.shape[0]
    if n > 50000:
        idx = np.random.choice(n, 50000, replace=False)
        x = x[idx]; y = y[idx]
    ax_scatter.scatter(x, y, s=4, c='tab:blue', alpha=0.25, edgecolors='none')
    # y=x reference
    try:
        xy_min = float(min(np.min(x), np.min(y)))
        xy_max = float(max(np.max(x), np.max(y)))
        ax_scatter.plot([xy_min, xy_max], [xy_min, xy_max], 'k--', linewidth=1)
        ax_scatter.set_xlim(xy_min, xy_max)
        ax_scatter.set_ylim(xy_min, xy_max)
    except Exception:
        pass
    ax_scatter.set_xlabel('L5PC Model (mV)')
    ax_scatter.set_ylabel('ANN (mV)')
    ax_scatter.grid(False)

    plt.tight_layout()
    fig = plt.gcf()
    try:
        # 先绘制，避免renderer为None
        fig.canvas.draw()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f'savefig tight failed ({e}), falling back to regular save')
        try:
            fig.canvas.draw()
            plt.savefig(save_path, dpi=300)
        except Exception as e2:
            print(f'regular save also failed ({e2}). Removing inset and retrying...')
            # 尝试移除所有inset artists后再次保存
            try:
                for ax in fig.axes:
                    # 删除非主轴上的子Axes（inset）
                    for child in list(ax.get_children()):
                        try:
                            from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
                            if hasattr(child, 'get_axes_locator') and isinstance(child.get_axes_locator(), InsetPosition):
                                child.remove()
                        except Exception:
                            continue
                fig.canvas.draw()
                plt.savefig(save_path, dpi=300)
            except Exception as e3:
                print(f'final save failed: {e3}')
    finally:
        plt.close(fig)
    print(f'summary panels saved to: {save_path}')


def main(models_dir, data_dir, model_string='NMDA', model_size='large'):
    # ## evel scrip params
    # model_string = 'NMDA'

    # # model_size = 'small'
    # model_size = 'large'

    # models_dir = './Models_TCN/Single_Neuron_InOut_SJC_funcgroup2_var2/models/NMDA_largeSpikeWeight/'
    # data_dir   = './Models_TCN/Single_Neuron_InOut_SJC_funcgroup2_var2/data/'

    if model_string == 'NMDA':
        # valid_data_dir     = data_dir + 'L5PC_NMDA_valid/'
        test_data_dir      = data_dir + 'L5PC_NMDA_test/'
        # output_figures_dir = '/Reseach/Single_Neuron_InOut/figures/NMDA/'
        
        if model_size == 'small':
            model_dir = models_dir + 'depth_3_filters_256_window_400/' #  '/NMDA_FCN__DxWxT_1x128x43/'
            # NN_illustration_filename = '/Models_TCN/Single_Neuron_InOut/figures/NN_Illustrations/TCN_3_layer.png'
        elif model_size == 'large':
            model_dir = models_dir + 'depth_7_filters_256_window_400/' #  '/NMDA_TCN__DxWxT_7x128x153/'
            # NN_illustration_filename = '/Models_TCN/Single_Neuron_InOut/figures/NN_Illustrations/TCN_3_layers.png'

    desired_false_positive_rate = 0.002

    # 使用字符串连接和in操作符一次性判断所有条件
    path_str = '/'.join(model_dir.split('/'))
    dataset_identifier = f'original_fpr{desired_false_positive_rate}'
    
    # 按优先级顺序检查条件
    if 'SJC_funcgroup2_var2' in path_str and 'fullStrategy' in path_str:
        dataset_identifier = f'SJC_funcgroup2_var2_fullStrategy_fpr{desired_false_positive_rate}'
    elif 'SJC_funcgroup2_var2' in path_str:
        dataset_identifier = f'SJC_funcgroup2_var2_fpr{desired_false_positive_rate}'
    elif 'largeSpikeWeight' in path_str:
        dataset_identifier = f'SJC_funcgroup2_var2_largeSpikeWeight_fpr{desired_false_positive_rate}'
    elif 'SJC_funcgroup2' in path_str:
        dataset_identifier = f'SJC_funcgroup2_fpr{desired_false_positive_rate}'
    elif 'SJC' in path_str:
        dataset_identifier = f'SJC_fpr{desired_false_positive_rate}'
    elif 'fullStrategy' in path_str:
        dataset_identifier = f'original_fullStrategy_fpr{desired_false_positive_rate}'

    output_dir = f"./results/main_figure_replication/{dataset_identifier}"
    os.makedirs(output_dir, exist_ok=True)
    
    print('-----------------------------------------------')
    print('finding data and model')
    print('-----------------------------------------------')

    # valid_files = sorted(glob.glob(valid_data_dir + '*_128x6_*'))
    test_files  = sorted(glob.glob(test_data_dir  + '*_128x6_*'))
    if len(test_files) == 0:
        test_files = sorted(glob.glob(os.path.join(test_data_dir, '*.p')))

    # model_filename = glob.glob(os.path.join(model_dir, "*.h5"))[0]
    # model_metadata_filename = glob.glob(os.path.join(model_dir, "*.pickle"))[0]

    # 使用find_best_model函数选择最佳模型
    model_filename, model_metadata_filename = find_best_model(model_dir)

    print('model found          : "%s"' %(model_filename.split('/')[-1]))
    print('model metadata found : "%s"' %(model_metadata_filename.split('/')[-1]))
    # print('number of validation files is %d' %(len(valid_files)))
    print('number of test files is %d' %(len(test_files)))
    print('-----------------------------------------------')

    ## load valid and test datasets

    print('----------------------------------------------------------------------------------------')
    print('loading testing files...')
    test_file_loading_start_time = time.time()

    v_threshold = -60

    # load test data
    X_test, y_spike_test, y_soma_test  = parse_multiple_sim_experiment_files(test_files)
    y_soma_test_transposed = y_soma_test.copy().T
    y_soma_test[y_soma_test > v_threshold] = v_threshold

    test_file_loading_duration_min = (time.time() - test_file_loading_start_time) / 60
    print('time took to load data is %.3f minutes' %(test_file_loading_duration_min))
    print('----------------------------------------------------------------------------------------')


    ## load model
    print('----------------------------------------------------------------------------------------')
    print('loading model "%s"' %(model_filename.split('/')[-1]))

    model_loading_start_time = time.time()

    temporal_conv_net = load_model(model_filename)
    temporal_conv_net.summary()

    input_window_size = temporal_conv_net.input_shape[1]

    # load metadata pickle file
    model_metadata_dict = pickle.load(open(model_metadata_filename, "rb" ), encoding='latin1')
    architecture_dict = model_metadata_dict['architecture_dict']
    time_window_T = (np.array(architecture_dict['filter_sizes_per_layer']) - 1).sum() + 1
    overlap_size = min(max(time_window_T + 1, min(150, input_window_size - 50)), 250)

    print('overlap_size = %d' %(overlap_size))
    print('time_window_T = %d' %(time_window_T))
    # print('input shape: %s' %(str(temporal_conv_net.get_input_shape_at(0))))
    print('input shape: %s' %(str(temporal_conv_net.input_shape)))

    model_loading_duration_min = (time.time() - model_loading_start_time) / 60
    print('time took to load model is %.3f minutes' %(model_loading_duration_min))
    print('----------------------------------------------------------------------------------------')

    ## create spike predictions on test set

    print('----------------------------------------------------------------------------------------')
    print('predicting using model...')

    prediction_start_time = time.time()

    y_train_soma_bias = -67.7

    X_test_for_TCN = np.transpose(X_test,axes=[2,1,0])
    y1_test_for_TCN = y_spike_test.T[:,:,np.newaxis]
    y2_test_for_TCN = y_soma_test.T[:,:,np.newaxis] - y_train_soma_bias

    y1_test_for_TCN_hat = np.zeros(y1_test_for_TCN.shape)
    y2_test_for_TCN_hat = np.zeros(y2_test_for_TCN.shape)

    num_test_splits = int(2 + (X_test_for_TCN.shape[1] - input_window_size) / (input_window_size - overlap_size))

    for k in range(num_test_splits):
        start_time_ind = k * (input_window_size - overlap_size)
        end_time_ind   = start_time_ind + input_window_size
        
        curr_X_test_for_TCN = X_test_for_TCN[:,start_time_ind:end_time_ind,:]
        
        if curr_X_test_for_TCN.shape[1] < input_window_size:
            padding_size = input_window_size - curr_X_test_for_TCN.shape[1]
            X_pad = np.zeros((curr_X_test_for_TCN.shape[0],padding_size,curr_X_test_for_TCN.shape[2]))
            curr_X_test_for_TCN = np.hstack((curr_X_test_for_TCN,X_pad))
            
        curr_y1_test_for_TCN, curr_y2_test_for_TCN = temporal_conv_net.predict(curr_X_test_for_TCN)

        if k == 0:
            y1_test_for_TCN_hat[:,:end_time_ind,:] = curr_y1_test_for_TCN
            y2_test_for_TCN_hat[:,:end_time_ind,:] = curr_y2_test_for_TCN
        elif k == (num_test_splits - 1):
            t0 = start_time_ind + overlap_size
            duration_to_fill = y1_test_for_TCN_hat.shape[1] - t0
            y1_test_for_TCN_hat[:,t0:,:] = curr_y1_test_for_TCN[:,overlap_size:(overlap_size + duration_to_fill),:]
            y2_test_for_TCN_hat[:,t0:,:] = curr_y2_test_for_TCN[:,overlap_size:(overlap_size + duration_to_fill),:]
        else:
            t0 = start_time_ind + overlap_size
            y1_test_for_TCN_hat[:,t0:end_time_ind,:] = curr_y1_test_for_TCN[:,overlap_size:,:]
            y2_test_for_TCN_hat[:,t0:end_time_ind,:] = curr_y2_test_for_TCN[:,overlap_size:,:]


    # zero score the prediction and align it with the actual test
    s_dst = y2_test_for_TCN.std()
    m_dst = y2_test_for_TCN.mean()

    s_src = y2_test_for_TCN_hat.std()
    m_src = y2_test_for_TCN_hat.mean()

    y2_test_for_TCN_hat = (y2_test_for_TCN_hat - m_src) / s_src
    y2_test_for_TCN_hat = s_dst * y2_test_for_TCN_hat + m_dst

    # convert to simple (num_simulations, num_time_points) format
    y_spikes_GT  = y1_test_for_TCN[:,:,0]
    y_spikes_hat = y1_test_for_TCN_hat[:,:,0]
    y_soma_GT    = y2_test_for_TCN[:,:,0]
    y_soma_hat   = y2_test_for_TCN_hat[:,:,0]

    # ------------------------------------------------------------ #

    num_spikes_per_sim = [0,24]
    ignore_time_at_start_ms = 500
    time_points_to_eval = np.arange(y_spikes_GT.shape[1]) >= ignore_time_at_start_ms
    simulations_to_eval = np.logical_and((y_spikes_GT.sum(axis=1) >= num_spikes_per_sim[0]),(y_spikes_GT.sum(axis=1) <= num_spikes_per_sim[1]))

    print('total amount of simualtions is %d' %(y_spikes_GT.shape[0]))
    print('percent of simulations kept = %.2f%s' %(100 * simulations_to_eval.mean(),'%'))

    y_spikes_GT_to_eval  = y_spikes_GT[simulations_to_eval,:][:,time_points_to_eval]
    y_spikes_hat_to_eval = y_spikes_hat[simulations_to_eval,:][:,time_points_to_eval]
    y_soma_GT_to_eval    = y_soma_GT[simulations_to_eval,:][:,time_points_to_eval]
    y_soma_hat_to_eval   = y_soma_hat[simulations_to_eval,:][:,time_points_to_eval]

    # ROC curve
    
    fpr, tpr, thresholds = roc_curve(y_spikes_GT_to_eval.ravel(), y_spikes_hat_to_eval.ravel()) # Shape: (num_simulations, num_time_points-ignore_time), e.g. (128, 5500)
    desired_fp_ind = np.argmin(abs(fpr - desired_false_positive_rate))
    if desired_fp_ind == 0:
        desired_fp_ind = 1

    actual_false_positive_rate = fpr[desired_fp_ind]
    print('desired_false_positive_rate = %.4f' %(desired_false_positive_rate))
    print('actual_false_positive_rate = %.4f' %(actual_false_positive_rate))

    # AUC_score = auc(fpr, tpr)

    # print('AUC = %.4f' %(AUC_score))
    # print('at %.4f FP rate, TP = %.4f' %(actual_false_positive_rate, tpr[desired_fp_ind]))

    # # cross correlation
    # half_time_window_size_ms = 50

    desired_threshold = thresholds[desired_fp_ind]
    print('desired_threshold = %.10f' %(desired_threshold))
    # 生成三个子图的figure（ROC/互相关/电压散点）
    plot_summary_panels(
        fpr, tpr, desired_fp_ind,
        y_spikes_GT_to_eval, y_spikes_hat_to_eval,
        y_soma_GT_to_eval, y_soma_hat_to_eval,
        desired_threshold,
        save_path=f'{output_dir}/summary_panels_%s.png' %(dataset_identifier)
    )
    # ------------------------------------------------------------ #

    xytick_labels_fontsize = 18
    title_fontsize = 29
    xylabels_fontsize = 22
    legend_fontsize = 18

    num_subplots = 5
    for fig_idx in range(5):
        num_spikes_per_simulation = y1_test_for_TCN.sum(axis=1)[:,0]
        possible_presentable_candidates = np.nonzero(np.logical_and(num_spikes_per_simulation >= 3, num_spikes_per_simulation <= 15))[0]
        # selected_traces = np.random.choice(possible_presentable_candidates, size=num_subplots)
        selected_traces = possible_presentable_candidates[fig_idx*num_subplots:(fig_idx+1)*num_subplots]

        fig, ax = plt.subplots(nrows=num_subplots, ncols=1, figsize=(12, 8), sharex=True)
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.01, hspace=0.01)

        for k, selected_trace in enumerate(selected_traces):
            
            spike_trace_GT   = y1_test_for_TCN[selected_trace,:,0]
            spike_trace_pred = y1_test_for_TCN_hat[selected_trace,:,0] > desired_threshold
            
            output_spike_times_in_ms_GT   = np.nonzero(spike_trace_GT)[0]
            output_spike_times_in_ms_pred = np.nonzero(spike_trace_pred)[0]
            
            soma_voltage_trace_GT   = y_soma_test_transposed[selected_trace,:]
            # soma_voltage_trace_GT   = y2_test_for_TCN[selected_trace,:,0] + y_train_soma_bias
            soma_voltage_trace_pred = y2_test_for_TCN_hat[selected_trace,:,0] + y_train_soma_bias
            
            # soma_voltage_trace_GT[output_spike_times_in_ms_GT] = 40
            soma_voltage_trace_pred[output_spike_times_in_ms_pred] = 40
                
            sim_duration_ms = spike_trace_GT.shape[0]
            sim_duration_sec = int(sim_duration_ms / 1000.0)
            # zoomout_scalebar_xloc = 0.95 * sim_duration_sec
            time_in_sec = np.arange(sim_duration_ms) / 1000.0

            # ax[k].axis('off')  # 移除或注释掉这一行
            ax[k].spines['top'].set_visible(False)
            ax[k].spines['right'].set_visible(False)
            ax[k].spines['bottom'].set_visible(True)  # 确保底部可见
            ax[k].spines['left'].set_visible(True)    # 确保左侧可见
            if k < num_subplots - 1:  # 隐藏除最底部子图外的所有 x 轴标签
                plt.setp(ax[k].get_xticklabels(), visible=False)
            ax[k].plot(time_in_sec,soma_voltage_trace_GT,c='c')
            ax[k].plot(time_in_sec,soma_voltage_trace_pred,c='m',linestyle=':')
            ax[k].set_xlim(0.02,sim_duration_sec)
            ax[k].set_ylabel('$V_m$ (mV)', fontsize=xylabels_fontsize)
            ax[-1].set_xlabel('Time (s)', fontsize=xylabels_fontsize)

            
        plt.savefig(f'{output_dir}/main_figure_replication_%d_%s.png' %(fig_idx, dataset_identifier), dpi=300, bbox_inches='tight')


if __name__ == "__main__":

    models_dir = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut_SJC_funcgroup2_var2/models/NMDA_fullStrategy/'
    data_dir   = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut_SJC_funcgroup2_var2/data/'
    
    main(models_dir, data_dir)


