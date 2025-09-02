import os
import glob
import time
import pickle
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from keras.models import load_model
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import explained_variance_score
from sklearn.metrics import roc_curve, auc
from utils.find_best_model import find_best_model
from utils.fit_CNN import parse_sim_experiment_file
# 设置matplotlib后端，避免渲染器问题
matplotlib.use('Agg')
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

def safe_save_fig(fig, save_path, dpi=300, bbox_inches='tight'):
    """
    安全保存matplotlib图像的函数，处理渲染器问题
    """
    try:
        # 强制绘制图像
        fig.canvas.draw()
        
        # 检查渲染器是否存在
        if fig.canvas.get_renderer() is None:
            fig.canvas.draw()
        
        # 尝试使用bbox_inches='tight'保存
        if bbox_inches == 'tight':
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        else:
            plt.savefig(save_path, dpi=dpi, facecolor='white', edgecolor='none')
            
    except Exception as e:
        print(f'savefig with bbox_inches failed ({e}), trying without bbox_inches...')
        try:
            # 不使用bbox_inches重试
            fig.canvas.draw()
            plt.savefig(save_path, dpi=dpi, facecolor='white', edgecolor='none')
        except Exception as e2:
            print(f'savefig without bbox_inches failed ({e2}), trying lower dpi...')
            try:
                # 降低DPI重试
                plt.savefig(save_path, dpi=150)
            except Exception as e3:
                print(f'All save attempts failed: {e3}')
                raise e3

def bin2dict(bin_spikes_matrix):
    spike_row_inds, spike_times = np.nonzero(bin_spikes_matrix)
    row_inds_spike_times_map = {}
    for row_ind, syn_time in zip(spike_row_inds,spike_times):
        if row_ind in row_inds_spike_times_map.keys():
            row_inds_spike_times_map[row_ind].append(syn_time)
        else:
            row_inds_spike_times_map[row_ind] = [syn_time]

    return row_inds_spike_times_map


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
    ax_scatter.set_xlim(-80, -57)
    ax_scatter.set_ylim(-80, -57)

    ax_scatter.grid(False)

    # plt.tight_layout()
    fig = plt.gcf()
    
    # 使用安全保存函数
    safe_save_fig(fig, save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'summary panels saved to: {save_path}')


def calculate_sample_specific_fpr(y_spikes_GT_sample, y_spikes_hat_sample, threshold):
    """
    计算单个样本的FPR和TPR
    
    Args:
        y_spikes_GT_sample: 单个样本的真实spike (1D array)
        y_spikes_hat_sample: 单个样本的预测spike概率 (1D array)
        threshold: 阈值
    
    Returns:
        dict: 包含FPR, TPR, 假阳性数量等信息
    """
    # 二值化预测
    y_pred_binary = (y_spikes_hat_sample > threshold).astype(int)
    
    # 计算混淆矩阵元素
    true_positives = np.sum((y_spikes_GT_sample == 1) & (y_pred_binary == 1))
    false_positives = np.sum((y_spikes_GT_sample == 0) & (y_pred_binary == 1))
    true_negatives = np.sum((y_spikes_GT_sample == 0) & (y_pred_binary == 0))
    false_negatives = np.sum((y_spikes_GT_sample == 1) & (y_pred_binary == 0))
    
    # 计算FPR和TPR
    fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
    tpr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # 计算其他有用指标
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = tpr  # 等同于TPR
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
        'false_positives': false_positives,
        'true_positives': true_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives,
        'total_spikes_gt': np.sum(y_spikes_GT_sample),
        'total_spikes_pred': np.sum(y_pred_binary)
    }

def find_optimal_threshold_for_sample(y_spikes_GT_sample, y_spikes_hat_sample, target_fpr=0.002):
    """
    为单个样本找到最优阈值，使其FPR接近目标值
    
    Args:
        y_spikes_GT_sample: 单个样本的真实spike
        y_spikes_hat_sample: 单个样本的预测spike概率
        target_fpr: 目标FPR
    
    Returns:
        dict: 包含最优阈值和对应的指标
    """
    # 生成候选阈值
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

def visualization_with_sample_fpr(y_spikes_GT, y_spikes_hat, y_voltage_GT, y_voltage_hat, 
                                         global_threshold, possible_presentable_candidates, output_dir, perfect_tpr_samples):
    """
    增强的可视化函数，显示每个样本的FPR信息
    """
    num_subplots = len(perfect_tpr_samples)
    

    for fig_idx in range(1):
        selected_traces = possible_presentable_candidates[perfect_tpr_samples]
        # selected_traces = possible_presentable_candidates[fig_idx*num_subplots:(fig_idx+1)*num_subplots]
        
        fig, axes = plt.subplots(nrows=num_subplots, ncols=1, figsize=(15, 10), sharex=True)
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.01, hspace=0.3)
        
        for k, selected_trace in enumerate(selected_traces):
            ax = axes[k]
            
            # 提取单个样本数据
            spike_trace_GT = y_spikes_GT[selected_trace, :]
            spike_trace_hat = y_spikes_hat[selected_trace, :]
            voltage_trace_GT = y_voltage_GT[selected_trace, :]
            voltage_trace_hat = y_voltage_hat[selected_trace, :]
            
            # 使用全局阈值
            spike_trace_pred_global = spike_trace_hat > global_threshold
            
            # 计算该样本的FPR指标
            global_metrics = calculate_sample_specific_fpr(spike_trace_GT, spike_trace_hat, global_threshold)
            
            # 寻找该样本的最优阈值
            optimal_result = find_optimal_threshold_for_sample(spike_trace_GT, spike_trace_hat, target_fpr=0.002)
            spike_trace_pred_optimal = spike_trace_hat > optimal_result['optimal_threshold']
            optimal_metrics = optimal_result['metrics']
            
            # 获取spike时间点
            output_spike_times_in_ms_GT = np.nonzero(spike_trace_GT)[0]
            output_spike_times_in_ms_pred_global = np.nonzero(spike_trace_pred_global)[0]
            output_spike_times_in_ms_pred_optimal = np.nonzero(spike_trace_pred_optimal)[0]
            
            # 准备电压轨迹（在spike时刻设置为40mV）
            voltage_trace_GT_plot = voltage_trace_GT.copy()
            voltage_trace_hat_plot = voltage_trace_hat.copy()
            voltage_trace_hat_plot[output_spike_times_in_ms_pred_global] = 40
            
            # 时间轴
            sim_duration_ms = spike_trace_GT.shape[0]
            sim_duration_sec = int(sim_duration_ms / 1000.0)
            time_in_sec = np.arange(sim_duration_ms) / 1000.0
            
            # 绘制电压轨迹
            ax.plot(time_in_sec, voltage_trace_GT_plot, c='c', linewidth=1.5, label='Ground Truth')
            ax.plot(time_in_sec, voltage_trace_hat_plot, c='m', linestyle=':', linewidth=1.5, label='Prediction')
            
            # 标记真实spike
            if len(output_spike_times_in_ms_GT) > 0:
                ax.scatter(output_spike_times_in_ms_GT/1000.0, 
                          np.full(len(output_spike_times_in_ms_GT), 40), 
                          c='blue', marker='|', s=100, label='GT Spikes')
            
            # 标记预测spike（全局阈值）
            if len(output_spike_times_in_ms_pred_global) > 0:
                ax.scatter(output_spike_times_in_ms_pred_global/1000.0, 
                          np.full(len(output_spike_times_in_ms_pred_global), 35), 
                          c='red', marker='|', s=100, label='Pred Spikes (Global)')
            
            # 标记预测spike（最优阈值）
            if len(output_spike_times_in_ms_pred_optimal) > 0:
                ax.scatter(output_spike_times_in_ms_pred_optimal/1000.0, 
                          np.full(len(output_spike_times_in_ms_pred_optimal), 30), 
                          c='orange', marker='|', s=100, label='Pred Spikes (Optimal)')
            
            # 设置坐标轴
            ax.set_xlim(0.02, sim_duration_sec)
            ax.set_ylim(-80, 45)
            ax.set_ylabel('$V_m$ (mV)', fontsize=12)
            
            # 添加FPR信息到标题
            title_text = (f'Sim {selected_trace}: Global FPR={global_metrics["fpr"]:.4f}, '
                         f'Optimal FPR={optimal_metrics["fpr"]:.4f}, '
                         f'FP={global_metrics["false_positives"]}, '
                         f'TP={global_metrics["true_positives"]}')
            ax.set_title(title_text, fontsize=10)
            
            # 设置图例（只在第一个子图显示）
            if k == 0:
                ax.legend(loc='upper right', fontsize=8)
            
            # 隐藏除最后一个子图外的x轴标签
            if k < num_subplots - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
        
        # 设置x轴标签
        axes[-1].set_xlabel('Time (s)', fontsize=12)
        
        # 保存图像
        fig = plt.gcf()
        save_path = f'{output_dir}/%d.png' % fig_idx
        safe_save_fig(fig, save_path, dpi=300, bbox_inches='tight')
        plt.close()

def analyze_fpr_distribution(y_spikes_GT, y_spikes_hat, global_threshold, target_fpr=0.002):
    """
    分析所有样本的FPR分布，打印每个样本的详细信息
    """
    sample_fprs = []
    sample_metrics = []
    sample_tprs = []  # 新增：保存所有样本的TPR值
    
    print(f"\n=== 详细FPR分析 ===")
    print(f"全局阈值: {global_threshold:.6f}")
    print(f"目标FPR: {target_fpr:.4f}")
    print(f"样本总数: {y_spikes_GT.shape[0]}")
    print(f"每个样本时长: {y_spikes_GT.shape[1]} ms")
    print("-" * 200)
    print(f"{'ID':<3} {'FP':<2} {'FPR':<5} {'TP':<2} {'TPR':<5} | {'ID':<3} {'FP':<2} {'FPR':<5} {'TP':<2} {'TPR':<5} | {'ID':<3} {'FP':<2} {'FPR':<5} {'TP':<2} {'TPR':<5} | {'ID':<3} {'FP':<2} {'FPR':<5} {'TP':<2} {'TPR':<5}")
    print("-" * 200)
    
    for i in range(0, y_spikes_GT.shape[0], 4):
        # 计算四个样本的指标
        metrics_list = []
        for j in range(4):
            if i + j < y_spikes_GT.shape[0]:
                metrics = calculate_sample_specific_fpr(y_spikes_GT[i+j, :], y_spikes_hat[i+j, :], global_threshold)
                sample_fprs.append(metrics['fpr'])
                sample_metrics.append(metrics)
                
                # 计算并保存TPR值
                if metrics['true_positives'] == 0:
                    tpr_value = 0.0
                else:
                    tpr_value = metrics['true_positives'] / metrics['total_spikes_gt'] if metrics['total_spikes_gt'] > 0 else 0.0
                sample_tprs.append(tpr_value)
                
                metrics_list.append((i+j, metrics, tpr_value))
            else:
                metrics_list.append(None)
        
        # 打印一行四个样本
        line_parts = []
        for j in range(4):
            if metrics_list[j] is not None:
                sample_id, metrics, tpr_value = metrics_list[j]
                line_parts.append(f"{sample_id:<3} {metrics['false_positives']:<2} {metrics['fpr']:<5.3f} {metrics['true_positives']:<2} {100*tpr_value:<5.1f}%")
            else:
                line_parts.append("")
        
        print(" | ".join(line_parts))
    
    sample_fprs = np.array(sample_fprs)
    
    print("-" * 80)
    print(f"\n=== FPR统计汇总 ===")
    print(f"样本FPR统计:")
    print(f"  均值: {np.mean(sample_fprs):.4f}")
    print(f"  中位数: {np.median(sample_fprs):.4f}")
    print(f"  标准差: {np.std(sample_fprs):.4f}")
    print(f"  最小值: {np.min(sample_fprs):.4f}")
    print(f"  最大值: {np.max(sample_fprs):.4f}")
    print(f"  25%分位数: {np.percentile(sample_fprs, 25):.4f}")
    print(f"  75%分位数: {np.percentile(sample_fprs, 75):.4f}")
    
    # 与目标FPR对比
    mean_fpr = np.mean(sample_fprs)
    fpr_difference = abs(mean_fpr - target_fpr)
    print(f"\n=== 与目标FPR对比 ===")
    print(f"目标FPR: {target_fpr:.4f}")
    print(f"实际平均FPR: {mean_fpr:.4f}")
    print(f"差异: {fpr_difference:.4f}")
    print(f"相对误差: {(fpr_difference/target_fpr)*100:.2f}%")
    
    # 找出FPR异常高的样本
    high_fpr_threshold = np.percentile(sample_fprs, 90)  # 90%分位数
    high_fpr_samples = np.where(sample_fprs > high_fpr_threshold)[0]
    
    print(f"\n高FPR样本 (>{high_fpr_threshold:.4f}): {len(high_fpr_samples)}个")
    for sample_idx in high_fpr_samples[:10]:  # 显示前10个
        metrics = sample_metrics[sample_idx]
        duration_sec = y_spikes_GT.shape[1] / 1000.0
        fpr_rate = metrics['false_positives'] / duration_sec
        print(f"  样本 {sample_idx}: FPR={metrics['fpr']:.4f}, FP={metrics['false_positives']}, TP={metrics['true_positives']}, FPR率={fpr_rate:.2f}/s")
    
    # 找出TPR为100%的样本（使用已计算的TPR值）
    sample_tprs = np.array(sample_tprs)
    perfect_tpr_samples = np.where(abs(sample_tprs - 1.0) < 1e-6)[0].tolist()  # TPR = 100% (考虑浮点数精度)
    
    print(f"\n=== TPR为100%的样本 ===")
    print(f"完美TPR样本数量: {len(perfect_tpr_samples)}个")
    if len(perfect_tpr_samples) > 0:
        print(f"样本ID列表: {perfect_tpr_samples}")
        # 显示前10个完美TPR样本的详细信息
        for sample_idx in perfect_tpr_samples[:10]:
            metrics = sample_metrics[sample_idx]
            duration_sec = y_spikes_GT.shape[1] / 1000.0
            fpr_rate = metrics['false_positives'] / duration_sec
            print(f"  样本 {sample_idx}: FPR={metrics['fpr']:.4f}, FP={metrics['false_positives']}, TP={metrics['true_positives']}, GT_spikes={metrics['total_spikes_gt']}, FPR率={fpr_rate:.2f}/s")
    else:
        print("没有找到TPR为100%的样本")
    
    return sample_fprs, sample_metrics, perfect_tpr_samples


def build_dataset_identifier(model_dir, model_size, desired_false_positive_rate):
    """
    动态构建dataset identifier
    """
    # 将路径按/分割
    path_parts = model_dir.split('/')
    
    # 找到包含'InOut'的部分的索引
    inout_index = None
    for i, part in enumerate(path_parts):
        if 'InOut' in part:
            inout_index = i
            break
    
    if inout_index is not None:
        # 提取'InOut'之后到下一个'/'前的内容
        inout_part = path_parts[inout_index]
        if 'InOut' in inout_part:
            # 获取'InOut'后的部分，去掉开头的下划线
            inout_suffix = inout_part.split('InOut')[-1]
            if inout_suffix.startswith('_'):
                inout_suffix = inout_suffix[1:]
        else:
            inout_suffix = inout_part
    else:
        inout_suffix = 'original'
    
    # 找到包含模型策略的部分（通常是倒数第3个部分）
    # 路径结构：.../models/AMPA_fullStrategy/depth_7_filters_256_window_400/
    if len(path_parts) >= 3:
        model_strategy_part = path_parts[-3]  # 倒数第3个部分
    else:
        model_strategy_part = path_parts[-1]
    
    # 从模型策略部分提取下划线后的内容
    if '_' in model_strategy_part:
        strategy_part = model_strategy_part.split('_', 1)[1]  # 获取第一个下划线后的部分
    else:
        strategy_part = model_strategy_part
    
    # 组合成base identifier
    base_identifier = f"{inout_suffix}_{strategy_part}"
    
    # 组合最终的dataset identifier
    dataset_identifier = f'{base_identifier}_{model_size}_fpr{desired_false_positive_rate}'
    return dataset_identifier


def main(models_dir, data_dir, model_string='NMDA', model_size='large'):
    if model_string == 'NMDA':
        test_data_dir      = data_dir + 'L5PC_NMDA_test/'
        if model_size == 'small':
            model_dir = models_dir + 'depth_3_filters_256_window_400/' 
        elif model_size == 'large':
            model_dir = models_dir + 'depth_7_filters_256_window_400/' 
    elif model_string == 'AMPA':
        test_data_dir      = data_dir + 'L5PC_AMPA_test/'
        if model_size == 'small':
            model_dir = models_dir + 'depth_1_filters_128_window_400/' 
        elif model_size == 'large':
            model_dir = models_dir + 'depth_7_filters_256_window_400/' 

    desired_false_positive_rate = 0.002

    # 动态构建dataset identifier
    dataset_identifier = build_dataset_identifier(model_dir, model_size, desired_false_positive_rate)

    output_dir = f"./results/5_main_figure_replication/{dataset_identifier}"
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

    v_threshold = -55

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
    
    # 分析FPR分布
    sample_fprs, sample_metrics, perfect_tpr_samples = analyze_fpr_distribution(
        y_spikes_GT_to_eval, y_spikes_hat_to_eval, desired_threshold, target_fpr=desired_false_positive_rate
    )
    
    # 生成三个子图的figure（ROC/互相关/电压散点）
    plot_summary_panels(
        fpr, tpr, desired_fp_ind,
        y_spikes_GT_to_eval, y_spikes_hat_to_eval,
        y_soma_GT_to_eval + y_train_soma_bias, y_soma_hat_to_eval + y_train_soma_bias, desired_threshold,
        save_path=f'{output_dir}/summary_panels.png'
    )
    # ------------------------------------------------------------ #  
    xylabels_fontsize = 22
    num_subplots = 5
    for fig_idx in range(5):
        num_spikes_per_simulation = y1_test_for_TCN.sum(axis=1)[:,0]
        if model_string == 'NMDA':
            possible_presentable_candidates = np.nonzero(np.logical_and(num_spikes_per_simulation >= 3, num_spikes_per_simulation <= 15))[0]
        elif model_string == 'AMPA':
            possible_presentable_candidates = np.nonzero(np.logical_and(num_spikes_per_simulation >= 0, num_spikes_per_simulation <= 15))[0]
        # selected_traces = np.random.choice(possible_presentable_candidates, size=num_subplots)
        # selected_traces = possible_presentable_candidates[fig_idx*num_subplots:(fig_idx+1)*num_subplots]


    visualization_with_sample_fpr(
        y_spikes_GT_to_eval, y_spikes_hat_to_eval,
        y_soma_GT_to_eval + y_train_soma_bias, 
        y_soma_hat_to_eval + y_train_soma_bias,
        desired_threshold, possible_presentable_candidates, output_dir, perfect_tpr_samples
    )

        # fig, ax = plt.subplots(nrows=num_subplots, ncols=1, figsize=(12, 8), sharex=True)
        # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.01, hspace=0.01)

        # for k, selected_trace in enumerate(selected_traces):
            
        #     spike_trace_GT   = y1_test_for_TCN[selected_trace,:,0]
        #     spike_trace_pred = y1_test_for_TCN_hat[selected_trace,:,0] > desired_threshold
            
        #     output_spike_times_in_ms_GT   = np.nonzero(spike_trace_GT)[0]
        #     output_spike_times_in_ms_pred = np.nonzero(spike_trace_pred)[0]
            
        #     soma_voltage_trace_GT   = y_soma_test_transposed[selected_trace,:]
        #     # soma_voltage_trace_GT   = y2_test_for_TCN[selected_trace,:,0] + y_train_soma_bias
        #     soma_voltage_trace_pred = y2_test_for_TCN_hat[selected_trace,:,0] + y_train_soma_bias
            
        #     # soma_voltage_trace_GT[output_spike_times_in_ms_GT] = 40
        #     soma_voltage_trace_pred[output_spike_times_in_ms_pred] = 40
                
        #     sim_duration_ms = spike_trace_GT.shape[0]
        #     sim_duration_sec = int(sim_duration_ms / 1000.0)
        #     # zoomout_scalebar_xloc = 0.95 * sim_duration_sec
        #     time_in_sec = np.arange(sim_duration_ms) / 1000.0

        #     # ax[k].axis('off')  # 移除或注释掉这一行
        #     ax[k].spines['top'].set_visible(False)
        #     ax[k].spines['right'].set_visible(False)
        #     ax[k].spines['bottom'].set_visible(True)  # 确保底部可见
        #     ax[k].spines['left'].set_visible(True)    # 确保左侧可见
        #     if k < num_subplots - 1:  # 隐藏除最底部子图外的所有 x 轴标签
        #         plt.setp(ax[k].get_xticklabels(), visible=False)
        #     ax[k].plot(time_in_sec,soma_voltage_trace_GT,c='c')
        #     ax[k].plot(time_in_sec,soma_voltage_trace_pred,c='m',linestyle=':')
        #     ax[k].set_xlim(0.02,sim_duration_sec)
        #     ax[k].set_ylim(-80, 40)
        #     ax[k].set_ylabel('$V_m$ (mV)', fontsize=xylabels_fontsize)
        #     ax[-1].set_xlabel('Time (s)', fontsize=xylabels_fontsize)
 
        # plt.savefig(f'{output_dir}/main_figure_replication_%d_%s.png' %(fig_idx, dataset_identifier), dpi=300, bbox_inches='tight')


if __name__ == "__main__":

    # models_dir = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut/models/NMDA_fullStrategy/'
    # data_dir   = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut/data/'
    
    models_dir = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut_SJC_funcgroup2_var2_AMPA/models/AMPA_fullStrategy/'
    data_dir   = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut_SJC_funcgroup2_var2_AMPA/data/'
    
    main(models_dir, data_dir, 'AMPA', 'large')


