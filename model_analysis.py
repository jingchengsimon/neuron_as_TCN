import pickle
import glob
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import os
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.models import load_model

def load_test_data(test_file):
    """
    加载测试数据 - 修复版本
    """
    try:
        # 使用与训练数据相同的解析函数
        from fit_CNN import parse_sim_experiment_file
        
        # 解析测试文件
        X_test, y_spike_test, y_soma_test = parse_sim_experiment_file(test_file)
        
        # 调整数据格式以匹配模型输入
        # 模型期望的输入格式：(batch_size, window_size, num_segments)
        # 但parse_sim_experiment_file返回：(num_segments, time_steps, num_simulations)
        
        # 转置数据以匹配模型输入格式
        X_test = np.transpose(X_test, axes=[2, 1, 0])  # (num_simulations, time_steps, num_segments)
        y_spike_test = y_spike_test.T[:, :, np.newaxis]  # (num_simulations, time_steps, 1)
        y_soma_test = y_soma_test.T[:, :, np.newaxis]   # (num_simulations, time_steps, 1)
        
        return X_test, y_spike_test, y_soma_test
        
    except Exception as e:
        print(f"Error loading test data from {test_file}: {e}")
        return None, None, None

def calculate_auc_metrics(model_path, test_data_dir):
    """
    计算模型的AUC指标 - 修复版本
    """
    try:
        # 清理GPU内存
        # tf.keras.backend.clear_session()
        # gc.collect()

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 强制使用CPU

        # 加载模型
        model = load_model(model_path.replace('.pickle', '.h5'))
        
        # 加载测试数据
        test_files = glob.glob(os.path.join(test_data_dir, '*.p'))
        if not test_files:
            print(f"No test files found in {test_data_dir}")
            return None
        
        # 使用第一个测试文件进行评估
        test_file = test_files[0]
        print(f"Using test file: {test_file}")
        
        # 加载测试数据
        X_test, y_spike_test, y_soma_test = load_test_data(test_file)
        
        if X_test is None:
            print("Failed to load test data")
            return None
        
        print(f"Test data shape: X={X_test.shape}, y_spike={y_spike_test.shape}, y_soma={y_soma_test.shape}")
        
        # 为了计算AUC，我们需要将数据分成小批次
        # 因为模型期望的输入是 (batch_size, window_size, num_segments)
        # 但我们的数据是 (num_simulations, time_steps, num_segments)
        
        # 获取模型的输入窗口大小
        input_window_size = model.input_shape[1]  # 假设输入是 (None, window_size, features)
        num_segments = model.input_shape[2]
        
        print(f"Model expects input shape: {model.input_shape}")
        print(f"Input window size: {input_window_size}, num_segments: {num_segments}")
        
        # 创建滑动窗口来生成测试样本
        batch_size = 32  # 可以根据内存调整
        all_predictions_spike = []
        all_predictions_soma = []
        all_targets_spike = []
        all_targets_soma = []
        
        # 对每个模拟进行预测
        for sim_idx in range(min(10, X_test.shape[0])):  # 限制模拟数量以节省时间
            sim_data = X_test[sim_idx]  # (time_steps, num_segments)
            sim_spike = y_spike_test[sim_idx]  # (time_steps, 1)
            sim_soma = y_soma_test[sim_idx]   # (time_steps, 1)
            
            # 创建滑动窗口
            for start_idx in range(0, sim_data.shape[0] - input_window_size + 1, input_window_size // 2):
                end_idx = start_idx + input_window_size
                
                # 提取窗口数据
                window_data = sim_data[start_idx:end_idx]  # (window_size, num_segments)
                window_spike = sim_spike[start_idx:end_idx]  # (window_size, 1)
                window_soma = sim_soma[start_idx:end_idx]   # (window_size, 1)
                
                # 添加batch维度
                window_data = window_data[np.newaxis, :, :]  # (1, window_size, num_segments)
                
                # 预测
                pred_spike, pred_soma = model.predict(window_data, verbose=0)
                
                # 收集结果
                all_predictions_spike.append(pred_spike.flatten())
                all_predictions_soma.append(pred_soma.flatten())
                all_targets_spike.append(window_spike.flatten())
                all_targets_soma.append(window_soma.flatten())
        
        # 合并所有预测结果
        if not all_predictions_spike:
            print("No predictions generated")
            return None
            
        y_pred_spike_flat = np.concatenate(all_predictions_spike)
        y_pred_soma_flat = np.concatenate(all_predictions_soma)
        y_spike_flat = np.concatenate(all_targets_spike)
        y_soma_flat = np.concatenate(all_targets_soma)
        
        print(f"Final shapes: pred_spike={y_pred_spike_flat.shape}, target_spike={y_spike_flat.shape}")
        
        # 计算AUC指标
        auc_metrics = {}
        
        # 1. ROC AUC for spike prediction
        try:
            auc_metrics['roc_auc_spike'] = roc_auc_score(y_spike_flat, y_pred_spike_flat)
        except Exception as e:
            print(f"Error calculating ROC AUC: {e}")
            auc_metrics['roc_auc_spike'] = 0.5
        
        # 2. Precision-Recall AUC for spike prediction
        try:
            auc_metrics['pr_auc_spike'] = average_precision_score(y_spike_flat, y_pred_spike_flat)
        except Exception as e:
            print(f"Error calculating PR AUC: {e}")
            auc_metrics['pr_auc_spike'] = 0.0
        
        # 3. 计算不同阈值下的AUC
        thresholds = np.arange(0.1, 1.0, 0.1)
        auc_at_thresholds = []
        
        for threshold in thresholds:
            try:
                y_pred_binary = (y_pred_spike_flat > threshold).astype(int)
                if len(np.unique(y_pred_binary)) > 1:  # 确保有正负样本
                    auc_at_thresholds.append(roc_auc_score(y_spike_flat, y_pred_binary))
                else:
                    auc_at_thresholds.append(0.5)
            except:
                auc_at_thresholds.append(0.5)
        
        auc_metrics['auc_at_thresholds'] = auc_at_thresholds
        auc_metrics['mean_auc_thresholds'] = np.mean(auc_at_thresholds)
        
        # 4. 计算somatic voltage的相关系数
        try:
            auc_metrics['soma_correlation'] = np.corrcoef(y_soma_flat, y_pred_soma_flat)[0, 1]
            if np.isnan(auc_metrics['soma_correlation']):
                auc_metrics['soma_correlation'] = 0.0
        except Exception as e:
            print(f"Error calculating somatic correlation: {e}")
            auc_metrics['soma_correlation'] = 0.0
        
        print(f"AUC metrics calculated: {auc_metrics}")
        return auc_metrics
        
    except Exception as e:
        print(f"Error calculating AUC for {model_path}: {e}")
        return None

def load_model_results(models_dir, test_data_dir):
    """
    加载所有保存的模型结果，包括AUC评估
    """
    model_pickles = glob.glob(os.path.join(models_dir, '*.pickle'))
    results = []
    
    for pkl in model_pickles:
        try:
            with open(pkl, 'rb') as f:
                data = pickle.load(f)
            
            # 提取基本信息
            training_history = data['training_history_dict']
            
            # 计算关键指标
            min_val_loss = min(training_history['val_loss'])
            min_val_spikes_loss = min(training_history['val_spikes_loss'])
            min_val_somatic_loss = min(training_history['val_somatic_loss'])
            
            # 找到最小loss对应的epoch
            min_val_loss_epoch = np.argmin(training_history['val_loss'])
            min_val_spikes_loss_epoch = np.argmin(training_history['val_spikes_loss'])
            
            # 提取模型文件名（不含路径）
            model_name = os.path.basename(pkl)
            
            # 计算AUC指标
            auc_metrics = calculate_auc_metrics(pkl, test_data_dir)
            
            results.append({
                'model_name': model_name,
                'model_path': pkl,
                'min_val_loss': min_val_loss,
                'min_val_spikes_loss': min_val_spikes_loss,
                'min_val_somatic_loss': min_val_somatic_loss,
                'min_val_loss_epoch': min_val_loss_epoch,
                'min_val_spikes_loss_epoch': min_val_spikes_loss_epoch,
                'training_history': training_history,
                'architecture_dict': data['architecture_dict'],
                'learning_schedule_dict': data['learning_schedule_dict'],
                'auc_metrics': auc_metrics
            })
            
        except Exception as e:
            print(f"Error loading {pkl}: {e}")
    
    return results

def plot_training_curves(results, save_dir):
    """
    绘制训练曲线
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    for i, result in enumerate(results):
        history = result['training_history']
        model_name = result['model_name'].replace('.pickle', '')
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Curves - {model_name}', fontsize=16)
        
        # 1. 总Loss
        axes[0, 0].plot(history['loss'], label='Training Loss', alpha=0.7)
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', alpha=0.7)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Spike Loss
        axes[0, 1].plot(history['spikes_loss'], label='Training Spike Loss', alpha=0.7)
        axes[0, 1].plot(history['val_spikes_loss'], label='Validation Spike Loss', alpha=0.7)
        axes[0, 1].set_title('Spike Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Somatic Loss
        axes[1, 0].plot(history['somatic_loss'], label='Training Somatic Loss', alpha=0.7)
        axes[1, 0].plot(history['val_somatic_loss'], label='Validation Somatic Loss', alpha=0.7)
        axes[1, 0].set_title('Somatic Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Learning Rate
        if 'learning_rate' in history:
            axes[1, 1].plot(history['learning_rate'], label='Learning Rate', color='red')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'training_curves_{i}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_model_comparison(results, save_dir):
    """
    绘制模型比较图
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建DataFrame用于分析
    df = pd.DataFrame(results)
    
    # 1. 模型性能比较柱状图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    # 按验证集总loss排序
    df_sorted = df.sort_values('min_val_loss')
    
    # 总Loss比较
    axes[0, 0].bar(range(len(df_sorted)), df_sorted['min_val_loss'])
    axes[0, 0].set_title('Minimum Validation Total Loss')
    axes[0, 0].set_xlabel('Model Index')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_xticks(range(len(df_sorted)))
    axes[0, 0].set_xticklabels([f'M{i}' for i in range(len(df_sorted))], rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Spike Loss比较
    axes[0, 1].bar(range(len(df_sorted)), df_sorted['min_val_spikes_loss'])
    axes[0, 1].set_title('Minimum Validation Spike Loss')
    axes[0, 1].set_xlabel('Model Index')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_xticks(range(len(df_sorted)))
    axes[0, 1].set_xticklabels([f'M{i}' for i in range(len(df_sorted))], rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Somatic Loss比较
    axes[1, 0].bar(range(len(df_sorted)), df_sorted['min_val_somatic_loss'])
    axes[1, 0].set_title('Minimum Validation Somatic Loss')
    axes[1, 0].set_xlabel('Model Index')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_xticks(range(len(df_sorted)))
    axes[1, 0].set_xticklabels([f'M{i}' for i in range(len(df_sorted))], rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 最佳epoch比较
    axes[1, 1].bar(range(len(df_sorted)), df_sorted['min_val_loss_epoch'])
    axes[1, 1].set_title('Best Epoch (Min Val Loss)')
    axes[1, 1].set_xlabel('Model Index')
    axes[1, 1].set_ylabel('Epoch')
    axes[1, 1].set_xticks(range(len(df_sorted)))
    axes[1, 1].set_xticklabels([f'M{i}' for i in range(len(df_sorted))], rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 散点图：Spike Loss vs Somatic Loss
    plt.figure(figsize=(10, 8))
    plt.scatter(df['min_val_spikes_loss'], df['min_val_somatic_loss'], alpha=0.7, s=100)
    plt.xlabel('Minimum Validation Spike Loss')
    plt.ylabel('Minimum Validation Somatic Loss')
    plt.title('Spike Loss vs Somatic Loss')
    plt.grid(True, alpha=0.3)
    
    # 添加模型标签
    for i, row in df.iterrows():
        plt.annotate(f'M{i}', (row['min_val_spikes_loss'], row['min_val_somatic_loss']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.savefig(os.path.join(save_dir, 'spike_vs_somatic_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_training_stability(results, save_dir):
    """
    分析训练稳定性
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Stability Analysis', fontsize=16)
    
    for i, result in enumerate(results):
        history = result['training_history']
        
        # 计算训练稳定性指标
        train_loss_std = np.std(history['loss'][-20:])  # 最后20个epoch的标准差
        val_loss_std = np.std(history['val_loss'][-20:])
        
        # 1. 训练loss的波动
        axes[0, 0].plot(history['loss'][-50:], alpha=0.7, label=f'Model {i}')
        axes[0, 0].set_title('Training Loss (Last 50 Epochs)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 验证loss的波动
        axes[0, 1].plot(history['val_loss'][-50:], alpha=0.7, label=f'Model {i}')
        axes[0, 1].set_title('Validation Loss (Last 50 Epochs)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 过拟合分析
        train_val_diff = np.array(history['loss']) - np.array(history['val_loss'])
        axes[1, 0].plot(train_val_diff[-50:], alpha=0.7, label=f'Model {i}')
        axes[1, 0].set_title('Train-Val Loss Difference (Last 50 Epochs)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss Difference')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 4. 收敛性分析
        axes[1, 1].plot(history['val_loss'], alpha=0.7, label=f'Model {i}')
        axes[1, 1].set_title('Full Validation Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 添加图例
    for ax in axes.flat:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_stability.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_auc_analysis(results, save_dir):
    """
    绘制AUC分析图
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 过滤掉没有AUC指标的模型
    results_with_auc = [r for r in results if r['auc_metrics'] is not None]
    
    if not results_with_auc:
        print("No models with AUC metrics found!")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(results_with_auc)
    
    # 1. AUC比较图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('AUC Analysis', fontsize=16)
    
    # 按ROC AUC排序
    df_sorted = df.sort_values('auc_metrics', key=lambda x: x.apply(lambda y: y['roc_auc_spike']))
    
    # ROC AUC比较
    roc_aucs = [r['auc_metrics']['roc_auc_spike'] for r in results_with_auc]
    axes[0, 0].bar(range(len(results_with_auc)), roc_aucs)
    axes[0, 0].set_title('ROC AUC for Spike Prediction')
    axes[0, 0].set_xlabel('Model Index')
    axes[0, 0].set_ylabel('ROC AUC')
    axes[0, 0].set_ylim(0.5, 1.0)
    axes[0, 0].set_xticks(range(len(results_with_auc)))
    axes[0, 0].set_xticklabels([f'M{i}' for i in range(len(results_with_auc))], rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # PR AUC比较
    pr_aucs = [r['auc_metrics']['pr_auc_spike'] for r in results_with_auc]
    axes[0, 1].bar(range(len(results_with_auc)), pr_aucs)
    axes[0, 1].set_title('Precision-Recall AUC for Spike Prediction')
    axes[0, 1].set_xlabel('Model Index')
    axes[0, 1].set_ylabel('PR AUC')
    axes[0, 1].set_xticks(range(len(results_with_auc)))
    axes[0, 1].set_xticklabels([f'M{i}' for i in range(len(results_with_auc))], rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Somatic correlation比较
    soma_corrs = [r['auc_metrics']['soma_correlation'] for r in results_with_auc]
    axes[1, 0].bar(range(len(results_with_auc)), soma_corrs)
    axes[1, 0].set_title('Somatic Voltage Correlation')
    axes[1, 0].set_xlabel('Model Index')
    axes[1, 0].set_ylabel('Correlation')
    axes[1, 0].set_xticks(range(len(results_with_auc)))
    axes[1, 0].set_xticklabels([f'M{i}' for i in range(len(results_with_auc))], rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 平均阈值AUC比较
    mean_threshold_aucs = [r['auc_metrics']['mean_auc_thresholds'] for r in results_with_auc]
    axes[1, 1].bar(range(len(results_with_auc)), mean_threshold_aucs)
    axes[1, 1].set_title('Mean AUC at Different Thresholds')
    axes[1, 1].set_xlabel('Model Index')
    axes[1, 1].set_ylabel('Mean AUC')
    axes[1, 1].set_xticks(range(len(results_with_auc)))
    axes[1, 1].set_xticklabels([f'M{i}' for i in range(len(results_with_auc))], rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'auc_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. AUC vs Loss散点图
    plt.figure(figsize=(12, 8))
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('AUC vs Loss Analysis', fontsize=16)
    
    # ROC AUC vs Spike Loss
    axes[0, 0].scatter([r['min_val_spikes_loss'] for r in results_with_auc], 
                      [r['auc_metrics']['roc_auc_spike'] for r in results_with_auc], alpha=0.7, s=100)
    axes[0, 0].set_xlabel('Validation Spike Loss')
    axes[0, 0].set_ylabel('ROC AUC')
    axes[0, 0].set_title('ROC AUC vs Spike Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # PR AUC vs Spike Loss
    axes[0, 1].scatter([r['min_val_spikes_loss'] for r in results_with_auc], 
                      [r['auc_metrics']['pr_auc_spike'] for r in results_with_auc], alpha=0.7, s=100)
    axes[0, 1].set_xlabel('Validation Spike Loss')
    axes[0, 1].set_ylabel('PR AUC')
    axes[0, 1].set_title('PR AUC vs Spike Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Somatic Correlation vs Somatic Loss
    axes[1, 0].scatter([r['min_val_somatic_loss'] for r in results_with_auc], 
                      [r['auc_metrics']['soma_correlation'] for r in results_with_auc], alpha=0.7, s=100)
    axes[1, 0].set_xlabel('Validation Somatic Loss')
    axes[1, 0].set_ylabel('Somatic Correlation')
    axes[1, 0].set_title('Somatic Correlation vs Somatic Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # ROC AUC vs Total Loss
    axes[1, 1].scatter([r['min_val_loss'] for r in results_with_auc], 
                      [r['auc_metrics']['roc_auc_spike'] for r in results_with_auc], alpha=0.7, s=100)
    axes[1, 1].set_xlabel('Validation Total Loss')
    axes[1, 1].set_ylabel('ROC AUC')
    axes[1, 1].set_title('ROC AUC vs Total Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加模型标签
    for i, result in enumerate(results_with_auc):
        axes[0, 0].annotate(f'M{i}', 
                           (result['min_val_spikes_loss'], result['auc_metrics']['roc_auc_spike']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0, 1].annotate(f'M{i}', 
                           (result['min_val_spikes_loss'], result['auc_metrics']['pr_auc_spike']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 0].annotate(f'M{i}', 
                           (result['min_val_somatic_loss'], result['auc_metrics']['soma_correlation']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 1].annotate(f'M{i}', 
                           (result['min_val_loss'], result['auc_metrics']['roc_auc_spike']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'auc_vs_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

def print_model_summary(results):
    """
    打印模型总结，包括AUC指标
    """
    print("=" * 80)
    print("MODEL ANALYSIS SUMMARY")
    print("=" * 80)
    
    # 过滤有AUC指标的模型
    results_with_auc = [r for r in results if r['auc_metrics'] is not None]
    
    if results_with_auc:
        # 按ROC AUC排序
        results_sorted = sorted(results_with_auc, 
                              key=lambda x: x['auc_metrics']['roc_auc_spike'], reverse=True)
        
        print(f"\nTotal models with AUC metrics: {len(results_with_auc)}")
        print("\n" + "=" * 60)
        print("MODELS RANKED BY ROC AUC (Best to Worst)")
        print("=" * 60)
        
        for i, result in enumerate(results_sorted):
            print(f"\n{i+1}. {result['model_name']}")
            print(f"   ROC AUC: {result['auc_metrics']['roc_auc_spike']:.4f}")
            print(f"   PR AUC: {result['auc_metrics']['pr_auc_spike']:.4f}")
            print(f"   Somatic Correlation: {result['auc_metrics']['soma_correlation']:.4f}")
            print(f"   Mean Threshold AUC: {result['auc_metrics']['mean_auc_thresholds']:.4f}")
            print(f"   Min Val Loss: {result['min_val_loss']:.6f}")
            print(f"   Min Val Spike Loss: {result['min_val_spikes_loss']:.6f}")
            print(f"   Min Val Somatic Loss: {result['min_val_somatic_loss']:.6f}")
            
            # 打印架构信息
            arch = result['architecture_dict']
            print(f"   Architecture: DxWxT_{arch['network_depth']}x{arch['num_filters_per_layer'][0]}x{sum(arch['filter_sizes_per_layer'])}")
        
        # 找出最佳AUC模型
        best_auc_model = results_sorted[0]
        print(f"\n" + "=" * 50)
        print("BEST MODEL BY AUC")
        print("=" * 50)
        print(f"Model: {best_auc_model['model_name']}")
        print(f"ROC AUC: {best_auc_model['auc_metrics']['roc_auc_spike']:.4f}")
        print(f"PR AUC: {best_auc_model['auc_metrics']['pr_auc_spike']:.4f}")
        print(f"Model Path: {best_auc_model['model_path']}")
        
        return best_auc_model
    else:
        print("No models with AUC metrics found!")
        return None

def main():
    """
    主函数：运行完整的模型分析，包括AUC评估
    """
    print("Loading model results...")
    models_dir = './Single_Neuron_InOut_SJC_funcgroup2_var2/models/NMDA/depth_7_filters_128_window_400_new_params/'
    test_data_dir = './Single_Neuron_InOut_SJC_funcgroup2_var2/data/L5PC_NMDA_test/'
    save_dir = './model_analysis_plots/SJC_funcgroup2_var2/depth_7_filters_128_window_400_new_params/'
    results = load_model_results(models_dir, test_data_dir)
    
    if not results:
        print("No model files found!")
        return
    
    print(f"Found {len(results)} models to analyze.")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 打印模型总结（包括AUC）
    best_model = print_model_summary(results)
    
    # 2. 绘制训练曲线
    print("\nGenerating training curves...")
    plot_training_curves(results, save_dir)
    
    # 3. 绘制模型比较图
    print("Generating model comparison plots...")
    plot_model_comparison(results, save_dir)
    
    # 4. 分析训练稳定性
    print("Analyzing training stability...")
    analyze_training_stability(results, save_dir)
    
    # 5. AUC分析
    print("Analyzing AUC metrics...")
    plot_auc_analysis(results, save_dir)
    
    print(f"\nAnalysis complete! All plots saved to: {save_dir}")
    if best_model:
        print(f"Best model by AUC: {best_model['model_name']}")
        print(f"Best ROC AUC: {best_model['auc_metrics']['roc_auc_spike']:.4f}")

if __name__ == "__main__":
    main()