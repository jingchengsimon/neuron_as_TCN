import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pickle
import tensorflow as tf
from keras.models import load_model
import random
from datetime import datetime
import time # Added missing import for time.time()
from activity_optimization import find_best_model
from sklearn.metrics import roc_curve
from tqdm.auto import tqdm
import sys
# 不再需要导入可视化函数，因为我们将使用plt.plot()和plt.hlines()

class ModelPredictionVisualizer:
    """模型预测可视化类，用于比较模型预测结果和真实输出"""
    
    def __init__(self, model_path, test_data_dir, sim_idx):
        """
        初始化可视化器
        
        Args:
            model_path: 训练好的模型文件路径 (.h5文件)
            test_data_dir: 测试数据目录
        """
        self.model_path = model_path
        self.test_data_dir = test_data_dir
        
        # 加载模型
        self.model = None
        self.model_info = None
        self.load_model()
        
        self.sim_idx = sim_idx
        
        # 创建输出目录 - 使用模型参数命名而不是时间戳
        model_dir_name = self.extract_model_directory_name(model_path)
        self.output_dir = f"./Results/model_prediction_analysis/{model_dir_name}/simu_trial_{sim_idx}/"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"模型预测可视化器初始化完成")
        print(f"输出目录: {self.output_dir}")
    
    def extract_model_directory_name(self, model_path):
        """
        从模型路径中提取目录名称，格式为：depth*filters*window_dataset
        """
        info = ModelPredictionVisualizer.parse_model_path_and_data_dir(model_path)
        return info['directory_name']

    @staticmethod
    def parse_model_path_and_data_dir(path: str):
        """
        统一解析函数：从模型或目录路径中提取
        - dataset_identifier
        - 参数目录(depth_*_filters_*_window_*)并生成directory_name
        - 对应的test_data_dir
        返回: dict{directory_name, test_data_dir, dataset_identifier}
        """
        try:
            path_parts = path.split('/')
            model_params_dir = None
            dataset_identifier = 'original'

            for part in path_parts:
                if 'depth_' in part and 'filters_' in part and 'window_' in part:
                    model_params_dir = part
                elif 'SJC_funcgroup2_var2' in part:
                    dataset_identifier = 'SJC_funcgroup2_var2'
                elif 'largeSpikeWeight' in part:
                    dataset_identifier = 'largeSpikeWeight'
                elif 'SJC_funcgroup2' in part:
                    dataset_identifier = 'SJC_funcgroup2'
                elif 'SJC' in part:
                    dataset_identifier = 'SJC'
                elif 'fullStrategy' in part:
                    dataset_identifier = 'original_fullStrategy'

            # 解析参数，生成目录名
            directory_name = f"unknown_model_{dataset_identifier}"
            if model_params_dir:
                import re
                depth_match = re.search(r'depth_(\d+)', model_params_dir)
                filters_match = re.search(r'filters_(\d+)', model_params_dir)
                window_match = re.search(r'window_(\d+)', model_params_dir)
                if depth_match and filters_match and window_match:
                    depth = depth_match.group(1)
                    filters = filters_match.group(1)
                    window = window_match.group(1)
                    directory_name = f"{depth}*{filters}*{window}_{dataset_identifier}"

            # 映射测试数据目录
            if dataset_identifier == 'original':
                test_data_dir = "./Models_TCN/Single_Neuron_InOut/data/L5PC_NMDA_test/"
            elif dataset_identifier == 'SJC':
                test_data_dir = "./Models_TCN/Single_Neuron_InOut_SJC/data/L5PC_NMDA_test/"
            elif dataset_identifier == 'SJC_funcgroup2':
                test_data_dir = "./Models_TCN/Single_Neuron_InOut_SJC_funcgroup2/data/L5PC_NMDA_test/"
            elif dataset_identifier == 'SJC_funcgroup2_var2' or dataset_identifier == 'largeSpikeWeight':
                test_data_dir = "./Models_TCN/Single_Neuron_InOut_SJC_funcgroup2_var2/data/L5PC_NMDA_test/"
            elif dataset_identifier == 'original_fullStrategy':
                test_data_dir = "./Models_TCN/Single_Neuron_InOut/data/L5PC_NMDA_test/"
            else:
                test_data_dir = "./Models_TCN/Single_Neuron_InOut/data/L5PC_NMDA_test/"

            return {
                'directory_name': directory_name,
                'test_data_dir': test_data_dir,
                'dataset_identifier': dataset_identifier,
            }
        except Exception as e:
            print(f"解析路径出错: {e}")
            return {
                'directory_name': 'unknown_model_original',
                'test_data_dir': "./Models_TCN/Single_Neuron_InOut/data/L5PC_NMDA_test/",
                'dataset_identifier': 'original',
            }

    @staticmethod
    def get_test_data_dir_for_model(models_dir):
        info = ModelPredictionVisualizer.parse_model_path_and_data_dir(models_dir)
        print(f"  检测到数据集标识: {info['dataset_identifier']}")
        print(f"  对应测试数据目录: {info['test_data_dir']}")
        return info['test_data_dir']
    
    def load_model(self):
        """加载训练好的模型和相关信息"""
        try:
            # 启用GPU显存按需增长，避免一次性占满导致OOM
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except Exception:
                        pass
            except Exception:
                pass

            print(f"正在加载模型: {self.model_path}")
            try:
                self.model = load_model(self.model_path)
            except Exception as e:
                print(f"GPU加载失败({e})，尝试在CPU上加载模型...")
                import tensorflow as tf
                with tf.device('/CPU:0'):
            self.model = load_model(self.model_path)
            print("模型加载成功!")
            
            # 获取模型信息
            print(f"模型输入形状: {self.model.input_shape}")
            print(f"模型输出形状: {self.model.output_shape}")
            
            # 尝试加载相关的pickle文件获取模型信息
            pickle_path = self.model_path.replace('.h5', '.pickle')
            if os.path.exists(pickle_path):
                with open(pickle_path, 'rb') as f:
                    self.model_info = pickle.load(f)
                print("模型信息加载成功!")
                
                # 打印关键参数
                if 'architecture_dict' in self.model_info:
                    arch = self.model_info['architecture_dict']
                    print(f"网络深度: {arch.get('network_depth', 'N/A')}")
                    print(f"输入窗口大小: {arch.get('input_window_size', 'N/A')} ms")
                    print(f"每层滤波器数量: {arch.get('num_filters_per_layer', 'N/A')}")
                
                if 'learning_schedule_dict' in self.model_info:
                    lr_sched = self.model_info['learning_schedule_dict']
                    print(f"训练轮数: {lr_sched.get('num_epochs', 'N/A')}")
                    print(f"批次大小: {lr_sched.get('batch_size_per_epoch', 'N/A')}")
            else:
                print("警告: 未找到相关的pickle文件，无法获取详细模型信息")
                
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def load_test_data(self, num_files=1, file_load=0.3):
        """加载测试数据"""
        test_files = glob.glob(os.path.join(self.test_data_dir, "*.p"))
        if not test_files:
            raise ValueError(f"在 {self.test_data_dir} 中未找到测试数据文件")
        
        print(f"找到 {len(test_files)} 个测试文件")
        
        # 随机选择文件
        selected_files = random.sample(test_files, min(num_files, len(test_files))) # There is only one file in the test_files
        print(f"选择文件: {[os.path.basename(f) for f in selected_files]}")
        
        # 加载数据
        test_data = []
        for file_path in selected_files:
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # 调试：打印数据结构
                print(f"成功加载文件: {os.path.basename(file_path)}")
                print(f"  数据类型: {type(data)}")
                if isinstance(data, dict):
                    print(f"  顶层键: {list(data.keys())}")
                    if 'Results' in data:
                        print(f"  Results键: {list(data['Results'].keys())}")
                        if 'listOfSingleSimulationDicts' in data['Results']:
                            sim_list = data['Results']['listOfSingleSimulationDicts']
                            print(f"  模拟数量: {len(sim_list)}")
                            if sim_list:
                                first_sim = sim_list[0]
                                print(f"  第一个模拟的键: {list(first_sim.keys())}")
                                if 'exInputSpikeTimes' in first_sim:
                                    print(f"  兴奋性segments数量: {len(first_sim['exInputSpikeTimes'])}")
                                if 'inhInputSpikeTimes' in first_sim:
                                    print(f"  抑制性segments数量: {len(first_sim['inhInputSpikeTimes'])}")
                                if 'somaVoltageLowRes' in first_sim:
                                    print(f"  Soma电压长度: {len(first_sim['somaVoltageLowRes'])}")
                
                test_data.append(data)
                
            except Exception as e:
                print(f"加载文件失败 {file_path}: {e}")
                import traceback
                traceback.print_exc()
        
        return test_data
    
    def select_trial_by_main_figure_logic(self, data):
        """
        按照main_figure_replication.py的逻辑选择trial（参考其选择逻辑）
        
        Args:
            data: 加载的数据
            
        Returns:
            selected_sim_index: 选中的simulation索引
        """
        print("按照main_figure_replication.py的逻辑选择trial...")
        
        # 获取模拟列表
        sim_list = data['Results']['listOfSingleSimulationDicts']
        if not sim_list:
            raise ValueError("模拟列表为空")
        
        print(f"找到 {len(sim_list)} 个模拟")
        
        # 计算每个simulation的spike数量（参考main_figure_replication.py）
        num_spikes_per_simulation = []
        for sim in sim_list:
            if 'outputSpikeTimes' in sim:
                output_spike_times = sim['outputSpikeTimes']
                if hasattr(output_spike_times, '__len__') and len(output_spike_times) > 0:
                    # 计算spike数量
                    if isinstance(output_spike_times, np.ndarray):
                        spike_count = len(output_spike_times)
                    else:
                        spike_count = len(output_spike_times)
                else:
                    spike_count = 0
            else:
                spike_count = 0
            num_spikes_per_simulation.append(spike_count)
        
        num_spikes_per_simulation = np.array(num_spikes_per_simulation)
        print(f"Spike数量统计: 最小={num_spikes_per_simulation.min()}, 最大={num_spikes_per_simulation.max()}")
        
        # 选择符合条件的simulation（参考main_figure_replication.py的逻辑）
        # 使用与main_figure_replication.py完全相同的过滤条件
        possible_presentable_candidates = np.nonzero(np.logical_and(num_spikes_per_simulation >= 3, 
                                                                    num_spikes_per_simulation <= 15))[0]  
        print(f"符合条件的simulation数量: {len(possible_presentable_candidates)}")
        
        # 按顺序选择（与main_figure_replication.py保持一致）
        # 使用sim_idx作为选择索引，确保与main_figure_replication.py选择相同的trial
        if len(possible_presentable_candidates) > 0:
            # 使用sim_idx作为选择索引，确保与main_figure_replication.py选择相同的trial
            # selection_index = self.sim_idx % len(possible_presentable_candidates)
            selected_sim_index = possible_presentable_candidates[self.sim_idx]
            
            print(f"使用sim_idx {self.sim_idx} 选择simulation索引: {selected_sim_index}")
            print(f"对应的spike数量: {num_spikes_per_simulation[selected_sim_index]}")
            
            return selected_sim_index
        else:
            # 如果没有符合条件的，使用默认选择
            print("没有符合条件的simulation，使用默认选择")
            return self.sim_idx % len(sim_list)

    def extract_input_output(self, data):
        """
        从数据中提取所有simulation的输入和输出数据（不指定特定的sim_idx）
        
        Args:
            data: 加载的数据
            
        Returns:
            input_data: 所有simulation的输入数据 (num_simulations, segments, full_time)
            target_voltage: 所有simulation的目标电压输出 (num_simulations, full_time)
            target_spikes: 所有simulation的目标spike输出 (num_simulations, full_time)
            segment_info: segment信息
        """
        print("正在解析所有simulation的数据格式...")
        
        # 检查数据是否为字典格式
        if not isinstance(data, dict):
            raise ValueError(f"数据必须是字典格式，当前类型: {type(data)}")
        
        # 检查是否包含必要的键
        if 'Results' not in data or 'listOfSingleSimulationDicts' not in data['Results']:
            raise ValueError("数据格式不正确，缺少 'Results' 或 'listOfSingleSimulationDicts' 键")
        
        # 获取模拟列表
        sim_list = data['Results']['listOfSingleSimulationDicts']
        if not sim_list:
            raise ValueError("模拟列表为空")
        
        print(f"找到 {len(sim_list)} 个simulation")
        
        # 获取兴奋性和抑制性segments数量（从第一个simulation推断）
        first_sim = sim_list[0]
        num_segments_exc = len(first_sim['exInputSpikeTimes'])
        num_segments_inh = len(first_sim['inhInputSpikeTimes'])
        num_segments_total = num_segments_exc + num_segments_inh
        
        # 检查模型期望的输入维度
        expected_features = self.model.input_shape[-1]
        
        # 如果数量不匹配，进行调整
        if num_segments_total != expected_features:
            print(f"特征数量不匹配，进行调整...")
            if num_segments_total < expected_features:
                padding_needed = expected_features - num_segments_total
                print(f"需要填充 {padding_needed} 个特征")
                num_segments_total = expected_features
            else:
                truncation_needed = num_segments_total - expected_features
                print(f"需要截断 {truncation_needed} 个特征")
                num_segments_total = expected_features
        
        # 获取模拟持续时间
        if 'Params' in data and 'STIM DURATION' in data['Params']:
            sim_duration_ms = data['Params']['STIM DURATION'] - 100
        elif 'Params' in data and 'totalSimDurationInSec' in data['Params']:
            sim_duration_ms = int(data['Params']['totalSimDurationInSec'] * 1000)
        else:
            # 从soma电压推断持续时间
            sim_duration_ms = len(first_sim['somaVoltageLowRes'])
        
        print(f"模拟持续时间: {sim_duration_ms} ms")
        
        # 初始化所有simulation的数据数组
        num_simulations = len(sim_list)
        input_data = np.zeros((num_simulations, num_segments_total, sim_duration_ms), dtype=bool)
        target_voltage = np.zeros((num_simulations, sim_duration_ms))
        target_spikes = np.zeros((num_simulations, sim_duration_ms), dtype=bool)
        
        # 使用进度条显示处理simulation的过程
        
        for sim_idx, selected_sim in tqdm(enumerate(sim_list), desc="处理simulations", unit="sim", total=len(sim_list), dynamic_ncols=True, leave=False, position=0):
            # 检查模拟数据是否包含必要的键
            required_keys = ['exInputSpikeTimes', 'inhInputSpikeTimes', 'outputSpikeTimes', 'somaVoltageLowRes']
            missing_keys = [key for key in required_keys if key not in selected_sim]
            if missing_keys:
                print(f"Simulation {sim_idx} 缺少必要的键: {missing_keys}")
                continue
            
            # 获取原始segment数量（用于数据填充/截断）
            original_exc_count = len(selected_sim['exInputSpikeTimes'])
            original_inh_count = len(selected_sim['inhInputSpikeTimes'])
            original_total = original_exc_count + original_inh_count
        
        # 处理兴奋性输入
        for i, (segment_id, spike_times) in enumerate(selected_sim['exInputSpikeTimes'].items()):
                if i < min(original_exc_count, num_segments_exc):
                    if hasattr(spike_times, '__len__') and len(spike_times) > 0:
                        if isinstance(spike_times, np.ndarray):
                            spike_times_array = spike_times.astype(float)
                        else:
                            spike_times_array = np.array(spike_times, dtype=float)
                        
                        mask = (spike_times_array >= 0) & (spike_times_array < sim_duration_ms)
                        valid_spike_times = spike_times_array[mask]
                        
                        if len(valid_spike_times) > 0:
                            spike_indices = valid_spike_times.astype(int)
                            input_data[sim_idx, i, spike_indices] = True
        
        # 处理抑制性输入
        for i, (segment_id, spike_times) in enumerate(selected_sim['inhInputSpikeTimes'].items()):
                if i < min(original_inh_count, num_segments_inh):
                    segment_idx = min(original_exc_count, num_segments_exc) + i
                    if segment_idx < num_segments_total:
                        if hasattr(spike_times, '__len__') and len(spike_times) > 0:
                            if isinstance(spike_times, np.ndarray):
                                spike_times_array = spike_times.astype(float)
                            else:
                                spike_times_array = np.array(spike_times, dtype=float)
                            
                            mask = (spike_times_array >= 0) & (spike_times_array < sim_duration_ms)
                            valid_spike_times = spike_times_array[mask]
                            
                            if len(valid_spike_times) > 0:
                                spike_indices = valid_spike_times.astype(int)
                                input_data[sim_idx, segment_idx, spike_indices] = True
            
            # 处理输出spikes
            output_spike_times = selected_sim['outputSpikeTimes']
            if hasattr(output_spike_times, '__len__') and len(output_spike_times) > 0:
                if isinstance(output_spike_times, np.ndarray):
                    output_spike_times_array = output_spike_times.astype(float)
            else:
                    output_spike_times_array = np.array(output_spike_times, dtype=float)
                
                mask = (output_spike_times_array >= 0) & (output_spike_times_array < sim_duration_ms)
                valid_output_spike_times = output_spike_times_array[mask]
                
                if len(valid_output_spike_times) > 0:
                    output_spike_indices = valid_output_spike_times.astype(int)
                    target_spikes[sim_idx, output_spike_indices] = True
            
            # 处理soma电压
            target_voltage[sim_idx, :] = selected_sim['somaVoltageLowRes'][:sim_duration_ms]
        
        segment_info = {
            'sim_duration_ms': sim_duration_ms,
            'num_segments': num_segments_total,
            'num_simulations': num_simulations,
            'num_segments_exc': num_segments_exc,
            'num_segments_inh': num_segments_inh
        }
        
        return input_data, target_voltage, target_spikes, segment_info
        
    def predict_output(self, input_data):
        """
        使用模型预测输出，使用滑动窗口技术处理完整的6秒模拟数据
        
        Args:
            input_data: 完整的输入数据 (segments, full_time) 或 (num_simulations, segments, full_time)
            
        Returns:
            predicted_voltage: 预测的完整电压输出
            predicted_spikes: 预测的完整spike输出
        """
        print("开始模型预测（使用滑动窗口技术）...")
        
        # 检查输入数据的维度
        has_simulation_dim = len(input_data.shape) == 3
        
        if has_simulation_dim:
            # 处理多个simulation的情况
            num_simulations, num_segments, full_time_steps = input_data.shape
            print(f"检测到多个simulation: {num_simulations}")
            
            # 初始化预测结果数组
            predicted_spikes_full = np.zeros((num_simulations, full_time_steps))
            predicted_voltage_full = np.zeros((num_simulations, full_time_steps))
            
            # 使用进度条显示处理simulation的过程
            for sim_idx in tqdm(range(num_simulations), desc="处理simulations", unit="sim", total=num_simulations, dynamic_ncols=True, leave=False, position=1):
                curr_input = input_data[sim_idx]  # (segments, full_time)
                
                # 递归调用自身处理单个simulation
                curr_pred_voltage, curr_pred_spikes = self._predict_single_simulation(curr_input)
                
                predicted_voltage_full[sim_idx] = curr_pred_voltage
                predicted_spikes_full[sim_idx] = curr_pred_spikes
            
            return predicted_voltage_full, predicted_spikes_full
        else:
            # 处理单个simulation的情况
            return self._predict_single_simulation(input_data)
    
    def _predict_single_simulation(self, input_data):
        """
        预测单个simulation的输出（内部方法）
        
        Args:
            input_data: 单个simulation的输入数据 (segments, full_time)
            
        Returns:
            predicted_voltage: 预测的完整电压输出
            predicted_spikes: 预测的完整spike输出
        """
        
        # 获取模型输入窗口大小
        input_window_size = self.model.input_shape[1]
        
        # 计算重叠大小（参考main_figure_replication.py）
        if self.model_info and 'architecture_dict' in self.model_info:
            arch = self.model_info['architecture_dict']
            filter_sizes = arch.get('filter_sizes_per_layer', [])
            if filter_sizes:
                time_window_T = (np.array(filter_sizes) - 1).sum() + 1
                overlap_size = min(max(time_window_T + 1, min(150, input_window_size - 50)), 250)
            else:
                overlap_size = 100  # 默认值
        else:
            overlap_size = 100  # 默认值
        
        # 准备输入数据格式
        if len(input_data.shape) == 2:
            # 转置为模型期望的格式: (time_steps, features)
            input_data_transposed = input_data.T  # (full_time, segments)
        else:
            input_data_transposed = input_data
        
        full_time_steps = input_data_transposed.shape[0]
        num_features = input_data_transposed.shape[1]
        
        # 初始化预测结果数组
        predicted_spikes_full = np.zeros(full_time_steps)
        predicted_voltage_full = np.zeros(full_time_steps)
        
        # 计算需要多少个测试分割
        num_test_splits = int(2 + (full_time_steps - input_window_size) / (input_window_size - overlap_size))

        # 使用滑动窗口进行预测
        for k in range(num_test_splits):
            start_time_ind = k * (input_window_size - overlap_size)
            end_time_ind = start_time_ind + input_window_size
            
            # 提取当前窗口的输入数据
            curr_input = input_data_transposed[start_time_ind:end_time_ind, :]
            
            # 如果当前窗口长度不足，进行填充
            if curr_input.shape[0] < input_window_size:
                padding_size = input_window_size - curr_input.shape[0]
                padding = np.zeros((padding_size, num_features))
                curr_input = np.vstack((curr_input, padding))
            
            # 添加batch维度
            model_input = np.expand_dims(curr_input, axis=0)  # (1, window_size, features)
            
            # 进行预测
            prediction = self.model.predict(model_input, verbose=0)

            # 处理预测结果
            if isinstance(prediction, list) and len(prediction) == 2:
                # 多输出模型：spikes和soma电压
                curr_pred_spikes = prediction[0][0] if len(prediction[0].shape) == 3 else prediction[0]
                curr_pred_soma = prediction[1][0] if len(prediction[1].shape) == 3 else prediction[1]
                
                # 确保是1D数组
                if len(curr_pred_spikes.shape) > 1:
                    curr_pred_spikes = np.squeeze(curr_pred_spikes)
                if len(curr_pred_soma.shape) > 1:
                    curr_pred_soma = np.squeeze(curr_pred_soma)
                
            else:
                # 单输出模型
                if isinstance(prediction, list):
                prediction = prediction[0]
                if len(prediction.shape) == 3:
                    prediction = prediction[0]
                if len(prediction.shape) > 1:
                    prediction = np.squeeze(prediction)
                
                # 假设是电压输出
                curr_pred_soma = prediction
                curr_pred_spikes = np.zeros_like(prediction, dtype=bool)
                
            # 应用电压偏置校正
            # y_train_soma_bias = -67.7
            # curr_pred_soma = curr_pred_soma + y_train_soma_bias
        
            # 将预测结果拼接到完整数组中
            if k == 0:
                # 第一个分割：直接填充
                predicted_spikes_full[:end_time_ind] = curr_pred_spikes
                predicted_voltage_full[:end_time_ind] = curr_pred_soma
            elif k == (num_test_splits - 1):
                # 最后一个分割：从重叠位置开始填充
                t0 = start_time_ind + overlap_size
                duration_to_fill = full_time_steps - t0
                if duration_to_fill > 0:
                    predicted_spikes_full[t0:] = curr_pred_spikes[overlap_size:(overlap_size + duration_to_fill)]
                    predicted_voltage_full[t0:] = curr_pred_soma[overlap_size:(overlap_size + duration_to_fill)]
            else:
                # 中间分割：从重叠位置开始填充到当前分割结束
                t0 = start_time_ind + overlap_size
                predicted_spikes_full[t0:end_time_ind] = curr_pred_spikes[overlap_size:]
                predicted_voltage_full[t0:end_time_ind] = curr_pred_soma[overlap_size:]
        
        return predicted_voltage_full, predicted_spikes_full
    
    def calculate_threshold_from_full_testset(self, test_data):
        """
        使用整个测试集计算ROC曲线和确定阈值，并保存所有trial的预测结果（参考main_figure_replication.py）
        
        Args:
            test_data: 整个测试数据集
            
        Returns:
            desired_threshold: 基于整个测试集确定的最佳阈值
            all_predictions: 包含所有trial预测结果的字典
        """
        print("正在使用整个测试集计算ROC曲线...")
        
        # 收集所有测试数据的target spikes和predicted spikes raw
        all_target_spikes = []
        all_target_voltage = []
        all_predicted_spikes_raw = []
        all_predicted_voltage = []
        all_segment_info = []
        all_input_data = []
        
        for i, data in enumerate(test_data):
            try:
                print(f"处理测试数据 {i+1}/{len(test_data)} 用于ROC计算...")
                
                # 提取输入输出数据 - 不指定sim_idx，处理所有simulation
                input_data, target_voltage, target_spikes, segment_info = self.extract_input_output(data)
                
                # 使用模型进行预测
                predicted_voltage, predicted_spikes_raw = self.predict_output(input_data)
                
                # 保存所有数据供后续使用
                all_input_data.append(input_data)
                all_target_voltage.append(target_voltage)
                all_predicted_voltage.append(predicted_voltage)
                all_target_spikes.append(target_spikes)
                all_predicted_spikes_raw.append(predicted_spikes_raw)
                all_segment_info.append(segment_info)
                
                # # 应用时间过滤（参考main_figure_replication.py）
                # ignore_time_at_start_ms = 500
                # time_points_to_eval = np.arange(target_spikes.shape[1]) >= ignore_time_at_start_ms
                
                # # 收集过滤后的数据 - 注意此时数据有额外的simulation维度
                # target_spikes_to_eval = target_spikes[:, time_points_to_eval]  # (num_simulations, time_points)
                # predicted_spikes_raw_to_eval = predicted_spikes_raw[:, time_points_to_eval]  # (num_simulations, time_points)
                
            except Exception as e:
                print(f"处理测试数据 {i+1} 时出错: {e}")
                continue
        
        if not all_target_spikes:
            raise ValueError("无法从测试数据中提取有效数据用于ROC计算")
        
        # 应用电压标准化（参考main_figure_replication.py）
        print("应用电压标准化处理...")
        
        # 将所有数据合并为单个数组进行标准化
        all_target_voltage_combined = np.concatenate(all_target_voltage, axis=0)  # (total_simulations, time_points)
        all_predicted_voltage_combined = np.concatenate(all_predicted_voltage, axis=0)  # (total_simulations, time_points)
        
        # 创建target voltage的copy，应用阈值限制和偏置校正（参考main_figure_replication.py）
        v_threshold = -55
        y_train_soma_bias = -67.7
        
        # 对target voltage应用阈值限制
        all_target_voltage_combined_copy = all_target_voltage_combined.copy()
        all_target_voltage_combined_copy[all_target_voltage_combined_copy > v_threshold] = v_threshold
        
        # 对predicted voltage应用偏置校正
        all_predicted_voltage_combined = all_predicted_voltage_combined - y_train_soma_bias
        
        # 计算标准化参数（参考main_figure_replication.py）
        s_dst = all_target_voltage_combined_copy.std()
        m_dst = all_target_voltage_combined_copy.mean()
        s_src = all_predicted_voltage_combined.std()
        m_src = all_predicted_voltage_combined.mean()
        
        # print(f"电压标准化参数:")
        # print(f"  Target (after threshold) - std: {s_dst:.4f}, mean: {m_dst:.4f}")
        # print(f"  Predicted (after bias) - std: {s_src:.4f}, mean: {m_src:.4f}")
        
        # 应用标准化：先标准化，再重新缩放（参考main_figure_replication.py）
        all_predicted_voltage_combined = (all_predicted_voltage_combined - m_src) / s_src
        all_predicted_voltage_combined = s_dst * all_predicted_voltage_combined + m_dst
        
        # 将标准化后的电压重新分配到各个文件中
        start_idx = 0
        for i in range(len(all_predicted_voltage)):
            num_simulations = all_predicted_voltage[i].shape[0]
            end_idx = start_idx + num_simulations
            all_predicted_voltage[i] = all_predicted_voltage_combined[start_idx:end_idx]
            start_idx = end_idx
        
        # 应用时间过滤（参考main_figure_replication.py）
        ignore_time_at_start_ms = 500
        time_points_to_eval = np.arange(all_target_spikes[0].shape[1]) >= ignore_time_at_start_ms
        
        # 收集过滤后的数据
        target_spikes_to_eval = []
        predicted_spikes_raw_to_eval = []
        
        for i in range(len(all_target_spikes)):
            # 应用时间过滤
            curr_target_spikes = all_target_spikes[i][:, time_points_to_eval]  # (num_simulations, time_points)
            curr_predicted_spikes_raw = all_predicted_spikes_raw[i][:, time_points_to_eval]  # (num_simulations, time_points)
            
            target_spikes_to_eval.append(curr_target_spikes)
            predicted_spikes_raw_to_eval.append(curr_predicted_spikes_raw)
        
        # 将所有过滤后的数据合并
        target_spikes_to_eval = np.concatenate(target_spikes_to_eval, axis=0)  # (total_simulations, time_points)
        predicted_spikes_raw_to_eval = np.concatenate(predicted_spikes_raw_to_eval, axis=0)  # (total_simulations, time_points)
        
        print(f"过滤后的数据形状:")
        print(f"  Target spikes: {target_spikes_to_eval.shape}")
        print(f"  Predicted spikes raw: {predicted_spikes_raw_to_eval.shape}")
        
        # 计算ROC曲线（参考main_figure_replication.py）
        desired_false_positive_rate = 0.002
        fpr, tpr, thresholds = roc_curve(target_spikes_to_eval.ravel(), predicted_spikes_raw_to_eval.ravel()) # Shape: (num_simulations, num_time_points-ignore_time), e.g. (128, 5500)
        
        # 找到最接近目标假阳性率的阈值
        desired_fp_ind = np.argmin(np.abs(fpr - desired_false_positive_rate))
        if desired_fp_ind == 0:
            desired_fp_ind = 1
        
        # 获取最佳阈值
        desired_threshold = thresholds[desired_fp_ind]
        actual_false_positive_rate = fpr[desired_fp_ind]
        true_positive_rate = tpr[desired_fp_ind]
        
        print(f"基于整个测试集的ROC计算结果:")
        print(f"  Desired False Positive Rate: {desired_false_positive_rate}")
        print(f"  Actual False Positive Rate: {actual_false_positive_rate:.4f}")
        print(f"  Actual True Positive Rate: {true_positive_rate:.4f}")
        print(f"  Desired Threshold: {desired_threshold:.10f}")
        
        # 保存所有预测结果供run_analysis使用
        all_predictions = {
            'input_data': all_input_data,
            'target_voltage': all_target_voltage,
            'predicted_voltage': all_predicted_voltage,
            'target_spikes': all_target_spikes,
            'predicted_spikes_raw': all_predicted_spikes_raw,
            'segment_info': all_segment_info,
            'desired_threshold': desired_threshold,
            'actual_false_positive_rate': actual_false_positive_rate,
            'true_positive_rate': true_positive_rate
        }
        
        return desired_threshold, all_predictions
  
    def run_analysis(self, num_samples=3):
        """
        运行完整的分析流程（处理完整6秒模拟数据）
        
        Args:
            num_samples: 分析的样本数量
        """
        print(f"\n{'='*60}")
        print("开始模型预测分析（完整6秒模拟）")
        print(f"{'='*60}")
        
        # 加载测试数据
        test_data = self.load_test_data(num_files=min(num_samples, 5))
        
        # 首先使用整个测试集计算ROC曲线和确定阈值，并获取所有trial的预测结果
        print("使用整个测试集计算ROC曲线和确定阈值...")
        desired_threshold, all_predictions = self.calculate_threshold_from_full_testset(test_data)
        
        # 分析每个样本 - 使用已保存的预测结果
        all_results = []
        for i in range(min(num_samples, len(test_data))):
            print(f"\n--- 分析样本 {i+1}/{min(num_samples, len(test_data))} ---")
            
            try:
                # 从保存的预测结果中提取对应的数据
                input_data = all_predictions['input_data'][i]  # (num_simulations, segments, time_points)
                target_voltage = all_predictions['target_voltage'][i]  # (num_simulations, time_points)
                predicted_voltage = all_predictions['predicted_voltage'][i]  # (num_simulations, time_points)
                target_spikes = all_predictions['target_spikes'][i]  # (num_simulations, time_points)
                predicted_spikes_raw = all_predictions['predicted_spikes_raw'][i]  # (num_simulations, time_points)
                segment_info = all_predictions['segment_info'][i]
                
                # 使用select_trial_by_main_figure_logic选择对应的simulation
                sim_idx_to_use = self.select_trial_by_main_figure_logic(test_data[i])
                print(f"使用select_trial_by_main_figure_logic选择simulation {sim_idx_to_use} (总共 {target_voltage.shape[0]} 个)")
                
                # 提取选中的simulation数据
                input_data_selected = input_data[sim_idx_to_use]  # (segments, time_points)
                target_voltage_selected = target_voltage[sim_idx_to_use]  # (time_points,)
                predicted_voltage_selected = predicted_voltage[sim_idx_to_use]  # (time_points,)
                target_spikes_selected = target_spikes[sim_idx_to_use]  # (time_points,)
                predicted_spikes_raw_selected = predicted_spikes_raw[sim_idx_to_use]  # (time_points,)
                
                # # 应用电压阈值限制
                # v_threshold = -55
                # target_voltage_selected[target_voltage_selected > v_threshold] = v_threshold
                
                # 使用从整个测试集确定的阈值将sigmoid输出转换为二进制spikes
                # print(f"样本 {i+1}: 使用从整个测试集确定的阈值: {desired_threshold:.6f}")
                predicted_spikes_selected = (predicted_spikes_raw_selected > desired_threshold).astype(bool)
                
                # 创建综合可视化图
                model_dir_name = self.extract_model_directory_name(self.model_path)
                save_prefix = f"{model_dir_name}_sim_index_{sim_idx_to_use}"
                
                self.create_comprehensive_visualization(
                    input_data_selected, predicted_voltage_selected, predicted_spikes_selected,
                    target_voltage_selected, target_spikes_selected, segment_info, save_prefix, sim_idx_to_use
                )
                
                # 保存结果
                result = {
                    'sample_index': i+1,
                    'segment_info': segment_info,
                    'input_shape': input_data.shape,
                    'target_voltage_shape': target_voltage_selected.shape,
                    'target_spikes_shape': target_spikes_selected.shape,
                    'predicted_voltage_shape': predicted_voltage_selected.shape,
                    'predicted_spikes_shape': predicted_spikes_selected.shape,
                    'roc_threshold_info': {
                        'desired_threshold': desired_threshold,
                        'actual_false_positive_rate': all_predictions['actual_false_positive_rate'],
                        'true_positive_rate': all_predictions['true_positive_rate']
                    }
                }
                all_results.append(result)
                
                print(f"样本 {i+1} 分析完成")
                
            except Exception as e:
                print(f"样本 {i+1} 分析失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 生成分析报告
        self.generate_analysis_report(all_results)
        
        print(f"\n{'='*60}")
        print("模型预测分析完成!")
        print(f"结果保存在: {self.output_dir}")
        print(f"{'='*60}")
        
        return all_results
    
    def create_comprehensive_visualization(self, input_data, predicted_voltage, predicted_spikes, 
                                         target_voltage, target_spikes, segment_info, 
                                         save_prefix="comprehensive", sim_idx=0):
        """
        创建综合可视化图，在同一个figure中显示所有信息（完整6秒模拟）
        
        Args:
            input_data: 完整的输入数据 (segments, full_time)
            predicted_voltage: 预测的完整电压
            predicted_spikes: 预测的完整spikes
            target_voltage: 完整的目标电压
            target_spikes: 完整的目标spikes
            segment_info: segment信息
            save_prefix: 保存文件前缀
        """
        print("生成综合可视化图（完整6秒模拟）...")
        
        # 创建保存路径 - 使用模型目录名称而不是时间戳
        model_dir_name = self.extract_model_directory_name(self.model_path)
        
        # 创建一个大figure，包含所有信息
        fig = plt.figure(figsize=(48, 20))
        
        # 创建网格布局：4行1列，单列布局
        gs = fig.add_gridspec(4, 1, height_ratios=[1.5, 3, 1, 0.5])
        
        # 获取时间信息
        sim_duration_ms = segment_info['sim_duration_ms']
        sim_duration_sec = sim_duration_ms / 1000.0
        time_axis_ms = np.arange(sim_duration_ms)
        time_axis_sec = time_axis_ms / 1000.0
        
        # 1. 输入spike图 (第一行，单列)
        ax_input = fig.add_subplot(gs[0, 0])
        
        # 分离exc和inh segments
        num_exc = segment_info['num_segments_exc']
        num_inh = segment_info['num_segments_inh']
        
        # 参考pickle load用vlines绘制InputSpiketimes
        # 绘制exc segments的spikes
        exc_spikes = input_data[:num_exc, :]
        for syn_id in range(num_exc):
            spike_times = np.where(exc_spikes[syn_id, :])[0]
            if len(spike_times) > 0:
                ax_input.vlines(spike_times, syn_id - 0.4, syn_id + 0.4, color='blue', linewidth=2)
        
        # 绘制inh segments的spikes
        inh_spikes = input_data[num_exc:, :]
        for syn_id in range(num_inh):
            spike_times = np.where(inh_spikes[syn_id, :])[0]
            if len(spike_times) > 0:
                ax_input.vlines(spike_times, syn_id + num_exc - 0.4, syn_id + num_exc + 0.4, color='red', linewidth=2)
        
        ax_input.set_title('Input Spikes (All Segments) - Full 6s Simulation', fontsize=16, fontweight='bold')
        ax_input.set_xlabel('Time (ms)', fontsize=14)
        ax_input.set_ylabel('Segment Index', fontsize=14)
        ax_input.set_xlim(0, sim_duration_ms)
        
        # 去除grid和边框
        ax_input.grid(False)
        ax_input.spines['top'].set_visible(False)
        ax_input.spines['right'].set_visible(False)
        
        # 添加exc和inh的分界线
        ax_input.axhline(y=num_exc-0.5, color='black', linestyle='--', alpha=0.7, linewidth=2)
        ax_input.text(0.02, 0.98, f'Exc: {num_exc} segments', transform=ax_input.transAxes, 
                     verticalalignment='top', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax_input.text(0.02, 0.02, f'Inh: {num_inh} segments', transform=ax_input.transAxes, 
                     verticalalignment='bottom', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # 2. 电压比较图 (第二行，单列)
        ax_voltage = fig.add_subplot(gs[1, 0], sharex=ax_input)
        
        # 获取spike时间点
        target_spike_times = np.where(target_spikes)[0]
        predicted_spike_times = np.where(predicted_spikes)[0]
        
        # 创建电压副本用于spike标记（参考main_figure_replication.py）
        target_voltage_with_spikes = target_voltage.copy()
        predicted_voltage_with_spikes = predicted_voltage.copy()
        
        # 将spike时间点的电压设置为40mV（参考main_figure_replication.py）
        # if len(target_spike_times) > 0:
        #     target_voltage_with_spikes[target_spike_times] = 40
        if len(predicted_spike_times) > 0:
            predicted_voltage_with_spikes[predicted_spike_times] = 40
        
        # 绘制电压 - 使用毫秒时间轴，包含spike标记
        ax_voltage.plot(time_axis_ms, target_voltage_with_spikes, 'c-', linewidth=2, label='Target Voltage', alpha=0.8)
        ax_voltage.plot(time_axis_ms, predicted_voltage_with_spikes, 'm', linestyle=':', linewidth=2, label='Predicted Voltage', alpha=0.8)
        
        # 绘制spikes - 使用hlines，时间轴为毫秒
        if len(target_spike_times) > 0:
            # 确保xmin和xmax是标量或者长度相同的数组
            if len(target_spike_times) == 1:
                ax_voltage.hlines(y=np.min(target_voltage) - 5, xmin=target_spike_times[0], xmax=target_spike_times[0] + 1, 
                                 colors='c', linewidth=3, label='Target Spikes', alpha=0.7)
            else:
                # 对于多个spikes，需要确保xmin和xmax都是数组且长度一致
                xmin_array = target_spike_times.astype(float)
                xmax_array = (target_spike_times + 1).astype(float)
                # 对于多个spikes，y值也需要是数组
                y_array = np.full(len(target_spike_times), np.min(target_voltage) - 5)
                ax_voltage.hlines(y=y_array, xmin=xmin_array, xmax=xmax_array, 
                                 colors='c', linewidth=3, label='Target Spikes', alpha=0.7)
        
        if len(predicted_spike_times) > 0:
            # 确保xmin和xmax是标量或者长度相同的数组
            if len(predicted_spike_times) == 1:
                ax_voltage.hlines(y=np.min(target_voltage) - 10, xmin=predicted_spike_times[0], xmax=predicted_spike_times[0] + 1, 
                                 colors='m', linewidth=3, label='Predicted Spikes', alpha=0.7)
            else:
                # 对于多个spikes，需要确保xmin和xmax都是数组且长度一致
                xmin_array = predicted_spike_times.astype(float)
                xmax_array = (predicted_spike_times + 1).astype(float)
                # 对于多个spikes，y值也需要是数组
                y_array = np.full(len(predicted_spike_times), np.min(target_voltage) - 10)
                ax_voltage.hlines(y=y_array, xmin=xmin_array, xmax=xmax_array, 
                                 colors='m', linewidth=3, label='Predicted Spikes', alpha=0.7)
        
        ax_voltage.set_title('Voltage Comparison - Full 6s Simulation', fontsize=14, fontweight='bold')
        ax_voltage.set_ylabel('Voltage (mV)', fontsize=12)
        ax_voltage.legend(fontsize=11)
        ax_voltage.set_xlabel('Time (ms)', fontsize=12)
        
        # 去除grid和边框
        ax_voltage.grid(False)
        ax_voltage.spines['top'].set_visible(False)
        ax_voltage.spines['right'].set_visible(False)
        
        # 3. 电压差异图 (第四行，单列)
        ax_diff = fig.add_subplot(gs[2, 0], sharex=ax_input)

        v_threshold = -55
        target_voltage[target_voltage > v_threshold] = v_threshold
        voltage_diff = predicted_voltage - target_voltage
        # 绘制电压差异 - 使用毫秒时间轴
        ax_diff.plot(time_axis_ms, voltage_diff, 'g-', linewidth=2, alpha=0.8)
        ax_diff.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax_diff.set_title('Voltage Difference (Predicted - Target) - Full 6s Simulation', fontsize=14, fontweight='bold')
        ax_diff.set_ylabel('Voltage Difference (mV)', fontsize=12)
        ax_diff.set_xlabel('Time (ms)', fontsize=12)
        
        # 去除grid和边框
        ax_diff.grid(False)
        ax_diff.spines['top'].set_visible(False)
        ax_diff.spines['right'].set_visible(False)
        

        # 4. Spike比较图 (第三行，单列)
        ax_spikes = fig.add_subplot(gs[3, 0], sharex=ax_input)
        
        # 绘制spikes - 使用毫秒时间轴
        ax_spikes.plot(time_axis_ms, target_spikes.astype(float), 'c-', linewidth=2, label='Target Spikes', alpha=0.8)
        ax_spikes.plot(time_axis_ms, predicted_spikes.astype(float), 'm-', linewidth=2, label='Predicted Spikes', alpha=0.8)
        
        ax_spikes.set_title('Spike Comparison - Full 6s Simulation', fontsize=14, fontweight='bold')
        ax_spikes.set_ylabel('Spike Value', fontsize=12)
        ax_spikes.legend(fontsize=11)
        ax_spikes.set_xlabel('Time (ms)', fontsize=12)
        
        # 去除grid和边框
        ax_spikes.grid(False)
        ax_spikes.spines['top'].set_visible(False)
        ax_spikes.spines['right'].set_visible(False)
        ax_spikes.set_ylim(-0.1, 1.1)


        # 设置总标题
        fig.suptitle(f'Comprehensive Model Prediction Analysis - Full 6s Simulation\n'
                    f'Simulation: {sim_idx}', 
                    fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.05, left=0.05, right=0.95)
        
        # 保存综合可视化图
        comprehensive_save_path = os.path.join(self.output_dir, f"{save_prefix}.png")
        plt.savefig(comprehensive_save_path, dpi=300, bbox_inches='tight')
        print(f"综合可视化图已保存: {comprehensive_save_path}")
        plt.close()
        
        return comprehensive_save_path
    
    def generate_analysis_report(self, results):
        """生成分析报告"""
        if not results:
            print("没有可用的结果来生成报告")
            return
        
        report_path = os.path.join(self.output_dir, "analysis_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("模型预测分析报告（完整6秒模拟）\n")
            f.write("=" * 60 + "\n\n")
            
            # 模型信息
            f.write("模型信息:\n")
            f.write(f"  模型文件: {self.model_path}\n")
            if self.model_info and 'architecture_dict' in self.model_info:
                arch = self.model_info['architecture_dict']
                f.write(f"  网络深度: {arch.get('network_depth', 'N/A')}\n")
                f.write(f"  输入窗口大小: {arch.get('input_window_size', 'N/A')} ms\n")
                f.write(f"  每层滤波器数量: {arch.get('num_filters_per_layer', 'N/A')}\n")
            f.write("\n")
            
            # 分析结果汇总
            f.write("分析结果汇总:\n")
            f.write("-" * 30 + "\n")
            
            for result in results:
                f.write(f"样本 {result['sample_index']}:\n")
                f.write(f"  模拟持续时间: {result['segment_info']['sim_duration_ms']} ms\n")
                f.write(f"  输入形状: {result['input_shape']}\n")
                f.write(f"  目标电压形状: {result['target_voltage_shape']}\n")
                f.write(f"  目标spikes形状: {result['target_spikes_shape']}\n")
                f.write(f"  预测电压形状: {result['predicted_voltage_shape']}\n")
                f.write(f"  预测spikes形状: {result['predicted_spikes_shape']}\n")
                
                # 添加ROC阈值信息
                if 'roc_threshold_info' in result:
                    roc_info = result['roc_threshold_info']
                    f.write(f"  ROC阈值信息:\n")
                    f.write(f"    最佳阈值: {roc_info['desired_threshold']:.4f}\n")
                    # f.write(f"    目标假阳性率: {roc_info['desired_false_positive_rate']:.4f}\n")
                    # f.write(f"    实际假阳性率: {roc_info['actual_false_positive_rate']:.4f}\n")
                    # f.write(f"    真阳性率: {roc_info['true_positive_rate']:.4f}\n")
                
                f.write("\n")
            
            # 总体统计
            f.write("总体统计:\n")
            f.write("-" * 20 + "\n")
            f.write(f"分析样本数量: {len(results)}\n")
            if results:
                f.write(f"模拟持续时间: {results[0]['segment_info']['sim_duration_ms']} ms\n")
                f.write(f"segment数量: {results[0]['input_shape'][0] if results else 'N/A'}\n")
            
            # 模型性能评估
            f.write("\n模型性能评估:\n")
            f.write("-" * 20 + "\n")
            f.write("✓ 完整6秒模拟数据可视化完成\n")
            f.write("✓ 使用滑动窗口技术处理长序列数据\n")
            f.write("✓ 使用plt.plot()绘制电压曲线\n")
            f.write("✓ 使用plt.hlines()绘制spike标记\n")
            f.write("✓ 生成综合比较图表\n")
            f.write("✓ 支持多输出模型（spikes和soma电压）\n")
            f.write("✓ 使用ROC曲线自动确定最佳阈值（参考main_figure_replication.py）\n")
        
        print(f"分析报告已生成: {report_path}")


def main():
    """主函数 - 演示如何使用ModelPredictionVisualizer"""
    
    # 配置参数 - 使用Single Neuron InOut中的数据和模型
    for sim_idx in range(5):
       for models_dir in [
                        #    "./Models_TCN/Single_Neuron_InOut/models/NMDA/depth_7_filters_256_window_400/",
                           "./Models_TCN/Single_Neuron_InOut/models/NMDA_fullStrategy/depth_7_filters_256_window_400/",
                        #    "./Models_TCN/Single_Neuron_InOut_SJC/models/NMDA/depth_7_filters_256_window_400/",
                        #    "./Models_TCN/Single_Neuron_InOut_SJC_funcgroup2_var2/models/NMDA/depth_7_filters_256_window_400_new_params/",
                        #    "./Models_TCN/Single_Neuron_InOut_SJC_funcgroup2_var2/models/NMDA_largeSpikeWeight/depth_7_filters_256_window_400/"
                           ]:

            test_data_dir = ModelPredictionVisualizer.get_test_data_dir_for_model(models_dir)
            
            print(f"\n--- 处理模型目录: {models_dir} ---")
            print(f"动态选择的测试数据目录: {test_data_dir}")
    
    # 查找最新的模型文件
            if os.path.isdir(models_dir):
                model_path, params_path = find_best_model(models_dir)
    
    # 检查文件是否存在
            if not os.path.exists(models_dir):
                print(f"模型目录不存在: {models_dir}")
        return
    
    if not os.path.exists(test_data_dir):
        print(f"测试数据目录不存在: {test_data_dir}")
        return
    
    try:
        # 创建可视化器
                visualizer = ModelPredictionVisualizer(model_path, test_data_dir, sim_idx)
        
                # 运行分析 - 需要加载足够的测试数据来计算ROC曲线
        results = visualizer.run_analysis(
                    num_samples=1  # 加载5个测试文件用于ROC计算
        )
        
        print(f"\n分析完成！共分析了 {len(results)} 个样本")
        print(f"所有结果和图表保存在: {visualizer.output_dir}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
