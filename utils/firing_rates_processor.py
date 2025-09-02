import numpy as np
from keras.models import load_model
import pickle
from utils.fit_CNN import parse_sim_experiment_file

class FiringRatesProcessor:
    """
    专门处理firing rates数据处理的类
    包含从加载firing rates到生成模型预测的所有数据处理功能
    """
    
    def __init__(self, model_path, model_params_path, time_duration_ms=300):
        """
        初始化数据处理器
        
        Args:
            model_path: 训练好的模型.h5文件路径
            model_params_path: 对应的参数.pickle文件路径
            time_duration_ms: 时间长度，默认300ms
        """
        self.model_path = model_path
        self.model_params_path = model_params_path
        self.time_duration_ms = time_duration_ms
        
        # 加载模型和参数
        self.model = load_model(model_path)
        with open(model_params_path, 'rb') as f:
            self.model_params = pickle.load(f)
        
        # 提取模型架构信息
        self.architecture_dict = self.model_params['architecture_dict']
        self.input_window_size = self.architecture_dict['input_window_size']
        
        # 从训练数据中获取segment数量信息
        train_files = self.model_params['data_dict']['train_files']
        if train_files:
            # 解析一个训练文件来获取segment信息
            X_sample, _, _ = parse_sim_experiment_file(train_files[0])
            self.num_segments_exc = 639  # 根据train_and_analyze.py中的设置
            self.num_segments_inh = 640  # SJC数据集的设置
            self.num_segments_total = self.num_segments_exc + self.num_segments_inh
        else:
            # 默认值
            self.num_segments_exc = 639
            self.num_segments_inh = 640
            self.num_segments_total = 1279
        
        print(f"FiringRatesProcessor初始化成功:")
        print(f"  输入窗口大小: {self.input_window_size}ms")
        print(f"  兴奋性segments: {self.num_segments_exc}")
        print(f"  抑制性segments: {self.num_segments_inh}")
        print(f"  总segments: {self.num_segments_total}")
    
    def load_init_firing_rates(self, firing_rates_path):
        """
        加载初始firing rates
        
        Args:
            firing_rates_path: firing rates的.npy文件路径
            
        Returns:
            firing_rates: (num_segments_total, full_time_duration) numpy数组
        """
        print(f"正在加载初始firing rates: {firing_rates_path}")
        try:
            firing_rates = np.load(firing_rates_path)
            print(f"  文件加载成功，形状: {firing_rates.shape}")
            print(f"  数据类型: {firing_rates.dtype}")
            print(f"  数值范围: [{np.min(firing_rates):.6f}, {np.max(firing_rates):.6f}]")
            
            # 验证数据格式
            if len(firing_rates.shape) != 2:
                raise ValueError(f"期望2D数组，但得到{len(firing_rates.shape)}D数组")
            
            num_segments, full_time = firing_rates.shape
            if num_segments != self.num_segments_total:
                print(f"警告：文件中的segments数量 ({num_segments}) 与模型期望不符 ({self.num_segments_total})")
            
            return firing_rates.astype(np.float32)
            
        except Exception as e:
            print(f"加载初始firing rates时出错: {e}")
            return None
    
    def prepare_firing_rates_for_optimization(self, firing_rates, batch_size=1, start_time_ms=100):
        """
        准备firing rates用于优化，提取后300ms数据（忽略前100ms）
        
        Args:
            firing_rates: (num_segments_total, full_time_duration) 或 (batch_size, num_segments_total, full_time_duration)
            batch_size: 批次大小
            start_time_ms: 开始时间，默认100ms（忽略前100ms）
            
        Returns:
            prepared_rates: (batch_size, num_segments_total, time_duration_ms)
        """
        if len(firing_rates.shape) == 2:
            # 单个样本，扩展为batch
            firing_rates = firing_rates[np.newaxis, :, :]  # (1, num_segments, full_time)
        
        current_batch_size, num_segments, full_time = firing_rates.shape
        
        # 处理segments数量不匹配的问题
        if num_segments != self.num_segments_total:
            print(f"调整segments数量: {num_segments} -> {self.num_segments_total}")
            if num_segments < self.num_segments_total:
                # 如果segments不足，用零填充
                padding_needed = self.num_segments_total - num_segments
                padding = np.zeros((current_batch_size, padding_needed, full_time), dtype=firing_rates.dtype)
                firing_rates = np.concatenate([firing_rates, padding], axis=1)
                print(f"  添加了 {padding_needed} 个零填充segments")
            else:
                # 如果segments过多，截取前面的部分
                firing_rates = firing_rates[:, :self.num_segments_total, :]
                print(f"  截取前 {self.num_segments_total} 个segments")
        
        # 计算提取的时间范围
        end_time_ms = start_time_ms + self.time_duration_ms
        
        if end_time_ms > full_time:
            raise ValueError(f"时间范围超出数据长度：需要{end_time_ms}ms，但数据只有{full_time}ms")
        
        # 提取指定时间段的数据
        extracted_rates = firing_rates[:, :, start_time_ms:end_time_ms]
        
        # 如果需要更多batch，复制数据
        if batch_size > current_batch_size:
            # 重复数据以达到所需的batch size
            repeat_times = batch_size // current_batch_size
            remainder = batch_size % current_batch_size
            
            repeated_rates = np.tile(extracted_rates, (repeat_times, 1, 1))
            if remainder > 0:
                extra_rates = extracted_rates[:remainder, :, :]
                extracted_rates = np.concatenate([repeated_rates, extra_rates], axis=0)
            else:
                extracted_rates = repeated_rates
        elif batch_size < current_batch_size:
            # 取前batch_size个样本
            extracted_rates = extracted_rates[:batch_size, :, :]
        
        print(f"Prepare for optimization:")
        print(f"  Original input shape: {firing_rates.shape}")
        print(f"  Extracted time: {start_time_ms}-{end_time_ms}ms")
        print(f"  Extracted input shape: {extracted_rates.shape}")
        
        return extracted_rates.astype(np.float32)
    
    def generate_background_firing_rates(self, batch_size=1):
        """
        生成背景firing rates (泊松过程)
        
        Args:
            batch_size: 批次大小
            
        Returns:
            firing_rates: (batch_size, time_duration_ms, num_segments_total)
        """
        # 为兴奋性和抑制性segments设置不同的背景firing rate
        exc_rate = 0.01  # 兴奋性背景firing rate (1%)
        inh_rate = 0.02  # 抑制性背景firing rate (2%)
        
        # 生成firing rates
        firing_rates = np.zeros((batch_size, self.time_duration_ms, self.num_segments_total))
        
        # 兴奋性segments
        firing_rates[:, :, :self.num_segments_exc] = np.random.uniform(
            0.005, 0.02, (batch_size, self.time_duration_ms, self.num_segments_exc)
        )
        
        # 抑制性segments
        firing_rates[:, :, self.num_segments_exc:] = np.random.uniform(
            0.01, 0.03, (batch_size, self.time_duration_ms, self.num_segments_inh)
        )
        
        return firing_rates.astype(np.float32)
    
    def generate_spikes_with_modification(self, firing_rates, fixed_exc_indices=None):
        """
        生成spikes with modification - 可复用函数
        
        Args:
            firing_rates: (batch_size, num_segments_total, time_duration_ms)
            fixed_exc_indices: 固定添加spikes的excitatory indices
            
        Returns:
            spike_trains: (batch_size, num_segments_total, time_duration_ms) binary
            fixed_exc_indices: 使用的fixed indices
        """
        batch_size = firing_rates.shape[0]
        
        # 只对前一半batch使用Poisson过程生成spikes
        first_half_spikes = np.random.poisson(firing_rates).astype(np.float32)
        first_half_spikes = np.clip(first_half_spikes, 0, 1)  # 限制为0或1 (binary)
        
        # 复制前一半到后一半，确保基础spikes完全相同
        second_half_spikes = first_half_spikes.copy()
        
        # 组合完整的spike trains
        spike_trains = np.concatenate([first_half_spikes, second_half_spikes], axis=0)
        
        # 如果没有指定固定的excitatory indices，随机选择三个
        if fixed_exc_indices is None:
            mono_syn_rnd = np.random.default_rng(42)
            fixed_exc_indices = mono_syn_rnd.choice(self.num_segments_exc, size=3, replace=False)
        
        # 对后一半batch添加固定spikes (在half window size处)
        if batch_size > 0:
            spike_time_mono_syn = self.input_window_size // 2  # half window size时间点
            if spike_time_mono_syn < self.time_duration_ms:
                # 在half window size处为指定的三个excitatory segments添加spikes
                for idx in fixed_exc_indices:
                    spike_trains[batch_size:, idx, spike_time_mono_syn] = 1.0
        
        return spike_trains, fixed_exc_indices
    
    def process_firing_rates_to_predictions(self, firing_rates, fixed_exc_indices):
        """
        可复用函数：将firing rates转换为模型预测
        
        Args:
            firing_rates: (batch_size, num_segments_total, time_duration_ms)
            fixed_exc_indices: 固定的excitatory indices
            
        Returns:
            spike_predictions: 模型预测的spike概率
            spike_trains: 生成的spike trains
        """
        # 使用可复用函数生成spikes
        spike_trains, _ = self.generate_spikes_with_modification(firing_rates, fixed_exc_indices)
        
        # 转换数据格式以匹配模型输入: (spike_batch_size, time_duration, num_segments)
        model_input = np.transpose(spike_trains, (0, 2, 1))
        
        # 处理输入窗口大小
        input_time_steps = model_input.shape[1]
        if input_time_steps < self.input_window_size:
            padding_needed = self.input_window_size - input_time_steps
            padding = np.zeros((model_input.shape[0], padding_needed, model_input.shape[2]))
            model_input = np.concatenate([padding, model_input], axis=1)
        elif input_time_steps > self.input_window_size:
            model_input = model_input[:, -self.input_window_size:, :]
        
        # 模型预测
        predictions = self.model.predict(model_input, verbose=0)
        spike_predictions = predictions[0]  # 第一个输出是spike预测
        
        return spike_predictions, spike_trains
    
    def get_model_info(self):
        """
        获取模型信息
        
        Returns:
            dict: 包含模型基本信息的字典
        """
        return {
            'input_window_size': self.input_window_size,
            'num_segments_exc': self.num_segments_exc,
            'num_segments_inh': self.num_segments_inh,
            'num_segments_total': self.num_segments_total,
            'time_duration_ms': self.time_duration_ms,
            'model_path': self.model_path
        }
