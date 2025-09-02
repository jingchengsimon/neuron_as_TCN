import numpy as np
import glob
import time
import os
import pickle
from datetime import datetime  
from itertools import product
# from fit_CNN_pytorch import train_and_save_pytorch
from utils.fit_CNN import create_temporaly_convolutional_model, SimulationDataGenerator
from utils.model_analysis import (
    load_model_results, print_model_summary, 
    plot_training_curves, plot_model_comparison, analyze_training_stability, plot_auc_analysis
)
import tensorflow as tf
from keras.optimizers import Nadam
from keras.callbacks import LearningRateScheduler

# 添加GPU监控库
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    print("Warning: pynvml not available. Install with: pip install nvidia-ml-py")
    GPU_MONITORING_AVAILABLE = False

class GPUMonitor:
    """GPU监控类，用于监控GPU使用情况和性能"""
    
    def __init__(self, gpu_index=0):
        """
        初始化GPU监控器
        
        Args:
            gpu_index: GPU设备索引，默认为0
        """
        self.gpu_index = gpu_index
        self.available = GPU_MONITORING_AVAILABLE
        self.handle = None
        
        if self.available:
            try:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                gpu_name_bytes = pynvml.nvmlDeviceGetName(self.handle)
                # 处理不同版本的pynvml返回格式
                if isinstance(gpu_name_bytes, bytes):
                    self.gpu_name = gpu_name_bytes.decode('utf-8')
                else:
                    self.gpu_name = str(gpu_name_bytes)
                self.total_memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle).total
            except Exception as e:
                print(f"Error initializing GPU monitor: {e}")
                self.available = False
    
    def get_utilization(self):
        """获取GPU利用率"""
        if not self.available or not self.handle:
            return "N/A"
        
        try:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            return utilization.gpu
        except:
            return "Error"
    
    def get_memory_usage(self):
        """获取GPU内存使用情况"""
        if not self.available or not self.handle:
            return "N/A"
        
        try:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            used_mb = memory_info.used / 1024 / 1024
            total_mb = memory_info.total / 1024 / 1024
            return f"{used_mb:.0f}/{total_mb:.0f} MB"
        except:
            return "Error"
    
    def get_memory_percent(self):
        """获取GPU内存使用百分比"""
        if not self.available or not self.handle:
            return "N/A"
        
        try:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return (memory_info.used / memory_info.total) * 100
        except:
            return "Error"
    
    def get_temperature(self):
        """获取GPU温度"""
        if not self.available or not self.handle:
            return "N/A"
        
        try:
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            return temp
        except:
            return "Error"
    
    def get_power_usage(self):
        """获取GPU功耗"""
        if not self.available or not self.handle:
            return "N/A"
        
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # 转换为瓦特
            return power
        except:
            return "Error"
    
    def get_comprehensive_info(self):
        """获取GPU综合信息"""
        if not self.available:
            return {
                'name': 'N/A',
                'utilization': 'N/A',
                'memory_usage': 'N/A',
                'memory_percent': 'N/A',
                'temperature': 'N/A',
                'power': 'N/A'
            }
        
        return {
            'name': self.gpu_name if hasattr(self, 'gpu_name') else 'N/A',
            'utilization': self.get_utilization(),
            'memory_usage': self.get_memory_usage(),
            'memory_percent': self.get_memory_percent(),
            'temperature': self.get_temperature(),
            'power': self.get_power_usage()
        }
    
    def print_status(self, prefix=""):
        """打印当前GPU状态"""
        info = self.get_comprehensive_info()
        print(f"{prefix}GPU: {info['name']}")
        print(f"{prefix}利用率: {info['utilization']}%")
        print(f"{prefix}内存: {info['memory_usage']} ({info['memory_percent']:.1f}%)")
        print(f"{prefix}温度: {info['temperature']}°C")
        print(f"{prefix}功耗: {info['power']}W")
    
    def monitor_continuously(self, duration_seconds=300, interval_seconds=5, output_file=None):
        """
        持续监控GPU使用情况
        
        Args:
            duration_seconds: 监控持续时间（秒）
            interval_seconds: 监控间隔（秒）
            output_file: 输出文件路径（可选）
        """
        if not self.available:
            print("GPU monitoring not available")
            return
        
        print(f"\n开始GPU监控 ({duration_seconds}秒, 每{interval_seconds}秒记录一次):")
        print("时间戳           | GPU利用率(%) | 内存使用(MB) | 内存使用率(%) | 温度(°C) | 功耗(W)")
        print("-" * 90)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write("timestamp,utilization,memory_used,memory_percent,temperature,power\n")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        try:
            while time.time() < end_time:
                elapsed = time.time() - start_time
                info = self.get_comprehensive_info()
                
                timestamp = time.strftime("%H:%M:%S")
                line = f"{timestamp} | {info['utilization']:11} | {info['memory_usage']:10} | {info['memory_percent']:12.1f} | {info['temperature']:8} | {info['power']:6}"
                print(line)
                
                if output_file:
                    with open(output_file, 'a') as f:
                        f.write(f"{timestamp},{info['utilization']},{info['memory_usage']},{info['memory_percent']},{info['temperature']},{info['power']}\n")
                
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\n监控已停止")
        
        print("-" * 90)
        print("GPU监控完成")
    
    def benchmark_performance(self, test_duration=60):
        """
        性能基准测试
        
        Args:
            test_duration: 测试持续时间（秒）
        """
        if not self.available:
            print("GPU monitoring not available for benchmarking")
            return
        
        print(f"\n开始GPU性能基准测试 ({test_duration}秒)...")
        
        # 收集性能数据
        utilizations = []
        memory_percents = []
        temperatures = []
        power_usages = []
        
        start_time = time.time()
        end_time = start_time + test_duration
        
        while time.time() < end_time:
            info = self.get_comprehensive_info()
            
            if info['utilization'] != 'N/A' and info['utilization'] != 'Error':
                utilizations.append(info['utilization'])
            if info['memory_percent'] != 'N/A' and info['memory_percent'] != 'Error':
                memory_percents.append(info['memory_percent'])
            if info['temperature'] != 'N/A' and info['temperature'] != 'Error':
                temperatures.append(info['temperature'])
            if info['power'] != 'N/A' and info['power'] != 'Error':
                power_usages.append(info['power'])
            
            time.sleep(1)
        
        # 计算统计信息
        print("\n=== 性能基准测试结果 ===")
        if utilizations:
            print(f"GPU利用率 - 平均: {np.mean(utilizations):.1f}%, 最大: {np.max(utilizations):.1f}%, 最小: {np.min(utilizations):.1f}%")
        if memory_percents:
            print(f"内存使用率 - 平均: {np.mean(memory_percents):.1f}%, 最大: {np.max(memory_percents):.1f}%, 最小: {np.min(memory_percents):.1f}%")
        if temperatures:
            print(f"GPU温度 - 平均: {np.mean(temperatures):.1f}°C, 最大: {np.max(temperatures):.1f}°C, 最小: {np.min(temperatures):.1f}°C")
        if power_usages:
            print(f"GPU功耗 - 平均: {np.mean(power_usages):.1f}W, 最大: {np.max(power_usages):.1f}W, 最小: {np.min(power_usages):.1f}W")
        
        # 性能评估
        avg_utilization = np.mean(utilizations) if utilizations else 0
        if avg_utilization > 80:
            print("✓ GPU利用率优秀 (>80%)")
        elif avg_utilization > 50:
            print("○ GPU利用率良好 (50-80%)")
        elif avg_utilization > 20:
            print("⚠ GPU利用率偏低 (20-50%)")
        else:
            print("✗ GPU利用率过低 (<20%)")
        
        print("========================")

def lr_warmup_decay(epoch, lr):
    warmup_epochs = 10
    init_lr = 0.0001
    max_lr = 0.001
    decay_rate = 0.95
    if epoch < warmup_epochs:
        # 线性warmup
        return init_lr + (max_lr - init_lr) * (epoch + 1) / warmup_epochs
    else:
        # 衰减
        return max_lr * (decay_rate ** (epoch - warmup_epochs))

def train_and_save(network_depth, num_filters_per_layer, input_window_size, num_epochs, 
                   train_data_dir, valid_data_dir, test_data_dir, models_dir,
                   use_improved_initialization=False, use_improved_sampling=False, spike_rich_ratio=0.5):

    # ========== GPU验证代码 - 添加在这里 ==========
    import tensorflow as tf
    print("Training device check:")
    print("TensorFlow built with CUDA:", tf.test.is_built_with_cuda())
    print("GPU devices:", tf.config.list_physical_devices('GPU'))
    # ===========================================

    use_multiprocessing = True  # 关闭多进程以避免内存问题
    num_workers = 4  # 减少worker数量

    print('------------------------------------------------------------------')
    print('use_multiprocessing = %s, num_workers = %d' %(str(use_multiprocessing), num_workers))
    print('------------------------------------------------------------------')

    # ------------------------------------------------------------------
    # basic configurations and directories
    # ------------------------------------------------------------------
    synapse_type = 'NMDA'
        
    # ------------------------------------------------------------------
    # learning schedule params
    # ------------------------------------------------------------------

    # Predefined hyper parameters
    num_filters_per_layer = [num_filters_per_layer] * network_depth # 64

    validation_fraction = 0.5
    train_file_load = 0.2  # 0.3 # 减少文件加载比例
    valid_file_load = 0.2  # 0.3 # 减少文件加载比例
    num_steps_multiplier = 10

    train_files_per_epoch = 6 # 4  # 减少训练文件数量
    valid_files_per_epoch = 2 #max(1, int(validation_fraction * train_files_per_epoch))

    batch_size_per_epoch        = [8] * num_epochs   # 16 # 减少批次大小从32到16
    learning_rate_per_epoch     = [0.0001] * len(batch_size_per_epoch)
    loss_weights_per_epoch      = [[1.0, 0.01]] * num_epochs # [[1.0, 0.0200]] * num_epochs # Even higher spike weight
    num_train_steps_per_epoch   = [100] * num_epochs  # 减少训练步数

    for i in range(40,num_epochs):
        batch_size_per_epoch[i]    = 8
        learning_rate_per_epoch[i] = 0.00003
        loss_weights_per_epoch[i]  = [2.0, 0.005] # [2.0, 0.0100]
        
    for i in range(80,num_epochs):
        batch_size_per_epoch[i]    = 8
        learning_rate_per_epoch[i] = 0.00001
        loss_weights_per_epoch[i]  = [3.0, 0.002] # [4.0, 0.0100]

    for i in range(120,num_epochs):
        batch_size_per_epoch[i]    = 8
        learning_rate_per_epoch[i] = 0.000003
        loss_weights_per_epoch[i]  = [4.0, 0.001] # [8.0, 0.0100]

    for i in range(160,num_epochs):
        batch_size_per_epoch[i]    = 8
        learning_rate_per_epoch[i] = 0.000001
        loss_weights_per_epoch[i]  = [5.0, 0.001] # [9.0, 0.0030]

    learning_schedule_dict = {}
    learning_schedule_dict['train_file_load']           = train_file_load
    learning_schedule_dict['valid_file_load']           = valid_file_load
    learning_schedule_dict['validation_fraction']       = validation_fraction
    learning_schedule_dict['num_epochs']                = num_epochs
    learning_schedule_dict['num_steps_multiplier']      = num_steps_multiplier
    learning_schedule_dict['batch_size_per_epoch']      = batch_size_per_epoch
    learning_schedule_dict['loss_weights_per_epoch']    = loss_weights_per_epoch
    learning_schedule_dict['learning_rate_per_epoch']   = learning_rate_per_epoch
    learning_schedule_dict['num_train_steps_per_epoch'] = num_train_steps_per_epoch

    # ------------------------------------------------------------------
    # define network architecture params
    # ------------------------------------------------------------------

    num_segments_exc  = 639 # 10042 + 16070
    if 'SJC' in train_data_dir:
        num_segments_inh  = 640 # 1023 + 1637 + 150 
    else:
        num_segments_inh  = 639 # 1023 + 1637 + 150 

    filter_sizes_per_layer = [54] + [24] * (network_depth - 1)
    initializer_per_layer         = [0.002] * network_depth
    activation_function_per_layer = ['relu'] * network_depth
    l2_regularization_per_layer   = [1e-6] * network_depth # 1e-8
    strides_per_layer             = [1] * network_depth
    dilation_rates_per_layer      = [1] * network_depth


    architecture_dict = {}
    architecture_dict['network_depth']                 = network_depth
    architecture_dict['input_window_size']             = input_window_size
    architecture_dict['num_filters_per_layer']         = num_filters_per_layer
    architecture_dict['initializer_per_layer']         = initializer_per_layer
    architecture_dict['filter_sizes_per_layer']        = filter_sizes_per_layer
    architecture_dict['l2_regularization_per_layer']   = l2_regularization_per_layer
    architecture_dict['activation_function_per_layer'] = activation_function_per_layer
    architecture_dict['strides_per_layer']             = strides_per_layer
    architecture_dict['dilation_rates_per_layer']      = dilation_rates_per_layer

    print('L2 regularization = %.9f' %(l2_regularization_per_layer[0]))
    print('activation function = "%s"' %(activation_function_per_layer[0]))
    
    print('-----------------------------------------------')
    print('finding data')
    print('-----------------------------------------------')

    train_files = glob.glob(train_data_dir + '*.p')
    valid_files = glob.glob(valid_data_dir + '*.p')
    test_files  = glob.glob(test_data_dir  + '*.p')

    data_dict = {}
    data_dict['train_files'] = train_files
    data_dict['valid_files'] = valid_files
    data_dict['test_files']  = test_files

    print('number of training files is %d' %(len(train_files)))
    print('number of validation files is %d' %(len(valid_files)))
    print('number of test files is %d' %(len(test_files)))
    print('-----------------------------------------------')

    
    assert(input_window_size > sum(filter_sizes_per_layer))
    temporal_conv_net = create_temporaly_convolutional_model(input_window_size, num_segments_exc, num_segments_inh, 
                                                            filter_sizes_per_layer, num_filters_per_layer,
                                                            activation_function_per_layer, l2_regularization_per_layer,
                                                            strides_per_layer, dilation_rates_per_layer, initializer_per_layer,
                                                            use_improved_initialization=use_improved_initialization)

    is_fully_connected = (network_depth == 1) or sum(filter_sizes_per_layer[1:]) == (network_depth -1)
    if is_fully_connected:
        model_prefix = '%s_FCN' %(synapse_type)
    else:
        model_prefix = '%s_TCN' %(synapse_type)
    network_average_width = int(np.array(num_filters_per_layer).mean())
    time_window_T = (np.array(filter_sizes_per_layer) - 1).sum() + 1
    architecture_overview = 'DxWxT_%dx%dx%d' %(network_depth,network_average_width,time_window_T)
    start_learning_schedule = 0
    num_training_samples = 0


    print('-----------------------------------------------')
    print('about to start training...')
    print('-----------------------------------------------')
    print(model_prefix)
    print(architecture_overview)
    print('-----------------------------------------------')

    
    num_learning_schedules = num_epochs # len(batch_size_per_epoch) # 8

    training_history_dict = {}
    for learning_schedule in range(start_learning_schedule, num_learning_schedules): # range(0, 8)
        epoch_start_time = time.time()
            
        batch_size    = batch_size_per_epoch[learning_schedule]
        learning_rate = learning_rate_per_epoch[learning_schedule]
        loss_weights  = loss_weights_per_epoch[learning_schedule]
        
        # prepare data generators
        if learning_schedule == 0 or (learning_schedule >= 1 and batch_size != batch_size_per_epoch[learning_schedule -1]):
            print('initializing generators')
            train_data_generator = SimulationDataGenerator(
                train_files, num_files_per_epoch=train_files_per_epoch, batch_size=batch_size, 
                window_size_ms=input_window_size, file_load=train_file_load,
                use_improved_sampling=use_improved_sampling, spike_rich_ratio=spike_rich_ratio)
            valid_data_generator = SimulationDataGenerator(
                valid_files, num_files_per_epoch=valid_files_per_epoch, batch_size=batch_size,
                window_size_ms=input_window_size, file_load=valid_file_load,
                use_improved_sampling=use_improved_sampling, spike_rich_ratio=spike_rich_ratio)
    
        train_steps_per_epoch = len(train_data_generator)
        
        # 优化器只初始化一次，学习率由callback动态调整
        if learning_schedule == 0:
            optimizer_to_use = Nadam(lr=learning_rate)
            temporal_conv_net.compile(optimizer=optimizer_to_use, loss=['binary_crossentropy','mse'], loss_weights=loss_weights)
        
        print('-----------------------------------------------')
        print('starting epoch %d:' %(learning_schedule))
        print('-----------------------------------------------')
        print('loss weights = %s' %(str(loss_weights)))
        # print('learning_rate = %.7f' %(learning_rate))
        print('batch_size = %d' %(batch_size))
        print('-----------------------------------------------')
        
        # 在训练循环中添加时间监控
        start_time = time.time()
        
        # 创建GPU监控器
        gpu_monitor = GPUMonitor()
        
        # 训练前检查GPU状态
        print("训练前GPU状态:")
        gpu_monitor.print_status("  ")
        
        # 可选：在后台启动GPU监控（注释掉以避免干扰训练输出）
        # import threading
        # monitor_thread = threading.Thread(target=gpu_monitor.monitor_continuously, args=(300, 10))
        # monitor_thread.daemon = True
        # monitor_thread.start()

        lr_scheduler = LearningRateScheduler(lr_warmup_decay, verbose=1)
        history = temporal_conv_net.fit_generator(generator=train_data_generator,
                                                epochs=num_steps_multiplier,
                                                validation_data=valid_data_generator,
                                                use_multiprocessing=use_multiprocessing,  # 关闭多进程以确保进度条正常显示
                                                workers=num_workers,  # 减少worker数量
                                                verbose=1,  # 显示详细进度条
                                                callbacks=[lr_scheduler])

        training_time = time.time() - start_time
        print(f"Training time: {training_time:.2f}s")
        print("训练后GPU状态:")
        gpu_monitor.print_status("  ")
        
        # store the loss values in training histogry dictionary and add some additional fields about the training schedule
        try:
            for key in history.history.keys():
                training_history_dict[key] += history.history[key]
            training_history_dict['learning_schedule'] += [learning_schedule] * num_steps_multiplier
            training_history_dict['batch_size']        += [batch_size] * num_steps_multiplier
            training_history_dict['learning_rate']     += [learning_rate] * num_steps_multiplier
            # training_history_dict['learning_rate']     += [optimizer_to_use.get_config()['learning_rate']] * num_steps_multiplier # 使用优化器获取当前学习率
            training_history_dict['loss_weights']      += [loss_weights] * num_steps_multiplier
            training_history_dict['num_train_samples'] += [batch_size * train_steps_per_epoch] * num_steps_multiplier
            training_history_dict['num_train_steps']   += [train_steps_per_epoch] * num_steps_multiplier
            training_history_dict['train_files_histogram'] += [train_data_generator.batches_per_file_dict]
            training_history_dict['valid_files_histogram'] += [valid_data_generator.batches_per_file_dict]
        except:
            for key in history.history.keys():
                training_history_dict[key] = history.history[key]
            training_history_dict['learning_schedule'] = [learning_schedule] * num_steps_multiplier
            training_history_dict['batch_size']        = [batch_size] * num_steps_multiplier
            training_history_dict['learning_rate']     = [learning_rate] * num_steps_multiplier
            # training_history_dict['learning_rate']     = [optimizer_to_use.get_config()['learning_rate']] * num_steps_multiplier # 使用优化器获取当前学习率
            training_history_dict['loss_weights']      = [loss_weights] * num_steps_multiplier
            training_history_dict['num_train_samples'] = [batch_size * train_steps_per_epoch] * num_steps_multiplier
            training_history_dict['num_train_steps']   = [train_steps_per_epoch] * num_steps_multiplier
            training_history_dict['train_files_histogram'] = [train_data_generator.batches_per_file_dict]
            training_history_dict['valid_files_histogram'] = [valid_data_generator.batches_per_file_dict]

        num_training_samples = num_training_samples + num_steps_multiplier * train_steps_per_epoch * batch_size
        
        print('-----------------------------------------------------------------------------------------')
        epoch_duration_sec = time.time() - epoch_start_time
        print('total time it took to calculate epoch was %.3f seconds (%.3f batches/second)' %(epoch_duration_sec, float(train_steps_per_epoch * num_steps_multiplier) / epoch_duration_sec))
        print('-----------------------------------------------------------------------------------------')
        
        # save model every once and a while
        if np.array(training_history_dict['val_spikes_loss'][-3:]).mean() < 0.03:
            model_ID = np.random.randint(100000)
            modelID_str = 'ID_%d' %(model_ID)
            train_string = 'samples_%d' %(num_training_samples)
            if len(training_history_dict['val_spikes_loss']) >= 10:
                train_MSE = 10000 * np.array(training_history_dict['spikes_loss'][-7:]).mean()
                valid_MSE = 10000 * np.array(training_history_dict['val_spikes_loss'][-7:]).mean()
            else:
                train_MSE = 10000 * np.array(training_history_dict['spikes_loss']).mean()
                valid_MSE = 10000 * np.array(training_history_dict['val_spikes_loss']).mean()
                
            results_overview = 'LogLoss_train_%d_valid_%d' %(train_MSE,valid_MSE)
            current_datetime = str(datetime.now())[:-10].replace(':','_').replace(' ','__')
            model_filename    = models_dir + '%s__%s__%s__%s__%s__%s.h5' %(model_prefix,architecture_overview,current_datetime,train_string,results_overview,modelID_str)
            auxilary_filename = models_dir + '%s__%s__%s__%s__%s__%s.pickle' %(model_prefix,architecture_overview,current_datetime,train_string,results_overview,modelID_str)

            print('-----------------------------------------------------------------------------------------')
            print('finished epoch %d/%d. saving...\n     "%s"\n     "%s"' %(learning_schedule +1, num_epochs, model_filename.split('/')[-1], auxilary_filename.split('/')[-1]))
            print('-----------------------------------------------------------------------------------------')

            temporal_conv_net.save(model_filename)
            
            # save all relevent training params (in raw and unprocessed way)
            model_hyperparams_and_training_dict = {}
            model_hyperparams_and_training_dict['data_dict']              = data_dict
            model_hyperparams_and_training_dict['architecture_dict']      = architecture_dict
            model_hyperparams_and_training_dict['learning_schedule_dict'] = learning_schedule_dict
            model_hyperparams_and_training_dict['training_history_dict']  = training_history_dict
            
            pickle.dump(model_hyperparams_and_training_dict, open(auxilary_filename, "wb"), protocol=2)

def analyze_and_save(models_dir, test_data_dir, save_dir):

    """
    运行完整的模型分析，包括AUC评估
    """
    print("Loading model results...")
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

def main():

    # ========== GPU配置代码 - 添加在这里 ==========
    
    # 检测GPU
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # 配置GPU内存自增长，避免显存不足
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(e)

    # 可选：指定使用特定GPU（如果有多块GPU）
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第0号GPU
    # ===========================================
    
    # GPU状态诊断
    print("\n=== GPU状态诊断 ===")
    print(f"TensorFlow版本: {tf.__version__}")
    print(f"CUDA可用: {tf.test.is_built_with_cuda()}")
    print(f"GPU设备数量: {len(tf.config.list_physical_devices('GPU'))}")
    
    # 使用GPU监控类
    gpu_monitor = GPUMonitor()
    if gpu_monitor.available:
        print("GPU监控器初始化成功")
        gpu_monitor.print_status("  ")
    else:
        print("GPU监控不可用，请安装: pip install nvidia-ml-py")
    print("==================\n")

    # 1. 定义超参数网格
    network_depth_list = [1]
    num_filters_per_layer_list = [128]  # 其它参数可固定或自行调整
    input_window_size_list = [400]  # 这里遍历不同的input_window_size

    num_epochs = 250

    # 2. 主控循环

    # 配置改进选项
    use_improved_initialization = False   # 设置为True启用改进的初始化策略
    use_improved_sampling = True        # 设置为True启用改进的数据采样策略
    spike_rich_ratio = 0.6              # 60%的样本包含spike
    
    print(f"\n=== 改进配置 ===")
    print(f"改进初始化策略: {'启用' if use_improved_initialization else '禁用'}")
    print(f"改进数据采样: {'启用' if use_improved_sampling else '禁用'}")
    if use_improved_sampling:
        print(f"Spike-rich样本比例: {spike_rich_ratio * 100:.0f}%")
    print(f"================\n")

    def build_analysis_suffix(base_path, model_suffix):
        """
        动态构建analysis suffix
        从base path中提取'InOut'后的部分，从model suffix中提取下划线后的部分
        """
        # 从base path中提取'InOut'后的部分
        if 'InOut' in base_path:
            inout_part = base_path.split('InOut')[-1]  # 获取'InOut'后的部分
            # 如果结果以下划线开头，去掉开头的下划线
            if inout_part.startswith('_'):
                inout_part = inout_part[1:]
        else:
            inout_part = 'original' # base_path.split('/')[-1]  # 如果没有'InOut'，取最后一部分
        
        # 从model suffix中提取下划线后的部分
        if '_' in model_suffix:
            model_part = model_suffix.split('_', 1)[1]  # 获取第一个下划线后的部分
        else:
            model_part = model_suffix
        
        # 组合成analysis suffix
        analysis_suffix = f"{inout_part}_{model_part}"
        return analysis_suffix

    for network_depth, num_filters_per_layer, input_window_size in product(network_depth_list, num_filters_per_layer_list, input_window_size_list):

        # 基础配置
        test_suffix = '_SJC_funcgroup2_var2_AMPA'
        base_path = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut' + test_suffix
        data_suffix = 'L5PC_AMPA'
        model_suffix = 'AMPA_fullStrategy'
        
        # 动态构建analysis suffix
        analysis_suffix = build_analysis_suffix(base_path, model_suffix)
    
        # 数据目录
        train_data_dir = f'{base_path}/data/{data_suffix}_train/'
        valid_data_dir = f'{base_path}/data/{data_suffix}_valid/'
        test_data_dir = f'{base_path}/data/{data_suffix}_test/'
        
        # 模型和分析目录
        model_dir = f'{base_path}/models/{model_suffix}/depth_{network_depth}_filters_{num_filters_per_layer}_window_{input_window_size}/'
        analysis_dir = f'./results/3_model_analysis_plots/{analysis_suffix}/depth_{network_depth}_filters_{num_filters_per_layer}_window_{input_window_size}/'
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(analysis_dir, exist_ok=True)

        print(f"\n==============================")
        print(f"开始训练: network_depth={network_depth}, num_filters_per_layer={num_filters_per_layer}, input_window_size={input_window_size}")
        print(f"模型保存目录: {model_dir}")
        print(f"分析结果目录: {analysis_dir}")
        print(f"==============================\n")
        
        # 3. 训练模型
        train_and_save(
            network_depth=network_depth,
            num_filters_per_layer=num_filters_per_layer,
            input_window_size=input_window_size,
            num_epochs=num_epochs,
            train_data_dir=train_data_dir,
            valid_data_dir=valid_data_dir,
            test_data_dir=test_data_dir,
            models_dir=model_dir,
            use_improved_initialization=use_improved_initialization,
            use_improved_sampling=use_improved_sampling,
            spike_rich_ratio=spike_rich_ratio
        )

        # train_and_save_pytorch(
        #     network_depth=network_depth,
        #     num_filters_per_layer=num_filters_per_layer,
        #     input_window_size=input_window_size,
        #     num_epochs=num_epochs,
        #     models_dir=model_dir
        # )

        # 4. 分析模型
        analyze_and_save(
            models_dir=model_dir,
            test_data_dir=test_data_dir,
            save_dir=analysis_dir
        )

    print("\n所有超参数组合训练与分析已完成！") 

if __name__ == "__main__":
    main()