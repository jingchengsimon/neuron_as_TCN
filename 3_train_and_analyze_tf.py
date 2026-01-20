import numpy as np
import glob
import time
import os
import pickle
import argparse
from datetime import datetime  
from itertools import product
# from fit_CNN_pytorch import train_and_save_pytorch
from utils.gpu_monitor import GPUMonitor
from utils.fit_CNN_tf import create_temporaly_convolutional_model, SimulationDataGenerator
from utils.model_analysis import (
    load_model_results, print_model_summary, 
    plot_training_curves, plot_model_comparison, analyze_training_stability, plot_auc_analysis
)
from utils.model_size_utils import get_model_size_info, analyze_model_size
import tensorflow as tf
from keras.optimizers import Nadam
from keras.callbacks import LearningRateScheduler, Callback
from tqdm import tqdm

# Import build_analysis_suffix from torch version to avoid code duplication
import importlib.util
spec = importlib.util.spec_from_file_location("train_torch", "3_train_and_analyze_torch.py")
train_torch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_torch)
build_analysis_suffix = train_torch.build_analysis_suffix

# Global variables for storing dynamic learning rates (will be set after model analysis)
_dynamic_init_lr = 0.0001
_dynamic_max_lr = 0.001

def set_warmup_lr(init_lr, max_lr):
    """Set learning rate parameters for warmup function"""
    global _dynamic_init_lr, _dynamic_max_lr
    _dynamic_init_lr = init_lr
    _dynamic_max_lr = max_lr

def lr_warmup_decay(epoch, lr):
    """Learning rate warmup and decay scheduling (using dynamically set learning rates)"""
    warmup_epochs = 10
    init_lr = _dynamic_init_lr
    max_lr = _dynamic_max_lr
    decay_rate = 0.95
    if epoch < warmup_epochs:
        # Linear warmup
        return init_lr + (max_lr - init_lr) * (epoch + 1) / warmup_epochs
    else:
        # Decay
        return max_lr * (decay_rate ** (epoch - warmup_epochs))

class TqdmProgressBar(Callback):
    """Custom tqdm progress bar Callback, displaying progress bar similar to PyTorch version"""
    
    def __init__(self, learning_schedule, num_epochs, num_steps_multiplier, train_steps_per_epoch, valid_steps=None):
        super().__init__()
        self.learning_schedule = learning_schedule
        self.num_epochs = num_epochs
        self.num_steps_multiplier = num_steps_multiplier
        self.train_steps_per_epoch = train_steps_per_epoch
        self.valid_steps = valid_steps
        self.train_pbar = None
        self.valid_pbar = None
        self.current_epoch = 0
        
    def on_epoch_begin(self, epoch, logs=None):
        """Create new training progress bar at the start of each epoch"""
        self.current_epoch = epoch
        # Calculate which mini_epoch this is (starting from 1)
        mini_epoch_num = epoch + 1
        desc = f"Train {self.learning_schedule+1}/{self.num_epochs} epoch {mini_epoch_num}/{self.num_steps_multiplier}"
        self.train_pbar = tqdm(total=self.train_steps_per_epoch, desc=desc, leave=False, ncols=100)
        
    def on_batch_end(self, batch, logs=None):
        """Update training progress bar at the end of each batch"""
        if self.train_pbar is not None:
            # Update progress bar, showing current loss information
            loss_info = ""
            if logs:
                if 'loss' in logs:
                    loss_info = f"loss={logs['loss']:.4f}"
                elif 'spikes_loss' in logs:
                    loss_info = f"spike_loss={logs['spikes_loss']:.4f}"
            self.train_pbar.set_postfix_str(loss_info)
            self.train_pbar.update(1)
    
    def on_test_begin(self, logs=None):
        """Create validation progress bar at the start of validation"""
        if self.valid_steps is not None:
            self.valid_pbar = tqdm(total=self.valid_steps, desc="Valid", leave=False, ncols=100)
    
    def on_test_batch_end(self, batch, logs=None):
        """Update validation progress bar at the end of each validation batch"""
        if self.valid_pbar is not None:
            self.valid_pbar.update(1)
    
    def on_test_end(self, logs=None):
        """Close validation progress bar at the end of validation"""
        if self.valid_pbar is not None:
            self.valid_pbar.close()
            self.valid_pbar = None
            
    def on_epoch_end(self, epoch, logs=None):
        """Close training progress bar at the end of each epoch"""
        if self.train_pbar is not None:
            self.train_pbar.close()
            self.train_pbar = None

def train_and_save(network_depth, num_filters_per_layer, input_window_size, num_epochs, 
                   train_data_dir, valid_data_dir, test_data_dir, models_dir,
                   use_improved_initialization=False, use_improved_sampling=False, spike_rich_ratio=0.5):

    # ========== GPU verification code - added here ==========
    import tensorflow as tf
    print("Training device check:")
    print("TensorFlow built with CUDA:", tf.test.is_built_with_cuda())
    print("GPU devices:", tf.config.list_physical_devices('GPU'))
    # ===========================================

    use_multiprocessing = True  # Disable multiprocessing to avoid memory issues
    num_workers = 4  # Reduce number of workers

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
    train_file_load = 0.2  # 0.3 # Reduce file loading ratio
    valid_file_load = 0.2  # 0.3 # Reduce file loading ratio
    num_steps_multiplier = 10

    train_files_per_epoch = 6 # 4  # Reduce number of training files
    valid_files_per_epoch = 2 #max(1, int(validation_fraction * train_files_per_epoch))

    # batch_size and learning rate will be dynamically set based on model size after model creation
    # Set default values first (will be updated after model analysis)
    batch_size = 64  # Default value, will be adjusted after model analysis
    loss_weights_per_epoch = [[1.0, 0.02]] * num_epochs # [[1.0, 0.0200]] * num_epochs # Even higher spike weight
    num_train_steps_per_epoch = [100] * num_epochs  # Reduce number of training steps

    # ------------------------------------------------------------------
    # define network architecture params
    # ------------------------------------------------------------------

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

    sample_file = train_files[0]
    with open(sample_file, 'rb') as f:
        sample_data = pickle.load(f)

    # Extract number of synapses from data
    if 'allSegmentsType' in sample_data['Params']:  # Original L5PC model
        num_segments_exc = len(sample_data['Params']['allSegmentsType'])
        num_segments_inh = len(sample_data['Params']['allSegmentsType'])
    else:  # IF model or SJC model
        num_segments_exc = len(sample_data['Results']['listOfSingleSimulationDicts'][0]['exInputSpikeTimes'])
        num_segments_inh = len(sample_data['Results']['listOfSingleSimulationDicts'][0]['inhInputSpikeTimes'])

    print(f'Detected from data: num_segments_exc = {num_segments_exc}, num_segments_inh = {num_segments_inh}')
    
    assert(input_window_size > sum(filter_sizes_per_layer))
    temporal_conv_net = create_temporaly_convolutional_model(input_window_size, num_segments_exc, num_segments_inh, 
                                                            filter_sizes_per_layer, num_filters_per_layer,
                                                            activation_function_per_layer, l2_regularization_per_layer,
                                                            strides_per_layer, dilation_rates_per_layer, initializer_per_layer,
                                                            use_improved_initialization=use_improved_initialization)

    # ========== Analyze model size and dynamically adjust batch_size and learning rate ==========
    # Quick model size overview (concise version)
    total_params, size_category, batch_range = get_model_size_info(temporal_conv_net)
    print(f"\nQuick model info: {size_category}, Parameters: {total_params:,}, Recommended batch_size: {batch_range}\n")
    
    # Detailed model size analysis (detailed version)
    model_info = analyze_model_size(temporal_conv_net, verbose=False)  # Don't print detailed info to avoid duplicate output
    
    # Dynamically set batch_size and learning rate based on model size
    if "小模型" in size_category or "Small" in model_info['size_category']:
        batch_size = 64
        base_lr = 0.0006  # Small models use smaller learning rate
        print(f"\nDetected small model, setting batch_size = {batch_size}, base_lr = {base_lr}")
    # elif "大模型" in size_category or "Large" in model_info['size_category']:
    #     batch_size = 256
    #     base_lr = 0.0006  # Large models use larger learning rate (from 64 to 256, learning rate ×2)
    #     print(f"\nDetected large model, setting batch_size = {batch_size}, base_lr = {base_lr}")
    else:
        batch_size = 8 # Because we set 400ms window size, the batch size should be smaller
        base_lr = 0.0001  # Because we set 400ms window size, the learning rate should be smaller
        print(f"\nDetected large model, setting batch_size = {batch_size}, base_lr = {base_lr}")
    
    # Set learning rate schedule based on batch_size
    batch_size_per_epoch = [batch_size] * num_epochs
    learning_rate_per_epoch = [base_lr] * num_epochs  # Initial learning rate (0-40 epochs)
    
    # Training stage configuration: (start_epoch, learning_rate_multiplier, loss_weights)
    # Learning rate will be dynamically calculated based on base_lr
    training_stages = [
        (40, 0.3, [2.0, 0.01]),   # 学习率 = base_lr * 0.3 # [2.0, 0.005]
        (80, 0.1, [4.0, 0.01]),   # 学习率 = base_lr * 0.1 # [4.0, 0.002]   
        (120, 0.03, [8.0, 0.01]), # 学习率 = base_lr * 0.03 # [8.0, 0.001]
        (160, 0.01, [9.0, 0.003]), # 学习率 = base_lr * 0.01 # [16.0, 0.001]
    ]
    
    for start_epoch, lr_ratio, loss_weights in training_stages:
        for i in range(start_epoch, num_epochs):
            learning_rate_per_epoch[i] = base_lr * lr_ratio
            loss_weights_per_epoch[i] = loss_weights
    
    print(f"Dynamic adjustment completed: batch_size = {batch_size}, initial learning rate = {base_lr}")
    print("=" * 60)
    
    # Set learning rate for warmup function (adjusted based on model size)
    max_lr = base_lr * 10  # max_lr is usually 10 times init_lr
    set_warmup_lr(base_lr, max_lr)
    
    # Save learning schedule to dictionary
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
        valid_steps_per_epoch = len(valid_data_generator)
        
        # Optimizer is initialized only once, learning rate is dynamically adjusted by callback
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
        
        # Add time monitoring in training loop
        start_time = time.time()
        
        # Create GPU monitor
        gpu_monitor = GPUMonitor()
        
        # Check GPU status before training
        print("GPU status before training:")
        gpu_monitor.print_status("  ")
        
    

        lr_scheduler = LearningRateScheduler(lr_warmup_decay, verbose=0)  # Don't print learning rate changes to avoid duplicate output
        # Create custom tqdm progress bar Callback
        tqdm_callback = TqdmProgressBar(learning_schedule, num_epochs, num_steps_multiplier, train_steps_per_epoch, valid_steps_per_epoch)
        history = temporal_conv_net.fit_generator(generator=train_data_generator,
                                                epochs=num_steps_multiplier,
                                                validation_data=valid_data_generator,
                                                use_multiprocessing=use_multiprocessing,
                                                workers=num_workers,
                                                verbose=0,  # Set to 0 to disable default progress bar, use custom tqdm progress bar
                                                callbacks=[lr_scheduler, tqdm_callback])

        training_time = time.time() - start_time
        print(f"Training time: {training_time:.2f}s")
        print("GPU status after training:")
        gpu_monitor.print_status("  ")
        
        # store the loss values in training histogry dictionary and add some additional fields about the training schedule
        try:
            for key in history.history.keys():
                training_history_dict[key] += history.history[key]
            training_history_dict['learning_schedule'] += [learning_schedule] * num_steps_multiplier
            training_history_dict['batch_size']        += [batch_size] * num_steps_multiplier
            training_history_dict['learning_rate']     += [learning_rate] * num_steps_multiplier
            # training_history_dict['learning_rate']     += [optimizer_to_use.get_config()['learning_rate']] * num_steps_multiplier # Use optimizer to get current learning rate
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
            # training_history_dict['learning_rate']     = [optimizer_to_use.get_config()['learning_rate']] * num_steps_multiplier # Use optimizer to get current learning rate
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
    Run complete model analysis, including AUC evaluation
    """
    print("Loading model results...")
    results = load_model_results(models_dir, test_data_dir)
    
    if not results:
        print("No model files found!")
        return
    print(f"Found {len(results)} models to analyze.")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Print model summary (including AUC)
    best_model = print_model_summary(results)
    
    # 2. Plot training curves
    print("\nGenerating training curves...")
    plot_training_curves(results, save_dir)
    
    # 3. Plot model comparison
    print("Generating model comparison plots...")
    plot_model_comparison(results, save_dir)
    
    # 4. Analyze training stability
    print("Analyzing training stability...")
    analyze_training_stability(results, save_dir)
    
    # 5. AUC analysis
    print("Analyzing AUC metrics...")
    plot_auc_analysis(results, save_dir)
    
    print(f"\nAnalysis complete! All plots saved to: {save_dir}")
    if best_model:
        print(f"Best model by AUC: {best_model['model_name']}")
        print(f"Best ROC AUC: {best_model['auc_metrics']['roc_auc_spike']:.4f}")

def main():
    # ========== Parse command line arguments ==========
    parser = argparse.ArgumentParser(description='Train and analyze TCN model (TensorFlow version)')
    
    # Hyperparameter grid arguments
    parser.add_argument('--network_depth', type=int, nargs='+', default=[7],
                        help='Network depth(s) to train (default: [7])')
    parser.add_argument('--num_filters_per_layer', type=int, nargs='+', default=[256],
                        help='Number of filters per layer (default: [256])')
    parser.add_argument('--input_window_size', type=int, nargs='+', default=[400],
                        help='Input window size(s) in ms (default: [400])')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    
    # Path and model configuration arguments
    parser.add_argument('--test_suffix', type=str, default='',
                        help='Test suffix to append to base path (default: empty string)')
    parser.add_argument('--base_subpath', type=str, default='Single_Neuron_InOut',
                        help='Base subpath for data and model directories (default: Single_Neuron_InOut)')
    parser.add_argument('--dataset_name', type=str, default='L5PC_NMDA',
                        help='Dataset name for train/valid/test directories (default: L5PC_NMDA)')
    parser.add_argument('--model_name', type=str, default='NMDA_tensorflow_ratio0.6',
                        help='Model name for model directory (default: NMDA_tensorflow_ratio0.6)')
    
    # Data sampling configuration arguments
    parser.add_argument('--use_improved_sampling', type=str, default='True', choices=['True', 'False', 'true', 'false'],
                        help='Whether to use improved data sampling strategy (default: True)')
    parser.add_argument('--spike_rich_ratio', type=float, default=0.6,
                        help='Ratio of samples containing spikes (default: 0.6)')
    
    args = parser.parse_args()
    
    # ========== Configuration: All variables defined together ==========
    # Hyperparameter grid
    network_depth_list = args.network_depth
    num_filters_per_layer_list = args.num_filters_per_layer
    input_window_size_list = args.input_window_size
    num_epochs = args.num_epochs
    
    # Path configuration
    test_suffix = args.test_suffix
    base_path = f'/G/results/aim2_sjc/Models_TCN/{args.base_subpath}{test_suffix}'
    dataset_name = args.dataset_name
    model_name = args.model_name
    
    # Configure improvement options
    use_improved_initialization = False   # Set to True to enable improved initialization strategy
    # Default to True if not explicitly set via command line (backward compatibility)
    use_improved_sampling = args.use_improved_sampling if hasattr(args, 'use_improved_sampling') and args.use_improved_sampling else True
    spike_rich_ratio = args.spike_rich_ratio

    # ========== GPU configuration code - added here ==========
    
    # Detect GPU
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Configure GPU memory growth to avoid out of memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(e)

    # Optional: specify which GPU to use (if multiple GPUs available)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only use GPU 0
    # ===========================================
    
    # GPU status diagnosis
    print("\n=== GPU Status Diagnosis ===")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"CUDA available: {tf.test.is_built_with_cuda()}")
    print(f"Number of GPU devices: {len(tf.config.list_physical_devices('GPU'))}")
    
    # Use GPU monitoring class
    gpu_monitor = GPUMonitor()
    if gpu_monitor.available:
        print("GPU monitor initialized successfully")
        gpu_monitor.print_status("  ")
    else:
        print("GPU monitoring not available, please install: pip install nvidia-ml-py")
    print("==================\n")
    
    print(f"\n=== Improvement Configuration ===")
    print(f"Improved initialization strategy: {'Enabled' if use_improved_initialization else 'Disabled'}")
    print(f"Improved data sampling: {'Enabled' if use_improved_sampling else 'Disabled'}")
    if use_improved_sampling:
        print(f"Spike-rich sample ratio: {spike_rich_ratio * 100:.0f}%")
    print(f"================\n")

    # 2. Main control loop
    for network_depth, num_filters_per_layer, input_window_size in product(network_depth_list, num_filters_per_layer_list, input_window_size_list):
        # Dynamically build analysis suffix
        analysis_suffix = build_analysis_suffix(base_path, model_name)
        
        # Add sampling configuration abbreviations to paths
        sampling_suffix = ''
        if use_improved_sampling:
            sampling_suffix = f'_ratio{spike_rich_ratio:.1f}'
        model_name_with_sampling = f'{model_name}{sampling_suffix}'
        analysis_suffix_with_sampling = f'{analysis_suffix}{sampling_suffix}'
    
        # Data directories
        train_data_dir = f'{base_path}/data/{dataset_name}_train/'
        valid_data_dir = f'{base_path}/data/{dataset_name}_valid/'
        test_data_dir = f'{base_path}/data/{dataset_name}_test/'
        
        # Model and analysis directories
        model_dir = f'{base_path}/models/{model_name_with_sampling}/depth_{network_depth}_filters_{num_filters_per_layer}_window_{input_window_size}/'
        analysis_dir = f'./results/3_model_analysis_plots/{analysis_suffix_with_sampling}/depth_{network_depth}_filters_{num_filters_per_layer}_window_{input_window_size}/'
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(analysis_dir, exist_ok=True)

        print(f"\n==============================")
        print(f"Starting training: network_depth={network_depth}, num_filters_per_layer={num_filters_per_layer}, input_window_size={input_window_size}")
        print(f"Model save directory: {model_dir}")
        print(f"Analysis results directory: {analysis_dir}")
        print(f"==============================\n")
        
        # 3. Train model
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

        # 4. Analyze model
        analyze_and_save(
            models_dir=model_dir,
            test_data_dir=test_data_dir,
            save_dir=analysis_dir
        )

    print("\nAll hyperparameter combination training and analysis completed!") 

if __name__ == "__main__":
    main()