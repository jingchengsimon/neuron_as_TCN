import numpy as np
import glob
import time
import os
import pickle
import argparse
from datetime import datetime  
from itertools import product
from utils.gpu_monitor import GPUMonitor, configure_pytorch_gpu, get_gpu_memory_info
from utils.fit_CNN_torch import TCNModel, SimulationDataGenerator
from utils.model_analysis import (
    load_model_results, print_model_summary, 
    plot_training_curves, plot_model_comparison, analyze_training_stability, plot_auc_analysis
)
from utils.model_size_utils import get_model_size_info, analyze_model_size
# TensorFlow/Keras imports removed - using PyTorch instead
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda

def create_lr_warmup_decay(init_lr, max_lr):
    """
    Create learning rate warmup and decay scheduling function (dynamic version)
    
    Args:
        init_lr: Initial learning rate
        max_lr: Maximum learning rate
    
    Returns:
        function: Learning rate scheduling function
    """
    warmup_epochs = 10
    decay_rate = 0.95
    
    def lr_warmup_decay(epoch, lr):
        """
        Learning rate warmup and decay scheduling
        Consistent with TF version: use passed init_lr and max_lr, ignore passed lr parameter
        """
        if epoch < warmup_epochs:
            # Linear warmup
            return init_lr + (max_lr - init_lr) * (epoch + 1) / warmup_epochs
        else:
            # Decay
            return max_lr * (decay_rate ** (epoch - warmup_epochs))
    
    return lr_warmup_decay

def train_and_save(network_depth, num_filters_per_layer, input_window_size, num_epochs, 
                   train_data_dir, valid_data_dir, test_data_dir, models_dir,
                   use_improved_initialization=False, use_improved_sampling=False, spike_rich_ratio=0.5):

    # ========== PyTorch GPU Configuration ==========
    device = configure_pytorch_gpu()
    print(f"Training will use device: {device}")
    # ===========================================

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
    loss_weights_per_epoch = [[1.0, 0.02]] * num_epochs # [1.0, 0.01]
    num_train_steps_per_epoch = [100] * num_epochs

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
    temporal_conv_net = TCNModel(input_window_size, num_segments_exc, num_segments_inh, 
                                filter_sizes_per_layer, num_filters_per_layer,
                                activation_function_per_layer, strides_per_layer, 
                                dilation_rates_per_layer, initializer_per_layer,
                                use_improved_initialization=use_improved_initialization)

    # Move model to specified device and create loss functions
    temporal_conv_net = temporal_conv_net.to(device)
    spike_criterion = nn.BCELoss()
    soma_criterion = nn.MSELoss()
    
    # Print model information and analyze model size
    # Quick model size overview (concise version)
    total_params, size_category, batch_range = get_model_size_info(temporal_conv_net)
    print(f"\nQuick model info: {size_category}, Parameters: {total_params:,}, Recommended batch_size: {batch_range}\n")
    
    # Detailed model size analysis (detailed version)
    model_info = analyze_model_size(temporal_conv_net, verbose=True)
    print(f"Model moved to device: {next(temporal_conv_net.parameters()).device}")
    
    # ========== Dynamically adjust batch_size and learning rate based on model size ==========
    # Determine model size and set batch_size
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
    
    # Create dynamic warmup function (adjusted based on model size)
    max_lr = base_lr * 10  # max_lr is usually 10 times init_lr
    lr_warmup_decay_func = create_lr_warmup_decay(base_lr, max_lr)
    
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

    
    num_learning_schedules = len(batch_size_per_epoch) # num_epochs

    training_history_dict = {}
    for learning_schedule in range(start_learning_schedule, num_learning_schedules): 
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
        
        # Optimizer is initialized only once; learning rate is set in each schedule
        if learning_schedule == 0:
            # Separate output layer parameters and other layer parameters to set smaller L2 regularization for output layer
            # Match TensorFlow version: convolutional layers use l2(1e-6), output layers use l2(1e-8)
            output_params = list(temporal_conv_net.spikes_head.parameters()) + \
                            list(temporal_conv_net.soma_head.parameters())
            other_params = [p for n, p in temporal_conv_net.named_parameters() 
                           if 'spikes_head' not in n and 'soma_head' not in n]
            
            try:
                # Use parameter groups to set different weight_decay for different layers
                # Convolutional layers use l2_regularization_per_layer[0] (1e-6)
                # Output layers use 1e-8, consistent with TensorFlow version
                optimizer = optim.NAdam([
                    {'params': other_params, 'weight_decay': l2_regularization_per_layer[0]},  # 1e-6
                    {'params': output_params, 'weight_decay': 1e-8}  # Output layers use 1e-8
                ], lr=learning_rate, eps=1e-7)  # eps=1e-7 matches TensorFlow NAdam default value
            except Exception:
                optimizer = optim.Adam([
                    {'params': other_params, 'weight_decay': l2_regularization_per_layer[0]},  # 1e-6
                    {'params': output_params, 'weight_decay': 1e-8}  # Output layers use 1e-8
                ], lr=learning_rate, eps=1e-7)
            
            print(f'Optimizer initialized with parameter groups:')
            print(f'  - Other layers: weight_decay={l2_regularization_per_layer[0]}')
            print(f'  - Output layers: weight_decay=1e-8')
        else:
            for g in optimizer.param_groups:
                g['lr'] = learning_rate
        
        print('-----------------------------------------------')
        print('starting epoch %d:' %(learning_schedule))
        print('-----------------------------------------------')
        print('loss weights = %s' %(str(loss_weights)))
        print('batch_size = %d' %(batch_size))
        print('-----------------------------------------------')
        
        # Add time monitoring in training loop
        start_time = time.time()
        
        # Create GPU monitor
        gpu_monitor = GPUMonitor()
        print("GPU status before training:")
        gpu_monitor.print_status("  ")
        memory_info = get_gpu_memory_info()
        if memory_info:
            print(f"PyTorch GPU memory - Allocated: {memory_info['allocated']:.2f}GB, Reserved: {memory_info['reserved']:.2f}GB")

        # Use PyTorch training loop instead of Keras fit_generator
        train_epoch_spike_losses = []
        train_epoch_soma_losses = []
        val_epoch_spike_losses = []
        val_epoch_soma_losses = []

        for mini_epoch in range(num_steps_multiplier):
            # Apply learning rate warmup/decay scheduling (matching TF version)
            # Calculate current learning rate based on mini_epoch within current learning_schedule
            # TF version uses lr_warmup_decay(epoch, lr) where epoch is the mini-epoch index
            # Note: Use dynamically created warmup function (adjusted based on model size)
            current_lr = lr_warmup_decay_func(mini_epoch, learning_rate)
            for g in optimizer.param_groups:
                g['lr'] = current_lr
            
            # Log learning rate for first mini_epoch of each learning_schedule
            if mini_epoch == 0:
                print(f'Learning rate for mini_epoch {mini_epoch+1}: {current_lr:.7f} (base_lr was {learning_rate:.7f})')
            
            # Train one "mini-epoch" (aligned with Keras epochs=num_steps_multiplier)
            temporal_conv_net.train()
            running_spike, running_soma, running_total = 0.0, 0.0, 0.0
            
            # Clear GPU memory
            if cuda.is_available():
                cuda.empty_cache()
            
            for step in tqdm(range(train_steps_per_epoch), desc=f"Train {learning_schedule+1}/{num_epochs} epoch {mini_epoch+1}/{num_steps_multiplier}", leave=False):
                X_batch, targets = train_data_generator[step]
                y_spike_batch, y_soma_batch = targets
                
                # Move data to GPU
                X_batch = X_batch.to(device, non_blocking=True)
                y_spike_batch = y_spike_batch.to(device, non_blocking=True)
                y_soma_batch = y_soma_batch.to(device, non_blocking=True)

                optimizer.zero_grad()
                pred_spike, pred_soma = temporal_conv_net(X_batch)

                loss_spike = spike_criterion(pred_spike, y_spike_batch)
                loss_soma = soma_criterion(pred_soma, y_soma_batch)
                loss = loss_weights[0] * loss_spike + loss_weights[1] * loss_soma

                loss.backward()
                optimizer.step()

                running_spike += loss_spike.item()
                running_soma += loss_soma.item()
                running_total += loss.item()
                
                # Clear GPU memory every 100 steps
                if step % 100 == 0 and cuda.is_available():
                    cuda.empty_cache()

            # Record training average loss (aligned with Keras one per epoch)
            train_epoch_spike_losses.append(running_spike / train_steps_per_epoch)
            train_epoch_soma_losses.append(running_soma / train_steps_per_epoch)

            # Validation
            temporal_conv_net.eval()
            with torch.no_grad():
                val_spike, val_soma = 0.0, 0.0
                for vstep in tqdm(range(len(valid_data_generator)), desc="Valid", leave=False):
                    Xb, targets_v = valid_data_generator[vstep]
                    ysb, yvb = targets_v
                    
                    # Move data to GPU
                    Xb = Xb.to(device, non_blocking=True)
                    ysb = ysb.to(device, non_blocking=True)
                    yvb = yvb.to(device, non_blocking=True)
                    
                    ps, pv = temporal_conv_net(Xb)
                    val_spike += spike_criterion(ps, ysb).item()
                    val_soma  += soma_criterion(pv, yvb).item()
                    
                    # Clear GPU memory
                    if vstep % 50 == 0 and cuda.is_available():
                        cuda.empty_cache()
                        
                val_epoch_spike_losses.append(val_spike / max(1, len(valid_data_generator)))
                val_epoch_soma_losses.append(val_soma / max(1, len(valid_data_generator)))

        # Construct structure equivalent to Keras history for subsequent statistics
        history = {
            'history': {
                'spikes_loss': train_epoch_spike_losses,
                'somatic_loss': train_epoch_soma_losses,
                'val_spikes_loss': val_epoch_spike_losses,
                'val_somatic_loss': val_epoch_soma_losses,
                'loss': [ws + wm for ws, wm in zip(train_epoch_spike_losses, train_epoch_soma_losses)],
                'val_loss': [ws + wm for ws, wm in zip(val_epoch_spike_losses, val_epoch_soma_losses)],
            }
        }

        training_time = time.time() - start_time
        print(f"Training time: {training_time:.2f}s")
        print("GPU status after training:")
        gpu_monitor.print_status("  ")
        
        # Print PyTorch GPU memory information
        memory_info = get_gpu_memory_info()
        if memory_info:
            print(f"PyTorch GPU memory - Allocated: {memory_info['allocated']:.2f}GB, Reserved: {memory_info['reserved']:.2f}GB, Max allocated: {memory_info['max_allocated']:.2f}GB")
        
        # Clear GPU memory
        if cuda.is_available():
            cuda.empty_cache()
            print("GPU memory cleared")
        
        # store the loss values in training histogry dictionary and add some additional fields about the training schedule
        try:
            for key in history['history'].keys():
                training_history_dict[key] += history['history'][key]
            training_history_dict['learning_schedule'] += [learning_schedule] * num_steps_multiplier
            training_history_dict['batch_size']        += [batch_size] * num_steps_multiplier
            training_history_dict['learning_rate']     += [learning_rate] * num_steps_multiplier
            training_history_dict['loss_weights']      += [loss_weights] * num_steps_multiplier
            training_history_dict['num_train_samples'] += [batch_size * train_steps_per_epoch] * num_steps_multiplier
            training_history_dict['num_train_steps']   += [train_steps_per_epoch] * num_steps_multiplier
            training_history_dict['train_files_histogram'] += [train_data_generator.batches_per_file_dict]
            training_history_dict['valid_files_histogram'] += [valid_data_generator.batches_per_file_dict]
        except:
            for key in history['history'].keys():
                training_history_dict[key] = history['history'][key]
            training_history_dict['learning_schedule'] = [learning_schedule] * num_steps_multiplier
            training_history_dict['batch_size']        = [batch_size] * num_steps_multiplier
            training_history_dict['learning_rate']     = [learning_rate] * num_steps_multiplier
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
            model_filename    = models_dir + '%s__%s__%s__%s__%s__%s.pt' %(model_prefix,architecture_overview,current_datetime,train_string,results_overview,modelID_str)
            auxilary_filename = models_dir + '%s__%s__%s__%s__%s__%s.pickle' %(model_prefix,architecture_overview,current_datetime,train_string,results_overview,modelID_str)

            print('-----------------------------------------------------------------------------------------')
            print('val_spikes_loss is: %f' %(np.array(training_history_dict['val_spikes_loss'][-3:]).mean()))
            print('finished epoch %d/%d. saving...\n     "%s"\n     "%s"' %(learning_schedule +1, num_epochs, model_filename.split('/')[-1], auxilary_filename.split('/')[-1]))
            print('-----------------------------------------------------------------------------------------')

            # Save PyTorch model parameters
            torch.save(temporal_conv_net.state_dict(), model_filename)
            
            # save all relevent training params (in raw and unprocessed way)
            model_hyperparams_and_training_dict = {}
            model_hyperparams_and_training_dict['data_dict']              = data_dict
            model_hyperparams_and_training_dict['architecture_dict']      = architecture_dict
            model_hyperparams_and_training_dict['learning_schedule_dict'] = learning_schedule_dict
            model_hyperparams_and_training_dict['training_history_dict']  = training_history_dict
            
            pickle.dump(model_hyperparams_and_training_dict, open(auxilary_filename, "wb"), protocol=2)
        else:
            print('-----------------------------------------------------------------------------------------')
            print('val_spikes_loss is: %f' %(np.array(training_history_dict['val_spikes_loss'][-3:]).mean()))
            print('No model saved for epoch %d/%d' %(learning_schedule +1, num_epochs))
            print('-----------------------------------------------------------------------------------------')

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
    parser = argparse.ArgumentParser(description='Train and analyze TCN model (PyTorch version)')
    
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
    parser.add_argument('--model_name', type=str, default='NMDA_torch_ratio0.6_2',
                        help='Model name for model directory (default: NMDA_torch_ratio0.6_2)')
    
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
    use_improved_sampling = True      # Set to True to enable improved data sampling strategy
    spike_rich_ratio = 0.6              # 60% of samples contain spikes
    
    # ========== PyTorch GPU Configuration ==========
    print("=== PyTorch GPU Configuration ===")
    device = configure_pytorch_gpu()
    
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

    def build_analysis_suffix(base_path, model_name):
        """
        Dynamically build analysis suffix
        Extract part after 'InOut' from base path, extract part after underscore from model name
        """
        # Extract part after 'InOut' from base path
        if 'InOut' in base_path:
            inout_part = base_path.split('InOut')[-1]  # Get part after 'InOut'
            # If result starts with underscore, remove leading underscore
            if inout_part.startswith('_'):
                inout_part = inout_part[1:]
        else:
            inout_part = 'original' # base_path.split('/')[-1]  # If no 'InOut', take last part
        
        # Extract part after underscore from model name
        if '_' in model_name:
            if model_name.startswith('IF_') or model_name.startswith('reduce_'):
                # For IF and reduce models, get the third part (e.g., "torch" from "reduce_model_torch")
                parts = model_name.split('_')
                if len(parts) >= 3:
                    model_part = parts[2]  # Get the third part
                else:
                    model_part = model_name.split('_', 1)[1]  # Fallback to second part
            else:
                # For single neuron, get the second part (e.g., "torch_ratio0.6_2" from "NMDA_torch_ratio0.6_2")
                model_part = model_name.split('_', 1)[1]  # Get part after first underscore
        else:
            model_part = model_name
        
        # Combine into analysis suffix
        analysis_suffix = f"{inout_part}_{model_part}"
        return analysis_suffix

    # Dynamically build analysis suffix
    analysis_suffix = build_analysis_suffix(test_suffix, model_name)

    # Data directories
    train_data_dir = f'{base_path}/data/{dataset_name}_train/'
    valid_data_dir = f'{base_path}/data/{dataset_name}_valid/'
    test_data_dir = f'{base_path}/data/{dataset_name}_test/'
    
    # 2. Main control loop
    for network_depth, num_filters_per_layer, input_window_size in product(network_depth_list, num_filters_per_layer_list, input_window_size_list): 
        
        # Model and analysis directories
        model_dir = f'{base_path}/models/{model_name}/depth_{network_depth}_filters_{num_filters_per_layer}_window_{input_window_size}/'
        analysis_dir = f'./results/3_model_analysis_plots/{analysis_suffix}/depth_{network_depth}_filters_{num_filters_per_layer}_window_{input_window_size}/'

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

        # 4. Analyze model
        analyze_and_save(
            models_dir=model_dir,
            test_data_dir=test_data_dir,
            save_dir=analysis_dir
        )

    print("\nAll hyperparameter combinations training and analysis completed!") 

if __name__ == "__main__":
    main()