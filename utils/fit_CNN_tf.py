import numpy as np
import time
import sys
import os
import pickle

# Set TensorFlow log level to reduce warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Disable TensorFlow graph optimizer warnings
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

import keras
from keras.models import Model
from keras.optimizers import Nadam
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import LeakyReLU
from keras.regularizers import l2
from keras import initializers


# share parse function from fit_CNN_torch
from utils.fit_CNN_torch import parse_sim_experiment_file

# some fixes for python 3
if sys.version_info[0]<3:
    import cPickle as pickle
else:
    import pickle
    basestring = str
    
def create_temporaly_convolutional_model(max_input_window_size, num_segments_exc, num_segments_inh, filter_sizes_per_layer,
                                                                                             num_filters_per_layer,
                                                                                             activation_function_per_layer,
                                                                                             l2_regularization_per_layer,
                                                                                             strides_per_layer,
                                                                                             dilation_rates_per_layer,
                                                                                             initializer_per_layer,
                                                                                             use_improved_initialization=False):
    """
    Create temporal convolutional network model
    
    Args:
        use_improved_initialization: Whether to use improved initialization strategy
            False: Use original scheme
            True: Use improved scheme (better spike output initialization, Focal Loss, etc.)
    """
    
    # define input and flatten it
    binary_input_mat = Input(shape=(max_input_window_size, num_segments_exc + num_segments_inh), name='input_layer')

    # define convolutional layers
    for k in range(len(filter_sizes_per_layer)):
        num_filters   = num_filters_per_layer[k]
        filter_size   = filter_sizes_per_layer[k]
        activation    = activation_function_per_layer[k]
        l2_reg        = l2_regularization_per_layer[k]
        stride        = strides_per_layer[k]
        dilation_rate = dilation_rates_per_layer[k]
        initializer   = initializer_per_layer[k]
        
        if activation == 'lrelu':
            leaky_relu_slope = 0.25
            activation = lambda x: LeakyReLU(alpha=leaky_relu_slope)(x)
            print('leaky relu slope = %.4f' %(leaky_relu_slope))
            
        if not isinstance(initializer, basestring):
            initializer = initializers.TruncatedNormal(stddev=initializer)
        
        # first layer
        if k == 0: 
            x = Conv1D(num_filters, filter_size, activation=activation, kernel_initializer=initializer, kernel_regularizer=l2(l2_reg),
                       strides=stride, dilation_rate=dilation_rate, padding='causal', name='layer_%d' %(k + 1))(binary_input_mat)
        # other layers
        else: 
            x = Conv1D(num_filters, filter_size, activation=activation, kernel_initializer=initializer, kernel_regularizer=l2(l2_reg),
                       strides=stride, dilation_rate=dilation_rate, padding='causal', name='layer_%d' %(k + 1))(x)
        x = BatchNormalization(name='layer_%d_BN' %(k + 1))(x)
    
    
    output_spike_init_weights = initializers.TruncatedNormal(stddev=0.001)
    output_spike_init_bias    = initializers.Constant(value=-2.0)
    output_soma_init = initializers.TruncatedNormal(stddev=0.03)

    output_spike_predictions = Conv1D(1, 1, activation='sigmoid', kernel_initializer=output_spike_init_weights, bias_initializer=output_spike_init_bias,
                                                                  kernel_regularizer=l2(1e-8), padding='causal', name='spikes')(x)
    output_soma_voltage_pred = Conv1D(1, 1, activation='linear' , kernel_initializer=output_soma_init, kernel_regularizer=l2(1e-8), padding='causal', name='somatic')(x)

    temporaly_convolutional_network_model = Model(inputs=binary_input_mat, outputs=
                                                  [output_spike_predictions, output_soma_voltage_pred])

    optimizer_to_use = Nadam(lr=0.0001)
    temporaly_convolutional_network_model.compile(optimizer=optimizer_to_use,
                                                      loss=['binary_crossentropy','mse'],
                                                      loss_weights=[1.0, 0.006])   
    temporaly_convolutional_network_model.summary()
    
    return temporaly_convolutional_network_model


class SimulationDataGenerator(keras.utils.Sequence):
    'thread-safe data generator for network training'

    def __init__(self, sim_experiment_files, num_files_per_epoch=10,
                 batch_size=8, window_size_ms=300, file_load=0.3, 
                 use_improved_sampling=False, spike_rich_ratio=0.5,
                 ignore_time_from_start=500, y_train_soma_bias=-67.7, y_soma_threshold=-55.0):
        """
        Args:
            use_improved_sampling: Whether to use improved data sampling strategy
                False: Use original random sampling
                True: Prioritize time windows containing spikes
            spike_rich_ratio: Ratio of samples containing spikes (only effective when use_improved_sampling=True)
        """
        'data generator initialization'
        
        self.sim_experiment_files = sim_experiment_files
        
        # Check file count, ensure num_files_per_epoch doesn't exceed available files
        if len(self.sim_experiment_files) == 0:
            raise ValueError(f"No experiment files found")
        
        if num_files_per_epoch > len(self.sim_experiment_files):
            print(f"Warning: Requested num_files_per_epoch ({num_files_per_epoch}) exceeds available file count ({len(self.sim_experiment_files)})")
            print(f"Adjusting num_files_per_epoch to {len(self.sim_experiment_files)}")
            num_files_per_epoch = len(self.sim_experiment_files)
        
        self.num_files_per_epoch = num_files_per_epoch
        self.batch_size = batch_size
        self.window_size_ms = window_size_ms
        self.ignore_time_from_start = ignore_time_from_start
        self.file_load = file_load
        self.y_train_soma_bias = y_train_soma_bias
        self.y_soma_threshold = y_soma_threshold
        self.use_improved_sampling = use_improved_sampling
        self.spike_rich_ratio = spike_rich_ratio
        
        self.curr_epoch_files_to_use = None
        self.on_epoch_end()
        self.curr_file_index = -1
        self.load_new_file()
        self.batches_per_file_dict = {}
        
        # gather information regarding the loaded file
        self.num_simulations_per_file, self.sim_duration_ms, self.num_segments = self.X.shape # 128 simulations, 6 seconds, 639 segments
        self.num_output_channels_y1 = self.y_spike.shape[2]
        self.num_output_channels_y2 = self.y_soma.shape[2]
        
        # determine how many batches in total can enter in the file
        self.max_batches_per_file = (self.num_simulations_per_file * self.sim_duration_ms) / (self.batch_size * self.window_size_ms) # (128*6000)/(2*400)=960 (20*5900)/(2*200)=295
        self.batches_per_file = int(self.file_load * self.max_batches_per_file) # int(0.2*240)=48 (59*6)=354
        self.batches_per_epoch = self.batches_per_file * self.num_files_per_epoch # 48*6=288 (59*6)=354

        print('-------------------------------------------------------------------------')
        print('file load = %.4f, max batches per file = %d, batches per epoch = %d' %(self.file_load,
                                                                                      self.max_batches_per_file,
                                                                                      self.batches_per_epoch))
        print('num batches per file = %d. coming from (%dx%d),(%dx%d)' %(self.batches_per_file, self.num_simulations_per_file,
                                                                         self.sim_duration_ms, self.batch_size, self.window_size_ms))
        if self.use_improved_sampling:
            print('Using improved data sampling strategy: spike-rich ratio = %.1f%%' %(self.spike_rich_ratio * 100))
        else:
            print('Using original random sampling strategy')
        print('-------------------------------------------------------------------------')

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, batch_ind_within_epoch):
        'Generate one batch of data'
        
        if ((batch_ind_within_epoch + 1) % self.batches_per_file) == 0:
            self.load_new_file()
            
        # Choose different sampling strategies based on configuration
        if self.use_improved_sampling:
            selected_sim_inds, selected_end_time_inds = self._select_balanced_windows()
        else:
            # Original random sampling strategy
            selected_sim_inds = np.random.choice(range(self.num_simulations_per_file), size=self.batch_size, replace=True)
            sampling_end_time = max(self.ignore_time_from_start, self.window_size_ms)
            selected_end_time_inds = np.random.choice(range(sampling_end_time, self.sim_duration_ms), size=self.batch_size, replace=False)
        
        # gather batch and yield it
        X_batch       = np.zeros((self.batch_size, self.window_size_ms, self.num_segments))
        y_spike_batch = np.zeros((self.batch_size, self.window_size_ms, self.num_output_channels_y1))
        y_soma_batch  = np.zeros((self.batch_size, self.window_size_ms, self.num_output_channels_y2))
        for k, (sim_ind, win_time) in enumerate(zip(selected_sim_inds, selected_end_time_inds)):
            X_batch[k,:,:]       = self.X[sim_ind, win_time - self.window_size_ms:win_time,:]
            y_spike_batch[k,:,:] = self.y_spike[sim_ind, win_time - self.window_size_ms:win_time,:]
            y_soma_batch[k,:,:]  = self.y_soma[sim_ind, win_time - self.window_size_ms:win_time,:]
        
        # increment the number of batches collected from each file
        try:
            self.batches_per_file_dict[self.curr_file_in_use] = self.batches_per_file_dict[self.curr_file_in_use] + 1
        except:
            self.batches_per_file_dict[self.curr_file_in_use] = 1
        
        # return the actual batch
        return (X_batch, [y_spike_batch, y_soma_batch])

    def _select_balanced_windows(self):
        """Improved sampling strategy: prioritize time windows containing spikes"""
        num_spike_rich = int(self.batch_size * self.spike_rich_ratio)
        num_random = self.batch_size - num_spike_rich
        
        # 1. Select time windows containing spikes
        spike_rich_windows = []
        sampling_end_time = max(self.ignore_time_from_start, self.window_size_ms)
        for sim_ind in range(self.num_simulations_per_file):
            spike_times = self.y_spike[sim_ind, :, 0]
            if np.sum(spike_times) > 0:  # If this simulation contains spikes
                # Select time windows centered on spikes
                for spike_time in np.where(spike_times)[0]:
                    # Calculate window end time (consistent with __getitem__, win_time is right-closed slice upper bound)
                    end_time = spike_time + self.window_size_ms // 2
                    start_time = end_time - self.window_size_ms
                    # Strong constraint: win_time >= ignore_time_from_start, and window completely falls within [0, sim_duration)
                    if (end_time >= sampling_end_time and
                        end_time < self.sim_duration_ms and
                        start_time >= 0):
                        spike_rich_windows.append((sim_ind, end_time))
        
        # If spike-rich windows are insufficient, repeat some
        if len(spike_rich_windows) > 0 and len(spike_rich_windows) < num_spike_rich:
            repeat_times = (num_spike_rich // len(spike_rich_windows)) + 1
            spike_rich_windows = np.tile(spike_rich_windows, (repeat_times, 1))
        
        # Randomly select spike-rich windows
        if len(spike_rich_windows) > 0:
            selected_spike_rich = np.random.choice(len(spike_rich_windows), 
                                                 size=min(num_spike_rich, len(spike_rich_windows)), 
                                                 replace=False)
            selected_wins_spike_rich = [spike_rich_windows[i] for i in selected_spike_rich]
        else:
            selected_wins_spike_rich = []
        
        # 2. Supplement with random windows
        selected_wins_random = []
        available_times = list(range(sampling_end_time, self.sim_duration_ms))
        
        # Avoid overlap with spike-rich windows
        for sim_ind, end_time in selected_wins_spike_rich:
            for t in range(max(sampling_end_time, end_time - self.window_size_ms), 
                          min(self.sim_duration_ms, end_time + self.window_size_ms)):
                if t in available_times:
                    available_times.remove(t)
        
        if len(available_times) >= num_random:
            selected_end_time_inds_random = np.random.choice(available_times, size=num_random, replace=False)
            selected_sim_inds_random = np.random.choice(range(self.num_simulations_per_file), size=num_random, replace=True)
            selected_wins_random = list(zip(selected_sim_inds_random, selected_end_time_inds_random))
        else:
            # If available times are insufficient, allow repetition
            selected_end_time_inds_random = np.random.choice(available_times, size=num_random, replace=True)
            selected_sim_inds_random = np.random.choice(range(self.num_simulations_per_file), size=num_random, replace=True)
            selected_wins_random = list(zip(selected_sim_inds_random, selected_end_time_inds_random))
        
        # 3. Combine and return
        all_windows = selected_wins_spike_rich + selected_wins_random
        np.random.shuffle(all_windows)  # Randomly shuffle order
        
        selected_sim_inds = [w[0] for w in all_windows]
        selected_end_time_inds = [w[1] for w in all_windows]
        
        return selected_sim_inds, selected_end_time_inds

    def on_epoch_end(self):
        'selects new subset of files to draw samples from'
        self.curr_epoch_files_to_use = np.random.choice(self.sim_experiment_files, size=self.num_files_per_epoch, replace=False)

    def load_new_file(self):
        'load new file to draw batches from'
        self.curr_file_index = (self.curr_file_index + 1) % self.num_files_per_epoch
        # update the current file in use
        self.curr_file_in_use = self.curr_epoch_files_to_use[self.curr_file_index]

        # load the file
        X, y_spike, y_soma = parse_sim_experiment_file(self.curr_file_in_use)

        # reshape to what is needed
        X  = np.transpose(X,axes=[2,1,0])
        y_spike = y_spike.T[:,:,np.newaxis]
        y_soma  = y_soma.T[:,:,np.newaxis]

        y_soma[y_soma > self.y_soma_threshold] = self.y_soma_threshold # Only save the subthreshold voltage for the voltage prediction

        y_soma = y_soma - self.y_train_soma_bias
        
        self.X, self.y_spike, self.y_soma = X, y_spike, y_soma
