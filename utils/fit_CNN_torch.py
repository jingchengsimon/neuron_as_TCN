import numpy as np
import time
import sys
import pickle

# Remove TensorFlow/Keras dependencies, use PyTorch instead
import torch
import torch.nn as nn
import torch.nn.functional as F

# some fixes for python 3
if sys.version_info[0]<3:
    import cPickle as pickle
else:
    import pickle
    basestring = str
    

def dict2bin(row_inds_spike_times_map, num_segments, sim_duration_ms, syn_type, data_type=None):
    """Convert dictionary of spike times to binary spike matrix"""
    # Batch process dictionary keys before the loop
    if syn_type == 'exc' and (data_type or '').lower() == 'sjc':
        # For excitatory synapse dictionary, subtract 1 from all keys in SJC data (from 1-639 to 0-638)
        adjusted_dict = {}
        for key, value in row_inds_spike_times_map.items():
            adjusted_dict[key - 1] = value
        row_inds_spike_times_map = adjusted_dict
    # For inh type, no need to modify keys

    bin_spikes_matrix = np.zeros((num_segments, sim_duration_ms), dtype='bool')            
    for row_ind in row_inds_spike_times_map.keys():
        for spike_time in row_inds_spike_times_map[row_ind]:
            bin_spikes_matrix[row_ind,spike_time] = 1.0
    
    return bin_spikes_matrix

def bin2dict(bin_spikes_matrix):
    """Convert binary spike matrix to dictionary format"""
    spike_row_inds, spike_times = np.nonzero(bin_spikes_matrix)
    row_inds_spike_times_map = {}
    for row_ind, syn_time in zip(spike_row_inds, spike_times):
        if row_ind not in row_inds_spike_times_map:
            row_inds_spike_times_map[row_ind] = []
            row_inds_spike_times_map[row_ind].append(syn_time)
    return row_inds_spike_times_map

def parse_multiple_sim_experiment_files(sim_experiment_files):
    """Parse multiple simulation experiment files"""
    if not sim_experiment_files:
        raise ValueError('No test files found. Please check test_data_dir and glob pattern.')
    
    data_list = []
    for sim_experiment_file in sim_experiment_files:
        X_curr, y_spike_curr, y_soma_curr = parse_sim_experiment_file(sim_experiment_file)
        data_list.append((X_curr, y_spike_curr, y_soma_curr))
    
    if not data_list:
        raise ValueError('Failed to assemble test data. Parsed zero files.')
    
    # Merge all data
    X = np.dstack([data[0] for data in data_list])
    y_spike = np.hstack([data[1] for data in data_list])
    y_soma = np.hstack([data[2] for data in data_list])
    
    return X, y_spike, y_soma

def parse_sim_experiment_file(sim_experiment_file, print_logs=False):
    
    if print_logs:
        print('-----------------------------------------------------------------')
        print("loading file: '" + sim_experiment_file.split("\\")[-1] + "'")
        loading_start_time = time.time()
        
    if sys.version_info[0]<3:
        experiment_dict = pickle.load(open(sim_experiment_file, "rb" ))
    else:
        experiment_dict = pickle.load(open(sim_experiment_file, "rb" ),encoding='latin1')
    
    # Detect data type from full path (not only basename)
    path_lower = str(sim_experiment_file).lower()
    data_type = 'sjc' if 'sjc' in path_lower else 'original'
    
    # Gather params
    num_simulations = len(experiment_dict['Results']['listOfSingleSimulationDicts'])
    if 'allSegmentsType' in experiment_dict['Params']: # model_original
        num_segments_exc  = len(experiment_dict['Params']['allSegmentsType'])
        num_segments_inh  = len(experiment_dict['Params']['allSegmentsType'])
    else: # model_aim1_sjc
        num_segments_exc  = len(experiment_dict['Results']['listOfSingleSimulationDicts'][0]['exInputSpikeTimes'])
        num_segments_inh  = len(experiment_dict['Results']['listOfSingleSimulationDicts'][0]['inhInputSpikeTimes'])
    if 'totalSimDurationInSec' in experiment_dict['Params']:
        sim_duration_ms = experiment_dict['Params']['totalSimDurationInSec'] * 1000
    else:
        sim_duration_ms = experiment_dict['Params']['STIM DURATION'] - 100
    num_ex_synapses  = num_segments_exc
    num_inh_synapses = num_segments_inh
    num_synapses = num_ex_synapses + num_inh_synapses
    
    # Collect X, y_spike, y_soma
    X = np.zeros((num_synapses, sim_duration_ms, num_simulations), dtype='bool')
    y_spike = np.zeros((sim_duration_ms,num_simulations))
    y_soma  = np.zeros((sim_duration_ms,num_simulations))
    
    # Go over all simulations in the experiment and collect their results
    for k, sim_dict in enumerate(experiment_dict['Results']['listOfSingleSimulationDicts']):
        X_ex  = dict2bin(sim_dict['exInputSpikeTimes'], num_segments_exc, sim_duration_ms, 'exc', data_type)
        X_inh = dict2bin(sim_dict['inhInputSpikeTimes'], num_segments_inh, sim_duration_ms, 'inh', data_type)
        X[:,:,k] = np.vstack((X_ex,X_inh))
        spike_times = (sim_dict['outputSpikeTimes'].astype(float) - 0.5).astype(int)
        y_spike[spike_times,k] = 1.0
        y_soma[:,k] = sim_dict['somaVoltageLowRes']
        
    if print_logs:
        loading_duration_sec = time.time() - loading_start_time
        print('loading took %.3f seconds' %(loading_duration_sec))
        print('-----------------------------------------------------------------')

    return X, y_spike, y_soma

class CausalConv1d(nn.Module):
    """
    Causal convolution: only pad on the left side, ensuring output time dimension matches input (when stride=1).
    Note: when stride>1, output will be downsampled by stride.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, dilation=dilation, padding=0, bias=bias)

    def forward(self, x):
        # x: (B, C, T)
        pad_left = (self.kernel_size - 1) * self.dilation
        x_padded = F.pad(x, (pad_left, 0))  # Only left padding
        return self.conv(x_padded)

class TCNModel(nn.Module):
    """
    PyTorch version of the Temporal Convolutional Network model.
    Maintains the same layer logic as the original model:
    - Several Conv1d (causal convolution) + BatchNorm1d + activation layers
    - Two output heads: spikes (sigmoid) and somatic (linear)
    Input: (batch, time, channels) to be consistent with the original data generator, then internally converted to (batch, channels, time)
    Output: spikes (batch, time, 1), somatic (batch, time, 1)
    """
    def __init__(self, max_input_window_size, num_segments_exc, num_segments_inh, 
                 filter_sizes_per_layer, num_filters_per_layer, activation_function_per_layer, strides_per_layer, 
                 dilation_rates_per_layer, initializer_per_layer, use_improved_initialization=False):
        super().__init__()
        
        self.input_window_size = max_input_window_size
        in_channels = num_segments_exc + num_segments_inh
        layers = []
        current_channels = in_channels
        
        # Activation function mapping
        def get_activation(name):
            if name == 'relu':
                return nn.ReLU(inplace=True)
            if name == 'lrelu':
                return nn.LeakyReLU(0.25, inplace=True)
            if name == 'elu':
                return nn.ELU(inplace=True)
            return nn.ReLU(inplace=True)
        
        for k in range(len(filter_sizes_per_layer)):
            num_filters   = num_filters_per_layer[k]
            filter_size   = filter_sizes_per_layer[k]
            activation    = activation_function_per_layer[k]
            stride        = strides_per_layer[k]
            dilation_rate = dilation_rates_per_layer[k]
            initializer   = initializer_per_layer[k]
            
            # Use causal convolution: only left padding, maintain length
            conv = CausalConv1d(in_channels=current_channels, out_channels=num_filters,
                                kernel_size=filter_size, stride=stride, dilation=dilation_rate, bias=True)
            
            # Initialize
            if isinstance(initializer, (int, float)):
                nn.init.trunc_normal_(conv.conv.weight, std=initializer)
            else:
                nn.init.kaiming_normal_(conv.conv.weight)

            if conv.conv.bias is not None:
                nn.init.zeros_(conv.conv.bias)
            
            # Follow Keras order: Conv -> Activation -> BatchNorm
            layers.append(conv)
            layers.append(get_activation(activation))
            layers.append(nn.BatchNorm1d(num_features=num_filters, momentum=0.01, eps=0.001))
            # layers.append(nn.BatchNorm1d(num_features=num_filters))
            current_channels = num_filters
        
        self.tcn = nn.Sequential(*layers)
        
        # Output layer
        if use_improved_initialization:
            spike_weight_init_std = 0.01
            spike_bias_init_val = 0.0
            soma_weight_init_std = 0.1
        else:
            spike_weight_init_std = 0.001
            spike_bias_init_val = -2.0
            soma_weight_init_std = 0.03
        
        self.spikes_head = nn.Conv1d(current_channels, 1, kernel_size=1, padding=0)
        nn.init.trunc_normal_(self.spikes_head.weight, std=spike_weight_init_std)
        nn.init.constant_(self.spikes_head.bias, spike_bias_init_val)
        
        self.soma_head = nn.Conv1d(current_channels, 1, kernel_size=1, padding=0)
        nn.init.trunc_normal_(self.soma_head.weight, std=soma_weight_init_std)
        if self.soma_head.bias is not None:
            nn.init.zeros_(self.soma_head.bias)

    def forward(self, x):
        # x: (batch, time, channels) -> (batch, channels, time)
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, time, channels), got shape {tuple(x.shape)}")
        x = x.permute(0, 2, 1)
        y = self.tcn(x)
        spikes = torch.sigmoid(self.spikes_head(y))   # (B, 1, T)
        soma   = self.soma_head(y)                    # (B, 1, T)
        # Convert back to (B, T, 1)
        spikes = spikes.permute(0, 2, 1)
        soma   = soma.permute(0, 2, 1)
        return spikes, soma

class SimulationDataGenerator:
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
        
        # self.y_DTV_threshold = y_DTV_threshold
        
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
            selected_sim_inds, selected_time_inds = self._select_balanced_windows()
        else:
            # Original random sampling strategy
            selected_sim_inds = np.random.choice(range(self.num_simulations_per_file), size=self.batch_size, replace=True)
            sampling_start_time = max(self.ignore_time_from_start, self.window_size_ms)
            selected_time_inds = np.random.choice(range(sampling_start_time, self.sim_duration_ms), size=self.batch_size, replace=False)
        
        # gather batch and yield it
        X_batch       = np.zeros((self.batch_size, self.window_size_ms, self.num_segments), dtype=np.float32)
        y_spike_batch = np.zeros((self.batch_size, self.window_size_ms, self.num_output_channels_y1), dtype=np.float32)
        y_soma_batch  = np.zeros((self.batch_size, self.window_size_ms, self.num_output_channels_y2), dtype=np.float32)
        for k, (sim_ind, win_time) in enumerate(zip(selected_sim_inds, selected_time_inds)):
            # print('sim_ind: %d, win_time: %d' %(sim_ind, win_time))
            X_batch[k,:,:]       = self.X[sim_ind, win_time - self.window_size_ms:win_time,:]
            y_spike_batch[k,:,:] = self.y_spike[sim_ind, win_time - self.window_size_ms:win_time,:]
            y_soma_batch[k,:,:]  = self.y_soma[sim_ind, win_time - self.window_size_ms:win_time,:]
        
        # Convert to torch.Tensor for direct feeding to PyTorch model
        X_batch_t       = torch.from_numpy(X_batch)           # (B, T, C)
        y_spike_batch_t = torch.from_numpy(y_spike_batch)     # (B, T, 1)
        y_soma_batch_t  = torch.from_numpy(y_soma_batch)      # (B, T, 1)
        
        # increment the number of batches collected from each file
        try:
            self.batches_per_file_dict[self.curr_file_in_use] = self.batches_per_file_dict[self.curr_file_in_use] + 1
        except:
            self.batches_per_file_dict[self.curr_file_in_use] = 1
        
        # return the actual batch
        return (X_batch_t, [y_spike_batch_t, y_soma_batch_t])

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
