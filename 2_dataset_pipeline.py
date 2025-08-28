import os
import json
import pandas as pd
import numpy as np
import pickle
import shutil
import glob
import ast
from scipy.signal import find_peaks
from multiprocessing import Pool, cpu_count
from functools import partial
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

class OptimizedDatasetPipeline:
    """
    Optimized version of DatasetPipeline for handling large numbers of experiments (>200)
    Uses multiprocessing and other optimization techniques
    """
    
    def __init__(self, root_folder_path, output_dir, train_dir, valid_dir, test_dir,
                 dt=1/40000, spike_threshold=-20, n_workers=None):
        """
        Initialize the optimized dataset pipeline
        
        Args:
            root_folder_path: Path to the root folder containing simulation data
            dt: Time step of the simulation (seconds)
            spike_threshold: Voltage threshold for spike detection (mV)
            n_workers: Number of worker processes (default: cpu_count())
        """
        self.root_folder_path = root_folder_path
        self.dt = dt
        self.spike_threshold = spike_threshold
        self.n_workers = n_workers if n_workers else min(cpu_count(), 8)  # Limit to 8 workers max
        
        # Output directories
        self.output_dir = output_dir
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create all necessary directories"""
        directories = [self.output_dir, self.train_dir, self.valid_dir, self.test_dir]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def load_data_single(self, exp):
        """Load simulation data for a single experiment (worker function)"""
        try:
            soma_path = [self.root_folder_path, exp, 'soma_v_array.npy']
            apic_v_path = [self.root_folder_path, exp, 'apic_v_array.npy']
            
            soma = np.load(os.path.join(*soma_path))
            apic_v = np.load(os.path.join(*apic_v_path))
        except FileNotFoundError:
            print(f"Warning: Data not found for exp: {exp}")
            return None

        # Read simulation parameters
        try:
            with open(os.path.join(self.root_folder_path, exp, 'simulation_params.json')) as f:
                simu_info = json.load(f)
            with open(os.path.join(self.root_folder_path, exp, 'section_synapse_df.csv')) as f:
                sec_syn_df = pd.read_csv(f)
        except Exception as e:
            print(f"Warning: Error reading params for exp {exp}: {e}")
            return None
        
        return soma, apic_v, simu_info, sec_syn_df

    def detect_spikes(self, voltage, threshold=None):
        """Detect spikes in voltage trace"""
        if threshold is None:
            threshold = self.spike_threshold
        peaks, _ = find_peaks(voltage, height=threshold)
        # Convert peak indices to actual time in milliseconds
        # peak_times = np.round(peaks).astype(int) # Low resolution
        peak_times = np.round(peaks * self.dt * 1000).astype(int) # High resolution
        return peak_times.tolist()

    def extract_spike_dict(self, sec_syn_df, syn_type, key='spike_train_bg', list_idx=0):
        """Extract spike times for a specific synapse type with segment matching"""
        # 加载 all_segments_noaxon.csv
        segments_df = pd.read_csv('all_segments_noaxon.csv')
        
        spike_dict = {}
        df = sec_syn_df[sec_syn_df['type'] == syn_type]
        
        for i, (idx, row) in enumerate(df.iterrows()):
            try:
                spike_list = ast.literal_eval(row[key])
                spike_times = spike_list[list_idx] if isinstance(spike_list, list) and len(spike_list) > abs(list_idx) else []
            except Exception:
                spike_times = []
            
            # 获取 section_synapse 和 loc
            section_synapse = row.get('section_synapse', '')
            loc = row.get('loc', 0.5)
            
            # 在 segments_df 中找到匹配的 section_name
            matching_segments = segments_df[segments_df['section_name'] == section_synapse]
            
            if not matching_segments.empty:
                # 找到最接近的 x_position
                x_positions = matching_segments['x_position'].values
                closest_idx = np.argmin(np.abs(x_positions - loc))
                segment_index = matching_segments.index[closest_idx]
                
                # 如果这个索引已经存在，合并 spike times
                if segment_index in spike_dict:
                    if isinstance(spike_dict[segment_index], list):
                        spike_dict[segment_index].extend(spike_times)
                    else:
                        spike_dict[segment_index] = [spike_dict[segment_index]] + spike_times
                else:
                    spike_dict[segment_index] = spike_times
            # else:
            #     # 如果没有找到匹配的 section，使用原来的索引
            #     if i in spike_dict:
            #         if isinstance(spike_dict[i], list):
            #             spike_dict[i].extend(spike_times)
            #         else:
            #             spike_dict[i] = [spike_dict[i]] + spike_times
            #     else:
            #         spike_dict[i] = spike_times
        
        if syn_type == 'A':
            spike_dict = {idx: spike_dict.get(idx, []) for idx in segments_df.index if idx != 0}
        else:
            spike_dict = {idx: spike_dict.get(idx, []) for idx in segments_df.index}
            
        return spike_dict

    def convert_single_experiment(self, exp):
        """Convert single experiment data to the required format (worker function)"""
        try:
            # Load data
            result = self.load_data_single(exp)
            if result is None:
                return None
            
            soma, apic_v, simu_info, sec_syn_df = result
            
            num_time = soma.shape[0]
            # Exclude first 100ms
            exclude_time_ms = 100
            start_idx = int(exclude_time_ms / (self.dt * 1000))
            
            # High resolution data (0.025ms)
            recordingTimeHighRes = np.arange(num_time - start_idx) * self.dt * 1000  # ms
            somaVoltageHighRes = soma[start_idx:, 0, 0] if soma.ndim == 3 else soma[start_idx:].squeeze()
            nexusVoltageHighRes = apic_v[start_idx:, 0, 0] if apic_v.ndim == 3 else apic_v[start_idx:].squeeze()
            
            # Resample to low resolution (1ms)
            ratio = int(1 / (self.dt * 1000))  # 40 for 1ms/0.025ms
            
            # Ensure length is a multiple of ratio
            new_length = (len(somaVoltageHighRes) // ratio) * ratio
            somaVoltageHighResTruncated = somaVoltageHighRes[:new_length]
            nexusVoltageHighResTruncated = nexusVoltageHighRes[:new_length]
            
            # Resample
            somaVoltageLowRes = np.mean(somaVoltageHighResTruncated.reshape(-1, ratio), axis=1)
            nexusVoltageLowRes = np.mean(nexusVoltageHighResTruncated.reshape(-1, ratio), axis=1)
            recordingTimeLowRes = np.arange(len(somaVoltageLowRes))  # 1ms intervals
            
            # Extract and filter spike times
            exInputSpikeTimes = self.extract_spike_dict(sec_syn_df, 'A', key='spike_train_bg', list_idx=0)
            inhInputSpikeTimes = self.extract_spike_dict(sec_syn_df, 'B', key='spike_train_bg', list_idx=0)
            
            # Filter input spike times (exclude first 100ms and adjust time)
            filtered_ex_spikes = {}
            filtered_inh_spikes = {}
            for syn_id, spike_times in exInputSpikeTimes.items():
                filtered_ex_spikes[syn_id] = np.array([t - exclude_time_ms for t in spike_times if t >= exclude_time_ms])
            for syn_id, spike_times in inhInputSpikeTimes.items():
                filtered_inh_spikes[syn_id] = np.array([t - exclude_time_ms for t in spike_times if t >= exclude_time_ms])
            
            # Detect output spikes
            # The time in the soma trace has already been adjusted by 100ms, so no need to subtract 100ms
            outputSpikeTimes = np.array(self.detect_spikes(somaVoltageHighRes)) # Must be high resolution !
            
            sim_dict = {
                'recordingTimeHighRes': recordingTimeHighRes,
                'somaVoltageHighRes': somaVoltageHighRes,
                'nexusVoltageHighRes': nexusVoltageHighRes,
                'recordingTimeLowRes': recordingTimeLowRes,
                'somaVoltageLowRes': somaVoltageLowRes,
                'nexusVoltageLowRes': nexusVoltageLowRes,
                'exInputSpikeTimes': filtered_ex_spikes,
                'inhInputSpikeTimes': filtered_inh_spikes,
                'outputSpikeTimes': outputSpikeTimes,
            }
            
            return sim_dict, simu_info
            
        except Exception as e:
            print(f"Error processing experiment {exp}: {e}")
            return None

    def convert_from_paths_parallel(self, exp_paths, batch_size=50):
        """Convert multiple experiments using parallel processing"""
        all_sim_dicts = []
        params_list = []
        
        print(f"Processing {len(exp_paths)} experiments using {self.n_workers} workers...")
        
        # Process in batches to avoid memory issues
        for i in tqdm.tqdm(range(0, len(exp_paths), batch_size), desc="Processing batches"):
            batch_paths = exp_paths[i:i + batch_size]
            
            # Use ProcessPoolExecutor for parallel processing
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                # Submit all tasks in the batch
                future_to_exp = {executor.submit(self.convert_single_experiment, exp): exp 
                               for exp in batch_paths}
                
                # Collect results as they complete
                for future in as_completed(future_to_exp):
                    exp = future_to_exp[future]
                    try:
                        result = future.result()
                        if result is not None:
                            sim_dict, simu_info = result
                            all_sim_dicts.append(sim_dict)
                            params_list.append(simu_info)
                    except Exception as e:
                        print(f"Error processing {exp}: {e}")
        
        params = params_list[0] if params_list else {}
        final_data = {
            'Params': params,
            'Results': {
                'listOfSingleSimulationDicts': all_sim_dicts
            }
        }
        
        print(f"Successfully processed {len(all_sim_dicts)} experiments")
        return final_data

    def save_to_pickle(self, final_data, filename):
        """Save data to pickle file"""
        with open(filename, 'wb') as f:
            pickle.dump(final_data, f)

    def split_pickle_file(self, input_file='output.pkl', num_files=10):
        """Split large pickle file into smaller files"""
        # Read the large pickle file
        with open(input_file, 'rb') as f:
            data = pickle.load(f)

        all_trials = data['Results']['listOfSingleSimulationDicts']
        total_trials = len(all_trials)
        trials_per_file = total_trials // num_files

        print(f"Splitting {total_trials} trials into {num_files} files...")

        for i in tqdm.tqdm(range(num_files), desc="Splitting files"):
            start = i * trials_per_file
            end = (i + 1) * trials_per_file if i < (num_files - 1) else total_trials
            sub_trials = all_trials[start:end]
            
            # Create sub-dictionary
            sub_data = {
                'Params': data['Params'],
                'Results': {
                    'listOfSingleSimulationDicts': sub_trials
                }
            }
            
            # Save as pickle
            filename = os.path.join(self.output_dir, f'L5PC_sim__Output_spikes_{i:04d}.p')
            with open(filename, 'wb') as f_out:
                pickle.dump(sub_data, f_out)

        print("Splitting completed!")

    def organize_dataset(self, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
        """Organize split files into train/validation/test sets"""
        # Get all .p files
        files = sorted(glob.glob(os.path.join(self.output_dir, '*.p')))
        
        if not files:
            print(f"No .p files found in {self.output_dir}")
            return
        
        total_files = len(files)
        train_count = int(total_files * train_ratio)
        valid_count = int(total_files * valid_ratio)
        
        # Split files
        train_files = files[:train_count]
        valid_files = files[train_count:train_count + valid_count]
        test_files = files[train_count + valid_count:]
        
        # Copy files to corresponding directories
        self._copy_files(train_files, self.train_dir)
        self._copy_files(valid_files, self.valid_dir)
        self._copy_files(test_files, self.test_dir)
        
        print(f"Dataset organized:")
        print(f"  Training set: {len(train_files)} files")
        print(f"  Validation set: {len(valid_files)} files")
        print(f"  Test set: {len(test_files)} files")

    def _copy_files(self, file_list, target_dir):
        """Copy files to target directory"""
        for file_path in tqdm.tqdm(file_list, desc=f"Copying to {os.path.basename(target_dir)}"):
            filename = os.path.basename(file_path)
            shutil.copy2(file_path, os.path.join(target_dir, filename))

    def run_full_pipeline(self, exp_paths, num_files=10, batch_size=50, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
        """
        Run the complete optimized pipeline from simulation data to organized dataset
        
        Args:
            exp_paths: List of experiment paths
            num_files: Number of files to split the data into
            train_ratio: Ratio of files for training
            valid_ratio: Ratio of files for validation
            test_ratio: Ratio of files for testing
            batch_size: Number of experiments to process in each batch
        """
        print("Starting optimized dataset pipeline...")
        print(f"Total experiments: {len(exp_paths)}")
        print(f"Workers: {self.n_workers}")
        print(f"Batch size: {batch_size}")
        
        # Step 1: Convert simulation data to pickle format (parallel)
        print("Step 1: Converting simulation data (parallel processing)...")
        final_data = self.convert_from_paths_parallel(exp_paths, batch_size)
        self.save_to_pickle(final_data, 'output.pkl')
        print("Step 1 completed: output.pkl created")
        
        # Step 2: Split large pickle file into smaller files
        print("Step 2: Splitting pickle file...")
        self.split_pickle_file('output.pkl', num_files)
        print("Step 2 completed: Files split")
        
        # Step 3: Organize files into train/validation/test sets
        print("Step 3: Organizing dataset...")
        self.organize_dataset(train_ratio, valid_ratio, test_ratio)
        print("Step 3 completed: Dataset organized")
        
        print("Optimized pipeline completed successfully!")


# Example usage
if __name__ == "__main__":
    # Define experiment paths
    exp_paths = [f'basal_range0_clus_invivo_NATURAL_funcgroup2_var2/1/{i}' for i in range(1, 3001)]
    
    output_dir = '/G/results/aim2_sjc/Data/full_output_dataset_funcgroup2_var2/'
    train_dir = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut_SJC_funcgroup2_var2/data/L5PC_NMDA_train/'
    valid_dir = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut_SJC_funcgroup2_var2/data/L5PC_NMDA_valid/'
    test_dir = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut_SJC_funcgroup2_var2/data/L5PC_NMDA_test/'

    # Create optimized pipeline instance
    pipeline = OptimizedDatasetPipeline(
        root_folder_path='/G/results/simulation/',
        output_dir=output_dir,
        train_dir=train_dir,
        valid_dir=valid_dir,
        test_dir=test_dir,
        n_workers=3,  # Adjust based on your CPU cores
        dt=1/40000,
        spike_threshold=-40
    )
    
    # Run the complete optimized pipeline
    pipeline.run_full_pipeline(
        exp_paths=exp_paths, 
        num_files=30, 
        batch_size=10  # Process 50 experiments at a time
    ) 