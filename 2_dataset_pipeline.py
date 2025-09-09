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
from pathlib import Path
import time
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
        self.root_folder_path = Path(root_folder_path)
        self.dt = dt
        self.spike_threshold = spike_threshold
        self.n_workers = n_workers if n_workers else min(cpu_count(), 12)  # Increased worker limit
        
        # Output directories
        self.output_dir = Path(output_dir)
        self.train_dir = Path(train_dir)
        self.valid_dir = Path(valid_dir)
        self.test_dir = Path(test_dir)
        
        # Pre-load segments data for efficiency
        self.segments_df = pd.read_csv('all_segments_noaxon.csv')
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create all necessary directories"""
        for directory in [self.output_dir, self.train_dir, self.valid_dir, self.test_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_data_single(self, exp):
        """Load simulation data for a single experiment (worker function)"""
        try:
            exp_path = self.root_folder_path / exp
            soma = np.load(exp_path / 'soma_v_array.npy')
            apic_v = np.load(exp_path / 'apic_v_array.npy')
            
            # Read simulation parameters
            with open(exp_path / 'simulation_params.json') as f:
                simu_info = json.load(f)
            sec_syn_df = pd.read_csv(exp_path / 'section_synapse_df.csv')
            
            return soma, apic_v, simu_info, sec_syn_df
        except Exception as e:
            print(f"Warning: Error loading exp {exp}: {e}")
            return None

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
        spike_dict = {}
        df = sec_syn_df[sec_syn_df['type'] == syn_type]
        
        for _, row in df.iterrows():
            try:
                spike_list = ast.literal_eval(row[key])
                spike_times = spike_list[list_idx] if isinstance(spike_list, list) and len(spike_list) > abs(list_idx) else []
            except Exception:
                spike_times = []
            
            if not spike_times:
                continue
                
            # Get section_synapse and loc
            section_synapse = row.get('section_synapse', '')
            loc = row.get('loc', 0.5)
            
            # Find matching section_name in segments_df
            matching_segments = self.segments_df[self.segments_df['section_name'] == section_synapse]
            
            if not matching_segments.empty:
                # Find closest x_position
                x_positions = matching_segments['x_position'].values
                closest_idx = np.argmin(np.abs(x_positions - loc))
                segment_index = matching_segments.index[closest_idx]
                
                # Merge spike times
                if segment_index in spike_dict:
                    spike_dict[segment_index].extend(spike_times)
                else:
                    spike_dict[segment_index] = spike_times
        
        # Ensure all segments have corresponding spike list
        if syn_type == 'A':
            spike_dict = {idx: spike_dict.get(idx, []) for idx in self.segments_df.index if idx != 0}
        else:
            spike_dict = {idx: spike_dict.get(idx, []) for idx in self.segments_df.index}
            
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
            exclude_time_ms = 100
            start_idx = int(exclude_time_ms / (self.dt * 1000))
            
            # Extract voltage data (memory efficient)
            soma_voltage = soma[start_idx:, 0, 0] if soma.ndim == 3 else soma[start_idx:].squeeze()
            nexus_voltage = apic_v[start_idx:, 0, 0] if apic_v.ndim == 3 else apic_v[start_idx:].squeeze()
            
            # Clear original arrays to save memory
            del soma, apic_v
            
            # Resample to low resolution (1ms)
            ratio = int(1 / (self.dt * 1000))  # 40 for 1ms/0.025ms
            new_length = (len(soma_voltage) // ratio) * ratio
            
            # Resample efficiently
            soma_voltage_low = np.mean(soma_voltage[:new_length].reshape(-1, ratio), axis=1)
            nexus_voltage_low = np.mean(nexus_voltage[:new_length].reshape(-1, ratio), axis=1)
            
            # Extract and filter spike times
            ex_spikes = self.extract_spike_dict(sec_syn_df, 'A', key='spike_train_bg', list_idx=0)
            inh_spikes = self.extract_spike_dict(sec_syn_df, 'B', key='spike_train_bg', list_idx=0)
            
            # Filter input spike times efficiently
            filtered_ex_spikes = {k: np.array([t - exclude_time_ms for t in v if t >= exclude_time_ms]) 
                                for k, v in ex_spikes.items()}
            filtered_inh_spikes = {k: np.array([t - exclude_time_ms for t in v if t >= exclude_time_ms]) 
                                 for k, v in inh_spikes.items()}
            
            # Detect output spikes
            output_spikes = np.array(self.detect_spikes(soma_voltage))
            
            sim_dict = {
                'recordingTimeHighRes': np.arange(num_time - start_idx) * self.dt * 1000,
                'somaVoltageHighRes': soma_voltage,
                'nexusVoltageHighRes': nexus_voltage,
                'recordingTimeLowRes': np.arange(len(soma_voltage_low)),
                'somaVoltageLowRes': soma_voltage_low,
                'nexusVoltageLowRes': nexus_voltage_low,
                'exInputSpikeTimes': filtered_ex_spikes,
                'inhInputSpikeTimes': filtered_inh_spikes,
                'outputSpikeTimes': output_spikes,
            }
            
            return sim_dict, simu_info
            
        except Exception as e:
            print(f"Error processing experiment {exp}: {e}")
            return None

    def convert_from_paths_parallel(self, exp_paths, batch_size=100):
        """Convert multiple experiments using optimized parallel processing"""
        all_sim_dicts = []
        trial_mapping = {}
        
        print(f"Processing {len(exp_paths)} experiments using {self.n_workers} workers...")
        
        # Use larger batch size and more efficient parallel processing
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks
            future_to_exp = {
                executor.submit(self.convert_single_experiment, exp): exp 
                for exp in exp_paths
            }
            
            # Collect results
            for future in tqdm.tqdm(as_completed(future_to_exp), 
                                  total=len(exp_paths), desc="Processing experiments"):
                exp = future_to_exp[future]
                try:
                    result = future.result()
                    if result is not None:
                        sim_dict, simu_info = result
                        sim_index = len(all_sim_dicts)
                        trial_index = self._extract_trial_index_from_path(exp)
                        
                        trial_mapping[sim_index] = {
                            'trial_index': trial_index,
                            'exp_path': exp,
                            'sim_index': sim_index
                        }
                        
                        all_sim_dicts.append(sim_dict)
                except Exception as e:
                    print(f"Error processing {exp}: {e}")
        
        final_data = {
            'Params': simu_info if all_sim_dicts else {},
            'Results': {'listOfSingleSimulationDicts': all_sim_dicts},
            'TrialMapping': trial_mapping
        }
        
        print(f"Successfully processed {len(all_sim_dicts)} experiments")
        return final_data

    def save_to_pickle(self, final_data, filename):
        """Save data to pickle file"""
        with open(filename, 'wb') as f:
            pickle.dump(final_data, f)

    def split_pickle_file(self, input_file='output.pkl', num_files=10):
        """Split large pickle file into smaller files with trial index tracking"""
        with open(input_file, 'rb') as f:
            data = pickle.load(f)

        all_trials = data['Results']['listOfSingleSimulationDicts']
        trial_mapping = data.get('TrialMapping', {})
        total_trials = len(all_trials)
        trials_per_file = total_trials // num_files

        print(f"Splitting {total_trials} trials into {num_files} files...")

        for i in tqdm.tqdm(range(num_files), desc="Splitting files"):
            start = i * trials_per_file
            end = (i + 1) * trials_per_file if i < (num_files - 1) else total_trials
            
            # Create sub-data
            sub_data = {
                'Params': data['Params'],
                'Results': {'listOfSingleSimulationDicts': all_trials[start:end]},
                'TrialMapping': {k: v for k, v in trial_mapping.items() if start <= k < end}
            }
            
            # Save file
            filename = self.output_dir / f'L5PC_sim__Output_spikes_{i:04d}.p'
            with open(filename, 'wb') as f_out:
                pickle.dump(sub_data, f_out)

        print("Splitting completed!")

    def organize_dataset(self, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
        """Organize split files into train/validation/test sets with trial index tracking"""
        files = sorted(self.output_dir.glob('*.p'))
        
        if not files:
            print(f"No .p files found in {self.output_dir}")
            return []
        
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
        
        # Extract test set trial indices
        test_trial_indices = self._extract_trial_indices_from_files(test_files)
        
        print(f"Dataset organized:")
        print(f"  Training set: {len(train_files)} files")
        print(f"  Validation set: {len(valid_files)} files")
        print(f"  Test set: {len(test_files)} files")
        print(f"  Test set trial indices: {sorted(test_trial_indices)}")
        
        # Save test set trial indices to file
        self._save_test_trial_indices(test_trial_indices)
        
        return test_trial_indices

    def _copy_files(self, file_list, target_dir):
        """Copy files to target directory"""
        for file_path in tqdm.tqdm(file_list, desc=f"Copying to {target_dir.name}"):
            shutil.copy2(file_path, target_dir / file_path.name)
    
    def _extract_trial_index_from_path(self, exp_path):
        """Extract trial index from experiment path"""
        # Assume path format: 'basal_range0_clus_invivo_NATURAL_funcgroup2_var2_AMPA/1/{trial_index}'
        parts = exp_path.split('/')
        if len(parts) >= 3:
            try:
                return int(parts[-1])  # Last part is trial index
            except ValueError:
                return None
        return None
    
    def _extract_trial_indices_from_files(self, file_list):
        """Extract trial indices from file list"""
        all_trial_indices = []
        
        for file_path in file_list:
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                trial_mapping = data.get('TrialMapping', {})
                file_trial_indices = [mapping['trial_index'] for mapping in trial_mapping.values()]
                all_trial_indices.extend(file_trial_indices)
                
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
        
        return all_trial_indices
    
    def _save_test_trial_indices(self, test_trial_indices):
        """Save test set trial indices to file"""
        test_indices_file = self.output_dir / 'test_trial_indices.json'
        with open(test_indices_file, 'w') as f:
            json.dump({
                'test_trial_indices': sorted(test_trial_indices),
                'total_test_trials': len(test_trial_indices)
            }, f, indent=2)
        
        print(f"Test trial indices saved to: {test_indices_file}")
    
    def load_test_trial_indices(self):
        """Load test set trial indices"""
        test_indices_file = self.output_dir / 'test_trial_indices.json'
        if test_indices_file.exists():
            with open(test_indices_file, 'r') as f:
                data = json.load(f)
            return data['test_trial_indices']
        else:
            print(f"Test trial indices file not found: {test_indices_file}")
            return []

    def run_full_pipeline(self, exp_paths, num_files=10, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
        """
        Run the complete optimized pipeline from simulation data to organized dataset
        
        Args:
            exp_paths: List of experiment paths
            num_files: Number of files to split the data into
            train_ratio: Ratio of files for training
            valid_ratio: Ratio of files for validation
            test_ratio: Ratio of files for testing
            
        Returns:
            test_trial_indices: List of trial indices in the test set
        """
        start_time = time.time()
        print(f"Starting optimized dataset pipeline...")
        print(f"Total experiments: {len(exp_paths)}")
        print(f"Workers: {self.n_workers}")
        
        # Step 1: Convert simulation data to pickle format (parallel)
        print("Step 1: Converting simulation data (parallel processing)...")
        step1_start = time.time()
        final_data = self.convert_from_paths_parallel(exp_paths)
        self.save_to_pickle(final_data, 'output.pkl')
        step1_time = time.time() - step1_start
        print(f"Step 1 completed: output.pkl created (took {step1_time:.2f}s)")
        
        # Step 2: Split large pickle file into smaller files
        print("Step 2: Splitting pickle file...")
        step2_start = time.time()
        self.split_pickle_file('output.pkl', num_files)
        step2_time = time.time() - step2_start
        print(f"Step 2 completed: Files split (took {step2_time:.2f}s)")
        
        # Step 3: Organize files into train/validation/test sets
        print("Step 3: Organizing dataset...")
        step3_start = time.time()
        test_trial_indices = self.organize_dataset(train_ratio, valid_ratio, test_ratio)
        step3_time = time.time() - step3_start
        print(f"Step 3 completed: Dataset organized (took {step3_time:.2f}s)")
        
        total_time = time.time() - start_time
        print("Optimized pipeline completed successfully!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per experiment: {total_time/len(exp_paths):.3f}s")
        print(f"Test set contains {len(test_trial_indices)} trials: {sorted(test_trial_indices)}")
        return test_trial_indices

# Example usage
if __name__ == "__main__":
    # Define base paths
    base_path = '/G/results/aim2_sjc'
    project_name = 'funcgroup2_var2'
    data_suffix = 'L5PC_NMDA'
    # Define experiment paths
    num_trials = 10000
    exp_paths = [f'basal_range0_clus_invivo_NATURAL_{project_name}/1/{i}' for i in range(1, num_trials + 1)]
    
    # Build directory paths using base paths
    output_dir = f'{base_path}/Data/full_output_dataset_{project_name}/'
    train_dir = f'{base_path}/Models_TCN/Single_Neuron_InOut_SJC_{project_name}/data/{data_suffix}_train/'
    valid_dir = f'{base_path}/Models_TCN/Single_Neuron_InOut_SJC_{project_name}/data/{data_suffix}_valid/'
    test_dir = f'{base_path}/Models_TCN/Single_Neuron_InOut_SJC_{project_name}/data/{data_suffix}_test/'

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
        num_files=num_trials // 100
    ) 