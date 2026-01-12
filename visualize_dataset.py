#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize reduce_model dataset distribution
Author: Jingcheng Shi
Date: 2025-01-XX
"""

import os
import sys
import glob
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for remote execution
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from contextlib import redirect_stdout
from io import StringIO

def merge_simulation_pickles(file_list):
    """
    Merge multiple simulation pickle files into a single list
    
    Args:
        file_list: List of pickle file paths
        
    Returns:
        simulation_list: List of simulation dictionaries
    """
    # Initialize merged data structure
    data = {
        'Params': None,  # Use parameters from first file
        'Results': {
            'listOfSingleSimulationDicts': []
        }
    }

    for file_path in file_list:
        with open(file_path, 'rb') as f:
            sing_data = pickle.load(f)

            # Record parameters (keep only first)
            if data['Params'] is None:
                data['Params'] = sing_data['Params']

            # Merge simulation results
            data['Results']['listOfSingleSimulationDicts'].extend(
                sing_data['Results']['listOfSingleSimulationDicts']
            )

    print(f"Loaded {len(file_list)} files.")
    print(f"Total simulations merged: {len(data['Results']['listOfSingleSimulationDicts'])}")

    results = data['Results']
    simulation_list = results['listOfSingleSimulationDicts']

    print(f"Number of simulations: {len(simulation_list)}")
    if len(simulation_list) > 0:
        print(len(simulation_list[0]['recordingTimeLowRes']))

    return simulation_list

def plot_firing_rate_histograms(simulation_list, res_label='Low', ex_syn_num=9, inh_syn_num=9, save_path=None):
    """
    Plot firing rate histograms for output, excitatory, and inhibitory inputs
    
    Args:
        simulation_list: List of simulation dictionaries
        res_label: Resolution label ('Low' or 'High')
        ex_syn_num: Number of excitatory synapses
        inh_syn_num: Number of inhibitory synapses
        save_path: Path to save the figure
    """
    firing_rates, ex_firing_rates, inh_firing_rates = [], [], []

    print('Number of simulations:', len(simulation_list))
    for simu_idx in range(len(simulation_list)):
        recording_time = simulation_list[simu_idx][f'recordingTime{res_label}Res']
        duration_seconds = recording_time[-1] / 1000.0  # ms -> s

        firing_rate = len(simulation_list[simu_idx]['outputSpikeTimes']) / duration_seconds
        
        ex_spikes = simulation_list[simu_idx]['exInputSpikeTimes']
        inh_spikes = simulation_list[simu_idx]['inhInputSpikeTimes']

        ex_rate = sum(len(spike_times) for spike_times in ex_spikes.values()) / (ex_syn_num * duration_seconds)
        inh_rate = sum(len(spike_times) for spike_times in inh_spikes.values()) / (inh_syn_num * duration_seconds) if inh_syn_num > 0 else 0
        
        firing_rates.append(firing_rate)
        ex_firing_rates.append(ex_rate)
        inh_firing_rates.append(inh_rate)

        if simu_idx == 0:
            print(f'Recording duration: {duration_seconds:.1f} seconds')
            print('number of ex segment:', len(ex_spikes))
            print('number of inh segment:', len(inh_spikes))

    avg_firing_rate = sum(firing_rates) / len(firing_rates)
    avg_ex_firing_rate = sum(ex_firing_rates) / len(ex_firing_rates)
    avg_inh_firing_rate = sum(inh_firing_rates) / len(inh_firing_rates)

    median_firing_rate = sorted(firing_rates)[len(firing_rates) // 2]
    median_ex_firing_rate = sorted(ex_firing_rates)[len(ex_firing_rates) // 2]
    median_inh_firing_rate = sorted(inh_firing_rates)[len(inh_firing_rates) // 2]

    plt.figure(figsize=(18, 4))
    for ax_idx in range(1, 4):
        plt.subplot(1, 3, ax_idx)
        plt.xlabel('Firing Rate (spikes per second)')
        plt.ylabel('Counts')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        
        if ax_idx == 1:
            plt.hist(firing_rates, bins=10, color='blue', alpha=0.7)
            plt.title('Output Firing Rates')
            plt.axvline(median_firing_rate, color='red', linestyle='dashed', linewidth=1, label=f'Median: {median_firing_rate:.2f}')
            plt.axvline(avg_firing_rate, color='black', linestyle='dashed', linewidth=1, label=f'Average: {avg_firing_rate:.2f}')
        elif ax_idx == 2:
            plt.hist(ex_firing_rates, bins=30, color='orange', alpha=0.7)
            plt.title('Excitatory Input Firing Rates')
            plt.axvline(median_ex_firing_rate, color='red', linestyle='dashed', linewidth=1, label=f'Median: {median_ex_firing_rate:.2f}')
            plt.axvline(avg_ex_firing_rate, color='black', linestyle='dashed', linewidth=1, label=f'Average: {avg_ex_firing_rate:.2f}')
        elif ax_idx == 3:
            plt.hist(inh_firing_rates, bins=30, color='green', alpha=0.7)
            plt.title('Inhibitory Input Firing Rates')
            plt.axvline(median_inh_firing_rate, color='red', linestyle='dashed', linewidth=1, label=f'Median: {median_inh_firing_rate:.2f}')
            plt.axvline(avg_inh_firing_rate, color='black', linestyle='dashed', linewidth=1, label=f'Average: {avg_inh_firing_rate:.2f}')
        plt.legend()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Histogram saved to: {save_path}")
    else:
        plt.savefig('firing_rate_histograms.png', dpi=150, bbox_inches='tight')
    
    plt.close()

def main():
    # Configuration
    root_folder_path = '/G/results/aim2_sjc/Data/reduce_model_output_dataset'
    output_dir = './results/2_dataset_load/reduce_model'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all pickle files
    file_pattern = os.path.join(root_folder_path, '*.p')
    file_list = sorted(glob.glob(file_pattern))
    
    if len(file_list) == 0:
        print(f"ERROR: No pickle files found in {root_folder_path}")
        print(f"Pattern used: {file_pattern}")
        sys.exit(1)
    
    print(f"Found {len(file_list)} pickle files")
    
    # Capture all print output
    output_buffer = StringIO()
    
    # Redirect stdout to capture prints
    with redirect_stdout(output_buffer):
        # Load and merge simulation data
        simulation_list = merge_simulation_pickles(file_list)
        
        # Detect number of segments from first simulation
        if len(simulation_list) > 0:
            first_sim = simulation_list[0]
            ex_syn_num = len(first_sim['exInputSpikeTimes'])
            inh_syn_num = len(first_sim['inhInputSpikeTimes'])
            print(f"Detected: ex_syn_num={ex_syn_num}, inh_syn_num={inh_syn_num}")
        else:
            ex_syn_num = 9  # Default fallback
            inh_syn_num = 9
            print(f"Warning: No simulations found, using default: ex_syn_num={ex_syn_num}, inh_syn_num={inh_syn_num}")
        
        # Plot histograms
        histogram_path = os.path.join(output_dir, 'firing_rate_histograms.png')
        plot_firing_rate_histograms(simulation_list, res_label='Low', 
                                   ex_syn_num=ex_syn_num, inh_syn_num=inh_syn_num,
                                   save_path=histogram_path)
    
    # Get captured output
    captured_output = output_buffer.getvalue()
    
    # Print to console as well
    print(captured_output)
    
    # Save output to text file
    output_text_path = os.path.join(output_dir, 'dataset_info.txt')
    with open(output_text_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Reduce Model Dataset Information\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Data directory: {root_folder_path}\n")
        f.write(f"Number of files: {len(file_list)}\n\n")
        f.write("=" * 60 + "\n")
        f.write("Dataset Statistics:\n")
        f.write("=" * 60 + "\n\n")
        f.write(captured_output)
    
    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"Results saved to: {output_dir}")
    print(f"  - Histogram: {histogram_path}")
    print(f"  - Statistics: {output_text_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

