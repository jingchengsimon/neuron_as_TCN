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

try:
    # Optional dependency used in the notebook version
    from scipy.ndimage import gaussian_filter1d
except Exception:
    gaussian_filter1d = None

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

def _gaussian_smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    """
    Smooth a 1D signal with Gaussian kernel. Uses scipy if available; otherwise falls back to a simple moving average.
    """
    if gaussian_filter1d is not None:
        return gaussian_filter1d(x, sigma=sigma)
    # Fallback: simple moving average with window size proportional to sigma
    win = max(1, int(round(sigma * 2)))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, kernel, mode='same')

def compute_firing_rate(spike_times, t_start, t_end, dt, sigma, num_syn):
    """
    Convert spike times to a smoothed firing rate (Hz) over [t_start, t_end] with bin size dt (ms).
    """
    n_bins = int((t_end - t_start) / dt) + 1
    spike_train = np.zeros(n_bins, dtype=np.float32)
    for t in spike_times:
        if t_start <= t < t_end:
            idx = int(t - t_start)
            if 0 <= idx < n_bins:
                spike_train[idx] += 1.0
    smoothed = _gaussian_smooth_1d(spike_train, sigma=sigma)
    # Convert to Hz and normalize by number of synapses
    denom = (dt / 1000.0) * max(1, int(num_syn))
    return smoothed / denom

def plot_simulation_detail(
    sim,
    t_start=0,
    t_end=6000,
    res_label='Low',
    ex_syn_num=9,
    inh_syn_num=9,
    sigma=20,
    save_path=None,
):
    """
    Plot a detailed view for a single simulation (voltage, output spikes, input rasters, firing rates).
    Adapted from `jupyter_notebooks/pickle_load.ipynb`.
    """
    time = sim[f'recordingTime{res_label}Res']
    mask = (time >= t_start) & (time <= t_end)
    time_sel = time[mask]
    soma_v = sim[f'somaVoltage{res_label}Res'][mask]

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(min(40, 10 * (t_end - t_start) / 1000.0), 6 * 4.3 / 2.8),
        sharex=True,
        gridspec_kw={'height_ratios': [1, 0.3, 1.5, 1.5]},
    )

    for ax in axes:
        ax.set_xlim(t_start, t_end)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Soma voltage
    axes[0].set_ylabel('Voltage (mV)')
    axes[0].set_ylim(-80, 0)
    axes[0].plot(time_sel, soma_v, label='Soma Voltage', color='k')
    axes[0].set_title(f'Soma Voltage Trace {res_label} Res')
    axes[0].legend(loc='upper left')

    # Output spikes
    output_spike_times = sim.get('outputSpikeTimes', np.array([]))
    output_spike_times = output_spike_times[(output_spike_times >= t_start) & (output_spike_times <= t_end)]
    axes[1].vlines(output_spike_times, 0, 1, color='purple', linewidth=1.0)
    axes[1].set_ylabel('Output')
    axes[1].set_yticks([1])
    axes[1].set_title('Output Spike Raster')

    # Input rasters
    spikes_exc = sim['exInputSpikeTimes']
    spikes_inh = sim['inhInputSpikeTimes']
    for spikes, color, offset in [(spikes_exc, 'b', 0), (spikes_inh, 'r', len(spikes_exc))]:
        for syn_id, spike_times in spikes.items():
            spike_times_sel = [t for t in spike_times if t_start <= t <= t_end]
            axes[2].vlines(
                spike_times_sel,
                offset + syn_id - 0.4,
                offset + syn_id + 0.4,
                color=color,
                linewidth=2,
            )
    axes[2].set_title('Excitatory and Inhibitory Input Raster')
    axes[2].set_ylabel('Input Syn ID')
    axes[2].set_xlabel('Time (ms)')

    # Firing rates
    all_spike_times_exc = []
    for spike_times in spikes_exc.values():
        all_spike_times_exc.extend([t for t in spike_times if t_start <= t <= t_end])
    all_spike_times_inh = []
    for spike_times in spikes_inh.values():
        all_spike_times_inh.extend([t for t in spike_times if t_start <= t <= t_end])

    dt = 1  # ms
    time_bins = np.arange(t_start, t_end + dt, dt)
    firing_rate_exc = compute_firing_rate(all_spike_times_exc, t_start, t_end, dt, sigma=sigma, num_syn=ex_syn_num)
    firing_rate_inh = compute_firing_rate(all_spike_times_inh, t_start, t_end, dt, sigma=sigma, num_syn=inh_syn_num)

    axes[3].plot(time_bins, firing_rate_exc, color='b', label='Excitatory')
    axes[3].plot(time_bins, firing_rate_inh, color='r', label='Inhibitory')
    axes[3].set_ylabel('Firing rate (Hz)')
    axes[3].set_xlabel('Time (ms)')
    axes[3].set_title('Instantaneous Firing Rate (Gaussian smoothing)')
    axes[3].legend()

    print(f'Average Excitatory firing rate: {np.mean(firing_rate_exc):.2f} Hz')
    print(f'Average Inhibitory firing rate: {np.mean(firing_rate_inh):.2f} Hz')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Simulation detail saved to: {save_path}")
    plt.close(fig)

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

        # Load a representative single-file set for detailed per-simulation visualization
        # (matches the notebook pattern: "*0009.p")
        single_file_list = sorted(glob.glob(os.path.join(root_folder_path, '*0009.p')))
        if len(single_file_list) == 0:
            # Fallback: use the first available file
            single_file_list = [file_list[0]]
        simulation_list_ori = merge_simulation_pickles(single_file_list)
        
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

        # Plot simulation details for selected indices (notebook snippet: epoch_idx in [2, 3])
        detail_indices_1based = [4, 5, 6, 7]
        for epoch_idx in detail_indices_1based:
            sim_idx = epoch_idx - 1
            if 0 <= sim_idx < len(simulation_list_ori):
                detail_path = os.path.join(output_dir, f'simulation_detail_{epoch_idx}.png')
                plot_simulation_detail(
                    simulation_list_ori[sim_idx],
                    t_start=0,
                    t_end=6000,
                    res_label='Low',
                    ex_syn_num=ex_syn_num,
                    inh_syn_num=inh_syn_num,
                    save_path=detail_path,
                )
    
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

