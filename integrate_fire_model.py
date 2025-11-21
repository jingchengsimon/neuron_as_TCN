"""
Integrate and Fire (I&F) Neuron Model Implementation

Based on the leaky I&F simulation described in the research paper.
Implements a neuron with exponential decay temporal kernel and Poisson input.

Model parameters:
- Membrane time constant: τ = 20ms
- Rest voltage: V_rest = -77mV
- Excitatory synapses: N_exc = 80, w_exc = 2mV, f_exc = 1.4Hz
- Inhibitory synapses: N_inh = 20, w_inh = -2mV, f_inh = 1.3Hz
- Simulation time: 6000ms
- Target output firing rate: ~0.9Hz
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import random
from pathlib import Path
import pickle


class IntegrateFireNeuron:
    """
    Integrate and Fire neuron model with exponential decay kernel.
    
    The membrane voltage is modeled as:
    V(t) = Σ w_i Σ K(t - t_i)
    
    Where K(t - t_i) = exp(-(t-t_i)/τ) * u(t-t_i) is the temporal kernel
    with exponential decay and τ = 20ms membrane time constant.
    """
    
    def __init__(self, 
                 tau: float = 20.0,  # membrane time constant (ms)
                 v_rest: float = -77.0,  # rest voltage (mV)
                 v_threshold: float = -60.0,  # spike threshold (mV) - adjusted based on voltage range
                 n_exc: int = 80,  # number of excitatory synapses
                 n_inh: int = 20,  # number of inhibitory synapses
                 w_exc: float = 2.8,  # excitatory synaptic weight (mV) - targeting ~0.9Hz
                 w_inh: float = -2.1,  # inhibitory synaptic weight (mV) - balanced
                 f_exc: float = 1.4,  # excitatory firing rate (Hz)
                 f_inh: float = 1.3,  # inhibitory firing rate (Hz)
                 dt: float = 0.1):  # time step (ms)
        
        self.tau = tau
        self.v_rest = v_rest
        self.v_threshold = v_threshold
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.w_exc = w_exc
        self.w_inh = w_inh
        self.f_exc = f_exc
        self.f_inh = f_inh
        self.dt = dt
        
        # Initialize membrane voltage
        self.v_membrane = v_rest
        
        # Store spike times for output
        self.output_spikes = []
        
    def generate_poisson_spikes(self, duration: float, rate: float, rng: np.random.RandomState = None) -> np.ndarray:
        """
        Generate spike times from a Poisson process.
        
        Args:
            duration: simulation duration (ms)
            rate: firing rate (Hz)
            rng: random number generator state (for independent processes)
            
        Returns:
            Array of spike times (ms)
        """
        # Use provided RNG or default to global numpy random
        if rng is None:
            rng = np.random
        
        # Convert rate from Hz to spikes per ms
        rate_per_ms = rate / 1000.0
        
        # Generate random spike times
        spike_times = []
        t = 0.0
        
        while t < duration:
            # Inter-spike interval from exponential distribution
            isi = rng.exponential(1.0 / rate_per_ms)
            t += isi
            if t < duration:
                spike_times.append(t)
                
        return np.array(spike_times)
    
    def exponential_kernel(self, t: float) -> float:
        """
        Exponential decay kernel: K(t) = exp(-t/τ) * u(t)
        
        Args:
            t: time since spike (ms)
            
        Returns:
            Kernel value
        """
        if t < 0:
            return 0.0
        return np.exp(-t / self.tau)
    
    def simulate(self, duration: float = 6000.0, base_seed: int = None) -> Tuple[np.ndarray, np.ndarray, List[float], List[np.ndarray], List[np.ndarray]]:
        """
        Simulate the I&F neuron for given duration.
        
        Args:
            duration: simulation duration (ms)
            base_seed: base random seed for generating independent Poisson processes
            
        Returns:
            Tuple of (time_array, voltage_trace, output_spike_times, exc_spikes_list, inh_spikes_list)
            exc_spikes_list and inh_spikes_list are lists of arrays, one per synapse
        """
        # Generate input spike trains
        print(f"Generating Poisson spike trains...")
        
        # Excitatory synapses - each with independent random seed
        exc_spikes = []
        total_exc_spikes = 0
        for i in range(self.n_exc):
            # Create independent random state for each synapse
            if base_seed is not None:
                # Use base_seed + synapse_index to ensure independence
                synapse_seed = base_seed * 10000 + i  # Multiply by large number to avoid overlap
            else:
                # Even without base_seed, use synapse index to ensure different sequences
                synapse_seed = i
            rng = np.random.RandomState(synapse_seed)
            spikes = self.generate_poisson_spikes(duration, self.f_exc, rng)
            exc_spikes.append(spikes)
            total_exc_spikes += len(spikes)
            
        # Inhibitory synapses - each with independent random seed
        inh_spikes = []
        total_inh_spikes = 0
        for i in range(self.n_inh):
            # Create independent random state for each synapse
            if base_seed is not None:
                # Use base_seed + synapse_index + offset to ensure independence from exc
                synapse_seed = base_seed * 10000 + 10000 + i  # Add offset for inhibitory synapses
            else:
                # Even without base_seed, use synapse index + offset to ensure different sequences
                synapse_seed = 10000 + i  # Offset for inhibitory synapses
            rng = np.random.RandomState(synapse_seed)
            spikes = self.generate_poisson_spikes(duration, self.f_inh, rng)
            inh_spikes.append(spikes)
            total_inh_spikes += len(spikes)
            
        print(f"Generated {total_exc_spikes} excitatory spikes across {self.n_exc} synapses")
        print(f"Generated {total_inh_spikes} inhibitory spikes across {self.n_inh} synapses")
        print(f"Expected exc rate: {total_exc_spikes/(self.n_exc * duration/1000):.2f} Hz per synapse")
        print(f"Expected inh rate: {total_inh_spikes/(self.n_inh * duration/1000):.2f} Hz per synapse")
        
        # Time array
        time_steps = int(duration / self.dt)
        time_array = np.arange(0, duration, self.dt)
        voltage_trace = np.zeros(time_steps)
        
        # Reset neuron state
        self.v_membrane = self.v_rest
        self.output_spikes = []
        
        print(f"Running simulation for {duration}ms...")
        
        # Track recent spikes for refractory period
        last_spike_time = -float('inf')
        refractory_period = 2.0  # ms
        
        # Track voltage statistics for debugging
        max_voltage = -float('inf')
        min_voltage = float('inf')
        
        # Main simulation loop
        for step, t in enumerate(time_array):
            # Skip if in refractory period (but still record actual voltage)
            if t - last_spike_time < refractory_period:
                # Keep voltage at rest during refractory, but record it
                self.v_membrane = self.v_rest
                voltage_trace[step] = self.v_rest
                continue
                
            # Calculate membrane voltage from all synaptic inputs
            v_total = 0.0
            
            # Excitatory contributions
            for syn_idx, spike_times in enumerate(exc_spikes):
                for spike_time in spike_times:
                    if spike_time <= t and spike_time > last_spike_time:
                        kernel_val = self.exponential_kernel(t - spike_time)
                        v_total += self.w_exc * kernel_val
            
            # Inhibitory contributions
            for syn_idx, spike_times in enumerate(inh_spikes):
                for spike_time in spike_times:
                    if spike_time <= t and spike_time > last_spike_time:
                        kernel_val = self.exponential_kernel(t - spike_time)
                        v_total += self.w_inh * kernel_val
            
            # Update membrane voltage
            self.v_membrane = self.v_rest + v_total
            voltage_trace[step] = self.v_membrane  # Record actual voltage value
            
            # Track voltage statistics
            max_voltage = max(max_voltage, self.v_membrane)
            min_voltage = min(min_voltage, self.v_membrane)
            
            # Check for spike threshold
            if self.v_membrane >= self.v_threshold:
                # Record output spike (voltage_trace already has the threshold/reached value)
                self.output_spikes.append(t)
                last_spike_time = t
                # Reset internal voltage for next step, but keep voltage_trace[step] as is
                self.v_membrane = self.v_rest
                # voltage_trace[step] will be modified in visualization only
        
        print(f"Voltage range: {min_voltage:.1f}mV to {max_voltage:.1f}mV")
        print(f"Threshold: {self.v_threshold}mV")
        print(f"Distance to threshold: {self.v_threshold - max_voltage:.1f}mV")
        
        # Return individual spike trains for each synapse (not concatenated)
        return time_array, voltage_trace, self.output_spikes, exc_spikes, inh_spikes
    
    def calculate_firing_rate(self, duration: float) -> float:
        """Calculate average firing rate from output spikes."""
        if len(self.output_spikes) == 0:
            return 0.0
        return len(self.output_spikes) / (duration / 1000.0)  # Convert to Hz
    
    def plot_results(self, time_array: np.ndarray, voltage_trace: np.ndarray, 
                    output_spikes: List[float], exc_spikes_list: List[np.ndarray], inh_spikes_list: List[np.ndarray],
                    duration: float, seed: int, save_fig: bool = True):
        """
        Plot simulation results.
        
        Args:
            time_array: time points (ms)
            voltage_trace: membrane voltage trace (mV) - original data, not modified
            output_spikes: output spike times (ms)
            exc_spikes_list: list of excitatory spike arrays, one per synapse
            inh_spikes_list: list of inhibitory spike arrays, one per synapse
            duration: simulation duration (ms)
            seed: random seed for filename
            save_fig: whether to save the figure
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Prepare voltage trace for visualization: set spike times to 0 (for visualization only)
        voltage_trace_plot = voltage_trace.copy()  # Original data unchanged
        if output_spikes:
            for spike_time in output_spikes:
                # Find the time step closest to spike time
                spike_idx = np.argmin(np.abs(time_array - spike_time))
                if spike_idx < len(voltage_trace_plot):
                    voltage_trace_plot[spike_idx] = 0.0  # Set to 0 for visualization only
        
        # Plot membrane voltage - show full duration
        ax1.plot(time_array, voltage_trace_plot, 'b-', linewidth=0.8, label='Membrane Voltage')
        ax1.axhline(y=self.v_threshold, color='r', linestyle='--', 
                   label=f'Threshold ({self.v_threshold}mV)')
        ax1.axhline(y=self.v_rest, color='g', linestyle='--', 
                   label=f'Rest ({self.v_rest}mV)')
        
        # Mark output spikes
        # if output_spikes:
        #     for spike_time in output_spikes:
        #         ax1.axvline(x=spike_time, color='r', alpha=0.7, linewidth=1)
            
        ax1.set_ylabel('Voltage (mV)')
        ax1.set_title('I&F Neuron Membrane Voltage')
        ax1.set_xlim(0, duration)  # Show full simulation duration
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot input spike raster - each synapse gets its own line
        if len(exc_spikes_list) > 0 or len(inh_spikes_list) > 0:
            spike_data = []
            colors = []
            lineoffsets = []
            
            # Plot each inhibitory synapse on separate lines (offset 0 to n_inh-1)
            if len(inh_spikes_list) > 0:
                for i, spikes in enumerate(inh_spikes_list):
                    if len(spikes) > 0:
                        spike_data.append(spikes)
                        colors.append('blue')
                        lineoffsets.append(i)
            
            # Plot each excitatory synapse on separate lines (offset n_inh to n_inh+n_exc-1)
            if len(exc_spikes_list) > 0:
                max_inh_offset = len(inh_spikes_list) if inh_spikes_list else 0
                for i, spikes in enumerate(exc_spikes_list):
                    if len(spikes) > 0:
                        spike_data.append(spikes)
                        colors.append('green')
                        lineoffsets.append(max_inh_offset + i)
            
            if spike_data:
                ax2.eventplot(spike_data, colors=colors, lineoffsets=lineoffsets,
                            linelengths=0.8, linewidths=1.5)
        
        ax2.set_ylabel('Synapse Index')
        ax2.set_title('Input Spike Trains (Green: Excitatory, Blue: Inhibitory) - Each line is one synapse')
        ax2.set_xlim(0, duration)  # Show full simulation duration
        if len(inh_spikes_list) > 0 or len(exc_spikes_list) > 0:
            max_offset = len(inh_spikes_list) + len(exc_spikes_list) - 1
            ax2.set_ylim(-0.5, max_offset + 0.5)
        else:
            ax2.set_ylim(-0.5, 1.5)
        ax2.grid(True, alpha=0.3)
        
        # Plot output spike raster
        if output_spikes:
            ax3.eventplot([output_spikes], colors=['red'], lineoffsets=0.5, 
                         linelengths=0.8, linewidths=2)
        ax3.set_ylabel('Output Spikes')
        ax3.set_xlabel('Time (ms)')
        ax3.set_title('Output Spike Train')
        ax3.set_xlim(0, duration)  # Show full simulation duration
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'/G/MIMOlab/Codes/neuron_as_TCN/IF_model_results/if_neuron_simulation_{seed}.png', 
                       dpi=300, bbox_inches='tight')
            print(f"Figure saved as 'if_neuron_simulation_{seed}.png'")
        
        # plt.show()


def spike_times_to_raster(spike_times_list, duration, dt, n_synapses):
    """
    Convert list of spike times per synapse to binary raster matrix.
    
    Args:
        spike_times_list: List of arrays, each containing spike times (in ms) for one synapse
        duration: Simulation duration in ms
        dt: Time step in ms
        n_synapses: Number of synapses
        
    Returns:
        raster: Binary matrix of shape (n_synapses, n_time_steps)
    """
    n_time_steps = int(duration / dt)
    raster = np.zeros((n_synapses, n_time_steps), dtype=np.float32)
    
    for syn_idx, spike_times in enumerate(spike_times_list):
        if len(spike_times) > 0:
            # Convert spike times to time step indices
            spike_indices = np.round(spike_times / dt).astype(int)
            # Keep only valid indices
            valid_mask = (spike_indices >= 0) & (spike_indices < n_time_steps)
            spike_indices = spike_indices[valid_mask]
            # Set spikes to 1
            raster[syn_idx, spike_indices] = 1.0
    
    return raster


def run_simulation(trial_id, save_dir=None, plot_results=False):
    """
    Run a single I&F neuron simulation and save results as a dictionary.
    
    Args:
        trial_id: trial identifier (used as seed and filename)
        save_dir: directory to save simulation data (if None, saves to current directory)
        plot_results: whether to plot and save results (default: False)
    
    Returns:
        sim_dict: Dictionary containing simulation data
    """
    print(f"=== Running simulation trial {trial_id} ===")
    
    # Set random seed
    np.random.seed(trial_id)
    random.seed(trial_id)
    
    # Create neuron model
    neuron = IntegrateFireNeuron()
    
    # Run simulation
    duration = 6000.0
    time_array, voltage_trace, output_spikes, exc_spikes_list, inh_spikes_list = neuron.simulate(
        duration=duration, base_seed=trial_id
    )
    
    # Calculate results
    firing_rate = neuron.calculate_firing_rate(duration)
    print(f"Trial {trial_id}: {len(output_spikes)} output spikes, {firing_rate:.2f} Hz")
    
    # Plot results if requested
    if plot_results:
        neuron.plot_results(time_array, voltage_trace, output_spikes, exc_spikes_list, inh_spikes_list, duration, trial_id)
    
    # Convert spike times to raster format (synapse x time_steps)
    ex_input_raster = spike_times_to_raster(exc_spikes_list, duration, neuron.dt, neuron.n_exc)
    inh_input_raster = spike_times_to_raster(inh_spikes_list, duration, neuron.dt, neuron.n_inh)
    
    # Convert output spikes to array
    output_spike_times = np.array(output_spikes, dtype=np.float32)
    
    # Create simulation dictionary
    sim_dict = {
        'voltage': voltage_trace.astype(np.float32),  # Array of voltage values
        'exInputSpikeTimes': ex_input_raster,  # Shape: (n_exc, n_time_steps)
        'inhInputSpikeTimes': inh_input_raster,  # Shape: (n_inh, n_time_steps)
        'outputSpikeTimes': output_spike_times,  # Array of output spike times in ms
    }
    
    # Prepare save directory
    if save_dir is None:
        save_dir = Path('.')
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle file
    pickle_path = save_dir / f'trial_{trial_id}.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(sim_dict, f)
    
    print(f"Trial {trial_id} data saved to {pickle_path}")
    
    return sim_dict


if __name__ == "__main__":
    # Run multiple trials and save as pickle files
    save_data_dir = Path('IF_model_simulations')
    save_data_dir.mkdir(exist_ok=True)
    
    # Number of trials to run
    num_trials = 9000
    start_trial_id = 42+1000
    
    print(f"Running {num_trials} IF neuron simulations...")
    print(f"Saving results to: {save_data_dir}")
    
    for trial_id in range(start_trial_id, start_trial_id + num_trials):
        run_simulation(trial_id, save_dir=save_data_dir, plot_results=False)
    
    print(f"\nAll {num_trials} simulations completed!")
    print(f"Results saved to: {save_data_dir}")
