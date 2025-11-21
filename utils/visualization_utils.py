import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Common utility functions for reducing code duplication
def _add_statistics_text(ax, stats_dict, position=(0.02, 0.98), fontsize=9, 
                        box_color='lightblue', alpha=0.8):
    """Add statistics text box to plot"""
    if not stats_dict:
        return
    
    stats_text = '\n'.join([f"{k}: {v}" for k, v in stats_dict.items()])
    ax.text(position[0], position[1], stats_text, transform=ax.transAxes, 
            fontsize=fontsize, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=box_color, alpha=alpha))

def _setup_plot_style(ax, title, xlabel, ylabel, fontsize=12, grid=True, legend=False, legend_fontsize=None, xlim=None, ylim=None):
    """Setup common plot styling with borders, labels, legend and optional axis limits"""
    ax.set_title(title, fontsize=fontsize, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=fontsize-1)
    ax.set_ylabel(ylabel, fontsize=fontsize-1)
    if grid:
        ax.grid(True, alpha=0.3)
    
    # Setup legend if requested
    if legend:
        legend_fontsize = legend_fontsize or fontsize-3
        ax.legend(fontsize=legend_fontsize)
    
    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def _setup_inset_axes(parent_ax, xlim, ylim, xlabel, ylabel, fontsize=8, tick_fontsize=7):
    """Setup inset axes with common styling"""
    try:
        axins = inset_axes(parent_ax, width='45%', height='45%', loc='lower right')
        axins.set_xlim(xlim)
        axins.set_ylim(ylim)
        axins.set_xlabel(xlabel, fontsize=fontsize)
        axins.set_ylabel(ylabel, fontsize=fontsize)
        axins.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        return axins
    except Exception:
        return None

def _process_segments_data(firing_rates, num_exc_segments, num_inh_segments, max_segments_to_show, specified_segments=None):
    """Process segments data and return separated data with indices"""
    num_segments_total, _ = firing_rates.shape
    num_inh_segments_calc = num_segments_total - num_exc_segments
    
    # Separate excitatory and inhibitory data
    exc_data = firing_rates[:num_exc_segments, :]
    inh_data = firing_rates[num_exc_segments:, :]
    
    # Process excitatory segments
    if specified_segments is not None and len(specified_segments) > 0:
        exc_specified = [idx for idx in specified_segments if idx < num_exc_segments]
        if len(exc_specified) > 0:
            exc_data_selected = exc_data[exc_specified, :]
            exc_selected_indices = np.array(exc_specified)
            exc_to_show = len(exc_specified)
        else:
            if num_exc_segments > max_segments_to_show // 2:
                exc_to_show = max_segments_to_show // 2
                exc_indices = np.linspace(0, num_exc_segments-1, exc_to_show, dtype=int)
                exc_data_selected = exc_data[exc_indices, :]
                exc_selected_indices = exc_indices
            else:
                exc_data_selected = exc_data
                exc_selected_indices = np.arange(num_exc_segments)
                exc_to_show = num_exc_segments
    else:
        if num_exc_segments > max_segments_to_show // 2:
            exc_to_show = max_segments_to_show // 2
            exc_indices = np.linspace(0, num_exc_segments-1, exc_to_show, dtype=int)
            exc_data_selected = exc_data[exc_indices, :]
            exc_selected_indices = exc_indices
        else:
            exc_data_selected = exc_data
            exc_selected_indices = np.arange(num_exc_segments)
            exc_to_show = num_exc_segments
    
    # Process inhibitory segments
    if specified_segments is not None and len(specified_segments) > 0:
        inh_specified = [idx for idx in specified_segments if idx < num_inh_segments]
        if len(inh_specified) > 0:
            inh_data_selected = inh_data[np.array(inh_specified) - num_exc_segments, :]
            inh_selected_indices = np.array(inh_specified)
            inh_to_show = len(inh_specified)
        else:
            if num_inh_segments_calc > max_segments_to_show // 2:
                inh_to_show = max_segments_to_show // 2
                inh_indices = np.linspace(0, num_inh_segments_calc-1, inh_to_show, dtype=int)
                inh_data_selected = inh_data[inh_indices, :]
                inh_selected_indices = inh_indices + num_exc_segments
            else:
                inh_data_selected = inh_data
                inh_selected_indices = np.arange(num_inh_segments_calc) + num_exc_segments
                inh_to_show = num_inh_segments_calc
    else:
        if num_inh_segments_calc > max_segments_to_show // 2:
            inh_to_show = max_segments_to_show // 2
            inh_indices = np.linspace(0, num_inh_segments_calc-1, inh_to_show, dtype=int)
            inh_data_selected = inh_data[inh_indices, :]
            inh_selected_indices = inh_indices + num_exc_segments
        else:
            inh_data_selected = inh_data
            inh_selected_indices = np.arange(num_inh_segments_calc) + num_exc_segments
            inh_to_show = num_inh_segments_calc
    
    return {
        'exc_data_selected': exc_data_selected,
        'exc_selected_indices': exc_selected_indices,
        'exc_to_show': exc_to_show,
        'inh_data_selected': inh_data_selected,
        'inh_selected_indices': inh_selected_indices,
        'inh_to_show': inh_to_show,
        'num_inh_segments_calc': num_inh_segments_calc
    }

def _setup_y_axis_ticks(ax, selected_indices, segment_type, to_show, monoconn_seg_indices=None):
    """Setup y-axis ticks with fixed indices logic"""
    if to_show > 0 and len(selected_indices) > 0:
        # When indices count is <= 20, set ticks for each segment
        if to_show <= 20:
            y_tick_positions = list(range(to_show))
            y_tick_labels = [f'{segment_type} {selected_indices[i]}' for i in y_tick_positions]
        elif to_show > 20 and monoconn_seg_indices is not None and len(monoconn_seg_indices) > 0:
            # When indices count > 20 and monoconn_seg_indices available, only show fixed indices
            y_tick_positions = []
            y_tick_labels = []
            for fixed_idx in monoconn_seg_indices:
                if fixed_idx in selected_indices:
                    # Find the position of fixed_idx in selected_indices
                    pos = list(selected_indices).index(fixed_idx)
                    y_tick_positions.append(pos)
                    y_tick_labels.append(f'{segment_type} {fixed_idx}')
        else:
            # When indices count > 20, use original sampling method
            y_tick_positions = [0, to_show//4, to_show//2, 3*to_show//4, to_show-1]
            y_tick_labels = [f'{segment_type} {selected_indices[i]}' for i in y_tick_positions if i < len(selected_indices)]
        
        ax.set_yticks(y_tick_positions[:len(y_tick_labels)])
        ax.set_yticklabels(y_tick_labels)
    else:
        # If no segments, set empty ticks
        ax.set_yticks([])
        ax.set_yticklabels([])

def _save_and_show_plot(fig, save_path=None, show_plot=False, dpi=300, 
                       bbox_inches='tight', message_prefix="Plot"):
    """Common save and show logic for plots with integrated safe saving"""
    if save_path:
        try:
            fig.canvas.draw()
            plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, 
                       facecolor='white', edgecolor='none')
        except Exception:
            try:
                plt.savefig(save_path, dpi=dpi, facecolor='white', edgecolor='none')
            except Exception:
                plt.savefig(save_path, dpi=150)
        #print(f"{message_prefix} generated: {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close(fig)
    return save_path

def _calculate_basic_stats(data):
    """Calculate basic statistics for data"""
    if not data or len(data) == 0:
        return {}
    
    return {
        'Final': f"{data[-1]:.6f}",
        'Min': f"{min(data):.6f}",
        'Max': f"{max(data):.6f}",
        'Mean': f"{np.mean(data):.6f}",
        'Improvement': f"{data[0] - data[-1]:.6f}" if len(data) > 1 else "N/A"
    }

# visualization functions for 5_main_figure_replication.py

def plot_summary_panels(fpr, tpr, desired_fp_ind, y_spikes_GT, y_spikes_hat, 
                       y_soma_GT, y_soma_hat, desired_threshold, save_path):
    """Plot summary panels"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # ROC curve
    ax_roc = axes[0]
    roc_auc = np.trapz(tpr, fpr)
    ax_roc.plot(fpr, tpr, 'k-', linewidth=2)
    ax_roc.plot(fpr[desired_fp_ind], tpr[desired_fp_ind], 'o', color='red', markersize=6)
    _setup_plot_style(ax_roc, '', 'False alarm rate', 'Hit rate', xlim=[0.0, 1.0], ylim=[0.0, 1.05], grid=False)
    ax_roc.text(0.98, 0.9, f"AUC={roc_auc:.4f}", transform=ax_roc.transAxes,
                ha='right', va='top', fontsize=11, color='dimgray')
    
    # Add zoomed inset
    axins = _setup_inset_axes(ax_roc, 
                            xlim=(0.0, max(0.04, float(fpr[min(len(fpr)-1, desired_fp_ind*3)]))),
                            ylim=(0.0, 1.0),
                            xlabel='False alarm rate',
                            ylabel='Hit rate')
    if axins:
        axins.plot(fpr, tpr, 'k-', linewidth=1.5)
        axins.plot(fpr[desired_fp_ind], tpr[desired_fp_ind], 'o', color='red', markersize=4)

    # Cross-correlation analysis
    ax_xcorr = axes[1]
    spikes_pred_bin = (y_spikes_hat > desired_threshold).astype(np.float32)
    spikes_gt_bin = (y_spikes_GT > 0.5).astype(np.float32)
    
    max_lag_ms = 50
    lags = np.arange(-max_lag_ms, max_lag_ms + 1)
    xcorr_sum = np.zeros(len(lags), dtype=np.float64)
    
    for i in range(spikes_gt_bin.shape[0]):
        cc = np.correlate(spikes_gt_bin[i], spikes_pred_bin[i], mode='full')
        mid = spikes_gt_bin.shape[1] - 1
        xcorr_sum += cc[mid - max_lag_ms: mid + max_lag_ms + 1]
    
    xcorr_hz = (xcorr_sum / max(1, spikes_gt_bin.shape[0])) * 1000.0
    ax_xcorr.plot(lags, xcorr_hz, 'k-', linewidth=1.5)
    ax_xcorr.axvline(0, color='k', linestyle=':', alpha=0.5)
    _setup_plot_style(ax_xcorr, '', 'Δt (ms)', 'spike rate (Hz)', grid=False)

    # Voltage scatter plot
    ax_scatter = axes[2]
    x, y = y_soma_GT.ravel(), y_soma_hat.ravel()
    
    # Downsample for performance
    if len(x) > 50000:
        idx = np.random.choice(len(x), 50000, replace=False)
        x, y = x[idx], y[idx]
    
    ax_scatter.scatter(x, y, s=4, c='tab:blue', alpha=0.25, edgecolors='none')
    _setup_plot_style(ax_scatter, '', 'L5PC Model (mV)', 'ANN (mV)', grid=False)# xlim=(-80, -57), ylim=(-80, -57), grid=False)

    return _save_and_show_plot(fig, save_path, show_plot=False, message_prefix="Summary panels")

def plot_voltage_traces(y_spikes_GT, y_spikes_hat, y_voltage_GT, y_voltage_hat, 
                       global_threshold, selected_traces, output_dir, 
                       per_sample_global_metrics=None, per_sample_optimal_metrics=None):
    """Plot voltage traces with pagination"""
    total = len(selected_traces)
    if total == 0:
        #print('No selected traces to plot.')
        return

    page_size = 5
    num_pages = int(np.ceil(total / page_size))

    figs = []
    axes_list = []

    for page_idx in range(num_pages):
        start = page_idx * page_size
        end = min((page_idx + 1) * page_size, total)
        traces_page = selected_traces[start:end]

        num_subplots = len(traces_page)
        fig, axes = plt.subplots(nrows=num_subplots, ncols=1, figsize=(15, 3*num_subplots), sharex=True)
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.01, hspace=0.3)

        # Normalize axes to list for both single and multiple subplot cases
        axes_norm = axes if isinstance(axes, (list, np.ndarray)) else [axes]

        for k, selected_trace in enumerate(traces_page):
            ax = axes_norm[k]

            y_train_soma_bias = -67.7

            # Extract data
            spike_GT = y_spikes_GT[selected_trace, :]
            spike_hat = y_spikes_hat[selected_trace, :]
            voltage_GT = y_voltage_GT[selected_trace, :]
            voltage_hat = y_voltage_hat[selected_trace, :] + y_train_soma_bias

            # Use precomputed metrics
            global_metrics = per_sample_global_metrics[selected_trace] if per_sample_global_metrics is not None else None
            optimal_metrics = per_sample_optimal_metrics[selected_trace] if per_sample_optimal_metrics is not None else None

            spike_trace_pred_global = spike_hat > global_threshold
            output_spike_times_in_ms_pred_global = np.nonzero(spike_trace_pred_global)[0]
            voltage_hat[output_spike_times_in_ms_pred_global] = 0

            # Time axis
            sim_duration_ms = spike_GT.shape[0]
            time_in_sec = np.arange(sim_duration_ms) / 1000.0

            # Plot voltage traces
            ax.plot(time_in_sec, voltage_GT, c='c', linewidth=2, label='Ground Truth')
            ax.plot(time_in_sec, voltage_hat, c='m', linestyle=':', linewidth=1.5, label='Prediction')

            # Add title
            if global_metrics is not None and optimal_metrics is not None:
                title_text = (f'Sim {selected_trace}: Global FPR={global_metrics["fpr"]:.4f}, '
                              f'Optimal FPR={optimal_metrics["fpr"]:.4f}, '
                              f'FP={global_metrics["false_positives"]}, '
                              f'TP={global_metrics["true_positives"]}')
            else:
                title_text = f'Sim {selected_trace}'
            
            # Set axes
            _setup_plot_style(ax, title_text, '', '$V_m$ (mV)', fontsize=10, 
                            xlim=(0.02, sim_duration_ms / 1000.0), ylim=(-80, 5), grid=False)

            # Set legend on the first subplot of each figure
            if k == 0:
                ax.legend(loc='upper right', fontsize=8)

            # Hide x-axis labels for non-last subplot
            if k < num_subplots - 1:
                plt.setp(ax.get_xticklabels(), visible=False)

        # Set x-axis label on the last subplot of each figure
        axes_norm[-1].set_xlabel('Time (s)', fontsize=12)

        figs.append(fig)
        axes_list.append(axes_norm)

    # Save figures
    if num_pages == 1:
        save_path = f'{output_dir}/voltage_traces.pdf'
        _save_and_show_plot(figs[0], save_path, show_plot=False, message_prefix="Voltage traces")
    else:
        for idx, fig in enumerate(figs, start=1):
            save_path = f'{output_dir}/voltage_traces_p{idx}.pdf'
            _save_and_show_plot(fig, save_path, show_plot=False, message_prefix="Voltage traces")


# Visualization functions for activity optimization

def visualize_firing_rates_trace(firing_rates, num_exc_segments=639, num_inh_segments=640, 
                                 save_path=None, title="Firing Rates Visualization", figsize=(18, 12), 
                                 time_step_ms=1, max_segments_to_show=200, specified_segments=None, monoconn_seg_indices=None):
    """Visualize firing rates using raster plot style with continuous curves"""
    # Process segments data using common function
    segments_data = _process_segments_data(firing_rates, num_exc_segments, num_inh_segments, 
                                         max_segments_to_show, specified_segments)
    
    exc_data_selected = segments_data['exc_data_selected']
    exc_selected_indices = segments_data['exc_selected_indices']
    exc_to_show = segments_data['exc_to_show']
    inh_data_selected = segments_data['inh_data_selected']
    inh_selected_indices = segments_data['inh_selected_indices']
    inh_to_show = segments_data['inh_to_show']
    num_inh_segments = segments_data['num_inh_segments_calc']
    
    # Get time duration from firing_rates shape
    time_duration_ms = firing_rates.shape[1]
    num_segments_total = firing_rates.shape[0]
    
    # Create time axis
    time_axis = np.arange(time_duration_ms) * time_step_ms
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # === Top plot: excitatory segments ===
    # Use specified indices as y-axis positions to ensure correct segment numbers are displayed
    y_positions_exc = np.arange(len(exc_selected_indices))
    
    # Plot firing rate for each excitatory segment
    for i in range(exc_data_selected.shape[0]):
        # Get firing rate for this segment (time series)
        segment_firing_rate = exc_data_selected[i, :]
        
        # Scale firing rate to appropriate display range
        scaled_firing_rate = segment_firing_rate * 4  # Scaling factor
        y_base = y_positions_exc[i]
        y_values = y_base + scaled_firing_rate
        
        # Plot curve (blue)
        ax1.plot(time_axis, y_values, color='blue', linewidth=0.5, alpha=0.7)
        
        # Plot baseline
        ax1.axhline(y=y_base, color='lightgray', linewidth=0.2, alpha=0.5)
    
    # Set top plot axes
    _setup_plot_style(ax1, f'Excitatory Segments (n={exc_to_show}/{num_exc_segments})', 'Time (ms)', 'Excitatory Segments', fontsize=14)
    ax1.tick_params(axis='y', colors='blue')
    
    # Set y-axis ticks and labels - excitatory
    _setup_y_axis_ticks(ax1, exc_selected_indices, 'Exc', exc_to_show, monoconn_seg_indices)
    
    # Add excitatory statistics
    exc_stats_text = f"Range: [{np.min(exc_data_selected):.4f}, {np.max(exc_data_selected):.4f}]\n"
    exc_stats_text += f"Mean: {np.mean(exc_data_selected):.4f} ± {np.std(exc_data_selected):.4f}"
    ax1.text(0.02, 0.98, exc_stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # === Bottom plot: inhibitory segments ===
    # Use specified indices as y-axis positions to ensure correct segment numbers are displayed
    y_positions_inh = np.arange(len(inh_selected_indices))
    
    # Plot firing rate for each inhibitory segment
    for i in range(inh_data_selected.shape[0]):
        # Get firing rate for this segment (time series)
        segment_firing_rate = inh_data_selected[i, :]
        
        # Scale firing rate to appropriate display range
        scaled_firing_rate = segment_firing_rate * 40  # Scaling factor
        y_base = y_positions_inh[i]
        y_values = y_base + scaled_firing_rate
        
        # Plot curve (red)
        ax2.plot(time_axis, y_values, color='red', linewidth=0.5, alpha=0.7)
        
        # Plot baseline
        ax2.axhline(y=y_base, color='lightgray', linewidth=0.2, alpha=0.5)
    
    # Set bottom plot axes
    _setup_plot_style(ax2, f'Inhibitory Segments (n={inh_to_show}/{num_inh_segments})', 'Time (ms)', 'Inhibitory Segments', fontsize=14)
    ax2.tick_params(axis='y', colors='red')
    
    # Set y-axis ticks and labels - inhibitory
    _setup_y_axis_ticks(ax2, inh_selected_indices, 'Inh', inh_to_show, monoconn_seg_indices)
    
    # Add inhibitory statistics
    inh_stats_text = f"Range: [{np.min(inh_data_selected):.4f}, {np.max(inh_data_selected):.4f}]\n"
    inh_stats_text += f"Mean: {np.mean(inh_data_selected):.4f} ± {np.std(inh_data_selected):.4f}"
    ax2.text(0.02, 0.98, inh_stats_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Set overall title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Leave space for overall title
    
    # Save or display image
    _save_and_show_plot(fig, save_path, show_plot=(save_path is None), message_prefix="Firing rates visualization")
    
    # Return some statistics
    stats = {
        'mean_firing_rate_exc': np.mean(exc_data_selected),
        'mean_firing_rate_inh': np.mean(inh_data_selected),
        'max_firing_rate_exc': np.max(exc_data_selected),
        'max_firing_rate_inh': np.max(inh_data_selected),
        'min_firing_rate_exc': np.min(exc_data_selected),
        'min_firing_rate_inh': np.min(inh_data_selected),
        'exc_segments_shown': exc_to_show,
        'inh_segments_shown': inh_to_show,
        'total_segments': num_segments_total
    }
    
    #print("Firing rates raster plot completed")
    
    return stats

def visualize_firing_rates_heatmap(firing_rates, num_exc_segments=639, num_inh_segments=640, 
                                   save_path=None, title="Firing Rates Heatmap", figsize=(18, 12), 
                                   max_segments_to_show=300, specified_segments=None, monoconn_seg_indices=None):
    """Visualize firing rates using heatmap style"""
    # Process segments data using common function
    segments_data = _process_segments_data(firing_rates, num_exc_segments, num_inh_segments, 
                                         max_segments_to_show, specified_segments)
    
    exc_data_selected = segments_data['exc_data_selected']
    exc_indices = segments_data['exc_selected_indices']
    exc_to_show = segments_data['exc_to_show']
    inh_data_selected = segments_data['inh_data_selected']
    inh_indices = segments_data['inh_selected_indices']
    inh_to_show = segments_data['inh_to_show']
    num_inh_segments = segments_data['num_inh_segments_calc']
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Top plot: excitatory segments heatmap (blue)
    exc_vmin = 0
    exc_vmax = 0.3
    im1 = ax1.imshow(exc_data_selected, aspect='auto', cmap='Blues', origin='lower', vmin=exc_vmin, vmax=exc_vmax)
    
    # Add blue colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Excitatory Firing Rate', rotation=270, labelpad=15, color='blue')
    cbar1.ax.tick_params(colors='blue')
    
    # Set top plot labels and title
    _setup_plot_style(ax1, f'Excitatory Segments (n={exc_to_show}/{num_exc_segments})', 'Time (ms)', 'Excitatory Segments', fontsize=14)
    ax1.tick_params(axis='y', colors='blue')
    
    # Set y-axis ticks - excitatory
    _setup_y_axis_ticks(ax1, exc_indices, 'Exc', exc_to_show, monoconn_seg_indices)
    
    # Add excitatory statistics
    exc_stats_text = f"Range: [{np.min(exc_data_selected):.4f}, {np.max(exc_data_selected):.4f}]\n"
    exc_stats_text += f"Mean: {np.mean(exc_data_selected):.4f} ± {np.std(exc_data_selected):.4f}"
    ax1.text(0.02, 0.98, exc_stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Bottom plot: inhibitory segments heatmap (red)
    inh_vmin = 0
    inh_vmax = 0.02
    im2 = ax2.imshow(inh_data_selected, aspect='auto', cmap='Reds', origin='lower', vmin=inh_vmin, vmax=inh_vmax)
    
    # Add red colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Inhibitory Firing Rate', rotation=270, labelpad=15, color='red')
    cbar2.ax.tick_params(colors='red')
    
    # Set bottom plot labels and title
    _setup_plot_style(ax2, f'Inhibitory Segments (n={inh_to_show}/{num_inh_segments})', 'Time (ms)', 'Inhibitory Segments', fontsize=14)
    ax2.tick_params(axis='y', colors='red')
    
    # Set y-axis ticks - inhibitory
    _setup_y_axis_ticks(ax2, inh_indices, 'Inh', inh_to_show, monoconn_seg_indices)
    
    # Add inhibitory statistics
    inh_stats_text = f"Range: [{np.min(inh_data_selected):.4f}, {np.max(inh_data_selected):.4f}]\n"
    inh_stats_text += f"Mean: {np.mean(inh_data_selected):.4f} ± {np.std(inh_data_selected):.4f}"
    ax2.text(0.02, 0.98, inh_stats_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Set overall title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Adjust subplot spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Leave space for overall title
    
    _save_and_show_plot(fig, save_path, show_plot=(save_path is None), message_prefix="Firing rates heatmap")
    
    # #print statistics
    #print("Firing rates heatmap completed")

def _plot_combined_trace(initial_sample, optimized_sample, num_exc_segments, num_inh_segments,
                        specified_segments, monoconn_seg_indices, save_path, title_prefix):
    """Plot combined 2x2 trace: left column (initial), right column (optimized)"""
    # Process segments data for both samples
    initial_data = _process_segments_data(initial_sample, num_exc_segments, num_inh_segments, 
                                        10, specified_segments)
    optimized_data = _process_segments_data(optimized_sample, num_exc_segments, num_inh_segments, 
                                          10, specified_segments)
    
    # Get time duration from sample shape
    time_duration_ms = initial_sample.shape[1]
    time_step_ms = 1
    time_axis = np.arange(time_duration_ms) * time_step_ms
    
    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(32, 16))
    
    # Top left: Initial Excitatory
    ax1 = axes[0, 0]
    exc_data = initial_data['exc_data_selected']
    exc_indices = initial_data['exc_selected_indices']
    exc_to_show = initial_data['exc_to_show']
    
    # Plot firing rate for each excitatory segment
    y_positions_exc = np.arange(len(exc_indices))
    for i in range(exc_data.shape[0]):
        segment_firing_rate = exc_data[i, :]
        scaled_firing_rate = segment_firing_rate * 4  # Scaling factor
        y_base = y_positions_exc[i]
        y_values = y_base + scaled_firing_rate
        ax1.plot(time_axis, y_values, color='blue', linewidth=0.5, alpha=0.7)
        ax1.axhline(y=y_base, color='lightgray', linewidth=0.2, alpha=0.5)
    
    _setup_plot_style(ax1, f'Initial - Excitatory (n={exc_to_show})', 'Time (ms)', 'Excitatory Segments', fontsize=12)
    ax1.tick_params(axis='y', colors='blue')
    _setup_y_axis_ticks(ax1, exc_indices, 'Exc', exc_to_show, monoconn_seg_indices)
    
    # Add excitatory statistics
    exc_stats_text = f"Range: [{np.min(exc_data):.4f}, {np.max(exc_data):.4f}]\n"
    exc_stats_text += f"Mean: {np.mean(exc_data):.4f} ± {np.std(exc_data):.4f}"
    ax1.text(0.02, 0.98, exc_stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Top right: Optimized Excitatory
    ax2 = axes[0, 1]
    opt_exc_data = optimized_data['exc_data_selected']
    opt_exc_indices = optimized_data['exc_selected_indices']
    opt_exc_to_show = optimized_data['exc_to_show']
    
    # Plot firing rate for each excitatory segment
    y_positions_opt_exc = np.arange(len(opt_exc_indices))
    for i in range(opt_exc_data.shape[0]):
        segment_firing_rate = opt_exc_data[i, :]
        scaled_firing_rate = segment_firing_rate * 4  # Scaling factor
        y_base = y_positions_opt_exc[i]
        y_values = y_base + scaled_firing_rate
        ax2.plot(time_axis, y_values, color='blue', linewidth=0.5, alpha=0.7)
        ax2.axhline(y=y_base, color='lightgray', linewidth=0.2, alpha=0.5)
    
    _setup_plot_style(ax2, f'Optimized - Excitatory (n={opt_exc_to_show})', 'Time (ms)', 'Excitatory Segments', fontsize=12)
    ax2.tick_params(axis='y', colors='blue')
    _setup_y_axis_ticks(ax2, opt_exc_indices, 'Exc', opt_exc_to_show, monoconn_seg_indices)
    
    # Add excitatory statistics
    opt_exc_stats_text = f"Range: [{np.min(opt_exc_data):.4f}, {np.max(opt_exc_data):.4f}]\n"
    opt_exc_stats_text += f"Mean: {np.mean(opt_exc_data):.4f} ± {np.std(opt_exc_data):.4f}"
    ax2.text(0.02, 0.98, opt_exc_stats_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Bottom left: Initial Inhibitory
    ax3 = axes[1, 0]
    inh_data = initial_data['inh_data_selected']
    inh_indices = initial_data['inh_selected_indices']
    inh_to_show = initial_data['inh_to_show']
    
    # Plot firing rate for each inhibitory segment
    y_positions_inh = np.arange(len(inh_indices))
    for i in range(inh_data.shape[0]):
        segment_firing_rate = inh_data[i, :]
        scaled_firing_rate = segment_firing_rate * 40  # Scaling factor
        y_base = y_positions_inh[i]
        y_values = y_base + scaled_firing_rate
        ax3.plot(time_axis, y_values, color='red', linewidth=0.5, alpha=0.7)
        ax3.axhline(y=y_base, color='lightgray', linewidth=0.2, alpha=0.5)
    
    _setup_plot_style(ax3, f'Initial - Inhibitory (n={inh_to_show})', 'Time (ms)', 'Inhibitory Segments', fontsize=12)
    ax3.tick_params(axis='y', colors='red')
    _setup_y_axis_ticks(ax3, inh_indices, 'Inh', inh_to_show, monoconn_seg_indices)
    
    # Add inhibitory statistics
    inh_stats_text = f"Range: [{np.min(inh_data):.4f}, {np.max(inh_data):.4f}]\n"
    inh_stats_text += f"Mean: {np.mean(inh_data):.4f} ± {np.std(inh_data):.4f}"
    ax3.text(0.02, 0.98, inh_stats_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Bottom right: Optimized Inhibitory
    ax4 = axes[1, 1]
    opt_inh_data = optimized_data['inh_data_selected']
    opt_inh_indices = optimized_data['inh_selected_indices']
    opt_inh_to_show = optimized_data['inh_to_show']
    
    # Plot firing rate for each inhibitory segment
    y_positions_opt_inh = np.arange(len(opt_inh_indices))
    for i in range(opt_inh_data.shape[0]):
        segment_firing_rate = opt_inh_data[i, :]
        scaled_firing_rate = segment_firing_rate * 40  # Scaling factor
        y_base = y_positions_opt_inh[i]
        y_values = y_base + scaled_firing_rate
        ax4.plot(time_axis, y_values, color='red', linewidth=0.5, alpha=0.7)
        ax4.axhline(y=y_base, color='lightgray', linewidth=0.2, alpha=0.5)
    
    _setup_plot_style(ax4, f'Optimized - Inhibitory (n={opt_inh_to_show})', 'Time (ms)', 'Inhibitory Segments', fontsize=12)
    ax4.tick_params(axis='y', colors='red')
    _setup_y_axis_ticks(ax4, opt_inh_indices, 'Inh', opt_inh_to_show, monoconn_seg_indices)
    
    # Add inhibitory statistics
    opt_inh_stats_text = f"Range: [{np.min(opt_inh_data):.4f}, {np.max(opt_inh_data):.4f}]\n"
    opt_inh_stats_text += f"Mean: {np.mean(opt_inh_data):.4f} ± {np.std(opt_inh_data):.4f}"
    ax4.text(0.02, 0.98, opt_inh_stats_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Set overall title
    fig.suptitle(f'{title_prefix} - Combined Trace Comparison', fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    return _save_and_show_plot(fig, save_path, show_plot=False, message_prefix="Combined trace")

def _plot_combined_heatmap(initial_sample, optimized_sample, num_exc_segments, num_inh_segments,
                          specified_segments, monoconn_seg_indices, save_path, title_prefix):
    """Plot combined 2x2 heatmap: left column (initial), right column (optimized)"""
    # Process segments data for both samples
    initial_data = _process_segments_data(initial_sample, num_exc_segments, num_inh_segments, 
                                        200, specified_segments)
    optimized_data = _process_segments_data(optimized_sample, num_exc_segments, num_inh_segments, 
                                          200, specified_segments)
    
    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(32, 16))
    
    # Top left: Initial Excitatory
    ax1 = axes[0, 0]
    exc_data = initial_data['exc_data_selected']
    exc_indices = initial_data['exc_selected_indices']
    exc_to_show = initial_data['exc_to_show']
    
    exc_vmin, exc_vmax = 0, 0.3
    im1 = ax1.imshow(exc_data, aspect='auto', cmap='Blues', origin='lower', 
                     vmin=exc_vmin, vmax=exc_vmax)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Excitatory Firing Rate', rotation=270, labelpad=15, color='blue')
    cbar1.ax.tick_params(colors='blue')
    
    _setup_plot_style(ax1, f'Initial - Excitatory (n={exc_to_show})', 'Time (ms)', 'Excitatory Segments', fontsize=12)
    ax1.tick_params(axis='y', colors='blue')
    _setup_y_axis_ticks(ax1, exc_indices, 'Exc', exc_to_show, monoconn_seg_indices)
    
    # Top right: Optimized Excitatory
    ax2 = axes[0, 1]
    opt_exc_data = optimized_data['exc_data_selected']
    opt_exc_indices = optimized_data['exc_selected_indices']
    opt_exc_to_show = optimized_data['exc_to_show']
    
    im2 = ax2.imshow(opt_exc_data, aspect='auto', cmap='Blues', origin='lower', 
                     vmin=exc_vmin, vmax=exc_vmax)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Excitatory Firing Rate', rotation=270, labelpad=15, color='blue')
    cbar2.ax.tick_params(colors='blue')
    
    _setup_plot_style(ax2, f'Optimized - Excitatory (n={opt_exc_to_show})', 'Time (ms)', 'Excitatory Segments', fontsize=12)
    ax2.tick_params(axis='y', colors='blue')
    _setup_y_axis_ticks(ax2, opt_exc_indices, 'Exc', opt_exc_to_show, monoconn_seg_indices)
    
    # Bottom left: Initial Inhibitory
    ax3 = axes[1, 0]
    inh_data = initial_data['inh_data_selected']
    inh_indices = initial_data['inh_selected_indices']
    inh_to_show = initial_data['inh_to_show']
    
    inh_vmin, inh_vmax = 0, 0.02
    im3 = ax3.imshow(inh_data, aspect='auto', cmap='Reds', origin='lower', 
                     vmin=inh_vmin, vmax=inh_vmax)
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('Inhibitory Firing Rate', rotation=270, labelpad=15, color='red')
    cbar3.ax.tick_params(colors='red')
    
    _setup_plot_style(ax3, f'Initial - Inhibitory (n={inh_to_show})', 'Time (ms)', 'Inhibitory Segments', fontsize=12)
    ax3.tick_params(axis='y', colors='red')
    _setup_y_axis_ticks(ax3, inh_indices, 'Inh', inh_to_show, monoconn_seg_indices)
    
    # Bottom right: Optimized Inhibitory
    ax4 = axes[1, 1]
    opt_inh_data = optimized_data['inh_data_selected']
    opt_inh_indices = optimized_data['inh_selected_indices']
    opt_inh_to_show = optimized_data['inh_to_show']
    
    im4 = ax4.imshow(opt_inh_data, aspect='auto', cmap='Reds', origin='lower', 
                     vmin=inh_vmin, vmax=inh_vmax)
    cbar4 = plt.colorbar(im4, ax=ax4)
    cbar4.set_label('Inhibitory Firing Rate', rotation=270, labelpad=15, color='red')
    cbar4.ax.tick_params(colors='red')
    
    _setup_plot_style(ax4, f'Optimized - Inhibitory (n={opt_inh_to_show})', 'Time (ms)', 'Inhibitory Segments', fontsize=12)
    ax4.tick_params(axis='y', colors='red')
    _setup_y_axis_ticks(ax4, opt_inh_indices, 'Inh', opt_inh_to_show, monoconn_seg_indices)
    
    # Set overall title
    fig.suptitle(f'{title_prefix} - Combined Heatmap Comparison', fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the plot
    _save_and_show_plot(fig, save_path, show_plot=False, message_prefix="Combined heatmap")
    
    # Return the four heatmap data arrays
    return {
        'initial_exc_data': exc_data,
        'optimized_exc_data': opt_exc_data, 
        'initial_inh_data': inh_data,
        'optimized_inh_data': opt_inh_data,
        'exc_indices': exc_indices,
        'opt_exc_indices': opt_exc_indices,
        'inh_indices': inh_indices,
        'opt_inh_indices': opt_inh_indices
    }

def visualize_optimized_firing_rates(initial_firing_rates, optimized_firing_rates, monoconn_seg_indices, 
                                   num_exc_segments=639, num_inh_segments=640,
                                   save_dir=None, title_prefix="Optimized Firing Rates"):
    """Visualize optimized firing rates"""
    
    # Use first batch for visualization
    initial_sample = np.mean(initial_firing_rates, axis=0)
    optimized_sample = np.mean(optimized_firing_rates, axis=0)
    # initial_sample = initial_firing_rates[0] if len(initial_firing_rates.shape) > 2 else initial_firing_rates
    # optimized_sample = optimized_firing_rates[0]  # (num_segments, time_duration)
    
    #print("Generating optimized firing rates visualization...")
    
    # Specify segments to visualize: prioritize monoconn_seg_indices, otherwise use default sampling
    specified_segments = None
    if monoconn_seg_indices is not None and len(monoconn_seg_indices) > 0:
        # Extend monoconn_seg_indices to include surrounding segments for better observation
        extended_indices = []
        for idx in monoconn_seg_indices:
            # Add k segments before and after each fixed index
            num_segments_to_add = 10
            start_idx = max(0, idx - num_segments_to_add)
            end_idx = min(num_inh_segments, idx + num_segments_to_add + 1)
            extended_indices.extend(range(start_idx, end_idx))
        
        # Remove duplicates and sort
        extended_indices = sorted(list(set(extended_indices)))
        specified_segments = extended_indices
        
    # # Combined trace - 2x2 subplots
    # combined_trace_save_path = os.path.join(save_dir, 'combined_firing_rate_trace.pdf')
    # _plot_combined_trace(
    #     initial_sample, optimized_sample, 
    #     num_exc_segments, num_inh_segments,
    #     specified_segments, monoconn_seg_indices,
    #     combined_trace_save_path, title_prefix
    # )
    
    # Combined heatmap - 2x2 subplots
    combined_heatmap_save_path = os.path.join(save_dir, 'combined_firing_rate_heatmap.pdf')
    heatmap_data = _plot_combined_heatmap(
        initial_sample, optimized_sample, 
        num_exc_segments, num_inh_segments,
        specified_segments, monoconn_seg_indices,
        combined_heatmap_save_path, title_prefix
    )
    
    print("Optimized firing rates visualization completed")
    return heatmap_data

def plot_average_heatmap(average_heatmap_data, monoconn_seg_indices, 
                        num_exc_segments=639, num_inh_segments=640,
                        save_path=None, title_prefix="Average Heatmap"):
    """Plot average heatmap from multiple seeds"""
    
    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(32, 16))
    
    # Extract data
    exc_data = average_heatmap_data['initial_exc_data']
    opt_exc_data = average_heatmap_data['optimized_exc_data']
    inh_data = average_heatmap_data['initial_inh_data']
    opt_inh_data = average_heatmap_data['optimized_inh_data']
    
    exc_indices = average_heatmap_data['exc_indices']
    opt_exc_indices = average_heatmap_data['opt_exc_indices']
    inh_indices = average_heatmap_data['inh_indices']
    opt_inh_indices = average_heatmap_data['opt_inh_indices']
    
    exc_to_show = len(exc_indices)
    opt_exc_to_show = len(opt_exc_indices)
    inh_to_show = len(inh_indices)
    opt_inh_to_show = len(opt_inh_indices)
    
    # Top left: Initial Excitatory
    ax1 = axes[0, 0]
    exc_vmin, exc_vmax = 0, 0.3
    im1 = ax1.imshow(exc_data, aspect='auto', cmap='Blues', origin='lower', 
                     vmin=exc_vmin, vmax=exc_vmax)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Excitatory Firing Rate', rotation=270, labelpad=15, color='blue')
    cbar1.ax.tick_params(colors='blue')
    
    _setup_plot_style(ax1, f'Initial - Excitatory (n={exc_to_show})', 'Time (ms)', 'Excitatory Segments', fontsize=12)
    ax1.tick_params(axis='y', colors='blue')
    _setup_y_axis_ticks(ax1, exc_indices, 'Exc', exc_to_show, monoconn_seg_indices)
    
    # Top right: Optimized Excitatory
    ax2 = axes[0, 1]
    im2 = ax2.imshow(opt_exc_data, aspect='auto', cmap='Blues', origin='lower', 
                     vmin=exc_vmin, vmax=exc_vmax)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Excitatory Firing Rate', rotation=270, labelpad=15, color='blue')
    cbar2.ax.tick_params(colors='blue')
    
    _setup_plot_style(ax2, f'Optimized - Excitatory (n={opt_exc_to_show})', 'Time (ms)', 'Excitatory Segments', fontsize=12)
    ax2.tick_params(axis='y', colors='blue')
    _setup_y_axis_ticks(ax2, opt_exc_indices, 'Exc', opt_exc_to_show, monoconn_seg_indices)
    
    # Bottom left: Initial Inhibitory
    ax3 = axes[1, 0]
    inh_vmin, inh_vmax = 0, 0.02
    im3 = ax3.imshow(inh_data, aspect='auto', cmap='Reds', origin='lower', 
                     vmin=inh_vmin, vmax=inh_vmax)
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('Inhibitory Firing Rate', rotation=270, labelpad=15, color='red')
    cbar3.ax.tick_params(colors='red')
    
    _setup_plot_style(ax3, f'Initial - Inhibitory (n={inh_to_show})', 'Time (ms)', 'Inhibitory Segments', fontsize=12)
    ax3.tick_params(axis='y', colors='red')
    _setup_y_axis_ticks(ax3, inh_indices, 'Inh', inh_to_show, monoconn_seg_indices)
    
    # Bottom right: Optimized Inhibitory
    ax4 = axes[1, 1]
    im4 = ax4.imshow(opt_inh_data, aspect='auto', cmap='Reds', origin='lower', 
                     vmin=inh_vmin, vmax=inh_vmax)
    cbar4 = plt.colorbar(im4, ax=ax4)
    cbar4.set_label('Inhibitory Firing Rate', rotation=270, labelpad=15, color='red')
    cbar4.ax.tick_params(colors='red')
    
    _setup_plot_style(ax4, f'Optimized - Inhibitory (n={opt_inh_to_show})', 'Time (ms)', 'Inhibitory Segments', fontsize=12)
    ax4.tick_params(axis='y', colors='red')
    _setup_y_axis_ticks(ax4, opt_inh_indices, 'Inh', opt_inh_to_show, monoconn_seg_indices)
    
    # Set overall title
    fig.suptitle(f'{title_prefix} - Average Across Seeds', fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    return _save_and_show_plot(fig, save_path, show_plot=False, message_prefix="Average heatmap")
    
def plot_gradient_norm_history(gradient_norm_history, title="Gradient Norm History",
                               figsize=(10, 6), save_path=None, show_plot=False):
    """Plot gradient norm history curve"""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot gradient norm history
    ax.plot(gradient_norm_history, linewidth=1.5, color='green', alpha=0.8)
    _setup_plot_style(ax, 'Gradient Norm History', 'Iteration', 'Gradient Norm')
    
    if len(gradient_norm_history) > 0:
        stats = _calculate_basic_stats(gradient_norm_history)
        stats_renamed = {
            'Final': stats['Final'], 
            'Min': stats['Min'], 
            'Max': stats['Max'],
            'Mean': stats['Mean']
        }
        _add_statistics_text(ax, stats_renamed, box_color='lightgreen')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return _save_and_show_plot(fig, save_path, show_plot, message_prefix="Gradient norm history")

def plot_loss_and_spike_preds_history(loss_history, spike_preds_history, gradient_norm_history=None,
                                     title="Activity Optimization History", 
                                     figsize=(20, 12), save_path=None, show_plot=False):
    """Plot loss history, spike prediction history, batch difference, correlation, and gradient norm (2x3 subplots)"""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    ax1, ax2, ax5 = axes[0]
    ax3, ax4, ax6 = axes[1]
    
    # Plot loss history (top-left)
    ax1.plot(loss_history, linewidth=1.5, color='blue', alpha=0.8)
    _setup_plot_style(ax1, 'Loss History', 'Iteration', 'Loss')
    
    if len(loss_history) > 0:
        stats = _calculate_basic_stats(loss_history)
        stats_renamed = {'Final': stats['Final'], 'Min': stats['Min'], 'Imp': stats['Improvement']}
        _add_statistics_text(ax1, stats_renamed, box_color='lightblue')
    
    # Plot spike prediction history and batch difference (top-center and bottom-left)
    differences = []
    if len(spike_preds_history) > 0:
        # Extract data once
        batch1_preds = [preds[0] for preds in spike_preds_history]
        batch2_preds = [preds[1] for preds in spike_preds_history]
        differences = [b2 - b1 for b1, b2 in zip(batch1_preds, batch2_preds)]
        iterations = range(len(spike_preds_history))
        
        # Spike prediction history (top-center)
        ax2.plot(iterations, batch1_preds, linewidth=1.5, color='green', alpha=0.8, 
                 label='Batch 1 (Control)', marker='o', markersize=2)
        ax2.plot(iterations, batch2_preds, linewidth=1.5, color='red', alpha=0.8, 
                 label='Batch 2 (Stimulated)', marker='s', markersize=2)
        _setup_plot_style(ax2, 'Spike Predictions Max History', 'Iteration', 'Max Spike Prediction', legend=True, legend_fontsize=9)
        
        # Add spike prediction statistics
        final_batch1 = batch1_preds[-1]
        final_batch2 = batch2_preds[-1]
        difference = final_batch2 - final_batch1
        spike_stats = {
            'Final B1': f"{final_batch1:.4f}",
            'Final B2': f"{final_batch2:.4f}",
            'Diff': f"{difference:.4f}"
        }
        _add_statistics_text(ax2, spike_stats, box_color='lightgreen')
        
        # Batch difference (bottom-left)
        ax3.plot(iterations, differences, linewidth=2, color='purple', alpha=0.8, 
                 marker='o', markersize=3, label='B2-B1 Difference')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        _setup_plot_style(ax3, 'Batch Difference (B2-B1)', 'Iteration', 'Spike Prediction Difference', legend=True, legend_fontsize=9)
        
        # Add difference statistics
        if len(differences) > 0:
            diff_stats = _calculate_basic_stats(differences)
            diff_stats_renamed = {
                'Final': diff_stats['Final'],
                'Max': diff_stats['Max'],
                'Min': diff_stats['Min'],
                'Mean': diff_stats['Mean']
            }
            _add_statistics_text(ax3, diff_stats_renamed, box_color='lightcoral')
    else:
        # No spike history data case
        ax2.text(0.5, 0.5, 'No data available', transform=ax2.transAxes, 
                 ha='center', va='center', fontsize=12)
        _setup_plot_style(ax2, 'Spike Predictions Max History', 'Iteration', 'Max Spike Prediction')
        
        ax3.text(0.5, 0.5, 'No data available', transform=ax3.transAxes, 
                 ha='center', va='center', fontsize=12)
        _setup_plot_style(ax3, 'Batch Difference (B2-B1)', 'Iteration', 'Spike Prediction Difference')
    
    # Correlation between batch difference and loss (top-right)
    try:
        if len(differences) > 0 and len(loss_history) > 0:
            # Align lengths
            n = min(len(differences), len(loss_history))
            dif_arr = np.array(differences[:n])
            loss_arr = np.array(loss_history[:n])
            # Keep only non-negative differences for X-axis
            mask = dif_arr >= 0
            dif_arr = dif_arr[mask]
            loss_arr = loss_arr[mask]
            ax5.scatter(dif_arr, loss_arr, s=6, alpha=0.5, c='tab:purple', edgecolors='none')
            _setup_plot_style(ax5, 'Corr: (B2-B1) vs Loss', 'B2-B1 Difference', 'Loss', grid=True)
            ax5.set_xlim(left=0)
            # Pearson r
            if len(dif_arr) > 1:
                r = np.corrcoef(dif_arr, loss_arr)[0, 1]
                _add_statistics_text(ax5, {'r': f"{r:.4f}", 'N': len(dif_arr)}, box_color='lavender')
        else:
            ax5.text(0.5, 0.5, 'No data available', transform=ax5.transAxes, 
                     ha='center', va='center', fontsize=12)
            _setup_plot_style(ax5, 'Corr: (B2-B1) vs Loss', 'B2-B1 Difference', 'Loss')
    except Exception:
        ax5.text(0.5, 0.5, 'Correlation failed', transform=ax5.transAxes, 
                 ha='center', va='center', fontsize=12)
        _setup_plot_style(ax5, 'Corr: (B2-B1) vs Loss', 'B2-B1 Difference', 'Loss')
    
    # Gradient norm history (bottom-center)
    if gradient_norm_history is not None and len(gradient_norm_history) > 0:
        ax4.plot(gradient_norm_history, linewidth=1.5, color='green', alpha=0.8)
        _setup_plot_style(ax4, 'Gradient Norm History', 'Iteration', 'Gradient Norm')
        stats = _calculate_basic_stats(gradient_norm_history)
        stats_renamed = {
            'Final': stats['Final'], 
            'Min': stats['Min'], 
            'Max': stats['Max'],
            'Mean': stats['Mean']
        }
        _add_statistics_text(ax4, stats_renamed, box_color='lightgreen')
    else:
        ax4.text(0.5, 0.5, 'No data available', transform=ax4.transAxes, 
                 ha='center', va='center', fontsize=12)
        _setup_plot_style(ax4, 'Gradient Norm History', 'Iteration', 'Gradient Norm')
    
    # Placeholder bottom-right (kept empty for balanced grid)
    ax6.axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return _save_and_show_plot(fig, save_path, show_plot, message_prefix="Loss, spike prediction and gradient history")

def plot_firing_rates_evolution(firing_rates_history, num_segments_exc, num_segments_inh, 
                               time_duration_ms, input_window_size, monoconn_seg_indices, 
                               title="Firing Rates Evolution", figsize=(24, 8), save_path=None, show_plot=False):
    """Plot firing rates evolution over iterations"""
    if not firing_rates_history:
        #print("Warning: firing_rates_history is empty, skipping plot")
        return None
    
    # Create 2 rows: first row for exc segments, second row for inh segments
    num_monoconn = len(monoconn_seg_indices)
    fig, axes = plt.subplots(2, num_monoconn, figsize=figsize)
    
    # Ensure axes is 2D array
    if num_monoconn == 1:
        axes = axes.reshape(2, 1)
    
    # Select several time points for visualization
    sample_times = [50, input_window_size // 2, 250]  # 50ms, half window size, 250ms
    
    # First row: Excitatory segments
    for i, segment_idx in enumerate(monoconn_seg_indices):
        ax = axes[0, i]
        for time_idx in sample_times:
                if time_idx < time_duration_ms:
                    values = [fr[0, segment_idx, time_idx] for fr in firing_rates_history]
                ax.plot(range(0, len(firing_rates_history)), values, 
                           label=f'Time {time_idx}ms', linewidth=1.5, alpha=0.8)
            
        _setup_plot_style(ax, f'Exc Segment {segment_idx}', 'Iteration', 'Firing Rate', 
                         fontsize=10, legend=True, legend_fontsize=9)
        
        # Add statistics
        if firing_rates_history:
                final_values = [fr[0, segment_idx, time_idx] for fr in firing_rates_history 
                              for time_idx in sample_times if time_idx < time_duration_ms]
                if final_values:
                    mean_val = np.mean(final_values)
                _add_statistics_text(ax, {'Mean': f"{mean_val:.4f}"}, box_color='lightyellow')
    
    # Second row: Corresponding inhibitory segments
    for i, segment_idx in enumerate(monoconn_seg_indices):
        ax = axes[1, i]
        # Calculate corresponding inhibitory segment index
        inh_segment_idx = segment_idx + num_segments_exc
        
        for time_idx in sample_times:
            if time_idx < time_duration_ms:
                values = [fr[0, inh_segment_idx, time_idx] for fr in firing_rates_history]
                ax.plot(range(0, len(firing_rates_history)*50, 50), values, 
                       label=f'Time {time_idx}ms', linewidth=1.5, alpha=0.8)
        
        _setup_plot_style(ax, f'Inh Segment {inh_segment_idx}', 'Iteration', 'Firing Rate', 
                         fontsize=10, legend=True, legend_fontsize=9)
        
        # Add statistics
        if firing_rates_history:
            final_values = [fr[0, inh_segment_idx, time_idx] for fr in firing_rates_history 
                          for time_idx in sample_times if time_idx < time_duration_ms]
            if final_values:
                mean_val = np.mean(final_values)
                _add_statistics_text(ax, {'Mean': f"{mean_val:.4f}"}, box_color='lightyellow')
    
    # Set overall title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    return _save_and_show_plot(fig, save_path, show_plot, message_prefix="Firing rates evolution")

# def plot_optimization_summary(loss_history, firing_rates_history, num_segments_exc, 
#                              num_segments_inh, time_duration_ms, input_window_size,
#                              title="Optimization Summary", figsize=(20, 12), 
#                              save_path=None, show_plot=False):
#     """Plot optimization summary with multiple subplots"""
#     fig = plt.figure(figsize=figsize)
    
#     # Create grid layout
#     gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
#     # 1. Loss history (top left, spans 2 rows)
#     ax1 = fig.add_subplot(gs[0:2, 0])
#     ax1.plot(loss_history, linewidth=2, color='blue', alpha=0.8)
#     _setup_plot_style(ax1, 'Loss History', 'Iteration', 'Loss')
    
#     # 2. Firing rates evolution and distribution (top right and bottom right)

#     if firing_rates_history:
#         # 2a. Firing rates evolution (top right, spans 2 rows)
#         ax2 = fig.add_subplot(gs[0:2, 1])
#         # Select representative segments
#         sample_segments = [0, num_segments_exc//2, num_segments_exc, 
#                          num_segments_exc + num_segments_inh//2]
#         sample_times = [input_window_size // 2]  # Only show key time points
        
#         for segment_idx in sample_segments:
#             if segment_idx < len(firing_rates_history[0][0]):
#                 values = [fr[0, segment_idx, sample_times[0]] for fr in firing_rates_history 
#                          if sample_times[0] < time_duration_ms]
#                 if values:
#                     segment_type = "Exc" if segment_idx < num_segments_exc else "Inh"
#                     color = 'blue' if segment_idx < num_segments_exc else 'red'
#                     ax2.plot(range(0, len(values)*50, 50), values, 
#                             label=f'{segment_type} {segment_idx}', color=color, alpha=0.8)
        
#         _setup_plot_style(ax2, 'Firing Rates Evolution', 'Iteration', 'Firing Rate', legend=True, legend_fontsize=8)
        
#         # 2b. Final firing rates distribution (bottom right)
#         ax3 = fig.add_subplot(gs[0:2, 2])
#         final_rates = firing_rates_history[-1][0]  # Take first sample from last batch
        
#         # Distribution of excitatory and inhibitory segments
#         exc_rates = final_rates[:num_segments_exc, :]
#         inh_rates = final_rates[num_segments_inh:, :]
        
#         # Calculate mean firing rate for each segment
#         exc_means = np.mean(exc_rates, axis=1)
#         inh_means = np.mean(inh_rates, axis=1)
        
#         # Plot distribution
#         ax3.hist(exc_means, bins=30, alpha=0.7, color='blue', label='Excitatory', density=True)
#         ax3.hist(inh_means, bins=30, alpha=0.7, color='red', label='Inhibitory', density=True)
#         _setup_plot_style(ax3, 'Final Firing Rates Distribution', 'Mean Firing Rate', 'Density', legend=True)
    
#     # 3. Optimization statistics (bottom, spans 3 columns)
#     ax4 = fig.add_subplot(gs[2, :])
#     ax4.axis('off')
    
#     # Create statistics text - combine all conditions
#     stats_text = "Optimization Statistics:\n"
    
#     # Loss history statistics
#     if len(loss_history) > 0:
#         final_loss = loss_history[-1]
#         min_loss = min(loss_history)
#         improvement = loss_history[0] - final_loss
        
#         # Add to plot
#         loss_stats_text = f"Final: {final_loss:.6f}\nMin: {min_loss:.6f}\nImp: {improvement:.6f}"
#         ax1.text(0.02, 0.98, loss_stats_text, transform=ax1.transAxes, fontsize=9,
#                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
#         # Add to summary
#         stats_text += f"• Total Iterations: {len(loss_history)}\n"
#         stats_text += f"• Initial Loss: {loss_history[0]:.6f}\n"
#         stats_text += f"• Final Loss: {final_loss:.6f}\n"
#         stats_text += f"• Loss Improvement: {improvement:.6f}\n"
#         stats_text += f"• Convergence: {'Yes' if abs(final_loss - loss_history[-10]) < 1e-6 else 'Partial'}\n"
    
#     # Firing rates statistics
#     if firing_rates_history:
#         stats_text += f"• Firing Rates History Points: {len(firing_rates_history)}\n"
#         stats_text += f"• Segments: {num_segments_exc} (Exc) + {num_segments_inh} (Inh) = {num_segments_exc + num_segments_inh}\n"
#         stats_text += f"• Time Duration: {time_duration_ms} ms\n"
#         stats_text += f"• Input Window: {input_window_size} ms"
    
#     ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=11,
#             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
#     # Set overall title
#     fig.suptitle(title, fontsize=16, fontweight='bold')
    
#     return _save_and_show_plot(fig, save_path, show_plot, message_prefix="Optimization summary")

def create_optimization_report(loss_history, firing_rates_history, spike_preds_history,
                               gradient_norm_history, monoconn_seg_indices, num_segments_exc, num_segments_inh, 
                              time_duration_ms, input_window_size,
                              save_dir, report_name="optimization_report"):
    """Create complete optimization report with all visualizations"""
    os.makedirs(save_dir, exist_ok=True)
    
    #print(f"Starting optimization report generation: {report_name}")
    
    
    # 1. Loss, spike predictions history and gradient norm (2x2 combined)
    if spike_preds_history or gradient_norm_history:
        plot_loss_and_spike_preds_history(
            loss_history, spike_preds_history, gradient_norm_history,
            save_path=os.path.join(save_dir, f'loss_spike_grad_history.pdf')
        )
    
    # # 2. Firing rates evolution
    # if firing_rates_history:
    #     plot_firing_rates_evolution(
    #         firing_rates_history, num_segments_exc, num_segments_inh,
    #         time_duration_ms, input_window_size, monoconn_seg_indices,
    #         save_path=os.path.join(save_dir, f'firing_rates_evolution.pdf')
    #     )
    
    # # 4. Optimization summary
    # plot_optimization_summary(
    #     loss_history, firing_rates_history, num_segments_exc, num_segments_inh,
    #     time_duration_ms, input_window_size,
    #     save_path=os.path.join(save_dir, f'summary.pdf')
    # )
    
    initial_firing_rates = firing_rates_history[0]
    optimized_firing_rates = firing_rates_history[-1]
    # 5. Optimized firing rates visualization
    heatmap_data = visualize_optimized_firing_rates(
        initial_firing_rates, optimized_firing_rates, 
        monoconn_seg_indices, 
        num_segments_exc, num_segments_inh,
        save_dir, title_prefix=f"{report_name.title()} - Optimized"
    )

    print("Optimization report completed \n")
    
    return heatmap_data
    