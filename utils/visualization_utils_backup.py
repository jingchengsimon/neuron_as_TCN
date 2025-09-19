import numpy as np
import matplotlib.pyplot as plt
import os

# visualization functions for 5_main_figure_replication.py

# Common utility functions
def _setup_segments_data(firing_rates, num_exc_segments, max_segments_to_show, specified_segments=None):
    """Extract and sample excitatory/inhibitory segments data"""
    num_segments_total, _ = firing_rates.shape
    num_inh_segments = num_segments_total - num_exc_segments
    
    exc_data = firing_rates[:num_exc_segments, :]
    inh_data = firing_rates[num_exc_segments:, :]
    
    # Process excitatory segments
    if specified_segments is not None and len(specified_segments) > 0:
        exc_specified = [idx for idx in specified_segments if idx < num_exc_segments]
        if len(exc_specified) > 0:
            exc_data_selected = exc_data[exc_specified, :]
            exc_indices = np.array(exc_specified)
            exc_to_show = len(exc_specified)
        else:
            exc_data_selected, exc_indices, exc_to_show = _sample_segments(exc_data, max_segments_to_show // 2, "excitatory")
    else:
        exc_data_selected, exc_indices, exc_to_show = _sample_segments(exc_data, max_segments_to_show // 2, "excitatory")
    
    # Process inhibitory segments
    if specified_segments is not None and len(specified_segments) > 0:
        inh_specified = [idx for idx in specified_segments if idx >= num_exc_segments and idx < num_segments_total]
        if len(inh_specified) > 0:
            inh_data_selected = inh_data[np.array(inh_specified) - num_exc_segments, :]
            inh_indices = np.array(inh_specified)
            inh_to_show = len(inh_specified)
        else:
            inh_data_selected, inh_indices, inh_to_show = _sample_segments(inh_data, max_segments_to_show // 2, "inhibitory")
    else:
        inh_data_selected, inh_indices, inh_to_show = _sample_segments(inh_data, max_segments_to_show // 2, "inhibitory")
    
    return (exc_data_selected, exc_indices, exc_to_show, 
            inh_data_selected, inh_indices, inh_to_show, 
            num_exc_segments, num_inh_segments)

def _sample_segments(data, max_to_show, segment_type):
    """Sample segments if needed"""
    if data.shape[0] > max_to_show:
        indices = np.linspace(0, data.shape[0]-1, max_to_show, dtype=int)
        data_selected = data[indices, :]
        return data_selected, indices, max_to_show
    else:
        return data, np.arange(data.shape[0]), data.shape[0]

def _setup_axes_labels(ax, segment_type, num_shown, total, indices, max_ticks=20):
    """Setup y-axis labels and ticks for segment plots"""
    if num_shown <= 0:
        ax.set_yticks([])
        ax.set_yticklabels([])
        return
    
    if num_shown <= max_ticks:
        y_ticks = list(range(num_shown))
        y_labels = [f'{segment_type[:3].title()} {indices[i]}' for i in y_ticks]
    else:
        y_ticks = [0, num_shown//4, num_shown//2, 3*num_shown//4, num_shown-1]
        y_labels = [f'{segment_type[:3].title()} {indices[i]}' for i in y_ticks if i < len(indices)]
    
    ax.set_yticks(y_ticks[:len(y_labels)])
    ax.set_yticklabels(y_labels)

def _add_statistics_text(ax, data, segment_type, color):
    """Add statistics text box to plot"""
    stats_text = f"Range: [{np.min(data):.4f}, {np.max(data):.4f}]\n"
    stats_text += f"Mean: {np.mean(data):.4f} ± {np.std(data):.4f}"
    
    box_color = 'lightblue' if segment_type == 'excitatory' else 'lightcoral'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))

def _safe_save_fig(fig, save_path, dpi=300, bbox_inches='tight'):
    """Safely save matplotlib figure with fallback options"""
    try:
        fig.canvas.draw()
        plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, 
                   facecolor='white', edgecolor='none')
    except Exception:
        try:
            plt.savefig(save_path, dpi=dpi, facecolor='white', edgecolor='none')
        except Exception:
            plt.savefig(save_path, dpi=150)

def plot_summary_panels(fpr, tpr, desired_fp_ind, y_spikes_GT, y_spikes_hat, 
                       y_soma_GT, y_soma_hat, desired_threshold, save_path):
    """Plot summary panels"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # ROC curve
    ax_roc = axes[0]
    ax_roc.plot(fpr, tpr, 'k-', linewidth=2)
    ax_roc.plot(fpr[desired_fp_ind], tpr[desired_fp_ind], 'o', color='red', markersize=6)
    ax_roc.set_xlabel('False alarm rate')
    ax_roc.set_ylabel('Hit rate')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    
    # Add zoomed inset
    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax_roc, width='45%', height='45%', loc='lower right')
        axins.plot(fpr, tpr, 'k-', linewidth=1.5)
        axins.plot(fpr[desired_fp_ind], tpr[desired_fp_ind], 'o', color='red', markersize=4)
        axins.set_xlim(0.0, max(0.04, float(fpr[min(len(fpr)-1, desired_fp_ind*3)])))
        axins.set_ylim(0.0, 1.0)
        axins.set_xlabel('False alarm rate', fontsize=8)
        axins.set_ylabel('Hit rate', fontsize=8)
        axins.tick_params(axis='both', which='major', labelsize=7)
    except Exception:
        pass

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
    ax_xcorr.set_xlabel('Δt (ms)')
    ax_xcorr.set_ylabel('spike rate (Hz)')

    # Voltage scatter plot
    ax_scatter = axes[2]
    x, y = y_soma_GT.ravel(), y_soma_hat.ravel()
    
    # Downsample for performance
    if len(x) > 50000:
        idx = np.random.choice(len(x), 50000, replace=False)
        x, y = x[idx], y[idx]
    
    ax_scatter.scatter(x, y, s=4, c='tab:blue', alpha=0.25, edgecolors='none')
    ax_scatter.set_xlabel('L5PC Model (mV)')
    ax_scatter.set_ylabel('ANN (mV)')
    ax_scatter.set_xlim(-80, -57)
    ax_scatter.set_ylim(-80, -57)

    _safe_save_fig(fig, save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Summary panels generated: {save_path}')

def plot_voltage_traces(y_spikes_GT, y_spikes_hat, y_voltage_GT, y_voltage_hat, 
                       global_threshold, selected_traces, output_dir, 
                       per_sample_global_metrics=None, per_sample_optimal_metrics=None):
    """Plot voltage traces with dynamic number of figures."""
    total = len(selected_traces)
    if total == 0:
        print('No selected traces to plot.')
        return

    page_size = 5
    num_pages = int(np.ceil(total / page_size))

    for page_idx in range(num_pages):
        start = page_idx * page_size
        end = min((page_idx + 1) * page_size, total)
        traces_page = selected_traces[start:end]

        num_subplots = len(traces_page)
        fig, axes = plt.subplots(nrows=num_subplots, ncols=1, figsize=(15, 3*num_subplots), sharex=True)
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.01, hspace=0.3)

        axes_norm = axes if isinstance(axes, (list, np.ndarray)) else [axes]

        for k, selected_trace in enumerate(traces_page):
            _plot_single_trace(axes_norm[k], selected_trace, traces_page, k, 
                             y_spikes_GT, y_spikes_hat, y_voltage_GT, y_voltage_hat,
                             global_threshold, per_sample_global_metrics, per_sample_optimal_metrics)

        axes_norm[-1].set_xlabel('Time (s)', fontsize=12)

        # Save figure
        if num_pages == 1:
            save_path = f'{output_dir}/voltage_traces.png'
        else:
            save_path = f'{output_dir}/voltage_traces_p{page_idx+1}.png'
        
        _safe_save_fig(fig, save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f'Voltage traces generated: {num_pages} page(s)')

def _plot_single_trace(ax, selected_trace, traces_page, k, y_spikes_GT, y_spikes_hat, 
                      y_voltage_GT, y_voltage_hat, global_threshold, 
                      per_sample_global_metrics, per_sample_optimal_metrics):
    """Plot a single voltage trace"""
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

            # Set axes
            ax.set_xlim(0.02, sim_duration_ms / 1000.0)
            ax.set_ylim(-80, 5)
            ax.set_ylabel('$V_m$ (mV)', fontsize=12)

            # Add title
            if global_metrics is not None and optimal_metrics is not None:
                title_text = (f'Sim {selected_trace}: Global FPR={global_metrics["fpr"]:.4f}, '
                              f'Optimal FPR={optimal_metrics["fpr"]:.4f}, '
                              f'FP={global_metrics["false_positives"]}, '
                              f'TP={global_metrics["true_positives"]}')
            else:
                title_text = f'Sim {selected_trace}'
            ax.set_title(title_text, fontsize=10)

            # Set legend on the first subplot of each figure
            if k == 0:
                ax.legend(loc='upper right', fontsize=8)

            # Hide x-axis labels for non-last subplot
    if k < len(traces_page) - 1:
                plt.setp(ax.get_xticklabels(), visible=False)


# Visualization functions for activity optimization process (6_activity_optimization.py)

def visualize_firing_rates_trace(firing_rates, num_exc_segments=639, num_inh_segments=640, save_path=None, 
                                 title="Firing Rates Visualization", figsize=(18, 12), 
                                 time_step_ms=1, max_segments_to_show=200, specified_segments=None):
    """Visualize firing rate array using raster plot style with continuous curves"""
    # Setup segments data
    (exc_data_selected, exc_indices, exc_to_show, 
     inh_data_selected, inh_indices, inh_to_show, 
     num_exc_segments, num_inh_segments) = _setup_segments_data(
        firing_rates, num_exc_segments, max_segments_to_show, specified_segments)
    
    # Create time axis
    time_duration_ms = firing_rates.shape[1]
    time_axis = np.arange(time_duration_ms) * time_step_ms
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot excitatory segments
    _plot_firing_rates_traces(ax1, exc_data_selected, exc_indices, time_axis, 
                             'excitatory', exc_to_show, num_exc_segments, 4)
    
    # Plot inhibitory segments  
    _plot_firing_rates_traces(ax2, inh_data_selected, inh_indices, time_axis,
                             'inhibitory', inh_to_show, num_inh_segments, 40)
    
    # Set overall title and layout
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save or display
    if save_path:
        _safe_save_fig(fig, save_path, dpi=300, bbox_inches='tight')
        print(f"Firing rates trace generated: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        'mean_firing_rate_exc': np.mean(exc_data_selected),
        'mean_firing_rate_inh': np.mean(inh_data_selected),
        'exc_segments_shown': exc_to_show,
        'inh_segments_shown': inh_to_show,
        'total_segments': firing_rates.shape[0]
    }

def _plot_firing_rates_traces(ax, data_selected, indices, time_axis, segment_type, 
                             num_shown, total_segments, scale_factor):
    """Plot firing rate traces for a segment type"""
    color = 'blue' if segment_type == 'excitatory' else 'red'
    y_positions = np.arange(len(indices))
    
    # Plot firing rate for each segment
    for i in range(data_selected.shape[0]):
        segment_firing_rate = data_selected[i, :]
        scaled_firing_rate = segment_firing_rate * scale_factor
        y_base = y_positions[i]
        y_values = y_base + scaled_firing_rate
        
        ax.plot(time_axis, y_values, color=color, linewidth=0.5, alpha=0.7)
        ax.axhline(y=y_base, color='lightgray', linewidth=0.2, alpha=0.5)
    
    # Set axes properties
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel(f'{segment_type.title()} Segments', fontsize=12, color=color)
    ax.set_title(f'{segment_type.title()} Segments (n={num_shown}/{total_segments})', 
                fontsize=14, color=color)
    ax.tick_params(axis='y', colors=color)
    ax.grid(True, alpha=0.3)
    
    # Setup labels and statistics
    _setup_axes_labels(ax, segment_type, num_shown, total_segments, indices)
    _add_statistics_text(ax, data_selected, segment_type, color)

def visualize_firing_rates_heatmap(firing_rates, num_exc_segments=639, num_inh_segments=640, save_path=None, 
                                  title="Firing Rates Heatmap", figsize=(18, 12), 
                                  max_segments_to_show=300, specified_segments=None, 
                                  monoconn_seg_indices=None):
    """Visualize firing rate array using heatmap style"""
    # Setup segments data
    (exc_data_selected, exc_indices, exc_to_show, 
     inh_data_selected, inh_indices, inh_to_show, 
     num_exc_segments, num_inh_segments) = _setup_segments_data(
        firing_rates, num_exc_segments, max_segments_to_show, specified_segments)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot excitatory heatmap
    _plot_firing_rates_heatmap(ax1, exc_data_selected, exc_indices, 
                              'excitatory', exc_to_show, num_exc_segments, 
                              monoconn_seg_indices, vmin=0, vmax=0.3, cmap='Blues')
    
    # Plot inhibitory heatmap
    _plot_firing_rates_heatmap(ax2, inh_data_selected, inh_indices,
                              'inhibitory', inh_to_show, num_inh_segments,
                              monoconn_seg_indices, vmin=0, vmax=0.02, cmap='Reds')
    
    # Set overall title and layout
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save or display
    if save_path:
        _safe_save_fig(fig, save_path, dpi=300, bbox_inches='tight')
        print(f"Firing rates heatmap generated: {save_path}")
            else:
        plt.show()
    
    plt.close()

def _plot_firing_rates_heatmap(ax, data_selected, indices, segment_type, num_shown, 
                              total_segments, monoconn_seg_indices, vmin, vmax, cmap):
    """Plot firing rate heatmap for a segment type"""
    color = 'blue' if segment_type == 'excitatory' else 'red'
    
    # Create heatmap
    im = ax.imshow(data_selected, aspect='auto', cmap=cmap, origin='lower', 
                   vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{segment_type.title()} Firing Rate', rotation=270, labelpad=15, color=color)
    cbar.ax.tick_params(colors=color)
    
    # Set axes properties
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel(f'{segment_type.title()} Segments', fontsize=12, color=color)
    ax.set_title(f'{segment_type.title()} Segments (n={num_shown}/{total_segments})', 
                fontsize=14, color=color)
    ax.tick_params(axis='y', colors=color)
    
    # Setup labels and statistics
    _setup_heatmap_labels(ax, segment_type, num_shown, indices, monoconn_seg_indices)
    _add_statistics_text(ax, data_selected, segment_type, color)

def _setup_heatmap_labels(ax, segment_type, num_shown, indices, monoconn_seg_indices=None):
    """Setup y-axis labels for heatmap plots"""
    if num_shown <= 0:
        ax.set_yticks([])
        ax.set_yticklabels([])
        return
    
    if num_shown <= 20:
        y_ticks = list(range(num_shown))
        y_labels = [f'{segment_type[:3].title()} {indices[i]}' for i in y_ticks]
        elif monoconn_seg_indices is not None and len(monoconn_seg_indices) > 0:
        # Show only monoconn segments if available
            fixed_positions = []
            for fixed_idx in monoconn_seg_indices:
            if fixed_idx in indices:
                pos = np.where(indices == fixed_idx)[0]
                    if len(pos) > 0:
                        fixed_positions.append(pos[0])
            
            if len(fixed_positions) > 0:
            y_ticks = fixed_positions
            y_labels = [f'{segment_type[:3].title()} {indices[pos]}' for pos in fixed_positions]
            else:
            y_ticks = [0, num_shown//4, num_shown//2, 3*num_shown//4, num_shown-1]
            y_labels = [f'{segment_type[:3].title()} {indices[i]}' for i in y_ticks if i < len(indices)]
        else:
        y_ticks = [0, num_shown//4, num_shown//2, 3*num_shown//4, num_shown-1]
        y_labels = [f'{segment_type[:3].title()} {indices[i]}' for i in y_ticks if i < len(indices)]
    
    ax.set_yticks(y_ticks[:len(y_labels)])
    ax.set_yticklabels(y_labels)
    
def visualize_optimized_firing_rates(optimized_firing_rates, monoconn_seg_indices, 
                                   num_exc_segments=639, num_inh_segments=640,
                                   save_dir=None, title_prefix="Optimized Firing Rates"):
    """Visualize optimized firing rates"""
    if save_dir is None:
        print("Warning: No save directory specified, skipping visualization")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    optimized_sample = optimized_firing_rates[0]  # (num_segments, time_duration)
    
    # Setup specified segments for visualization
    specified_segments = _setup_optimized_segments(monoconn_seg_indices, num_inh_segments)
    
    # Generate raster plot
    raster_save_path = os.path.join(save_dir, 'optimized_firing_rate_raster.png')
    visualize_firing_rates_trace(
        firing_rates=optimized_sample,
        num_exc_segments=num_exc_segments,
        num_inh_segments=num_inh_segments,
        save_path=raster_save_path,
        title=f"{title_prefix} - Raster Plot",
        max_segments_to_show=10,
        specified_segments=specified_segments
    )
    
    # Generate heatmap
    heatmap_save_path = os.path.join(save_dir, 'optimized_firing_rate_heatmap.png')
    visualize_firing_rates_heatmap(
        firing_rates=optimized_sample,
        num_exc_segments=num_exc_segments,
        num_inh_segments=num_inh_segments,
        save_path=heatmap_save_path,
        title=f"{title_prefix} - Heatmap",
        max_segments_to_show=400,
        specified_segments=specified_segments,
        monoconn_seg_indices=monoconn_seg_indices
    )
    
    print("Optimized firing rates visualization completed")
    return {'raster_path': raster_save_path, 'heatmap_path': heatmap_save_path}

def _setup_optimized_segments(monoconn_seg_indices, num_inh_segments):
    """Setup specified segments for optimized visualization"""
    if monoconn_seg_indices is None or len(monoconn_seg_indices) == 0:
        return None
    
    # Extend monoconn_seg_indices to include surrounding segments
    extended_indices = []
    for idx in monoconn_seg_indices:
        num_segments_to_add = 30
        start_idx = max(0, idx - num_segments_to_add)
        end_idx = min(num_inh_segments, idx + num_segments_to_add + 1)
        extended_indices.extend(range(start_idx, end_idx))
    
    return sorted(list(set(extended_indices)))

def plot_loss_and_spike_preds_history(loss_history, spike_preds_history, 
                                     title="Activity Optimization History", 
                                     figsize=(20, 6), save_path=None, show_plot=False):
    """Plot loss history curve, spike prediction history curve, and batch difference"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Plot loss history
    _plot_loss_history(ax1, loss_history)
    
    # Plot spike prediction history
    _plot_spike_preds_history(ax2, spike_preds_history)
    
    # Plot batch difference
    _plot_batch_difference(ax3, spike_preds_history)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save or display
    if save_path:
        _safe_save_fig(fig, save_path, dpi=300, bbox_inches='tight')
        print(f"Loss and spike prediction history generated: {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()
    return save_path

def _plot_loss_history(ax, loss_history):
    """Plot loss history with statistics"""
    ax.plot(loss_history, linewidth=1.5, color='blue', alpha=0.8)
    ax.set_title('Loss History', fontsize=12, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if len(loss_history) > 0:
        final_loss = loss_history[-1]
        min_loss = min(loss_history)
        improvement = loss_history[0] - final_loss
        
        stats_text = f"Final: {final_loss:.6f}\nMin: {min_loss:.6f}\nImp: {improvement:.6f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

def _plot_spike_preds_history(ax, spike_preds_history):
    """Plot spike prediction history with statistics"""
    if len(spike_preds_history) == 0:
        return
    
        batch1_preds = [preds[0] for preds in spike_preds_history]
        batch2_preds = [preds[1] for preds in spike_preds_history]
        iterations = range(len(spike_preds_history))
        
    ax.plot(iterations, batch1_preds, linewidth=1.5, color='green', alpha=0.8, 
                label='Batch 1 (Control)', marker='o', markersize=2)
    ax.plot(iterations, batch2_preds, linewidth=1.5, color='red', alpha=0.8, 
                label='Batch 2 (Stimulated)', marker='s', markersize=2)
        
    ax.set_title('Spike Predictions Max History', fontsize=12, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Max Spike Prediction', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
        final_batch1 = batch1_preds[-1]
        final_batch2 = batch2_preds[-1]
        difference = final_batch2 - final_batch1
        
    stats_text = f"Final B1: {final_batch1:.4f}\nFinal B2: {final_batch2:.4f}\nDiff: {difference:.4f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

def _plot_batch_difference(ax, spike_preds_history):
    """Plot batch difference (batch2 - batch1) over iterations"""
    if len(spike_preds_history) == 0:
        ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12)
        ax.set_title('Batch Difference (B2-B1)', fontsize=12, fontweight='bold')
        return
    
    batch1_preds = [preds[0] for preds in spike_preds_history]
    batch2_preds = [preds[1] for preds in spike_preds_history]
    differences = [b2 - b1 for b1, b2 in zip(batch1_preds, batch2_preds)]
    iterations = range(len(spike_preds_history))
    
    # Plot difference curve
    ax.plot(iterations, differences, linewidth=2, color='purple', alpha=0.8, 
            marker='o', markersize=3, label='B2-B1 Difference')
    
    # Add zero line for reference
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_title('Batch Difference (B2-B1)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Spike Prediction Difference', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Add statistics
    if len(differences) > 0:
        final_diff = differences[-1]
        max_diff = max(differences)
        min_diff = min(differences)
        mean_diff = np.mean(differences)
        
        stats_text = f"Final: {final_diff:.4f}\nMax: {max_diff:.4f}\nMin: {min_diff:.4f}\nMean: {mean_diff:.4f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

def plot_firing_rates_evolution(firing_rates_history, num_segments_exc, num_segments_inh, 
                               time_duration_ms, input_window_size, title="Firing Rates Evolution",
                               figsize=(15, 10), save_path=None, show_plot=False):
    """Plot firing rates evolution over iterations"""
    if not firing_rates_history:
        print("Warning: firing_rates_history is empty, skipping plot")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Select sample points for visualization
    sample_times = [50, input_window_size // 2, 250]
    sample_segments = [0, num_segments_exc//2, num_segments_exc, 
                      num_segments_exc + num_segments_inh//2]
    
    for i, (ax, segment_idx) in enumerate(zip(axes.flat, sample_segments)):
        if i < len(sample_segments):
            _plot_segment_evolution(ax, firing_rates_history, segment_idx, sample_times, 
                                  time_duration_ms, num_segments_exc)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    if save_path:
        _safe_save_fig(fig, save_path, dpi=300, bbox_inches='tight')
        print(f"Firing rates evolution generated: {save_path}")
    if show_plot:
        plt.show()
    plt.close()
    
    return save_path

def _plot_segment_evolution(ax, firing_rates_history, segment_idx, sample_times, 
                          time_duration_ms, num_segments_exc):
    """Plot evolution for a single segment"""
            for time_idx in sample_times:
                if time_idx < time_duration_ms:
                    values = [fr[0, segment_idx, time_idx] for fr in firing_rates_history]
            ax.plot(range(len(firing_rates_history)), values, 
                           label=f'Time {time_idx}ms', linewidth=1.5, alpha=0.8)
            
            segment_type = "Exc" if segment_idx < num_segments_exc else "Inh"
            ax.set_title(f'Segment {segment_idx} ({segment_type})', fontsize=12, fontweight='bold')
            ax.set_xlabel('Iteration', fontsize=10)
            ax.set_ylabel('Firing Rate', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
    # Add mean statistics
            if firing_rates_history:
                final_values = [fr[0, segment_idx, time_idx] for fr in firing_rates_history 
                              for time_idx in sample_times if time_idx < time_duration_ms]
                if final_values:
                    mean_val = np.mean(final_values)
            ax.text(0.02, 0.98, f'Mean: {mean_val:.4f}', transform=ax.transAxes, 
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

def plot_optimization_summary(loss_history, firing_rates_history, num_segments_exc, 
                             num_segments_inh, time_duration_ms, input_window_size,
                             title="Optimization Summary", figsize=(20, 12), 
                             save_path=None, show_plot=False):
    """Plot optimization process summary with multiple subplots"""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot loss history
    ax1 = fig.add_subplot(gs[0:2, 0])
    _plot_loss_history(ax1, loss_history)
    
    # Plot firing rates evolution
    if firing_rates_history:
        ax2 = fig.add_subplot(gs[0:2, 1])
        _plot_summary_firing_rates_evolution(ax2, firing_rates_history, num_segments_exc, 
                                           num_segments_inh, time_duration_ms, input_window_size)
    
    # Plot final distribution
    if firing_rates_history:
        ax3 = fig.add_subplot(gs[0:2, 2])
        _plot_final_distribution(ax3, firing_rates_history[-1][0], num_segments_exc, num_segments_inh)
    
    # Add statistics text
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    _add_optimization_stats(ax4, loss_history, firing_rates_history, 
                           num_segments_exc, num_segments_inh, time_duration_ms, input_window_size)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    if save_path:
        _safe_save_fig(fig, save_path, dpi=300, bbox_inches='tight')
        print(f"Optimization summary generated: {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()
    return save_path

def _plot_summary_firing_rates_evolution(ax, firing_rates_history, num_segments_exc, 
                                       num_segments_inh, time_duration_ms, input_window_size):
    """Plot firing rates evolution for summary"""
        sample_segments = [0, num_segments_exc//2, num_segments_exc, 
                         num_segments_exc + num_segments_inh//2]
    sample_times = [input_window_size // 2]
        
        for segment_idx in sample_segments:
            if segment_idx < len(firing_rates_history[0][0]):
                values = [fr[0, segment_idx, sample_times[0]] for fr in firing_rates_history 
                         if sample_times[0] < time_duration_ms]
                if values:
                    segment_type = "Exc" if segment_idx < num_segments_exc else "Inh"
                    color = 'blue' if segment_idx < num_segments_exc else 'red'
                ax.plot(range(0, len(values)*10, 10), values, 
                            label=f'{segment_type} {segment_idx}', color=color, alpha=0.8)
        
    ax.set_title('Firing Rates Evolution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Firing Rate')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def _plot_final_distribution(ax, final_rates, num_segments_exc, num_segments_inh):
    """Plot final firing rates distribution"""
        exc_rates = final_rates[:num_segments_exc, :]
        inh_rates = final_rates[num_segments_inh:, :]
        
        exc_means = np.mean(exc_rates, axis=1)
        inh_means = np.mean(inh_rates, axis=1)
        
    ax.hist(exc_means, bins=30, alpha=0.7, color='blue', label='Excitatory', density=True)
    ax.hist(inh_means, bins=30, alpha=0.7, color='red', label='Inhibitory', density=True)
    ax.set_title('Final Firing Rates Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Mean Firing Rate')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

def _add_optimization_stats(ax, loss_history, firing_rates_history, 
                          num_segments_exc, num_segments_inh, time_duration_ms, input_window_size):
    """Add optimization statistics text"""
    stats_text = "Optimization Statistics:\n"
    
    if len(loss_history) > 0:
        stats_text += f"• Total Iterations: {len(loss_history)}\n"
        stats_text += f"• Initial Loss: {loss_history[0]:.6f}\n"
        stats_text += f"• Final Loss: {loss_history[-1]:.6f}\n"
        stats_text += f"• Loss Improvement: {loss_history[0] - loss_history[-1]:.6f}\n"
        convergence = 'Yes' if len(loss_history) > 10 and abs(loss_history[-1] - loss_history[-10]) < 1e-6 else 'Partial'
        stats_text += f"• Convergence: {convergence}\n"
    
    if firing_rates_history:
        stats_text += f"• Firing Rates History Points: {len(firing_rates_history)}\n"
        stats_text += f"• Segments: {num_segments_exc} (Exc) + {num_segments_inh} (Inh) = {num_segments_exc + num_segments_inh}\n"
        stats_text += f"• Time Duration: {time_duration_ms} ms\n"
        stats_text += f"• Input Window: {input_window_size} ms"
    
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

def create_optimization_report(loss_history, firing_rates_history, spike_preds_history,
                              optimized_firing_rates, monoconn_seg_indices,
                              num_segments_exc, num_segments_inh, 
                              time_duration_ms, input_window_size,
                              save_dir, report_name="optimization_report"):
    """Create complete optimization report with all visualization content"""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Generating optimization report: {report_name}")
    
    # Generate all visualization components
    _generate_report_components(loss_history, spike_preds_history, firing_rates_history,
                              optimized_firing_rates, monoconn_seg_indices,
                              num_segments_exc, num_segments_inh, 
                              time_duration_ms, input_window_size, save_dir, report_name)
    
    print(f"Optimization report completed: {save_dir}")
    return save_dir

def _generate_report_components(loss_history, spike_preds_history, firing_rates_history,
                              optimized_firing_rates, monoconn_seg_indices,
                              num_segments_exc, num_segments_inh, 
                              time_duration_ms, input_window_size, save_dir, report_name):
    """Generate all components of the optimization report"""
    # Loss and spike predictions history
    plot_loss_and_spike_preds_history(
        loss_history=loss_history,
        spike_preds_history=spike_preds_history,
        title="Activity Optimization: Loss and Spike Predictions History",
        save_path=os.path.join(save_dir, 'loss_and_spike_preds_history.png')
    )
    
    # # Firing rates evolution
    # if firing_rates_history:
    #     plot_firing_rates_evolution(
    #         firing_rates_history, num_segments_exc, num_segments_inh,
    #         time_duration_ms, input_window_size,
    #         save_path=os.path.join(save_dir, 'firing_rates_evolution.png')
    #     )
    
    # # Optimization summary
    # plot_optimization_summary(
    #     loss_history, firing_rates_history, num_segments_exc, num_segments_inh,
    #     time_duration_ms, input_window_size,
    #     save_path=os.path.join(save_dir, 'summary.png')
    # )
    
    # Optimized firing rates visualization
    visualize_optimized_firing_rates(
        optimized_firing_rates, monoconn_seg_indices, 
        num_segments_exc, num_segments_inh,
        save_dir, title_prefix=f"{report_name.title()} - Optimized"
    )


    