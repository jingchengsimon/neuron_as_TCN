import numpy as np
import matplotlib.pyplot as plt
import os

def demo_visualization():
    """
    Demonstrate how to use visualization functions
    """
    print("=== Firing Rates Visualization Demo ===")
    
    # Create sample data, format: (num_segments, time_duration)
    num_exc_segments = 639  # Corrected to 639
    num_inh_segments = 640  
    num_total_segments = num_exc_segments + num_inh_segments  # 1279
    time_duration = 310
    
    print(f"Generating sample data: {num_total_segments} segments, {time_duration} ms")
    print(f"Data format: (segments, time) = ({num_total_segments}, {time_duration})")
    
    # Generate simulated firing rates
    np.random.seed(42)  # For reproducibility
    
    # Excitatory segments: lower baseline firing rate with peaks at certain time points
    exc_firing_rates = np.random.uniform(0.005, 0.02, (num_exc_segments, time_duration))
    
    # Add some peaks around 150ms
    peak_time = 150
    peak_width = 20
    for i in range(0, num_exc_segments, 50):  # Add a peak every 50 segments
        start_time = max(0, peak_time - peak_width//2)
        end_time = min(time_duration, peak_time + peak_width//2)
        exc_firing_rates[i:min(i+10, num_exc_segments), start_time:end_time] += 0.03
    
    # Inhibitory segments: slightly higher baseline firing rate
    inh_firing_rates = np.random.uniform(0.01, 0.03, (num_inh_segments, time_duration))
    
    # 组合数据: (num_total_segments, time_duration)
    firing_rates = np.concatenate([exc_firing_rates, inh_firing_rates], axis=0)
    
    print(f"最终数据形状: {firing_rates.shape}")
    print(f"数据范围: [{np.min(firing_rates):.4f}, {np.max(firing_rates):.4f}]")
    
    # 创建保存目录
    save_dir = "./firing_rates_visualization_demo/"
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 类似raster plot的可视化
    print("\n1. 生成类似raster plot的可视化...")
    raster_save_path = os.path.join(save_dir, "firing_rates_raster_demo.png")
    stats = visualize_firing_rates_trace(
        firing_rates=firing_rates,
        num_exc_segments=num_exc_segments,
        save_path=raster_save_path,
        title="Demo: Firing Rates Raster Plot Style (639 Exc + 640 Inh)",
        max_segments_to_show=200
    )
    
    # 2. 热图可视化
    print("\n2. 生成热图可视化...")
    heatmap_save_path = os.path.join(save_dir, "firing_rates_heatmap_demo.png")
    visualize_firing_rates_heatmap(
        firing_rates=firing_rates,
        num_exc_segments=num_exc_segments,
        save_path=heatmap_save_path,
        title="Demo: Firing Rates Heatmap (639 Exc + 640 Inh)",
        max_segments_to_show=300
    )
    
    print(f"\n演示完成！图片已保存到: {save_dir}")
    return firing_rates

# visualization functions for 5_main_figure_replication.py

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

    safe_save_fig(fig, save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Summary panels saved to: {save_path}')

def plot_voltage_traces(y_spikes_GT, y_spikes_hat, y_voltage_GT, y_voltage_hat, 
                       global_threshold, selected_traces, output_dir, 
                       per_sample_global_metrics=None, per_sample_optimal_metrics=None):
    """Plot voltage traces with dynamic number of figures.
    - Paginate selected_traces into pages of up to 5 traces each.
    - Works for both single-figure and multi-figure cases by treating figures as a list.
    """
    total = len(selected_traces)
    if total == 0:
        print('No selected traces to plot.')
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
            if k < num_subplots - 1:
                plt.setp(ax.get_xticklabels(), visible=False)

        # Set x-axis label on the last subplot of each figure
        axes_norm[-1].set_xlabel('Time (s)', fontsize=12)

        figs.append(fig)
        axes_list.append(axes_norm)

    # Save figures
    if num_pages == 1:
        save_path = f'{output_dir}/voltage_traces.png'
        safe_save_fig(figs[0], save_path, dpi=300, bbox_inches='tight')
        plt.close(figs[0])
        print(f'Voltage traces saved to: {save_path}')
    else:
        for idx, fig in enumerate(figs, start=1):
            save_path = f'{output_dir}/voltage_traces_p{idx}.png'
            safe_save_fig(fig, save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'Voltage traces saved to: {save_path}')

def safe_save_fig(fig, save_path, dpi=300, bbox_inches='tight'):
    """Safely save matplotlib figure"""
    try:
        fig.canvas.draw()
        plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, 
                       facecolor='white', edgecolor='none')
    except Exception:
        try:
            plt.savefig(save_path, dpi=dpi, facecolor='white', edgecolor='none')
        except Exception:
                plt.savefig(save_path, dpi=150)

# 添加优化过程相关的可视化函数 for 6_activity_optimization.py

def visualize_firing_rates_trace(firing_rates, num_exc_segments=640, save_path=None, 
                                 title="Firing Rates Visualization", figsize=(18, 12), 
                                 time_step_ms=1, max_segments_to_show=200, specified_segments=None):
    """
    Visualize firing rate array using raster plot style but displaying continuous curves, separately showing excitatory and inhibitory segments
    
    Args:
        firing_rates: (num_segments_total, time_duration_ms) numpy array
        num_exc_segments: Number of excitatory segments, default 640
        save_path: Path to save the image, if None then display the image
        title: Image title
        figsize: Image size
        time_step_ms: Time step in milliseconds
        max_segments_to_show: Maximum number of segments to display (to avoid overcrowding)
        specified_segments: List of specified segment indices to display, if None use default sampling
    """
    # Data format: (num_segments_total, time_duration_ms)
    num_segments_total, time_duration_ms = firing_rates.shape
    num_inh_segments = num_segments_total - num_exc_segments
    
    print(f"Data shape: {firing_rates.shape}")
    print(f"Total segments: {num_segments_total} (excitatory: {num_exc_segments}, inhibitory: {num_inh_segments})")
    print(f"Time duration: {time_duration_ms} ms")
    
    # Separate excitatory and inhibitory data
    exc_data = firing_rates[:num_exc_segments, :]
    inh_data = firing_rates[num_exc_segments:, :]
    
    # If segments are specified, prioritize using them
    if specified_segments is not None and len(specified_segments) > 0:
        # Separate excitatory and inhibitory segments
        exc_specified = [idx for idx in specified_segments if idx < num_exc_segments]
        inh_specified = [idx for idx in specified_segments if idx >= num_exc_segments]
        
        if len(exc_specified) > 0:
            exc_data_selected = exc_data[exc_specified, :]
            exc_selected_indices = np.array(exc_specified)
            exc_to_show = len(exc_specified)
            print(f"Specified excitatory segments: {exc_specified}")
        else:
            # If no excitatory segments specified, use default sampling
            if num_exc_segments > max_segments_to_show // 2:
                exc_to_show = max_segments_to_show // 2
                exc_indices = np.linspace(0, num_exc_segments-1, exc_to_show, dtype=int)
                exc_data_selected = exc_data[exc_indices, :]
                exc_selected_indices = exc_indices
                print(f"Excitatory segments sampling: {exc_to_show}/{num_exc_segments}")
            else:
                exc_data_selected = exc_data
                exc_selected_indices = np.arange(num_exc_segments)
                exc_to_show = num_exc_segments
        
        if len(inh_specified) > 0:
            inh_data_selected = inh_data[inh_specified, :]
            inh_selected_indices = np.array(inh_specified)
            inh_to_show = len(inh_specified)
            print(f"Specified inhibitory segments: {inh_specified}")
        else:
            # If no inhibitory segments specified, use default sampling
            if num_inh_segments > max_segments_to_show // 2:
                inh_to_show = max_segments_to_show // 2
                inh_indices = np.linspace(0, num_inh_segments-1, inh_to_show, dtype=int)
                inh_data_selected = inh_data[inh_indices, :]
                inh_selected_indices = inh_indices
                print(f"Inhibitory segments sampling: {inh_to_show}/{num_inh_segments}")
            else:
                inh_data_selected = inh_data
                inh_selected_indices = np.arange(num_inh_segments)
                inh_to_show = num_inh_segments
            
    else:
        # If no segments specified, use default sampling
        if num_exc_segments > max_segments_to_show // 2:
            exc_to_show = max_segments_to_show // 2
            exc_indices = np.linspace(0, num_exc_segments-1, exc_to_show, dtype=int)
            exc_data_selected = exc_data[exc_indices, :]
            exc_selected_indices = exc_indices
            print(f"Excitatory segments sampling: {exc_to_show}/{num_exc_segments}")
        else:
            exc_data_selected = exc_data
            exc_selected_indices = np.arange(num_exc_segments)
            exc_to_show = num_exc_segments
        
        if num_inh_segments > max_segments_to_show // 2:
            inh_to_show = max_segments_to_show // 2
            inh_indices = np.linspace(0, num_inh_segments-1, inh_to_show, dtype=int)
            inh_data_selected = inh_data[inh_indices, :]
            inh_selected_indices = inh_indices
            print(f"Inhibitory segments sampling: {inh_to_show}/{num_inh_segments}")
        else:
            inh_data_selected = inh_data
            inh_selected_indices = np.arange(num_inh_segments)
            inh_to_show = num_inh_segments
    
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
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Excitatory Segments', fontsize=12, color='blue')
    ax1.set_title(f'Excitatory Segments (n={exc_to_show}/{num_exc_segments})', fontsize=14, color='blue')
    ax1.tick_params(axis='y', colors='blue')
    
    # Set y-axis ticks and labels - excitatory
    # Add safety check to ensure indices are valid
    if exc_to_show > 0 and len(exc_selected_indices) > 0:
        # When indices count is <= 20, set ticks for each segment
        if exc_to_show <= 20:
            exc_y_tick_positions = list(range(exc_to_show))
            exc_y_tick_labels = [f'Exc {exc_selected_indices[i]}' for i in exc_y_tick_positions]
        else:
            # When indices count > 20, use original sampling method
            exc_y_tick_positions = [0, exc_to_show//4, exc_to_show//2, 3*exc_to_show//4, exc_to_show-1]
            exc_y_tick_labels = [f'Exc {exc_selected_indices[i]}' for i in exc_y_tick_positions if i < len(exc_selected_indices)]
        
        ax1.set_yticks(exc_y_tick_positions[:len(exc_y_tick_labels)])
        ax1.set_yticklabels(exc_y_tick_labels)
    else:
        # If no excitatory segments, set empty ticks
        ax1.set_yticks([])
        ax1.set_yticklabels([])
    
    # Set grid
    ax1.grid(True, alpha=0.3)
    
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
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Inhibitory Segments', fontsize=12, color='red')
    ax2.set_title(f'Inhibitory Segments (n={inh_to_show}/{num_inh_segments})', fontsize=14, color='red')
    ax2.tick_params(axis='y', colors='red')
    
    # Set y-axis ticks and labels - inhibitory
    # Add safety check to ensure indices are valid
    if inh_to_show > 0 and len(inh_selected_indices) > 0:
        # When indices count is <= 20, set ticks for each segment
        if inh_to_show <= 20:
            inh_y_tick_positions = list(range(inh_to_show))
            inh_y_tick_labels = [f'Inh {inh_selected_indices[i]}' for i in inh_y_tick_positions]
        else:
            # When indices count > 20, use original sampling method
            inh_y_tick_positions = [0, inh_to_show//4, inh_to_show//2, 3*inh_to_show//4, inh_to_show-1]
            inh_y_tick_labels = [f'Inh {inh_selected_indices[i]}' for i in inh_y_tick_positions if i < len(inh_selected_indices)]
        
        ax2.set_yticks(inh_y_tick_positions[:len(inh_y_tick_labels)])
        ax2.set_yticklabels(inh_y_tick_labels)
    else:
        # If no inhibitory segments, set empty ticks
        ax2.set_y_ticklabels([])
        ax2.set_yticks([])
    
    # Set grid
    ax2.grid(True, alpha=0.3)
    
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Firing rates visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
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
    
    print(f"\nRaster plot statistics:")
    print(f"  Excitatory segments:")
    print(f"    Display count: {exc_to_show}/{num_exc_segments}")
    print(f"    Average firing rate: {stats['mean_firing_rate_exc']:.4f}")
    print(f"    Value range: [{stats['min_firing_rate_exc']:.4f}, {stats['max_firing_rate_exc']:.4f}]")
    print(f"  Inhibitory segments:")
    print(f"    Display count: {inh_to_show}/{num_inh_segments}")
    print(f"    Average firing rate: {stats['mean_firing_rate_inh']:.4f}")
    print(f"    Value range: [{stats['min_firing_rate_inh']:.4f}, {stats['max_firing_rate_inh']:.4f}]")
    
    return stats

def visualize_firing_rates_heatmap(firing_rates, num_exc_segments=640, save_path=None, 
                                  title="Firing Rates Heatmap", figsize=(18, 10), 
                                  max_segments_to_show=300, specified_segments=None):
    """
    Visualize firing rate array using heatmap style, separately displaying excitatory and inhibitory segments
    
    Args:
        firing_rates: (num_segments_total, time_duration_ms) numpy array
        num_exc_segments: Number of excitatory segments, default 640
        save_path: Path to save the image
        title: Image title
        figsize: Image size
        max_segments_to_show: Maximum number of segments to display
        specified_segments: List of specified segment indices to display, if None use default sampling
    """
    # Data format: (num_segments_total, time_duration_ms)
    num_segments_total, time_duration_ms = firing_rates.shape
    num_inh_segments = num_segments_total - num_exc_segments
    
    print(f"Heatmap data shape: {firing_rates.shape}")
    print(f"Segments: {num_segments_total} (excitatory: {num_exc_segments}, inhibitory: {num_inh_segments})")
    
    # Separate excitatory and inhibitory data
    exc_data = firing_rates[:num_exc_segments, :]
    inh_data = firing_rates[num_exc_segments:, :]
    
    # If segments are specified, prioritize using them
    if specified_segments is not None and len(specified_segments) > 0:
        # Separate excitatory and inhibitory segments
        exc_specified = [idx for idx in specified_segments if idx < num_exc_segments]
        inh_specified = [idx for idx in specified_segments if idx >= num_exc_segments]
        
        if len(exc_specified) > 0:
            exc_data_selected = exc_data[exc_specified, :]
            exc_indices = np.array(exc_specified)
            exc_to_show = len(exc_specified)
            print(f"Specified excitatory segments: {exc_specified}")
        else:
            # If no excitatory segments specified, use default sampling
            if num_exc_segments > max_segments_to_show // 2:
                exc_to_show = max_segments_to_show // 2
                exc_indices = np.linspace(0, num_exc_segments-1, exc_to_show, dtype=int)
                exc_data_selected = exc_data[exc_indices, :]
                print(f"Excitatory segments sampling: {exc_to_show}/{num_exc_segments}")
            else:
                exc_data_selected = exc_data
                exc_indices = np.arange(num_exc_segments)
                exc_to_show = num_exc_segments
        
        if len(inh_specified) > 0:
            inh_data_selected = inh_data[inh_specified, :]
            inh_indices = np.array(inh_specified)
            inh_to_show = len(inh_specified)
            print(f"Specified inhibitory segments: {inh_specified}")
        else:
            # If no inhibitory segments specified, use default sampling
            if num_inh_segments > max_segments_to_show // 2:
                inh_to_show = max_segments_to_show // 2
                inh_indices = np.linspace(0, num_inh_segments-1, inh_to_show, dtype=int)
                inh_data_selected = inh_data[inh_indices, :]
                print(f"Inhibitory segments sampling: {inh_to_show}/{num_inh_segments}")
            else:
                inh_data_selected = inh_data
                inh_indices = np.arange(num_inh_segments)
                inh_to_show = num_inh_segments
    else:
        # If no segments specified, use default sampling
        if num_exc_segments > max_segments_to_show // 2:
            exc_to_show = max_segments_to_show // 2
            exc_indices = np.linspace(0, num_exc_segments-1, exc_to_show, dtype=int)
            exc_data_selected = exc_data[exc_indices, :]
            print(f"Excitatory segments sampling: {exc_to_show}/{num_exc_segments}")
        else:
            exc_data_selected = exc_data
            exc_indices = np.arange(num_exc_segments)
            exc_to_show = num_exc_segments
        
        if num_inh_segments > max_segments_to_show // 2:
            inh_to_show = max_segments_to_show // 2
            inh_indices = np.linspace(0, num_inh_segments-1, inh_to_show, dtype=int)
            inh_data_selected = inh_data[inh_indices, :]
            print(f"Inhibitory segments sampling: {inh_to_show}/{num_inh_segments}")
        else:
            inh_data_selected = inh_data
            inh_to_show = num_inh_segments
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Top plot: excitatory segments heatmap (blue)
    im1 = ax1.imshow(exc_data_selected, aspect='auto', cmap='Blues', origin='lower')
    
    # Add blue colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Excitatory Firing Rate', rotation=270, labelpad=15, color='blue')
    cbar1.ax.tick_params(colors='blue')
    
    # Set top plot labels and title
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Excitatory Segments', fontsize=12, color='blue')
    ax1.set_title(f'Excitatory Segments (n={exc_to_show}/{num_exc_segments})', fontsize=14, color='blue')
    ax1.tick_params(axis='y', colors='blue')
    
    # Set y-axis ticks - excitatory
    # Add safety check to ensure indices are valid
    if exc_to_show > 0:
        # When indices count is <= 20, set ticks for each segment
        if exc_to_show <= 20:
            exc_y_ticks = list(range(exc_to_show))
            exc_y_labels = [f'Exc {exc_indices[i]}' for i in exc_y_ticks]
        else:
            # When indices count > 20, use original sampling method
            exc_y_ticks = [0, exc_to_show//4, exc_to_show//2, 3*exc_to_show//4, exc_to_show-1]
            exc_y_labels = [f'Exc {exc_indices[i]}' for i in exc_y_ticks if i < len(exc_indices)]
        
        ax1.set_yticks(exc_y_ticks[:len(exc_y_labels)])
        ax1.set_yticklabels(exc_y_labels)
    else:
        # If no excitatory segments, set empty ticks
        ax1.set_yticks([])
        ax1.set_yticklabels([])
    
    # Add excitatory statistics
    exc_stats_text = f"Range: [{np.min(exc_data_selected):.4f}, {np.max(exc_data_selected):.4f}]\n"
    exc_stats_text += f"Mean: {np.mean(exc_data_selected):.4f} ± {np.std(exc_data_selected):.4f}"
    ax1.text(0.02, 0.98, exc_stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Bottom plot: inhibitory segments heatmap (red)
    im2 = ax2.imshow(inh_data_selected, aspect='auto', cmap='Reds', origin='lower')
    
    # Add red colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Inhibitory Firing Rate', rotation=270, labelpad=15, color='red')
    cbar2.ax.tick_params(colors='red')
    
    # Set bottom plot labels and title
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Inhibitory Segments', fontsize=12, color='red')
    ax2.set_title(f'Inhibitory Segments (n={inh_to_show}/{num_inh_segments})', fontsize=14, color='red')
    ax2.tick_params(axis='y', colors='red')
    
    # Set y-axis ticks - inhibitory
    # Add safety check to ensure indices are valid
    if inh_to_show > 0:
        # When indices count is <= 20, set ticks for each segment
        if inh_to_show <= 20:
            inh_y_ticks = list(range(inh_to_show))
            inh_y_labels = [f'Inh {inh_indices[i]}' for i in inh_y_ticks]
        else:
            # When indices count > 20, use original sampling method
            inh_y_ticks = [0, inh_to_show//4, inh_to_show//2, 3*inh_to_show//4, inh_to_show-1]
            inh_y_labels = [f'Inh {inh_indices[i]}' for i in inh_y_ticks if i < len(inh_indices)]
        
        ax2.set_yticks(inh_y_ticks[:len(inh_y_labels)])
        ax2.set_yticklabels(inh_y_labels)
    else:
        # If no inhibitory segments, set empty ticks
        ax2.set_yticks([])
        ax2.set_yticklabels([])
    
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
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Firing rates heatmap saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print statistics
    print(f"\nHeatmap statistics:")
    print(f"  Excitatory segments:")
    print(f"    Display count: {exc_to_show}/{num_exc_segments}")
    print(f"    Value range: [{np.min(exc_data_selected):.6f}, {np.max(exc_data_selected):.6f}]")
    print(f"    Average: {np.mean(exc_data_selected):.6f} ± {np.std(exc_data_selected):.6f}")
    print(f"  Inhibitory segments:")
    print(f"    Display count: {inh_to_show}/{num_inh_segments}")
    print(f"    Value range: [{np.min(inh_data_selected):.6f}, {np.max(inh_data_selected):.6f}]")
    print(f"    Average: {np.mean(inh_data_selected):.6f} ± {np.std(inh_data_selected):.6f}")

def visualize_optimized_firing_rates(optimized_firing_rates, fixed_exc_indices, 
                                   num_exc_segments=639, save_dir=None, 
                                   title_prefix="Optimized Firing Rates"):
    """
    可视化优化后的firing rates
    
    Args:
        optimized_firing_rates: 优化后的firing rates
        fixed_exc_indices: 固定的excitatory indices
        num_exc_segments: 兴奋性segments数量
        save_dir: 保存目录
        title_prefix: 标题前缀
    """
    if save_dir is None:
        print("警告: 未指定保存目录，跳过可视化")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 取第一个batch进行可视化
    optimized_sample = optimized_firing_rates[0]  # (num_segments, time_duration)
    
    print("\n生成优化后的firing rates可视化...")
    
    # 指定要可视化的segments：优先显示fixed_exc_indices，如果没有则使用默认采样
    specified_segments = None
    if fixed_exc_indices is not None and len(fixed_exc_indices) > 0:
        # 扩展fixed_exc_indices，包含周围的一些segments以便更好地观察
        extended_indices = []
        for idx in fixed_exc_indices:
            # 为每个fixed index添加前后各2个segments
            start_idx = max(0, idx - 2)
            end_idx = min(optimized_sample.shape[0], idx + 3)
            extended_indices.extend(range(start_idx, end_idx))
        
        # 去重并排序
        extended_indices = sorted(list(set(extended_indices)))
        specified_segments = extended_indices
        
        print(f"指定可视化segments: {specified_segments}")
        print(f"包含fixed_exc_indices: {fixed_exc_indices}")
    
    # Raster plot
    raster_save_path = os.path.join(save_dir, 'optimized_firing_rates_raster.png')
    visualize_firing_rates_trace(
        firing_rates=optimized_sample,
        num_exc_segments=num_exc_segments,
        save_path=raster_save_path,
        title=f"{title_prefix} - Raster Plot",
        max_segments_to_show=10,
        specified_segments=specified_segments
    )
    
    # Heatmap
    heatmap_save_path = os.path.join(save_dir, 'optimized_firing_rates_heatmap.png')
    visualize_firing_rates_heatmap(
        firing_rates=optimized_sample,
        num_exc_segments=num_exc_segments,
        save_path=heatmap_save_path,
        title=f"{title_prefix} - Heatmap",
        max_segments_to_show=10,
        specified_segments=specified_segments
    )
    
    print("优化后的firing rates可视化已保存")
    
    return {
        'raster_path': raster_save_path,
        'heatmap_path': heatmap_save_path
    }

def plot_loss_history(loss_history, title="Activity Optimization Loss History", 
                     figsize=(10, 6), save_path=None, show_plot=False):
    """
    绘制损失历史曲线
    
    Args:
        loss_history: 损失历史列表
        title: Image title
        figsize: Image size
        save_path: 保存路径
        show_plot: 是否显示图片
    """
    plt.figure(figsize=figsize)
    plt.plot(loss_history, linewidth=1.5, color='blue', alpha=0.8)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    if len(loss_history) > 0:
        final_loss = loss_history[-1]
        min_loss = min(loss_history)
        improvement = loss_history[0] - final_loss
        
        stats_text = f"Final Loss: {final_loss:.6f}\n"
        stats_text += f"Min Loss: {min_loss:.6f}\n"
        stats_text += f"Improvement: {improvement:.6f}"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"损失历史图已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()
    
    return save_path

def plot_firing_rates_evolution(firing_rates_history, num_segments_exc, num_segments_inh, 
                               time_duration_ms, input_window_size, title="Firing Rates Evolution",
                               figsize=(15, 10), save_path=None, show_plot=False):
    """
    绘制firing rates随迭代的变化
    
    Args:
        firing_rates_history: firing rates历史列表
        num_segments_exc: 兴奋性segments数量
        num_segments_inh: Number of inhibitory segments
        time_duration_ms: 时间长度
        input_window_size: 输入窗口大小
        title: Image title
        figsize: Image size
        save_path: 保存路径
        show_plot: 是否显示图片
    """
    if not firing_rates_history:
        print("警告: firing_rates_history为空，跳过绘制")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 选择几个时间点和segments进行可视化
    sample_times = [50, input_window_size // 2, 250]  # 50ms, half window size, 250ms
    sample_segments = [0, num_segments_exc//2, num_segments_exc, 
                      num_segments_exc + num_segments_inh//2]
    
    for i, (ax, segment_idx) in enumerate(zip(axes.flat, sample_segments)):
        if i < len(sample_segments):
            for time_idx in sample_times:
                if time_idx < time_duration_ms:
                    values = [fr[0, segment_idx, time_idx] for fr in firing_rates_history]
                    ax.plot(range(0, len(firing_rates_history)*10, 10), values, 
                           label=f'Time {time_idx}ms', linewidth=1.5, alpha=0.8)
            
            segment_type = "Exc" if segment_idx < num_segments_exc else "Inh"
            ax.set_title(f'Segment {segment_idx} ({segment_type})', fontsize=12, fontweight='bold')
            ax.set_xlabel('Iteration', fontsize=10)
            ax.set_ylabel('Firing Rate', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息
            if firing_rates_history:
                final_values = [fr[0, segment_idx, time_idx] for fr in firing_rates_history 
                              for time_idx in sample_times if time_idx < time_duration_ms]
                if final_values:
                    mean_val = np.mean(final_values)
                    ax.text(0.02, 0.98, f'Mean: {mean_val:.4f}', 
                           transform=ax.transAxes, fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Set overall title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Firing rates演化图已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()
    
    return save_path

def plot_optimization_summary(loss_history, firing_rates_history, num_segments_exc, 
                             num_segments_inh, time_duration_ms, input_window_size,
                             title="Optimization Summary", figsize=(20, 12), 
                             save_path=None, show_plot=False):
    """
    绘制优化过程总结图，包含多个子图
    
    Args:
        loss_history: 损失历史
        firing_rates_history: firing rates历史
        num_segments_exc: 兴奋性segments数量
        num_segments_inh: Number of inhibitory segments
        time_duration_ms: 时间长度
        input_window_size: 输入窗口大小
        title: Image title
        figsize: Image size
        save_path: 保存路径
        show_plot: 是否显示图片
    """
    fig = plt.figure(figsize=figsize)
    
    # 创建网格布局
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 损失历史 (左上角，跨越2行)
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.plot(loss_history, linewidth=2, color='blue', alpha=0.8)
    ax1.set_title('Loss History', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # 添加损失统计
    if len(loss_history) > 0:
        final_loss = loss_history[-1]
        min_loss = min(loss_history)
        improvement = loss_history[0] - final_loss
        stats_text = f"Final: {final_loss:.6f}\nMin: {min_loss:.6f}\nImp: {improvement:.6f}"
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 2. Firing rates演化 (右上角，跨越2行)
    if firing_rates_history:
        ax2 = fig.add_subplot(gs[0:2, 1])
        # 选择代表性segments
        sample_segments = [0, num_segments_exc//2, num_segments_exc, 
                         num_segments_exc + num_segments_inh//2]
        sample_times = [input_window_size // 2]  # 只显示关键时间点
        
        for segment_idx in sample_segments:
            if segment_idx < len(firing_rates_history[0][0]):
                values = [fr[0, segment_idx, sample_times[0]] for fr in firing_rates_history 
                         if sample_times[0] < time_duration_ms]
                if values:
                    segment_type = "Exc" if segment_idx < num_segments_exc else "Inh"
                    color = 'blue' if segment_idx < num_segments_exc else 'red'
                    ax2.plot(range(0, len(values)*10, 10), values, 
                            label=f'{segment_type} {segment_idx}', color=color, alpha=0.8)
        
        ax2.set_title('Firing Rates Evolution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Firing Rate')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
    
    # 3. 最终firing rates分布 (右下角)
    if firing_rates_history:
        ax3 = fig.add_subplot(gs[0:2, 2])
        final_rates = firing_rates_history[-1][0]  # 取最后一个batch的第一个样本
        
        # Distribution of excitatory and inhibitory segments
        exc_rates = final_rates[:num_segments_exc, :]
        inh_rates = final_rates[num_segments_inh:, :]
        
        # 计算每个segment的平均firing rate
        exc_means = np.mean(exc_rates, axis=1)
        inh_means = np.mean(inh_rates, axis=1)
        
        # 绘制分布
        ax3.hist(exc_means, bins=30, alpha=0.7, color='blue', label='Excitatory', density=True)
        ax3.hist(inh_means, bins=30, alpha=0.7, color='red', label='Inhibitory', density=True)
        ax3.set_title('Final Firing Rates Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Mean Firing Rate')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. 优化统计信息 (底部，跨越3列)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # 创建统计信息文本
    stats_text = "Optimization Statistics:\n"
    if len(loss_history) > 0:
        stats_text += f"• Total Iterations: {len(loss_history)}\n"
        stats_text += f"• Initial Loss: {loss_history[0]:.6f}\n"
        stats_text += f"• Final Loss: {loss_history[-1]:.6f}\n"
        stats_text += f"• Loss Improvement: {loss_history[0] - loss_history[-1]:.6f}\n"
        stats_text += f"• Convergence: {'Yes' if abs(loss_history[-1] - loss_history[-10]) < 1e-6 else 'Partial'}\n"
    
    if firing_rates_history:
        stats_text += f"• Firing Rates History Points: {len(firing_rates_history)}\n"
        stats_text += f"• Segments: {num_segments_exc} (Exc) + {num_segments_inh} (Inh) = {num_segments_exc + num_segments_inh}\n"
        stats_text += f"• Time Duration: {time_duration_ms} ms\n"
        stats_text += f"• Input Window: {input_window_size} ms"
    
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Set overall title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"优化总结图已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()
    
    return save_path

def create_optimization_report(loss_history, firing_rates_history, 
                              optimized_firing_rates, fixed_exc_indices,
                              num_segments_exc, num_segments_inh, 
                              time_duration_ms, input_window_size,
                              save_dir, report_name="optimization_report"):
    """
    创建完整的优化报告，包含所有可视化内容
    
    Args:
        loss_history: 损失历史
        firing_rates_history: firing rates历史
        optimized_firing_rates: 优化后的firing rates
        fixed_exc_indices: 固定的excitatory indices
        num_segments_exc: 兴奋性segments数量
        num_segments_inh: Number of inhibitory segments
        time_duration_ms: 时间长度
        input_window_size: 输入窗口大小
        save_dir: 保存目录
        report_name: 报告名称
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n开始生成优化报告: {report_name}")
    
    # 1. 损失历史
    plot_loss_history(loss_history, save_path=os.path.join(save_dir, f'loss_history.png'))
    
    # 2. Firing rates演化
    if firing_rates_history:
        plot_firing_rates_evolution(
            firing_rates_history, num_segments_exc, num_segments_inh,
            time_duration_ms, input_window_size,
            save_path=os.path.join(save_dir, f'firing_rates_evolution.png')
        )
    
    # 3. 优化总结
    plot_optimization_summary(
        loss_history, firing_rates_history, num_segments_exc, num_segments_inh,
        time_duration_ms, input_window_size,
        save_path=os.path.join(save_dir, f'summary.png')
    )
    
    # 4. 优化后的firing rates可视化
    visualize_optimized_firing_rates(
        optimized_firing_rates, fixed_exc_indices, num_segments_exc,
        save_dir, title_prefix=f"{report_name.title()} - Optimized"
    )
    
    print(f"优化报告已生成，保存在: {save_dir}")
    
    return save_dir


if __name__ == "__main__":
    # 运行演示
    demo_firing_rates = demo_visualization()
    
    print("\n=== 使用说明 ===")
    print("要可视化你自己的firing rate array，请使用以下代码:")
    print()
    print("# 假设你的firing_rates是 (1279, 310) 的numpy数组")
    print("# First 639 segments are excitatory, last 640 are inhibitory")
    print("from utils.visualization_utils import visualize_firing_rates_trace, visualize_firing_rates_heatmap")
    print()
    print("# 类似raster plot的可视化")
    print("visualize_firing_rates_trace(")
    print("    firing_rates=your_firing_rates,")
    print("    num_exc_segments=639,")
    print("    save_path='your_output_path.png',")
    print("    title='Your Title'")
    print(")")
    print()
    print("# 热图可视化")
    print("visualize_firing_rates_heatmap(")
    print("    firing_rates=your_firing_rates,")
    print("    num_exc_segments=639,")
    print("    save_path='your_heatmap_path.png'")
    print(")") 