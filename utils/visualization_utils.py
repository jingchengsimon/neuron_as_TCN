import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_firing_rates_trace(firing_rates, num_exc_segments=640, save_path=None, 
                                 title="Firing Rates Visualization", figsize=(18, 12), 
                                 time_step_ms=1, max_segments_to_show=200, specified_segments=None):
    """
    可视化firing rate array，类似raster plot但显示连续曲线，分别显示兴奋性和抑制性segments
    
    Args:
        firing_rates: (num_segments_total, time_duration_ms) numpy数组
        num_exc_segments: 兴奋性segments的数量，默认640
        save_path: 保存图片的路径，如果为None则显示图片
        title: 图片标题
        figsize: 图片大小
        time_step_ms: 时间步长（毫秒）
        max_segments_to_show: 最大显示的segments数量（避免图片过于拥挤）
        specified_segments: 指定要显示的segments索引列表，如果为None则使用默认采样
    """
    # 数据格式: (num_segments_total, time_duration_ms)
    num_segments_total, time_duration_ms = firing_rates.shape
    num_inh_segments = num_segments_total - num_exc_segments
    
    print(f"数据形状: {firing_rates.shape}")
    print(f"Segments总数: {num_segments_total} (兴奋性: {num_exc_segments}, 抑制性: {num_inh_segments})")
    print(f"时间长度: {time_duration_ms} ms")
    
    # 分离兴奋性和抑制性数据
    exc_data = firing_rates[:num_exc_segments, :]
    inh_data = firing_rates[num_exc_segments:, :]
    
    # 如果指定了segments，优先使用指定的
    if specified_segments is not None and len(specified_segments) > 0:
        # 分离兴奋性和抑制性segments
        exc_specified = [idx for idx in specified_segments if idx < num_exc_segments]
        inh_specified = [idx for idx in specified_segments if idx >= num_exc_segments]
        
        if len(exc_specified) > 0:
            exc_data_selected = exc_data[exc_specified, :]
            exc_selected_indices = np.array(exc_specified)
            exc_to_show = len(exc_specified)
            print(f"指定兴奋性segments: {exc_specified}")
        else:
            # 如果没有指定兴奋性segments，使用默认采样
            if num_exc_segments > max_segments_to_show // 2:
                exc_to_show = max_segments_to_show // 2
                exc_indices = np.linspace(0, num_exc_segments-1, exc_to_show, dtype=int)
                exc_data_selected = exc_data[exc_indices, :]
                exc_selected_indices = exc_indices
                print(f"兴奋性segments采样: {exc_to_show}/{num_exc_segments}")
            else:
                exc_data_selected = exc_data
                exc_selected_indices = np.arange(num_exc_segments)
                exc_to_show = num_exc_segments
        
        if len(inh_specified) > 0:
            inh_data_selected = inh_data[inh_specified, :]
            inh_selected_indices = np.array(inh_specified)
            inh_to_show = len(inh_specified)
            print(f"指定抑制性segments: {inh_specified}")
        else:
            # 如果没有指定抑制性segments，使用默认采样
            if num_inh_segments > max_segments_to_show // 2:
                inh_to_show = max_segments_to_show // 2
                inh_indices = np.linspace(0, num_inh_segments-1, inh_to_show, dtype=int)
                inh_data_selected = inh_data[inh_indices, :]
                inh_selected_indices = inh_indices
                print(f"抑制性segments采样: {inh_to_show}/{num_inh_segments}")
            else:
                inh_data_selected = inh_data
                inh_selected_indices = np.arange(num_inh_segments)
                inh_to_show = num_inh_segments
            
    else:
        # 如果没有指定segments，使用默认采样
        if num_exc_segments > max_segments_to_show // 2:
            exc_to_show = max_segments_to_show // 2
            exc_indices = np.linspace(0, num_exc_segments-1, exc_to_show, dtype=int)
            exc_data_selected = exc_data[exc_indices, :]
            exc_selected_indices = exc_indices
            print(f"兴奋性segments采样: {exc_to_show}/{num_exc_segments}")
        else:
            exc_data_selected = exc_data
            exc_selected_indices = np.arange(num_exc_segments)
            exc_to_show = num_exc_segments
        
        if num_inh_segments > max_segments_to_show // 2:
            inh_to_show = max_segments_to_show // 2
            inh_indices = np.linspace(0, num_inh_segments-1, inh_to_show, dtype=int)
            inh_data_selected = inh_data[inh_indices, :]
            inh_selected_indices = inh_indices
            print(f"抑制性segments采样: {inh_to_show}/{num_inh_segments}")
        else:
            inh_data_selected = inh_data
            inh_selected_indices = np.arange(num_inh_segments)
            inh_to_show = num_inh_segments
    
    # 创建时间轴
    time_axis = np.arange(time_duration_ms) * time_step_ms
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # === 上图：兴奋性segments ===
    # 使用指定的索引作为y轴位置，确保显示正确的segment编号
    y_positions_exc = np.arange(len(exc_selected_indices))
    
    # 绘制每个兴奋性segment的firing rate
    for i in range(exc_data_selected.shape[0]):
        # 获取该segment的firing rate (时间序列)
        segment_firing_rate = exc_data_selected[i, :]
        
        # 将firing rate缩放到合适的显示范围
        scaled_firing_rate = segment_firing_rate * 4  # 缩放因子
        y_base = y_positions_exc[i]
        y_values = y_base + scaled_firing_rate
        
        # 绘制曲线 (蓝色)
        ax1.plot(time_axis, y_values, color='blue', linewidth=0.5, alpha=0.7)
        
        # 绘制基线
        ax1.axhline(y=y_base, color='lightgray', linewidth=0.2, alpha=0.5)
    
    # 设置上图坐标轴
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Excitatory Segments', fontsize=12, color='blue')
    ax1.set_title(f'Excitatory Segments (n={exc_to_show}/{num_exc_segments})', fontsize=14, color='blue')
    ax1.tick_params(axis='y', colors='blue')
    
    # 设置y轴刻度和标签 - 兴奋性
    # 添加安全检查，确保索引有效
    if exc_to_show > 0 and len(exc_selected_indices) > 0:
        # 当indices数量不超过20时，为每个segment设置刻度
        if exc_to_show <= 20:
            exc_y_tick_positions = list(range(exc_to_show))
            exc_y_tick_labels = [f'Exc {exc_selected_indices[i]}' for i in exc_y_tick_positions]
        else:
            # 当indices数量超过20时，使用原来的采样方式
            exc_y_tick_positions = [0, exc_to_show//4, exc_to_show//2, 3*exc_to_show//4, exc_to_show-1]
            exc_y_tick_labels = [f'Exc {exc_selected_indices[i]}' for i in exc_y_tick_positions if i < len(exc_selected_indices)]
        
        ax1.set_yticks(exc_y_tick_positions[:len(exc_y_tick_labels)])
        ax1.set_yticklabels(exc_y_tick_labels)
    else:
        # 如果没有兴奋性segments，设置空的刻度
        ax1.set_yticks([])
        ax1.set_yticklabels([])
    
    # 设置网格
    ax1.grid(True, alpha=0.3)
    
    # 添加兴奋性统计信息
    exc_stats_text = f"Range: [{np.min(exc_data_selected):.4f}, {np.max(exc_data_selected):.4f}]\n"
    exc_stats_text += f"Mean: {np.mean(exc_data_selected):.4f} ± {np.std(exc_data_selected):.4f}"
    ax1.text(0.02, 0.98, exc_stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # === 下图：抑制性segments ===
    # 使用指定的索引作为y轴位置，确保显示正确的segment编号
    y_positions_inh = np.arange(len(inh_selected_indices))
    
    # 绘制每个抑制性segment的firing rate
    for i in range(inh_data_selected.shape[0]):
        # 获取该segment的firing rate (时间序列)
        segment_firing_rate = inh_data_selected[i, :]
        
        # 将firing rate缩放到合适的显示范围
        scaled_firing_rate = segment_firing_rate * 40  # 缩放因子
        y_base = y_positions_inh[i]
        y_values = y_base + scaled_firing_rate
        
        # 绘制曲线 (红色)
        ax2.plot(time_axis, y_values, color='red', linewidth=0.5, alpha=0.7)
        
        # 绘制基线
        ax2.axhline(y=y_base, color='lightgray', linewidth=0.2, alpha=0.5)
    
    # 设置下图坐标轴
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Inhibitory Segments', fontsize=12, color='red')
    ax2.set_title(f'Inhibitory Segments (n={inh_to_show}/{num_inh_segments})', fontsize=14, color='red')
    ax2.tick_params(axis='y', colors='red')
    
    # 设置y轴刻度和标签 - 抑制性
    # 添加安全检查，确保索引有效
    if inh_to_show > 0 and len(inh_selected_indices) > 0:
        # 当indices数量不超过20时，为每个segment设置刻度
        if inh_to_show <= 20:
            inh_y_tick_positions = list(range(inh_to_show))
            inh_y_tick_labels = [f'Inh {inh_selected_indices[i]}' for i in inh_y_tick_positions]
        else:
            # 当indices数量超过20时，使用原来的采样方式
            inh_y_tick_positions = [0, inh_to_show//4, inh_to_show//2, 3*inh_to_show//4, inh_to_show-1]
            inh_y_tick_labels = [f'Inh {inh_selected_indices[i]}' for i in inh_y_tick_positions if i < len(inh_selected_indices)]
        
        ax2.set_yticks(inh_y_tick_positions[:len(inh_y_tick_labels)])
        ax2.set_yticklabels(inh_y_tick_labels)
    else:
        # 如果没有抑制性segments，设置空的刻度
        ax2.set_y_ticklabels([])
        ax2.set_yticks([])
    
    # 设置网格
    ax2.grid(True, alpha=0.3)
    
    # 添加抑制性统计信息
    inh_stats_text = f"Range: [{np.min(inh_data_selected):.4f}, {np.max(inh_data_selected):.4f}]\n"
    inh_stats_text += f"Mean: {np.mean(inh_data_selected):.4f} ± {np.std(inh_data_selected):.4f}"
    ax2.text(0.02, 0.98, inh_stats_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # 设置总标题
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # 为总标题留出空间
    
    # 保存或显示图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Firing rates visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # 返回一些统计信息
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
    
    print(f"\nRaster plot统计信息:")
    print(f"  兴奋性segments:")
    print(f"    显示数量: {exc_to_show}/{num_exc_segments}")
    print(f"    平均firing rate: {stats['mean_firing_rate_exc']:.4f}")
    print(f"    数值范围: [{stats['min_firing_rate_exc']:.4f}, {stats['max_firing_rate_exc']:.4f}]")
    print(f"  抑制性segments:")
    print(f"    显示数量: {inh_to_show}/{num_inh_segments}")
    print(f"    平均firing rate: {stats['mean_firing_rate_inh']:.4f}")
    print(f"    数值范围: [{stats['min_firing_rate_inh']:.4f}, {stats['max_firing_rate_inh']:.4f}]")
    
    return stats

def visualize_firing_rates_heatmap(firing_rates, num_exc_segments=640, save_path=None, 
                                  title="Firing Rates Heatmap", figsize=(18, 10), 
                                  max_segments_to_show=300, specified_segments=None):
    """
    使用热图方式可视化firing rate array，分别显示兴奋性和抑制性segments
    
    Args:
        firing_rates: (num_segments_total, time_duration_ms) numpy数组
        num_exc_segments: 兴奋性segments的数量，默认640
        save_path: 保存图片的路径
        title: 图片标题
        figsize: 图片大小
        max_segments_to_show: 最大显示的segments数量
        specified_segments: 指定要显示的segments索引列表，如果为None则使用默认采样
    """
    # 数据格式: (num_segments_total, time_duration_ms)
    num_segments_total, time_duration_ms = firing_rates.shape
    num_inh_segments = num_segments_total - num_exc_segments
    
    print(f"热图数据形状: {firing_rates.shape}")
    print(f"Segments: {num_segments_total} (兴奋性: {num_exc_segments}, 抑制性: {num_inh_segments})")
    
    # 分离兴奋性和抑制性数据
    exc_data = firing_rates[:num_exc_segments, :]
    inh_data = firing_rates[num_exc_segments:, :]
    
    # 如果指定了segments，优先使用指定的
    if specified_segments is not None and len(specified_segments) > 0:
        # 分离兴奋性和抑制性segments
        exc_specified = [idx for idx in specified_segments if idx < num_exc_segments]
        inh_specified = [idx for idx in specified_segments if idx >= num_exc_segments]
        
        if len(exc_specified) > 0:
            exc_data_selected = exc_data[exc_specified, :]
            exc_indices = np.array(exc_specified)
            exc_to_show = len(exc_specified)
            print(f"指定兴奋性segments: {exc_specified}")
        else:
            # 如果没有指定兴奋性segments，使用默认采样
            if num_exc_segments > max_segments_to_show // 2:
                exc_to_show = max_segments_to_show // 2
                exc_indices = np.linspace(0, num_exc_segments-1, exc_to_show, dtype=int)
                exc_data_selected = exc_data[exc_indices, :]
                print(f"兴奋性segments采样: {exc_to_show}/{num_exc_segments}")
            else:
                exc_data_selected = exc_data
                exc_indices = np.arange(num_exc_segments)
                exc_to_show = num_exc_segments
        
        if len(inh_specified) > 0:
            inh_data_selected = inh_data[inh_specified, :]
            inh_indices = np.array(inh_specified)
            inh_to_show = len(inh_specified)
            print(f"指定抑制性segments: {inh_specified}")
        else:
            # 如果没有指定抑制性segments，使用默认采样
            if num_inh_segments > max_segments_to_show // 2:
                inh_to_show = max_segments_to_show // 2
                inh_indices = np.linspace(0, num_inh_segments-1, inh_to_show, dtype=int)
                inh_data_selected = inh_data[inh_indices, :]
                print(f"抑制性segments采样: {inh_to_show}/{num_inh_segments}")
            else:
                inh_data_selected = inh_data
                inh_indices = np.arange(num_inh_segments)
                inh_to_show = num_inh_segments
    else:
        # 如果没有指定segments，使用默认采样
        if num_exc_segments > max_segments_to_show // 2:
            exc_to_show = max_segments_to_show // 2
            exc_indices = np.linspace(0, num_exc_segments-1, exc_to_show, dtype=int)
            exc_data_selected = exc_data[exc_indices, :]
            print(f"兴奋性segments采样: {exc_to_show}/{num_exc_segments}")
        else:
            exc_data_selected = exc_data
            exc_indices = np.arange(num_exc_segments)
            exc_to_show = num_exc_segments
        
        if num_inh_segments > max_segments_to_show // 2:
            inh_to_show = max_segments_to_show // 2
            inh_indices = np.linspace(0, num_inh_segments-1, inh_to_show, dtype=int)
            inh_data_selected = inh_data[inh_indices, :]
            print(f"抑制性segments采样: {inh_to_show}/{num_inh_segments}")
        else:
            inh_data_selected = inh_data
            inh_to_show = num_inh_segments
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # 上图：兴奋性segments热图（蓝色）
    im1 = ax1.imshow(exc_data_selected, aspect='auto', cmap='Blues', origin='lower')
    
    # 添加蓝色颜色条
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Excitatory Firing Rate', rotation=270, labelpad=15, color='blue')
    cbar1.ax.tick_params(colors='blue')
    
    # 设置上图标签和标题
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Excitatory Segments', fontsize=12, color='blue')
    ax1.set_title(f'Excitatory Segments (n={exc_to_show}/{num_exc_segments})', fontsize=14, color='blue')
    ax1.tick_params(axis='y', colors='blue')
    
    # 设置y轴刻度 - 兴奋性
    # 添加安全检查，确保索引有效
    if exc_to_show > 0:
        # 当indices数量不超过20时，为每个segment设置刻度
        if exc_to_show <= 20:
            exc_y_ticks = list(range(exc_to_show))
            exc_y_labels = [f'Exc {exc_indices[i]}' for i in exc_y_ticks]
        else:
            # 当indices数量超过20时，使用原来的采样方式
            exc_y_ticks = [0, exc_to_show//4, exc_to_show//2, 3*exc_to_show//4, exc_to_show-1]
            exc_y_labels = [f'Exc {exc_indices[i]}' for i in exc_y_ticks if i < len(exc_indices)]
        
        ax1.set_yticks(exc_y_ticks[:len(exc_y_labels)])
        ax1.set_yticklabels(exc_y_labels)
    else:
        # 如果没有兴奋性segments，设置空的刻度
        ax1.set_yticks([])
        ax1.set_yticklabels([])
    
    # 添加兴奋性统计信息
    exc_stats_text = f"Range: [{np.min(exc_data_selected):.4f}, {np.max(exc_data_selected):.4f}]\n"
    exc_stats_text += f"Mean: {np.mean(exc_data_selected):.4f} ± {np.std(exc_data_selected):.4f}"
    ax1.text(0.02, 0.98, exc_stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 下图：抑制性segments热图（红色）
    im2 = ax2.imshow(inh_data_selected, aspect='auto', cmap='Reds', origin='lower')
    
    # 添加红色颜色条
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Inhibitory Firing Rate', rotation=270, labelpad=15, color='red')
    cbar2.ax.tick_params(colors='red')
    
    # 设置下图标签和标题
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Inhibitory Segments', fontsize=12, color='red')
    ax2.set_title(f'Inhibitory Segments (n={inh_to_show}/{num_inh_segments})', fontsize=14, color='red')
    ax2.tick_params(axis='y', colors='red')
    
    # 设置y轴刻度 - 抑制性
    # 添加安全检查，确保索引有效
    if inh_to_show > 0:
        # 当indices数量不超过20时，为每个segment设置刻度
        if inh_to_show <= 20:
            inh_y_ticks = list(range(inh_to_show))
            inh_y_labels = [f'Inh {inh_indices[i]}' for i in inh_y_ticks]
        else:
            # 当indices数量超过20时，使用原来的采样方式
            inh_y_ticks = [0, inh_to_show//4, inh_to_show//2, 3*inh_to_show//4, inh_to_show-1]
            inh_y_labels = [f'Inh {inh_indices[i]}' for i in inh_y_ticks if i < len(inh_indices)]
        
        ax2.set_yticks(inh_y_ticks[:len(inh_y_labels)])
        ax2.set_yticklabels(inh_y_labels)
    else:
        # 如果没有抑制性segments，设置空的刻度
        ax2.set_yticks([])
        ax2.set_yticklabels([])
    
    # 添加抑制性统计信息
    inh_stats_text = f"Range: [{np.min(inh_data_selected):.4f}, {np.max(inh_data_selected):.4f}]\n"
    inh_stats_text += f"Mean: {np.mean(inh_data_selected):.4f} ± {np.std(inh_data_selected):.4f}"
    ax2.text(0.02, 0.98, inh_stats_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # 设置总标题
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 调整子图间距
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # 为总标题留出空间
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Firing rates heatmap saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # 打印统计信息
    print(f"\n热图统计信息:")
    print(f"  兴奋性segments:")
    print(f"    显示数量: {exc_to_show}/{num_exc_segments}")
    print(f"    数值范围: [{np.min(exc_data_selected):.6f}, {np.max(exc_data_selected):.6f}]")
    print(f"    平均值: {np.mean(exc_data_selected):.6f} ± {np.std(exc_data_selected):.6f}")
    print(f"  抑制性segments:")
    print(f"    显示数量: {inh_to_show}/{num_inh_segments}")
    print(f"    数值范围: [{np.min(inh_data_selected):.6f}, {np.max(inh_data_selected):.6f}]")
    print(f"    平均值: {np.mean(inh_data_selected):.6f} ± {np.std(inh_data_selected):.6f}")

def demo_visualization():
    """
    演示如何使用可视化函数
    """
    print("=== Firing Rates 可视化演示 ===")
    
    # 创建示例数据，格式为 (num_segments, time_duration)
    num_exc_segments = 639  # 修正为639
    num_inh_segments = 640  
    num_total_segments = num_exc_segments + num_inh_segments  # 1279
    time_duration = 310
    
    print(f"生成示例数据: {num_total_segments} segments, {time_duration} ms")
    print(f"数据格式: (segments, time) = ({num_total_segments}, {time_duration})")
    
    # 生成模拟的firing rates
    np.random.seed(42)  # 为了可重复性
    
    # 兴奋性segments: 较低的基础firing rate，在某些时间点有峰值
    exc_firing_rates = np.random.uniform(0.005, 0.02, (num_exc_segments, time_duration))
    
    # 在150ms附近添加一些峰值
    peak_time = 150
    peak_width = 20
    for i in range(0, num_exc_segments, 50):  # 每50个segments添加一个峰值
        start_time = max(0, peak_time - peak_width//2)
        end_time = min(time_duration, peak_time + peak_width//2)
        exc_firing_rates[i:min(i+10, num_exc_segments), start_time:end_time] += 0.03
    
    # 抑制性segments: 稍高的基础firing rate
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

# 添加优化过程相关的可视化函数

def plot_loss_history(loss_history, title="Activity Optimization Loss History", 
                     figsize=(10, 6), save_path=None, show_plot=False):
    """
    绘制损失历史曲线
    
    Args:
        loss_history: 损失历史列表
        title: 图片标题
        figsize: 图片大小
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
        num_segments_inh: 抑制性segments数量
        time_duration_ms: 时间长度
        input_window_size: 输入窗口大小
        title: 图片标题
        figsize: 图片大小
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
    
    # 设置总标题
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
        num_segments_inh: 抑制性segments数量
        time_duration_ms: 时间长度
        input_window_size: 输入窗口大小
        title: 图片标题
        figsize: 图片大小
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
        
        # 兴奋性和抑制性segments的分布
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
    
    # 设置总标题
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
        num_segments_inh: 抑制性segments数量
        time_duration_ms: 时间长度
        input_window_size: 输入窗口大小
        save_dir: 保存目录
        report_name: 报告名称
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n开始生成优化报告: {report_name}")
    
    # 1. 损失历史
    plot_loss_history(loss_history, save_path=os.path.join(save_dir, f'{report_name}_loss_history.png'))
    
    # 2. Firing rates演化
    if firing_rates_history:
        plot_firing_rates_evolution(
            firing_rates_history, num_segments_exc, num_segments_inh,
            time_duration_ms, input_window_size,
            save_path=os.path.join(save_dir, f'{report_name}_firing_rates_evolution.png')
        )
    
    # 3. 优化总结
    plot_optimization_summary(
        loss_history, firing_rates_history, num_segments_exc, num_segments_inh,
        time_duration_ms, input_window_size,
        save_path=os.path.join(save_dir, f'{report_name}_summary.png')
    )
    
    # 4. 优化后的firing rates可视化
    visualize_optimized_firing_rates(
        optimized_firing_rates, fixed_exc_indices, num_segments_exc,
        save_dir, title_prefix=f"{report_name.title()} - Optimized"
    )
    
    print(f"优化报告已生成，保存在: {save_dir}")
    
    return save_dir

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

if __name__ == "__main__":
    # 运行演示
    demo_firing_rates = demo_visualization()
    
    print("\n=== 使用说明 ===")
    print("要可视化你自己的firing rate array，请使用以下代码:")
    print()
    print("# 假设你的firing_rates是 (1279, 310) 的numpy数组")
    print("# 前639个segments是兴奋性，后640个是抑制性")
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