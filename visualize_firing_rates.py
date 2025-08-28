import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

def visualize_firing_rates_raster(firing_rates, num_exc_segments=640, save_path=None, 
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
    stats = visualize_firing_rates_raster(
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

if __name__ == "__main__":
    # 运行演示
    demo_firing_rates = demo_visualization()
    
    print("\n=== 使用说明 ===")
    print("要可视化你自己的firing rate array，请使用以下代码:")
    print()
    print("# 假设你的firing_rates是 (1279, 310) 的numpy数组")
    print("# 前639个segments是兴奋性，后640个是抑制性")
    print("from visualize_firing_rates import visualize_firing_rates_raster, visualize_firing_rates_heatmap")
    print()
    print("# 类似raster plot的可视化")
    print("visualize_firing_rates_raster(")
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