import numpy as np
import os
from utils.visualization_utils import visualize_firing_rates_trace, visualize_firing_rates_heatmap

def load_and_visualize_firing_rates(npy_file_path, num_exc_segments=639, save_dir=None):
    """
    读取npy格式的firing rate array并进行可视化
    
    Args:
        npy_file_path: npy文件路径
        num_exc_segments: 兴奋性segments数量，默认639
        save_dir: 保存图片的目录，如果为None则使用默认目录
    """
    print("=== 加载和可视化 Firing Rate Array ===")
    
    # 检查文件是否存在
    if not os.path.exists(npy_file_path):
        print(f"错误：文件 {npy_file_path} 不存在！")
        return
    
    # 读取npy文件
    print(f"正在读取文件: {npy_file_path}")
    try:
        firing_rates = np.load(npy_file_path)
        print(f"文件读取成功！")
        print(f"数据类型: {firing_rates.dtype}")
        print(f"数据形状: {firing_rates.shape}")
        print(f"数据范围: [{np.min(firing_rates):.6f}, {np.max(firing_rates):.6f}]")
        print(f"数据均值: {np.mean(firing_rates):.6f}")
        print(f"数据标准差: {np.std(firing_rates):.6f}")
        
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 检查数据维度
    if len(firing_rates.shape) != 2:
        print(f"错误：期望2D数组 (segments, time)，但得到 {len(firing_rates.shape)}D 数组")
        return
    
    num_segments, time_duration = firing_rates.shape
    num_inh_segments = num_segments - num_exc_segments
    
    print(f"\n数据分析:")
    print(f"  总segments数: {num_segments}")
    print(f"  兴奋性segments: {num_exc_segments} (前{num_exc_segments}个)")
    print(f"  抑制性segments: {num_inh_segments} (后{num_inh_segments}个)")
    print(f"  时间长度: {time_duration} ms")
    
    # 检查segments数量是否合理
    if num_exc_segments > num_segments:
        print(f"警告：兴奋性segments数量 ({num_exc_segments}) 大于总segments数 ({num_segments})")
        num_exc_segments = num_segments // 2
        print(f"自动调整为: {num_exc_segments}")
    
    # 分析兴奋性和抑制性segments的统计信息
    exc_data = firing_rates[:num_exc_segments, :]
    inh_data = firing_rates[num_exc_segments:, :]
    
    print(f"\n分组统计:")
    print(f"  兴奋性segments:")
    print(f"    均值: {np.mean(exc_data):.6f}")
    print(f"    标准差: {np.std(exc_data):.6f}")
    print(f"    范围: [{np.min(exc_data):.6f}, {np.max(exc_data):.6f}]")
    
    print(f"  抑制性segments:")
    print(f"    均值: {np.mean(inh_data):.6f}")
    print(f"    标准差: {np.std(inh_data):.6f}")
    print(f"    范围: [{np.min(inh_data):.6f}, {np.max(inh_data):.6f}]")
    
    # 设置保存目录
    if save_dir is None:
        save_dir = "./firing_rate_visualization/"
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成文件名（基于原文件名）
    base_name = os.path.splitext(os.path.basename(npy_file_path))[0]
    
    # 1. 生成类似raster plot的可视化
    print(f"\n正在生成类似raster plot的可视化...")
    raster_save_path = os.path.join(save_dir, f"{base_name}_raster_plot.png")
    
    try:
        stats = visualize_firing_rates_trace(
            firing_rates=firing_rates,
            num_exc_segments=num_exc_segments,
            save_path=raster_save_path,
            title=f"Firing Rates Raster Plot - {base_name}",
            max_segments_to_show=10,
            figsize=(18, 12)
        )
        print(f"Raster plot 已保存")
        
    except Exception as e:
        print(f"生成raster plot时出错: {e}")
    
    # 2. 生成热图可视化
    print(f"\n正在生成热图可视化...")
    heatmap_save_path = os.path.join(save_dir, f"{base_name}_heatmap.png")
    
    try:
        visualize_firing_rates_heatmap(
            firing_rates=firing_rates,
            num_exc_segments=num_exc_segments,
            save_path=heatmap_save_path,
            title=f"Firing Rates Heatmap - {base_name}",
            max_segments_to_show=400,
            figsize=(18, 12)
        )
        print(f"Heatmap 已保存")
        
    except Exception as e:
        print(f"生成heatmap时出错: {e}")
    
    # # 3. 生成时间序列的统计图
    # print(f"\n正在生成时间序列统计图...")
    # time_stats_path = os.path.join(save_dir, f"{base_name}_time_statistics.png")
    
    # try:
    #     create_time_statistics_plot(firing_rates, num_exc_segments, time_stats_path, base_name)
    #     print(f"时间序列统计图已保存")
    # except Exception as e:
    #     print(f"生成时间序列统计图时出错: {e}")
    
    # print(f"\n=== 可视化完成 ===")
    # print(f"所有图片已保存到: {save_dir}")
    
    # return firing_rates

def create_time_statistics_plot(firing_rates, num_exc_segments, save_path, base_name):
    """
    创建时间序列统计图，显示兴奋性和抑制性segments随时间的平均firing rate变化
    """
    import matplotlib.pyplot as plt
    
    num_segments, time_duration = firing_rates.shape
    
    # 计算每个时间点的平均firing rate
    exc_data = firing_rates[:num_exc_segments, :]
    inh_data = firing_rates[num_exc_segments:, :]
    
    exc_mean_over_time = np.mean(exc_data, axis=0)  # 每个时间点的兴奋性平均
    inh_mean_over_time = np.mean(inh_data, axis=0)  # 每个时间点的抑制性平均
    
    exc_std_over_time = np.std(exc_data, axis=0)    # 每个时间点的兴奋性标准差
    inh_std_over_time = np.std(inh_data, axis=0)    # 每个时间点的抑制性标准差
    
    # 时间轴
    time_axis = np.arange(time_duration)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # 上图：平均firing rate随时间变化
    ax1.plot(time_axis, exc_mean_over_time, color='blue', linewidth=2, label=f'Excitatory (n={num_exc_segments})', alpha=0.8)
    ax1.fill_between(time_axis, 
                     exc_mean_over_time - exc_std_over_time, 
                     exc_mean_over_time + exc_std_over_time, 
                     color='blue', alpha=0.2)
    
    ax1.plot(time_axis, inh_mean_over_time, color='red', linewidth=2, label=f'Inhibitory (n={num_segments-num_exc_segments})', alpha=0.8)
    ax1.fill_between(time_axis, 
                     inh_mean_over_time - inh_std_over_time, 
                     inh_mean_over_time + inh_std_over_time, 
                     color='red', alpha=0.2)
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Mean Firing Rate')
    ax1.set_title(f'Mean Firing Rate Over Time - {base_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 下图：总体firing rate随时间变化
    total_mean_over_time = np.mean(firing_rates, axis=0)
    total_std_over_time = np.std(firing_rates, axis=0)
    
    ax2.plot(time_axis, total_mean_over_time, color='black', linewidth=2, label=f'All Segments (n={num_segments})')
    ax2.fill_between(time_axis, 
                     total_mean_over_time - total_std_over_time, 
                     total_mean_over_time + total_std_over_time, 
                     color='gray', alpha=0.3)
    
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Mean Firing Rate')
    ax2.set_title('Overall Mean Firing Rate Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f"Data shape: {firing_rates.shape}\n"
    stats_text += f"Exc mean: {np.mean(exc_data):.4f} ± {np.std(exc_data):.4f}\n"
    stats_text += f"Inh mean: {np.mean(inh_data):.4f} ± {np.std(inh_data):.4f}\n"
    stats_text += f"Overall: {np.mean(firing_rates):.4f} ± {np.std(firing_rates):.4f}"
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    主函数：读取init_firing_rate_array.npy并可视化
    """
    # 默认文件路径
    npy_file = "init_firing_rate_array.npy"
    
    # 检查文件是否存在
    if not os.path.exists(npy_file):
        print(f"未找到文件: {npy_file}")
        print("请确保 init_firing_rate_array.npy 文件在当前目录中")
        return
    
    # 读取并可视化
    firing_rates = load_and_visualize_firing_rates(
        npy_file_path=npy_file,
        num_exc_segments=639,  # 前639个是兴奋性
        save_dir="./results/1_firing_rate_visualization/"
    )
    
    if firing_rates is not None:
        print(f"\n成功加载数据，形状: {firing_rates.shape}")
        print("可视化文件已生成在 ./results/1_firing_rate_visualization/ 目录中")

if __name__ == "__main__":
    main() 