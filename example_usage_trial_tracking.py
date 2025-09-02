#!/usr/bin/env python3
"""
使用修改后的OptimizedDatasetPipeline进行trial index跟踪的示例
"""

from dataset_pipeline import OptimizedDatasetPipeline
import os

def main():
    """主函数：演示如何使用修改后的pipeline"""
    
    # 定义路径
    base_path = '/G/results/aim2_sjc'
    project_name = 'funcgroup2_var2_AMPA'
    
    # 定义实验路径（使用较小的数据集进行测试）
    num_trials = 100  # 使用100个trials进行测试
    exp_paths = [f'basal_range0_clus_invivo_NATURAL_{project_name}/1/{i}' for i in range(1, num_trials + 1)]
    
    # 构建目录路径
    output_dir = f'{base_path}/Data/full_output_dataset_{project_name}/'
    train_dir = f'{base_path}/Models_TCN/Single_Neuron_InOut_SJC_{project_name}/data/L5PC_AMPA_train/'
    valid_dir = f'{base_path}/Models_TCN/Single_Neuron_InOut_SJC_{project_name}/data/L5PC_AMPA_valid/'
    test_dir = f'{base_path}/Models_TCN/Single_Neuron_InOut_SJC_{project_name}/data/L5PC_AMPA_test/'

    # 创建pipeline实例
    pipeline = OptimizedDatasetPipeline(
        root_folder_path='/G/results/simulation/',
        output_dir=output_dir,
        train_dir=train_dir,
        valid_dir=valid_dir,
        test_dir=test_dir,
        n_workers=3,
        dt=1/40000,
        spike_threshold=-40
    )
    
    # 运行完整的pipeline
    print("开始运行数据集管道...")
    test_trial_indices = pipeline.run_full_pipeline(
        exp_paths=exp_paths, 
        num_files=num_trials // 10,  # 每10个trials一个文件
        batch_size=10,
        train_ratio=0.7, 
        valid_ratio=0.2, 
        test_ratio=0.1
    )
    
    # 显示结果
    print(f"\n=== 结果总结 ===")
    print(f"Test set包含 {len(test_trial_indices)} 个trials")
    print(f"Test trial indices: {sorted(test_trial_indices)}")
    
    # 验证保存的trial indices文件
    print(f"\n=== 验证保存的文件 ===")
    loaded_indices = pipeline.load_test_trial_indices()
    print(f"从文件加载的trial indices: {loaded_indices}")
    print(f"加载的indices与原始indices是否一致: {set(loaded_indices) == set(test_trial_indices)}")
    
    return test_trial_indices

def demonstrate_trial_mapping():
    """演示如何从test set文件中提取trial mapping信息"""
    
    # 假设pipeline已经运行完成
    base_path = '/G/results/aim2_sjc'
    project_name = 'funcgroup2_var2_AMPA'
    output_dir = f'{base_path}/Data/full_output_dataset_{project_name}/'
    
    pipeline = OptimizedDatasetPipeline(
        root_folder_path='/G/results/simulation/',
        output_dir=output_dir,
        train_dir='',  # 不需要这些目录
        valid_dir='',
        test_dir=''
    )
    
    # 加载test trial indices
    test_indices = pipeline.load_test_trial_indices()
    print(f"Test trial indices: {test_indices}")
    
    # 从test set文件中提取详细的mapping信息
    import glob
    import pickle
    
    test_files = sorted(glob.glob(os.path.join(output_dir, 'L5PC_AMPA_test', '*.p')))
    
    print(f"\n=== Test set文件中的详细mapping信息 ===")
    for test_file in test_files:
        print(f"\n文件: {os.path.basename(test_file)}")
        try:
            with open(test_file, 'rb') as f:
                data = pickle.load(f)
            
            trial_mapping = data.get('TrialMapping', {})
            print(f"  包含的simulation indices: {list(trial_mapping.keys())}")
            
            for sim_idx, mapping in trial_mapping.items():
                print(f"    Simulation {sim_idx} -> Trial {mapping['trial_index']} (路径: {mapping['exp_path']})")
                
        except Exception as e:
            print(f"  读取文件时出错: {e}")

if __name__ == "__main__":
    # 运行主函数
    test_indices = main()
    
    # 演示mapping信息提取
    print(f"\n{'='*50}")
    print("演示trial mapping信息提取")
    print(f"{'='*50}")
    demonstrate_trial_mapping()
