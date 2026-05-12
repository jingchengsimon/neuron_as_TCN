import os
import shutil

def clean_epoch_folders(root_folder_path, exp_base, start_epoch=1, end_epoch=50):
    """
    删除每个epoch文件夹中除了load_data需要的文件之外的其他文件
    
    Args:
        root_folder_path: 根文件夹路径
        exp_base: 实验基础路径
        start_epoch: 起始epoch
        end_epoch: 结束epoch
    """
    # load_data方法需要的文件
    required_files = [
        'dend_v_array.npy',
        'soma_v_array.npy', 
        'apic_v_array.npy',
        'simulation_params.json',
        'section_synapse_df.csv'
    ]
    
    cleaned_epochs = []
    skipped_epochs = []
    
    for epoch in range(start_epoch, end_epoch + 1):
        epoch_path = os.path.join(root_folder_path, exp_base, '1', str(epoch))
        
        if not os.path.exists(epoch_path):
            print(f"Epoch {epoch}: 文件夹不存在，跳过")
            skipped_epochs.append(epoch)
            continue
            
        # 检查是否所有必需文件都存在
        missing_required = []
        for file_name in required_files:
            file_path = os.path.join(epoch_path, file_name)
            if not os.path.exists(file_path):
                missing_required.append(file_name)
        
        if missing_required:
            print(f"Epoch {epoch}: 缺少必需文件 {missing_required}，跳过清理")
            skipped_epochs.append(epoch)
            continue
        
        # 获取文件夹中的所有文件
        all_files = []
        for root, dirs, files in os.walk(epoch_path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, epoch_path)
                all_files.append(rel_path)
        
        # 删除不需要的文件
        deleted_files = []
        for file_path in all_files:
            if file_path not in required_files:
                full_path = os.path.join(epoch_path, file_path)
                try:
                    os.remove(full_path)
                    deleted_files.append(file_path)
                except Exception as e:
                    print(f"Epoch {epoch}: 删除文件 {file_path} 失败 - {e}")
        
        if deleted_files:
            print(f"Epoch {epoch}: 已删除 {len(deleted_files)} 个文件")
            print(f"  删除的文件: {deleted_files}")
            cleaned_epochs.append(epoch)
        else:
            print(f"Epoch {epoch}: 无需删除文件")
    
    print(f"\n总结:")
    print(f"检查的epoch范围: {start_epoch} - {end_epoch}")
    print(f"已清理的epoch: {cleaned_epochs}")
    print(f"跳过的epoch: {skipped_epochs}")
    print(f"已清理的epoch数量: {len(cleaned_epochs)}")
    print(f"跳过的epoch数量: {len(skipped_epochs)}")

if __name__ == "__main__":
    root_folder_path = '/G/results/simulation/'
    exp_base = 'basal_range0_clus_invivo_NATURAL_exc1.3'
    
    clean_epoch_folders(root_folder_path, exp_base, 1, 50) 