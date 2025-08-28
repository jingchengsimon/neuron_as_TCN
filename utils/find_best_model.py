import glob
import os
import pickle

def find_best_model(models_dir):
    """
    找到最佳模型（基于验证损失）
    
    Args:
        models_dir: 模型目录
        
    Returns:
        best_model_path: 最佳模型的.h5文件路径
        best_params_path: 对应的.pickle文件路径
    """
    pickle_files = glob.glob(os.path.join(models_dir, '*.pickle'))
    
    if not pickle_files:
        raise ValueError(f"在{models_dir}中未找到模型文件")
    
    best_val_loss = float('inf')
    best_model_path = None
    best_params_path = None
    
    for pickle_path in pickle_files:
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            # 获取最小验证损失
            val_losses = data['training_history_dict']['val_spikes_loss']
            min_val_loss = min(val_losses)
            
            if min_val_loss < best_val_loss:
                best_val_loss = min_val_loss
                best_params_path = pickle_path
                best_model_path = pickle_path.replace('.pickle', '.h5')
        
        except Exception as e:
            print(f"读取{pickle_path}时出错: {e}")
    
    if best_model_path is None:
        raise ValueError("未找到有效的模型文件")
    
    print(f"找到最佳模型: {best_model_path}")
    
    return best_model_path, best_params_path
