import pickle
import glob
import os

def check_params_across_files(directory, param_key):
    """
    检查目录中所有pickle文件的指定参数值
    
    Args:
        directory: 包含pickle文件的目录
        param_key: 要检查的参数键名
    """
    pickle_files = glob.glob(os.path.join(directory, "*.p"))
    
    print(f"检查目录 {directory} 中的 {len(pickle_files)} 个文件...")
    print("-" * 50)
    
    for file_path in sorted(pickle_files):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if 'Params' in data and param_key in data['Params']:
                value = data['Params'][param_key]
                filename = os.path.basename(file_path)
                print(f"{filename}: {param_key} = {value}")
            else:
                filename = os.path.basename(file_path)
                print(f"{filename}: 参数 {param_key} 不存在")
                
        except Exception as e:
            filename = os.path.basename(file_path)
            print(f"{filename}: 读取错误 - {e}")

if __name__ == "__main__":
    # 检查output.pkl
    print("检查 output.pkl:")
    check_params_across_files(".", "NUM_EX_BASAL_EXC")
    
    # 检查分割后的文件
    print("\n检查分割后的文件:")
    check_params_across_files("./full_output_dataset", "NUM_EX_BASAL_EXC") 