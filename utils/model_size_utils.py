"""
模型规模分析工具
提供快速判断模型大小的函数
支持PyTorch和Keras/TensorFlow模型
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def _count_params_pytorch(model):
    """计算PyTorch模型的参数量"""
    return sum(p.numel() for p in model.parameters())


def _count_params_keras(model):
    """计算Keras/TensorFlow模型的参数量"""
    return model.count_params()


def get_model_size_info(model):
    """
    快速获取模型规模和参数量（简洁版）
    
    Args:
        model: PyTorch模型或Keras/TensorFlow模型
    
    Returns:
        tuple: (总参数量, 模型规模分类, 推荐batch_size范围)
    
    Example:
        from utils.model_size_utils import get_model_size_info
        
        total_params, category, batch_range = get_model_size_info(model)
        print(f"模型: {category}, 参数量: {total_params:,}, 推荐batch_size: {batch_range}")
    """
    # 判断模型类型
    if TORCH_AVAILABLE and isinstance(model, (torch.nn.Module,)):
        total_params = _count_params_pytorch(model)
    elif TF_AVAILABLE and (hasattr(model, 'count_params') or isinstance(model, (tf.keras.Model, keras.Model))):
        total_params = _count_params_keras(model)
    else:
        # 尝试PyTorch方式
        try:
            total_params = sum(p.numel() for p in model.parameters())
        except:
            # 尝试Keras方式
            try:
                total_params = model.count_params()
            except:
                raise ValueError("无法识别模型类型，请确保模型是PyTorch或Keras/TensorFlow模型")
    
    if total_params < 1_000_000:  # < 1M
        return total_params, "小模型", "32-128"
    elif total_params < 10_000_000:  # < 10M
        return total_params, "小-中等模型", "64-256"
    elif total_params < 100_000_000:  # < 100M
        return total_params, "中等模型", "128-512"
    elif total_params < 1_000_000_000:  # < 1B
        return total_params, "大模型", "256-1024"
    else:  # >= 1B
        return total_params, "超大模型", "512-2048+"


def analyze_model_size(model, verbose=True):
    """
    详细分析模型参数量并判断模型规模
    
    Args:
        model: PyTorch模型或Keras/TensorFlow模型
        verbose: 是否打印详细信息
    
    Returns:
        dict: 包含参数量、模型大小、规模分类等信息
    
    Example:
        from utils.model_size_utils import analyze_model_size
        
        info = analyze_model_size(model)
        print(f"模型规模: {info['size_category']}")
        print(f"推荐batch_size: {info['recommended_batch_size']}")
    """
    # 判断模型类型并计算参数量
    if TORCH_AVAILABLE and isinstance(model, (torch.nn.Module,)):
        total_params = _count_params_pytorch(model)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
    elif TF_AVAILABLE and (hasattr(model, 'count_params') or isinstance(model, (tf.keras.Model, keras.Model))):
        total_params = _count_params_keras(model)
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
    else:
        # 尝试PyTorch方式
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            non_trainable_params = total_params - trainable_params
        except:
            # 尝试Keras方式
            try:
                total_params = model.count_params()
                trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
                non_trainable_params = total_params - trainable_params
            except:
                raise ValueError("无法识别模型类型，请确保模型是PyTorch或Keras/TensorFlow模型")
    
    # 计算模型大小（假设float32，每个参数4字节）
    model_size_mb = total_params * 4 / (1024 ** 2)  # MB
    model_size_gb = model_size_mb / 1024  # GB
    
    # 判断模型规模（基于参数量）
    if total_params < 1_000_000:  # < 1M
        size_category = "小模型 (Small)"
        recommended_batch_size = "32-128"
        size_threshold = "< 1M"
    elif total_params < 10_000_000:  # < 10M
        size_category = "小-中等模型 (Small-Medium)"
        recommended_batch_size = "64-256"
        size_threshold = "1M - 10M"
    elif total_params < 100_000_000:  # < 100M
        size_category = "中等模型 (Medium)"
        recommended_batch_size = "128-512"
        size_threshold = "10M - 100M"
    elif total_params < 1_000_000_000:  # < 1B
        size_category = "大模型 (Large)"
        recommended_batch_size = "256-1024"
        size_threshold = "100M - 1B"
    else:  # >= 1B
        size_category = "超大模型 (Very Large)"
        recommended_batch_size = "512-2048+"
        size_threshold = ">= 1B"
    
    result = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'model_size_mb': model_size_mb,
        'model_size_gb': model_size_gb,
        'size_category': size_category,
        'size_threshold': size_threshold,
        'recommended_batch_size': recommended_batch_size
    }
    
    if verbose:
        print("=" * 60)
        print("模型规模分析 (Model Size Analysis)")
        print("=" * 60)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        if non_trainable_params > 0:
            print(f"不可训练参数: {non_trainable_params:,}")
        print(f"模型大小: {model_size_mb:.2f} MB ({model_size_gb:.4f} GB)")
        print(f"模型规模: {size_category}")
        print(f"参数量范围: {size_threshold}")
        print(f"推荐 Batch Size: {recommended_batch_size}")
        print("=" * 60)
    
    return result


def print_model_summary(model):
    """
    打印模型摘要信息（最简洁版）
    
    Args:
        model: PyTorch模型
    
    Example:
        from utils.model_size_utils import print_model_summary
        print_model_summary(model)
    """
    total_params, category, batch_range = get_model_size_info(model)
    print(f"模型规模: {category}")
    print(f"参数量: {total_params:,}")
    print(f"推荐batch_size: {batch_range}")


if __name__ == "__main__":
    # 示例：创建一个简单的模型进行测试
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(10, 64, 3)
            self.conv2 = nn.Conv1d(64, 128, 3)
            self.fc = nn.Linear(128, 1)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.mean(dim=2)
            return self.fc(x)
    
    # 测试简洁版
    model = SimpleModel()
    print("=" * 60)
    print("简洁版输出:")
    print("=" * 60)
    total_params, category, batch_range = get_model_size_info(model)
    print(f"参数量: {total_params:,}")
    print(f"模型规模: {category}")
    print(f"推荐batch_size: {batch_range}")
    
    print("\n" + "=" * 60)
    print("详细版输出:")
    print("=" * 60)
    analyze_model_size(model, verbose=True)

