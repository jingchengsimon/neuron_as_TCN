#!/usr/bin/env python3
"""
TensorFlow到PyTorch模型转换器
将.h5格式的Keras模型转换为PyTorch格式
"""

import os
import sys
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("警告: TensorFlow未安装，无法进行模型转换")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("错误: PyTorch未安装，无法进行模型转换")

class TCNModelConverter:
    """
    TCN模型转换器：从TensorFlow/Keras到PyTorch
    """
    
    def __init__(self, tf_model_path: str, model_params_path: str = None):
        """
        初始化转换器
        
        Args:
            tf_model_path: TensorFlow模型文件路径(.h5)
            model_params_path: 模型参数文件路径(.pickle)
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow未安装，无法加载.h5模型")
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch未安装，无法创建PyTorch模型")
        
        self.tf_model_path = tf_model_path
        self.model_params_path = model_params_path
        self.tf_model = None
        self.model_params = None
        self.architecture_dict = None
        
        # 加载TensorFlow模型和参数
        self._load_tf_model()
        self._load_model_params()
    
    def _load_tf_model(self):
        """加载TensorFlow模型"""
        try:
            print(f"正在加载TensorFlow模型: {self.tf_model_path}")
            self.tf_model = keras.models.load_model(self.tf_model_path, compile=False)
            print(f"✓ TensorFlow模型加载成功")
            print(f"  模型类型: {type(self.tf_model)}")
            print(f"  输入形状: {self.tf_model.input_shape}")
            print(f"  输出形状: {self.tf_model.output_shape}")
            
            # 打印模型结构
            print("\n模型结构:")
            self.tf_model.summary()
            
        except Exception as e:
            raise RuntimeError(f"加载TensorFlow模型失败: {e}")
    
    def _load_model_params(self):
        """加载模型参数"""
        if self.model_params_path and os.path.exists(self.model_params_path):
            try:
                with open(self.model_params_path, 'rb') as f:
                    self.model_params = pickle.load(f)
                
                if 'architecture_dict' in self.model_params:
                    self.architecture_dict = self.model_params['architecture_dict']
                    print(f"✓ 模型参数加载成功")
                    print(f"  架构信息: {self.architecture_dict}")
                else:
                    print("警告: 模型参数中未找到架构信息")
                    
            except Exception as e:
                print(f"警告: 加载模型参数失败: {e}")
        else:
            print("警告: 模型参数文件不存在，将使用默认架构")
    
    def _extract_layer_info(self):
        """提取模型层信息"""
        layer_info = []
        
        for i, layer in enumerate(self.tf_model.layers):
            layer_type = type(layer).__name__
            config = layer.get_config()
            
            info = {
                'index': i,
                'name': layer.name,
                'type': layer_type,
                'config': config,
                'input_shape': layer.input_shape,
                'output_shape': layer.output_shape
            }
            
            # 提取特定层的参数
            if hasattr(layer, 'filters'):
                info['filters'] = layer.filters
            if hasattr(layer, 'kernel_size'):
                info['kernel_size'] = layer.kernel_size
            if hasattr(layer, 'strides'):
                info['strides'] = layer.strides
            if hasattr(layer, 'padding'):
                info['padding'] = layer.padding
            if hasattr(layer, 'activation'):
                info['activation'] = str(layer.activation)
            
            layer_info.append(info)
        
        return layer_info
    
    def _create_pytorch_tcn(self, layer_info):
        """根据层信息创建PyTorch TCN模型"""
        print("\n正在创建PyTorch TCN模型...")
        
        # 分析输入形状
        input_shape = self.tf_model.input_shape
        if len(input_shape) == 3:  # (batch, time, features)
            batch_size, time_steps, num_features = input_shape
        else:
            raise ValueError(f"不支持的输入形状: {input_shape}")
        
        # 分析输出形状
        output_shape = self.tf_model.output_shape
        if len(output_shape) == 3:  # (batch, time, output_features)
            _, _, output_features = output_shape
        else:
            output_features = 1  # 默认输出特征数
        
        print(f"  输入形状: {input_shape}")
        print(f"  输出形状: {output_shape}")
        print(f"  时间步数: {time_steps}")
        print(f"  输入特征: {num_features}")
        print(f"  输出特征: {output_features}")
        
        # 创建PyTorch TCN模型
        pytorch_model = self._create_tcn_model(input_size, hidden_sizes, output_features)
        
        print("✓ PyTorch TCN模型创建成功")
        return pytorch_model
        
        # 根据原始模型结构确定隐藏层大小
        # 分析卷积层
        conv_layers = [layer for layer in layer_info if 'Conv' in layer['type']]
        
        if conv_layers:
            # 使用原始模型的卷积层配置
            hidden_sizes = []
            for layer in conv_layers[:-1]:  # 除了最后一层
                if 'filters' in layer:
                    hidden_sizes.append(layer['filters'])
            
            if not hidden_sizes:
                # 如果没有找到filters信息，使用默认配置
                hidden_sizes = [256, 128, 64]
        else:
            # 使用默认配置
            hidden_sizes = [256, 128, 64]
        
        print(f"  隐藏层大小: {hidden_sizes}")
        
        # 创建模型
        pytorch_model = self._create_tcn_model(input_size, hidden_sizes, output_features)
        
        print("✓ PyTorch TCN模型创建成功")
        return pytorch_model
    
    def _copy_weights(self, pytorch_model):
        """复制权重（如果可能）"""
        print("\n尝试复制权重...")
        
        try:
            # 获取TensorFlow模型的权重
            tf_weights = self.tf_model.get_weights()
            
            # 获取PyTorch模型的权重
            pytorch_state_dict = pytorch_model.state_dict()
            
            print(f"  TensorFlow权重数量: {len(tf_weights)}")
            print(f"  PyTorch参数数量: {len(pytorch_state_dict)}")
            
            # 尝试匹配和复制权重
            copied_count = 0
            for i, (name, param) in enumerate(pytorch_state_dict.items()):
                if i < len(tf_weights):
                    tf_weight = tf_weights[i]
                    
                    # 检查形状是否兼容
                    if tf_weight.shape == param.shape:
                        # 直接复制
                        param.data = torch.from_numpy(tf_weight).float()
                        copied_count += 1
                        print(f"    ✓ 复制权重: {name} {tf_weight.shape}")
                    elif len(tf_weight.shape) == 2 and len(param.shape) == 3:
                        # 2D -> 3D 转换
                        if tf_weight.shape[1] == param.shape[1]:
                            # 扩展维度
                            expanded_weight = tf_weight[np.newaxis, :, :]
                            param.data = torch.from_numpy(expanded_weight).float()
                            copied_count += 1
                            print(f"    ✓ 复制权重(维度转换): {name} {tf_weight.shape} -> {param.shape}")
                        else:
                            print(f"    ⚠ 形状不匹配: {name} {tf_weight.shape} vs {param.shape}")
                    else:
                        print(f"    ⚠ 形状不匹配: {name} {tf_weight.shape} vs {param.shape}")
                else:
                    print(f"    ⚠ 权重数量不匹配: {name}")
            
            print(f"  成功复制 {copied_count}/{len(pytorch_state_dict)} 个参数")
            
            if copied_count > 0:
                print("✓ 部分权重复制成功")
            else:
                print("⚠ 无法复制权重，将使用随机初始化")
                
        except Exception as e:
            print(f"⚠ 权重复制失败: {e}")
            print("  将使用随机初始化的权重")
    
    def convert(self, output_path: str = None):
        """
        执行模型转换
        
        Args:
            output_path: 输出PyTorch模型路径
            
        Returns:
            pytorch_model: 转换后的PyTorch模型
        """
        print("=" * 60)
        print("开始TensorFlow到PyTorch模型转换")
        print("=" * 60)
        
        # 提取层信息
        layer_info = self._extract_layer_info()
        print(f"\n提取到 {len(layer_info)} 个层的信息")
        
        # 创建PyTorch模型
        pytorch_model = self._create_pytorch_tcn(layer_info)
        
        # 尝试复制权重
        self._copy_weights(pytorch_model)
        
        # 保存模型
        if output_path is None:
            # 生成默认输出路径
            tf_model_name = Path(self.tf_model_path).stem
            output_path = f"{tf_model_name}_converted.pth"
        
        try:
            # 保存PyTorch模型
            torch.save(pytorch_model, output_path)
            print(f"\n✓ PyTorch模型已保存到: {output_path}")
            
            # 保存模型信息
            info_path = output_path.replace('.pth', '_info.pickle')
            model_info = {
                'original_tf_model': self.tf_model_path,
                'architecture_dict': self.architecture_dict,
                'input_shape': self.tf_model.input_shape,
                'output_shape': self.tf_model.output_shape,
                'conversion_info': {
                    'converter_version': '1.0',
                    'conversion_date': str(np.datetime64('now')),
                    'layer_info': layer_info
                }
            }
            
            with open(info_path, 'wb') as f:
                pickle.dump(model_info, f)
            print(f"✓ 模型信息已保存到: {info_path}")
            
        except Exception as e:
            print(f"⚠ 保存模型失败: {e}")
        
        print("\n" + "=" * 60)
        print("模型转换完成！")
        print("=" * 60)
        
        return pytorch_model
    
    def _create_tcn_model(self, input_size, hidden_sizes, output_size):
        """
        创建PyTorch TCN模型
        
        Args:
            input_size: 输入特征数
            hidden_sizes: 隐藏层大小列表
            output_size: 输出特征数
            
        Returns:
            pytorch_model: PyTorch TCN模型
        """
        import torch
        import torch.nn as nn
        
        class PyTorchTCN(nn.Module):
            def __init__(self, input_size, hidden_sizes, output_size, kernel_size=3, dropout=0.1):
                super(PyTorchTCN, self).__init__()
                
                self.input_size = input_size
                self.hidden_sizes = hidden_sizes
                self.output_size = output_size
                self.kernel_size = kernel_size
                self.dropout = dropout
                
                # 构建卷积层
                layers = []
                in_channels = input_size
                
                for hidden_size in hidden_sizes:
                    layers.extend([
                        nn.Conv1d(in_channels, hidden_size, kernel_size, padding=kernel_size//2),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
                    in_channels = hidden_size
                
                # 输出层
                layers.append(nn.Conv1d(in_channels, output_size, kernel_size, padding=kernel_size//2))
                layers.append(nn.Sigmoid())
                
                self.conv_layers = nn.Sequential(*layers)
                
            def forward(self, x):
                # 输入: (batch, time, features) -> (batch, features, time)
                x = x.transpose(1, 2)
                
                # 通过卷积层
                x = self.conv_layers(x)
                
                # 输出: (batch, features, time) -> (batch, time, features)
                x = x.transpose(1, 2)
                
                return x
        
        # 创建模型实例
        pytorch_model = PyTorchTCN(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            kernel_size=3,
            dropout=0.1
        )
        
        return pytorch_model
    
    def test_conversion(self, pytorch_model, test_input=None):
        """测试转换后的模型"""
        print("\n测试转换后的模型...")
        
        try:
            # 创建测试输入
            if test_input is None:
                input_shape = self.tf_model.input_shape
                if len(input_shape) == 3:
                    batch_size, time_steps, features = input_shape
                    test_input = np.random.random((1, time_steps, features)).astype(np.float32)
                else:
                    raise ValueError(f"不支持的输入形状: {input_shape}")
            
            print(f"  测试输入形状: {test_input.shape}")
            
            # TensorFlow模型预测
            with tf.device('/CPU:0'):  # 强制使用CPU避免GPU问题
                tf_output = self.tf_model.predict(test_input, verbose=0)
            
            # PyTorch模型预测
            pytorch_model.eval()
            with torch.no_grad():
                test_input_torch = torch.from_numpy(test_input).float()
                torch_output = pytorch_model(test_input_torch)
                torch_output_np = torch_output.numpy()
            
            print(f"  TensorFlow输出形状: {tf_output.shape}")
            print(f"  PyTorch输出形状: {torch_output_np.shape}")
            
            # 比较输出
            if tf_output.shape == torch_output_np.shape:
                # 计算差异
                diff = np.abs(tf_output - torch_output_np)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                print(f"  最大差异: {max_diff:.6f}")
                print(f"  平均差异: {mean_diff:.6f}")
                
                if max_diff < 0.1:  # 允许一定的差异
                    print("✓ 模型转换测试通过")
                else:
                    print("⚠ 模型输出差异较大，可能需要检查转换逻辑")
            else:
                print("⚠ 输出形状不匹配")
                
        except Exception as e:
            print(f"⚠ 模型测试失败: {e}")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python model_converter.py <tf_model_path> [model_params_path] [output_path]")
        print("示例: python model_converter.py model.h5 model_params.pickle converted_model.pth")
        return
    
    tf_model_path = sys.argv[1]
    model_params_path = sys.argv[2] if len(sys.argv) > 2 else None
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        # 创建转换器
        converter = TCNModelConverter(tf_model_path, model_params_path)
        
        # 执行转换
        pytorch_model = converter.convert(output_path)
        
        # 测试转换
        converter.test_conversion(pytorch_model)
        
    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
