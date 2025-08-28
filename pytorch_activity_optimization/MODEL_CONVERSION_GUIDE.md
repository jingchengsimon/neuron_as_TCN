# TensorFlow 到 PyTorch 模型转换指南

## 🎯 概述

本指南介绍如何将 TensorFlow/Keras 格式的 `.h5` 模型转换为 PyTorch 格式的 `.pth` 模型，以便在 PyTorch Activity Optimization 项目中使用。

## 🔧 安装依赖

### 1. 安装 TensorFlow
```bash
# 安装 TensorFlow (CPU 版本)
pip install tensorflow

# 或者安装 GPU 版本
pip install tensorflow-gpu
```

### 2. 确保 PyTorch 已安装
```bash
# 检查 PyTorch 是否已安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
```

## 🚀 使用方法

### 方法 1: 自动转换（推荐）

主程序会自动检测 TensorFlow 模型并尝试转换：

```bash
python main.py
```

程序会：
1. 查找最佳模型（基于验证损失）
2. 检查是否存在对应的 PyTorch 模型
3. 如果不存在，自动查找对应的 `.h5` 文件
4. 执行模型转换
5. 使用转换后的模型进行优化

### 方法 2: 手动转换

使用模型转换器手动转换：

```bash
# 基本用法
python model_converter.py model.h5

# 指定参数文件
python model_converter.py model.h5 model_params.pickle

# 指定输出路径
python model_converter.py model.h5 model_params.pickle converted_model.pth
```

### 方法 3: 在代码中使用

```python
from model_converter import TCNModelConverter

# 创建转换器
converter = TCNModelConverter(
    tf_model_path="model.h5",
    model_params_path="model_params.pickle"
)

# 执行转换
pytorch_model = converter.convert("converted_model.pth")

# 测试转换
converter.test_conversion(pytorch_model)
```

## 📁 文件结构

转换过程会生成以下文件：

```
model.h5                           # 原始 TensorFlow 模型
model_params.pickle               # 模型参数文件
model_converted.pth              # 转换后的 PyTorch 模型
model_converted_info.pickle      # 转换信息和元数据
```

## 🔍 转换过程详解

### 1. 模型分析
- 加载 `.h5` 模型文件
- 提取模型架构信息
- 分析层结构和参数

### 2. 架构重建
- 根据原始模型创建对应的 PyTorch 架构
- 保持输入输出形状一致
- 重建卷积层、激活函数等

### 3. 权重复制
- 尝试直接复制兼容的权重
- 处理维度不匹配的情况
- 对于无法复制的权重使用随机初始化

### 4. 模型测试
- 使用相同输入测试两个模型
- 比较输出差异
- 验证转换的正确性

## ⚠️ 注意事项

### 1. 模型兼容性
- 目前支持标准的 TCN 架构
- 主要支持 Conv1D、BatchNorm、ReLU、Dropout 等层
- 复杂的自定义层可能需要手动调整

### 2. 权重转换
- 完全兼容的权重会直接复制
- 部分兼容的权重会进行维度调整
- 不兼容的权重会使用随机初始化

### 3. 性能差异
- 转换后的模型可能与原模型有轻微的性能差异
- 差异主要来自：
  - 权重初始化方式不同
  - 数值精度差异
  - 框架实现细节差异

## 🐛 常见问题

### 问题 1: TensorFlow 未安装
```
ImportError: TensorFlow未安装，无法加载.h5模型
```
**解决方案**: 安装 TensorFlow
```bash
pip install tensorflow
```

### 问题 2: 模型加载失败
```
RuntimeError: 加载TensorFlow模型失败
```
**可能原因**:
- 模型文件损坏
- 模型使用了不兼容的层
- 内存不足

**解决方案**:
- 检查模型文件完整性
- 使用较小的模型进行测试
- 检查模型架构兼容性

### 问题 3: 权重转换失败
```
⚠ 权重复制失败
```
**可能原因**:
- 模型架构差异较大
- 权重形状不匹配
- 层类型不支持

**解决方案**:
- 检查模型架构信息
- 手动调整模型结构
- 使用随机初始化的权重

### 问题 4: 输出差异较大
```
⚠ 模型输出差异较大，可能需要检查转换逻辑
```
**可能原因**:
- 权重转换不完整
- 激活函数实现差异
- 数值精度问题

**解决方案**:
- 检查权重转换日志
- 验证模型架构匹配
- 考虑重新训练模型

## 📊 转换质量评估

### 1. 输出形状匹配
- 确保输入输出形状一致
- 检查批次维度处理

### 2. 数值差异
- 最大差异 < 0.1 为良好
- 平均差异 < 0.01 为优秀
- 差异过大需要检查转换逻辑

### 3. 功能验证
- 使用相同输入测试
- 比较预测结果
- 验证关键功能

## 🔮 高级功能

### 1. 自定义层支持
可以扩展转换器支持更多层类型：

```python
def _convert_custom_layer(self, layer_info):
    """转换自定义层"""
    if layer_info['type'] == 'CustomLayer':
        # 实现自定义转换逻辑
        pass
```

### 2. 权重优化
- 支持权重量化
- 支持模型剪枝
- 支持混合精度

### 3. 批量转换
```python
def batch_convert(self, model_dir, output_dir):
    """批量转换模型"""
    for model_file in glob.glob(os.path.join(model_dir, "*.h5")):
        self.convert(model_file, output_dir)
```

## 📈 性能优化建议

### 1. 内存管理
- 使用 `weights_only=True` 加载模型
- 分批处理大型模型
- 及时释放不需要的变量

### 2. 并行处理
- 支持多进程转换
- 利用 GPU 加速（如果可用）
- 异步 I/O 操作

### 3. 缓存机制
- 缓存已转换的模型
- 避免重复转换
- 增量更新支持

## 🎉 总结

通过本转换器，你可以：

1. **无缝迁移**: 从 TensorFlow 平滑过渡到 PyTorch
2. **保持功能**: 维持原有模型的预测能力
3. **提升性能**: 利用 PyTorch 的优化特性
4. **简化部署**: 统一模型格式，简化部署流程

转换过程自动化程度高，支持批量处理，是模型迁移的理想解决方案。

