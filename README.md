# Neuron as Temporal Convolutional Network (TCN)

这个项目实现了将神经元活动建模为时间卷积网络（TCN）的框架，用于预测神经元的spike输出和膜电位。

## 项目结构

- `fit_CNN.py` - 核心TCN模型定义和训练
- `train_and_analyze.py` - 主训练脚本
- `model_analysis.py` - 模型分析和评估
- `model_prediction_visualization.py` - 模型预测可视化
- `main_figure_replication.py` - 主要结果图复制
- `activity_optimization.py` - 活动优化算法
- `codefortest/` - 测试和诊断工具
- `dataset_pipeline.py` - 数据集处理管道

## 主要功能

1. **时间卷积网络模型**：支持不同深度和滤波器数量的TCN架构
2. **神经元数据训练**：使用L5PC神经元模拟数据进行训练
3. **多输出预测**：同时预测spike输出和膜电位
4. **GPU加速**：支持CUDA加速训练
5. **模型分析**：AUC评估、训练曲线分析等

## 环境要求

- Python 3.8+
- TensorFlow/Keras
- NumPy, Pandas
- CUDA支持（可选）

## 使用方法

1. 安装依赖：`pip install -r requirements.txt`
2. 运行训练：`python train_and_analyze.py`
3. 模型分析：`python model_analysis.py`

## 配置远程仓库

要将代码推送到GitHub或其他Git托管服务：

```bash
# 添加远程仓库（替换为你的仓库URL）
git remote add origin https://github.com/你的用户名/仓库名.git

# 推送到远程仓库
git push -u origin master
```

## 许可证

[在此添加许可证信息]

## 贡献

欢迎提交Issue和Pull Request！ 

# CNN模型改进选项说明

本文档说明了如何在`fit_CNN.py`中使用改进的架构和数据增强选项，以便对比不同设计下的训练结果。

## 主要改进内容

### 1. 架构改进 (`use_improved_architecture`)

当设置 `use_improved_architecture=True` 时，模型将包含以下改进：

#### 1.1 正则化改进
- **Dropout层**: 在BatchNormalization后添加Dropout(0.2)，防止过拟合
- **残差连接**: 对于相同通道数的层，添加残差连接以改善梯度流动

#### 1.2 初始化策略改进
- **权重初始化**: 使用HeNormal初始化替代TruncatedNormal
- **偏置调整**: 将spike输出的偏置从-2.0调整为-3.0

#### 1.3 优化器改进
- **优化器**: 使用Adam替代Nadam，参数：lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-7
- **损失权重**: 将soma电压的损失权重从0.006调整为0.01

### 2. 数据增强改进 (`use_improved_data_augmentation`)

当设置 `use_improved_data_augmentation=True` 时，数据生成器将包含以下改进：

#### 2.1 时间增强
- **时间偏移**: 30%概率应用±5个时间步的随机偏移
- **数据滚动**: 使用numpy.roll实现时间窗口的循环移位

#### 2.2 噪声增强
- **高斯噪声**: 20%概率添加标准差为0.01的高斯噪声
- **值域限制**: 确保添加噪声后的值仍在[0,1]范围内

#### 2.3 掩码增强
- **随机掩码**: 10%概率随机掩码5%的输入数据
- **缺失数据模拟**: 模拟真实环境中的数据缺失情况

## 使用方法

### 基本用法

```python
# 1. 创建原有架构模型
model_original = create_temporaly_convolutional_model(
    max_input_window_size, num_segments_exc, num_segments_inh,
    filter_sizes_per_layer, num_filters_per_layer, activation_function_per_layer,
    l2_regularization_per_layer, strides_per_layer, dilation_rates_per_layer,
    initializer_per_layer, 
    use_improved_architecture=False  # 使用原有方案
)

# 2. 创建改进架构模型
model_improved = create_temporaly_convolutional_model(
    max_input_window_size, num_segments_exc, num_segments_inh,
    filter_sizes_per_layer, num_filters_per_layer, activation_function_per_layer,
    l2_regularization_per_layer, strides_per_layer, dilation_rates_per_layer,
    initializer_per_layer, 
    use_improved_architecture=True  # 使用改进方案
)

# 3. 创建原有数据生成器
data_gen_original = SimulationDataGenerator(
    sim_files, 
    use_improved_data_augmentation=False  # 使用原有数据增强
)

# 4. 创建改进数据生成器
data_gen_improved = SimulationDataGenerator(
    sim_files, 
    use_improved_data_augmentation=True  # 使用改进数据增强
)
```

### 对比实验设置

为了进行公平对比，建议按以下方式设置实验：

#### 实验1: 基线模型
```python
use_improved_architecture = False
use_improved_data_augmentation = False
```

#### 实验2: 仅架构改进
```python
use_improved_architecture = True
use_improved_data_augmentation = False
```

#### 实验3: 仅数据增强改进
```python
use_improved_architecture = False
use_improved_data_augmentation = True
```

#### 实验4: 完整改进
```python
use_improved_architecture = True
use_improved_data_augmentation = True
```

## 参数说明

### 架构改进参数
- `use_improved_architecture`: 布尔值，控制是否使用改进的架构
  - `False`: 使用原有架构（默认）
  - `True`: 使用改进架构

### 数据增强参数
- `use_improved_data_augmentation`: 布尔值，控制是否使用改进的数据增强
  - `False`: 使用原有数据增强（默认）
  - `True`: 使用改进数据增强

## 预期效果

### 架构改进的预期效果
1. **更好的泛化能力**: Dropout层减少过拟合
2. **更稳定的训练**: 残差连接改善梯度流动
3. **更好的收敛性**: HeNormal初始化和Adam优化器
4. **更平衡的损失**: 调整的损失权重

### 数据增强的预期效果
1. **更强的鲁棒性**: 时间偏移和噪声增强
2. **更好的泛化**: 模拟真实环境的不确定性
3. **数据效率提升**: 通过增强减少对大量数据的依赖

## 注意事项

1. **计算开销**: 改进方案会增加一定的计算开销，特别是残差连接
2. **超参数调优**: 改进方案可能需要重新调整学习率等超参数
3. **兼容性**: 确保TensorFlow和Keras版本兼容
4. **结果对比**: 建议在相同的数据集和训练条件下进行对比实验

## 运行示例

运行以下命令查看示例用法：

```bash
python fit_CNN.py
```

这将展示如何创建和对比不同配置的模型。

## 版本历史

- v1.0: 添加架构改进选项
- v1.1: 添加数据增强改进选项
- v1.2: 完善文档和示例代码 