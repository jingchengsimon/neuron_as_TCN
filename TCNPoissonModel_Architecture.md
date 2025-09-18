# TCNPoissonModel 架构设计

## 概述

`TCNPoissonModel`是一个专门为activity optimization设计的PyTorch模块，将TCN模型与可微分的Poisson spike生成结合在一起。

## 架构设计

### 核心组件

```
TCNPoissonModel (nn.Module)
├── tcn_model (预训练的TCN)
├── forward() (训练时前向传播，包含STE Poisson采样)
└── predict_eval() (评估时预测)
```

### 数据流

1. **训练模式** (`forward`):
   ```
   firing_rates → STE Poisson → TCN → spike_predictions
        ↑                                      ↓
        └────────── gradients ←─────────────────┘
   ```

2. **评估模式** (`predict_eval`):
   ```
   firing_rates → numpy Poisson → TCN (no_grad) → spike_predictions
   ```

## 关键特性

### 1. Straight-Through Estimator (STE)

```python
# 前向传播：真实Poisson采样
with torch.no_grad():
    poisson_samples = torch.poisson(safe_rates)
    poisson_binary = torch.clamp(poisson_samples, 0, 1)

# 反向传播：身份梯度
first_half_spikes = poisson_binary + safe_rates - safe_rates.detach()
```

**优势**：
- 保持Poisson过程的统计特性
- 允许梯度传播
- 数值稳定

### 2. 双重spike生成策略

- **训练时**：使用STE确保可微分性
- **评估时**：使用真实numpy Poisson确保统计准确性

### 3. 模块化设计

- **独立性**：可以单独使用，不依赖其他处理器
- **PyTorch原生**：继承`nn.Module`，支持标准PyTorch工作流
- **GPU兼容**：自动处理设备管理

## 使用方式

### 基本用法

```python
from utils.tcn_poisson_model import TCNPoissonModel

# 创建模型
tcn_model = TCNPoissonModel(
    model_path="path/to/model.pt",
    model_params=model_params_dict,
    input_window_size=400,
    num_segments_exc=639,
    num_segments_inh=640,
    time_duration_ms=400
)

# 训练模式（支持梯度）
firing_rates = torch.rand(2, 1279, 400, requires_grad=True)
spike_preds, spike_trains = tcn_model(firing_rates, fixed_exc_indices)

# 评估模式（无梯度，更准确）
eval_preds, eval_trains = tcn_model.predict_eval(firing_rates, fixed_exc_indices)
```

### 在优化中使用

```python
# 在activity optimization中
class ActivityOptimizer:
    def __init__(self, ...):
        self.tcn_poisson_model = TCNPoissonModel(...)
    
    def optimize_activity(self, ...):
        for epoch in range(num_iterations):
            optimizer.zero_grad()
            
            # 使用TCN Poisson模型
            spike_preds, _ = self.tcn_poisson_model(firing_rates, fixed_exc_indices)
            
            loss = self.compute_loss(firing_rates, spike_preds, ...)
            loss.backward()
            optimizer.step()
```

## 重构后的模块分工

### FiringRatesProcessor (轻量化)
- **专注于**：数据处理和预处理
- **功能**：
  - 加载初始firing rates
  - 生成背景firing rates  
  - 数据格式转换和预处理
  - 为优化准备数据
- **不包含**：模型加载、spike生成、预测计算

### TCNPoissonModel (核心计算)
- **专注于**：模型预测和spike生成
- **功能**：
  - TCN模型管理
  - STE Poisson spike生成
  - 梯度使能的前向传播
  - 评估模式预测

## 与原有架构对比

| 方面 | 原有设计 | 新设计 |
|------|----------|-------|
| 模块化 | 功能混合在FiringRatesProcessor | 清晰的职责分离 |
| 可维护性 | 单个类承担过多职责 | 单一职责原则 |
| 可测试性 | 难以单独测试模型部分 | 独立可测试 |
| 代码复用 | 重复代码较多 | 高度复用 |
| PyTorch集成 | 非标准模型使用 | 标准nn.Module |
| 依赖关系 | 紧耦合 | 松耦合 |

## 优势

1. **简化使用**：一个调用完成所有spike生成和预测
2. **标准化**：遵循PyTorch nn.Module约定
3. **高效**：减少不必要的数据转换
4. **灵活**：支持训练和评估两种模式
5. **可扩展**：易于添加新功能

## 未来扩展

- 支持不同的随机采样策略
- 添加批量归一化等正则化技术
- 支持多GPU并行计算
- 集成更多的神经建模功能
