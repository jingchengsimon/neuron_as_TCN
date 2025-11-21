# TensorFlow vs PyTorch 实现差异分析

## 问题描述
TF版本200个epoch能得到很多符合条件的模型（val_spikes_loss < 0.05），但PyTorch版本只能得到1-2个。

## 关键差异点

### 1. **BatchNormalization 参数差异** ⚠️ **关键问题**

#### TensorFlow (Keras):
```python
x = BatchNormalization(name='layer_%d_BN' %(k + 1))(x)
```
- **默认参数**:
  - `momentum=0.99` (Keras默认)
  - `epsilon=1e-3` (Keras默认)
  - `axis=-1` (默认，对通道维度归一化)

#### PyTorch:
```python
nn.BatchNorm1d(num_features=num_filters, momentum=0.01, eps=0.001)
```

**关键差异**:
- **momentum定义相反**:
  - Keras: `momentum=0.99` 表示使用99%的旧统计量，1%的新统计量
  - PyTorch: `momentum=0.01` 表示使用1%的新统计量，99%的旧统计量
  - **结论**: PyTorch的`momentum=0.01`对应Keras的`momentum=0.99` ✅ **已正确**

- **epsilon**: 两者都是`1e-3` ✅ **一致**

- **训练/评估模式**:
  - TF: `fit_generator`自动处理train/eval模式
  - PyTorch: 需要手动调用`model.train()`和`model.eval()`
  - **问题**: 在验证时，PyTorch代码中使用了`temporal_conv_net.eval()`和`torch.no_grad()` ✅ **已正确**

### 2. **优化器参数差异** ⚠️ **可能问题**

#### TensorFlow:
```python
optimizer_to_use = Nadam(lr=learning_rate)
# 默认参数:
# beta_1=0.9, beta_2=0.999, epsilon=1e-7
```

#### PyTorch:
```python
optimizer = optim.NAdam(temporal_conv_net.parameters(), lr=learning_rate, weight_decay=l2_regularization_per_layer[0])
# 默认参数:
# betas=(0.9, 0.999), eps=1e-8, weight_decay=0
```

**关键差异**:
- **epsilon**: TF默认`1e-7` vs PyTorch默认`1e-8` ❌ **不一致**
- **weight_decay**: 
  - TF: L2正则化通过`kernel_regularizer=l2(l2_reg)`在每个层单独应用
  - PyTorch: `weight_decay`是全局的，应用到所有参数
  - **问题**: 在TF中，输出层使用`l2(1e-8)`，但PyTorch对所有层使用`l2_regularization_per_layer[0]`（通常是`1e-6`）❌ **不一致**

### 3. **损失函数和权重** ⚠️ **需要验证**

#### TensorFlow:
```python
loss=['binary_crossentropy','mse']
loss_weights=[1.0, 0.02]  # 在训练循环中动态变化
```

#### PyTorch:
```python
spike_criterion = nn.BCELoss()  # 默认reduction='mean'
soma_criterion = nn.MSELoss()   # 默认reduction='mean'
loss = loss_weights[0] * loss_spike + loss_weights[1] * loss_soma
```

**潜在问题**:
- **reduction**: 两者都默认`mean` ✅ **一致**
- **loss计算**: 需要确认TF和PyTorch的BCELoss实现是否完全一致
- **loss_weights**: 代码中看起来一致，但需要确认在训练循环中的使用方式

### 4. **数据采样和随机性** ⚠️ **可能问题**

#### 随机种子:
- **两个版本都没有设置随机种子** ❌
- 这会导致每次运行结果不同，但不会解释为什么PyTorch版本性能更差

#### 数据生成器:
- TF: `keras.utils.Sequence`，支持多进程
- PyTorch: 普通Python类，单进程
- **采样逻辑**: 看起来一致，但需要确认`_select_balanced_windows()`的实现

### 5. **初始化差异** ⚠️ **需要验证**

#### TensorFlow:
```python
initializer = initializers.TruncatedNormal(stddev=initializer)
# 输出层:
output_spike_init_weights = initializers.TruncatedNormal(stddev=0.001)
output_spike_init_bias = initializers.Constant(value=-2.0)
output_soma_init = initializers.TruncatedNormal(stddev=0.03)
```

#### PyTorch:
```python
nn.init.trunc_normal_(conv.conv.weight, std=initializer)
# 输出层:
nn.init.trunc_normal_(self.spikes_head.weight, std=spike_weight_init_std)
nn.init.constant_(self.spikes_head.bias, spike_bias_init_val)
nn.init.trunc_normal_(self.soma_head.weight, std=soma_weight_init_std)
```

**潜在问题**:
- `trunc_normal_`的实现可能不同（截断范围、分布参数等）
- 需要验证两个框架的TruncatedNormal是否产生相同的分布

### 6. **训练循环差异** ⚠️ **关键问题**

#### TensorFlow:
```python
history = temporal_conv_net.fit_generator(
    generator=train_data_generator,
    epochs=num_steps_multiplier,  # 通常是10
    validation_data=valid_data_generator,
    use_multiprocessing=use_multiprocessing,
    workers=num_workers,
    verbose=1,
    callbacks=[lr_scheduler]
)
```

#### PyTorch:
```python
for mini_epoch in range(num_steps_multiplier):
    temporal_conv_net.train()
    # 训练循环...
    temporal_conv_net.eval()
    with torch.no_grad():
        # 验证循环...
```

**关键差异**:
- **验证频率**: 
  - TF: 每个epoch结束后验证一次
  - PyTorch: 每个mini_epoch都验证一次
  - **影响**: PyTorch版本验证更频繁，但应该不影响最终性能

- **学习率调度**:
  - TF: 使用`LearningRateScheduler` callback，每个epoch调用一次
  - PyTorch: **没有实现学习率调度** ❌ **缺失**
  - **影响**: 这可能是导致性能差异的主要原因！

### 7. **数值精度和dtype** ⚠️ **需要检查**

#### TensorFlow:
- 默认使用`float32`
- Keras自动处理dtype

#### PyTorch:
```python
X_batch = X_batch.to(device, non_blocking=True)  # 默认float32
```
- 需要确认数据是否都是`float32`
- GPU上的计算可能有精度差异

### 8. **梯度计算和更新** ⚠️ **需要验证**

#### TensorFlow:
- 自动处理梯度计算和更新
- 默认使用`tf.GradientTape`（TF 2.x）或自动微分（Keras）

#### PyTorch:
```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```
- 需要确认梯度裁剪是否存在（两个版本都没有）
- 需要确认梯度累积是否一致

### 9. **L2正则化应用方式** ⚠️ **关键问题**

#### TensorFlow:
```python
kernel_regularizer=l2(l2_reg)  # 每层单独设置
# 输出层使用 l2(1e-8)
```

#### PyTorch:
```python
weight_decay=l2_regularization_per_layer[0]  # 全局应用到所有参数
# 输出层也受到相同的weight_decay影响
```

**问题**: 
- TF中输出层的L2正则化是`1e-8`，但PyTorch中所有层（包括输出层）都使用`1e-6`
- 这可能导致输出层过度正则化，影响spike预测性能

### 10. **模型保存条件** ⚠️ **需要确认**

#### TensorFlow:
```python
if np.array(training_history_dict['val_spikes_loss'][-3:]).mean() < 0.03:
```

#### PyTorch:
```python
if np.array(training_history_dict['val_spikes_loss'][-3:]).mean() < 0.05:
```

**差异**: PyTorch版本的阈值更宽松（0.05 vs 0.03），但性能仍然更差，说明问题不在阈值设置。

## 最可能的原因排序

### 🔴 **高优先级问题**:

1. **学习率调度缺失** (PyTorch版本) ⚠️⚠️⚠️ **最可能的原因**
   - **TF版本**: 使用`LearningRateScheduler` callback，在每个mini-epoch（`num_steps_multiplier=10`）内都会调用`lr_warmup_decay`函数
   - **PyTorch版本**: 只在每个`learning_schedule`（大epoch）开始时设置学习率，在10个mini-epoch内保持不变
   - **影响**: 
     - TF版本在每个mini-epoch内都有warmup/decay调度
     - PyTorch版本学习率固定，无法利用细粒度的学习率调整
     - 这可能导致训练不稳定，难以收敛到好的解
   - **代码位置**:
     - TF: `3_train_and_analyze_tf.py:463` - `callbacks=[lr_scheduler]`
     - PyTorch: `3_train_and_analyze.py:226-227` - 只在大epoch开始时更新

2. **L2正则化应用不一致** ⚠️⚠️ **关键问题**
   - **TF版本**: 
     ```python
     # 卷积层
     kernel_regularizer=l2(1e-6)  # l2_regularization_per_layer[0]
     # 输出层
     kernel_regularizer=l2(1e-8)  # 硬编码
     ```
   - **PyTorch版本**:
     ```python
     # 所有层（包括输出层）
     weight_decay=l2_regularization_per_layer[0]  # 1e-6
     ```
   - **影响**: 
     - 输出层（特别是spike预测头）在PyTorch中受到更强的L2正则化（1e-6 vs 1e-8）
     - 这可能导致spike预测性能下降，因为输出层参数被过度约束
   - **代码位置**:
     - TF: `fit_CNN.py:160-162` - 输出层使用`l2(1e-8)`
     - PyTorch: `3_train_and_analyze.py:222` - 全局`weight_decay=1e-6`

3. **优化器epsilon差异** ⚠️ **可能影响**
   - **TF NAdam默认**: `epsilon=1e-7`
   - **PyTorch NAdam默认**: `eps=1e-8`
   - **影响**: 虽然差异小，但在梯度很小的情况下可能累积影响优化稳定性
   - **建议**: 在PyTorch中显式设置`eps=1e-7`以匹配TF

### 🟡 **中优先级问题**:

4. **TruncatedNormal初始化实现差异**
   - 两个框架的实现可能不完全相同

5. **数据采样和随机性**
   - 没有设置随机种子，可能导致结果不稳定

### 🟢 **低优先级问题**:

6. **验证频率差异**
   - 应该不影响最终性能

7. **数值精度**
   - 通常不是主要问题

## 建议修复方案

### 1. **添加学习率调度器**（最重要）🔴

在PyTorch训练循环中添加学习率调度：

```python
# 在每个mini_epoch内调整学习率
for mini_epoch in range(num_steps_multiplier):
    # 计算当前mini_epoch的学习率（考虑warmup和decay）
    current_lr = lr_warmup_decay(mini_epoch, learning_rate, ...)
    for g in optimizer.param_groups:
        g['lr'] = current_lr
    
    # 训练循环...
```

或者使用PyTorch的`lr_scheduler`:

```python
from torch.optim.lr_scheduler import LambdaLR

# 创建调度器
def lr_lambda(epoch):
    return lr_warmup_decay(epoch, learning_rate, ...) / learning_rate

scheduler = LambdaLR(optimizer, lr_lambda)

# 在每个mini_epoch后调用
for mini_epoch in range(num_steps_multiplier):
    # 训练...
    scheduler.step()
```

### 2. **修正L2正则化**（关键）🔴

**方案A**: 使用参数组为输出层设置不同的weight_decay

```python
# 分离输出层和其他层
output_params = list(temporal_conv_net.spikes_head.parameters()) + \
                list(temporal_conv_net.soma_head.parameters())
other_params = [p for n, p in temporal_conv_net.named_parameters() 
                if 'spikes_head' not in n and 'soma_head' not in n]

optimizer = optim.NAdam([
    {'params': other_params, 'weight_decay': l2_regularization_per_layer[0]},  # 1e-6
    {'params': output_params, 'weight_decay': 1e-8}  # 输出层使用1e-8
], lr=learning_rate, eps=1e-7)
```

**方案B**: 在损失函数中手动添加L2正则化（更灵活但更复杂）

### 3. **统一优化器参数**🟡

```python
optimizer = optim.NAdam(
    temporal_conv_net.parameters(), 
    lr=learning_rate, 
    weight_decay=l2_regularization_per_layer[0],
    eps=1e-7,  # 匹配TF的默认值
    betas=(0.9, 0.999)  # 显式设置（虽然已经是默认值）
)
```

### 4. **添加随机种子设置**🟡

在训练开始前设置：

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 在main()或train_and_save()开始时调用
set_seed(42)
```

### 5. **验证TruncatedNormal初始化**🟢

可以添加测试代码验证两个框架的初始化分布是否一致。

### 6. **检查BatchNorm的track_running_stats**🟢

确保BatchNorm在训练和评估时行为一致：

```python
nn.BatchNorm1d(
    num_features=num_filters, 
    momentum=0.01, 
    eps=0.001,
    track_running_stats=True  # 显式设置（默认True）
)
```

## 修复优先级

1. **立即修复**: 学习率调度器 + L2正则化分离
2. **尽快修复**: 优化器epsilon参数
3. **建议修复**: 随机种子设置
4. **可选修复**: 其他细节优化

