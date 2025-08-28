# GPU监控使用说明

## 安装依赖

首先安装GPU监控所需的库：

```bash
pip install nvidia-ml-py
```

或者使用requirements.txt：

```bash
pip install -r requirements.txt
```

## GPUMonitor类功能

`GPUMonitor`类提供了完整的GPU监控功能：

### 主要方法

- `get_utilization()`: 获取GPU利用率
- `get_memory_usage()`: 获取GPU内存使用情况
- `get_memory_percent()`: 获取GPU内存使用百分比
- `get_temperature()`: 获取GPU温度
- `get_power_usage()`: 获取GPU功耗
- `get_comprehensive_info()`: 获取所有GPU信息
- `print_status()`: 打印当前GPU状态
- `monitor_continuously()`: 持续监控GPU
- `benchmark_performance()`: 性能基准测试

## 使用方法

### 方法1：在训练代码中使用

训练代码已经集成了`GPUMonitor`类，会自动显示：
- 训练前的GPU状态
- 训练后的GPU状态
- 训练时间

```python
# 创建GPU监控器
gpu_monitor = GPUMonitor()

# 打印GPU状态
gpu_monitor.print_status()

# 获取综合信息
info = gpu_monitor.get_comprehensive_info()
print(f"GPU利用率: {info['utilization']}%")
```

### 方法2：使用独立的GPU监控脚本

在训练过程中，打开另一个终端窗口运行：

```bash
# 监控60分钟，每2秒更新一次
python gpu_monitor.py 60 2

# 监控30分钟，每5秒更新一次，并保存到文件
python gpu_monitor.py 30 5 gpu_log.csv

# 使用默认设置（60分钟，2秒间隔）
python gpu_monitor.py
```

### 方法3：在代码中进行性能基准测试

```python
gpu_monitor = GPUMonitor()
gpu_monitor.benchmark_performance(test_duration=60)  # 60秒基准测试
```

## 如何诊断GPU性能问题

### 1. 检查GPU是否被正确识别

运行训练代码时，会显示：
```
=== GPU状态诊断 ===
TensorFlow版本: 2.x.x
CUDA可用: True
GPU设备数量: 1
GPU监控器初始化成功
  GPU: NVIDIA GeForce RTX 3080
  利用率: 5%
  内存: 1024/10240 MB (10.0%)
  温度: 45°C
  功耗: 25W
==================
```

### 2. 监控训练过程中的GPU使用情况

如果GPU利用率很低（<20%），可能的原因：

#### 数据加载瓶颈
- 数据生成器加载速度慢
- 文件I/O成为瓶颈
- 数据预处理耗时过长

#### 批次大小过小
- 增加batch_size（如从32增加到64或128）
- 确保GPU内存足够

#### 多进程配置问题
- 尝试关闭多进程：`use_multiprocessing=False`
- 减少worker数量：`num_workers=2`

### 3. 使用性能基准测试

```python
gpu_monitor = GPUMonitor()
gpu_monitor.benchmark_performance(60)  # 60秒测试
```

输出示例：
```
=== 性能基准测试结果 ===
GPU利用率 - 平均: 85.2%, 最大: 98.0%, 最小: 72.1%
内存使用率 - 平均: 65.3%, 最大: 78.2%, 最小: 52.1%
GPU温度 - 平均: 68.5°C, 最大: 72.0°C, 最小: 65.0°C
GPU功耗 - 平均: 180.5W, 最大: 220.0W, 最小: 150.0W
✓ GPU利用率优秀 (>80%)
========================
```

### 4. 优化建议

#### 提高GPU利用率：
```python
# 增加批次大小
batch_size_per_epoch = [64] * num_epochs  # 或更大

# 减少数据加载开销
train_file_load = 0.3  # 减少到0.3或更小
```

#### 使用tf.data.Dataset替代自定义生成器：
```python
# 在fit_CNN.py中实现tf.data.Dataset版本
```

#### 预加载数据到内存：
```python
# 在训练前预加载所有数据，避免训练时的文件I/O
```

## 高级功能

### 1. 多GPU监控

```python
# 监控多个GPU
gpu_monitor_0 = GPUMonitor(gpu_index=0)
gpu_monitor_1 = GPUMonitor(gpu_index=1)

print("GPU 0:", gpu_monitor_0.get_utilization())
print("GPU 1:", gpu_monitor_1.get_utilization())
```

### 2. 保存监控数据

```python
# 保存监控数据到CSV文件
gpu_monitor.monitor_continuously(
    duration_seconds=300, 
    interval_seconds=5, 
    output_file="gpu_monitoring.csv"
)
```

### 3. 自定义监控间隔

```python
# 高频率监控（每1秒）
gpu_monitor.monitor_continuously(60, 1)

# 低频率监控（每10秒）
gpu_monitor.monitor_continuously(300, 10)
```

## 常见问题

### Q: GPU利用率显示为"N/A"或"Error"
A: 检查是否正确安装了nvidia-ml-py，确保有NVIDIA GPU驱动

### Q: GPU利用率很低但训练很慢
A: 可能是数据加载瓶颈，尝试上述优化建议

### Q: GPU内存不足
A: 减少batch_size或使用GPU内存增长模式（代码中已启用）

### Q: 温度过高
A: 检查GPU散热，可能需要清理风扇或调整环境温度

## 性能基准

理想的GPU使用情况：
- GPU利用率：>80%
- GPU内存使用：>50%
- GPU温度：<80°C
- 训练速度：比CPU快3-10倍

如果达不到这些基准，请按照上述建议进行优化。 