# 分批处理内存管理问题修复方案

## 问题描述

在分批处理优化过程中，使用 `tf.keras.backend.clear_session()` 清理TensorFlow会话会导致以下问题：

1. **模型计算图丢失**：`clear_session()` 会清除整个TensorFlow会话，包括模型的计算图
2. **评估阶段失败**：当优化完成后，模型无法进行预测，出现错误：
   ```
   Tensor input_layer:0, specified in either feed_devices or fetch_devices was not found in the Graph
   ```
3. **内存清理过度**：每次批次都进行完整清理，影响性能

## 解决方案

### 1. 智能内存清理策略

创建了 `_smart_memory_cleanup()` 方法，只在必要时进行清理：

- **定期清理**：每3个批次清理一次，而不是每个批次
- **最终清理**：最后一个批次完成后进行完整清理
- **条件清理**：根据内存使用情况决定是否清理

### 2. 模型重新加载机制

在关键节点重新加载模型：

- **最后一个批次完成后**：重新加载模型，为评估做准备
- **评估阶段开始时**：再次确认模型可用性
- **错误恢复**：如果模型失效，自动重新加载

### 3. 评估阶段容错处理

增强了评估阶段的错误处理：

- **模型重新加载**：评估开始前确保模型可用
- **进度监控**：每5次评估显示进度
- **错误恢复**：单次评估失败时使用默认值，继续其他评估

## 修复后的代码结构

### 智能内存清理方法

```python
def _smart_memory_cleanup(self, batch_idx, num_batches):
    """智能内存清理：只在必要时进行清理"""
    # 条件1: 每3个批次清理一次
    if (batch_idx + 1) % 3 == 0:
        should_cleanup = True
    
    # 条件2: 最后一个批次，为评估做准备
    if batch_idx == num_batches - 1:
        should_cleanup = True
    
    if should_cleanup:
        # 执行清理操作
        tf.keras.backend.clear_session()
        gc.collect()
        # GPU内存清理...
    
    # 最后一个批次后重新加载模型
    if batch_idx == num_batches - 1:
        self.processor.model = load_model(self.processor.model_path)
```

### 评估阶段模型恢复

```python
def evaluate_optimized_activity(self, optimized_firing_rates, fixed_exc_indices, num_evaluations=10):
    """评估优化后的activity"""
    # 重新加载模型以确保评估正常进行
    print("重新加载模型以确保评估正常进行...")
    try:
        from keras.models import load_model
        self.processor.model = load_model(self.processor.model_path)
        print("✓ 模型重新加载成功")
    except Exception as e:
        print(f"✗ 模型重新加载失败: {e}")
    
    # 评估循环...
```

## 优势

1. **内存效率**：减少不必要的内存清理，提高性能
2. **稳定性**：确保模型在评估阶段正常工作
3. **容错性**：即使出现错误也能继续执行
4. **可维护性**：代码结构清晰，易于调试和维护

## 使用建议

1. **监控内存使用**：观察内存清理的效果，调整清理频率
2. **测试评估阶段**：确保模型重新加载正常工作
3. **错误日志**：关注错误信息，及时调整参数
4. **性能优化**：根据实际情况调整清理策略

## 注意事项

1. **模型重新加载**：会增加一些时间开销，但确保稳定性
2. **内存清理频率**：过于频繁会影响性能，过于稀疏可能导致OOM
3. **错误处理**：评估失败时使用默认值，可能影响结果准确性
4. **依赖关系**：确保 `keras.models.load_model` 可用

这个修复方案在保持内存管理效果的同时，解决了分批处理中模型失效的问题，确保整个优化流程能够顺利完成。
