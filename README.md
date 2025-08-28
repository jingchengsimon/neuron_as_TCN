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