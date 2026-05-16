import os
import random
import numpy as np
import time

# TensorFlow/Keras (for Keras test)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)
from keras.optimizers import Nadam

# PyTorch (for Torch test)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda

# Reuse project utilities
from utils.fit_CNN import create_temporaly_convolutional_model as create_keras_tcn
from utils.fit_CNN import SimulationDataGenerator as KerasDataGen
from utils.fit_CNN_torch import create_temporaly_convolutional_model as create_torch_tcn
from utils.fit_CNN_torch import SimulationDataGenerator as TorchDataGen

def set_global_seed(seed: int = 1234, deterministic: bool = True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.set_random_seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def build_paths():
    # 保持与 3_train_and_analyze.py 一致，以便复现
    test_suffix = '_SJC_funcgroup2_var2'
    base_path = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut' + test_suffix
    data_suffix = 'L5PC_NMDA'
    model_suffix = 'NMDA_torch'  # 仅用于打印标签
    train_dir = f'{base_path}/data/{data_suffix}_train/'
    valid_dir = f'{base_path}/data/{data_suffix}_valid/'
    return train_dir, valid_dir, model_suffix

def common_arch():
    # 与 3_train_and_analyze.py 保持一致
    network_depth = 7
    num_filters_per_layer_single = 256
    input_window_size = 400

    num_segments_exc = 639
    # SJC 数据抑制性通道为 640，否则 639；这里按路径名判定交由各自的数据装载器
    num_segments_inh_default = 640 

    filter_sizes_per_layer = [54] + [24] * (network_depth - 1)
    initializer_per_layer = [0.002] * network_depth
    activation_function_per_layer = ['relu'] * network_depth
    l2_regularization_per_layer = [1e-6] * network_depth
    strides_per_layer = [1] * network_depth
    dilation_rates_per_layer = [1] * network_depth
    num_filters_per_layer = [num_filters_per_layer_single] * network_depth

    return dict(
        network_depth=network_depth,
        input_window_size=input_window_size,
        num_segments_exc=num_segments_exc,
        num_segments_inh_default=num_segments_inh_default,
        filter_sizes_per_layer=filter_sizes_per_layer,
        initializer_per_layer=initializer_per_layer,
        activation_function_per_layer=activation_function_per_layer,
        l2_regularization_per_layer=l2_regularization_per_layer,
        strides_per_layer=strides_per_layer,
        dilation_rates_per_layer=dilation_rates_per_layer,
        num_filters_per_layer=num_filters_per_layer,
    )

def run_keras(train_dir, valid_dir, arch, epochs=10, files_per_epoch_train=4, files_per_epoch_valid=2, file_load=0.2, batch_size=8):
    print("\n=== Running Keras TCN ===")
    # 数据生成器（Keras）
    train_gen = KerasDataGen(
        sim_experiment_files=sorted([f for f in glob_glob(train_dir)]),
        num_files_per_epoch=files_per_epoch_train,
        batch_size=batch_size,
        window_size_ms=arch['input_window_size'],
        file_load=file_load,
        use_improved_sampling=True,
        spike_rich_ratio=0.6,
    )
    valid_gen = KerasDataGen(
        sim_experiment_files=sorted([f for f in glob_glob(valid_dir)]),
        num_files_per_epoch=files_per_epoch_valid,
        batch_size=batch_size,
        window_size_ms=arch['input_window_size'],
        file_load=file_load,
        use_improved_sampling=True,
        spike_rich_ratio=0.6,
    )

    # 模型（Keras）
    model = create_keras_tcn(
        max_input_window_size=arch['input_window_size'],
        num_segments_exc=arch['num_segments_exc'],
        num_segments_inh=arch['num_segments_inh_default'],
        filter_sizes_per_layer=arch['filter_sizes_per_layer'],
        num_filters_per_layer=arch['num_filters_per_layer'],
        activation_function_per_layer=arch['activation_function_per_layer'],
        l2_regularization_per_layer=arch['l2_regularization_per_layer'],
        strides_per_layer=arch['strides_per_layer'],
        dilation_rates_per_layer=arch['dilation_rates_per_layer'],
        initializer_per_layer=arch['initializer_per_layer'],
        use_improved_initialization=False
    )
    # 确认优化器学习率
    try:
        model.optimizer.lr.assign(0.0001)
    except Exception:
        pass

    steps_per_epoch = len(train_gen)
    val_steps = len(valid_gen)

    start = time.time()
    history = model.fit_generator(
        generator=train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=valid_gen,
        validation_steps=val_steps,
        verbose=1,
        workers=0,
        use_multiprocessing=False
    )
    duration = time.time() - start
    print(f"Keras training finished in {duration:.2f}s")

    # 取 spike loss
    spikes_loss = history.history.get('spikes_loss', [])
    val_spikes_loss = history.history.get('val_spikes_loss', [])
    return spikes_loss, val_spikes_loss

def run_torch(train_dir, valid_dir, arch, epochs=10, files_per_epoch_train=4, files_per_epoch_valid=2, file_load=0.2, batch_size=8, spike_weight=1.0, soma_weight=0.006):
    print("\n=== Running PyTorch TCN ===")
    device = get_device()
    print(f"Using device: {device}")

    # 数据生成器（Torch）
    train_gen = TorchDataGen(
        sim_experiment_files=sorted([f for f in glob_glob(train_dir)]),
        num_files_per_epoch=files_per_epoch_train,
        batch_size=batch_size,
        window_size_ms=arch['input_window_size'],
        file_load=file_load,
        use_improved_sampling=True,
        spike_rich_ratio=0.6,
    )
    valid_gen = TorchDataGen(
        sim_experiment_files=sorted([f for f in glob_glob(valid_dir)]),
        num_files_per_epoch=files_per_epoch_valid,
        batch_size=batch_size,
        window_size_ms=arch['input_window_size'],
        file_load=file_load,
        use_improved_sampling=True,
        spike_rich_ratio=0.6,
    )

    # 模型（Torch）
    model = create_torch_tcn(
        max_input_window_size=arch['input_window_size'],
        num_segments_exc=arch['num_segments_exc'],
        num_segments_inh=arch['num_segments_inh_default'],
        filter_sizes_per_layer=arch['filter_sizes_per_layer'],
        num_filters_per_layer=arch['num_filters_per_layer'],
        activation_function_per_layer=arch['activation_function_per_layer'],
        l2_regularization_per_layer=arch['l2_regularization_per_layer'],
        strides_per_layer=arch['strides_per_layer'],
        dilation_rates_per_layer=arch['dilation_rates_per_layer'],
        initializer_per_layer=arch['initializer_per_layer'],
        use_improved_initialization=False
    ).to(device)

    criterion_spike = nn.BCELoss()
    criterion_soma = nn.MSELoss()
    try:
        optimizer = optim.NAdam(model.parameters(), lr=1e-4)
    except Exception:
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

    steps_per_epoch = len(train_gen)
    val_steps = len(valid_gen)
    spikes_loss_hist = []
    val_spikes_loss_hist = []

    start = time.time()
    for ep in range(epochs):
        model.train()
        running_spike = 0.0

        if cuda.is_available():
            cuda.empty_cache()

        for step in range(steps_per_epoch):
            xb, targets = train_gen[step]
            ysb, yvb = targets
            xb = xb.to(device, non_blocking=True)
            ysb = ysb.to(device, non_blocking=True)
            yvb = yvb.to(device, non_blocking=True)

            optimizer.zero_grad()
            ps, pv = model(xb)
            loss_spike = criterion_spike(ps, ysb)
            loss_soma = criterion_soma(pv, yvb)
            loss = spike_weight * loss_spike + soma_weight * loss_soma
            loss.backward()
            optimizer.step()
            running_spike += loss_spike.item()

        spikes_loss_hist.append(running_spike / max(1, steps_per_epoch))

        # 验证
        model.eval()
        with torch.no_grad():
            val_spike_running = 0.0
            for vstep in range(val_steps):
                xb, targets = valid_gen[vstep]
                ysb, yvb = targets
                xb = xb.to(device, non_blocking=True)
                ysb = ysb.to(device, non_blocking=True)
                yvb = yvb.to(device, non_blocking=True)
                ps, pv = model(xb)
                val_spike_running += criterion_spike(ps, ysb).item()
            val_spikes_loss_hist.append(val_spike_running / max(1, val_steps))

        print(f"Epoch {ep+1}/{epochs} | Torch spikes_loss={spikes_loss_hist[-1]:.6f} val_spikes_loss={val_spikes_loss_hist[-1]:.6f}")

    duration = time.time() - start
    print(f"Torch training finished in {duration:.2f}s")
    return spikes_loss_hist, val_spikes_loss_hist

def glob_glob(dir_path):
    import glob
    return glob.glob(os.path.join(dir_path, '*.p'))

def main():
    set_global_seed(1234, deterministic=True)
    train_dir, valid_dir, tag = build_paths()
    arch = common_arch()

    # 训练 10 epoch
    epochs = 50
    files_per_epoch_train = 4
    files_per_epoch_valid = 2
    file_load = 0.2
    batch_size = 8

    keras_train_spike, keras_val_spike = run_keras(
        train_dir, valid_dir, arch, epochs, files_per_epoch_train, files_per_epoch_valid, file_load, batch_size
    )
    torch_train_spike, torch_val_spike = run_torch(
        train_dir, valid_dir, arch, epochs, files_per_epoch_train, files_per_epoch_valid, file_load, batch_size,
        spike_weight=1.0, soma_weight=0.006
    )

    print("\n=== Spike Loss Comparison (Train) ===")
    for i in range(epochs):
        k = keras_train_spike[i] if i < len(keras_train_spike) else float('nan')
        t = torch_train_spike[i] if i < len(torch_train_spike) else float('nan')
        print(f"Epoch {i+1:02d} | Keras: {k:.6f} | Torch: {t:.6f}")

    print("\n=== Spike Loss Comparison (Valid) ===")
    for i in range(epochs):
        k = keras_val_spike[i] if i < len(keras_val_spike) else float('nan')
        t = torch_val_spike[i] if i < len(torch_val_spike) else float('nan')
        print(f"Epoch {i+1:02d} | Keras: {k:.6f} | Torch: {t:.6f}")

if __name__ == "__main__":
    main()