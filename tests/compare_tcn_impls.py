import os
import sys
import numpy as np
import torch

# Fix seeds for reproducibility
np.random.seed(1234)
torch.manual_seed(1234)

# Ensure project root is importable for `utils.*`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Try import TensorFlow (required for Keras baseline). If unavailable, exit gracefully.
TF_AVAILABLE = True
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    tf.set_random_seed(1234)
except Exception as _e:
    TF_AVAILABLE = False
    print("TensorFlow is not installed. Install TensorFlow to run the Keras vs PyTorch comparison.")
    print("pip install 'tensorflow<2.16'  # or appropriate version for your environment")
    # We continue defining the file so it can be run later when TF is available

if TF_AVAILABLE:
    from utils.fit_CNN import create_temporaly_convolutional_model as create_keras_tcn
from utils.fit_CNN_torch import create_temporaly_convolutional_model as create_torch_tcn


def build_models():
    # Small, identical architectures
    max_input_window_size = 50
    num_segments_exc = 2
    num_segments_inh = 2

    filter_sizes_per_layer = [5, 3]
    num_filters_per_layer = [4, 3]
    activation_function_per_layer = ['relu', 'relu']
    l2_regularization_per_layer = [0.0, 0.0]
    strides_per_layer = [1, 1]
    dilation_rates_per_layer = [1, 1]

    # Use numeric initializer to force TruncatedNormal(std) on Keras and trunc_normal_ on Torch
    initializer_per_layer = [0.01, 0.01]

    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for this comparison test but is not available.")

    keras_model = create_keras_tcn(
        max_input_window_size,
        num_segments_exc,
        num_segments_inh,
        filter_sizes_per_layer,
        num_filters_per_layer,
        activation_function_per_layer,
        l2_regularization_per_layer,
        strides_per_layer,
        dilation_rates_per_layer,
        initializer_per_layer,
        use_improved_initialization=False,
    )

    torch_model = create_torch_tcn(
        max_input_window_size,
        num_segments_exc,
        num_segments_inh,
        filter_sizes_per_layer,
        num_filters_per_layer,
        activation_function_per_layer,
        l2_regularization_per_layer,
        strides_per_layer,
        dilation_rates_per_layer,
        initializer_per_layer,
        use_improved_initialization=False,
    )

    return keras_model, torch_model, max_input_window_size, (num_segments_exc + num_segments_inh)


def set_identical_weights(keras_model, torch_model):
    """Assign deterministic, identical weights to both models (considering layout differences)."""
    # 1) Convolutional layers (named 'layer_1', 'layer_2', ... in Keras; torch_model.tcn contains CausalConv1d modules)
    torch_convs = []
    for m in torch_model.tcn:
        # CausalConv1d in our impl has attribute 'conv'
        if hasattr(m, 'conv') and isinstance(m.conv, torch.nn.Conv1d):
            torch_convs.append(m.conv)

    keras_layer_idx = 1
    for conv in torch_convs:
        k_layer = keras_model.get_layer(name=f'layer_{keras_layer_idx}')
        # Shapes: Keras (kernel, in_channels, out_channels), Torch (out_channels, in_channels, kernel)
        k_w_shape = k_layer.get_weights()[0].shape
        k_b_shape = k_layer.get_weights()[1].shape

        # Deterministic weights via numpy
        W_ker = np.linspace(-0.1, 0.1, num=np.prod(k_w_shape), dtype=np.float32).reshape(k_w_shape)
        b_ker = np.linspace(-0.05, 0.05, num=np.prod(k_b_shape), dtype=np.float32)

        # Assign to Keras
        k_layer.set_weights([W_ker, b_ker])

        # Convert and assign to Torch
        W_torch = np.transpose(W_ker, (2, 1, 0))  # (out_channels, in_channels, kernel)
        with torch.no_grad():
            conv.weight.copy_(torch.from_numpy(W_torch))
            if conv.bias is not None:
                conv.bias.copy_(torch.from_numpy(b_ker.astype(np.float32)))

        # BatchNorm for this layer (Keras: 'layer_k_BN', Torch: next module in sequence)
        try:
            k_bn = keras_model.get_layer(name=f'layer_{keras_layer_idx}_BN')
            # gamma, beta, moving_mean, moving_variance
            gamma = np.ones((k_b_shape[0],), dtype=np.float32)
            beta = np.zeros((k_b_shape[0],), dtype=np.float32)
            moving_mean = np.zeros((k_b_shape[0],), dtype=np.float32)
            moving_var = np.ones((k_b_shape[0],), dtype=np.float32)
            k_bn.set_weights([gamma, beta, moving_mean, moving_var])
        except Exception:
            pass

        keras_layer_idx += 1

    # Match BN eps/momentum with Keras defaults on Torch side
    # Keras: epsilon=1e-3, momentum=0.99
    for m in torch_model.tcn:
        if isinstance(m, torch.nn.BatchNorm1d):
            m.eps = 1e-3
            m.momentum = 0.99
            with torch.no_grad():
                m.weight.fill_(1.0)
                m.bias.fill_(0.0)
                m.running_mean.zero_()
                m.running_var.fill_(1.0)

    # 2) Output heads: 'spikes' and 'somatic' in Keras, spikes_head/soma_head in Torch
    k_spikes = keras_model.get_layer('spikes')
    k_soma = keras_model.get_layer('somatic')

    # Keras shapes: (1, channels, 1); biases (1,)
    k_sw_shape = k_spikes.get_weights()[0].shape
    k_sb_shape = k_spikes.get_weights()[1].shape
    k_vw_shape = k_soma.get_weights()[0].shape
    k_vb_shape = k_soma.get_weights()[1].shape

    W_spike = np.linspace(-0.02, 0.02, num=np.prod(k_sw_shape), dtype=np.float32).reshape(k_sw_shape)
    b_spike = np.array([-2.0], dtype=np.float32)  # match default
    W_soma = np.linspace(-0.03, 0.03, num=np.prod(k_vw_shape), dtype=np.float32).reshape(k_vw_shape)
    b_soma = np.array([0.0], dtype=np.float32)

    k_spikes.set_weights([W_spike, b_spike])
    k_soma.set_weights([W_soma, b_soma])

    # Torch heads
    with torch.no_grad():
        torch_model.spikes_head.weight.copy_(torch.from_numpy(np.transpose(W_spike, (2, 1, 0))))
        torch_model.spikes_head.bias.copy_(torch.from_numpy(b_spike))
        torch_model.soma_head.weight.copy_(torch.from_numpy(np.transpose(W_soma, (2, 1, 0))))
        torch_model.soma_head.bias.copy_(torch.from_numpy(b_soma))

    # Eval mode for both (use running stats in BN)
    torch_model.eval()


def main():
    keras_model, torch_model, T, C = build_models()
    set_identical_weights(keras_model, torch_model)

    # Create deterministic input (batch=2)
    X = np.linspace(0.0, 1.0, num=2 * T * C, dtype=np.float32).reshape(2, T, C)

    # Keras predict
    y_spike_k, y_soma_k = keras_model.predict(X, batch_size=2, verbose=0)

    # Torch forward
    X_t = torch.from_numpy(X)
    with torch.no_grad():
        y_spike_t, y_soma_t = torch_model(X_t)
    y_spike_t = y_spike_t.detach().cpu().numpy()
    y_soma_t = y_soma_t.detach().cpu().numpy()

    # Compare
    same_spike = np.array_equal(y_spike_k, y_spike_t)
    same_soma = np.array_equal(y_soma_k, y_soma_t)

    max_diff_spike = float(np.max(np.abs(y_spike_k - y_spike_t)))
    max_diff_soma = float(np.max(np.abs(y_soma_k - y_soma_t)))

    print("Comparison (exact equality):")
    print(f"  spikes identical: {same_spike}")
    print(f"  soma   identical: {same_soma}")
    print("Max absolute differences:")
    print(f"  spikes: {max_diff_spike:.10f}")
    print(f"  soma  : {max_diff_soma:.10f}")

    # Also assert close within tight tolerance
    atol = 1e-7
    close_spike = np.allclose(y_spike_k, y_spike_t, atol=atol, rtol=0.0)
    close_soma = np.allclose(y_soma_k, y_soma_t, atol=atol, rtol=0.0)
    print(f"Allclose (atol={atol}): spikes={close_spike}, soma={close_soma}")


if __name__ == "__main__":
    main()


