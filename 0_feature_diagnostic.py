"""
0_feature_diagnostic.py
=======================
Diagnostic script that compares three input feature representations to isolate
whether the per-segment mean aggregation is the bottleneck limiting TCN spike
prediction precision.

Three representations (same underlying data, different compression):
  Rep A — mean rate       : current pipeline behaviour (may destroy coincidence signal)
  Rep B — binary (clipped): first-synapse proxy, restores sparsity like reference impl.
  Rep C — coincidence     : raw count + causal NMDA-like smoothing (tau=3ms)

All reps feed the same small TCNModel (depth=3, filters=32) trained with BCE loss.
Comparison metrics: ROC-AUC, PR-AUC, Victor-Purpura distance, input sparsity.

Usage:
    python 0_feature_diagnostic.py --test_dir /path/to/test/ --n_trials 100

Outputs → ./results/0_feature_diagnostic/
"""

import argparse
import os
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

from utils.fit_CNN_torch import TCNModel, parse_sim_experiment_file
from utils.visualization_utils import _setup_plot_style

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_SEGMENTS_EXC = 639
SEED = 42
SMALL_TCN_DEPTH = 3
SMALL_TCN_FILTERS = 32
INPUT_WINDOW_SIZE = 200   # ms
DEFAULT_EPOCHS = 20
DEFAULT_BATCH = 64
TAU_COINCIDENCE_MS = 3    # NMDA integration time constant
VP_COST_PER_MS = 1.0      # Victor-Purpura cost per ms of spike shift
REFRACTORY_MS = 3         # minimum inter-spike interval for peak detection
SPIKE_DECODE_THRESHOLD = 0.5
SAVE_DIR = './results/0_feature_diagnostic/'

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Victor-Purpura distance (self-contained, not added to utils)
# ---------------------------------------------------------------------------
def victor_purpura_distance(spike_train_a: np.ndarray,
                             spike_train_b: np.ndarray,
                             cost_per_ms: float = VP_COST_PER_MS) -> float:
    """
    Victor-Purpura spike train dissimilarity metric.

    Args:
        spike_train_a: 1-D array of spike times in ms (sorted)
        spike_train_b: 1-D array of spike times in ms (sorted)
        cost_per_ms:   cost of shifting a spike by 1 ms

    Returns:
        Scalar distance (lower = more similar timing).
    """
    n, m = len(spike_train_a), len(spike_train_b)
    if n == 0 and m == 0:
        return 0.0
    if n == 0:
        return float(m)
    if m == 0:
        return float(n)

    D = np.zeros((n + 1, m + 1), dtype=np.float64)
    D[:, 0] = np.arange(n + 1, dtype=np.float64)
    D[0, :] = np.arange(m + 1, dtype=np.float64)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            move_cost = cost_per_ms * abs(float(spike_train_a[i - 1]) -
                                          float(spike_train_b[j - 1]))
            D[i, j] = min(
                D[i - 1, j] + 1.0,        # delete spike from a
                D[i, j - 1] + 1.0,        # insert spike into a
                D[i - 1, j - 1] + move_cost  # move spike
            )
    return float(D[n, m])


# ---------------------------------------------------------------------------
# Feature extraction: three representations
# ---------------------------------------------------------------------------
def _spikes_dict_to_binary(spike_dict: dict,
                            num_segments: int,
                            sim_duration_ms: int) -> np.ndarray:
    """
    Convert per-segment spike-time dict to a dense integer count matrix.

    Returns:
        count_matrix: (num_segments, sim_duration_ms) int16
                      value = number of spikes in that 1 ms bin
    """
    count_matrix = np.zeros((num_segments, sim_duration_ms), dtype=np.int16)
    for seg_idx, spike_times in spike_dict.items():
        if not hasattr(spike_times, '__iter__'):
            continue
        for t in spike_times:
            t_int = int(t)
            if 0 <= t_int < sim_duration_ms:
                count_matrix[seg_idx, t_int] += 1
    return count_matrix


def build_rep_A(exc_dict: dict, inh_dict: dict,
                num_segments_exc: int, num_segments_inh: int,
                sim_duration_ms: int,
                synapses_per_segment_exc: float = 40.0,
                synapses_per_segment_inh: float = 4.0) -> np.ndarray:
    """
    Rep A — mean rate (current pipeline behaviour).

    Divides raw spike count by estimated synapses-per-segment so the value
    approximates a per-synapse firing probability in each ms bin.
    Falls back to raw binary if divisor unknown.

    Returns:
        X: (num_segments_exc + num_segments_inh, sim_duration_ms) float32  # C × T
    """
    exc_counts = _spikes_dict_to_binary(exc_dict, num_segments_exc, sim_duration_ms)
    inh_counts = _spikes_dict_to_binary(inh_dict, num_segments_inh, sim_duration_ms)

    # FALLBACK: if actual synapse count unavailable, divisor defaults give approx mean
    exc_mean = exc_counts.astype(np.float32) / synapses_per_segment_exc
    inh_mean = inh_counts.astype(np.float32) / synapses_per_segment_inh

    return np.vstack([exc_mean, inh_mean])  # (C, T)


def build_rep_B(exc_dict: dict, inh_dict: dict,
                num_segments_exc: int, num_segments_inh: int,
                sim_duration_ms: int) -> np.ndarray:
    """
    Rep B — binary clipped (first-synapse sparse proxy).

    Treats the aggregated segment spike list as if it came from a single synapse:
    any activity in a bin → 1, otherwise 0.  Restores the sparsity structure
    present in the Beniaguev et al. reference implementation.

    Returns:
        X: (num_segments_exc + num_segments_inh, sim_duration_ms) float32  # C × T
    """
    exc_counts = _spikes_dict_to_binary(exc_dict, num_segments_exc, sim_duration_ms)
    inh_counts = _spikes_dict_to_binary(inh_dict, num_segments_inh, sim_duration_ms)

    exc_bin = np.clip(exc_counts, 0, 1).astype(np.float32)
    inh_bin = np.clip(inh_counts, 0, 1).astype(np.float32)

    return np.vstack([exc_bin, inh_bin])  # (C, T)


def _causal_exp_kernel(tau_ms: int) -> np.ndarray:
    """Causal exponential decay kernel, length = 4*tau_ms + 1."""
    t = np.arange(4 * tau_ms + 1, dtype=np.float32)
    kernel = np.exp(-t / tau_ms).astype(np.float32)
    kernel /= kernel.sum()
    return kernel


def build_rep_C(exc_dict: dict, inh_dict: dict,
                num_segments_exc: int, num_segments_inh: int,
                sim_duration_ms: int,
                tau_ms: int = TAU_COINCIDENCE_MS) -> np.ndarray:
    """
    Rep C — coincidence rate with causal NMDA-like smoothing.

    Raw multi-synapse spike count per bin (not divided), smoothed with a
    causal exponential kernel (tau=3ms, mimicking NMDA integration window).
    Preserves coincidence information: 10 simultaneous spikes → 10× signal.

    Returns:
        X: (num_segments_exc + num_segments_inh, sim_duration_ms) float32  # C × T
    """
    exc_counts = _spikes_dict_to_binary(exc_dict, num_segments_exc, sim_duration_ms)
    inh_counts = _spikes_dict_to_binary(inh_dict, num_segments_inh, sim_duration_ms)

    kernel = _causal_exp_kernel(tau_ms)
    num_segments = num_segments_exc + num_segments_inh
    X = np.zeros((num_segments, sim_duration_ms), dtype=np.float32)  # (C, T)

    for seg_idx in range(num_segments_exc):
        raw = exc_counts[seg_idx].astype(np.float32)
        smoothed = np.convolve(raw, kernel, mode='full')[:sim_duration_ms]
        X[seg_idx] = smoothed

    for seg_idx in range(num_segments_inh):
        raw = inh_counts[seg_idx].astype(np.float32)
        smoothed = np.convolve(raw, kernel, mode='full')[:sim_duration_ms]
        X[num_segments_exc + seg_idx] = smoothed

    return X  # (C, T)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FixedWindowDataset(Dataset):
    """
    Sliding-window dataset over pre-computed (X, y_spike) arrays.

    Args:
        X_list      : list of (C, T) float32 arrays, one per trial
        y_list      : list of (T,) float32 binary arrays, one per trial
        window_size : input window in ms
        stride      : sliding step in ms (default = window_size // 4)
    """

    def __init__(self, X_list, y_list, window_size: int = INPUT_WINDOW_SIZE,
                 stride: int = None):
        self.window_size = window_size
        self.stride = stride or window_size // 4
        self.samples = []  # list of (X_window, y_window)

        for X, y in zip(X_list, y_list):
            T = X.shape[1]
            for end in range(window_size, T + 1, self.stride):
                x_win = X[:, end - window_size:end].T  # (T, C) for TCNModel: (B,T,C)
                y_win = y[end - window_size:end, np.newaxis]  # (T, 1)
                self.samples.append((
                    torch.from_numpy(x_win.astype(np.float32)),   # (T, C)
                    torch.from_numpy(y_win.astype(np.float32))    # (T, 1)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_trials(test_dir: str, n_trials: int, seed: int):
    """
    Load up to n_trials from .p files in test_dir using parse_sim_experiment_file.

    Returns:
        trials: list of sim_dict (each with exInputSpikeTimes, inhInputSpikeTimes,
                outputSpikeTimes, somaVoltageLowRes)
        sim_duration_ms: int
    """
    p_files = sorted(glob.glob(os.path.join(test_dir, '*.p')))
    if not p_files:
        raise FileNotFoundError(f'No .p files found in {test_dir}')

    rng = np.random.default_rng(seed)
    rng.shuffle(p_files)

    trials = []
    sim_duration_ms = None

    for p_file in p_files:
        if len(trials) >= n_trials:
            break
        try:
            with open(p_file, 'rb') as f:
                data = pickle.load(f)
            sim_dicts = data['Results']['listOfSingleSimulationDicts']
            for sd in sim_dicts:
                if len(trials) >= n_trials:
                    break
                trials.append(sd)
                if sim_duration_ms is None:
                    sim_duration_ms = len(sd['somaVoltageLowRes'])
        except Exception as e:
            print(f'Warning: could not load {p_file}: {e}')

    if not trials:
        raise RuntimeError('No trials loaded. Check --test_dir.')

    print(f'Loaded {len(trials)} trials, sim_duration_ms={sim_duration_ms}')
    return trials, sim_duration_ms


def build_arrays(trials, sim_duration_ms: int, rep: str):
    """
    Build (X, y_spike) arrays for all trials under the specified representation.

    Args:
        trials         : list of sim_dict
        sim_duration_ms: int
        rep            : 'A', 'B', or 'C'

    Returns:
        X_list : list of (C, T) float32
        y_list : list of (T,) float32 binary
        sparsity: mean fraction of zero-valued bins
    """
    X_list, y_list = [], []
    num_inh = None

    build_fn = {'A': build_rep_A, 'B': build_rep_B, 'C': build_rep_C}[rep]

    for sd in trials:
        exc_dict = sd['exInputSpikeTimes']
        inh_dict = sd['inhInputSpikeTimes']

        # Infer num_segments_inh from data on first trial
        if num_inh is None:
            num_inh = max((int(k) for k in inh_dict.keys()), default=110) + 1
            # Clamp to reasonable range
            num_inh = max(num_inh, 111)

        if rep == 'A':
            X = build_fn(exc_dict, inh_dict, NUM_SEGMENTS_EXC, num_inh, sim_duration_ms)
        else:
            X = build_fn(exc_dict, inh_dict, NUM_SEGMENTS_EXC, num_inh, sim_duration_ms)

        # Binary spike target
        y = np.zeros(sim_duration_ms, dtype=np.float32)
        for t in sd['outputSpikeTimes']:
            t_int = int(t)
            if 0 <= t_int < sim_duration_ms:
                y[t_int] = 1.0

        X_list.append(X)
        y_list.append(y)

    total_bins = sum(x.size for x in X_list)
    zero_bins = sum(np.sum(x == 0) for x in X_list)
    sparsity = zero_bins / total_bins if total_bins > 0 else 0.0

    return X_list, y_list, sparsity, num_inh


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def build_small_tcn(num_segments_exc: int, num_segments_inh: int,
                    device: torch.device) -> TCNModel:
    """Instantiate the fixed small TCN used for all three representations."""
    depth = SMALL_TCN_DEPTH
    model = TCNModel(
        max_input_window_size=INPUT_WINDOW_SIZE,
        num_segments_exc=num_segments_exc,
        num_segments_inh=num_segments_inh,
        filter_sizes_per_layer=[8] * depth,
        num_filters_per_layer=[SMALL_TCN_FILTERS] * depth,
        activation_function_per_layer=['relu'] * depth,
        strides_per_layer=[1] * depth,
        dilation_rates_per_layer=[2 ** i for i in range(depth)],
        initializer_per_layer=['glorot_uniform'] * depth,
        use_improved_initialization=False,
    ).to(device)
    return model


def train_one_rep(X_list, y_list, num_inh: int, n_epochs: int,
                  batch_size: int, device: torch.device,
                  rep_label: str, save_dir: str) -> tuple:
    """
    Train small TCN on one representation. Returns (model, val_loss_history).

    Checkpoint saved to save_dir/model_rep_{rep_label}.pt
    """
    set_seed(SEED)
    n = len(X_list)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train_ds = FixedWindowDataset(X_list[:n_train], y_list[:n_train])
    val_ds = FixedWindowDataset(X_list[n_train:n_train + n_val], y_list[n_train:n_train + n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=device.type == 'cuda')

    model = build_small_tcn(NUM_SEGMENTS_EXC, num_inh, device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    ckpt_path = os.path.join(save_dir, f'model_rep_{rep_label}.pt')
    val_history = []

    for epoch in range(n_epochs):
        # --- train ---
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)    # (B, T, C)
            y_batch = y_batch.to(device)    # (B, T, 1)
            optimizer.zero_grad()
            pred_spike, _ = model(X_batch)  # (B, T, 1)
            loss = criterion(pred_spike, y_batch)
            loss.backward()
            optimizer.step()

        # --- validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                pred_spike, _ = model(X_batch)
                val_losses.append(criterion(pred_spike, y_batch).item())
        val_loss = float(np.mean(val_losses))
        val_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

        print(f'  Rep {rep_label} | epoch {epoch+1:03d}/{n_epochs} '
              f'| val_loss={val_loss:.5f}  (best={best_val_loss:.5f})')

    # Reload best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, val_history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def decode_spike_times(pred_proba: np.ndarray) -> np.ndarray:
    """
    Convert predicted probability sequence to spike time array using peak detection.

    Args:
        pred_proba: (T,) float array

    Returns:
        spike_times: sorted int array of detected spike times in ms
    """
    peaks, _ = find_peaks(pred_proba,
                          height=SPIKE_DECODE_THRESHOLD,
                          distance=REFRACTORY_MS)
    return peaks.astype(np.int32)


def evaluate_rep(model: TCNModel, X_list, y_list,
                 num_inh: int, device: torch.device) -> dict:
    """
    Run evaluation on test trials (last 15% of list).

    Returns dict with roc_auc, pr_auc, mean_vp_dist.
    """
    n = len(X_list)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    test_X = X_list[n_train + n_val:]
    test_y = y_list[n_train + n_val:]

    model.eval()
    all_pred, all_true = [], []
    vp_distances = []

    with torch.no_grad():
        for X, y in zip(test_X, test_y):
            T = X.shape[1]
            preds = np.zeros(T, dtype=np.float32)

            # Sliding-window inference
            for end in range(INPUT_WINDOW_SIZE, T + 1, INPUT_WINDOW_SIZE // 4):
                x_win = X[:, end - INPUT_WINDOW_SIZE:end].T  # (T_win, C)
                x_t = torch.from_numpy(x_win.astype(np.float32)).unsqueeze(0).to(device)
                # x_t shape: (1, T_win, C) — (B, T, C)
                pred_t, _ = model(x_t)
                pred_np = pred_t.squeeze().cpu().numpy()  # (T_win,)
                # Average overlapping predictions
                preds[end - INPUT_WINDOW_SIZE:end] = np.maximum(
                    preds[end - INPUT_WINDOW_SIZE:end], pred_np
                )

            all_pred.append(preds)
            all_true.append(y)

            # Victor-Purpura distance
            pred_spikes = decode_spike_times(preds)
            true_spikes = np.where(y > 0.5)[0].astype(np.int32)
            vp = victor_purpura_distance(pred_spikes, true_spikes)
            vp_distances.append(vp)

    pred_flat = np.concatenate(all_pred)
    true_flat = np.concatenate(all_true)

    # Guard against degenerate cases (all 0 or all 1 targets)
    if true_flat.sum() == 0 or true_flat.sum() == len(true_flat):
        roc_auc = float('nan')
        pr_auc = float('nan')
    else:
        roc_auc = roc_auc_score(true_flat, pred_flat)
        pr_auc = average_precision_score(true_flat, pred_flat)

    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'mean_vp_dist': float(np.mean(vp_distances)),
        # Also return arrays for visualization
        '_pred_example': all_pred[0] if all_pred else np.array([]),
        '_true_example': test_y[0] if test_y else np.array([]),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def plot_results(results: dict, save_dir: str):
    """
    2×2 summary figure:
      TL: ROC-AUC bar   TR: PR-AUC bar
      BL: VP-dist bar   BR: example spike train comparison (500 ms window)
    """
    reps = ['A', 'B', 'C']
    rep_labels = ['A: mean rate\n(current)', 'B: binary\n(sparse proxy)',
                  'C: coincidence\n(NMDA smooth)']
    colors = ['#4C72B0', '#DD8452', '#55A868']

    roc_vals  = [results[r]['roc_auc'] for r in reps]
    pr_vals   = [results[r]['pr_auc'] for r in reps]
    vp_vals   = [results[r]['mean_vp_dist'] for r in reps]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    def bar_panel(ax, values, title, ylabel, higher_better=True):
        bars = ax.bar(rep_labels, values, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        _setup_plot_style(ax, title=title, xlabel='Representation', ylabel=ylabel,
                          fontsize=11, grid=True)
        note = '↑ higher is better' if higher_better else '↓ lower is better'
        ax.set_title(f'{title}\n({note})', fontsize=11, fontweight='bold')

    bar_panel(axes[0, 0], roc_vals, 'ROC AUC', 'AUC', higher_better=True)
    bar_panel(axes[0, 1], pr_vals,  'PR AUC',  'Average Precision', higher_better=True)
    bar_panel(axes[1, 0], vp_vals,  'Victor–Purpura Distance', 'Mean VP dist (ms)',
              higher_better=False)

    # Example spike train panel — 500 ms window
    ax = axes[1, 1]
    window_ms = 500
    for rep, color, label in zip(reps, colors, rep_labels):
        pred = results[rep].get('_pred_example', np.array([]))
        if len(pred) >= window_ms:
            ax.plot(pred[:window_ms], color=color, alpha=0.8,
                    linewidth=1.2, label=label.replace('\n', ' '))

    # True spikes
    true_y = results['A'].get('_true_example', np.array([]))
    if len(true_y) >= window_ms:
        spike_t = np.where(true_y[:window_ms] > 0.5)[0]
        ax.vlines(spike_t, 0.9, 1.05, colors='black', linewidth=1.5, label='True spikes')

    _setup_plot_style(ax, title='Predicted spike probability (500 ms)',
                      xlabel='Time (ms)', ylabel='P(spike)', fontsize=11,
                      grid=True, legend=True, ylim=(-0.05, 1.1))

    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'diagnostic_figure.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Figure saved → {fig_path}')


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_and_save_summary(results: dict, sparsities: dict, save_dir: str):
    """Print summary table and save as CSV."""
    import csv

    header = ['rep', 'roc_auc', 'pr_auc', 'mean_vp_dist', 'mean_input_sparsity']
    rows = []
    rep_names = {
        'A': 'A_mean_rate',
        'B': 'B_binary_sparse',
        'C': 'C_coincidence',
    }

    print('\n' + '=' * 70)
    print(f'{"Rep":<20} {"ROC-AUC":>10} {"PR-AUC":>10} '
          f'{"VP-dist":>12} {"Sparsity":>12}')
    print('-' * 70)
    for rep in ['A', 'B', 'C']:
        r = results[rep]
        row = [rep_names[rep],
               f"{r['roc_auc']:.4f}",
               f"{r['pr_auc']:.4f}",
               f"{r['mean_vp_dist']:.2f}",
               f"{sparsities[rep]:.4f}"]
        rows.append(row)
        print(f'{rep_names[rep]:<20} {row[1]:>10} {row[2]:>10} {row[3]:>12} {row[4]:>12}')
    print('=' * 70)

    csv_path = os.path.join(save_dir, 'diagnostic_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f'Summary saved → {csv_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Feature representation diagnostic for TCN spike prediction'
    )
    parser.add_argument('--test_dir', type=str,
                        default='./data/L5PC_NMDA_test/',
                        help='Directory containing test .p files')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of trials to load (max)')
    parser.add_argument('--n_epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Training epochs per representation')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ---- Load trials -------------------------------------------------------
    print(f'\n[1/5] Loading up to {args.n_trials} trials from {args.test_dir}')
    trials, sim_duration_ms = load_trials(args.test_dir, args.n_trials, args.seed)

    # ---- Build three feature sets ------------------------------------------
    print('\n[2/5] Building feature representations ...')
    rep_configs = {
        'A': {'fn': build_rep_A,  'label': 'mean rate'},
        'B': {'fn': build_rep_B,  'label': 'binary (sparse proxy)'},
        'C': {'fn': build_rep_C,  'label': 'coincidence (NMDA smooth)'},
    }

    all_X, all_y, sparsities, num_inh_global = {}, {}, {}, None

    for rep in ['A', 'B', 'C']:
        print(f'  Building Rep {rep}: {rep_configs[rep]["label"]}')
        X_list, y_list, sparsity, num_inh = build_arrays(
            trials, sim_duration_ms, rep
        )
        all_X[rep] = X_list
        all_y[rep] = y_list
        sparsities[rep] = sparsity
        if num_inh_global is None:
            num_inh_global = num_inh
        print(f'    input sparsity (fraction zero bins): {sparsity:.4f}')

    # ---- Train -------------------------------------------------------------
    print('\n[3/5] Training small TCN (depth=3, filters=32) per representation ...')
    models = {}
    for rep in ['A', 'B', 'C']:
        print(f'\n--- Rep {rep}: {rep_configs[rep]["label"]} ---')
        model, _ = train_one_rep(
            all_X[rep], all_y[rep],
            num_inh=num_inh_global,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            device=device,
            rep_label=rep,
            save_dir=SAVE_DIR,
        )
        models[rep] = model

    # ---- Evaluate ----------------------------------------------------------
    print('\n[4/5] Evaluating on held-out test trials ...')
    results = {}
    for rep in ['A', 'B', 'C']:
        print(f'  Evaluating Rep {rep} ...')
        metrics = evaluate_rep(
            models[rep], all_X[rep], all_y[rep],
            num_inh=num_inh_global, device=device
        )
        results[rep] = metrics
        print(f'    ROC-AUC={metrics["roc_auc"]:.4f}  '
              f'PR-AUC={metrics["pr_auc"]:.4f}  '
              f'VP-dist={metrics["mean_vp_dist"]:.2f}')

    # ---- Summary & figure --------------------------------------------------
    print('\n[5/5] Generating summary and figure ...')
    print_and_save_summary(results, sparsities, SAVE_DIR)
    plot_results(results, SAVE_DIR)

    print(f'\nDiagnostic complete. All outputs in: {SAVE_DIR}')


if __name__ == '__main__':
    main()
