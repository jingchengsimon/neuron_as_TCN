# AGENT.md — Project Constraints for AI Coding Agents

> **Read this file first. It is the single source of truth for how to work in this codebase.**
> For deep dives, see `docs/CONVENTIONS.md` and `docs/ARCHITECTURE.md`.

---

## 1. Project Overview

This project trains and evaluates **Temporal Convolutional Network (TCN)** models that predict single-neuron spike output from multi-segment synaptic input. The pipeline runs as a numbered sequence of scripts (`1_` → `6_`). Two framework backends co-exist: **PyTorch** (primary, actively developed) and **TensorFlow/Keras** (legacy, kept for reference).

**Domain vocabulary:**
- `segments` = dendritic compartments (excitatory or inhibitory synaptic inputs)
- `sim_duration_ms` = simulation time in milliseconds
- `y_spike` / `y_soma` = spike train output / somatic voltage output
- `STE` = Straight-Through Estimator (used in `tcn_poisson_model.py`)
- `monoconn_seg_indices` = fixed excitatory segment indices for monosynaptic stimulation

---

## 2. Filesystem Layout

```
project_root/
├── 1_load_and_visualize.py       # Step 1: data loading & EDA
├── 2_dataset_pipeline.py         # Step 2: build train/val/test splits
├── 3_train_and_analyze_torch.py  # Step 3: training (PyTorch, primary)
├── 3_train_and_analyze_tf.py     # Step 3: training (TF, legacy)
├── 4_model_prediction_visualization.py
├── 5_main_figure_replication_torch.py
├── 5_main_figure_replication_tf.py
├── 6_activity_optimization.py    # Step 6: gradient-based input optimization
│
├── utils/                        # Reusable library modules (no __main__ entry points)
│   ├── fit_CNN_torch.py          # TCNModel, CausalConv1d, SimulationDataGenerator, helpers
│   ├── fit_CNN_tf.py             # TF/Keras equivalents (legacy)
│   ├── model_analysis.py         # AUC metrics, training-curve plots
│   ├── model_size_utils.py       # Parameter counting, size classification
│   ├── visualization_utils.py    # All matplotlib helpers
│   ├── visualization_utils_backup.py  # DO NOT EDIT — reference snapshot
│   ├── tcn_poisson_model.py      # TCNPoissonModel (STE + Poisson)
│   ├── find_best_model.py        # Best-model selector by val loss
│   └── gpu_monitor.py            # GPUMonitor class, configure_pytorch_gpu()
│
├── tests/                        # Comparison / validation scripts
├── docs/                         # Human-readable design docs
├── AGENT.md                      # ← you are here
└── results/                      # Runtime output (auto-created, never commit)
```

**Rules:**
- All reusable logic lives in `utils/`. Top-level numbered scripts are **orchestrators only**.
- `visualization_utils_backup.py` is a read-only snapshot — **never modify it**.
- Runtime outputs (`results/`, `*.log`, `*.pt`, `*.pickle`, `*.h5`) are never committed.

---

## 3. Naming Conventions

### 3.1 Variables

| Concept | Canonical name | Example |
|---|---|---|
| Excitatory segment count | `num_segments_exc` | `num_segments_exc = 639` |
| Inhibitory segment count | `num_segments_inh` | `num_segments_inh = 111` |
| Total segment count | `num_segments` or `num_segments_total` | — |
| Simulation duration | `sim_duration_ms` | time in **milliseconds** |
| Time duration | `time_duration_ms` | same unit |
| Input window | `input_window_size` | in ms |
| Spike binary matrix | `bin_spikes_matrix` | shape `(num_segments, sim_duration_ms)` |
| Spike dict | `row_inds_spike_times_map` | key=segment_index, value=list of spike times |
| Model input batch | `X_batch` (numpy) / `X_batch_t` (tensor) | `_t` suffix = torch.Tensor |
| Spike target | `y_spike`, `y_spike_batch`, `y_spike_batch_t` | — |
| Somatic voltage target | `y_soma`, `y_soma_batch`, `y_soma_batch_t` | — |
| Firing rate array | `firing_rates` | shape `(num_segments, time_duration_ms)` |
| Safe/clipped version | `safe_<original_name>` | `safe_firing_rates` |
| Model metadata dict | `model_params` or `meta` | loaded from `.pickle` |
| Architecture sub-dict | `architecture_dict` | key inside `model_params` |
| Training history dict | `training_history_dict` | key inside saved pickle |

### 3.2 Tensor / Array Dimension Order

```
# Model input (PyTorch)
X : (batch_size, time_steps, num_segments)   # B × T × C

# Spike matrix (internal)
bin_spikes_matrix : (num_segments, sim_duration_ms)   # C × T

# Firing rates
firing_rates : (num_segments, time_duration_ms)       # C × T  (same as spike matrix)

# TCN forward: permute before conv
model_input = spike_trains.permute(0, 2, 1)  # B×C×T → conv → permute back
```

Always add a comment like `# (B, T, C)` on lines where a tensor is created or transposed.

### 3.3 Functions and Classes

- Functions: `snake_case`, verb-first — `load_test_data`, `calculate_auc_metrics`, `plot_training_curves`
- Classes: `PascalCase` — `TCNModel`, `TCNPoissonModel`, `GPUMonitor`, `FiringRatesProcessor`, `SimulationDataGenerator`
- Private helpers inside a module: leading underscore — `_add_statistics_text`, `_setup_plot_style`, `_count_params_pytorch`
- Boolean flags: `use_<feature>` — `use_improved_sampling`, `use_improved_initialization`, `use_torch`

### 3.4 Files

- Top-level pipeline scripts: `<step_number>_<verb>_<noun>[_<backend>].py`
  - e.g., `3_train_and_analyze_torch.py`
- Utility modules: `<noun>_<noun>.py` (no step number, no verb prefix)
- Avoid creating new top-level scripts outside the numbered sequence without prior discussion.

### 3.5 Saved Artefacts

| Artefact | Naming pattern |
|---|---|
| PyTorch weights | `<descriptor>_torch.pt` |
| TF weights | `<descriptor>_tf.h5` |
| Metadata / history | `<descriptor>.pickle` (same stem as weights) |
| Plots | `<descriptor>_<plot_type>.pdf` (prefer PDF for publication) or `.png` (for quick previews) |

---

## 4. Architecture Constraints

### 4.1 Framework Separation

- PyTorch code lives in `*_torch.py` files; TF/Keras code in `*_tf.py` files.
- **Never mix** `import tensorflow` and `import torch` in the same module, except `model_size_utils.py` which explicitly handles both with try/except guards.
- New feature development targets **PyTorch only**. TF files are maintained for reproducibility, not extended.

### 4.2 Model Architecture

- `TCNModel` uses `CausalConv1d` (left-only padding). **Do not replace with standard `nn.Conv1d` with symmetric padding**.
- Dilation rates, filter sizes, strides, and activation functions are passed as **per-layer lists** (`*_per_layer` suffix). Always keep these lists the same length as `network_depth`.
- `TCNPoissonModel` wraps a frozen `TCNModel`. The inner model's `requires_grad` is always `False`. STE is implemented via `poisson_samples + safe_firing_rates - safe_firing_rates.detach()` — do not simplify this expression.

### 4.3 Data Shapes — Critical Invariants

```python
# Segment layout (always in this order):
X[:num_segments_exc, :]   # excitatory
X[num_segments_exc:, :]   # inhibitory

# SJC dataset key adjustment (only in dict2bin, only for 'exc' + 'sjc'):
adjusted_key = original_key - 1   # converts 1-based → 0-based

# Monosynaptic stimulation (TCNPoissonModel):
spike_time_mono_syn = input_window_size // 2   # must remain integer division
```

### 4.4 Device Handling

Always auto-detect device; never hardcode `'cuda'` or `'cpu'`:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
Exception: `calculate_auc_metrics` deliberately uses CPU — that is intentional.

### 4.5 File Pairing

`.pickle` and `.pt` (or `.h5`) files **always** share the same stem. Functions that load one must be able to derive the other via simple string replacement (`.replace('.pickle', '.pt')`). Do not change this pairing convention.

---

## 5. Coding Style

- **Python 3.8+ only.** The `if sys.version_info[0]<3` guards in `fit_CNN_torch.py` are legacy — do not add new Python 2 compatibility code.
- Type hints are encouraged for new public functions but not required for internal helpers.
- Docstrings: use the existing **Google style** (`Args:`, `Returns:`, `Example:`).
- `print()` is acceptable for training progress; use `f-strings` (not `.format()` or `%`).
- Wrap optional imports with try/except and set a `<LIB>_AVAILABLE` boolean flag (see `gpu_monitor.py`, `model_size_utils.py`).
- `os.makedirs(path, exist_ok=True)` — always use `exist_ok=True`.
- Prefer `np.nan_to_num(..., nan=0.0, posinf=1.0, neginf=0.0)` over manual NaN checks.

---

## 6. What NOT to Do

- ❌ Do **not** modify `visualization_utils_backup.py`.
- ❌ Do **not** add `torch.backends.cudnn.benchmark = True` inside model classes — it belongs in `configure_pytorch_gpu()` only.
- ❌ Do **not** hardcode absolute paths like `/home/user/...`. Use relative paths or pass directories as arguments.
- ❌ Do **not** commit large data files (`.npy`, `.p`, `.pickle`, `.pt`, `.h5`, `.csv` data).
- ❌ Do **not** create a new `utils/` module that duplicates functionality already in an existing one.
- ❌ Do **not** change the excitatory/inhibitory segment ordering in `X` arrays.
- ❌ Do **not** remove the `strict=False` in `tcn_poisson_model.py`'s `load_state_dict` — it is intentional for partial weight loading.
- ❌ Do **not** add symmetric padding to `CausalConv1d`.

---

## 7. Adding New Code

1. **New reusable function?** → Add to the relevant `utils/` module or create a new `utils/<noun>_<noun>.py`.
2. **New pipeline step?** → Create `<N>_<verb>_<noun>[_<backend>].py` at project root; import from `utils/`.
3. **New plot type?** → Add to `utils/visualization_utils.py`, following the `_setup_plot_style` / `_add_statistics_text` helper pattern.
4. **New model variant?** → Sub-class `TCNModel` or create a new class in `utils/fit_CNN_torch.py`; do not fork the whole file.
5. **Touching training loop?** → Preserve the dual-output structure `(y_spike, y_soma)` and the `training_history_dict` key schema.

---

## 8. Key Constants (do not change without updating all callers)

```python
NUM_SEGMENTS_EXC_DEFAULT = 639    # L5PC model default
NUM_SEGMENTS_INH_DEFAULT = 111    # L5PC model default
Y_SOMA_THRESHOLD         = -25    # mV clipping for somatic voltage
SYNAPSE_TYPE             = 'NMDA'
SPIKE_RICH_RATIO_DEFAULT = 0.5    # balanced sampling default
```