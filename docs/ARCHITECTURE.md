# ARCHITECTURE.md — Module Responsibilities & Data Flow

## System Overview

This project implements a **single-neuron input-output mapping** pipeline. A biologically realistic L5 pyramidal cell (L5PC) receives synaptic input across hundreds of dendritic segments, and a TCN model learns to predict somatic spike output and membrane voltage from that input.

```
Simulation data (.p files)
        │
        ▼
parse_sim_experiment_file()          ← utils/fit_CNN_torch.py
        │
        ├── dict2bin()               # spike dict → binary matrix
        │
        ▼
SimulationDataGenerator              ← utils/fit_CNN_torch.py
  (balanced sampling, windowing)
        │
        ▼
TCNModel.forward(X)                  ← utils/fit_CNN_torch.py
  CausalConv1d stack
        │
        ├── pred_spike  (B, T, 1)   # Bernoulli probability
        └── pred_soma   (B, T, 1)   # subthreshold voltage
        │
        ▼
Loss: BCE(spike) + MSE(soma)
        │
        ▼
Save: <run_id>.pt + <run_id>.pickle
        │
        ▼
find_best_model()                    ← utils/find_best_model.py
        │
        ▼
calculate_auc_metrics()              ← utils/model_analysis.py
        │
        ▼
TCNPoissonModel (Step 6)             ← utils/tcn_poisson_model.py
  Frozen TCNModel + STE Poisson
  Gradient-based firing rate optimization
```

## Module Dependency Graph

```
3_train_and_analyze_torch.py
    ├── utils/fit_CNN_torch.py       [TCNModel, SimulationDataGenerator, parse_*]
    ├── utils/gpu_monitor.py         [GPUMonitor, configure_pytorch_gpu]
    ├── utils/model_analysis.py      [load_model_results, plot_*, prune_*]
    └── utils/model_size_utils.py    [get_model_size_info, analyze_model_size]

6_activity_optimization.py
    ├── utils/tcn_poisson_model.py   [TCNPoissonModel]
    │       └── utils/fit_CNN_torch.py  [TCNModel]
    ├── utils/find_best_model.py     [find_best_model]
    └── utils/visualization_utils.py [create_optimization_report, plot_average_heatmap]

utils/model_analysis.py
    ├── utils/fit_CNN_torch.py       [TCNModel, parse_sim_experiment_file]
    └── (matplotlib, sklearn)

utils/model_size_utils.py
    ├── (torch)                      optional
    └── (tensorflow)                 optional
```

**Rules:**
- `utils/` modules may import from each other only when the dependency is one-directional (no cycles).
- Top-level numbered scripts import from `utils/`; never the reverse.
- `tests/` imports from both `utils/fit_CNN_torch.py` and `utils/fit_CNN_tf.py` for cross-validation.

## TCNModel Architecture

```
Input: (B, T, C)   where C = num_segments_exc + num_segments_inh

→ Permute to (B, C, T)
→ [CausalConv1d → Activation] × network_depth    (dilation doubles each layer)
→ Permute to (B, T, C_out)
→ Linear → sigmoid   → pred_spike  (B, T, 1)
→ Linear → identity  → pred_soma   (B, T, 1)
```

Key properties:
- **Causal**: no future time leakage (left-only padding in `CausalConv1d`).
- **Dilated**: receptive field grows exponentially with depth.
- Per-layer hyperparameters always stored as lists of length `network_depth`.

## TCNPoissonModel Architecture (Step 6)

```
firing_rates (B, C, T)
    → nan_to_num + clamp ≥ 0
    → torch.poisson(·)                   # forward: discrete samples
    → STE: samples + rates - rates.detach()   # gradient flows through rates
    → first_half / second_half split
    → inject monosynaptic spike at T//2 in second half
    → permute → frozen TCNModel → pred_spike
```

The model produces `batch_size * 2` outputs: first half = baseline, second half = with monosynaptic perturbation.

## Visualization Utilities (`utils/visualization_utils.py`)

All plot functions should:
1. Accept `save_path` (optional, None = display only).
2. Call `_setup_plot_style(ax, ...)` for consistent styling.
3. Call `_add_statistics_text(ax, stats_dict)` for optional stat boxes.
4. Use `plt.savefig(save_path, dpi=300, bbox_inches='tight')` then `plt.close()`.
5. Return the figure object for further manipulation if needed.

Internal helpers (leading underscore):
- `_add_statistics_text` — adds a text box with key-value stats to an axis.
- `_setup_plot_style` — sets title, labels, grid, legend, spine style.
- `_setup_inset_axes` — creates a zoomed inset axes in the lower-right corner.

## File Format Summary

| Extension | Content | Typical size |
|---|---|---|
| `.p` (pickle) | Raw simulation experiment dict | 10–500 MB |
| `.pickle` | Saved model params + training history | <1 MB |
| `.pt` | PyTorch `state_dict` | 1–50 MB |
| `.h5` | Keras/TF weights (legacy) | 1–50 MB |
| `.npy` | NumPy arrays (firing rates, segment info) | varies |
| `.pdf` | Publication-quality figures | <5 MB |
| `.graphml` | Dendritic segment graph (`DiG.graphml`) | <1 MB |
| `.csv` | Segment metadata (`all_segments_*.csv`) | <1 MB |
