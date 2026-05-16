# Agent Task: Feature Representation Diagnostic Script

## Background (read before writing any code)

This project trains a TCN model (`TCNModel` in `utils/fit_CNN_torch.py`) to predict
single-neuron spike output from synaptic input across dendritic segments of an L5PC
biophysical model. The existing pipeline compresses ~26k synapses into 639 exc + 639 inh
segment-level channels by **averaging spike counts per segment per ms**, then trains with
standard BCE loss.

**Hypothesis being tested:** The per-segment mean is destroying the discriminative signal.
Specifically, because each segment receives ~40 exc synapses, the mean is never zero even
during low-activity periods, whereas the original reference implementation (Beniaguev et al.)
has exactly 1 synapse per segment and therefore clean sparse binary inputs.

The diagnostic compares **three input representations** on the same held-out trials to
determine whether the bottleneck is in the feature extraction step.

## Data Format (critical — read carefully)

Processed `.p` files are loaded via `parse_sim_experiment_file()` in `utils/fit_CNN_torch.py`.
Each file contains a dict with key `'Results'` → `'listOfSingleSimulationDicts'`, where each
trial dict has:
- `'exInputSpikeTimes'`  : `dict[segment_index → list[int]]`  (ms, 0-based, 639 segments)
- `'inhInputSpikeTimes'` : `dict[segment_index → list[int]]`  (ms, 0-based, 639 segments)
- `'outputSpikeTimes'`   : `np.ndarray` of int ms timestamps
- `'somaVoltageLowRes'`  : `np.ndarray` shape `(sim_duration_ms,)`

Raw experiment folders (pre-pipeline) each contain:
- `section_synapse_df.csv` — per-synapse metadata including `type` ('A'=exc, 'B'=inh),
  `section_synapse` (section name), `loc` (0–1 position), `spike_train_bg` (spike times list)
- `soma_v_array.npy`
- `simulation_params.json`

Segment metadata is in `all_segments_noaxon.csv` at project root.
Segment ordering: indices `0 … 638` = exc, `639 … 749` (or similar) = inh.
`num_segments_exc = 639`, `num_segments_inh` determined from data.

## Task

Create `0_feature_diagnostic.py` at project root. This script is **self-contained and
read-only** with respect to existing data — it writes results only to
`./results/0_feature_diagnostic/`.

### What the script must do

**Step 1 — Load a small held-out subset**

Load up to `N_TRIALS = 100` trials from the test `.p` files (directory passed as CLI arg
`--test_dir`). Use the existing `parse_sim_experiment_file()` function. Do not re-implement
parsing.

**Step 2 — Build three input representations from the same trials**

All three must produce arrays of shape `(num_segments_exc + num_segments_inh, sim_duration_ms)`,
i.e. they are drop-in replacements for the existing `bin_spikes_matrix` fed to `TCNModel`.

- **Rep A — "mean" (current baseline)**
  For each segment, count how many synapses fired in each 1 ms bin, divide by total
  synapses mapped to that segment. This reproduces the existing `2_dataset_pipeline.py`
  behaviour. Use `exInputSpikeTimes` / `inhInputSpikeTimes` directly from the parsed dict.
  Since the parsed `.p` files already contain the aggregated dicts (one entry per segment,
  values = merged spike times from all synapses on that segment), the "mean" here is:
  `spike_count_in_bin / num_synapses_on_segment`. If `num_synapses_on_segment` is not
  stored, fall back to using raw binary (treat each segment as having 1 synapse) — document
  this fallback clearly with a `# FALLBACK` comment.

- **Rep B — "first-synapse binary" (sparse proxy)**
  For each segment, treat the aggregated spike list as if it came from a single synapse:
  create a binary `(0/1)` spike train where `bin[t] = 1` if any spike time falls in ms `t`.
  This approximates what the reference implementation sees (1 synapse per segment, binary).
  Use `np.clip(..., 0, 1)` on the binned count. Do not modify `parse_sim_experiment_file`.

- **Rep C — "coincidence" (instantaneous count, normalised)**
  For each segment, compute the raw count of spikes in each 1 ms bin (not divided by
  anything). Then apply a causal exponential smoothing kernel with `tau_ms = 3`
  (mimicking NMDA integration):
  ```
  kernel[t] = exp(-t / tau_ms),  t = 0 … 4*tau_ms
  kernel /= kernel.sum()
  smoothed = np.convolve(raw_count, kernel, mode='full')[:sim_duration_ms]
  ```
  This preserves the coincidence information (10 synapses firing simultaneously vs 1 = 10×
  count, not averaged away) while remaining causal.

**Step 3 — Train a small TCN on each representation**

For each of the three representations, train a **small fixed TCN** (do not search
hyperparameters):
```
network_depth = 3
num_filters    = 32
input_window_size = 200  # ms
num_epochs     = 20
batch_size     = 64
```

Use the existing `TCNModel` and `SimulationDataGenerator` from `utils/fit_CNN_torch.py`.
Split the 100 trials: 70 train, 15 val, 15 test (fixed split, seed=42).

Training must use the **existing BCE loss** (do not introduce Focal Loss here — this
diagnostic isolates the feature variable only).

Save the best checkpoint (by `val_spikes_loss`) for each representation as
`results/0_feature_diagnostic/model_rep_{A|B|C}.pt`.

**Step 4 — Evaluate and compare**

On the 15 held-out test trials, compute for each representation:

1. `roc_auc` — ROC AUC on all time bins (use `sklearn.metrics.roc_auc_score`)
2. `pr_auc`  — Precision-Recall AUC (use `sklearn.metrics.average_precision_score`)
3. `vp_dist` — Mean Victor-Purpura distance between predicted and true spike trains
   - Decode predicted spike train using `scipy.signal.find_peaks` with
     `height=0.5, distance=3` (3 ms refractory)
   - Use `cost_per_ms = 1.0` in VP distance
   - VP distance implementation must be self-contained in this script (do not add to utils)
4. `mean_input_sparsity` — fraction of zero-valued bins in the input array, averaged over
   trials and segments (diagnostic: should be highest for Rep B)

Print a summary table to stdout and save to
`results/0_feature_diagnostic/diagnostic_summary.csv`.

**Step 5 — Visualisation**

Generate a single 2×2 figure saved as
`results/0_feature_diagnostic/diagnostic_figure.pdf`:
- Top-left:  ROC AUC bar chart (A, B, C)
- Top-right: PR AUC bar chart
- Bottom-left: VP distance bar chart (lower = better timing precision)
- Bottom-right: example predicted vs true spike train for 1 trial, 500ms window,
  for all 3 representations overlaid (use `utils/visualization_utils._setup_plot_style`)

## Constraints (from AGENT.md — apply all)

- Tensor shape: always comment `# (B, T, C)` on creation/transpose.
- Torch tensors from numpy: use `torch.from_numpy(arr.astype(np.float32))`.
- Device: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`.
- No hardcoded absolute paths. All directories via CLI args with `argparse`.
- `os.makedirs(save_dir, exist_ok=True)` everywhere.
- Do NOT modify any file in `utils/`. Read-only.
- Do NOT modify any `.p` data files.
- `visualization_utils_backup.py` — do not touch.
- New module not needed — keep VP distance implementation local to this script.
- All outputs go to `./results/0_feature_diagnostic/`.

## CLI Interface

```
python 0_feature_diagnostic.py \
    --test_dir   /path/to/test_data/   \
    --n_trials   100                   \
    --n_epochs   20                    \
    --seed       42
```

All arguments have defaults so the script runs without any flags if data is in the
standard location expected by `3_train_and_analyze_torch.py`.

## Expected Output

```
results/0_feature_diagnostic/
├── model_rep_A.pt
├── model_rep_B.pt
├── model_rep_C.pt
├── diagnostic_summary.csv      ← columns: rep, roc_auc, pr_auc, vp_dist, mean_input_sparsity
└── diagnostic_figure.pdf
```

## What NOT to implement

- Do not implement `argparse` subcommands.
- Do not implement distributed training.
- Do not re-implement `parse_sim_experiment_file` — import and call it.
- Do not add any new keys to the `.p` file format.
- Do not run `2_dataset_pipeline.py` as a subprocess.
- Do not implement Rep A by going back to raw `section_synapse_df.csv` — use the already-
  processed `.p` files which contain aggregated per-segment spike dicts.
