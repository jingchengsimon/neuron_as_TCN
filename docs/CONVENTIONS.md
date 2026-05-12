# CONVENTIONS.md — Naming & Abbreviation Reference

## Abbreviation Table

| Abbreviation | Full meaning | Context |
|---|---|---|
| `exc` | excitatory | segment type, e.g. `num_segments_exc` |
| `inh` | inhibitory | segment type, e.g. `num_segments_inh` |
| `ms` | milliseconds | always appended to time variables |
| `sim` | simulation | e.g. `sim_duration_ms`, `sim_experiment_file` |
| `seg` | segment | e.g. `monoconn_seg_indices` |
| `bin` | binary | e.g. `bin_spikes_matrix` |
| `pred` | prediction/predicted | e.g. `pred_spike_t`, `y_pred_spike_flat` |
| `val` | validation | e.g. `val_spikes_loss`, `valid_data_dir` |
| `lr` | learning rate | e.g. `init_lr`, `max_lr` |
| `auc` | area under curve | e.g. `roc_auc_spike`, `pr_auc_spike` |
| `tcn` | temporal convolutional network | model class prefix |
| `ste` | straight-through estimator | gradient trick in `tcn_poisson_model.py` |
| `sjc` | dataset variant name | affects key indexing in `dict2bin` |
| `mono` / `monoconn` | monosynaptic connection | fixed-spike stimulation protocol |
| `B` | batch size | tensor shape comment |
| `T` | time steps | tensor shape comment |
| `C` | channels = segments | tensor shape comment |
| `N` | number of simulations | tensor shape comment |
| `_t` | torch.Tensor (suffix) | e.g. `X_batch_t` |
| `_flat` | flattened 1D array | e.g. `y_spike_flat` |
| `_dir` | directory path | e.g. `train_data_dir`, `save_dir` |
| `_path` | file path | e.g. `model_path`, `pickle_path` |

## Magic Numbers — Use Named Constants

These values recur throughout the codebase. Always reference by name, never hardcode:

```python
NUM_SEGMENTS_EXC_DEFAULT = 639    # L5PC NMDA model, excitatory
NUM_SEGMENTS_INH_DEFAULT = 111    # L5PC NMDA model, inhibitory
Y_SOMA_THRESHOLD         = -25    # mV — clipping for subthreshold voltage
SYNAPSE_TYPE             = 'NMDA'
SPIKE_RICH_RATIO_DEFAULT = 0.5
WARMUP_EPOCHS            = 10
LR_DECAY_RATE            = 0.95
```

## Dict Key Schema

### `model_params` (loaded from `.pickle`)
```python
{
  'architecture_dict': {
      'input_window_size': int,
      'network_depth': int,
      'filter_sizes_per_layer': list[int],
      'num_filters_per_layer': list[int],
      'activation_function_per_layer': list[str],
      'strides_per_layer': list[int],
      'dilation_rates_per_layer': list[int],
      'initializer_per_layer': list[str],
      'l2_regularization_per_layer': list[float],
  },
  'training_history_dict': {
      'val_spikes_loss': list[float],
      'val_somatic_loss': list[float],
      'val_loss': list[float],
      # ... other keys
  }
}
```

### `sim_experiment_file` parsed dict (experiment_dict)
```python
{
  'Params': { 'totalSimDurationInSec': float, ... },
  'Results': {
      'listOfSingleSimulationDicts': [
          {
              'exInputSpikeTimes': dict,   # key: segment_index (1-based in SJC)
              'inhInputSpikeTimes': dict,
              'outputSpikeTimes': np.ndarray,
              'somaVoltageLowRes': np.ndarray,
          }, ...
      ]
  }
}
```
