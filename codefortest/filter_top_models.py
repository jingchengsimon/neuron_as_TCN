#!/usr/bin/env python3
"""
筛选并可选删除模型脚本。

注意：默认不执行文件删除，仅显示将要删除的文件列表。使用 --delete 开关来执行实际删除操作。
"""

import os
import argparse
import glob
import pickle


def main():
    parser = argparse.ArgumentParser(description='Filter top-N models by val_spikes_loss (min) and optionally delete others.')
    parser.add_argument('--models-dir', required=True, help='目录，包含模型的 .pickle 文件（以及 .pt 权重）')
    parser.add_argument('--keep', type=int, default=10, help='保留前 N 个模型（默认 10）')
    parser.add_argument('--dry-run', action='store_true', help='仅打印将要删除的文件，不实际删除')
    parser.add_argument('--delete', action='store_true', help='执行删除操作（危险操作，慎用）')
    args = parser.parse_args()

    models_dir = args.models_dir
    keep_n = args.keep

    print(f"Scanning .pickle files in: {models_dir}")
    pickle_files = sorted(glob.glob(os.path.join(models_dir, '*.pickle')))
    if not pickle_files:
        print(f"No .pickle model files found in {models_dir}")
        return

    # Read minimal metadata from pickles (fast)
    model_infos = []
    for pkl in pickle_files:
        try:
            with open(pkl, 'rb') as f:
                data = pickle.load(f)
            training_history = data.get('training_history_dict') or data.get('training_history') or {}
            val_spikes = training_history.get('val_spikes_loss', [])
            min_val_spikes = min(val_spikes) if val_spikes else float('inf')
            model_infos.append({'model_path': pkl, 'min_val_spikes_loss': min_val_spikes})
        except Exception as e:
            print(f"Failed to read {pkl}: {e}")

    if not model_infos:
        print('No readable model pickles found.')
        return

    # Sort ascending by min val_spikes_loss and select top-N to keep
    model_infos_sorted = sorted(model_infos, key=lambda x: x['min_val_spikes_loss'])
    keep_models = model_infos_sorted[:keep_n]
    keep_paths = set(m['model_path'] for m in keep_models)

    # Prepare delete list (all pickles not in keep + corresponding .pt/.h5 files)
    files_to_delete = []
    for m in model_infos_sorted[keep_n:]:
        pkl = m['model_path']
        files_to_delete.append(pkl)
        pt = pkl.replace('.pickle', '.pt')
        h5 = pkl.replace('.pickle', '.h5')
        if os.path.exists(pt):
            files_to_delete.append(pt)
        elif os.path.exists(h5):
            files_to_delete.append(h5)

    print(f"Selected top {len(keep_models)} models by min(val_spikes_loss).")
    for i, km in enumerate(keep_models):
        print(f"  {i+1}. {os.path.basename(km['model_path'])}  min_val_spikes_loss={km['min_val_spikes_loss']}")

    if not files_to_delete:
        print('No files to delete (all models are within the top-N).')
        return

    print(f"Found {len(files_to_delete)} files that would be deleted (including .pt/.h5 weight files).")

    if args.dry_run or not args.delete:
        print('Dry-run or --delete not given: the following files would be deleted:')
        for f in files_to_delete:
            print(f"  {f}")
        print('To actually delete files, re-run with --delete (and without --dry-run).')
        return

    # Execute deletions
    deleted = []
    failed = []
    for f in files_to_delete:
        try:
            if os.path.exists(f):
                os.remove(f)
                deleted.append(f)
            else:
                failed.append((f, 'not found'))
        except Exception as e:
            failed.append((f, str(e)))

    print(f"Deleted {len(deleted)} files.")
    if failed:
        print(f"Failed to delete {len(failed)} files:")
        for f, err in failed:
            print(f"  {f}: {err}")


if __name__ == '__main__':
    main()
