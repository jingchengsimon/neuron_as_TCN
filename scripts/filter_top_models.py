#!/usr/bin/env python3
"""
筛选并可选删除模型脚本。

用途：针对一个包含多个训练结果（*.pickle）的目录，计算每个模型的 AUC（使用 utils.model_analysis 中的函数），
按 ROC AUC 排序并保留前 N 个模型；将不是前 N 的模型（包含对应的 .pickle 与 .pt 权重文件）删除（可选）。

示例：
    python scripts/filter_top_models.py --models-dir /G/results/aim2_sjc/Models_TCN \
        --test-data-dir /G/results/aim2_sjc/TestData --keep 10 --dry-run

注意：默认不执行文件删除，仅显示将要删除的文件列表。使用 --delete 开关来执行实际删除操作。
"""

import os
import argparse
from utils.model_analysis import load_model_results, prune_results_keep_top


def main():
    parser = argparse.ArgumentParser(description='Filter top-N models by ROC AUC and optionally delete others.')
    parser.add_argument('--models-dir', required=True, help='目录，包含模型的 .pickle 文件（以及 .pt 权重）')
    parser.add_argument('--test-data-dir', required=True, help='用于计算 AUC 的测试数据目录')
    parser.add_argument('--keep', type=int, default=10, help='保留前 N 个模型（默认 10）')
    parser.add_argument('--dry-run', action='store_true', help='仅打印将要删除的文件，不实际删除')
    parser.add_argument('--delete', action='store_true', help='执行删除操作（危险操作，慎用）')
    args = parser.parse_args()

    models_dir = args.models_dir
    test_data_dir = args.test_data_dir
    keep_n = args.keep

    print(f"Loading models from: {models_dir}")
    results = load_model_results(models_dir, test_data_dir)

    if not results:
        print("No model files found or failed to load any models.")
        return

    # 确保所有模型都计算出 AUC，以便比较
    results_with_auc = [r for r in results if r.get('auc_metrics') is not None]
    if not results_with_auc:
        print("No models with AUC metrics were found. Aborting.")
        return

    # 通过 prune 函数获得要保留的前 keep_n 个模型（内存层面）
    keep_results = prune_results_keep_top(results, top_n=keep_n, delete_files=False)
    keep_paths = set(r['model_path'] for r in keep_results)

    # 准备删除列表（.pickle 以及对应 .pt 权重文件）
    files_to_delete = []
    for r in results:
        pkl = r['model_path']
        if pkl not in keep_paths:
            pt = pkl.replace('.pickle', '.pt')
            files_to_delete.append(pkl)
            if os.path.exists(pt):
                files_to_delete.append(pt)

    if not files_to_delete:
        print("No files to delete (all models are in the top-N set).")
        return

    print(f"Found {len(files_to_delete)} files that would be deleted (including .pt files).")

    # dry-run: 只打印，不执行
    if args.dry_run or not args.delete:
        print("Dry-run mode or --delete not given: the following files would be deleted:")
        for f in files_to_delete:
            print(f"  {f}")
        print("To actually delete files, re-run with --delete (and without --dry-run).")
        return

    # 执行删除
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
