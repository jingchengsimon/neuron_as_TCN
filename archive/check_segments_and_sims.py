#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按fit_CNN的读取方式，检查每个.p文件中的：
- num_segments_exc（兴奋性段数量）
- num_segments_inh（抑制性段数量）
- num_simulations（模拟条目数量）
并输出一致性报告与异常文件列表。
"""

import os
import pickle
import glob
from collections import defaultdict, Counter

import numpy as np


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def load_experiment(file_path):
    try:
        with open(file_path, 'rb') as f:
            exp = pickle.load(f)
        return exp
    except Exception as e:
        print(f"  加载失败: {file_path} -> {e}")
        return None


def extract_segments_info(file_path):
    exp = load_experiment(file_path)
    if exp is None:
        return None

    try:
        sims = exp['Results']['listOfSingleSimulationDicts']
        num_simulations = len(sims)

        per_sim_segments = []  # [(exc_count, inh_count), ...]

        for idx, sim in enumerate(sims):
            ex_count = len(sim['exInputSpikeTimes'])
            inh_count = len(sim['inhInputSpikeTimes'])
            per_sim_segments.append((ex_count, inh_count))

        # 获取第一个模拟的段数量
        if per_sim_segments:
            first_exc = per_sim_segments[0][0]  # 第一个模拟的兴奋性段数
            first_inh = per_sim_segments[0][1]  # 第一个模拟的抑制性段数
        else:
            first_exc = None
            first_inh = None

        # 从Params中获取总段数
        try:
            num_segments = len(exp['Params']['allSegmentsType'])
        except (KeyError, TypeError):
            num_segments = None

        return {
            'file': file_path,
            'num_simulations': num_simulations,
            'first_sim_exc': first_exc,
            'first_sim_inh': first_inh,
            'num_segments': num_segments,
            'per_sim_segments': per_sim_segments,
        }
    except Exception as e:
        print(f"  解析失败: {file_path} -> {e}")
        return None


def check_segments_and_sims(base_dir):
    # 创建输出文件
    output_file = "segments_check_output.txt"
    output_lines = []
    
    def print_and_save(text):
        """同时打印到控制台和保存到输出列表"""
        print(text)
        output_lines.append(text)
    
    print_and_save("=" * 90)
    print_and_save("检查 .p 文件中的段数量与模拟数量（按fit_CNN读取逻辑）")
    print_and_save("=" * 90)

    files = []
    for root, _, fnames in os.walk(base_dir):
        for n in fnames:
            if n.endswith('.p'):
                files.append(os.path.join(root, n))

    print_and_save(f"找到 .p 文件: {len(files)} 个\n")
    if not files:
        return

    results = []
    mismatched_files = []
    for i, f in enumerate(sorted(files)[:300]):
        info = extract_segments_info(f)
        if info:
            results.append(info)
            # 检查是否符合要求：文件中的模拟数为128，且第一个模拟的exc和inh段数量均为639
            num_segments_str = f"total_segments={info.get('num_segments', 'N/A')}"
            first_exc = info.get('first_sim_exc', 'N/A')
            first_inh = info.get('first_sim_inh', 'N/A')
            first_sim_str = f"first_sim(exc={first_exc},inh={first_inh})"
            
            # 检查第一个模拟的段数是否为639
            first_sim_correct = (first_exc == 639 and first_inh == 639)
            
            if (info['num_simulations'] != 128) or not first_sim_correct:
                mismatched_files.append(info)
                # 单行输出：文件名 | sims | total_segments | first_sim段数
                print_and_save(f"[{i+1}/{len(files)}] {os.path.basename(f)} | sims={info['num_simulations']} | {num_segments_str} | {first_sim_str}")
            else:
                # 符合要求的文件也打印基本信息
                print_and_save(f"[{i+1}/{len(files)}] {os.path.basename(f)} | sims={info['num_simulations']} | {num_segments_str} | {first_sim_str} | ✓ 符合要求")

    if not results:
        print_and_save("未成功解析任何文件。")
        return

    # 统计不符合要求的文件
    total_files = len(results)
    mismatched_count = len(mismatched_files)
    
    print_and_save(f"\n=== 检查结果 ===")
    print_and_save(f"总文件数: {total_files}")
    print_and_save(f"符合要求 (sims=128, first_sim_exc=639, first_sim_inh=639): {total_files - mismatched_count}")
    print_and_save(f"不符合要求: {mismatched_count}")
    
    if mismatched_count == 0:
        print_and_save("\n✓ 所有文件都符合要求，可以安全进行训练！")
    else:
        print_and_save(f"\n✗ 发现 {mismatched_count} 个不符合要求的文件，请检查或排除后再训练。")
        print_and_save("\n不符合要求的文件详情:")
        for info in mismatched_files:
            filename = os.path.basename(info['file'])
            sims = info['num_simulations']
            first_exc = info.get('first_sim_exc')
            first_inh = info.get('first_sim_inh')
            issues = []
            if sims != 128:
                issues.append(f"sims={sims}(应为128)")
            if first_exc != 639:
                issues.append(f"first_sim_exc={first_exc}(应为639)")
            if first_inh != 639:
                issues.append(f"first_sim_inh={first_inh}(应为639)")
            print_and_save(f"  {filename} | 问题: {', '.join(issues)}")
    
    # 保存输出到文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in output_lines:
                f.write(line + '\n')
        print_and_save(f"\n=== 输出已保存到文件: {output_file} ===")
    except Exception as e:
        print_and_save(f"\n警告: 无法保存输出文件: {e}")


def main():
    base_dir = '/G/results/aim2_sjc/Models_TCN/Single_Neuron_InOut/data'
    print(f"检查目录: {base_dir}")
    if not os.path.exists(base_dir):
        print(f"目录不存在: {base_dir}")
        return
    check_segments_and_sims(base_dir)


if __name__ == '__main__':
    main() 