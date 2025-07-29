#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import pandas as pd
import tyro
from pathlib import Path
from typing import List

# 这个辅助函数与之前完全相同，无需更改。
def calculate_similarity_score(
    query_chunk: np.ndarray, all_actions: List[np.ndarray]
) -> float:
    """
    为给定的动作序列片段(query_chunk)计算相似度分数。

    该分数通过在数据集中找到最接近的两个匹配项来计算，其定义为：
    score = 1 - (到最近匹配项的距离 / 最近两个匹配项之间的距离)

    Args:
        query_chunk (np.ndarray): 用于查询的动作序列片段。
        all_actions (List[np.ndarray]): 数据集中所有动作轨迹的列表。

    Returns:
        float: 计算出的相似度分数。如果在数据集中找不到至少两个匹配项，则返回 np.nan。
    """
    action_horizon = query_chunk.shape[0]
    best_matches = []
    for i, trajectory in enumerate(all_actions):
        if trajectory.shape[0] < action_horizon:
            continue
        best_dist_in_traj = np.inf
        best_start_idx_in_traj = -1
        for j in range(trajectory.shape[0] - action_horizon + 1):
            window = trajectory[j : j + action_horizon]
            dist = np.linalg.norm(query_chunk - window)
            if dist < best_dist_in_traj:
                best_dist_in_traj = dist
                best_start_idx_in_traj = j
        if best_start_idx_in_traj != -1:
            best_matches.append((best_dist_in_traj, i, best_start_idx_in_traj))

    if len(best_matches) < 2:
        return np.nan

    best_matches.sort(key=lambda x: x[0])
    dist_to_closest = best_matches[0][0]
    closest_traj_idx, closest_start_idx = best_matches[0][1], best_matches[0][2]
    second_closest_traj_idx, second_closest_start_idx = best_matches[1][1], best_matches[1][2]
    closest_action_seq = all_actions[closest_traj_idx][closest_start_idx : closest_start_idx + action_horizon]
    second_closest_action_seq = all_actions[second_closest_traj_idx][second_closest_start_idx : second_closest_start_idx + action_horizon]
    dist_between_top_two = np.linalg.norm(second_closest_action_seq - closest_action_seq)

    if dist_between_top_two == 0:
        return 1.0
    return 1 - (dist_to_closest / dist_between_top_two)


def main(
    h5_path: Path,
    query_action_dir: Path,
    action_horizon: int = 10,
    overwrite: bool = False,
):
    """
    针对数据集计算多个查询动作轨迹的相似度分数。

    该脚本会从 'query_action_dir' 开始，递归地查找所有名为 'trajs' 的子目录，
    并处理其中的每一个 .npz 文件。对于每个文件，它会以 'action_horizon' 大小的
    非重叠块处理轨迹，并计算相似度分数。结果会保存到与原 .npz 文件同目录的 .csv 文件中。

    Args:
        h5_path: 包含动作轨迹数据集的HDF5文件的路径。
                 文件应包含如 'traj_0', 'traj_1' 等数据集，每个数据集都有一个 'actions' 字段。
        query_action_dir: 开始搜索的根目录。脚本将递归查找所有 'trajs' 子目录
                          并处理其中的 .npz 文件。
        action_horizon: 用于比较的动作序列窗口的长度（块大小）。
    """
    # --- 1. 一次性加载共享的 HDF5 数据 ---
    print(f"🔄 正在从以下位置加载动作轨迹: {h5_path}")
    all_actions = []
    try:
        with h5py.File(h5_path, 'r') as h5_file:
            sorted_keys = sorted(h5_file.keys(), key=lambda k: int(k.split('_')[-1]))
            for key in sorted_keys:
                if 'actions' in h5_file[key]:
                    all_actions.append(h5_file[key]['actions'][:])
    except Exception as e:
        print(f"❌ 加载HDF5文件时出错: {e}")
        return
    
    if not all_actions:
        print("❌ HDF5 文件中未找到任何动作轨迹。")
        return
    print(f"✅ 已从 HDF5 加载 {len(all_actions)} 条轨迹。")

    # --- 2. 查找并处理每个查询文件 ---
    print(f"🔍 正在 '{query_action_dir}' 目录下递归搜索匹配 '**/trajs/*.npz' 的文件...")
    query_files = sorted(list(query_action_dir.glob('**/trajs/*.npz')))
    
    if not query_files:
        print(f"❌ 未找到任何匹配 '**/trajs/*.npz' 模式的 .npz 文件。")
        return
    
    print(f"✅ 找到 {len(query_files)} 个查询文件待处理。")
    
    # 循环处理每个找到的 .npz 文件
    for i, query_action_path in enumerate(query_files):
        print("\n" + "="*60)
        # 使用相对路径打印，使输出更简洁
        relative_path = query_action_path.relative_to(query_action_dir)
        print(f"▶️  处理文件 {i+1}/{len(query_files)}: {relative_path}")
        print("="*60)
        
        # -- 2x. 检查 scores.csv 是否存在 --
        scores_csv_path = query_action_path.with_suffix('.csv')
        if scores_csv_path.exists() and not overwrite:
            print(f"⚠️  跳过已存在的文件: {scores_csv_path.relative_to(query_action_dir)}")
            continue

        # -- 2a. 加载单个查询动作 --
        print("🔄 正在加载查询动作...")
        try:
            query_action_full = np.load(query_action_path)['action']
        except Exception as e:
            print(f"❌ 加载查询文件 '{query_action_path.name}' 时出错: {e}")
            continue

        print(f"✅ 查询动作已加载，形状为: {query_action_full.shape}")

        if query_action_full.shape[0] < action_horizon:
            print(f"⚠️  警告: 查询动作长度 ({query_action_full.shape[0]}) 小于 action_horizon ({action_horizon})。已跳过此文件。")
            continue

        # -- 2b. 计算分数 --
        print(f"⚙️  正在计算分数 (action horizon: {action_horizon})...")
        results_data = []
        for j in range(0, query_action_full.shape[0] - action_horizon + 1, action_horizon):
            query_chunk = query_action_full[j : j + action_horizon]
            score = calculate_similarity_score(query_chunk, all_actions)
            results_data.append({'timestamp': j, 'score': score})

        print(f"✅ 已完成对 {relative_path} 的计算。")
        
        # -- 2c. 保存结果 --
        output_path = query_action_path.with_suffix('.csv')
        
        print(f"💾 正在将结果保存到: {output_path.relative_to(query_action_dir)}")
        try:
            df = pd.DataFrame(results_data, columns=['timestamp', 'score'])
            df.to_csv(output_path, index=False)
        except Exception as e:
            print(f"❌ 保存CSV文件时出错: {e}")
    
    print("\n✨ 所有文件处理完毕！")


if __name__ == "__main__":
    tyro.cli(main)