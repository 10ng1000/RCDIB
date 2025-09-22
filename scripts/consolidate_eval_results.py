#!/usr/bin/env python3
"""
整合zero-shot-eval文件夹下每个子文件夹的评估结果
提取四个数据集（imagenet_sketch, imagenet-r, imagenet1k, imagenetv2）的acc1和acc5指标
"""

import os
import json
import pandas as pd
from pathlib import Path

def extract_metrics_from_json(file_path):
    """从JSON文件中提取acc1和acc5指标"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metrics = data.get('metrics', {})
        acc1 = metrics.get('acc1', 'N/A')
        acc5 = metrics.get('acc5', 'N/A')
        
        # 将数值乘以100转换为百分比形式
        if isinstance(acc1, (int, float)):
            acc1 = acc1 * 100
        if isinstance(acc5, (int, float)):
            acc5 = acc5 * 100
            
        return {
            'acc1': acc1,
            'acc5': acc5
        }
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {'acc1': 'Error', 'acc5': 'Error'}

def consolidate_folder_results(folder_path):
    """整合一个文件夹内的所有数据集结果"""
    datasets = ['imagenet_sketch', 'imagenet-r', 'imagenet1k', 'imagenetv2']
    results = {}
    
    for dataset in datasets:
        # 查找包含dataset名称的JSON文件
        json_files = list(Path(folder_path).glob(f"{dataset}*.json"))
        
        if json_files:
            # 如果找到文件，取第一个（通常每个数据集只有一个文件）
            json_file = json_files[0]
            metrics = extract_metrics_from_json(json_file)
            results[dataset] = metrics
        else:
            # 如果没有找到文件，标记为缺失
            results[dataset] = {'acc1': 'Missing', 'acc5': 'Missing'}
    
    return results

def main():
    zero_shot_eval_path = Path('/home/h3c/lyl/open_clip/scripts/zero-shot-eval')
    
    if not zero_shot_eval_path.exists():
        print(f"Error: Path {zero_shot_eval_path} does not exist!")
        return
    
    all_results = {}
    
    # 遍历zero-shot-eval下的所有子文件夹
    for item in zero_shot_eval_path.iterdir():
        if item.is_dir():
            folder_name = item.name
            print(f"Processing folder: {folder_name}")
            
            # 整合该文件夹的结果
            folder_results = consolidate_folder_results(item)
            all_results[folder_name] = folder_results
    
    # 创建结果表格
    rows = []
    for folder_name, folder_data in all_results.items():
        for dataset, metrics in folder_data.items():
            rows.append({
                'Folder': folder_name,
                'Dataset': dataset,
                'ACC1': metrics['acc1'],
                'ACC5': metrics['acc5']
            })
    
    # 创建DataFrame
    df = pd.DataFrame(rows)
    
    # 按文件夹和数据集排序
    df = df.sort_values(['Folder', 'Dataset'])
    
    # 保存为CSV文件
    output_file = zero_shot_eval_path / 'consolidated_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # 也创建一个更易读的格式（每个文件夹为一行）
    pivot_rows = []
    for folder_name, folder_data in all_results.items():
        row = {'Folder': folder_name}
        for dataset in ['imagenet_sketch', 'imagenet-r', 'imagenet1k', 'imagenetv2']:
            if dataset in folder_data:
                row[f'{dataset}_acc1'] = folder_data[dataset]['acc1']
                row[f'{dataset}_acc5'] = folder_data[dataset]['acc5']
            else:
                row[f'{dataset}_acc1'] = 'Missing'
                row[f'{dataset}_acc5'] = 'Missing'
        pivot_rows.append(row)
    
    pivot_df = pd.DataFrame(pivot_rows)
    pivot_df = pivot_df.sort_values('Folder')
    
    # 保存透视表格式
    pivot_output_file = zero_shot_eval_path / 'consolidated_results_pivot.csv'
    pivot_df.to_csv(pivot_output_file, index=False)
    print(f"Pivot table saved to: {pivot_output_file}")
    
    # 显示结果预览
    print("\n=== Results Preview ===")
    print("Long format (first 10 rows):")
    print(df.head(10).to_string(index=False))
    
    print("\nPivot format (first 5 rows):")
    # 只显示前几列以避免输出太宽
    preview_cols = ['Folder', 'imagenet1k_acc1', 'imagenet1k_acc5', 'imagenet_sketch_acc1', 'imagenet_sketch_acc5']
    available_cols = [col for col in preview_cols if col in pivot_df.columns]
    print(pivot_df[available_cols].head().to_string(index=False))

if __name__ == "__main__":
    main()
