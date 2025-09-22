#!/usr/bin/env python3
"""
脚本用于从保存的CLIP模型检查点中移除IBM（Information Bottleneck Module）模块
使用方法:
    python remove_ibm_from_checkpoint.py --input_checkpoint model_with_ibm.pt --output_checkpoint model_without_ibm.pt
    或者
    python remove_ibm_from_checkpoint.py --input_dir /path/to/checkpoint/dir --output_dir /path/to/clean/dir
"""

import argparse
import os
import sys
import torch
from pathlib import Path


def remove_ibm_from_state_dict(state_dict):
    """
    从state_dict中移除IBM相关的参数
    
    Args:
        state_dict: 模型的state_dict
        
    Returns:
        cleaned_state_dict: 移除IBM模块后的state_dict
        removed_keys: 被移除的键列表
    """
    cleaned_state_dict = {}
    removed_keys = []
    
    for key, value in state_dict.items():
        # 检查是否是IBM相关的参数
        if any(ibm_key in key for ibm_key in ['image_ibm', 'text_ibm']):
            removed_keys.append(key)
            print(f"Removing IBM parameter: {key}")
        else:
            cleaned_state_dict[key] = value
    
    return cleaned_state_dict, removed_keys


def process_checkpoint(input_path, output_path):
    """
    处理单个检查点文件
    
    Args:
        input_path: 输入检查点路径
        output_path: 输出检查点路径
    """
    print(f"Processing checkpoint: {input_path}")
    
    try:
        # 加载检查点
        checkpoint = torch.load(input_path, map_location='cpu')
        
        # 检查检查点格式
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                # 训练检查点格式（包含epoch, optimizer等信息）
                print("Detected training checkpoint format")
                original_state_dict = checkpoint['state_dict']
                cleaned_state_dict, removed_keys = remove_ibm_from_state_dict(original_state_dict)
                checkpoint['state_dict'] = cleaned_state_dict
                
                # 更新检查点中的其他信息
                if 'model_cfg' in checkpoint and 'use_ibm' in checkpoint['model_cfg']:
                    checkpoint['model_cfg']['use_ibm'] = False
                    print("Updated model_cfg: set use_ibm to False")
                    
            else:
                # 简单的state_dict格式
                print("Detected simple state_dict format")
                cleaned_state_dict, removed_keys = remove_ibm_from_state_dict(checkpoint)
                checkpoint = cleaned_state_dict
        else:
            print("Warning: Unexpected checkpoint format")
            cleaned_state_dict, removed_keys = remove_ibm_from_state_dict(checkpoint)
            checkpoint = cleaned_state_dict
        
        # 保存清理后的检查点
        torch.save(checkpoint, output_path)
        
        print(f"Cleaned checkpoint saved to: {output_path}")
        print(f"Removed {len(removed_keys)} IBM-related parameters:")
        for key in removed_keys:
            print(f"  - {key}")
        print()
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False


def process_directory(input_dir, output_dir):
    """
    处理目录中的所有.pt文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有.pt文件
    pt_files = list(input_path.glob("*.pt"))
    
    if not pt_files:
        print(f"No .pt files found in {input_dir}")
        return
    
    print(f"Found {len(pt_files)} checkpoint files to process")
    
    success_count = 0
    for pt_file in pt_files:
        input_file = pt_file
        output_file = output_path / pt_file.name
        
        if process_checkpoint(input_file, output_file):
            success_count += 1
    
    print(f"Successfully processed {success_count}/{len(pt_files)} files")


def validate_checkpoint(checkpoint_path):
    """
    验证检查点是否包含IBM模块
    
    Args:
        checkpoint_path: 检查点路径
        
    Returns:
        bool: 是否包含IBM模块
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 获取state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 检查是否有IBM相关的键
        ibm_keys = [key for key in state_dict.keys() if any(ibm_key in key for ibm_key in ['image_ibm', 'text_ibm'])]
        
        if ibm_keys:
            print(f"Found IBM modules in {checkpoint_path}:")
            for key in ibm_keys:
                print(f"  - {key}")
            return True
        else:
            print(f"No IBM modules found in {checkpoint_path}")
            return False
            
    except Exception as e:
        print(f"Error validating {checkpoint_path}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Remove IBM modules from CLIP model checkpoints")
    parser.add_argument('--input_checkpoint', type=str, help='Input checkpoint file path')
    parser.add_argument('--output_checkpoint', type=str, help='Output checkpoint file path')
    parser.add_argument('--input_dir', type=str, help='Input directory containing checkpoint files')
    parser.add_argument('--output_dir', type=str, help='Output directory for cleaned checkpoint files')
    parser.add_argument('--validate_only', action='store_true', help='Only validate if checkpoint contains IBM modules')
    parser.add_argument('--force', action='store_true', help='Overwrite output files if they exist')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.input_checkpoint and args.input_dir:
        print("Error: Cannot specify both --input_checkpoint and --input_dir")
        sys.exit(1)
    
    if not args.input_checkpoint and not args.input_dir:
        print("Error: Must specify either --input_checkpoint or --input_dir")
        sys.exit(1)
    
    if args.validate_only:
        # 仅验证模式
        if args.input_checkpoint:
            validate_checkpoint(args.input_checkpoint)
        elif args.input_dir:
            input_path = Path(args.input_dir)
            pt_files = list(input_path.glob("*.pt"))
            for pt_file in pt_files:
                validate_checkpoint(pt_file)
        return
    
    # 处理单个文件
    if args.input_checkpoint:
        if not args.output_checkpoint:
            args.output_checkpoint = args.input_checkpoint.replace('.pt', '_cleaned.pt')
        
        if not os.path.exists(args.input_checkpoint):
            print(f"Error: Input checkpoint file does not exist: {args.input_checkpoint}")
            sys.exit(1)
        
        if os.path.exists(args.output_checkpoint) and not args.force:
            print(f"Error: Output file already exists: {args.output_checkpoint}")
            print("Use --force to overwrite")
            sys.exit(1)
        
        # 先验证输入文件是否包含IBM模块
        if not validate_checkpoint(args.input_checkpoint):
            print("No IBM modules found in input checkpoint. Nothing to do.")
            return
        
        success = process_checkpoint(args.input_checkpoint, args.output_checkpoint)
        if success:
            print("Successfully removed IBM modules from checkpoint!")
        else:
            print("Failed to process checkpoint!")
            sys.exit(1)
    
    # 处理目录
    elif args.input_dir:
        if not args.output_dir:
            print("Error: --output_dir is required when using --input_dir")
            sys.exit(1)
        
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory does not exist: {args.input_dir}")
            sys.exit(1)
        
        if os.path.exists(args.output_dir) and not args.force:
            print(f"Warning: Output directory already exists: {args.output_dir}")
            response = input("Continue? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(1)
        
        process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
