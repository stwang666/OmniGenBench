#!/usr/bin/env python3
"""
独立脚本：预计算所有样本的结构信息并保存到文件
可以在训练前单独运行，避免训练时内存不足
"""

import os
import sys
import pickle
import pandas as pd
import torch
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))
from rna_pyg import rna_structure_to_graph_with_edge_types
from data.wrapper import preprocess_item

def compute_structure_info_worker(args):
    """工作函数：计算单个样本的结构信息"""
    sequence, structure, max_spatial_pos = args
    
    if structure is None or pd.isna(structure):
        return None
    
    cache_key = f"{sequence}_{structure}"
    
    try:
        # 1. 将RNA转换为PyG图
        graph = rna_structure_to_graph_with_edge_types(sequence, structure)
        
        # 2. 使用Graphormer的preprocess_item计算结构信息
        item = graph
        item.idx = 0
        
        # 3. 调用preprocess_item
        processed_item = preprocess_item(item)
        
        # 4. 提取结构信息
        structure_info = {
            'spatial_pos': processed_item.spatial_pos,  # (N, N)
            'edge_input': processed_item.edge_input,    # (N, N, max_dist, edge_dim)
            'attn_edge_type': processed_item.attn_edge_type,  # (N, N, edge_dim)
            'attn_bias': processed_item.attn_bias,      # (N+1, N+1)
            'x': processed_item.x,                      # (N, 1)
        }
        
        # 限制spatial_pos的最大值
        structure_info['spatial_pos'] = torch.clamp(
            structure_info['spatial_pos'], 
            max=max_spatial_pos
        )
        
        return (cache_key, structure_info)
        
    except Exception as e:
        print(f"Error computing structure for {sequence[:20]}...: {e}")
        return None

def precompute_structures(csv_file, output_file, max_spatial_pos=32, num_workers=None, batch_size=10000, overwrite=False):
    """
    预计算所有样本的结构信息（分批处理以避免OOM）
    
    Args:
        csv_file: 输入CSV文件路径
        output_file: 输出pickle文件路径
        max_spatial_pos: 最大空间位置
        num_workers: 进程数，如果为None则自动检测
        batch_size: 每批处理的样本数（默认10000，减少可降低内存占用）
        overwrite: 如果输出文件已存在，是否覆盖（默认False，跳过）
    """
    # 检查输出文件是否已存在
    if os.path.exists(output_file) and not overwrite:
        print(f"⚠️ 输出文件已存在: {output_file}")
        print(f"   跳过预计算（如需重新计算，请使用 --overwrite 参数）")
        return
    
    print(f"读取数据文件: {csv_file}")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"输入文件不存在: {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    # 检测序列和结构字段名
    seq_col = None
    struct_col = None
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['seq', 'sequence', 'text', 'dna', 'rna']:
            seq_col = col
        if col_lower in ['structure', 'ss', 'struct']:
            struct_col = col
    
    if seq_col is None:
        raise ValueError("未找到序列字段")
    if struct_col is None:
        print("警告：未找到结构字段，将跳过结构信息计算")
        return
    
    print(f"序列字段: {seq_col}")
    print(f"结构字段: {struct_col}")
    print(f"总样本数: {len(df)}")
    
    # 收集需要计算的样本
    tasks = []
    for idx, row in df.iterrows():
        sequence = str(row[seq_col]) if pd.notna(row[seq_col]) else None
        structure = str(row[struct_col]) if pd.notna(row[struct_col]) else None
        if sequence and structure and structure != 'nan':
            tasks.append((sequence, structure, max_spatial_pos))
    
    print(f"需要计算结构信息的样本数: {len(tasks)}")
    
    if len(tasks) == 0:
        print("没有需要计算的样本")
        return
    
    # 确定进程数
    if num_workers is None:
        # 优先使用SLURM分配的核心数
        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
        if slurm_cpus:
            total_cpus = int(slurm_cpus)
            print(f"从SLURM环境变量检测到 {total_cpus} 个CPU核心")
        else:
            total_cpus = cpu_count()
            print(f"从系统检测到 {total_cpus} 个CPU核心")
        
        # 使用非常保守的进程数（12.5%），避免OOM
        # 每个进程启动时需要加载torch等库，占用大量内存
        # 8个进程 × 300MB ≈ 2.4GB（仅启动开销）
        num_workers = min(total_cpus, max(4, int(total_cpus * 0.125)))
    
    print(f"使用 {num_workers} 个进程并行计算...")
    print(f"分批处理，每批 {batch_size} 个样本...")
    
    # 分批处理，避免一次性启动所有进程导致OOM
    structure_cache = {}
    total_batches = (len(tasks) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(tasks))
        batch_tasks = tasks[start_idx:end_idx]
        
        print(f"\n处理批次 {batch_idx + 1}/{total_batches} (样本 {start_idx}-{end_idx})...")
        
        # 每批单独创建进程池，处理完后释放，避免内存累积
        batch_cache = {}
        # 使用chunksize减少进程间通信次数，提高效率
        # chunksize=50表示每次给每个进程分配50个任务，减少通信开销
        # 对于计算密集的任务，较小的chunksize可以更快看到进度
        chunksize = max(1, len(batch_tasks) // (num_workers * 8))
        with Pool(processes=num_workers) as pool:
            # 使用imap_unordered可以更快看到结果（不保证顺序，但更快）
            results = list(tqdm(
                pool.imap_unordered(compute_structure_info_worker, batch_tasks, chunksize=chunksize),
                total=len(batch_tasks),
                desc=f"批次 {batch_idx + 1}/{total_batches}"
            ))
        
        # 收集当前批次的结果
        for result in results:
            if result is not None:
                cache_key, structure_info = result
                batch_cache[cache_key] = structure_info
        
        # 合并到总缓存
        structure_cache.update(batch_cache)
        
        # 每批处理后清理内存
        del batch_cache, results
        import gc
        gc.collect()
        
        print(f"批次 {batch_idx + 1} 完成，当前缓存 {len(structure_cache)} 个样本")
    
    success_count = len(structure_cache)
    print(f"\n成功计算 {success_count}/{len(tasks)} 个样本的结构信息")
    
    # 保存到文件
    print(f"保存到文件: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(structure_cache, f)
    
    print(f"完成！缓存了 {len(structure_cache)} 个样本的结构信息")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='预计算结构信息')
    parser.add_argument('--csv_file', type=str, required=True, help='输入CSV文件路径')
    parser.add_argument('--output_file', type=str, required=True, help='输出pickle文件路径')
    parser.add_argument('--max_spatial_pos', type=int, default=32, help='最大空间位置')
    parser.add_argument('--num_workers', type=int, default=None, help='进程数（默认自动检测）')
    parser.add_argument('--batch_size', type=int, default=10000, help='每批处理的样本数（默认10000）')
    parser.add_argument('--overwrite', action='store_true', help='如果输出文件已存在，是否覆盖')
    
    args = parser.parse_args()
    
    precompute_structures(
        csv_file=args.csv_file,
        output_file=args.output_file,
        max_spatial_pos=args.max_spatial_pos,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        overwrite=args.overwrite
    )

