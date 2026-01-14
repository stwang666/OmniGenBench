#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出Attention信息到.atten格式文件

使用embedding_mixin.py中的批量处理函数，支持：
- 选择输出模型的哪一层
- 选择各个头之间attention如何聚合
"""

import argparse
import json
import csv
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
from tqdm import tqdm

from omnigenbench import ModelHub, OmniTokenizer, OmniDatasetForSequenceClassification


def aggregate_attention_heads(
    attention_tensor: torch.Tensor,
    head_aggregation: str = "mean"
) -> torch.Tensor:
    """
    聚合attention头
    
    Args:
        attention_tensor: shape (layers, heads, seq_len, seq_len) 或 (heads, seq_len, seq_len)
        head_aggregation: 聚合方式，可选 "mean", "max", "sum"
    
    Returns:
        聚合后的attention tensor
    """
    if attention_tensor.ndim == 4:
        # (layers, heads, seq_len, seq_len)
        if head_aggregation == "mean":
            return attention_tensor.mean(dim=1)
        elif head_aggregation == "max":
            return attention_tensor.max(dim=1)[0]
        elif head_aggregation == "sum":
            return attention_tensor.sum(dim=1)
        else:
            raise ValueError(f"不支持的head_aggregation: {head_aggregation}")
    elif attention_tensor.ndim == 3:
        # (heads, seq_len, seq_len)
        if head_aggregation == "mean":
            return attention_tensor.mean(dim=0)
        elif head_aggregation == "max":
            return attention_tensor.max(dim=0)[0]
        elif head_aggregation == "sum":
            return attention_tensor.sum(dim=0)
        else:
            raise ValueError(f"不支持的head_aggregation: {head_aggregation}")
    else:
        raise ValueError(f"不支持的attention tensor维度: {attention_tensor.ndim}")


def extract_position_attention(
    attention_matrix: torch.Tensor,
    attention_mask: torch.Tensor,
    aggregation_method: str = "row_mean"
) -> np.ndarray:
    """
    从attention矩阵中提取每个位置的attention权重
    
    Args:
        attention_matrix: shape (seq_len, seq_len) 的attention矩阵
        attention_mask: shape (seq_len,) 的mask，1表示有效位置
        aggregation_method: 聚合方法
            - "row_mean": 对每行求平均（该位置关注其他位置的平均权重）
            - "col_mean": 对每列求平均（其他位置关注该位置的平均权重）
            - "row_max": 对每行求最大值
            - "col_max": 对每列求最大值
            - "diag": 对角线值（自注意力）
    
    Returns:
        每个位置的attention权重，shape (seq_len,)
    """
    attention_matrix = attention_matrix.cpu().numpy()
    attention_mask = attention_mask.cpu().numpy()
    
    # 只考虑有效位置
    valid_len = int(attention_mask.sum())
    attention_matrix = attention_matrix[:valid_len, :valid_len]
    
    if aggregation_method == "row_mean":
        # 每行平均：该位置关注其他位置的平均权重
        position_attention = attention_matrix.mean(axis=1)
    elif aggregation_method == "col_mean":
        # 每列平均：其他位置关注该位置的平均权重
        position_attention = attention_matrix.mean(axis=0)
    elif aggregation_method == "row_max":
        position_attention = attention_matrix.max(axis=1)
    elif aggregation_method == "col_max":
        position_attention = attention_matrix.max(axis=0)
    elif aggregation_method == "diag":
        position_attention = np.diag(attention_matrix)
    else:
        raise ValueError(f"不支持的aggregation_method: {aggregation_method}")
    
    return position_attention


def export_attention_to_file(
    model_path: str,
    dataset_path: Optional[str] = None,
    sequences: Optional[List[str]] = None,
    labels: Optional[List[int]] = None,
    other_info: Optional[List[Dict]] = None,
    output_file: str = "attention_export.atten",
    layer_indices: Optional[Union[int, List[int]]] = None,
    head_aggregation: str = "mean",
    position_aggregation: str = "row_mean",
    batch_size: int = 4,
    max_length: int = 512,
    device: Optional[str] = None,
):
    """
    导出attention信息到.atten格式文件
    
    Args:
        model_path: 模型路径
        dataset_path: 数据集路径（可选，如果提供则从数据集加载数据）
        sequences: 序列列表（可选，如果提供则直接使用）
        labels: 标签列表（可选）
        other_info: 其他信息列表，每个元素是一个字典（可选）
        output_file: 输出文件路径
        layer_indices: 要提取的层索引，可以是单个int或列表，None表示所有层
        head_aggregation: 头的聚合方式，"mean", "max", "sum"
        position_aggregation: 位置的聚合方式，"row_mean", "col_mean", "row_max", "col_max", "diag"
        batch_size: 批处理大小
        max_length: 最大序列长度
        device: 设备，None表示自动选择
    """
    # 加载模型和tokenizer
    print(f"📦 加载模型: {model_path}")
    model = ModelHub.load(model_path)
    tokenizer = OmniTokenizer.from_pretrained(model_path)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.model.config.output_attentions = True
    
    # 准备数据
    if dataset_path:
        print(f"📂 从数据集加载数据: {dataset_path}")
        # 从数据集加载
        dataset = OmniDatasetForSequenceClassification.from_hub(
            dataset_path,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        # 假设使用test集，可以根据需要修改
        if "test" in dataset:
            test_dataset = dataset["test"]
        elif "valid" in dataset:
            test_dataset = dataset["valid"]
        else:
            test_dataset = list(dataset.values())[0]
        
        sequences = []
        labels = []
        other_info = []
        
        for item in test_dataset:
            if isinstance(item, dict):
                seq = item.get("sequence") or item.get("seq") or item.get("text")
                if seq:
                    sequences.append(seq)
                    labels.append(item.get("label", -1))
                    # 提取其他信息
                    other = {k: v for k, v in item.items() 
                            if k not in ["sequence", "seq", "text", "label", "labels"]}
                    other_info.append(other if other else {})
    else:
        if sequences is None:
            raise ValueError("必须提供dataset_path或sequences")
        if labels is None:
            labels = [-1] * len(sequences)
        if other_info is None:
            other_info = [{}] * len(sequences)
    
    print(f"📊 共 {len(sequences)} 条序列需要处理")
    
    # 处理layer_indices
    if layer_indices is not None:
        if isinstance(layer_indices, int):
            layer_indices = [layer_indices]
        elif isinstance(layer_indices, list):
            pass
        else:
            raise ValueError(f"layer_indices必须是int或List[int]，得到: {type(layer_indices)}")
    
    # 批量提取attention
    print(f"🔍 提取attention（层: {layer_indices if layer_indices else 'all'}, "
          f"头聚合: {head_aggregation}, 位置聚合: {position_aggregation}）")
    
    attention_results = model.batch_extract_attention_scores(
        sequences=sequences,
        batch_size=batch_size,
        max_length=max_length,
        layer_indices=layer_indices,
        head_indices=None,  # 提取所有头，稍后聚合
        return_on_cpu=True,
    )
    
    # 进行预测获取probabilities
    print("🔮 进行模型预测...")
    predictions = []
    probabilities_list = []
    
    # 尝试使用inference方法，如果不存在则使用forward
    has_inference = hasattr(model, 'inference')
    
    for i in tqdm(range(0, len(sequences), batch_size), desc="预测"):
        batch_sequences = sequences[i:i+batch_size]
        outputs = []
        
        if has_inference:
            # 使用inference方法
            for seq in batch_sequences:
                try:
                    output = model.inference(seq)
                    outputs.append(output)
                except Exception as e:
                    print(f"警告: inference失败，使用forward方法: {e}")
                    has_inference = False
                    break
        
        if not has_inference:
            # 使用forward方法
            inputs = tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                model_outputs = model.model(**inputs)
                logits = model_outputs.logits
                probs_tensor = torch.softmax(logits, dim=-1).cpu().numpy()
                preds_tensor = torch.argmax(logits, dim=-1).cpu().numpy()
            
            for j, seq in enumerate(batch_sequences):
                pred = int(preds_tensor[j])
                probs = probs_tensor[j].tolist()
                outputs.append({
                    'predictions': pred,
                    'probabilities': probs
                })
        
        for output in outputs:
            pred = output.get('predictions', output.get('prediction', -1))
            probs = output.get('probabilities', output.get('confidence', []))
            
            # 处理probabilities格式
            if isinstance(probs, (list, tuple, np.ndarray)):
                if len(probs) == 1:
                    # 二分类情况，需要补充另一个概率
                    prob_0 = 1 - probs[0] if isinstance(probs[0], (int, float)) else 1 - float(probs[0])
                    prob_1 = probs[0] if isinstance(probs[0], (int, float)) else float(probs[0])
                    probs = [prob_0, prob_1]
                prob_str = ",".join(map(str, probs))
            elif isinstance(probs, (int, float)):
                # 单个概率值，假设是二分类
                prob_str = f"{1-probs},{probs}"
            else:
                prob_str = str(probs)
            
            predictions.append(pred)
            probabilities_list.append(prob_str)
    
    # 处理attention并写入文件
    print("💾 写入文件...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['Sequence', 'Prediction', 'Probabilities', 
                       'Actual Label', 'Attention', 'Other'],
            delimiter='\t'
        )
        writer.writeheader()
        
        for i, (seq, label, other, attn_result, pred, probs) in enumerate(
            tqdm(zip(sequences, labels, other_info, attention_results, 
                    predictions, probabilities_list), 
                 desc="处理", total=len(sequences))
        ):
            # 获取attention tensor: (layers, heads, seq_len, seq_len)
            attentions = attn_result['attentions']
            attention_mask = attn_result['attention_mask']
            
            # 聚合头
            if attentions.ndim == 4:
                # (layers, heads, seq_len, seq_len) -> (layers, seq_len, seq_len)
                attentions_agg = aggregate_attention_heads(attentions, head_aggregation)
            else:
                # 已经是聚合后的
                attentions_agg = attentions
            
            # 选择层
            if attentions_agg.ndim == 3:
                # 有多个层，选择或平均
                if layer_indices is not None and len(layer_indices) == 1:
                    # 选择单个层
                    attention_matrix = attentions_agg[layer_indices[0]]
                elif layer_indices is not None and len(layer_indices) > 1:
                    # 选择多个层并平均
                    attention_matrix = attentions_agg[layer_indices].mean(dim=0)
                else:
                    # 平均所有层
                    attention_matrix = attentions_agg.mean(dim=0)
            else:
                # 已经是单层
                attention_matrix = attentions_agg
            
            # 提取位置级别的attention
            position_attention = extract_position_attention(
                attention_matrix,
                attention_mask,
                position_aggregation
            )
            
            # 格式化为字符串
            attention_str = ",".join(map(str, position_attention))
            
            # 格式化other信息
            other_str = json.dumps(other, ensure_ascii=False)
            
            # 写入行
            writer.writerow({
                'Sequence': seq,
                'Prediction': pred,
                'Probabilities': probs,
                'Actual Label': label,
                'Attention': attention_str,
                'Other': other_str
            })
    
    print(f"✅ 完成！输出文件: {output_path}")
    print(f"   - 共处理 {len(sequences)} 条序列")
    print(f"   - 层索引: {layer_indices if layer_indices else 'all'}")
    print(f"   - 头聚合: {head_aggregation}")
    print(f"   - 位置聚合: {position_aggregation}")


def main():
    parser = argparse.ArgumentParser(
        description="导出Attention信息到.atten格式文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 从数据集导出，使用最后一层，平均所有头
  python export_attention.py \\
    --model_path /path/to/model \\
    --dataset_path /path/to/dataset \\
    --output_file output.atten \\
    --layer_indices -1 \\
    --head_aggregation mean \\
    --position_aggregation row_mean

  # 从序列列表导出，使用所有层平均
  python export_attention.py \\
    --model_path /path/to/model \\
    --sequences seq1.txt seq2.txt \\
    --output_file output.atten \\
    --head_aggregation mean \\
    --position_aggregation col_mean
        """
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='模型路径'
    )
    
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
        help='数据集路径（可选，如果提供则从数据集加载）'
    )
    
    parser.add_argument(
        '--sequences',
        type=str,
        nargs='+',
        default=None,
        help='序列列表（可选，如果提供则直接使用）'
    )
    
    parser.add_argument(
        '--labels',
        type=int,
        nargs='+',
        default=None,
        help='标签列表（可选）'
    )
    
    parser.add_argument(
        '--output_file',
        type=str,
        default='attention_export.atten',
        help='输出文件路径（默认: attention_export.atten）'
    )
    
    parser.add_argument(
        '--layer_indices',
        type=int,
        nargs='+',
        default=None,
        help='要提取的层索引，可以是单个或多个，例如: -1 或 0 5 10（默认: 所有层）'
    )
    
    parser.add_argument(
        '--head_aggregation',
        type=str,
        choices=['mean', 'max', 'sum'],
        default='mean',
        help='头的聚合方式（默认: mean）'
    )
    
    parser.add_argument(
        '--position_aggregation',
        type=str,
        choices=['row_mean', 'col_mean', 'row_max', 'col_max', 'diag'],
        default='row_mean',
        help='位置的聚合方式（默认: row_mean）'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='批处理大小（默认: 4）'
    )
    
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='最大序列长度（默认: 512）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='设备（默认: 自动选择）'
    )
    
    args = parser.parse_args()
    
    # 处理sequences参数（如果从文件读取）
    sequences = args.sequences
    if sequences and len(sequences) == 1:
        # 可能是文件路径
        seq_file = Path(sequences[0])
        if seq_file.exists() and seq_file.is_file():
            with open(seq_file, 'r') as f:
                sequences = [line.strip() for line in f if line.strip()]
    
    export_attention_to_file(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        sequences=sequences,
        labels=args.labels,
        output_file=args.output_file,
        layer_indices=args.layer_indices,
        head_aggregation=args.head_aggregation,
        position_aggregation=args.position_aggregation,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )


if __name__ == "__main__":
    main()

