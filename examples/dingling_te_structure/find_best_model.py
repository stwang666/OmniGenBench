"""
查找并测试历史所有checkpoint中最好的模型。
可以根据不同的metric（accuracy_score, f1_score等）来选择最佳模型。
"""

import os
import re
import torch
import gc
from typing import Optional, Dict, Any, List

torch.cuda.empty_cache()
gc.collect()

from omnigenbench import (
    ClassificationMetric,
    AccelerateTrainer,
    ModelHub,
    OmniTokenizer,
    OmniDatasetForSequenceClassification,
)


# 自定义数据集类（与训练时相同）
class OmniDatasetWithTissue(OmniDatasetForSequenceClassification):
    """支持tissue信息的数据集类"""
    
    def __init__(self, dataset_name_or_path, tokenizer, max_length=None, **kwargs):
        self.tissues = [
            'root', 'seedling', 'leaf', 'FMI', 'FOD',
            'Prophase-I-pollen', 'Tricellular-pollen', 'flag', 'grain'
        ]
        self.tissue2id = {t: i for i, t in enumerate(self.tissues)}
        super().__init__(dataset_name_or_path, tokenizer, max_length, **kwargs)
        
        for item in self.data:
            if "tissue_id" in item:
                tid = item["tissue_id"]
                if tid.ndim == 0:
                    item["tissue_id"] = tid.unsqueeze(0)
                elif tid.ndim > 1:
                    item["tissue_id"] = tid.flatten()[:1]
    
    def prepare_input(self, instance, **kwargs):
        labels = -100
        tissue_id = None
        
        if isinstance(instance, str):
            sequence = instance
        elif isinstance(instance, dict):
            sequence = (
                instance.get("seq", None)
                if "seq" in instance
                else instance.get("sequence", None)
            )
            label = instance.get("label", None)
            labels = instance.get("labels", None)
            labels = labels if labels is not None else label
            
            tissue_name = instance["tissue"]
            tissue_id = self.tissue2id[tissue_name]
        else:
            raise Exception("Unknown instance format.")

        tokenized_inputs = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        for col in tokenized_inputs:
            tokenized_inputs[col] = tokenized_inputs[col].squeeze()

        if labels is not None:
            labels = self.label2id.get(str(labels), -100)
            if not isinstance(labels, int):
                raise Exception("The label must be an integer for sequence classification.")
        tokenized_inputs["labels"] = torch.tensor(labels)
        
        if tissue_id is not None:
            tokenized_inputs["tissue_id"] = torch.tensor([tissue_id], dtype=torch.long)
        else:
            tokenized_inputs["tissue_id"] = torch.tensor([0], dtype=torch.long)
        
        return tokenized_inputs
    
    def _pad_and_truncate(self, pad_value=0):
        tissue_ids = []
        for item in self.data:
            if "tissue_id" in item:
                tissue_ids.append(item.pop("tissue_id"))
            else:
                tissue_ids.append(None)
        
        super()._pad_and_truncate(pad_value)
        
        for i, item in enumerate(self.data):
            if tissue_ids[i] is not None:
                tid = tissue_ids[i]
                if tid.ndim == 0:
                    item["tissue_id"] = tid.unsqueeze(0)
                elif tid.ndim == 1:
                    item["tissue_id"] = tid
                else:
                    item["tissue_id"] = tid.flatten()[:1]
            else:
                item["tissue_id"] = torch.tensor([0], dtype=torch.long)


def find_all_checkpoints(
    checkpoint_dir: str,
    pattern_prefix: str = "ogb_te_2class_finetuned_v1.5_seq_tissue"
) -> List[Dict[str, Any]]:
    """
    查找目录中所有符合pattern的checkpoint。
    
    Args:
        checkpoint_dir: checkpoint所在目录
        pattern_prefix: checkpoint名称前缀
    
    Returns:
        包含所有checkpoint信息的列表
    """
    checkpoints = []
    
    for item in os.listdir(checkpoint_dir):
        item_path = os.path.join(checkpoint_dir, item)
        if not os.path.isdir(item_path):
            continue
        
        # 检查是否符合pattern
        if not item.startswith(pattern_prefix):
            continue
        
        # 提取epoch编号
        epoch_match = re.search(r"epoch_(\d+)", item)
        if not epoch_match:
            continue
        epoch = int(epoch_match.group(1))
        
        # 提取所有metric
        metrics = {}
        for metric_name in ["accuracy_score", "f1_score"]:
            pattern = f"{metric_name}_([0-9.]+)"
            match = re.search(pattern, item)
            if match:
                metrics[metric_name] = float(match.group(1))
        
        if metrics:  # 只有当至少有一个metric时才添加
            checkpoints.append({
                'path': item_path,
                'name': item,
                'epoch': epoch,
                'metrics': metrics
            })
    
    return checkpoints


def display_checkpoints_table(checkpoints: List[Dict[str, Any]]) -> None:
    """
    以表格形式显示所有checkpoint的信息。
    """
    if not checkpoints:
        print("⚠️ 未找到任何checkpoint")
        return
    
    # 按epoch排序
    checkpoints.sort(key=lambda x: x['epoch'])
    
    print("\n" + "=" * 120)
    print(f"{'Epoch':^8} | {'Accuracy':^12} | {'F1 Score':^12} | {'Checkpoint Name':^70}")
    print("=" * 120)
    
    for ckpt in checkpoints:
        epoch = ckpt['epoch']
        metrics = ckpt['metrics']
        acc = metrics.get('accuracy_score', -1)
        f1 = metrics.get('f1_score', -1)
        name = ckpt['name']
        
        print(f"{epoch:^8} | {acc:^12.4f} | {f1:^12.4f} | {name[:70]}")
    
    print("=" * 120)


def find_best_checkpoint(
    checkpoints: List[Dict[str, Any]],
    metric_name: str = "accuracy_score",
    top_k: int = 5
) -> Optional[Dict[str, Any]]:
    """
    从checkpoints中找出指定metric最好的那个。
    
    Args:
        checkpoints: checkpoint列表
        metric_name: 要优化的metric名称
        top_k: 显示前k个最好的checkpoint
    
    Returns:
        最好的checkpoint信息
    """
    if not checkpoints:
        return None
    
    # 过滤出有该metric的checkpoint
    valid_checkpoints = [
        ckpt for ckpt in checkpoints 
        if metric_name in ckpt['metrics']
    ]
    
    if not valid_checkpoints:
        print(f"⚠️ 没有checkpoint包含metric: {metric_name}")
        return None
    
    # 按metric排序
    valid_checkpoints.sort(key=lambda x: x['metrics'][metric_name], reverse=True)
    
    # 显示top k
    print(f"\n📊 按 {metric_name} 排序的Top {top_k} checkpoint:")
    print("=" * 120)
    for i, ckpt in enumerate(valid_checkpoints[:top_k]):
        epoch = ckpt['epoch']
        metric_value = ckpt['metrics'][metric_name]
        other_metrics = {k: v for k, v in ckpt['metrics'].items() if k != metric_name}
        other_str = ", ".join([f"{k}={v:.4f}" for k, v in other_metrics.items()])
        print(f"  {i+1}. Epoch {epoch:2d}: {metric_name}={metric_value:.4f} ({other_str})")
    print("=" * 120)
    
    best = valid_checkpoints[0]
    print(f"\n✅ 最佳checkpoint (按{metric_name}):")
    print(f"   - Epoch: {best['epoch']}")
    print(f"   - {metric_name}: {best['metrics'][metric_name]:.4f}")
    for k, v in best['metrics'].items():
        if k != metric_name:
            print(f"   - {k}: {v:.4f}")
    print(f"   - 路径: {best['path']}")
    
    return best


def test_checkpoint(
    checkpoint_path: str,
    datasets: Dict[str, Any],
    metric_functions: List,
    device: str = "cuda:0"
) -> Dict[str, float]:
    """
    加载checkpoint并在测试集上评估。
    
    Args:
        checkpoint_path: checkpoint路径
        datasets: 包含train/valid/test的数据集字典
        metric_functions: metric函数列表
        device: 运行设备
    
    Returns:
        测试集上的metric结果
    """
    print(f"\n🔄 加载checkpoint: {os.path.basename(checkpoint_path)}")
    
    # 不使用custom_model.py（会执行训练代码），直接重新定义模型类
    # 这样避免执行checkpoint中保存的训练代码
    import torch.nn as nn
    from omnigenbench import OmniModelForSequenceClassification, OmniTokenizer
    
    # 在函数内部重新定义模型类，避免执行custom_model.py中的训练代码
    class OmniModelForSequenceClassificationWithTissue(OmniModelForSequenceClassification):
        """支持tissue嵌入的序列分类模型"""
        
        def __init__(self, config_or_model, tokenizer, *args, **kwargs):
            super().__init__(config_or_model, tokenizer, *args, **kwargs)
            self.metadata["model_name"] = self.__class__.__name__
            
            self.tissue_embed_dim = self.config.hidden_size // 4
            self.tissue_embedding = nn.Embedding(
                num_embeddings=9,
                embedding_dim=self.tissue_embed_dim
            )
            
            self.classifier = nn.Linear(
                self.config.hidden_size + self.tissue_embed_dim,
                self.config.num_labels
            )

        def forward(self, **inputs):
            labels = inputs.pop("labels", None)
            tissue_id = inputs.pop("tissue_id", None)
            
            last_hidden_state = self.last_hidden_state_forward(**inputs)
            last_hidden_state = self.dropout(last_hidden_state)
            last_hidden_state = self.activation(last_hidden_state)
            
            pooled_state = self.pooler(inputs, last_hidden_state)
            
            if tissue_id is not None:
                if tissue_id.ndim > 1:
                    tissue_id = tissue_id.squeeze(-1)
                tissue_embed = self.tissue_embedding(tissue_id)
            else:
                batch_size = last_hidden_state.shape[0]
                tissue_embed = torch.zeros(batch_size, self.tissue_embed_dim, device=last_hidden_state.device)

            combined_features = torch.cat([pooled_state, tissue_embed], dim=-1)
            logits = self.classifier(combined_features)
            
            outputs = {
                "logits": logits,
                "last_hidden_state": last_hidden_state, 
                "labels": labels,
            }
            return outputs
    
    # 加载tokenizer
    tokenizer = OmniTokenizer.from_pretrained(checkpoint_path)
    
    # 创建模型实例并加载权重
    model = OmniModelForSequenceClassificationWithTissue(checkpoint_path, tokenizer, num_labels=2)
    
    # 加载state dict
    model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    print(f"✅ 模型加载成功")
    
    # 创建trainer用于测试（batch_size设为1，最稳定）
    print(f"🧪 在测试集上评估（batch_size=1，避免维度不匹配）...")
    trainer = AccelerateTrainer(
        model=model,
        epochs=1,
        learning_rate=2e-5,
        batch_size=1,  # 使用batch_size=1避免最后一个batch维度不匹配的问题
        train_dataset=datasets["train"],
        eval_dataset=datasets["valid"],
        test_dataset=datasets["test"],
        compute_metrics=metric_functions,
        device=torch.device(device),
    )
    
    test_metrics = trainer.test()
    
    print("\n" + "=" * 80)
    print("📊 测试集结果：")
    print("=" * 80)
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("=" * 80)
    
    # 保存测试结果到文件
    checkpoint_name = os.path.basename(checkpoint_path)
    result_file = os.path.join(
        os.path.dirname(checkpoint_path),
        f"test_results_{checkpoint_name}.json"
    )
    
    import json
    result_data = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_name": checkpoint_name,
        "test_metrics": test_metrics,
        "test_time": str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu",
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 测试结果已保存到: {result_file}")
    
    return test_metrics


def main(test_mode: str = "3"):
    """
    主函数
    
    Args:
        test_mode: 测试模式 ("1"=测试最佳accuracy, "2"=测试最佳f1, "3"=都测试, "4"=不测试)
    """
    # 配置参数
    CHECKPOINT_DIR = "/home/sw1136/OmniGenBench/examples/dingling_te_structure"
    CHECKPOINT_PREFIX = "ogb_te_2class_finetuned_v1.5_seq_tissue"
    MODEL_NAME = "yangheng/OmniGenome-v1.5"
    DATA_DIR = "/home/sw1136/OmniGenBench/examples/dingling_te_structure/biclass_seq"
    
    print("🔍 查找所有checkpoint...")
    print(f"  - 目录: {CHECKPOINT_DIR}")
    print(f"  - 前缀: {CHECKPOINT_PREFIX}")
    
    # 查找所有checkpoint
    checkpoints = find_all_checkpoints(CHECKPOINT_DIR, CHECKPOINT_PREFIX)
    print(f"\n✅ 找到 {len(checkpoints)} 个checkpoint")
    
    if not checkpoints:
        print("❌ 未找到任何checkpoint，程序退出")
        return
    
    # 显示所有checkpoint的表格
    display_checkpoints_table(checkpoints)
    
    # 找出最佳checkpoint（可以选择不同的metric）
    print("\n" + "=" * 120)
    print("按不同metric查找最佳checkpoint:")
    print("=" * 120)
    
    # 1. 按accuracy_score查找
    best_by_accuracy = find_best_checkpoint(checkpoints, metric_name="accuracy_score", top_k=5)
    
    # 2. 按f1_score查找
    best_by_f1 = find_best_checkpoint(checkpoints, metric_name="f1_score", top_k=5)
    
    # 询问用户要测试哪个（或使用传入的test_mode参数）
    print("\n" + "=" * 120)
    print("选择要在测试集上评估的模型：")
    print("  1. 按accuracy_score最佳的模型")
    print("  2. 按f1_score最佳的模型")
    print("  3. 两个都测试")
    print("  4. 不测试，仅显示排名")
    print("=" * 120)
    
    # 尝试交互式输入，如果失败则使用默认的test_mode
    try:
        choice = input("请输入选择 (1/2/3/4，默认3): ").strip() or test_mode
    except EOFError:
        # 非交互式环境，使用参数指定的test_mode
        choice = test_mode
        print(f"使用默认选择: {choice}")
    
    if choice == "4":
        print("\n✅ 完成！")
        return
    
    # 加载数据集
    print("\n📂 加载数据集...")
    label2id = {"0": 0, "1": 1}
    tokenizer = OmniTokenizer.from_pretrained(MODEL_NAME)
    datasets = OmniDatasetWithTissue.from_hub(
        DATA_DIR,
        tokenizer=tokenizer,
        max_length=512,
        label2id=label2id,
    )
    print(f"✅ 数据集加载完成")
    
    metric_functions = [
        ClassificationMetric().accuracy_score,
        ClassificationMetric().f1_score
    ]
    
    # 测试模型（内部使用batch_size=1避免维度问题）
    if choice in ["1", "3"] and best_by_accuracy:
        print("\n" + "=" * 120)
        print("🎯 测试按accuracy_score最佳的模型")
        print("=" * 120)
        test_checkpoint(
            best_by_accuracy['path'],
            datasets,
            metric_functions
        )
    
    if choice in ["2", "3"] and best_by_f1:
        # 如果两个模型相同，跳过重复测试
        if best_by_f1['path'] == best_by_accuracy['path']:
            print("\n" + "=" * 120)
            print("ℹ️  最佳accuracy和最佳f1是同一个模型，跳过重复测试")
            print("=" * 120)
        else:
            print("\n" + "=" * 120)
            print("🎯 测试按f1_score最佳的模型")
            print("=" * 120)
            test_checkpoint(
                best_by_f1['path'],
                datasets,
                metric_functions
            )
    
    print("\n✅ 所有任务完成！")


if __name__ == "__main__":
    import sys
    
    # 支持命令行参数: python find_best_model.py [1/2/3/4]
    # 1: 测试最佳accuracy模型
    # 2: 测试最佳f1模型
    # 3: 两个都测试（默认）
    # 4: 只显示排名，不测试
    test_mode = sys.argv[1] if len(sys.argv) > 1 else "3"
    main(test_mode)

