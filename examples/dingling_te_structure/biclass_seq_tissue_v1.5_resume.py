# Step 1: Data Preparation
import torch
import gc
torch.cuda.empty_cache()
gc.collect()

from omnigenbench import (
    ClassificationMetric,
    AccelerateTrainer,
    ModelHub,
    OmniTokenizer,
    OmniDatasetForSequenceClassification,
    OmniModelForSequenceClassification,
)
import torch.nn as nn
from typing import Dict, Any, Optional
import os
import re
import json

# model_name_or_path = "yangheng/OmniGenome-52M"
model_name_or_path = "yangheng/OmniGenome-v1.5"

# 自定义数据集类：处理tissue字段并转换为0-8编号
class OmniDatasetWithTissue(OmniDatasetForSequenceClassification):
    """
    支持tissue信息的数据集类。
    将tissue名称映射为0-8的编号，并在prepare_input中返回tissue_id。
    """
    
    def __init__(self, dataset_name_or_path, tokenizer, max_length=None, **kwargs):
        # 在调用super().__init__()之前初始化tissue2id
        # 因为父类的__init__会在初始化过程中调用prepare_input
        # 定义组织映射
        self.tissues = [
            'root', 'seedling', 'leaf', 'FMI', 'FOD',
            'Prophase-I-pollen', 'Tricellular-pollen', 'flag', 'grain'
        ]
        self.tissue2id = {t: i for i, t in enumerate(self.tissues)}
        
        super().__init__(dataset_name_or_path, tokenizer, max_length, **kwargs)
        
        # 在父类__init__完成后，确保所有tissue_id都是1维张量
        # 因为父类的__init__会对所有值调用squeeze(0)，会把[1]压缩成0维
        for item in self.data:
            if "tissue_id" in item:
                tid = item["tissue_id"]
                if tid.ndim == 0:
                    # 如果是0维标量，转换为1维
                    item["tissue_id"] = tid.unsqueeze(0)
                elif tid.ndim > 1:
                    # 如果是多维，展平后取第一个
                    item["tissue_id"] = tid.flatten()[:1]
                # 如果已经是1维，保持不变
    
    def prepare_input(self, instance, **kwargs):
        """
        准备输入数据，包括tissue信息。
        """
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
            
            # 获取tissue信息
            tissue_name = instance["tissue"]
            tissue_id =self.tissue2id[tissue_name]
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
                raise Exception(
                    "The label must be an integer for sequence classification."
                )
        tokenized_inputs["labels"] = torch.tensor(labels)
        
        # 添加tissue_id（确保是1维张量，避免在_pad_and_truncate中出错）
        if tissue_id is not None:
            tokenized_inputs["tissue_id"] = torch.tensor([tissue_id], dtype=torch.long)  # 使用列表创建1维张量
        else:
            tokenized_inputs["tissue_id"] = torch.tensor([0], dtype=torch.long)  # 默认值，1维张量
        
        return tokenized_inputs
    
    def _pad_and_truncate(self, pad_value=0):
        """
        重写_pad_and_truncate方法，跳过tissue_id的padding处理。
        tissue_id是标量值，不应该被padding。
        """
        # 临时移除tissue_id，避免被padding处理
        tissue_ids = []
        for item in self.data:
            if "tissue_id" in item:
                tissue_ids.append(item.pop("tissue_id"))
            else:
                tissue_ids.append(None)
        
        # 调用父类的_pad_and_truncate处理其他字段
        super()._pad_and_truncate(pad_value)
        
        # 恢复tissue_id，确保是1维张量
        for i, item in enumerate(self.data):
            if tissue_ids[i] is not None:
                tid = tissue_ids[i]
                # 确保tissue_id是1维张量 [1]
                if tid.ndim == 0:
                    # 如果是0维标量，转换为1维
                    item["tissue_id"] = tid.unsqueeze(0)
                elif tid.ndim == 1:
                    # 如果已经是1维，直接使用
                    item["tissue_id"] = tid
                else:
                    # 如果是多维，展平后取第一个
                    item["tissue_id"] = tid.flatten()[:1]
            else:
                # 如果没有tissue_id，使用默认值
                item["tissue_id"] = torch.tensor([0], dtype=torch.long)


# 自定义模型类：添加tissue嵌入层
class OmniModelForSequenceClassificationWithTissue(OmniModelForSequenceClassification):
    """
    支持tissue嵌入的序列分类模型。
    将tissue嵌入拼接到last_hidden_state（在pooler之前，每个token位置都会包含tissue信息）。
    """
    
    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        
        # 定义tissue嵌入层：9个tissue类别（0-8），嵌入维度设为hidden_size的一部分
        # 这里使用hidden_size的1/4作为tissue嵌入维度
        self.tissue_embed_dim = self.config.hidden_size // 4
        self.tissue_embedding = nn.Embedding(
            num_embeddings=9,  # 0-8共9个tissue
            embedding_dim=self.tissue_embed_dim
        )
        
        # 重新定义classifier，因为输入维度变为 hidden_size + tissue_embed_dim
        self.classifier = nn.Linear(
            self.config.hidden_size + self.tissue_embed_dim,
            self.config.num_labels
        )

    def forward(self, **inputs):
        labels = inputs.pop("labels", None)
        tissue_id = inputs.pop("tissue_id", None)
        
        # 1. 获取序列的 last_hidden_state (不拼接 Tissue)
        # Shape: (batch_size, seq_len, hidden_size)
        last_hidden_state = self.last_hidden_state_forward(**inputs)
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        
        # 2. 直接池化序列特征 (使用原始 hidden_size，无需切片)
        # Shape: (batch_size, hidden_size)
        pooled_state = self.pooler(inputs, last_hidden_state)
        
        # 3. 获取 Tissue Embedding (无需扩展到 seq_len)
        if tissue_id is not None:
            if tissue_id.ndim > 1:
                tissue_id = tissue_id.squeeze(-1)
            # Shape: (batch_size, tissue_embed_dim)
            tissue_embed = self.tissue_embedding(tissue_id)
        else:
            batch_size = last_hidden_state.shape[0]
            tissue_embed = torch.zeros(batch_size, self.tissue_embed_dim, device=last_hidden_state.device)

        # 4. 在池化后进行拼接 (Late Fusion)
        # Shape: (batch_size, hidden_size + tissue_embed_dim)
        combined_features = torch.cat([pooled_state, tissue_embed], dim=-1)
        
        # 5. 分类
        logits = self.classifier(combined_features)
        
        outputs = {
            "logits": logits,
            # 如果下游需要 last_hidden_state，返回原始的即可
            "last_hidden_state": last_hidden_state, 
            "labels": labels,
        }
        return outputs


# 自定义AccelerateTrainer，支持从指定epoch开始训练
class ResumableAccelerateTrainer(AccelerateTrainer):
    """
    支持断点续训的Trainer，可以设置起始epoch编号。
    """
    
    def __init__(self, *args, start_epoch: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch
    
    def train(self, path_to_save: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        从start_epoch开始训练。
        """
        # 修复导入路径：直接使用父类的方法，不需要额外导入
        import random
        import numpy as np
        
        # seed_everything功能
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Initialize early stopping flag for distributed coordination
        self.early_stop_flag = torch.tensor(0, device=self.accelerator.device)

        # Ensure all processes sync before starting
        self.accelerator.wait_for_everyone()

        # Initial evaluation
        if self.eval_loader is not None and len(self.eval_loader) > 0:
            initial_metrics = self.evaluate()
        else:
            initial_metrics = self.test()

        # Only main process handles metric comparison and model saving
        if self.accelerator.is_main_process:
            if self._is_metric_better(initial_metrics, stage="valid"):
                self._save_state_dict()
                self.early_stop_flag = torch.tensor(0, device=self.accelerator.device)

        # Synchronize early stopping flag across all processes
        gathered_flags = self.accelerator.gather(self.early_stop_flag)
        self.early_stop_flag = (
            gathered_flags if gathered_flags.ndim == 0 else gathered_flags[0]
        )

        # Main training loop - 从start_epoch开始
        for epoch in range(self.epochs):
            # 实际的epoch编号
            actual_epoch = epoch + self.start_epoch
            
            # Train for one epoch (显示实际epoch编号)
            avg_loss = self._train_epoch(actual_epoch)

            # Synchronize all processes before evaluation
            self.accelerator.wait_for_everyone()

            # Evaluate after each epoch
            if self.eval_loader is not None and len(self.eval_loader) > 0:
                valid_metrics = self.evaluate()
            else:
                valid_metrics = self.test()

            # Only main process handles metric comparison and early stopping
            if self.accelerator.is_main_process:
                if self._is_metric_better(valid_metrics, stage="valid"):
                    self._save_state_dict()
                    self.early_stop_flag = torch.tensor(
                        0, device=self.accelerator.device
                    )
                else:
                    self.early_stop_flag += 1

            # Synchronize early stopping flag across all processes
            gathered_flags = self.accelerator.gather(self.early_stop_flag)
            self.early_stop_flag = (
                gathered_flags if gathered_flags.ndim == 0 else gathered_flags[0]
            )

            # Check for early stopping
            if self.early_stop_flag.item() > self.patience:
                if self.accelerator.is_main_process:
                    print(f"Early stopping at epoch {actual_epoch + 1}.")
                break

            # Save epoch checkpoint (only main process) - 使用实际epoch编号
            if path_to_save and self.accelerator.is_main_process:
                self._save_epoch_checkpoint_with_actual_epoch(
                    path_to_save, actual_epoch, valid_metrics, **kwargs
                )

            # Ensure all processes sync before next epoch
            self.accelerator.wait_for_everyone()

        # Final testing with best model
        if self.test_loader is not None and len(self.test_loader) > 0:
            self._load_state_dict()
            self.accelerator.wait_for_everyone()
            test_metrics = self.test()
            if self.accelerator.is_main_process:
                self._is_metric_better(test_metrics, stage="test")

        # Save final model (only main process)
        if path_to_save and self.accelerator.is_main_process:
            self._save_final_model(path_to_save, **kwargs)

        # Clean up
        self._remove_state_dict()
        self.accelerator.free_memory(
            self.model,
            self.optimizer,
            self.train_loader,
            self.eval_loader,
            self.test_loader,
        )
        return self.metrics
    
    def _save_epoch_checkpoint_with_actual_epoch(
        self, path_to_save: str, actual_epoch: int, metrics: Dict[str, Any], **kwargs
    ) -> None:
        """
        使用实际epoch编号保存checkpoint。
        """
        checkpoint_path = f"{path_to_save}_epoch_{actual_epoch + 1}"

        if metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    checkpoint_path += f"_seed_{self.seed}_{key}_{value:.4f}"

        self.save_model(checkpoint_path, **kwargs)


def find_best_checkpoint(checkpoint_dir: str, metric_name: str = "accuracy_score") -> Optional[str]:
    """
    查找所有checkpoint中指定metric最好的那个。
    
    Args:
        checkpoint_dir: checkpoint所在目录
        metric_name: 要优化的metric名称（如'accuracy_score', 'f1_score'）
    
    Returns:
        最好的checkpoint路径
    """
    checkpoints = []
    
    # 遍历目录查找所有checkpoint
    for item in os.listdir(checkpoint_dir):
        item_path = os.path.join(checkpoint_dir, item)
        if not os.path.isdir(item_path):
            continue
        
        # 从目录名中提取metric值
        # 格式：ogb_te_2class_finetuned_v1.5_seq_tissue_epoch_19_seed_42_accuracy_score_0.7517_seed_42_f1_score_0.7234
        pattern = f"{metric_name}_([0-9.]+)"
        match = re.search(pattern, item)
        if match:
            metric_value = float(match.group(1))
            
            # 提取epoch编号
            epoch_match = re.search(r"epoch_(\d+)", item)
            epoch = int(epoch_match.group(1)) if epoch_match else -1
            
            checkpoints.append({
                'path': item_path,
                'name': item,
                'metric_value': metric_value,
                'epoch': epoch
            })
    
    if not checkpoints:
        print(f"⚠️ 在 {checkpoint_dir} 中未找到任何checkpoint")
        return None
    
    # 按metric_value排序
    checkpoints.sort(key=lambda x: x['metric_value'], reverse=True)
    
    # 打印所有checkpoint的排名
    print(f"\n📊 所有checkpoint按{metric_name}排序：")
    print("=" * 100)
    for i, ckpt in enumerate(checkpoints[:10]):  # 只显示前10个
        print(f"  {i+1}. Epoch {ckpt['epoch']:2d}: {metric_name}={ckpt['metric_value']:.4f} - {ckpt['name']}")
    
    best = checkpoints[0]
    print("=" * 100)
    print(f"✅ 最佳checkpoint: Epoch {best['epoch']}, {metric_name}={best['metric_value']:.4f}")
    print(f"   路径: {best['path']}")
    
    return best['path']


# ==================== 主程序 ====================

# We define the label mapping in the training
label2id = {"0": 0, "1": 1}  # 0: Low TE, 1: High TE

# Initialize tokenizer
tokenizer = OmniTokenizer.from_pretrained(model_name_or_path)

# Load datasets
datasets = OmniDatasetWithTissue.from_hub(
    "/home/sw1136/OmniGenBench/examples/dingling_te_structure/biclass_seq",
    tokenizer=tokenizer,
    max_length=512,
    label2id=label2id,
)
print(f"📊 Loaded datasets: {list(datasets.keys())}")
for split, dataset in datasets.items():
    print(f"  - {split}: {len(dataset)} samples")

# ===== 配置参数 =====
RESUME_FROM_CHECKPOINT = True  # 是否从checkpoint恢复
CHECKPOINT_PATH = "ogb_te_2class_finetuned_v1.5_seq_tissue_epoch_19_seed_42_accuracy_score_0.7517_seed_42_f1_score_0.7234"
START_EPOCH = 19  # 从第19个epoch继续（显示为epoch 20, 21, ...）
TOTAL_EPOCHS = 20  # 总共要训练60个epoch
REMAINING_EPOCHS = TOTAL_EPOCHS - START_EPOCH  # 还需要训练的epoch数

# Step 2: Model Initialization
if RESUME_FROM_CHECKPOINT:
    print(f"\n🔄 从checkpoint恢复训练...")
    print(f"  - Checkpoint: {CHECKPOINT_PATH}")
    print(f"  - 已完成epoch: {START_EPOCH}")
    print(f"  - 剩余epoch: {REMAINING_EPOCHS}")
    
    # 加载checkpoint - 直接使用自定义模型类避免labels参数问题
    # 创建新模型实例
    model = OmniModelForSequenceClassificationWithTissue(
        model_name_or_path,
        tokenizer,
        num_labels=len(list(label2id.keys())),
    )
    
    # 加载checkpoint的权重
    checkpoint_model_path = os.path.join(CHECKPOINT_PATH, "pytorch_model.bin")
    if os.path.exists(checkpoint_model_path):
        state_dict = torch.load(checkpoint_model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"✅ 成功加载checkpoint权重")
    else:
        print(f"⚠️ 未找到checkpoint权重文件，将使用预训练模型")
        print(f"   预期路径: {checkpoint_model_path}")
else:
    print(f"\n🆕 从头开始训练...")
    model = OmniModelForSequenceClassificationWithTissue(
        model_name_or_path,
        tokenizer,
        num_labels=len(list(label2id.keys())),
    )
    START_EPOCH = 0
    REMAINING_EPOCHS = TOTAL_EPOCHS

# Step 3: Model Training
metric_functions = [
    ClassificationMetric().accuracy_score,
    ClassificationMetric().f1_score
]

# 使用可恢复的trainer
trainer = ResumableAccelerateTrainer(
    model=model,
    epochs=REMAINING_EPOCHS,  # 只训练剩余的epoch
    start_epoch=START_EPOCH,  # 设置起始epoch编号
    learning_rate=2e-5,
    batch_size=4,  # 降低batch_size避免OOM
    train_dataset=datasets["train"],
    eval_dataset=datasets["valid"],
    test_dataset=datasets["test"],
    compute_metrics=metric_functions,
    gradient_accumulation_steps=4,  # 有效batch_size = 4 * 4 = 16
    device=torch.device("cuda:0"),
    monitor='valid_accuracy_score',
    load_best_model_at_end=True,
)

print("\n🎓 开始训练...")
print(f"  - Epoch范围: {START_EPOCH + 1} 到 {TOTAL_EPOCHS}")
print(f"  - Batch size: 4 (有效batch size: 16)")
print(f"  - Learning rate: 2e-5")
print("=" * 100)

# 训练
metrics = trainer.train(
    path_to_save="ogb_te_2class_finetuned_v1.5_seq_tissue_resumed",
    dataset_class=OmniDatasetWithTissue
)
print('\n📊 训练完成! Final Metrics:', metrics)

# ===== 训练完成后，查找历史最佳模型 =====
print("\n" + "=" * 100)
print("🔍 开始查找历史最佳模型...")
print("=" * 100)

checkpoint_dir = "/home/sw1136/OmniGenBench/examples/dingling_te_structure"
best_checkpoint = find_best_checkpoint(checkpoint_dir, metric_name="accuracy_score")

if best_checkpoint:
    print(f"\n🎯 使用历史最佳模型进行最终测试...")
    
    # 直接加载模型权重，避免labels参数问题
    best_model = OmniModelForSequenceClassificationWithTissue(
        model_name_or_path,
        tokenizer,
        num_labels=len(list(label2id.keys())),
    )
    best_model_path = os.path.join(best_checkpoint, "pytorch_model.bin")
    if os.path.exists(best_model_path):
        state_dict = torch.load(best_model_path, map_location="cpu")
        best_model.load_state_dict(state_dict)
        print("✅ 最佳模型加载成功")
        
        # 使用最佳模型创建新的trainer进行测试
        test_trainer = AccelerateTrainer(
            model=best_model,
            epochs=1,
            learning_rate=2e-5,
            batch_size=8,
            train_dataset=datasets["train"],
            eval_dataset=datasets["valid"],
            test_dataset=datasets["test"],
            compute_metrics=metric_functions,
            device=torch.device("cuda:0"),
        )
        
        final_test_metrics = test_trainer.test()
        print("\n" + "=" * 100)
        print("📊 使用历史最佳模型的测试集结果：")
        print("=" * 100)
        for key, value in final_test_metrics.items():
            print(f"  {key}: {value:.4f}")
        print("=" * 100)
        
        # 构建最终结果文件夹名称
        accuracy = final_test_metrics.get('test_accuracy_score', 0.0)
        f1 = final_test_metrics.get('test_f1_score', 0.0)
        final_result_folder = os.path.join(
            checkpoint_dir,
            f"ogb_te_2class_finetuned_v1.5_seq_tissue_seed_42_accuracy_score_{accuracy:.4f}_f1_score_{f1:.4f}"
        )
        
        # 创建文件夹
        os.makedirs(final_result_folder, exist_ok=True)
        
        # 保存测试结果到JSON文件
        result_json_path = os.path.join(final_result_folder, "final_test_results.json")
        checkpoint_name = os.path.basename(best_checkpoint)
        
        result_data = {
            "best_checkpoint_path": best_checkpoint,
            "best_checkpoint_name": checkpoint_name,
            "test_metrics": final_test_metrics,
            "training_config": {
                "model_name": model_name_or_path,
                "total_epochs": TOTAL_EPOCHS,
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "learning_rate": 2e-5,
            }
        }
        
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        # 保存最佳模型到最终结果文件夹
        test_trainer.save_model(final_result_folder, dataset_class=OmniDatasetWithTissue)
        
        print(f"\n💾 最终测试结果已保存到文件夹: {final_result_folder}")
        print(f"   - 测试结果JSON: {result_json_path}")
        print(f"   - 模型文件: {os.path.join(final_result_folder, 'pytorch_model.bin')}")
        
    else:
        print(f"❌ 未找到最佳模型权重文件: {best_model_path}")
else:
    print("❌ 未找到历史checkpoint，使用当前训练的模型")
