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


# # 自定义数据集类：使用已有的结构信息，而不是自动预测
# class OmniDatasetWithExistingStructure(OmniDatasetForSequenceClassification):
#     """
#     使用数据中已有的结构信息的数据集类。
    # 如果数据中包含 'structure' 字段，直接使用；否则回退到自动预测。
    # """
    
    # def _preprocessing(self):
    #     """
    #     重写预处理方法：优先使用数据中已有的结构信息。
    #     """
    #     for idx, ex in enumerate(self.examples):
    #         # 处理不同的序列字段名（SEQ, seq, sequence, text）
    #         if "SEQ" in self.examples[idx]:
    #             self.examples[idx]["sequence"] = self.examples[idx]["SEQ"]
    #             del self.examples[idx]["SEQ"]
    #         if "seq" in self.examples[idx]:
    #             self.examples[idx]["sequence"] = self.examples[idx]["seq"]
    #             del self.examples[idx]["seq"]
    #         if "text" in self.examples[idx]:
    #             self.examples[idx]["sequence"] = self.examples[idx]["text"]
    #             del self.examples[idx]["text"]

    #         if "sequence" not in self.examples[idx]:
    #             import warnings
    #             warnings.warn("The 'sequence' field is missing in the raw dataset.")
        
    #     if len(self.examples) > 0 and "sequence" in self.examples[0]:
    #         sequences = [ex["sequence"] for ex in self.examples]
    #         if self.structure_in:
    #             # 检查数据中是否已经包含结构信息（支持不同的大小写）
    #             has_structure = False
    #             structure_key = None
    #             for key in ["structure", "Structure", "STRUCTURE"]:
    #                 if key in self.examples[0]:
    #                     has_structure = True
    #                     structure_key = key
    #                     break
                
    #             if has_structure:
    #                 # 使用已有的结构信息
    #                 for idx, ex in enumerate(self.examples):
    #                     structure = ex.get(structure_key, "")
    #                     sequence = ex["sequence"]
    #                     self.examples[idx]["sequence"] = f"{sequence}{self.tokenizer.eos_token}{structure}"
    #             else:
    #                 # 如果没有结构信息，则自动预测（回退到原始行为）
    #                 structures = self.rna2structure.fold(sequences)
    #                 for idx, (sequence, structure) in enumerate(zip(sequences, structures)):
    #                     self.examples[idx]["sequence"] = f"{sequence}{self.tokenizer.eos_token}{structure}"

# model_name_or_path = "yangheng/OmniGenome-52M"
model_name_or_path = "yangheng/OmniGenome-v1.5"
# dataset_name = "translation_efficiency_prediction"

# Model and Tokenizer

# We define the label mapping in the training
label2id = {"0": 0, "1": 1}  # 0: Low TE, 1: High TE

# Initialize tokenizer
tokenizer = OmniTokenizer.from_pretrained(model_name_or_path)

# Load datasets
# 使用自定义数据集类的 from_hub 方法，它会自动使用已有的结构信息
# 注意：当 structure_in=True 时，需要增加 max_length 以容纳序列+结构信息
# 如果序列长度约512，结构信息长度也约512，建议设置 max_length=1024 或更大
# 但要注意不能超过模型的最大位置嵌入限制（model.config.max_position_embeddings）
datasets = OmniDatasetForSequenceClassification.from_hub(
    "/home/sw1136/OmniGenBench/examples/dingling_te_structure/biclass_seq", # 指定具体的数据目录
    tokenizer=tokenizer,
    max_length=512,  # 增加 max_length 以容纳序列+结构信息（原512 + 结构信息512 ≈ 1024）
    label2id=label2id,
    # structure_in=True,  # 启用结构信息：使用数据中已有的结构信息（如果存在），否则自动预测
)
print(f"📊 Loaded datasets: {list(datasets.keys())}")
for split, dataset in datasets.items():
    print(f"  - {split}: {len(dataset)} samples")

# Step 2: Model Initialization
# === Model Initialization ===
# We support all genomic foundation models from Hugging Face Hub.
model = OmniModelForSequenceClassification(
    model_name_or_path,
    tokenizer,
    num_labels=len(list(label2id.keys())),  # Binary classification: Low TE vs High TE
)

# Step 3: Model Training
metric_functions = [
    ClassificationMetric().accuracy_score,  # 准确率：正确预测的样本数 / 总样本数
    ClassificationMetric().f1_score]


trainer = AccelerateTrainer(
    model=model,
    epochs=60,
    learning_rate=2e-5,
    batch_size=8,
    train_dataset=datasets["train"],
    eval_dataset=datasets["valid"],
    test_dataset=datasets["test"],
    compute_metrics=metric_functions,
    gradient_accumulation_steps=8,
    device=torch.device("cuda:0"),
    # 早停和最佳模型配置
    early_stopping=True,
    patience=8,  # 8个epoch验证集准确率不提升就停止
    monitor='valid_accuracy_score',  # 监控验证集准确率
    load_best_model_at_end=True,  # 训练结束自动加载最佳模型
)
print("🎓 Starting training...")

# metrics = trainer.train()
# trainer.save_model("ogb_te_finetuned")

# # trainer.save_model(path_to_save="ogb_te_3class_finetuned", dataset_class=TriClassTEDataset)
metrics = trainer.train(path_to_save="ogb_te_2class_finetuned_v1.5_seq", dataset_class=OmniDatasetForSequenceClassification)
print('Final Metrics:', metrics)


# # Step 4: Model Inference and Interpretation
# inference_model = ModelHub.load("yangheng/ogb_te_finetuned")

# sample_sequences = {
#     "Optimized sequence": "AAACCAACAAAATGCAGTAGAAGTACTCTCGAGCTATAGTCGCGACGTGCTGCCCCGCAGGAGTACAGTAGTAGTACAACGTAAGCGGGAGCAACAGACTCCCCCCCTGCAACCCACTGTGCCTGTGCCCTCGACGCGTCTCCGTCGCTTTGGCAAATGTCACGTACATATTACCGTCTCAGGCTCTCAGCCATGCTCCCTACCACCCCTGCAGCGAAGCAAAAGCCACGCACGCGGCGCCTGACATGTAACAGGACTAGACCATCTTGTTCATTTCCCGCACCCCCTCCTCTCCTCTTCCTCCATCTGCCTCTTTAAAACAGTAAAAATAACCGTGCATCCCCTGGGCAAAATCTCTCCCATACATACACTACAGCGGCGAACCTTTCCTTATTCTCGCAACGCCTCGGTAACGGGCAGCGCCTGCTCCGCGCCGCGGTTGCGAGTTCGGGAAGGCGGCCGGAGTCGCGGGGAGGAGAGGGAGGATTCGATCGGCCAGA",
#     "Suboptimal sequence": "TGGAGATGGGCAGATGGCACACAAAACATGAATAGAAAACCCAAAAGGAAGGATGAAAAAAACACACACACACACACACACAAAACACAGAGAGAGAGAGAGAGAGAGCGAGAAAAGAAAAGAAAAAACCAATTCTTTTGGTCTCTTCCCTCTCCGTTTGTCGTGTCGAAGCCTTTGCCCCCACCACCTCCTCCTCTCCTCTCCCTTCCTCCCCTCCTCCCCATCTCGCTCTCCTCCCTCCTCTCTCCTCTCCTCGTCTCCTCTTCCTCTCCATTCCATTGGCCATTCCATTCCATTCCACCCCCCATGAAACCCCAAACCCTCGTCGGCCTCGCCGCGCTCGCGTAGCGCACCCGCCCTTCTCCTCTCGCCGGTGGTCCGCCGCCAGCCTCCCCCCACCCGATCCCGCCGCCCCCCCCGCCTTCACCCCGCCCACGCGGACGCATCCGATCCCGCCGCATCGCCGCGCGGGGGGGGGGGGGGGGGGGGGGGGGAGGGCACG",
#     "Random sequence": "AUGC" * (128 // 4),
# }
# for seq_name, sequence in sample_sequences.items():
#     outputs = inference_model.inference(sequence)

#     # —— Result Interpretation ——
#     prediction = outputs['predictions']
#     confidence = outputs['confidence']
#     print(f"  - Predicted Translation Efficiency: {prediction} (Confidence: {confidence:.2f})")
