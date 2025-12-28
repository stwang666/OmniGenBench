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

# model_name_or_path = "yangheng/OmniGenome-52M"
# model_name_or_path = "yangheng/OmniGenome-v1.5"
model_name_or_path = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
# dataset_name = "translation_efficiency_prediction"

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
        
        # 重新定义pooler的输入维度（需要自定义pooler来处理扩展后的hidden_size）
        # 但为了简化，我们创建一个新的pooler来处理拼接后的hidden state
        # 实际上，我们可以先拼接，然后pooler只对原始部分池化，最后再拼接tissue嵌入
        # 或者，我们可以修改pooler来支持扩展的hidden_size
        
        # 重新定义classifier，因为输入维度变为 hidden_size + tissue_embed_dim
        self.classifier = nn.Linear(
            self.config.hidden_size + self.tissue_embed_dim,
            self.config.num_labels
        )
        
        # 🔑 NEW: Store dataset class reference for saving
        self.dataset_class = kwargs.pop('dataset_class', OmniDatasetWithTissue)

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
        
        # 6. 计算损失（如果提供了labels）
        loss = None
        if labels is not None:
            # Use CrossEntropyLoss for binary classification
            logits_flat = logits.view(-1, self.config.num_labels)  # [batch, num_labels]
            labels_flat = labels.view(-1)  # [batch]
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits_flat, labels_flat)
        
        outputs = {
            "loss": loss,
            "logits": logits,
            # 如果下游需要 last_hidden_state，返回原始的即可
            "last_hidden_state": last_hidden_state, 
        }
        return outputs
    
    # def forward(self, **inputs):
    #     """
    #     Forward pass with tissue embedding.
    #     在pooler之前将tissue嵌入拼接到last_hidden_state的每个token位置。
    #     """
    #     labels = inputs.pop("labels", None)
    #     tissue_id = inputs.pop("tissue_id", None)
        
    #     # 获取last_hidden_state: (batch_size, seq_len, hidden_size)
    #     last_hidden_state = self.last_hidden_state_forward(**inputs)
    #     last_hidden_state = self.dropout(last_hidden_state)
    #     last_hidden_state = self.activation(last_hidden_state)
        
    #     # 获取tissue嵌入并扩展到每个token位置
    #     if tissue_id is not None:
    #         # 处理tissue_id可能是[batch_size, 1]的情况（DataLoader批处理后的形状）
    #         if tissue_id.ndim > 1:
    #             tissue_id = tissue_id.squeeze(-1)  # 压缩为[batch_size]
    #         tissue_embed = self.tissue_embedding(tissue_id)  # (batch_size, tissue_embed_dim)
    #         # 扩展到每个token位置: (batch_size, 1, tissue_embed_dim) -> (batch_size, seq_len, tissue_embed_dim)
    #         seq_len = last_hidden_state.shape[1]
    #         tissue_embed_expanded = tissue_embed.unsqueeze(1).expand(-1, seq_len, -1)
    #         # 拼接到last_hidden_state的每个token位置
    #         last_hidden_state = torch.cat([last_hidden_state, tissue_embed_expanded], dim=-1)  # (batch_size, seq_len, hidden_size + tissue_embed_dim)
    #     else:
    #         # 如果没有tissue_id，使用零向量
    #         batch_size, seq_len = last_hidden_state.shape[0], last_hidden_state.shape[1]
    #         device = last_hidden_state.device
    #         zero_tissue_embed = torch.zeros(batch_size, seq_len, self.tissue_embed_dim, device=device)
    #         last_hidden_state = torch.cat([last_hidden_state, zero_tissue_embed], dim=-1)
        
    #     # 池化操作：需要修改pooler来处理扩展后的hidden_size
    #     # 为了兼容现有的pooler，我们创建一个临时的inputs，但pooler内部会使用原始的hidden_size部分
    #     # 实际上，我们需要自定义pooler逻辑
    #     # 简单方法：先对原始部分进行池化，然后拼接tissue嵌入
    #     # 但这样tissue信息不会影响pooling过程
        
    #     # 更好的方法：创建一个自定义的pooling，或者修改pooler的输入
    #     # 为了简化，我们使用原始hidden_size部分进行pooling，然后拼接tissue嵌入
    #     original_hidden_size = self.config.hidden_size
    #     original_hidden_state = last_hidden_state[:, :, :original_hidden_size]
    #     pooled_state = self.pooler(inputs, original_hidden_state)  # (batch_size, hidden_size)
        
    #     # 从拼接后的hidden_state中提取tissue嵌入（取第一个token位置的tissue嵌入，因为所有token的tissue嵌入相同）
    #     tissue_embed_pooled = last_hidden_state[:, 0, original_hidden_size:]  # (batch_size, tissue_embed_dim)
        
    #     # 拼接pooled state和tissue嵌入
    #     last_hidden_state = torch.cat([pooled_state, tissue_embed_pooled], dim=-1)  # (batch_size, hidden_size + tissue_embed_dim)
        
    #     # 分类
    #     logits = self.classifier(last_hidden_state)
    #     logits = self.softmax(logits)
        
    #     outputs = {
    #         "logits": logits,
    #         "last_hidden_state": last_hidden_state,
    #         "labels": labels,
    #     }
    #     return outputs


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
datasets = OmniDatasetWithTissue.from_hub(
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
model = OmniModelForSequenceClassificationWithTissue(
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
    epochs=30,
    learning_rate=2e-5,
    batch_size=16,
    train_dataset=datasets["train"],
    eval_dataset=datasets["valid"],
    test_dataset=datasets["test"],
    compute_metrics=metric_functions,
    gradient_accumulation_steps=4,
    device=torch.device("cuda:0"),
    # 早停和最佳模型配置
    # early_stopping=True,
    # patience=8,  # 8个epoch验证集准确率不提升就停止
    monitor='valid_accuracy_score',  # 监控验证集准确率
    load_best_model_at_end=True,  # 训练结束自动加载最佳模型
)
print("🎓 Starting training...")

# metrics = trainer.train()
# trainer.save_model("ogb_te_finetuned")

# # trainer.save_model(path_to_save="ogb_te_3class_finetuned", dataset_class=TriClassTEDataset)
metrics = trainer.train(path_to_save="ogb_te_2class_finetuned_insta_seq_tissue", dataset_class=OmniDatasetWithTissue)
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
