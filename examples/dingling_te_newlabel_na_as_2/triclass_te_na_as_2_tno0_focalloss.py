# -*- coding: utf-8 -*-
# file: triclass_te_na_as_2.py
# time: 09:35 07/10/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.

import torch
import math
from omnigenbench import (
    ClassificationMetric,
    Trainer,
    ModelHub,
    OmniTokenizer,
    OmniDatasetForMultiLabelClassification,
    OmniModelForMultiLabelSequenceClassification,
    OmniPooling,
)

# model_name_or_path = "yangheng/OmniGenome-52M"
model_name_or_path = "yangheng/OmniGenome-v1.5" #和186M模型差不多，修改了一些东西
#model_name_or_path = "multimolecule/splicebert"
# model_name_or_path = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"

# Load tokenizer
tokenizer = OmniTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)


class TriClassTEDataset(OmniDatasetForMultiLabelClassification):
    """Dataset for 3-class (Low/Medium/High) multi-label TE classification
    
    继承说明：
    - 继承自 OmniDatasetForMultiLabelClassification，获得多标签分类数据集的基础功能
    - 重写 prepare_input() 方法以适配 TE 3分类任务的特殊需求
    """

    def __init__(self, **kwargs):
        # 调用父类的初始化方法，继承父类的属性和行为
        super().__init__(**kwargs)

    def prepare_input(self, instance, **kwargs):
        # Map labels to indices: 0=0, 1=1, 2=2, 空值=-100
        def safe_label_mapping(label_value):
            """安全地映射标签值，处理空值、NaN和字符串"""
            if label_value is None or label_value == '' or str(label_value).strip() == '':
                return -100  # 空值用-100表示，将被ignore
            label_str = str(label_value).strip()
            if label_str in ['0.0', '0']:
                return 0
            elif label_str in ['1.0', '1']:
                return 1
            elif label_str in ['2.0', '2']:
                return 2
            elif label_str.lower() in ['nan', 'na', 'null']:
                return -100
            else:
                # 对于其他情况，默认为空值
                return -100

        # Extract labels for all 9 tissues
        root_TE_label = safe_label_mapping(instance["root_TE_label"])
        seedling_TE_label = safe_label_mapping(instance["seedling_TE_label"])
        leaf_TE_label = safe_label_mapping(instance["leaf_TE_label"])
        FMI_TE_label = safe_label_mapping(instance["FMI_TE_label"])
        FOD_TE_label = safe_label_mapping(instance["FOD_TE_label"])
        Prophase_I_pollen_TE_label = safe_label_mapping(instance["Prophase-I-pollen_TE_label"])
        Tricellular_pollen_TE_label = safe_label_mapping(instance["Tricellular-pollen_TE_label"])
        flag_TE_label = safe_label_mapping(instance["flag_TE_label"])
        grain_TE_label = safe_label_mapping(instance["grain_TE_label"])
        sequence = instance["sequence"]

        # Tokenize sequence
        tokenized_inputs = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Stack all labels
        labels = torch.tensor([
            root_TE_label,
            seedling_TE_label,
            leaf_TE_label,
            FMI_TE_label,
            FOD_TE_label,
            Prophase_I_pollen_TE_label,
            Tricellular_pollen_TE_label,
            flag_TE_label,
            grain_TE_label,
        ], dtype=torch.long)  # Use long for CrossEntropyLoss

        tokenized_inputs["labels"] = labels

        return tokenized_inputs

class FocalLoss(torch.nn.Module):
    """Focal Loss for multi-class classification"""
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 类别权重，可以是tensor或None
        self.gamma = gamma  # 聚焦参数
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # inputs: [batch * num_labels, num_classes]
        # targets: [batch * num_labels]
        
        # 🔧 检查输入是否包含 NaN 或 inf
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            print("⚠️ Warning: inputs contains NaN or inf!")
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 处理ignore_index（先创建掩码）
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index)
        else:
            mask = torch.ones_like(targets, dtype=torch.bool)
        
        # 🔧 防止所有样本都被忽略的情况
        valid_count = mask.float().sum()
        if valid_count == 0:
            # 🔧 如果所有样本都被忽略，返回 0.0（这会导致 loss 从 0 开始）
            # 这通常发生在某些 batch 中所有标签都是 -100 的情况
            return torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype, requires_grad=True)
        
        # 计算交叉熵
        # F.cross_entropy 可以处理 [N, C] 形状的inputs和 [N] 形状的targets
        # 其中 N = batch * num_labels, C = num_classes
        # 这里 inputs 是 [batch * num_labels, num_classes]，targets 是 [batch * num_labels]
        # 完全符合 cross_entropy 的输入要求
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, 
                                   ignore_index=self.ignore_index, 
                                   reduction='none')
        
        # 🔧 检查 ce_loss 是否包含 NaN 或 inf
        if torch.isnan(ce_loss).any() or torch.isinf(ce_loss).any():
            print("⚠️ Warning: ce_loss contains NaN or inf!")
            ce_loss = torch.nan_to_num(ce_loss, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 计算概率（使用 clamp 防止数值溢出）
        # p_t = exp(-ce_loss)，当 ce_loss 很大时，p_t 接近 0
        # 使用 clamp 确保数值稳定，并限制 ce_loss 的范围
        ce_loss_clamped = ce_loss.clamp(max=50.0)  # 防止 exp 溢出
        p_t = torch.exp(-ce_loss_clamped).clamp(min=1e-4, max=1.0-1e-4)
        
        # 计算focal loss
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        
        # 🔧 检查 focal_loss 是否包含 NaN 或 inf
        if torch.isnan(focal_loss).any() or torch.isinf(focal_loss).any():
            print("⚠️ Warning: focal_loss contains NaN or inf!")
            focal_loss = torch.nan_to_num(focal_loss, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 应用类别权重（如果提供）
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                # 🔧 FIX: 标量需要转换为与 targets 相同形状的张量
                alpha_t = torch.full_like(targets, float(self.alpha), dtype=torch.float32)
            else:
                # 🔧 FIX: 确保 alpha 和 targets 在同一个设备上
                alpha = self.alpha.to(targets.device)
                # 对于 -100 (ignore_index)，使用默认权重 1.0；其他值正常索引
                valid_mask = (targets >= 0) & (targets < len(alpha))
                alpha_t = torch.ones_like(targets, dtype=torch.float32)
                alpha_t[valid_mask] = alpha[targets[valid_mask]]
            focal_loss = alpha_t * focal_loss
        
        # 🔧 再次检查 focal_loss
        if torch.isnan(focal_loss).any() or torch.isinf(focal_loss).any():
            print("⚠️ Warning: focal_loss after alpha contains NaN or inf!")
            focal_loss = torch.nan_to_num(focal_loss, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 应用掩码（将 ignore_index 位置的 loss 设为 0）
        focal_loss = focal_loss * mask.float()
        
        # 计算最终的 loss
        if self.reduction == 'mean':
            if self.ignore_index is not None:
                # 🔧 FIX: 防止除以零
                if valid_count > 0:
                    loss = focal_loss.sum() / valid_count
                else:
                    loss = torch.tensor(0.0, device=focal_loss.device, dtype=focal_loss.dtype, requires_grad=True)
            else:
                loss = focal_loss.mean()
        elif self.reduction == 'sum':
            loss = focal_loss.sum()
        else:
            loss = focal_loss
        
        # 🔧 最终检查
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ Warning: Final loss is NaN or inf! Returning 0.0")
            loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype, requires_grad=True)
        
        return loss

class OmniModelForTriClassTESequenceClassification(OmniModelForMultiLabelSequenceClassification):
    """Model for 3-class multi-label TE classification"""

    def __init__(self, config_or_model, tokenizer, num_labels=9, num_classes=3, *args, **kwargs):
        # For multi-label with 3 classes each, output should be [batch, num_labels, num_classes]
        super().__init__(config_or_model, tokenizer, num_labels=num_labels * num_classes, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.num_labels = num_labels  # 9 tissues
        self.num_classes = num_classes  # 3 classes (0/1/NA)
        self.pooler = OmniPooling(self.config)
        self.classifier = torch.nn.Linear(self.config.hidden_size, self.num_classes * self.num_labels)
        
        # 🔑 使用 Focal Loss 替代 CrossEntropyLoss
        # alpha: 类别权重，可以是tensor([1.0, 1.0, 1.0])或None
        # gamma: 聚焦参数，通常设置为2.0
        self.loss_fn = FocalLoss(
            alpha=torch.tensor([1.3925, 1.4825, 0.217], dtype=torch.float32),  # 🔧 L1归一化权重：从[1.3925, 1.4825, 0.217]归一化到[0.4504, 0.4795, 0.0702]
            # 归一化方法：每个权重除以总和(3.092)，使得权重和为1.0，同时保持相对比例不变(6.42:6.83:1.0)
            # 优点：权重范围更小(0-1)，降低梯度幅度，提高数值稳定性，同时保持类别间的相对重要性
            gamma=2.0,   # 聚焦参数，值越大，对难样本的关注越多
            ignore_index=-100,
            reduction="mean"
        )

        # 🔑 NEW: Store dataset class reference for saving
        self.dataset_class = kwargs.pop('dataset_class', TriClassTEDataset)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass with proper reshaping for multi-label multi-class"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        # Get the logits from classifier head
        base_logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        # 🔧 检查基础模型输出
        if torch.isnan(base_logits).any() or torch.isinf(base_logits).any():
            print("⚠️ Warning: base_logits (from model) contains NaN or inf!")
            # 🔧 检查基础模型权重是否包含 NaN/inf（可能是梯度爆炸导致）
            nan_params = []
            inf_params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if torch.isnan(param).any():
                        nan_params.append(name)
                    if torch.isinf(param).any():
                        inf_params.append(name)
            if nan_params:
                print(f"⚠️ CRITICAL: Model parameters with NaN: {nan_params[:5]}...")  # 只显示前5个
            if inf_params:
                print(f"⚠️ CRITICAL: Model parameters with inf: {inf_params[:5]}...")  # 只显示前5个
            # 🔧 检查输入是否异常
            if torch.isnan(input_ids).any() or torch.isinf(input_ids.float()).any():
                print("⚠️ Warning: input_ids contains NaN or inf!")
            base_logits = torch.nan_to_num(base_logits, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Pooling
        pooled_output = self.pooler(input_ids, base_logits)
        
        # 🔧 检查 pooling 输出
        if torch.isnan(pooled_output).any() or torch.isinf(pooled_output).any():
            print("⚠️ Warning: pooled_output contains NaN or inf!")
            pooled_output = torch.nan_to_num(pooled_output, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Classifier
        # 🔧 先检查 classifier 权重是否包含 NaN/inf（在计算之前检查）
        classifier_has_nan = False
        for name, param in self.classifier.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"⚠️ Warning: classifier.{name} contains NaN or inf!")
                classifier_has_nan = True
        
        logits = self.classifier(pooled_output)
        
        # 🔧 检查 classifier 输出
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("⚠️ Warning: logits (from classifier) contains NaN or inf! Clamping values...")
            # 🔧 如果 classifier 权重有 NaN/inf，这是严重问题，需要停止训练
            if classifier_has_nan:
                print("⚠️ CRITICAL: Classifier weights contain NaN/inf! Model parameters are corrupted!")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Reshape logits from [batch, num_labels * num_classes] to [batch, num_labels, num_classes]
        batch_size = logits.shape[0]
        logits = logits.view(batch_size, self.num_labels, self.num_classes)

        loss = None
        if labels is not None:
            # labels shape: [batch, num_labels]
            # Flatten for CrossEntropyLoss
            logits_flat = logits.view(-1, self.num_classes)  # [batch * num_labels, num_classes]
            labels_flat = labels.view(-1)  # [batch * num_labels]

            # # 🔧 FIX: 使用带权重的loss计算，权重会自动在正确的设备上
            # # 
            # # 权重传递示例：
            # # 假设 self.class_weights = tensor([1.0, 1.0, 0.1])
            # # 假设 labels_flat = tensor([0, 1, 2, 0, 1, -100])  # 6个样本
            # # 
            # # CrossEntropyLoss 会为每个样本应用对应类别的权重：
            # # - 样本1 (label=0): 使用权重 1.0
            # # - 样本2 (label=1): 使用权重 1.0
            # # - 样本3 (label=2): 使用权重 0.1  ← NA类别，权重降低
            # # - 样本4 (label=0): 使用权重 1.0
            # # - 样本5 (label=1): 使用权重 1.0
            # # - 样本6 (label=-100): 被ignore_index忽略，不参与loss计算
            # # 
            # # 最终loss = (loss_1 * 1.0 + loss_2 * 1.0 + loss_3 * 0.1 + loss_4 * 1.0 + loss_5 * 1.0) / 5
            # # 注意：被忽略的样本不计入分母
            # loss = torch.nn.functional.cross_entropy(
            #     logits_flat, 
            #     labels_flat, 
            #     weight=self.class_weights,  # tensor([1.0, 1.0, 0.1]) 自动匹配到GPU
            #     ignore_index=-100,
            #     reduction="mean"
            # )

            # 🔧 确认损失函数调用：logits_flat 和 labels_flat 的形状正确
            # logits_flat: [batch * num_labels, num_classes]
            # labels_flat: [batch * num_labels]
            loss = self.loss_fn(logits_flat, labels_flat)

        return {
            "loss": loss,
            "logits": logits,
            "last_hidden_state": outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else None,
        }

    def predict(self, sequence_or_inputs, **kwargs):
        """Prediction with softmax for multi-class"""
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        # Apply softmax to get probabilities for each label
        probabilities = torch.softmax(logits, dim=-1)  # [batch, num_labels, num_classes]

        # Get predicted class for each label
        predictions = torch.argmax(probabilities, dim=-1)  # [batch, num_labels]

        outputs = {
            "predictions": predictions,
            "logits": logits,
            "probabilities": probabilities,
            "last_hidden_state": last_hidden_state,
        }

        return outputs

    def inference(self, sequence_or_inputs, **kwargs):
        """Inference wrapper"""
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        # Apply softmax
        probabilities = torch.softmax(logits, dim=-1)

        # Get predictions
        predictions = torch.argmax(probabilities, dim=-1)

        # Get confidence (max probability for each label)
        confidence, _ = torch.max(probabilities, dim=-1)

        if not isinstance(sequence_or_inputs, list):
            outputs = {
                "predictions": predictions[0],
                "logits": logits[0],
                "probabilities": probabilities[0],
                "confidence": confidence[0],
                "last_hidden_state": last_hidden_state[0] if last_hidden_state is not None else None,
            }
        else:
            outputs = {
                "predictions": predictions,
                "logits": logits,
                "probabilities": probabilities,
                "confidence": confidence,
                "last_hidden_state": last_hidden_state,
            }

        return outputs


# Load datasets
print("📊 Loading datasets...")
datasets = TriClassTEDataset.from_hub(
    "/home/sw1136/OmniGenBench/examples/dingling_te_newlabel_na_as_2/preprocess_data_Tno0_02/original_data",  # 指定具体的数据目录
    tokenizer=tokenizer,
    max_length=512,
    force_padding=False
)

print("📝 Data loading completed!")
print(f"📊 Loaded datasets: {list(datasets.keys())}")
for split, dataset in datasets.items():
    print(f"  - {split}: {len(dataset)} samples")

# Initialize model
print("\n🚀 Initializing model...")
model = OmniModelForTriClassTESequenceClassification(
    model_name_or_path,
    tokenizer,
    num_labels=9,  # 9 tissues
    num_classes=3,  # 3 classes: 0, 1, NA
    trust_remote_code=True
)

# 🔧 模型自检：检查模型初始化后是否包含 NaN/inf
print("\n🔍 执行模型自检...")

# 1. 检查模型参数是否包含 NaN/inf
print("  1. 检查模型参数...")
nan_params = []
inf_params = []
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        nan_params.append(name)
    if torch.isinf(param).any():
        inf_params.append(name)

if nan_params:
    print(f"  ❌ 发现 {len(nan_params)} 个参数包含 NaN:")
    for name in nan_params[:10]:  # 只显示前10个
        print(f"     - {name}")
    raise RuntimeError("模型参数包含 NaN！请重新初始化模型或检查预训练权重。")
if inf_params:
    print(f"  ❌ 发现 {len(inf_params)} 个参数包含 inf:")
    for name in inf_params[:10]:  # 只显示前10个
        print(f"     - {name}")
    raise RuntimeError("模型参数包含 inf！请重新初始化模型或检查预训练权重。")
print("  ✅ 模型参数检查通过：无 NaN/inf")

# 2. 检查模型前向传播是否正常
print("  2. 检查模型前向传播...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 获取一个样本进行测试
try:
    sample = datasets["train"][0]
    # 数据集返回的是 OmniDict，可以直接访问字段
    input_ids = sample["input_ids"]
    attention_mask = sample.get("attention_mask", None)
    labels = sample.get("labels", None)
    
    # 添加 batch 维度并移动到设备
    input_ids = input_ids.unsqueeze(0).to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.unsqueeze(0).to(device)
    if labels is not None:
        labels = labels.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # 检查输出是否包含 NaN/inf
        for name, tensor in outputs.items():
            if tensor is not None:
                if torch.isnan(tensor).any():
                    raise RuntimeError(f"模型输出 {name} 包含 NaN！")
                if torch.isinf(tensor).any():
                    raise RuntimeError(f"模型输出 {name} 包含 inf！")
        
        print(f"  ✅ 前向传播检查通过：loss = {outputs['loss'].item():.6f}")
except Exception as e:
    print(f"  ⚠️ 前向传播测试跳过：{e}")
    print("  （这可能是数据格式问题，不影响训练）")

# 3. 检查损失函数是否正常
print("  3. 检查损失函数...")
try:
    # 创建测试输入
    test_logits = torch.randn(9, 3).to(device)  # [num_labels, num_classes]
    test_labels = torch.randint(0, 3, (9,)).to(device)  # [num_labels]
    test_labels[0] = -100  # 测试 ignore_index
    
    test_loss = model.loss_fn(test_logits, test_labels)
    if torch.isnan(test_loss) or torch.isinf(test_loss):
        raise RuntimeError(f"损失函数返回 NaN/inf：{test_loss}")
    print(f"  ✅ 损失函数检查通过：test_loss = {test_loss.item():.6f}")
except Exception as e:
    print(f"  ❌ 损失函数检查失败：{e}")
    raise

print("✅ 模型自检全部通过！可以开始训练。\n")

# Define metrics: accuracy and F1 score
# - accuracy_score: 计算整体分类准确率，忽略标签为-100的样本（通常用于padding或无效标签）
# - f1_score: 计算F1分数（精确率和召回率的调和平均），使用macro平均（对每个类别计算F1后取平均，适合类别不平衡的情况）
# 
# 计算时机：这些metrics会在训练过程中应用于eval_dataset（验证集）和test_dataset（测试集）
# 计算方式：
#   1. 模型对验证集/测试集进行前向传播，得到预测结果（logits）
#   2. 将logits转换为预测类别（argmax）
#   3. 将预测类别与真实标签进行比较，忽略标签为-100的位置
#   4. accuracy_score: 正确预测数 / 有效样本总数
#   5. f1_score (macro): 对每个类别分别计算F1值，然后取平均值
metric_functions = [
    ClassificationMetric(ignore_y=-100).accuracy_score,  # 准确率：正确预测的样本数 / 总样本数
    ClassificationMetric(ignore_y=-100, average='macro').f1_score,  # 宏平均F1：(TP) / (TP + 0.5*(FP+FN))，对所有类别取平均
    ClassificationMetric(ignore_y=-100).classification_report,
]

# Initialize trainer
 # batch_size: 每次从数据集中加载并处理的样本数量
    # - 这里设置为16，表示每个训练步骤会处理16个序列样本
    # - 较小的batch_size可以减少GPU内存占用，但训练可能不够稳定
    # - 较大的batch_size可以提高训练稳定性和速度，但需要更多GPU内存

# gradient_accumulation_steps: 梯度累积步数
    # - 这里设置为4，表示每4个batch才进行一次参数更新
    # - 实际有效batch_size = batch_size × gradient_accumulation_steps = 16 × 4 = 64
    # - 作用：在GPU内存有限的情况下，通过累积多个小batch的梯度来模拟大batch训练
    # - 工作原理：
    #   1. 前向传播和反向传播计算梯度（但不更新参数）
    #   2. 将梯度累加到之前的梯度上
    #   3. 重复步骤1-2共4次
    #   4. 第4次后，使用累积的梯度更新模型参数，然后清零梯度
    # - 优点：可以用较小的GPU内存训练出与大batch相当的效果
    
# 训练时不会用到test_dataset，它仅在训练完成后用于最终评估
# - train_dataset: 用于模型训练，更新模型参数
# - eval_dataset: 用于训练过程中的验证，监控过拟合，选择最佳模型
# - test_dataset: 仅在训练完成后用于最终性能评估，不参与训练过程

trainer = Trainer(
    model=model,
    epochs=30,
    learning_rate=5e-6,  # 🔧 大幅降低学习率：从 2e-5 降到 5e-6，防止梯度爆炸导致模型参数变成 NaN/inf
    batch_size=16,  # 每次训练的样本数量
    train_dataset=datasets["train"],
    eval_dataset=datasets["valid"],
    test_dataset=datasets["test"],  # 仅用于训练后的最终测试，不影响训练过程
    compute_metrics=metric_functions,
    gradient_accumulation_steps=1,  # 🔧 关闭梯度累积，减少累积梯度导致的数值不稳定
    max_grad_norm=0.3,  # 🔧 启用并增强梯度裁剪：设置为 0.3，非常严格地限制梯度，防止梯度爆炸
    # 说明：
    # - 不是限制单个梯度的值，而是限制所有梯度的 L2 范数（总长度）
    # - 如果梯度的 L2 范数 > 0.3，会将所有梯度按比例缩放，使得总范数 = 0.3
    # - 例如：如果梯度范数是 1.0，会缩放为 0.3（缩放因子 = 0.3/1.0 = 0.3）
    # - 非常低的阈值（0.3）意味着非常严格地限制梯度，防止梯度爆炸
    # - 这有助于解决 Focal Loss + alpha 权重导致的 NaN/inf 问题
    weight_decay=0.01,  # 添加权重衰减
    early_stopping=True,
    patience=3,  # 验证集指标3个epoch不提升就停止
    monitor='valid_f1_score',  # 监控验证集F1分数
    warmup_steps=100,  # 学习率预热
    autocast="float32",  # 🔧 关闭混合精度：设置为 "float32" 或 "fp32" 使用全精度训练，避免 FP16 导致的数值不稳定和 NaN/inf
    # 注意：autocast=None 会被映射为默认的 float16，必须显式设置为 "float32" 才能关闭混合精度
    # eval_steps=50,  # 更频繁的验证
    # save_strategy="steps",
    # save_steps=50,
    # load_best_model_at_end=True,
    # metric_for_best_model="accuracy_score",
    # greater_is_better=True,
    # save_total_limit=3,
)
# trainer.save_model(path_to_save="ogb_te_3class_finetuned", dataset_class=TriClassTEDataset)
metrics = trainer.train(path_to_save="ogb_te_3class_finetuned_na_as_2_tno0_02_v1.5_focalloss", dataset_class=TriClassTEDataset)
print('📊 Final Metrics:', metrics)

# === Model Inference ===
print("\n🔮 Starting inference on test samples...")

inference_model = ModelHub.load("/home/sw1136/OmniGenBench/examples/dingling_te/ogb_te_3class_finetuned_epoch_19_seed_42_accuracy_score_0.9900_seed_42_f1_score_0.9900")

# Get some test samples
# sample_sequences = datasets['test'].sample(1000).examples
#sample_sequences = datasets['valid'].sample(1000).examples
sample_sequences = datasets['train'].examples[:1]

label_names = ['0.0', '1.0', '2.0']
tissue_names = [
    'root', 'seedling', 'leaf', 'FMI', 'FOD',
    'Prophase-I-pollen', 'Tricellular-pollen', 'flag', 'grain'
]

with torch.no_grad():
    for row in sample_sequences:
        sequence = row["sequence"]
        print(f"\n{'='*60}")
        print(f"🧬 Sample ID: {row['ID']}")
        print(f"📏 Sequence length: {len(sequence)} bp")

        outputs = inference_model.inference(sequence, **row)
        predictions = outputs['predictions'].cpu().numpy() # tensor([0, 2, 1, 2, 0, 2, 2, 1, 2], device='cuda:0') 9个tissue的预测类别
        probabilities = outputs['probabilities'].cpu().numpy() # 9*3的tensor，每个tissue的3个类别的概率 （logits --> softmax）
        confidence = outputs['confidence'].cpu().numpy() # 9个tissue的预测置信度 tensor([0.9990, 1.0000, 0.5112, 0.9834, 1.0000, 0.9985, 0.9990, 0.9995, 1.0000], probabilities中的最大值
        # last_hidden_state = outputs['last_hidden_state'].cpu().numpy() # 9*512的tensor，每个tissue的512个token的隐藏状态


        print(f"\n📊 Predictions for 9 tissues:")
        for i, tissue in enumerate(tissue_names):
            pred_class = predictions[i]
            pred_label = label_names[pred_class]
            conf = confidence[i]
            probs = probabilities[i]

            # Get ground truth if available
            gt_col = f"{tissue}_TE_label"
            if gt_col in row:
                gt_label = row[gt_col]
                if isinstance(gt_label, float) and math.isnan(gt_label):
                    continue
                match_emoji = "✅" if pred_label == gt_label else "❌"
                print(f"  {match_emoji} {tissue:25s}: {pred_label:6s} (conf: {conf:.3f}) [GT: {gt_label}]")
            else:
                print(f"  🔹 {tissue:25s}: {pred_label:6s} (conf: {conf:.3f})")

            # Show probability distribution
            print(f"      Probs - 0: {probs[0]:.3f}, 1: {probs[1]:.3f}, 2: {probs[2]:.3f}")

print("\n🎉 All tasks completed!")