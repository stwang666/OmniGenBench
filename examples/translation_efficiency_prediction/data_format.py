# --- 准备工作 ---
from omnigenbench import OmniTokenizer, OmniModelForSequenceClassification
import torch

# 加载分词器和模型
model_name = "yangheng/OmniGenome-52M"
tokenizer = OmniTokenizer.from_pretrained(model_name)
model = OmniModelForSequenceClassification(model_name, tokenizer, num_labels=2)

# 准备一个样本序列
sequence = "AUGCAUGC"
inputs = tokenizer(sequence, return_tensors="pt")
input_ids = inputs['input_ids']

print(f"1. 原始序列: '{sequence}'")
print(f"2. 分词后的整数ID (Input IDs): {input_ids.tolist()}")
print(f"   - 数据类型: {input_ids.dtype}")
print(f"   - 形状 (Shape): {input_ids.shape}")
print("-" * 30)

# --- 查看模型内部变化 ---
# 关闭梯度计算，我们只做前向传播观察
with torch.no_grad():
    # 手动通过 Embedding 层
    # model.model 访问的是基础的 OmniGenome 模型，不包括最后的分类头
    embedding_layer = model.model.embeddings
    embeddings = embedding_layer(input_ids)

    print("3. 经过 Embedding 层后的高级特征:")
    print(f"   - 数据类型: {embeddings.dtype}")
    print(f"   - 形状 (Shape): {embeddings.shape}")
    print("   - 第一个核苷酸 'A' 对应的特征向量 (只显示前10个值):")
    print(f"     {embeddings[0, 1, :10].tolist()}")  # [0, 1, ..] 是因为 tokenizer 会在前面加一个特殊token
    print("-" * 30)

    # 完整通过模型，获取最终输出
    outputs = model(**inputs)

    print("4. 模型的最终原始输出 (Logits):")
    print(f"   - Logits: {outputs.logits.tolist()}")
    print(f"   - 形状 (Shape): {outputs.logits.shape}")