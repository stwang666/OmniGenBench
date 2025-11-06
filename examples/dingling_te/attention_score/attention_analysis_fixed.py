import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

# 添加模型目录到路径，以便导入自定义模型
MODEL_PATH = "/home/sw1136/OmniGenBench/examples/dingling_te/ogb_te_3class_finetuned_epoch_19_seed_42_accuracy_score_0.9900_seed_42_f1_score_0.9900"
sys.path.insert(0, MODEL_PATH)

# 导入自定义模型和数据集类
from custom_model import OmniModelForTriClassTESequenceClassification
from custom_dataset import TriClassTEDataset

from omnigenbench import (
    OmniTokenizer,
    ModelHub
)

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

print("="*80)
print(" 🚀 启动正确加载的模型注意力诊断分析...")
print("="*80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" 正在加载模型: {MODEL_PATH}")
print(f" 目标设备: {device}")

# 🔑 关键：使用 ModelHub.load 正确加载自定义模型
print(" 正在使用 ModelHub 加载自定义模型（包含微调权重）...")
model = ModelHub.load(MODEL_PATH)

# 验证权重是否正确加载
print("\n 验证权重加载...")
saved_weights = torch.load(f"{MODEL_PATH}/pytorch_model.bin", map_location='cpu')

# 访问内部模型获取注意力权重
if hasattr(model, 'model') and hasattr(model.model, 'encoder'):
    encoder = model.model.encoder
elif hasattr(model, 'encoder'):
    encoder = model.encoder
else:
    print("❌ 无法找到 encoder，尝试访问包装器内部模型")
    if hasattr(model, 'model'):
        model = model.model
        if hasattr(model, 'model'):
            encoder = model.model.encoder
        else:
            encoder = model.encoder

# 对比第一层的query权重
try:
    loaded_query_weight = encoder.layer[0].attention.self.query.weight
    saved_query_weight = saved_weights['model.encoder.layer.0.attention.self.query.weight']
    
    # 确保在同一设备上比较
    loaded_query_weight_cpu = loaded_query_weight.detach().cpu().float()
    saved_query_weight_cpu = saved_query_weight.float()
    
    weights_match = torch.allclose(loaded_query_weight_cpu, saved_query_weight_cpu, rtol=1e-4, atol=1e-4)
    
    print(f"  ✓ 加载的权重形状: {loaded_query_weight.shape}")
    print(f"  ✓ 保存的权重形状: {saved_query_weight.shape}")
    print(f"  ✓ 权重匹配: {'✅ 是' if weights_match else '❌ 否'}")
    print(f"  ✓ 加载权重前5个值: {loaded_query_weight_cpu.flatten()[:5]}")
    print(f"  ✓ 保存权重前5个值: {saved_query_weight_cpu.flatten()[:5]}")
    
    if not weights_match:
        print("\n❌ 警告：加载的权重与保存的权重不匹配！")
        print("   这意味着您在分析的是未经微调的模型！")
        print("   注意力图将不会反映您微调后的模型行为！")
    else:
        print("\n✅ 确认：成功加载了微调后的权重！")
except Exception as e:
    print(f"  ❌ 权重验证失败: {e}")
    import traceback
    traceback.print_exc()

# 获取tokenizer
tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else OmniTokenizer.from_pretrained(MODEL_PATH)

model = model.to(device)
model.eval()

print(f"\n  模型加载成功!")
print(f"    模型类型: {model.__class__.__name__}")
print(f"    设备: {next(model.parameters()).device}")

# --- 准备要分析的序列 ---
SEQUENCE_TO_ANALYZE = "GTGGCGGCCGCTGCAAAACCCGGGGCGCGAGCCCGGGCGGAGCGGCCGTCGGTGCAGATCTTGGTGGTAGTAGCAAATATTCAAATGAGAACTTTGAAGGCCGAAGAGGAGAAAGGTTCCATGTGAACAGCACTTGCACATGGGTAAGCCGATTCTAAGGGACGGGGTAACCCCGGCAGATAGCGCGATCACGCGCATCCCCCGAAAGGGAATCGGGTTAAGATTTCCCGAGCCGGGATGTGGCAGTTGACGGTGACGTTAGGAAGTCCGGAGACGCCGGCGGGGGCCTCGGGAAGAGTTATCTTTTCTGCTTAACGGCCTGCCAACCCTGGAAACGGTTCAGCCGGAGGTAGGGTCCAGTGGCCGGAAGATCACCGGACGTCGCGCGGTGTCCAGTGCGCCCCCGGCGGCCCATGAAAATCCGGAGGACCGAGTACCGTTCACACCCGGTCGTACTCATAACCGCATCAGGTCTCCAAGGTGAACAGCCTCTGGCCA"

print(f"\n 正在分析序列（长度: {len(SEQUENCE_TO_ANALYZE)} bp）")
print(f"    序列预览: {SEQUENCE_TO_ANALYZE[:50]}...")

# --- 提取注意力分数 ---
# 🔑 对于自定义模型，需要手动提取注意力
print(" 正在提取注意力分数...")

try:
    # Tokenize输入
    inputs = tokenizer(
        SEQUENCE_TO_ANALYZE,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 前向传播并获取注意力
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # 提取注意力
    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        attentions = outputs.attentions
    elif isinstance(outputs, dict) and 'attentions' in outputs:
        attentions = outputs['attentions']
    else:
        print("❌ 无法从输出中提取注意力！")
        print(f"   输出键: {outputs.keys() if isinstance(outputs, dict) else dir(outputs)}")
        exit(1)
    
    # 转换为numpy格式
    # attentions是一个tuple，每个元素是 [batch, num_heads, seq_len, seq_len]
    attention_list = []
    for layer_attention in attentions:
        attention_list.append(layer_attention.cpu().numpy())
    
    # 堆叠为 [num_layers, batch, num_heads, seq_len, seq_len]
    attention_array = np.stack(attention_list, axis=0)
    # 去掉batch维度 -> [num_layers, num_heads, seq_len, seq_len]
    attention_array = attention_array[:, 0, :, :, :]
    
    # 获取attention mask
    attention_mask = inputs['attention_mask'].cpu().numpy()[0]
    
    # 构建结果字典（模仿 extract_attention_scores 的输出格式）
    attention_result = {
        'attentions': torch.from_numpy(attention_array),
        'attention_mask': torch.from_numpy(attention_mask),
        'tokens': tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu().numpy())
    }
    
    num_layers = attention_result['attentions'].shape[0]
    num_heads = attention_result['attentions'].shape[1]
    seq_len = attention_result['attentions'].shape[2]
    
    print(f" ✅ 提取成功!")
    print(f"   > 形状 (层, 头, 序列长, 序列长): {attention_result['attentions'].shape}")
    print(f"   > 有效序列长度: {attention_mask.sum()}")

except Exception as e:
    print(f"  ❌ 提取注意力失败: {e}")
    import traceback
    traceback.print_exc()
    exit()

# --- 分析并可视化（使用之前的Step 1-3代码）---
print("\n" + "="*80)
print(" 📊 注意力可视化分析")
print("="*80)

# Step 1: 平均注意力
print("\n【Step 1】平均所有头的注意力（快速诊断）")
print("-" * 80)

try:
    layer_idx = -1
    avg_attention = attention_result['attentions'][layer_idx].mean(dim=0).numpy()
    attention_mask_np = attention_result['attention_mask'].numpy()
    seq_len_actual = int(attention_mask_np.sum())
    avg_attention_trimmed = avg_attention[:seq_len_actual, :seq_len_actual]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(avg_attention_trimmed, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, label="Attention Weight")
    ax.set_title(f"Attention Pattern - Layer {layer_idx} (Average of All {num_heads} Heads)\n✅ 使用微调后的模型", fontsize=14)
    ax.set_xlabel("Key Positions", fontsize=12)
    ax.set_ylabel("Query Positions", fontsize=12)
    plt.tight_layout()
    plt.savefig("fixed_step1_attention_avg_all_heads.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"   ✅ 平均注意力热图已保存: fixed_step1_attention_avg_all_heads.png")
    print(f"   📊 注意力矩阵形状: {avg_attention_trimmed.shape}")
    print(f"   📊 平均注意力值: {avg_attention_trimmed.mean():.6f}")
    print(f"   📊 最大注意力值: {avg_attention_trimmed.max():.6f}")
    
except Exception as e:
    print(f"   ❌ Step 1 可视化失败: {e}")

# Step 2: 逐个头分析
print("\n【Step 2】逐个头的注意力分析")
print("-" * 80)

actual_layer_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx

# 计算每个头的统计信息
head_stats = []
for head_idx in range(num_heads):
    head_attention = attention_result['attentions'][layer_idx, head_idx].numpy()
    head_attention_trimmed = head_attention[:seq_len_actual, :seq_len_actual]
    
    # 计算熵
    epsilon = 1e-10
    entropy = -np.sum(head_attention_trimmed * np.log(head_attention_trimmed + epsilon), axis=-1).mean()
    
    # 计算集中度
    concentration = head_attention_trimmed.max()
    
    head_stats.append({
        'head_idx': head_idx,
        'entropy': entropy,
        'concentration': concentration,
        'mean_attention': head_attention_trimmed.mean(),
        'max_attention': head_attention_trimmed.max()
    })

print("\n   各个头的统计信息:")
print(f"   {'Head':>6} | {'熵(Entropy)':>15} | {'集中度':>12} | {'平均值':>10} | {'最大值':>10}")
print("   " + "-" * 70)
for stat in head_stats:
    print(f"   {stat['head_idx']:>6} | {stat['entropy']:>15.4f} | {stat['concentration']:>12.6f} | "
          f"{stat['mean_attention']:>10.6f} | {stat['max_attention']:>10.6f}")

# Step 3: 代表性头
print("\n【Step 3】选择最有代表性的头")
print("-" * 80)

head_lowest_entropy = min(head_stats, key=lambda x: x['entropy'])
head_highest_concentration = max(head_stats, key=lambda x: x['concentration'])

print(f"\n   📊 代表性头部分析:")
print(f"   1️⃣  熵最低 (注意力最集中): Head {head_lowest_entropy['head_idx']}")
print(f"       - 熵值: {head_lowest_entropy['entropy']:.4f}")
print(f"       - 集中度: {head_lowest_entropy['concentration']:.6f}")

print(f"\n   2️⃣  集中度最高 (最明确的关注): Head {head_highest_concentration['head_idx']}")
print(f"       - 熵值: {head_highest_concentration['entropy']:.4f}")
print(f"       - 集中度: {head_highest_concentration['concentration']:.6f}")

# 为代表性头生成可视化
representative_heads = [
    ('lowest_entropy', head_lowest_entropy['head_idx'], '熵最低'),
    ('highest_concentration', head_highest_concentration['head_idx'], '集中度最高'),
]

print(f"\n   正在为代表性头生成可视化...")
for name, head_idx, description in representative_heads:
    try:
        head_attention = attention_result['attentions'][layer_idx, head_idx].numpy()
        head_attention_trimmed = head_attention[:seq_len_actual, :seq_len_actual]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(head_attention_trimmed, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax, label="Attention Weight")
        ax.set_title(f"Attention Pattern - Layer {layer_idx}, Head {head_idx}\n{description} - ✅ 微调后的模型", fontsize=14)
        ax.set_xlabel("Key Positions", fontsize=12)
        ax.set_ylabel("Query Positions", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"fixed_step3_representative_{name}_head{head_idx:02d}.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"   ✅ {description} (Head {head_idx}): fixed_step3_representative_{name}_head{head_idx:02d}.png")
    except Exception as e:
        print(f"   ⚠️  {description} (Head {head_idx}) 可视化失败: {e}")

print("\n" + "="*80)
print(" ✅ 分析完成！使用微调后的模型权重")
print("="*80)
print("\n 💡 提示:")
print("   - 对比 'fixed_*' 和之前的图片")
print("   - 微调后的模型应该显示更清晰、更有意义的注意力模式")
print("   - 如果仍然是'斜状式'，说明微调可能没有显著改变注意力模式")

