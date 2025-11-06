import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 导入 OmnigenBench 组件
from omnigenbench import (
    OmniTokenizer,
    OmniModelForEmbedding,  # 使用基础的 Embedding 模型类进行注意力提取
)

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

print("="*80)
print(" 启动模型注意力诊断分析...")
print("="*80)

# --- 1. 加载模型（使用基础模型类进行注意力分析）---
MODEL_PATH = "/home/sw1136/OmniGenBench/examples/dingling_te/ogb_te_3class_finetuned_epoch_19_seed_42_accuracy_score_0.9900_seed_42_f1_score_0.9900"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" 正在加载模型: {MODEL_PATH}")
print(f" 目标设备: {device}")

# 🔑 使用基础的 OmniModelForEmbedding 类加载模型
# 这个类包含了所有注意力提取功能，不需要自定义类的特殊参数
print(" 正在加载模型（用于注意力分析）...")
model = OmniModelForEmbedding(
    MODEL_PATH,
    trust_remote_code=True
)

# 加载 tokenizer
tokenizer = model.tokenizer

model = model.to(device)
model.eval()

print(f"  模型加载成功!")
print(f"    模型类型: {model.__class__.__name__}")
print(f"    设备: {device}")
print(f"\n 说明: 使用 OmniModelForEmbedding 加载，可以提取注意力但不包含自定义分类头")
print(f"    这不影响注意力分析，因为注意力来自底层 Transformer 层")


# --- 3. 准备要分析的序列 ---
# 🔑 替换为您要分析的序列（例如：验证集/测试集上预测错误的序列）
SEQUENCE_TO_ANALYZE = "GTGGCGGCCGCTGCAAAACCCGGGGCGCGAGCCCGGGCGGAGCGGCCGTCGGTGCAGATCTTGGTGGTAGTAGCAAATATTCAAATGAGAACTTTGAAGGCCGAAGAGGAGAAAGGTTCCATGTGAACAGCACTTGCACATGGGTAAGCCGATTCTAAGGGACGGGGTAACCCCGGCAGATAGCGCGATCACGCGCATCCCCCGAAAGGGAATCGGGTTAAGATTTCCCGAGCCGGGATGTGGCAGTTGACGGTGACGTTAGGAAGTCCGGAGACGCCGGCGGGGGCCTCGGGAAGAGTTATCTTTTCTGCTTAACGGCCTGCCAACCCTGGAAACGGTTCAGCCGGAGGTAGGGTCCAGTGGCCGGAAGATCACCGGACGTCGCGCGGTGTCCAGTGCGCCCCCGGCGGCCCATGAAAATCCGGAGGACCGAGTACCGTTCACACCCGGTCGTACTCATAACCGCATCAGGTCTCCAAGGTGAACAGCCTCTGGCCA"

# 可选：如果您知道某个生物学上重要的区域（如 motif），可以标记它用于对比
KNOWN_MOTIF_START = 50
KNOWN_MOTIF_END = 70

# --- 4. 提取所有层的注意力分数 ---
print(f"\n 正在分析序列（长度: {len(SEQUENCE_TO_ANALYZE)} bp）")
print(f"    序列预览: {SEQUENCE_TO_ANALYZE[:50]}...")
print(" 正在提取所有层和所有头的注意力分数...")

try:
    attention_result = model.extract_attention_scores(
        sequence=SEQUENCE_TO_ANALYZE,
        max_length=512,          # 您可以根据需要调整
        layer_indices=None,      # "None" 表示提取所有层
        head_indices=None,       # "None" 表示提取所有头
        return_on_cpu=True       # 推荐，防止OOM
    )
    
    num_layers = attention_result['attentions'].shape[0]
    num_heads = attention_result['attentions'].shape[1]
    seq_len = attention_result['attentions'].shape[2]
    
    print(f" 提取成功!")
    print(f"   > 形状 (层, 头, 序列长, 序列长): {attention_result['attentions'].shape}")
    print(f"   > Tokens: {attention_result['tokens']}")

except Exception as e:
    print(f"  提取注意力失败: {e}")
    import traceback
    traceback.print_exc()
    exit()

# --- 3. 分析 1 & 2: 检查胡乱猜测 (高熵) 与 关注点偏移 (可视化) ---
print("\n" + "="*80)
print(" 分析 1 & 2: 总体注意力统计 (检查是否胡乱猜测) 与 可视化 (检查关注点偏移)")
print("="*80)

# Compute Attention Statistics 计算所有层和所有头聚合后的平均统计数据
stats = model.get_attention_statistics(
    attention_result['attentions'],
    attention_result['attention_mask'],
    layer_aggregation="mean",  # 在所有层上取平均
    head_aggregation="mean"    # 在所有头上取平均
)

avg_entropy = stats['attention_entropy'].mean()
# print(f" 总体平均注意力熵 (Attention Entropy): {avg_entropy:.4f}")
# # print(f"Attention entropy: {stats['attention_entropy'].mean():.4f}")
# print(f"Self-attention: {stats['self_attention_scores'].mean():.4f}")
# print(f"Concentration: {stats['attention_concentration'].max():.4f}")
print("📈 Attention Statistics:")
print(f"  Attention matrix shape: {stats['attention_matrix'].shape}")
print(f"  Average attention entropy: {stats['attention_entropy'].mean():.4f}")
print(f"  Max attention concentration: {stats['attention_concentration'].max():.4f}")
print(f"  Average self-attention score: {stats['self_attention_scores'].mean():.4f}")
print(f"  Max attention per position (top 5): {stats['max_attention_per_position'][:5]}")

# 分析"胡乱猜测"
if avg_entropy > 3.0: # (阈值需要经验判断，高熵通常 > 3-4)
    print("   >  诊断: 注意力熵非常高 (High Entropy)。")
    print("   > 结论: 模型可能没有学到清晰的模式，注意力高度分散，接近'胡乱猜测'。")
else:
    print("   >  诊断: 注意力熵较低 (Low Entropy)。")
    print("   > 结论: 模型已学会关注特定位置。")

# 分析"关注点偏移" - 多步骤可视化分析
print("\n" + "="*80)
print(" 📊 多步骤注意力可视化分析")
print("="*80)

# ============================================================================
# Step 1: 平均注意力可视化（快速诊断）
# ============================================================================
print("\n【Step 1】平均所有头的注意力（快速诊断）")
print("-" * 80)
print("   目的: 获得模型的整体注意力模式，快速判断模型是否学到有意义的模式")

try:
    # 提取最后一层，平均所有头
    layer_idx = -1  # 最后一层
    avg_attention = attention_result['attentions'][layer_idx].mean(dim=0).numpy()
    attention_mask = attention_result['attention_mask'].numpy()
    seq_len_actual = int(attention_mask.sum())
    avg_attention_trimmed = avg_attention[:seq_len_actual, :seq_len_actual]
    
    # 手动绘制平均注意力热图
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(avg_attention_trimmed, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, label="Attention Weight")
    ax.set_title(f"Attention Pattern - Layer {layer_idx} (Average of All {num_heads} Heads)", fontsize=14)
    ax.set_xlabel("Key Positions", fontsize=12)
    ax.set_ylabel("Query Positions", fontsize=12)
    plt.tight_layout()
    plt.savefig("step1_attention_avg_all_heads.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"   ✅ 平均注意力热图已保存: step1_attention_avg_all_heads.png")
    print(f"   📊 注意力矩阵形状: {avg_attention_trimmed.shape}")
    print(f"   📊 平均注意力值: {avg_attention_trimmed.mean():.6f}")
    print(f"   📊 最大注意力值: {avg_attention_trimmed.max():.6f}")
    
except Exception as e:
    print(f"   ❌ Step 1 可视化失败: {e}")

# ============================================================================
# Step 2: 逐个头可视化（深入分析）
# ============================================================================
print("\n【Step 2】逐个头的注意力可视化（深入分析）")
print("-" * 80)
print(f"   目的: 观察每个头学到的不同注意力模式")
print(f"   将为 {num_heads} 个头分别生成热图...")

# 🔑 将负索引转换为正索引（避免切片时出现空张量）
actual_layer_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx

# 先计算每个头的统计信息
head_stats = []
for head_idx in range(num_heads):
    head_attention = attention_result['attentions'][layer_idx, head_idx].numpy()
    head_attention_trimmed = head_attention[:seq_len_actual, :seq_len_actual]
    
    # 计算该头的熵和集中度
    # 🔑 使用正数索引进行切片，避免 -1:0 产生空张量
    head_attention_tensor = attention_result['attentions'][actual_layer_idx:actual_layer_idx+1, head_idx:head_idx+1]
    stats = model.get_attention_statistics(
        head_attention_tensor,
        attention_result['attention_mask'],
        layer_aggregation="mean",
        head_aggregation="mean"
    )
    
    entropy = stats['attention_entropy'].mean().item()
    concentration = stats['attention_concentration'].max().item()
    
    head_stats.append({
        'head_idx': head_idx,
        'entropy': entropy,
        'concentration': concentration,
        'mean_attention': head_attention_trimmed.mean(),
        'max_attention': head_attention_trimmed.max()
    })

# 打印每个头的统计信息
print("\n   各个头的统计信息:")
print(f"   {'Head':>6} | {'熵(Entropy)':>15} | {'集中度':>12} | {'平均值':>10} | {'最大值':>10}")
print("   " + "-" * 70)
for stat in head_stats:
    print(f"   {stat['head_idx']:>6} | {stat['entropy']:>15.4f} | {stat['concentration']:>12.6f} | "
          f"{stat['mean_attention']:>10.6f} | {stat['max_attention']:>10.6f}")

# 可视化每个头（可选择性生成，避免文件过多）
print(f"\n   正在为每个头生成可视化...")
visualize_all_heads = True #False else  # 设置为 False 可跳过生成所有头的图片

if visualize_all_heads:
    for head_idx in range(num_heads):
        try:
            fig = model.visualize_attention_pattern(
                attention_result=attention_result,
                layer_idx=-1,
                head_idx=head_idx,
                save_path=f"step2_attention_head_{head_idx:02d}.png",
                figsize=(12, 10)
            )
            plt.close()
        except Exception as e:
            print(f"   ⚠️  Head {head_idx} 可视化失败: {e}")
    
    print(f"   ✅ 所有 {num_heads} 个头的热图已保存: step2_attention_head_XX.png")
else:
    print(f"   ⏭️  跳过生成所有头的图片（设置 visualize_all_heads=True 以生成）")

# ============================================================================
# Step 3: 选择最有代表性的头进行深入分析
# ============================================================================
print("\n【Step 3】选择最有代表性的头（深入分析）")
print("-" * 80)
print("   策略:")
print("   1️⃣  选择熵最低的头（注意力最集中）")
print("   2️⃣  选择熵最高的头（注意力最分散）")
print("   3️⃣  选择集中度最高的头（最明确的关注点）")

# 找出最有代表性的头
head_lowest_entropy = min(head_stats, key=lambda x: x['entropy'])
head_highest_entropy = max(head_stats, key=lambda x: x['entropy'])
head_highest_concentration = max(head_stats, key=lambda x: x['concentration'])

print(f"\n   📊 代表性头部分析:")
print(f"   1️⃣  熵最低 (注意力最集中): Head {head_lowest_entropy['head_idx']}")
print(f"       - 熵值: {head_lowest_entropy['entropy']:.4f}")
print(f"       - 集中度: {head_lowest_entropy['concentration']:.6f}")
print(f"       - 这个头学到了最明确的注意力模式")

print(f"\n   2️⃣  熵最高 (注意力最分散): Head {head_highest_entropy['head_idx']}")
print(f"       - 熵值: {head_highest_entropy['entropy']:.4f}")
print(f"       - 集中度: {head_highest_entropy['concentration']:.6f}")
print(f"       - 这个头的注意力最分散，可能未学到有用模式")

print(f"\n   3️⃣  集中度最高 (最明确的关注): Head {head_highest_concentration['head_idx']}")
print(f"       - 熵值: {head_highest_concentration['entropy']:.4f}")
print(f"       - 集中度: {head_highest_concentration['concentration']:.6f}")
print(f"       - 这个头对某些特定位置的关注最强")

# 为这三个代表性头生成详细的可视化
representative_heads = [
    ('lowest_entropy', head_lowest_entropy['head_idx'], '熵最低'),
    ('highest_concentration', head_highest_concentration['head_idx'], '集中度最高'),
]

print(f"\n   正在为代表性头生成详细可视化...")
for name, head_idx, description in representative_heads:
    try:
        fig = model.visualize_attention_pattern(
            attention_result=attention_result,
            layer_idx=-1,
            head_idx=head_idx,
            save_path=f"step3_representative_{name}_head{head_idx:02d}.png",
            figsize=(12, 10)
        )
        plt.close()
        print(f"   ✅ {description} (Head {head_idx}): step3_representative_{name}_head{head_idx:02d}.png")
    except Exception as e:
        print(f"   ⚠️  {description} (Head {head_idx}) 可视化失败: {e}")

print("\n" + "-" * 80)
print("   💡 分析建议:")
print("   - 如果熵最低的头仍然很高 (>4)，说明模型可能未学到清晰模式")
print("   - 检查熵最低的头是否关注了生物学上有意义的区域")
print("   - 比较不同头的注意力模式，理解模型的多样性")

# --- 4. 分析 3: Layer-wise Analysis (逐层分析) ---
print("\n" + "="*80)
print(" 分析 3: Layer-wise Analysis (逐层注意力熵)")
print("="*80)
print("   > 观察熵值如何随层数加深而变化。")
print("   > 理想情况: 高层 (如最后几层) 的熵应较低，表明模型最终学会了聚焦。")
print("-" * 80)

layer_entropies = []
for layer_idx in range(num_layers):
    # 提取第 i 层的注意力 (保持形状 [1, num_heads, seq_len, seq_len])
    layer_attention = attention_result['attentions'][layer_idx:layer_idx+1]
    
    # 计算该层(平均所有头)的统计数据
    stats = model.get_attention_statistics(
        layer_attention,
        attention_result['attention_mask'],
        layer_aggregation="mean", # (因为只有1层，所以无所谓)
        head_aggregation="mean"   # 平均该层的所有头
    )
    
    layer_avg_entropy = stats['attention_entropy'].mean()
    layer_entropies.append(layer_avg_entropy)
    print(f"   Layer {layer_idx:02d}: 平均注意力熵 = {layer_avg_entropy:.4f}")

print("-" * 80)

# 绘制逐层熵变化图
try:
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_layers), layer_entropies, marker='o', linestyle='-')
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Mean Attention Entropy", fontsize=12)
    plt.title("Layer-wise Attention Entropy Analysis", fontsize=14)
    plt.xticks(range(num_layers))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("attention_entropy_layer_wise.png")
    print(" 逐层熵变化图已保存到: attention_entropy_layer_wise.png")
    # plt.show()
except Exception as e:
    print(f" 绘制逐层熵图失败: {e}")

print("\n" + "="*80)
print(" 分析完成。")
print("="*80)