import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 导入教程中使用的所有必要组件
from omnigenbench import (
    OmniModelForSequenceClassification, # 您可以使用任何模型类型
    OmniTokenizer,
)
# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

print("="*80)
print(" 启动模型注意力诊断分析...")
print("="*80)

# --- 1. 设置与加载模型 ---
# 替换为您微调好的模型路径，或者使用一个基础模型作为示例
MODEL_PATH = "/home/sw1136/OmniGenBench/examples/dingling_te/ogb_te_3class_finetuned_epoch_19_seed_42_accuracy_score_0.9900_seed_42_f1_score_0.9900" 
# 替换为一条您在验证集/测试集上预测错误的序列
SEQUENCE_TO_ANALYZE = "TAATTTGCTAAACCACACCTTGGTCCAAACGGAGTTCAAATTCATATTTTAAAGTACAGTTAAATTAATCATTTCTTCAAACTGCAAAACAAAAGTGTTCATAATATTTCAAATAATCTACTAAATTTTCCATGAAGAAACCCCACTTGAGAGAGTTTCTAAAAGGCCCCTAACTAATTTTAGAATGTACCAACATCATTTAAAGTGGTCCAATTAAATGCAAACACAATTATAGACTTTTCCAGTTGGCCTCAAACTTTTTGAGCAACCTCAGAATATTCCAAAGAATTTATGTACCTAGTTTCATAATTAGCAAAAATTGTTTGGACACTCATTTAATTAGAAAACAGTGGCAAAAAATGCTAAAGACGTAGTGAACTAATGGGCCATAAGTGGCTGCAGCAGCACGACCCAGCCCACGAACGGCCTTCTCCTTCTTCTGTACATAGTAGGAGGCTGGAGATGTGCGTGTCAAACATCCACGGCTGCCATGGCCATGG"
# 替换为您认为这条序列中"真正重要"的生物学区域（用于人工对比）
# 示例：假设我们认为 'CCCCGGGG' 这个 G/C 簇是关键
KNOWN_MOTIF_START = 8
KNOWN_MOTIF_END = 16

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" 正在加载模型: {MODEL_PATH} (到 {device})")

# 加载分词器
tokenizer = OmniTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True) # 假设模型需要，基于教程示例

# 加载模型 - 注意：您可以使用任何 OmniModel 类型
# Option 1: Use dedicated embedding model
# model = OmniModelForEmbedding(model_name, trust_remote_code=True)

# 这里我们用分类模型作为示例
# 如果加载您自己微调的模型，请确保 num_labels 匹配
model = OmniModelForSequenceClassification(
    MODEL_PATH,
    tokenizer=tokenizer,
    num_labels=2, # 示例，请根据您的模型修改
    trust_remote_code=True
)
model = model.to(device)
model.eval()
print(f" Model loaded: {type(model).__name__}")
print(" 模型加载成功。")

# --- 2. 提取所有层的注意力分数 ---
print(f"\n 正在分析序列: {SEQUENCE_TO_ANALYZE}")
print(" 正在提取所有层和所有头的注意力分数...")

try:
    attention_result = model.extract_attention_scores(
        sequence=SEQUENCE_TO_ANALYZE,
        max_length=128,          # 您可以根据需要调整
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
    print(f" 提取注意力失败: {e}")
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
print(f" 总体平均注意力熵 (Attention Entropy): {avg_entropy:.4f}")
print(f"Attention entropy: {stats['attention_entropy'].mean():.4f}")
print(f"Self-attention: {stats['self_attention_scores'].mean():.4f}")
print(f"Concentration: {stats['attention_concentration'].max():.4f}")
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

# 分析"关注点偏移"
print("\n 正在生成注意力热图 (平均所有头，最后一层)...")
print(f"   > 请人工对比热图上的高亮区域与您已知的关键区域 (如 {KNOWN_MOTIF_START}-{KNOWN_MOTIF_END})。")


# Visualize Attention Heatmap 可视化注意力热图
# 将可视化最后一层 (layer_idx=-1)，并平均所有头 (head_idx=None)
# 注意：visualize_attention_pattern 默认聚合所有头
# Create heatmap for specific layer and head
try:
    fig = model.visualize_attention_pattern(
        attention_result=attention_result,
        layer_idx=-1,  # 索引-1代表最后一层；0 first layer
        head_idx=-1, # "None" 表示自动平均该层的所有头，该方法不支持；0 first attention head
        save_path="attention_heatmap_LastLayer_Avg.png",
        figsize=(12, 10)
    )
    if fig:
        print(f" 热图已保存到: attention_heatmap_LastLayer_Avg.png")
        # plt.show() # 如果在脚本中运行，可以注释掉这行
    else:
        print(" 可视化跳过 (可能缺少 matplotlib)。")
        
except Exception as e:
    print(f" 可视化失败: {e}")

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