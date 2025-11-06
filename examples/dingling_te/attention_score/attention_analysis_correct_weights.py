import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from omnigenbench import (
    OmniTokenizer,
    OmniModelForEmbedding,
)

# Set up plotting
plt.style.use('seaborn-v0_8')
# 使用蓝色到红色的渐变色
sns.set_palette(sns.color_palette("coolwarm", n_colors=10))

print("="*80)
print(" 🔧 启动注意力分析 - 手动加载微调权重")
print("="*80)

MODEL_PATH = "/home/sw1136/OmniGenBench/examples/dingling_te/ogb_te_3class_finetuned_epoch_19_seed_42_accuracy_score_0.9900_seed_42_f1_score_0.9900"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" 模型路径: {MODEL_PATH}")
print(f" 目标设备: {device}")

# Step 1: 加载基础模型（不带权重）
print("\n 步骤 1: 加载基础模型架构...")
tokenizer = OmniTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = OmniModelForEmbedding(MODEL_PATH, trust_remote_code=True)

# Step 2: 加载保存的权重
print(" 步骤 2: 加载保存的权重...")
saved_weights = torch.load(f"{MODEL_PATH}/pytorch_model.bin", map_location='cpu')
print(f"   - 加载了 {len(saved_weights)} 个权重张量")

# Step 3: 移除 'model.' 前缀并加载到模型中
print(" 步骤 3: 修正权重键名并加载...")
# 创建一个新的state_dict，移除 'model.' 前缀
corrected_state_dict = {}
for key, value in saved_weights.items():
    if key.startswith('model.'):
        # 移除 'model.' 前缀
        new_key = key[6:]  # 'model.' 有6个字符
        corrected_state_dict[new_key] = value
    else:
        corrected_state_dict[key] = value

# 只加载encoder相关的权重（忽略classifier等自定义头部）
model_state_dict = model.model.state_dict()
weights_to_load = {}
skipped_keys = []

for key in model_state_dict.keys():
    if key in corrected_state_dict:
        weights_to_load[key] = corrected_state_dict[key]
    else:
        skipped_keys.append(key)

print(f"   - 匹配的权重: {len(weights_to_load)} / {len(model_state_dict)}")
print(f"   - 未匹配的权重: {len(skipped_keys)}")
if len(skipped_keys) > 0 and len(skipped_keys) < 20:
    print(f"   - 未匹配的键: {skipped_keys[:10]}")

# 加载权重
missing_keys, unexpected_keys = model.model.load_state_dict(weights_to_load, strict=False)

print(f" 步骤 4: 验证权重加载...")
# 验证：检查第一层的query权重
loaded_query = model.model.encoder.layer[0].attention.self.query.weight
saved_query = corrected_state_dict['encoder.layer.0.attention.self.query.weight']

loaded_query_cpu = loaded_query.detach().cpu().float()
saved_query_cpu = saved_query.float()

weights_match = torch.allclose(loaded_query_cpu, saved_query_cpu, rtol=1e-5, atol=1e-5)

print(f"   ✓ 权重验证: {'✅ 成功匹配' if weights_match else '❌ 不匹配'}")
print(f"   ✓ 加载的权重前5个值: {loaded_query_cpu.flatten()[:5]}")
print(f"   ✓ 保存的权重前5个值: {saved_query_cpu.flatten()[:5]}")

if not weights_match:
    print("\n❌ 警告：权重加载失败！")
    exit(1)

# 将模型移到GPU
model = model.to(device)
model.eval()

print(f"\n✅ 模型加载成功！")
print(f"   - 模型类型: {model.__class__.__name__}")
print(f"   - 设备: {next(model.parameters()).device}")
print(f"   - 确认使用微调后的权重")

# --- 准备要分析的序列 ---
# SEQUENCE_TO_ANALYZE = "GTGGCGGCCGCTGCAAAACCCGGGGCGCGAGCCCGGGCGGAGCGGCCGTCGGTGCAGATCTTGGTGGTAGTAGCAAATATTCAAATGAGAACTTTGAAGGCCGAAGAGGAGAAAGGTTCCATGTGAACAGCACTTGCACATGGGTAAGCCGATTCTAAGGGACGGGGTAACCCCGGCAGATAGCGCGATCACGCGCATCCCCCGAAAGGGAATCGGGTTAAGATTTCCCGAGCCGGGATGTGGCAGTTGACGGTGACGTTAGGAAGTCCGGAGACGCCGGCGGGGGCCTCGGGAAGAGTTATCTTTTCTGCTTAACGGCCTGCCAACCCTGGAAACGGTTCAGCCGGAGGTAGGGTCCAGTGGCCGGAAGATCACCGGACGTCGCGCGGTGTCCAGTGCGCCCCCGGCGGCCCATGAAAATCCGGAGGACCGAGTACCGTTCACACCCGGTCGTACTCATAACCGCATCAGGTCTCCAAGGTGAACAGCCTCTGGCCA"
# SEQUENCE_TO_ANALYZE = "ACACAGAGGGCTAAGAAAAAGGGGGAAAACAGAGTTGAGGAAGAAACGAGCC"
SEQUENCE_TO_ANALYZE = "GGATATATCATGAGAAATATCGACCATAAGGAAAGCCACAGTAGAGAAAAGAAATAGGACAACATTTCTG"

print(f"\n 正在分析序列（长度: {len(SEQUENCE_TO_ANALYZE)} bp）")
print(f"    序列预览: {SEQUENCE_TO_ANALYZE[:50]}...")
print(" 正在提取注意力分数...")

try:
    attention_result = model.extract_attention_scores(
        sequence=SEQUENCE_TO_ANALYZE,
        max_length=512,
        layer_indices=None,
        head_indices=None,
        return_on_cpu=True
    )
    
    num_layers = attention_result['attentions'].shape[0]
    num_heads = attention_result['attentions'].shape[1]
    seq_len = attention_result['attentions'].shape[2]
    
    print(f" ✅ 提取成功!")
    print(f"   > 形状 (层, 头, 序列长, 序列长): {attention_result['attentions'].shape}")
    print(f"   > 有效序列长度: {attention_result['attention_mask'].sum()}")

except Exception as e:
    print(f"  ❌ 提取注意力失败: {e}")
    import traceback
    traceback.print_exc()
    exit()

# --- 分析并可视化 ---
print("\n" + "="*80)
print(" 📊 注意力可视化分析（✅ 使用微调权重）")
print("="*80)

# Step 1: 平均注意力
print("\n【Step 1】平均所有头的注意力")
print("-" * 80)

try:
    layer_idx = -1
    avg_attention = attention_result['attentions'][layer_idx].mean(dim=0).numpy()
    attention_mask_np = attention_result['attention_mask'].numpy()
    seq_len_actual = int(attention_mask_np.sum())
    avg_attention_trimmed = avg_attention[:seq_len_actual, :seq_len_actual]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    # im = ax.imshow(avg_attention_trimmed, cmap="coolwarm", aspect="auto")
    im = ax.imshow(avg_attention_trimmed, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, label="Attention Weight")
    ax.set_title(f"Attention Pattern - Layer {layer_idx} (Average of All {num_heads} Heads)\nFine-tuned Model", fontsize=14, color='green')
    ax.set_xlabel("Key Positions", fontsize=12)
    ax.set_ylabel("Query Positions", fontsize=12)
    plt.tight_layout()
    plt.savefig("CORRECT_step1_attention_avg.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"   ✅ 平均注意力热图: CORRECT_step1_attention_avg.png")
    print(f"   📊 注意力矩阵形状: {avg_attention_trimmed.shape}")
    print(f"   📊 平均注意力值: {avg_attention_trimmed.mean():.6f}")
    print(f"   📊 最大注意力值: {avg_attention_trimmed.max():.6f}")
    
    # 分析注意力模式
    # 计算对角线强度
    diagonal_strength = np.mean([avg_attention_trimmed[i, i] for i in range(min(seq_len_actual, 10))])
    off_diagonal_strength = avg_attention_trimmed.mean()
    
    print(f"   📊 对角线注意力强度: {diagonal_strength:.6f}")
    print(f"   📊 整体平均注意力: {off_diagonal_strength:.6f}")
    
    if diagonal_strength > off_diagonal_strength * 2:
        print("   ⚠️  注意：对角线注意力很强，可能是'斜状式'模式")
        print("      这可能意味着：")
        print("      1) 微调没有显著改变注意力模式")
        print("      2) 任务不需要长距离依赖")
        print("      3) 模型主要关注局部特征")
    else:
        print("   ✅ 注意力模式显示长距离依赖")
    
except Exception as e:
    print(f"   ❌ Step 1 可视化失败: {e}")
    import traceback
    traceback.print_exc()

# Step 2: 头统计
print("\n【Step 2】各个头的统计分析")
print("-" * 80)

actual_layer_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx

head_stats = []
for head_idx in range(num_heads):
    head_attention = attention_result['attentions'][layer_idx, head_idx].numpy()
    head_attention_trimmed = head_attention[:seq_len_actual, :seq_len_actual]
    
    # 计算熵
    epsilon = 1e-10
    entropy = -np.sum(head_attention_trimmed * np.log(head_attention_trimmed + epsilon), axis=-1).mean()
    
    # 计算集中度
    concentration = head_attention_trimmed.max()
    
    # 计算对角线vs非对角线比例
    diagonal_mean = np.mean([head_attention_trimmed[i, i] for i in range(min(seq_len_actual, 10))])
    
    head_stats.append({
        'head_idx': head_idx,
        'entropy': entropy,
        'concentration': concentration,
        'mean_attention': head_attention_trimmed.mean(),
        'max_attention': head_attention_trimmed.max(),
        'diagonal_strength': diagonal_mean
    })

print("\n   各个头的详细统计:")
print(f"   {'Head':>6} | {'熵':>12} | {'集中度':>12} | {'平均值':>10} | {'对角线':>10}")
print("   " + "-" * 70)
for stat in head_stats:
    print(f"   {stat['head_idx']:>6} | {stat['entropy']:>12.4f} | {stat['concentration']:>12.6f} | "
          f"{stat['mean_attention']:>10.6f} | {stat['diagonal_strength']:>10.6f}")

# Step 3: 代表性头可视化
print("\n【Step 3】代表性头可视化")
print("-" * 80)

head_lowest_entropy = min(head_stats, key=lambda x: x['entropy'])
head_highest_concentration = max(head_stats, key=lambda x: x['concentration'])
# 找到对角线最弱的头（可能学到了长距离依赖）
head_weakest_diagonal = min(head_stats, key=lambda x: x['diagonal_strength'])

print(f"\n   📊 关键头部:")
print(f"   1️⃣  熵最低 (最集中): Head {head_lowest_entropy['head_idx']}")
print(f"   2️⃣  集中度最高: Head {head_highest_concentration['head_idx']}")
print(f"   3️⃣  对角线最弱 (可能学到长距离依赖): Head {head_weakest_diagonal['head_idx']}")

representative_heads = [
    ('lowest_entropy', head_lowest_entropy['head_idx'], 'Lowest Entropy'),
    ('highest_concentration', head_highest_concentration['head_idx'], 'Highest Concentration'),
    ('weakest_diagonal', head_weakest_diagonal['head_idx'], 'Weakest Diagonal'),
]

print(f"\n   正在生成可视化...")
for name, head_idx, description in representative_heads:
    try:
        head_attention = attention_result['attentions'][layer_idx, head_idx].numpy()
        head_attention_trimmed = head_attention[:seq_len_actual, :seq_len_actual]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        # im = ax.imshow(head_attention_trimmed, cmap="coolwarm", aspect="auto")
        im = ax.imshow(head_attention_trimmed, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax, label="Attention Weight")
        ax.set_title(f"Attention - Layer {layer_idx}, Head {head_idx}\n{description} - Fine-tuned Model", 
                    fontsize=14, color='green')
        ax.set_xlabel("Key Positions", fontsize=12)
        ax.set_ylabel("Query Positions", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"CORRECT_step3_{name}_head{head_idx:02d}.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"   ✅ {description} (Head {head_idx}): CORRECT_step3_{name}_head{head_idx:02d}.png")
    except Exception as e:
        print(f"   ⚠️  {description} 失败: {e}")

print("\n" + "="*80)
print(" ✅ 分析完成！")
print("="*80)
print("\n 📝 结论:")
print("   1. 对比 'CORRECT_*' 和之前的 'step*' 图片")
print("   2. 如果两者相似，说明：")
print("      - 微调主要改变了分类头，而非注意力模式")
print("      - 预训练的注意力模式已经足够好")
print("      - 任务可能不需要学习新的注意力模式")
print("   3. 如果'斜状式'模式持续存在：")
print("      - 这可能是正常的！很多任务主要依赖局部特征")
print("      - 检查其他层的注意力（不只是最后一层）")
print("      - 查看中间层是否有更多长距离依赖")
