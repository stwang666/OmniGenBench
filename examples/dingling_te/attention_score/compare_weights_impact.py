"""
对比分析：错误权重 vs 正确权重的注意力差异
"""
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from omnigenbench import OmniTokenizer, OmniModelForEmbedding

MODEL_PATH = "/home/sw1136/OmniGenBench/examples/dingling_te/ogb_te_3class_finetuned_epoch_19_seed_42_accuracy_score_0.9900_seed_42_f1_score_0.9900"
SEQUENCE = "GTGGCGGCCGCTGCAAAACCCGGGGCGCGAGCCCGGGCGGAGCGGCCGTCGGTGCAGATCTTGGTGGTAGTAGCAAATATTCAAATGAGAACTTTGAAGGCCGAAGAGGAGAAAGGTTCCATGTGAACAGCACTTGCACATGGGTAAGCCGATTCTAAGGGACGGGGTAACCCCGGCAGATAGCGCGATCACGCGCATCCCCCGAAAGGGAATCGGGTTAAGATTTCCCGAGCCGGGATGTGGCAGTTGACGGTGACGTTAGGAAGTCCGGAGACGCCGGCGGGGGCCTCGGGAAGAGTTATCTTTTCTGCTTAACGGCCTGCCAACCCTGGAAACGGTTCAGCCGGAGGTAGGGTCCAGTGGCCGGAAGATCACCGGACGTCGCGCGGTGTCCAGTGCGCCCCCGGCGGCCCATGAAAATCCGGAGGACCGAGTACCGTTCACACCCGGTCGTACTCATAACCGCATCAGGTCTCCAAGGTGAACAGCCTCTGGCCA"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*80)
print(" 🔬 对比分析：错误权重 vs 正确权重")
print("="*80)

def load_model_with_wrong_weights():
    """加载模型，使用错误的权重（原方法）"""
    print("\n📌 方法1: 加载错误权重（OmniModelForEmbedding直接加载）")
    model = OmniModelForEmbedding(MODEL_PATH, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    return model

def load_model_with_correct_weights():
    """加载模型，手动修正权重键名"""
    print("\n📌 方法2: 加载正确权重（手动修正键名）")
    model = OmniModelForEmbedding(MODEL_PATH, trust_remote_code=True)
    
    # 加载并修正权重
    saved_weights = torch.load(f"{MODEL_PATH}/pytorch_model.bin", map_location='cpu')
    corrected_state_dict = {}
    for key, value in saved_weights.items():
        if key.startswith('model.'):
            new_key = key[6:]
            corrected_state_dict[new_key] = value
        else:
            corrected_state_dict[key] = value
    
    # 只加载encoder相关权重
    model_state_dict = model.model.state_dict()
    weights_to_load = {k: corrected_state_dict[k] for k in model_state_dict.keys() if k in corrected_state_dict}
    
    model.model.load_state_dict(weights_to_load, strict=False)
    model = model.to(device)
    model.eval()
    return model

def extract_attention(model, sequence):
    """提取注意力"""
    result = model.extract_attention_scores(
        sequence=sequence,
        max_length=512,
        layer_indices=None,
        head_indices=None,
        return_on_cpu=True
    )
    return result

# 加载两个版本的模型
print("\n" + "="*80)
print(" 步骤 1: 加载两个版本的模型")
print("="*80)

model_wrong = load_model_with_wrong_weights()
model_correct = load_model_with_correct_weights()

# 提取注意力
print("\n" + "="*80)
print(" 步骤 2: 提取注意力分数")
print("="*80)

print("\n 🔴 提取错误权重模型的注意力...")
attention_wrong = extract_attention(model_wrong, SEQUENCE)

print("\n 🟢 提取正确权重模型的注意力...")
attention_correct = extract_attention(model_correct, SEQUENCE)

# 对比分析
print("\n" + "="*80)
print(" 步骤 3: 对比分析")
print("="*80)

layer_idx = -1
seq_len_actual = int(attention_correct['attention_mask'].sum())

# 平均所有头
avg_wrong = attention_wrong['attentions'][layer_idx].mean(dim=0).numpy()[:seq_len_actual, :seq_len_actual]
avg_correct = attention_correct['attentions'][layer_idx].mean(dim=0).numpy()[:seq_len_actual, :seq_len_actual]

print(f"\n📊 统计对比:")
print(f"   {'指标':<30} | {'错误权重':>15} | {'正确权重':>15} | {'差异':>15}")
print("   " + "-"*75)

# 对角线强度
diag_wrong = np.mean([avg_wrong[i,i] for i in range(min(seq_len_actual, 10))])
diag_correct = np.mean([avg_correct[i,i] for i in range(min(seq_len_actual, 10))])
print(f"   {'对角线注意力强度':<30} | {diag_wrong:>15.6f} | {diag_correct:>15.6f} | {abs(diag_wrong-diag_correct):>15.6f}")

# 整体平均
mean_wrong = avg_wrong.mean()
mean_correct = avg_correct.mean()
print(f"   {'整体平均注意力':<30} | {mean_wrong:>15.6f} | {mean_correct:>15.6f} | {abs(mean_wrong-mean_correct):>15.6f}")

# 最大值
max_wrong = avg_wrong.max()
max_correct = avg_correct.max()
print(f"   {'最大注意力值':<30} | {max_wrong:>15.6f} | {max_correct:>15.6f} | {abs(max_wrong-max_correct):>15.6f}")

# 计算两个注意力矩阵的相似度
similarity = np.corrcoef(avg_wrong.flatten(), avg_correct.flatten())[0, 1]
print(f"   {'相关系数':<30} | {'':<15} | {'':<15} | {similarity:>15.6f}")

# L2距离
l2_distance = np.linalg.norm(avg_wrong - avg_correct)
print(f"   {'L2距离':<30} | {'':<15} | {'':<15} | {l2_distance:>15.6f}")

# 可视化对比
print("\n" + "="*80)
print(" 步骤 4: 可视化对比")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 错误权重
im1 = axes[0].imshow(avg_wrong, cmap="Blues", aspect="auto")
axes[0].set_title("❌ 错误权重（随机初始化）\n对角线强度: {:.4f}".format(diag_wrong), fontsize=12, color='red')
axes[0].set_xlabel("Key Positions")
axes[0].set_ylabel("Query Positions")
plt.colorbar(im1, ax=axes[0])

# 正确权重
im2 = axes[1].imshow(avg_correct, cmap="Blues", aspect="auto")
axes[1].set_title("✅ 正确权重（微调后）\n对角线强度: {:.4f}".format(diag_correct), fontsize=12, color='green')
axes[1].set_xlabel("Key Positions")
axes[1].set_ylabel("Query Positions")
plt.colorbar(im2, ax=axes[1])

# 差异热图
diff = np.abs(avg_wrong - avg_correct)
im3 = axes[2].imshow(diff, cmap="Reds", aspect="auto")
axes[2].set_title("📊 绝对差异\nL2距离: {:.4f}".format(l2_distance), fontsize=12)
axes[2].set_xlabel("Key Positions")
axes[2].set_ylabel("Query Positions")
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig("COMPARISON_wrong_vs_correct_weights.png", dpi=300, bbox_inches="tight")
print("\n✅ 对比图已保存: COMPARISON_wrong_vs_correct_weights.png")

# 结论
print("\n" + "="*80)
print(" 🎯 结论")
print("="*80)

if similarity > 0.95:
    print("""
   ✅ 相关系数 > 0.95: 两个注意力模式**非常相似**
   
   💡 这说明：
      1. 微调主要改变了分类头（classifier），而非底层的注意力模式
      2. 预训练的注意力模式已经足够好，不需要大幅调整
      3. 您看到的"斜状式"注意力图是**正常的**，不是因为权重加载错误
      4. 对于基因组序列任务，局部特征（对角线）可能比长距离依赖更重要
   
   📝 建议：
      - 查看其他层（不只是最后一层）的注意力，中间层可能有不同模式
      - 分析预测错误的样本，看注意力是否有异常
      - 这种"斜状式"模式在很多序列任务中都是常见的
    """)
elif similarity > 0.8:
    print("""
   ⚠️ 相关系数在 0.8-0.95: 两个注意力模式有明显差异
   
   💡 这说明：
      1. 微调确实改变了部分注意力模式
      2. 但整体结构仍然相似
      3. 您之前看到的图可能混合了随机权重和微调权重的影响
    """)
else:
    print("""
   ❌ 相关系数 < 0.8: 两个注意力模式**完全不同**
   
   💡 这说明：
      1. 之前使用的确实是错误的权重
      2. 微调显著改变了注意力模式
      3. 现在使用 'CORRECT_*' 图片才是您微调模型的真实表现
    """)

print("\n📊 请查看生成的对比图: COMPARISON_wrong_vs_correct_weights.png")
print("="*80)

