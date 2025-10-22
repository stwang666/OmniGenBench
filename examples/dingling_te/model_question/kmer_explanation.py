# -*- coding: utf-8 -*-
# k-mer详解与应用

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

print("="*70)
print("🧬 k-mer详解")
print("="*70)

# ============= 1. 什么是k-mer？ =============
print("\n📌 1. 什么是k-mer？")
print("-"*70)

print("""
k-mer定义：
  📖 长度为k的连续子序列

类比：
  如果把DNA序列比作一个句子，k-mer就像"n-gram"
  
  句子: "I love DNA"
  2-gram: "I ", " l", "lo", "ov", "ve", "e ", " D", "DN", "NA"
  
  DNA: "ATCGATCG"
  2-mer: "AT", "TC", "CG", "GA", "AT", "TC", "CG"
  3-mer: "ATC", "TCG", "CGA", "GAT", "ATC", "TCG"
""")

# ============= 2. k-mer示例 =============
print("\n📌 2. k-mer分解示例")
print("-"*70)

sequence = "ATCGATCGTAGC"
print(f"DNA序列: {sequence}")
print(f"序列长度: {len(sequence)}\n")

# 不同k值的k-mer
for k in [1, 2, 3, 4, 6]:
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        kmers.append(kmer)
    
    print(f"{k}-mer (共{len(kmers)}个):")
    print(f"  {' | '.join(kmers)}")
    
    # 统计频率
    kmer_counts = Counter(kmers)
    if len(kmer_counts) <= 10:  # 只显示不太多的情况
        print(f"  频率: {dict(kmer_counts)}")
    print()

# ============= 3. k-mer的性质 =============
print("\n📌 3. k-mer的重要性质")
print("-"*70)

print("""
1️⃣ k-mer数量
   对于长度为L的序列：
   - k-mer总数 = L - k + 1
   - 唯一k-mer数 ≤ min(4^k, L - k + 1)
   
   例如: 序列长度L=100, k=3
   - 总k-mer数: 100 - 3 + 1 = 98个
   - 可能的唯一3-mer: 4^3 = 64种

2️⃣ k值的选择
   k太小 (k=1,2):
     ✅ 统计稳定
     ❌ 信息量少
   
   k适中 (k=3,4,5,6):
     ✅ 平衡信息量和频率
     ✅ 常用于motif识别
   
   k太大 (k>10):
     ✅ 信息量大
     ❌ 稀疏，很多k-mer只出现1次

3️⃣ 反向互补
   DNA双链，需考虑反向互补：
   - "ATG" 和 "CAT" 本质相同
   - 可以合并计数
""")

# ============= 4. k-mer在tokenization中的应用 =============
print("\n\n📌 4. k-mer Tokenization（OmniGenome使用）")
print("-"*70)

print("""
为什么用k-mer做tokenization？

传统方法（单碱基）:
  序列: ATCG
  Token: ['A', 'T', 'C', 'G']
  
  ❌ 问题：丢失碱基间的关联信息
     - "ATG"（起始密码子）被拆成'A', 'T', 'G'
     - 模型需要重新学习这种组合

k-mer方法 (k=3):
  序列: ATCGATCG
  Token: ['ATC', 'TCG', 'CGA', 'GAT', 'ATC', 'TCG']
  
  ✅ 优点：
     - 保留了局部上下文信息
     - 直接编码常见motif
     - 减少序列长度（压缩）
     - 类似于BPE（Byte Pair Encoding）
""")

# 示例：不同tokenization方法的对比
sequence_example = "ATGCATGC"
print(f"\n示例序列: {sequence_example}")
print(f"\n方法对比:")

# 方法1: 单碱基
tokens_1mer = list(sequence_example)
print(f"  单碱基:  {tokens_1mer}")
print(f"  Token数: {len(tokens_1mer)}")

# 方法2: 3-mer (overlapping)
tokens_3mer_overlap = [sequence_example[i:i+3] for i in range(len(sequence_example)-2)]
print(f"  3-mer (重叠): {tokens_3mer_overlap}")
print(f"  Token数: {len(tokens_3mer_overlap)}")

# 方法3: 3-mer (non-overlapping)
tokens_3mer_non = [sequence_example[i:i+3] for i in range(0, len(sequence_example), 3)]
print(f"  3-mer (不重叠): {tokens_3mer_non}")
print(f"  Token数: {len(tokens_3mer_non)}")

# ============= 5. k-mer频谱分析 =============
print("\n\n📌 5. k-mer频谱分析")
print("-"*70)

# 生成一个较长的模拟序列
np.random.seed(42)
long_sequence = ''.join(np.random.choice(['A', 'T', 'C', 'G'], size=500))

# 计算3-mer频率
k = 3
kmers_list = [long_sequence[i:i+k] for i in range(len(long_sequence) - k + 1)]
kmer_counts = Counter(kmers_list)

print(f"模拟序列长度: {len(long_sequence)}")
print(f"3-mer总数: {len(kmers_list)}")
print(f"唯一3-mer数: {len(kmer_counts)}")
print(f"理论最大值: 4^3 = {4**3}")

# 可视化k-mer频率分布
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图1: k-mer频率条形图（Top 20）
ax1 = axes[0, 0]
top_20 = kmer_counts.most_common(20)
kmers_top, counts_top = zip(*top_20)
colors_map = {
    'A': '#FF6B6B', 'T': '#4ECDC4', 'C': '#45B7D1', 'G': '#95E1D3'
}
# 根据第一个碱基着色
bar_colors = [colors_map[kmer[0]] for kmer in kmers_top]

ax1.bar(range(len(kmers_top)), counts_top, color=bar_colors, alpha=0.7, edgecolor='black')
ax1.set_xticks(range(len(kmers_top)))
ax1.set_xticklabels(kmers_top, rotation=45, ha='right')
ax1.set_xlabel('3-mer', fontsize=11)
ax1.set_ylabel('频率', fontsize=11)
ax1.set_title('Top 20 最常见的3-mer', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 图2: 频率分布直方图
ax2 = axes[0, 1]
freq_values = list(kmer_counts.values())
ax2.hist(freq_values, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax2.set_xlabel('出现次数', fontsize=11)
ax2.set_ylabel('k-mer数量', fontsize=11)
ax2.set_title('k-mer频率分布\n（大多数k-mer出现次数相近）', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 图3: 不同k值的唯一k-mer数
ax3 = axes[1, 0]
k_values = range(1, 9)
unique_counts = []
total_counts = []

for k_val in k_values:
    kmers_k = [long_sequence[i:i+k_val] for i in range(len(long_sequence) - k_val + 1)]
    unique_counts.append(len(set(kmers_k)))
    total_counts.append(len(kmers_k))

ax3.plot(k_values, unique_counts, marker='o', linewidth=2, markersize=8, 
        label='唯一k-mer数', color='red')
ax3.plot(k_values, [4**k for k in k_values], marker='s', linewidth=2, 
        markersize=8, label='理论最大值 (4^k)', color='blue', linestyle='--')
ax3.set_xlabel('k值', fontsize=11)
ax3.set_ylabel('k-mer数量', fontsize=11)
ax3.set_title('不同k值的k-mer空间大小', fontsize=12, fontweight='bold')
ax3.set_xticks(k_values)
ax3.legend()
ax3.grid(alpha=0.3)
ax3.set_yscale('log')

# 图4: GC含量与k-mer的关系
ax4 = axes[1, 1]
gc_content = []
for kmer in kmer_counts.keys():
    gc = (kmer.count('G') + kmer.count('C')) / len(kmer)
    gc_content.extend([gc] * kmer_counts[kmer])

ax4.hist(gc_content, bins=20, color='green', alpha=0.7, edgecolor='black')
ax4.set_xlabel('GC含量', fontsize=11)
ax4.set_ylabel('k-mer数量', fontsize=11)
ax4.set_title('k-mer的GC含量分布\n（随机序列应该接近0.5）', fontsize=12, fontweight='bold')
ax4.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='期望值=0.5')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/sw1136/OmniGenBench/examples/dingling_te/kmer_analysis.png', 
           dpi=300, bbox_inches='tight')
print("💾 k-mer分析图已保存: kmer_analysis.png\n")

# ============= 6. k-mer在生物信息学中的应用 =============
print("\n\n📌 6. k-mer的实际应用")
print("-"*70)

print("""
1️⃣ 序列比对和组装
   - 快速查找相似序列
   - De novo基因组组装
   - 重叠群(Contig)构建

2️⃣ Motif识别
   - 转录因子结合位点（TFBS）
   - 启动子识别
   - 剪接位点预测

3️⃣ 物种鉴定
   - k-mer频谱指纹
   - 宏基因组分析
   - 病原体检测

4️⃣ 序列分类
   - 编码/非编码区分
   - 转座子分类
   - 功能预测

5️⃣ 深度学习
   - Tokenization（如OmniGenome）
   - 特征提取
   - 嵌入学习
""")

# ============= 7. OmniGenome的k-mer tokenization =============
print("\n\n📌 7. OmniGenome如何使用k-mer？")
print("-"*70)

print("""
OmniGenome的Tokenization策略：

方法: BPE-like k-mer tokenization

步骤:
  1. 初始化：所有单碱基作为基础token
     词表: ['A', 'T', 'C', 'G', ...]
  
  2. 统计：在大规模语料中找最频繁的碱基对
     例如: 'AT' 出现很频繁 → 加入词表
  
  3. 迭代：继续合并频繁的token对
     'AT' + 'G' → 'ATG'（起始密码子）
     'TA' + 'TA' → 'TATA'（TATA box）
  
  4. 最终词表：
     - 单碱基: A, T, C, G
     - 常见2-mer: AT, TA, CG, GC, ...
     - 常见3-mer: ATG, TAA, TGA, ...
     - 功能motif: TATA, CCAAT, ...
     - 特殊token: [PAD], [CLS], [SEP], [MASK]

优势:
  ✅ 自动学习生物学上有意义的motif
  ✅ 适应数据分布
  ✅ 压缩序列长度
  ✅ 提高训练效率

示例tokenization:
  序列: "ATGTATAGATAG"
  
  可能的token化:
  ['ATG', 'TATA', 'GAT', 'AG']
  
  而不是:
  ['A', 'T', 'G', 'T', 'A', 'T', 'A', 'G', 'A', 'T', 'A', 'G']
  
  token数: 4 vs 12（压缩3倍）
""")

# ============= 8. 实践建议 =============
print("\n\n📌 8. k-mer使用建议")
print("-"*70)

print("""
选择k值的原则：

基因组分析:
  - k=4-6: motif发现
  - k=21-31: 组装和比对
  - k=3: 密码子分析

机器学习:
  - k=3-6: 特征提取
  - BPE: 让模型自己学（推荐）
  - 考虑序列长度限制

注意事项:
  ⚠️ k越大，词表越大（4^k）
  ⚠️ 需要考虑反向互补
  ⚠️ 稀有k-mer可能是噪声
  ⚠️ 计算和存储开销
""")

plt.show()

print("\n" + "="*70)
print("✅ k-mer详解完成！")
print("="*70)



