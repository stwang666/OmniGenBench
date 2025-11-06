import torch
import numpy as np

# 模拟单个头的注意力数据
print("="*80)
print("测试 get_attention_statistics 的 nan 问题")
print("="*80)

# 创建测试数据
attention_scores = torch.randn(1, 1, 10, 10)
attention_scores = torch.softmax(attention_scores, dim=-1)  # 确保是概率分布
attention_mask = torch.ones(10)

print(f"\n输入形状:")
print(f"  attention_scores: {attention_scores.shape}")
print(f"  attention_mask: {attention_mask.shape}")

# 模拟 get_attention_statistics 的处理
print(f"\n处理步骤:")

# Step 1: 应用mask
mask = attention_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len)
print(f"  mask after unsqueeze: {mask.shape}")

mask = mask * attention_mask.unsqueeze(0).unsqueeze(-1)  # (1, 1, seq_len, seq_len)
print(f"  mask final: {mask.shape}")

attention_scores_masked = attention_scores * mask
print(f"  attention_scores after mask: {attention_scores_masked.shape}")

# Step 2: Head aggregation
head_aggregated = attention_scores_masked.mean(dim=1)
print(f"  after head aggregation (dim=1): {head_aggregated.shape}")

# Step 3: Layer aggregation  
layer_aggregated = head_aggregated.mean(dim=0)
print(f"  after layer aggregation (dim=0): {layer_aggregated.shape}")

# Step 4: 计算统计量
print(f"\n计算统计量:")
print(f"  layer_aggregated stats:")
print(f"    - min: {layer_aggregated.min():.6f}")
print(f"    - max: {layer_aggregated.max():.6f}")
print(f"    - mean: {layer_aggregated.mean():.6f}")
print(f"    - sum per row: {layer_aggregated.sum(dim=-1)[:3]}")  # 前3行

# 计算熵
entropy = -torch.sum(layer_aggregated * torch.log(layer_aggregated + 1e-9), dim=-1)
print(f"\n  entropy: {entropy}")
print(f"  entropy stats: min={entropy.min():.4f}, max={entropy.max():.4f}, mean={entropy.mean():.4f}")
print(f"  Contains nan: {torch.isnan(entropy).any()}")

# 计算concentration
concentration = (layer_aggregated**2).sum(dim=-1)
print(f"\n  concentration: {concentration}")
print(f"  concentration stats: min={concentration.min():.6f}, max={concentration.max():.6f}")
print(f"  Contains nan: {torch.isnan(concentration).any()}")

# 问题诊断
print(f"\n" + "="*80)
print("问题诊断:")
print("="*80)

# 检查注意力权重和是否为1
row_sums = layer_aggregated.sum(dim=-1)
print(f"每行的和（应该接近1.0）: {row_sums[:5]}")
print(f"和的平均值: {row_sums.mean():.6f}")

if row_sums.mean() < 0.5:
    print("\n⚠️  问题: 注意力权重的和太小!")
    print("   原因: mask操作后，注意力权重被大幅降低，但没有重新归一化")
    print("   解决: 需要在mask后重新归一化注意力分布")

