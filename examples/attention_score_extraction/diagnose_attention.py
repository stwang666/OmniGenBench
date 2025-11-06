import torch
from omnigenbench import OmniTokenizer, OmniModelForEmbedding

MODEL_PATH = "/home/sw1136/OmniGenBench/examples/dingling_te/ogb_te_3class_finetuned_epoch_19_seed_42_accuracy_score_0.9900_seed_42_f1_score_0.9900"
SEQUENCE = "GTGGCGGCCGCTGCAAAACCCGGGGCGCGAGCCCGGGCGGAGCGGCCGTCGGTGCAGATCTTGGTGGTAGTAGCAAATATTCAAATGAGAACTTTGAAGGCCGAAGAGGAGAAAGGTTCCATGTGAACAGCACTTGCACATGGGTAAGCCGATTCTAAGGGACGGGGTAACCCCGGCAGATAGCGCGATCACGCGCATCCCCCGAAAGGGAATCGGGTTAAGATTTCCCGAGCCGGGATGTGGCAGTTGACGGTGACGTTAGGAAGTCCGGAGACGCCGGCGGGGGCCTCGGGAAGAGTTATCTTTTCTGCTTAACGGCCTGCCAACCCTGGAAACGGTTCAGCCGGAGGTAGGGTCCAGTGGCCGGAAGATCACCGGACGTCGCGCGGTGTCCAGTGCGCCCCCGGCGGCCCATGAAAATCCGGAGGACCGAGTACCGTTCACACCCGGTCGTACTCATAACCGCATCAGGTCTCCAAGGTGAACAGCCTCTGGCCA"

print("="*80)
print("诊断实际attention数据的nan问题")
print("="*80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OmniModelForEmbedding(MODEL_PATH, trust_remote_code=True).to(device).eval()

print(f"\n提取attention...")
attention_result = model.extract_attention_scores(
    sequence=SEQUENCE,
    max_length=512,
    layer_indices=None,
    head_indices=None,
    return_on_cpu=True
)

print(f"Attention shape: {attention_result['attentions'].shape}")
print(f"Attention mask shape: {attention_result['attention_mask'].shape}")
print(f"Mask sum (实际序列长度): {attention_result['attention_mask'].sum()}")

# 检查最后一层第0个头
layer_idx = -1
head_idx = 0
head_attention = attention_result['attentions'][layer_idx, head_idx]

print(f"\n检查 Layer {layer_idx}, Head {head_idx}:")
print(f"  Shape: {head_attention.shape}")
print(f"  Min: {head_attention.min():.8f}")
print(f"  Max: {head_attention.max():.8f}")
print(f"  Mean: {head_attention.mean():.8f}")
print(f"  Contains nan: {torch.isnan(head_attention).any()}")
print(f"  Contains inf: {torch.isinf(head_attention).any()}")

# 检查每行的和
row_sums = head_attention.sum(dim=-1)
print(f"\n  Row sums (前10个): {row_sums[:10]}")
print(f"  Row sums mean: {row_sums.mean():.8f}")
print(f"  Row sums std: {row_sums.std():.8f}")

# 现在测试 get_attention_statistics 对单个头的处理
print(f"\n测试单个头的统计计算:")
head_attention_tensor = attention_result['attentions'][layer_idx:layer_idx+1, head_idx:head_idx+1]
print(f"  Input shape: {head_attention_tensor.shape}")

try:
    stats = model.get_attention_statistics(
        head_attention_tensor,
        attention_result['attention_mask'],
        layer_aggregation="mean",
        head_aggregation="mean"
    )
    
    print(f"\n  Stats computed successfully!")
    print(f"  Entropy shape: {stats['attention_entropy'].shape}")
    print(f"  Entropy: {stats['attention_entropy'][:10]}")
    print(f"  Entropy contains nan: {torch.isnan(stats['attention_entropy']).any()}")
    print(f"  Entropy mean: {stats['attention_entropy'].mean()}")
    
    print(f"\n  Concentration shape: {stats['attention_concentration'].shape}")
    print(f"  Concentration: {stats['attention_concentration'][:10]}")
    print(f"  Concentration contains nan: {torch.isnan(stats['attention_concentration']).any()}")
    print(f"  Concentration max: {stats['attention_concentration'].max()}")
    
    print(f"\n  Attention matrix shape: {stats['attention_matrix'].shape}")
    print(f"  Attention matrix row sums (前5个): {stats['attention_matrix'].sum(dim=-1)[:5]}")
    
except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()

