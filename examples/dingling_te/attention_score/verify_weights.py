import torch
import sys
sys.path.insert(0, '/home/sw1136/OmniGenBench/examples/dingling_te/ogb_te_3class_finetuned_epoch_19_seed_42_accuracy_score_0.9900_seed_42_f1_score_0.9900')

from omnigenbench import OmniModelForEmbedding, ModelHub

MODEL_PATH = "/home/sw1136/OmniGenBench/examples/dingling_te/ogb_te_3class_finetuned_epoch_19_seed_42_accuracy_score_0.9900_seed_42_f1_score_0.9900"

print("="*80)
print("对比测试：验证权重加载")
print("="*80)

# 加载保存的权重
saved_weights = torch.load(f"{MODEL_PATH}/pytorch_model.bin", map_location='cpu')

# 方法1: 使用 OmniModelForEmbedding（当前方法）
print("\n方法1: OmniModelForEmbedding")
model1 = OmniModelForEmbedding(MODEL_PATH, trust_remote_code=True)

# 检查第一层第一个注意力头的query权重
try:
    loaded_weight1 = model1.model.encoder.layer[0].attention.self.query.weight
    saved_weight = saved_weights['model.encoder.layer.0.attention.self.query.weight']
    
    print(f"  加载的权重形状: {loaded_weight1.shape}")
    print(f"  保存的权重形状: {saved_weight.shape}")
    print(f"  权重是否相同: {torch.allclose(loaded_weight1, saved_weight)}")
    print(f"  加载权重的前5个值: {loaded_weight1.flatten()[:5]}")
    print(f"  保存权重的前5个值: {saved_weight.flatten()[:5]}")
except Exception as e:
    print(f"  ❌ 错误: {e}")

# 方法2: 使用 ModelHub.load（正确方法）
print("\n方法2: ModelHub.load")
try:
    model2 = ModelHub.load(MODEL_PATH)
    
    # 访问内部模型
    if hasattr(model2, 'model'):
        inner_model = model2.model
        if hasattr(inner_model, 'model'):
            inner_model = inner_model.model
    else:
        inner_model = model2
    
    loaded_weight2 = inner_model.encoder.layer[0].attention.self.query.weight
    saved_weight = saved_weights['model.encoder.layer.0.attention.self.query.weight']
    
    print(f"  加载的权重形状: {loaded_weight2.shape}")
    print(f"  保存的权重形状: {saved_weight.shape}")
    print(f"  权重是否相同: {torch.allclose(loaded_weight2, saved_weight)}")
    print(f"  加载权重的前5个值: {loaded_weight2.flatten()[:5]}")
    print(f"  保存权重的前5个值: {saved_weight.flatten()[:5]}")
except Exception as e:
    print(f"  ❌ 错误: {e}")
    import traceback
    traceback.print_exc()

