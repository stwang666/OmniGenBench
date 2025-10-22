# -*- coding: utf-8 -*-
# 检查模型配置和多头注意力机制

import torch
from omnigenbench import OmniTokenizer, ModelHub

print("="*70)
print("🔍 检查模型配置")
print("="*70)

# 加载模型
model_path = "yangheng/OmniGenome-52M"
print(f"\n正在加载模型: {model_path}")

try:
    tokenizer = OmniTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 方法1：直接加载模型查看配置
    from transformers import AutoModel, AutoConfig
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    print("\n📊 模型配置信息：")
    print("-"*70)
    
    # 关键配置
    if hasattr(config, 'hidden_size'):
        print(f"✅ hidden_size (隐藏层维度): {config.hidden_size}")
    
    if hasattr(config, 'num_attention_heads'):
        print(f"✅ num_attention_heads (注意力头数): {config.num_attention_heads}")
        if hasattr(config, 'hidden_size'):
            head_dim = config.hidden_size // config.num_attention_heads
            print(f"✅ 每个头的维度: {head_dim} (= {config.hidden_size} / {config.num_attention_heads})")
    
    if hasattr(config, 'num_hidden_layers'):
        print(f"✅ num_hidden_layers (Transformer层数): {config.num_hidden_layers}")
    
    if hasattr(config, 'intermediate_size'):
        print(f"✅ intermediate_size (FFN中间层维度): {config.intermediate_size}")
    
    if hasattr(config, 'vocab_size'):
        print(f"✅ vocab_size (词表大小): {config.vocab_size}")
    
    if hasattr(config, 'max_position_embeddings'):
        print(f"✅ max_position_embeddings (最大序列长度): {config.max_position_embeddings}")
    
    print("\n完整配置：")
    print("-"*70)
    print(config)
    
except Exception as e:
    print(f"⚠️ 加载配置时出错: {e}")
    print("\n使用默认配置进行说明...")

print("\n" + "="*70)


