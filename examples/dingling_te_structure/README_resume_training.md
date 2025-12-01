# 断点续训和最佳模型查找指南

本文档介绍如何从checkpoint继续训练，以及如何从历史所有checkpoint中找出最佳模型进行测试。

## 📁 相关文件

- `biclass_seq_tissue_v1.5_resume.py` - 支持断点续训的训练脚本
- `find_best_model.py` - 查找和测试历史最佳模型的独立脚本

## 🔄 功能1：从Epoch 19继续训练

### 特点

✅ **Epoch编号连续** - 从epoch 20开始显示，而不是重新从epoch 0开始  
✅ **自动加载checkpoint** - 加载epoch 19的模型权重  
✅ **避免OOM** - 降低batch_size从16到4（有效batch_size通过梯度累积保持为16）  
✅ **训练结束自动找最佳模型** - 训练完成后自动查找历史所有checkpoint中最好的模型并测试

### 使用方法

```bash
cd /home/sw1136/OmniGenBench/examples/dingling_te_structure
python biclass_seq_tissue_v1.5_resume.py
```

### 配置参数（脚本中可修改）

```python
# 在脚本中可以修改这些参数：
RESUME_FROM_CHECKPOINT = True  # 是否从checkpoint恢复
CHECKPOINT_PATH = "ogb_te_2class_finetuned_v1.5_seq_tissue_epoch_19_..."  # checkpoint路径
START_EPOCH = 19  # 从第19个epoch继续（显示为epoch 20, 21, ...）
TOTAL_EPOCHS = 60  # 总共要训练60个epoch
```

### 训练过程

1. **加载checkpoint** - 从epoch 19的保存点加载模型权重
2. **继续训练** - Epoch编号从20开始，训练到60
3. **保存新checkpoint** - 每个epoch保存为 `ogb_te_2class_finetuned_v1.5_seq_tissue_resumed_epoch_XX`
4. **查找最佳模型** - 训练结束后自动扫描所有历史checkpoint
5. **测试最佳模型** - 使用找到的最佳模型在测试集上评估

### 输出示例

```
🔄 从checkpoint恢复训练...
  - Checkpoint: ogb_te_2class_finetuned_v1.5_seq_tissue_epoch_19_...
  - 已完成epoch: 19
  - 剩余epoch: 41
✅ 成功加载checkpoint

🎓 开始训练...
  - Epoch范围: 20 到 60
  - Batch size: 4 (有效batch size: 16)
  - Learning rate: 2e-5

Epoch 20/60 Loss: ...
Epoch 21/60 Loss: ...
...

🔍 开始查找历史最佳模型...
📊 所有checkpoint按accuracy_score排序：
  1. Epoch 18: accuracy_score=0.7605 - ogb_te_2class_finetuned_v1.5_seq_tissue_epoch_18_...
  2. Epoch  7: accuracy_score=0.7599 - ogb_te_2class_finetuned_v1.5_seq_tissue_epoch_7_...
  ...
✅ 最佳checkpoint: Epoch 18, accuracy_score=0.7605

🎯 使用历史最佳模型进行最终测试...
📊 使用历史最佳模型的测试集结果：
  accuracy_score: 0.7580
  f1_score: 0.7365
```

## 🏆 功能2：独立查找和测试最佳模型

如果你不想重新训练，只想从现有的checkpoint中找出最佳模型进行测试，可以使用独立脚本。

### 使用方法

```bash
cd /home/sw1136/OmniGenBench/examples/dingling_te_structure
python find_best_model.py
```

### 功能特点

✅ **完整表格展示** - 显示所有checkpoint的完整信息  
✅ **多metric排序** - 可以按accuracy_score或f1_score排序  
✅ **Top-K显示** - 显示前5个最佳checkpoint  
✅ **交互式选择** - 让用户选择要测试哪个模型  
✅ **自动测试** - 在测试集上评估选定的模型

### 运行示例

```bash
$ python find_best_model.py

🔍 查找所有checkpoint...
  - 目录: /home/sw1136/OmniGenBench/examples/dingling_te_structure
  - 前缀: ogb_te_2class_finetuned_v1.5_seq_tissue

✅ 找到 19 个checkpoint

========================================================================================================================
  Epoch   |   Accuracy   |   F1 Score   |                     Checkpoint Name                                       
========================================================================================================================
    1     |    0.7045    |    0.6482    | ogb_te_2class_finetuned_v1.5_seq_tissue_epoch_1_seed_42_accuracy_...
    2     |    0.7383    |    0.6781    | ogb_te_2class_finetuned_v1.5_seq_tissue_epoch_2_seed_42_accuracy_...
    ...
   19     |    0.7517    |    0.7234    | ogb_te_2class_finetuned_v1.5_seq_tissue_epoch_19_seed_42_accuracy_...
========================================================================================================================

📊 按 accuracy_score 排序的Top 5 checkpoint:
========================================================================================================================
  1. Epoch 18: accuracy_score=0.7605 (f1_score=0.7377)
  2. Epoch  7: accuracy_score=0.7599 (f1_score=0.7321)
  3. Epoch 16: accuracy_score=0.7582 (f1_score=0.7365)
  4. Epoch 13: accuracy_score=0.7558 (f1_score=0.7263)
  5. Epoch  6: accuracy_score=0.7547 (f1_score=0.7272)
========================================================================================================================

✅ 最佳checkpoint (按accuracy_score):
   - Epoch: 18
   - accuracy_score: 0.7605
   - f1_score: 0.7377
   - 路径: /home/sw1136/OmniGenBench/examples/dingling_te_structure/ogb_te_2class_finetuned_v1.5_seq_tissue_epoch_18_...

选择要在测试集上评估的模型：
  1. 按accuracy_score最佳的模型
  2. 按f1_score最佳的模型
  3. 两个都测试
  4. 不测试，仅显示排名
请输入选择 (1/2/3/4，默认3): 1

🔄 加载checkpoint: ogb_te_2class_finetuned_v1.5_seq_tissue_epoch_18_...
✅ 模型加载成功
🧪 在测试集上评估...

================================================================================
📊 测试集结果：
================================================================================
  accuracy_score: 0.7580
  f1_score: 0.7365
================================================================================
```

## 🎯 推荐使用流程

### 场景1：训练中断，需要继续

```bash
# 1. 修改 biclass_seq_tissue_v1.5_resume.py 中的参数
#    - CHECKPOINT_PATH: 设置为最新的checkpoint
#    - START_EPOCH: 设置为已完成的epoch数
#    - TOTAL_EPOCHS: 设置为计划的总epoch数

# 2. 运行续训脚本
python biclass_seq_tissue_v1.5_resume.py

# 训练结束后会自动找出并测试历史最佳模型
```

### 场景2：训练已完成，查找最佳模型

```bash
# 直接运行查找脚本
python find_best_model.py

# 选择要测试的模型（按accuracy或f1排序）
```

### 场景3：降低显存使用

如果继续遇到OOM错误，可以进一步调整参数：

```python
# 在 biclass_seq_tissue_v1.5_resume.py 中修改：
trainer = ResumableAccelerateTrainer(
    batch_size=2,  # 从4降到2
    gradient_accumulation_steps=8,  # 从4增加到8，保持有效batch_size=16
    ...
)
```

## 📊 当前最佳结果（基于epoch 1-19）

根据验证集表现：

| Metric | Best Epoch | Score |
|--------|-----------|-------|
| **Accuracy** | Epoch 18 | 0.7605 |
| **F1 Score** | Epoch 15 | 0.7377 |

建议重点关注：
- **Epoch 18**: 综合表现最好（accuracy=0.7605, f1=0.7377）
- **Epoch 16**: 次优选择（accuracy=0.7582, f1=0.7365）

## ⚠️ 注意事项

1. **Optimizer状态**：从checkpoint恢复时，optimizer和学习率调度器会重新初始化，不会保留历史状态
2. **Early stopping计数器**：patience计数器会重置
3. **显存管理**：batch_size已调整为4，如仍OOM可进一步降低
4. **Checkpoint命名**：续训的checkpoint会带有"resumed"标记

## 🤔 FAQ

**Q: 为什么不直接保存optimizer状态？**  
A: 当前框架的checkpoint机制只保存模型权重。如需完整状态，需修改框架代码。

**Q: Epoch编号为什么从20开始而不是0？**  
A: 使用了自定义的`ResumableAccelerateTrainer`，设置了`start_epoch=19`参数。

**Q: 如何选择按accuracy还是f1？**  
A: 取决于你的任务：
- 类别平衡 → accuracy
- 类别不平衡 → f1_score
- 不确定 → 看两者，选综合最好的

**Q: 能否同时从多个checkpoint继续？**  
A: 只能选择一个checkpoint作为起点，但训练结束后会比较所有历史checkpoint。







