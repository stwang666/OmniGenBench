# 改进的训练配置
trainer = Trainer(
    model=model,
    epochs=10,
    learning_rate=1e-5,  # 降低学习率
    batch_size=16,
    train_dataset=datasets["train"],
    eval_dataset=datasets["valid"],
    test_dataset=datasets["test"],
    compute_metrics=metric_functions,
    gradient_accumulation_steps=4,  # 添加梯度累积
    max_grad_norm=1.0,  # 添加梯度裁剪
    weight_decay=0.01,  # 添加权重衰减
    warmup_steps=100,  # 学习率预热
    eval_steps=50,  # 更频繁的验证
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy_score",
    greater_is_better=True,
    save_total_limit=3,
)