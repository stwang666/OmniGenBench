# Step 1: Data Preparation
from omnigenbench import (
    ClassificationMetric,
    AccelerateTrainer,
    ModelHub,
    OmniTokenizer,
    OmniDatasetForSequenceClassification,
    OmniModelForSequenceClassification,
)

model_name_or_path = "yangheng/OmniGenome-52M"
dataset_name = "translation_efficiency_prediction"

# Model and Tokenizer

# We define the label mapping in the training
label2id = {"0": 0, "1": 1}  # 0: Low TE, 1: High TE

# Initialize tokenizer
tokenizer = OmniTokenizer.from_pretrained(model_name_or_path)

datasets = OmniDatasetForSequenceClassification.from_huggingface(
    dataset_name="translation_efficiency_prediction",
    tokenizer=tokenizer,
    max_length=512,
    label2id=label2id,
)

print(f"📊 Loaded datasets: {list(datasets.keys())}")
for split, dataset in datasets.items():
    print(f"  - {split}: {len(dataset)} samples")

# Step 2: Model Initialization
# === Model Initialization ===
# We support all genomic foundation models from Hugging Face Hub.
model = OmniModelForSequenceClassification(
    model_name_or_path,
    tokenizer,
    num_labels=len(list(label2id.keys())),  # Binary classification: Low TE vs High TE
)

# Step 3: Model Training
metric_functions = [ClassificationMetric().f1_score]

trainer = AccelerateTrainer(
    model=model,
    train_dataset=datasets["train"],
    eval_dataset=datasets["valid"],
    test_dataset=datasets["test"],
    compute_metrics=metric_functions,
)
print("🎓 Starting training...")

metrics = trainer.train()
trainer.save_model("ogb_te_finetuned")

print('Metrics:', metrics)

# Step 4: Model Inference and Interpretation
inference_model = ModelHub.load("yangheng/ogb_te_finetuned")

sample_sequences = {
    "Optimized sequence": "AAACCAACAAAATGCAGTAGAAGTACTCTCGAGCTATAGTCGCGACGTGCTGCCCCGCAGGAGTACAGTAGTAGTACAACGTAAGCGGGAGCAACAGACTCCCCCCCTGCAACCCACTGTGCCTGTGCCCTCGACGCGTCTCCGTCGCTTTGGCAAATGTCACGTACATATTACCGTCTCAGGCTCTCAGCCATGCTCCCTACCACCCCTGCAGCGAAGCAAAAGCCACGCACGCGGCGCCTGACATGTAACAGGACTAGACCATCTTGTTCATTTCCCGCACCCCCTCCTCTCCTCTTCCTCCATCTGCCTCTTTAAAACAGTAAAAATAACCGTGCATCCCCTGGGCAAAATCTCTCCCATACATACACTACAGCGGCGAACCTTTCCTTATTCTCGCAACGCCTCGGTAACGGGCAGCGCCTGCTCCGCGCCGCGGTTGCGAGTTCGGGAAGGCGGCCGGAGTCGCGGGGAGGAGAGGGAGGATTCGATCGGCCAGA",
    "Suboptimal sequence": "TGGAGATGGGCAGATGGCACACAAAACATGAATAGAAAACCCAAAAGGAAGGATGAAAAAAACACACACACACACACACACAAAACACAGAGAGAGAGAGAGAGAGAGCGAGAAAAGAAAAGAAAAAACCAATTCTTTTGGTCTCTTCCCTCTCCGTTTGTCGTGTCGAAGCCTTTGCCCCCACCACCTCCTCCTCTCCTCTCCCTTCCTCCCCTCCTCCCCATCTCGCTCTCCTCCCTCCTCTCTCCTCTCCTCGTCTCCTCTTCCTCTCCATTCCATTGGCCATTCCATTCCATTCCACCCCCCATGAAACCCCAAACCCTCGTCGGCCTCGCCGCGCTCGCGTAGCGCACCCGCCCTTCTCCTCTCGCCGGTGGTCCGCCGCCAGCCTCCCCCCACCCGATCCCGCCGCCCCCCCCGCCTTCACCCCGCCCACGCGGACGCATCCGATCCCGCCGCATCGCCGCGCGGGGGGGGGGGGGGGGGGGGGGGGGAGGGCACG",
    "Random sequence": "AUGC" * (128 // 4),
}
for seq_name, sequence in sample_sequences.items():
    outputs = inference_model.inference(sequence)

    # —— Result Interpretation ——
    prediction = outputs['predictions']
    confidence = outputs['confidence']
    print(f"  - Predicted Translation Efficiency: {prediction} (Confidence: {confidence:.2f})")
