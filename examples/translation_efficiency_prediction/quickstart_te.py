# -*- coding: utf-8 -*-
# file: quickstart_te.py
# Translation Efficiency Prediction Quick Start Script with Multi-Model Testing
# ==============================================================================

"""
🧬 Translation Efficiency Prediction with Foundation Model

This script demonstrates how to predict Translation Efficiency (TE) from mRNA sequences
using OmniGenBench foundation models. It supports automatic testing across multiple models.

Usage:
    # Test single model (default)
    python quickstart_te.py
    
    # Test all available models
    python quickstart_te.py --test-all-models
    
    # Test specific model
    python quickstart_te.py --model yangheng/OmniGenome-52M
"""

import argparse
import sys
import traceback
from typing import Dict, List, Optional

# ============================================================================
# Available Models for Testing
# ============================================================================

AVAILABLE_GFMS = [
    'yangheng/OmniGenome-52M',
    'yangheng/OmniGenome-186M',
    'yangheng/OmniGenome-v1.5',
    'zhihan1996/DNABERT-2-117M',
    'LongSafari/hyenadna-large-1m-seqlen-hf',
    'InstaDeepAI/nucleotide-transformer-v2-100m-multi-species',
    'kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16',
    'multimolecule/rnafm',
]

from omnigenbench import (
    ClassificationMetric,
    AccelerateTrainer,
    ModelHub,
    OmniTokenizer,
    OmniDatasetForSequenceClassification,
    OmniModelForSequenceClassification,
)

# ============================================================================
# Training and Testing Functions
# ============================================================================

def train_single_model(
    model_name_or_path: str = "yangheng/PlantRNA-FM",
    dataset_name: str = "translation_efficiency_prediction",
    max_length: int = 512,
    batch_size: int = 8,
    epochs: int = 10,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Train a single model for translation efficiency prediction.
    
    Args:
        model_name_or_path: HuggingFace model name or local path
        dataset_name: Dataset name for loading from hub
        max_length: Maximum sequence length
        batch_size: Training batch size
        epochs: Number of training epochs
        output_dir: Directory to save the trained model
        
    Returns:
        Dictionary containing training metrics
    """
    print("\n" + "=" * 80)
    print(f"🧬 Training Translation Efficiency Model")
    print(f"   Model: {model_name_or_path}")
    print("=" * 80)
    
    # Default output directory based on model name
    if output_dir is None:
        model_short_name = model_name_or_path.split('/')[-1]
        output_dir = f"te_finetuned_{model_short_name}"
    
    # Label mapping
    label2id = {"0": 0, "1": 1}  # 0: Low TE, 1: High TE
    
    # Step 1: Initialize tokenizer
    print(f"\n📝 Step 1: Loading tokenizer...")
    tokenizer = OmniTokenizer.from_pretrained(model_name_or_path)
    print(f"   ✅ Tokenizer loaded")
    
    # Step 2: Load datasets
    print(f"\n📊 Step 2: Loading datasets...")
    datasets = OmniDatasetForSequenceClassification.from_hub(
        dataset_name_or_path=dataset_name,
        tokenizer=tokenizer,
        max_length=max_length,
        label2id=label2id,
    )
    print(f"   📊 Loaded datasets: {list(datasets.keys())}")
    for split, dataset in datasets.items():
        print(f"      - {split}: {len(dataset)} samples")
    
    # Step 3: Initialize model
    print(f"\n🤖 Step 3: Initializing model...")
    model = OmniModelForSequenceClassification(
        model_name_or_path,
        tokenizer,
        num_labels=len(list(label2id.keys())),
    )
    print(f"   ✅ Model initialized")
    
    # Step 4: Training
    print(f"\n🎓 Step 4: Training...")
    metric_functions = [ClassificationMetric().f1_score]
    
    trainer = AccelerateTrainer(
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["valid"],
        test_dataset=datasets["test"],
        compute_metrics=metric_functions,
    )
    
    metrics = trainer.train()
    trainer.save_model(output_dir)
    
    print(f"\n   ✅ Training completed!")
    print(f"   📁 Model saved to: {output_dir}")
    print(f"   📊 Metrics: {metrics}")
    
    return {
        "model": model_name_or_path,
        "metrics": metrics,
        "output_dir": output_dir,
        "status": "SUCCESS"
    }


def test_inference(model_path: str, sample_sequences: Optional[Dict[str, str]] = None) -> Dict:
    """
    Test model inference on sample sequences.
    
    Args:
        model_path: Path to the trained model or HuggingFace model name
        sample_sequences: Dictionary of {name: sequence} to test
        
    Returns:
        Dictionary containing inference results
    """
    print(f"\n🔮 Testing inference with model: {model_path}")
    
    if sample_sequences is None:
        sample_sequences = {
            "Optimized sequence": "AAACCAACAAAATGCAGTAGAAGTACTCTCGAGCTATAGTCGCGACGTGCTGCCCCGCAGGAGTACAGTAGTAGTACAACGTAAGCGGGAGCAACAGACTCCCCCCCTGCAACCCACTGTGCCTGTGCCCTCGACGCGTCTCCGTCGCTTTGGCAAATGTCACGTACATATTACCGTCTCAGGCTCTCAGCCATGCTCCCTACCACCCCTGCAGCGAAGCAAAAGCCACGCACGCGGCGCCTGACATGTAACAGGACTAGACCATCTTGTTCATTTCCCGCACCCCCTCCTCTCCTCTTCCTCCATCTGCCTCTTTAAAACAGTAAAAATAACCGTGCATCCCCTGGGCAAAATCTCTCCCATACATACACTACAGCGGCGAACCTTTCCTTATTCTCGCAACGCCTCGGTAACGGGCAGCGCCTGCTCCGCGCCGCGGTTGCGAGTTCGGGAAGGCGGCCGGAGTCGCGGGGAGGAGAGGGAGGATTCGATCGGCCAGA",
            "Suboptimal sequence": "TGGAGATGGGCAGATGGCACACAAAACATGAATAGAAAACCCAAAAGGAAGGATGAAAAAAACACACACACACACACACACAAAACACAGAGAGAGAGAGAGAGAGAGCGAGAAAAGAAAAGAAAAAACCAATTCTTTTGGTCTCTTCCCTCTCCGTTTGTCGTGTCGAAGCCTTTGCCCCCACCACCTCCTCCTCTCCTCTCCCTTCCTCCCCTCCTCCCCATCTCGCTCTCCTCCCTCCTCTCTCCTCTCCTCGTCTCCTCTTCCTCTCCATTCCATTGGCCATTCCATTCCATTCCACCCCCCATGAAACCCCAAACCCTCGTCGGCCTCGCCGCGCTCGCGTAGCGCACCCGCCCTTCTCCTCTCGCCGGTGGTCCGCCGCCAGCCTCCCCCCACCCGATCCCGCCGCCCCCCCCGCCTTCACCCCGCCCACGCGGACGCATCCGATCCCGCCGCATCGCCGCGCGGGGGGGGGGGGGGGGGGGGGGGGGAGGGCACG",
            "Random sequence": "AUGC" * (128 // 4),
        }
    
    inference_model = ModelHub.load(model_path)
    results = {}
    
    for seq_name, sequence in sample_sequences.items():
        outputs = inference_model.inference(sequence)
        prediction = outputs.get('predictions', [0])[0]
        confidence = outputs.get('confidence', 0.5)
        
        results[seq_name] = {
            "prediction": prediction,
            "confidence": confidence,
            "label": "High TE" if prediction == 1 else "Low TE"
        }
        
        print(f"   {seq_name}: {results[seq_name]['label']} (confidence: {confidence:.2f})")
    
    return results


def test_all_models(
    models: List[str] = None,
    skip_training: bool = False,
    epochs: int = 5,
    batch_size: int = 8
) -> Dict[str, Dict]:
    """
    Test all available models sequentially.
    
    Args:
        models: List of model names to test. If None, uses AVAILABLE_GFMS
        skip_training: If True, only test inference on pre-trained models
        epochs: Number of training epochs for each model
        batch_size: Batch size for training
        
    Returns:
        Dictionary mapping model names to their test results
    """
    if models is None:
        models = AVAILABLE_GFMS
    
    print("\n" + "=" * 80)
    print("🧪 MULTI-MODEL TESTING FOR TRANSLATION EFFICIENCY PREDICTION")
    print("=" * 80)
    print(f"\n📋 Models to test: {len(models)}")
    for i, model in enumerate(models, 1):
        print(f"   {i}. {model}")
    
    all_results = {}
    
    for i, model_name in enumerate(models, 1):
        print(f"\n{'='*80}")
        print(f"📍 Testing Model {i}/{len(models)}: {model_name}")
        print(f"{'='*80}")
        
        try:
            if skip_training:
                # Only test inference
                result = {
                    "model": model_name,
                    "status": "INFERENCE_ONLY",
                    "inference": test_inference(model_name)
                }
            else:
                # Full training and inference
                result = train_single_model(
                    model_name_or_path=model_name,
                    epochs=epochs,
                    batch_size=batch_size
                )
                # Test inference with the trained model
                result["inference"] = test_inference(result["output_dir"])
            
            all_results[model_name] = result
            print(f"\n✅ Model {model_name} completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Model {model_name} failed with error:")
            print(f"   {str(e)}")
            traceback.print_exc()
            all_results[model_name] = {
                "model": model_name,
                "status": "FAILED",
                "error": str(e)
            }
    
    # Print summary
    print_test_summary(all_results)
    
    return all_results


def print_test_summary(results: Dict[str, Dict]):
    """Print a summary of all model test results."""
    print("\n" + "=" * 80)
    print("📊 MULTI-MODEL TEST SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results.values() if r.get("status") != "FAILED")
    failed = len(results) - successful
    
    print(f"\n✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(results)}")
    
    print(f"\n{'Model':<60} {'Status':<15}")
    print("-" * 75)
    
    for model_name, result in results.items():
        status = result.get("status", "UNKNOWN")
        status_emoji = "✅" if status != "FAILED" else "❌"
        print(f"{status_emoji} {model_name:<58} {status:<15}")
    
    print("\n" + "=" * 80)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Translation Efficiency Prediction with Multiple Model Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default model (PlantRNA-FM)
    python quickstart_te.py
    
    # Test all available models
    python quickstart_te.py --test-all-models
    
    # Train with specific model
    python quickstart_te.py --model yangheng/OmniGenome-52M
    
    # Test all models with custom epochs
    python quickstart_te.py --test-all-models --epochs 5 --batch-size 4
    
    # Skip training, only test inference
    python quickstart_te.py --test-all-models --skip-training
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yangheng/PlantRNA-FM",
        help="Model name or path (default: yangheng/PlantRNA-FM)"
    )
    parser.add_argument(
        "--test-all-models", "-a",
        action="store_true",
        help="Test all models in AVAILABLE_GFMS list"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8,
        help="Training batch size (default: 8)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, only test inference on pre-trained models"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit"
    )
    
    args = parser.parse_args()
    
    # List models and exit
    if args.list_models:
        print("\n📋 Available Models for Testing:")
        print("-" * 50)
        for i, model in enumerate(AVAILABLE_GFMS, 1):
            print(f"   {i}. {model}")
        print("\nTo enable more models, edit the AVAILABLE_GFMS list in quickstart_te.py")
        return
    
    print("\n" + "=" * 80)
    print("🧬 Translation Efficiency Prediction - Quick Start")
    print("=" * 80)
    
    if args.test_all_models:
        # Test all available models
        results = test_all_models(
            models=AVAILABLE_GFMS,
            skip_training=args.skip_training,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    else:
        # Train single model
        if args.skip_training:
            print(f"\n🔮 Testing inference with pre-trained model: {args.model}")
            test_inference(args.model)
        else:
            result = train_single_model(
                model_name_or_path=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                output_dir=args.output_dir
            )
            # Test inference with trained model
            print(f"\n🔮 Testing inference with trained model...")
            test_inference(result["output_dir"])
    
    print("\n🎉 All tasks completed!")


if __name__ == "__main__":
    main()
