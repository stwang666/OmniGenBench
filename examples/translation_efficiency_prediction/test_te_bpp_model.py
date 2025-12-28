# -*- coding: utf-8 -*-
# file: test_te_bpp_model.py
# time: 12:30 01/11/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Copyright (C) 2019-2025. All Rights Reserved.

"""
Test script for TE-BPP model implementation with multi-model support.

This script validates the core functionality of the TEModelWithBPP:
1. Dataset creation and BPP computation
2. Model initialization
3. Forward pass
4. Training pipeline
5. Inference
6. Multi-model compatibility testing

Usage:
    # Run all tests with default model
    python test_te_bpp_model.py
    
    # Test all available models
    python test_te_bpp_model.py --test-all-models
    
    # Test specific model
    python test_te_bpp_model.py --model yangheng/OmniGenome-52M
    
    # List available models
    python test_te_bpp_model.py --list-models
"""

import os
import sys
import argparse
import torch
import numpy as np
import traceback
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from te_bpp_model import (
    TEDatasetWithBPP,
    TEModelWithBPP,
    BPPProcessor,
    FeatureFusion,
    create_demo_dataset,
)

from omnigenbench import OmniTokenizer

# ============================================================================
# Available Models for Testing
# ============================================================================

AVAILABLE_GFMS = [
    'yangheng/OmniGenome-52M',
    # 'yangheng/OmniGenome-186M',
    # 'yangheng/OmniGenome-v1.5',
    # 'zhihan1996/DNABERT-2-117M',
    # 'LongSafari/hyenadna-large-1m-seqlen-hf',
    # 'InstaDeepAI/nucleotide-transformer-v2-100m-multi-species',
    # 'kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16',
    # 'multimolecule/rnafm',
]


def test_bpp_computation(model_name: str = "yangheng/PlantRNA-FM"):
    """Test BPP matrix computation."""
    print("\n" + "=" * 80)
    print("TEST 1: BPP Matrix Computation")
    print(f"   Model: {model_name}")
    print("=" * 80)
    
    # Create a simple RNA sequence
    sequence = "AUGCAUGCGCGCGCGCAUGCAUGC"
    
    print(f"[INFO] Test sequence: {sequence}")
    print(f"[INFO] Length: {len(sequence)}")
    
    # Initialize dataset to access BPP computation method
    tokenizer = OmniTokenizer.from_pretrained(model_name)
    dataset = TEDatasetWithBPP(
        dataset_name_or_path=[],  # Empty dataset for testing
        tokenizer=tokenizer,
        max_length=512
    )
    
    # Compute BPP
    bpp_matrix = dataset.compute_bpp_matrix(sequence)
    
    # Validate BPP matrix
    print(f"\n[CHECK] BPP matrix shape: {bpp_matrix.shape}")
    assert bpp_matrix.shape == (len(sequence), len(sequence)), "BPP shape mismatch"
    
    print(f"[CHECK] BPP matrix dtype: {bpp_matrix.dtype}")
    assert bpp_matrix.dtype == np.float32, "BPP dtype should be float32"
    
    print(f"[CHECK] BPP symmetry: ", end="")
    is_symmetric = np.allclose(bpp_matrix, bpp_matrix.T)
    assert is_symmetric, "BPP matrix should be symmetric"
    print("PASS")
    
    print(f"[CHECK] BPP value range: [{bpp_matrix.min():.4f}, {bpp_matrix.max():.4f}]")
    assert bpp_matrix.min() >= 0 and bpp_matrix.max() <= 1, "BPP values out of range"
    
    print(f"[CHECK] Diagonal elements: {np.diag(bpp_matrix).sum()}")
    assert np.allclose(np.diag(bpp_matrix), 0), "Diagonal should be zero"
    
    print(f"\n[SUCCESS] BPP computation test passed!")
    return True


def test_dataset_preparation(model_name: str = "yangheng/PlantRNA-FM"):
    """Test dataset with BPP feature preparation."""
    print("\n" + "=" * 80)
    print("TEST 2: Dataset Preparation")
    print(f"   Model: {model_name}")
    print("=" * 80)
    
    # Create demo dataset
    dataset_dir = create_demo_dataset("test_te_dataset")
    
    # Load dataset
    tokenizer = OmniTokenizer.from_pretrained(model_name)
    label2id = {"0": 0, "1": 1}
    
    dataset = TEDatasetWithBPP(
        dataset_name_or_path=f"{dataset_dir}/train.json",
        tokenizer=tokenizer,
        max_length=512,
        label2id=label2id
    )
    
    print(f"[INFO] Dataset size: {len(dataset)}")
    assert len(dataset) > 0, "Dataset should not be empty"
    
    # Get a sample
    sample = dataset[0]
    
    print(f"\n[CHECK] Sample keys: {list(sample.keys())}")
    required_keys = ["input_ids", "attention_mask", "bpp_matrix", "labels"]
    for key in required_keys:
        assert key in sample, f"Missing key: {key}"
    
    print(f"[CHECK] input_ids shape: {sample['input_ids'].shape}")
    print(f"[CHECK] attention_mask shape: {sample['attention_mask'].shape}")
    print(f"[CHECK] bpp_matrix shape: {sample['bpp_matrix'].shape}")
    print(f"[CHECK] labels shape: {sample['labels'].shape}")
    
    assert sample['bpp_matrix'].shape == (512, 512), "BPP matrix should be 512x512"
    assert sample['input_ids'].shape[0] == 512, "Input should be padded to 512"
    
    print(f"\n[SUCCESS] Dataset preparation test passed!")
    
    # Cleanup
    import shutil
    shutil.rmtree(dataset_dir)
    
    return True


def test_bpp_processor():
    """Test BPP processor module."""
    print("\n" + "=" * 80)
    print("TEST 3: BPP Processor")
    print("=" * 80)
    
    # Create processor
    processor = BPPProcessor(output_dim=128)
    
    # Create dummy BPP matrix
    batch_size = 4
    seq_len = 512
    bpp_matrix = torch.randn(batch_size, seq_len, seq_len)
    
    print(f"[INFO] Input shape: {bpp_matrix.shape}")
    
    # Forward pass
    features = processor(bpp_matrix)
    
    print(f"[CHECK] Output shape: {features.shape}")
    assert features.shape == (batch_size, 128), "Output shape mismatch"
    
    print(f"[CHECK] Output dtype: {features.dtype}")
    assert features.dtype == torch.float32, "Output should be float32"
    
    print(f"\n[SUCCESS] BPP processor test passed!")
    return True


def test_feature_fusion():
    """Test feature fusion module."""
    print("\n" + "=" * 80)
    print("TEST 4: Feature Fusion")
    print("=" * 80)
    
    # Create fusion module
    seq_dim = 768
    bpp_dim = 128
    fusion = FeatureFusion(seq_dim=seq_dim, bpp_dim=bpp_dim, output_dim=256)
    
    # Create dummy features
    batch_size = 4
    seq_features = torch.randn(batch_size, seq_dim)
    bpp_features = torch.randn(batch_size, bpp_dim)
    
    print(f"[INFO] Sequence features shape: {seq_features.shape}")
    print(f"[INFO] BPP features shape: {bpp_features.shape}")
    
    # Forward pass
    fused = fusion(seq_features, bpp_features)
    
    print(f"[CHECK] Fused features shape: {fused.shape}")
    assert fused.shape == (batch_size, 256), "Fused shape mismatch"
    
    print(f"\n[SUCCESS] Feature fusion test passed!")
    return True


def test_model_forward(model_name: str = "yangheng/PlantRNA-FM"):
    """Test complete model forward pass."""
    print("\n" + "=" * 80)
    print("TEST 5: Model Forward Pass")
    print(f"   Model: {model_name}")
    print("=" * 80)
    
    # Initialize model
    tokenizer = OmniTokenizer.from_pretrained(model_name)
    model = TEModelWithBPP(
        config_or_model=model_name,
        tokenizer=tokenizer,
        num_labels=2,
        label2id={"0": 0, "1": 1}
    )
    
    print(f"[INFO] Model initialized")
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 512
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    bpp_matrix = torch.randn(batch_size, seq_len, seq_len)
    labels = torch.tensor([0, 1])
    
    print(f"[INFO] Input shapes:")
    print(f"  - input_ids: {input_ids.shape}")
    print(f"  - attention_mask: {attention_mask.shape}")
    print(f"  - bpp_matrix: {bpp_matrix.shape}")
    print(f"  - labels: {labels.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bpp_matrix=bpp_matrix,
            labels=labels
        )
    
    print(f"\n[CHECK] Output keys: {list(outputs.keys())}")
    required_keys = ["loss", "logits", "sequence_features", "bpp_features"]
    for key in required_keys:
        assert key in outputs, f"Missing output key: {key}"
    
    print(f"[CHECK] Loss: {outputs['loss'].item():.4f}")
    assert outputs['loss'].item() >= 0, "Loss should be non-negative"
    
    print(f"[CHECK] Logits shape: {outputs['logits'].shape}")
    assert outputs['logits'].shape == (batch_size, 2), "Logits shape mismatch"
    
    print(f"[CHECK] Sequence features shape: {outputs['sequence_features'].shape}")
    print(f"[CHECK] BPP features shape: {outputs['bpp_features'].shape}")
    
    print(f"\n[SUCCESS] Model forward pass test passed!")
    return True


def test_inference(model_name: str = "yangheng/PlantRNA-FM"):
    """Test model inference."""
    print("\n" + "=" * 80)
    print("TEST 6: Model Inference")
    print(f"   Model: {model_name}")
    print("=" * 80)
    
    # Initialize model
    tokenizer = OmniTokenizer.from_pretrained(model_name)
    model = TEModelWithBPP(
        config_or_model=model_name,
        tokenizer=tokenizer,
        num_labels=2,
        label2id={"0": 0, "1": 1},
        dataset_class=TEDatasetWithBPP
    )
    
    print(f"[INFO] Model initialized for inference")
    
    # Test sequence
    test_sequence = "AUGCAUGCAUGCGCGCGCGC" * 20
    
    print(f"[INFO] Test sequence length: {len(test_sequence)}")
    
    # Run inference
    model.eval()
    outputs = model.inference({"sequence": test_sequence})
    
    print(f"\n[CHECK] Inference output keys: {list(outputs.keys())}")
    
    if "predictions" in outputs:
        prediction = outputs["predictions"][0]
        print(f"[CHECK] Prediction: {prediction} ({'High TE' if prediction == 1 else 'Low TE'})")
        assert prediction in [0, 1], "Prediction should be 0 or 1"
    
    if "confidence" in outputs:
        confidence = outputs["confidence"]
        print(f"[CHECK] Confidence: {confidence:.4f}")
        assert 0 <= confidence <= 1, "Confidence should be in [0, 1]"
    
    print(f"\n[SUCCESS] Inference test passed!")
    return True


def test_training_mini(model_name: str = "yangheng/PlantRNA-FM"):
    """Test minimal training loop."""
    print("\n" + "=" * 80)
    print("TEST 7: Mini Training Loop")
    print(f"   Model: {model_name}")
    print("=" * 80)
    
    from te_bpp_model import train_te_model
    
    print(f"[INFO] Starting mini training with demo data...")
    print(f"[INFO] This will use only 10 samples and 2 epochs for testing")
    
    model_short_name = model_name.split('/')[-1]
    output_dir = f"test_te_model_{model_short_name}"
    
    try:
        results = train_te_model(
            model_name=model_name,
            use_demo=True,
            batch_size=2,
            epochs=2,  # Very short training for testing
            output_dir=output_dir
        )
        
        print(f"\n[CHECK] Training completed")
        print(f"[CHECK] Results: {results}")
        
        # Validate results (check for various possible metric names)
        has_metrics = any(key in results for key in ["f1_score", "f1", "accuracy", "test_f1_score"])
        if not has_metrics:
            print(f"[WARNING] No standard metrics found in results")
        
        print(f"\n[SUCCESS] Training test passed!")
        
        # Cleanup
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if os.path.exists("translation_efficiency_prediction"):
            shutil.rmtree("translation_efficiency_prediction")
        
        return True
        
    except Exception as e:
        print(f"\n[WARNING] Training test failed: {e}")
        print(f"[INFO] This is expected if GPU/resources are limited")
        traceback.print_exc()
        return False


def run_all_tests(model_name: str = "yangheng/PlantRNA-FM"):
    """Run all tests for a single model."""
    print("\n" + "=" * 80)
    print(f"RUNNING ALL TESTS FOR TE-BPP MODEL")
    print(f"Model: {model_name}")
    print("=" * 80)
    
    # Tests that require model
    model_tests = [
        ("BPP Computation", test_bpp_computation),
        ("Dataset Preparation", test_dataset_preparation),
        ("Model Forward", test_model_forward),
        ("Model Inference", test_inference),
        ("Mini Training", test_training_mini),
    ]
    
    # Tests that don't require model (model-agnostic)
    generic_tests = [
        ("BPP Processor", test_bpp_processor),
        ("Feature Fusion", test_feature_fusion),
    ]
    
    results = {}
    
    # Run generic tests first (only once)
    for test_name, test_func in generic_tests:
        try:
            success = test_func()
            results[test_name] = "PASS" if success else "FAIL"
        except Exception as e:
            print(f"\n[ERROR] {test_name} failed with exception:")
            print(f"  {str(e)}")
            results[test_name] = "ERROR"
            traceback.print_exc()
    
    # Run model-specific tests
    for test_name, test_func in model_tests:
        try:
            success = test_func(model_name)
            results[test_name] = "PASS" if success else "FAIL"
        except Exception as e:
            print(f"\n[ERROR] {test_name} failed with exception:")
            print(f"  {str(e)}")
            results[test_name] = "ERROR"
            traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"TEST SUMMARY FOR: {model_name}")
    print("=" * 80)
    
    for test_name, result in results.items():
        status_symbol = {
            "PASS": "✓",
            "FAIL": "✗",
            "ERROR": "⚠"
        }.get(result, "?")
        print(f"  {status_symbol} {test_name}: {result}")
    
    passed = sum(1 for r in results.values() if r == "PASS")
    total = len(results)
    
    print(f"\n[RESULT] {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All tests passed! 🎉")
        return True
    else:
        print("[WARNING] Some tests failed. Please review the errors above.")
        return False


def run_all_models_tests(models: List[str] = None) -> Dict[str, Dict]:
    """
    Run all tests for multiple models.
    
    Args:
        models: List of model names to test. If None, uses AVAILABLE_GFMS
        
    Returns:
        Dictionary mapping model names to their test results
    """
    if models is None:
        models = AVAILABLE_GFMS
    
    print("\n" + "=" * 80)
    print("🧪 MULTI-MODEL TESTING FOR TE-BPP MODEL")
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
            success = run_all_tests(model_name)
            all_results[model_name] = {
                "status": "PASS" if success else "FAIL",
                "all_passed": success
            }
            
            status_emoji = "✅" if success else "⚠️"
            print(f"\n{status_emoji} Model {model_name} testing completed!")
            
        except Exception as e:
            print(f"\n❌ Model {model_name} failed with error:")
            print(f"   {str(e)}")
            traceback.print_exc()
            all_results[model_name] = {
                "status": "ERROR",
                "error": str(e),
                "all_passed": False
            }
    
    # Print final summary
    print("\n" + "=" * 80)
    print("📊 MULTI-MODEL TEST FINAL SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in all_results.values() if r.get("all_passed", False))
    failed = len(all_results) - successful
    
    print(f"\n✅ Fully Passed: {successful}")
    print(f"❌ Failed/Partial: {failed}")
    print(f"📊 Total: {len(all_results)}")
    
    print(f"\n{'Model':<60} {'Status':<15}")
    print("-" * 75)
    
    for model_name, result in all_results.items():
        status = result.get("status", "UNKNOWN")
        all_passed = result.get("all_passed", False)
        status_emoji = "✅" if all_passed else ("⚠️" if status == "FAIL" else "❌")
        print(f"{status_emoji} {model_name:<58} {status:<15}")
    
    print("\n" + "=" * 80)
    
    return all_results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Test TE-BPP Model Implementation with Multi-Model Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all tests with default model
    python test_te_bpp_model.py
    
    # Test all available models
    python test_te_bpp_model.py --test-all-models
    
    # Test specific model
    python test_te_bpp_model.py --model yangheng/OmniGenome-52M
    
    # List available models
    python test_te_bpp_model.py --list-models
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yangheng/PlantRNA-FM",
        help="Model name or path to test (default: yangheng/PlantRNA-FM)"
    )
    parser.add_argument(
        "--test-all-models", "-a",
        action="store_true",
        help="Test all models in AVAILABLE_GFMS list"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit"
    )
    
    args = parser.parse_args()
    
    import warnings
    warnings.filterwarnings("ignore")
    
    # List models and exit
    if args.list_models:
        print("\n📋 Available Models for Testing:")
        print("-" * 50)
        for i, model in enumerate(AVAILABLE_GFMS, 1):
            print(f"   {i}. {model}")
        print("\nTo enable more models, edit the AVAILABLE_GFMS list in test_te_bpp_model.py")
        return
    
    print("\n" + "=" * 80)
    print("🧬 TE-BPP Model Test Suite")
    print("=" * 80)
    
    if args.test_all_models:
        # Test all available models
        results = run_all_models_tests(AVAILABLE_GFMS)
        all_passed = all(r.get("all_passed", False) for r in results.values())
        sys.exit(0 if all_passed else 1)
    else:
        # Test single model
        success = run_all_tests(args.model)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
