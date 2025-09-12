#!/usr/bin/env python3
"""
Per-Appliance Evaluation Script for REDD NILM Results
Provides detailed performance metrics for each individual appliance
"""

import numpy as np
import json
import os
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import sys

# Add the src directory to the path
sys.path.append('src')
sys.path.append('.')

# Import WeightedRF from the correct path
from src.SER.WeightedRF import WeightedRandomForest

def sample_from_large_categories(Xt, yt, n_samples=1):
    """Sample from large categories for tuning"""
    wanted_appl = np.unique(yt)
    idx = []
    for name in wanted_appl:
        appl_idx = np.where(yt == name)[0]
        for i in range(0, len(appl_idx), 10):
            appl_idx_i = np.random.choice(appl_idx[i:i+10], n_samples, replace=False)
            idx.append(appl_idx_i)
    
    idx = np.array(idx).flatten()
    Xtune, ytune = Xt[idx], yt[idx]
    Xtune, ytune = Xtune.reshape(len(idx), -1), ytune.reshape(-1)
    
    Xtest, ytest = np.delete(Xt, idx, axis=0), np.delete(yt, idx, axis=0)
    return Xtest, Xtune, ytest, ytune

def get_appliance_names():
    """Get appliance names mapping"""
    return {
        0: 'dishwasher',
        1: 'fridge', 
        2: 'microwave',
        3: 'washingmachine'
    }

def evaluate_per_appliance(y_true, y_pred, appliance_names):
    """Calculate per-appliance performance metrics"""
    results = {}
    
    # Overall metrics
    overall_accuracy = accuracy_score(y_true, y_pred) * 100
    overall_f1 = f1_score(y_true, y_pred, average='macro') * 100
    overall_precision = precision_score(y_true, y_pred, average='macro') * 100
    overall_recall = recall_score(y_true, y_pred, average='macro') * 100
    
    results['Overall'] = {
        'Accuracy': float(overall_accuracy),
        'F1_macro': float(overall_f1),
        'Precision': float(overall_precision),
        'Recall': float(overall_recall),
        'Total_samples': int(len(y_true))
    }
    
    # Per-appliance metrics
    per_class_precision = precision_score(y_true, y_pred, average=None)
    per_class_recall = recall_score(y_true, y_pred, average=None)
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    
    # Confusion matrix for per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    
    results['Per_Appliance'] = {}
    
    for class_idx in np.unique(y_true):
        appliance_name = appliance_names.get(class_idx, f'class_{class_idx}')
        
        # True samples for this appliance
        true_samples = np.sum(y_true == class_idx)
        
        # Correctly predicted samples
        correct_predictions = cm[class_idx, class_idx] if class_idx < len(cm) else 0
        
        # Per-class accuracy
        class_accuracy = (correct_predictions / true_samples * 100) if true_samples > 0 else 0
        
        results['Per_Appliance'][appliance_name] = {
            'Accuracy': float(class_accuracy),
            'Precision': float(per_class_precision[class_idx] * 100) if class_idx < len(per_class_precision) else 0.0,
            'Recall': float(per_class_recall[class_idx] * 100) if class_idx < len(per_class_recall) else 0.0,
            'F1_score': float(per_class_f1[class_idx] * 100) if class_idx < len(per_class_f1) else 0.0,
            'True_samples': int(true_samples),
            'Correct_predictions': int(correct_predictions)
        }
    
    return results

def run_per_appliance_evaluation(k_target=1, n_trees=9):
    """Run complete per-appliance evaluation"""
    
    print("=== REDD Per-Appliance Evaluation ===")
    print("Loading REDD datasets...")
    
    # Load training dataset
    Xtrain = np.load("src/data/redd/training_datasets/X.npy")
    ytrain = np.load("src/data/redd/training_datasets/Y.npy")
    
    # Load validation dataset  
    X_val = np.load("src/data/redd/validation_datasets/X.npy")
    y_val = np.load("src/data/redd/validation_datasets/Y.npy")
    
    # Load testing dataset (cross-house evaluation: House 1)
    Xtest = np.load("src/data/redd/testing_datasets/X.npy")
    ytest = np.load("src/data/redd/testing_datasets/Y.npy")
    
    print(f"Data Loaded:")
    print(f"  Training: {Xtrain.shape[0]} samples from House 3")
    print(f"  Validation: {X_val.shape[0]} samples from House 3")
    print(f"  Testing: {Xtest.shape[0]} samples from House 1 (cross-house)")
    
    # Get appliance distribution
    appliance_names = get_appliance_names()
    print(f"\nAppliance Distribution in Testing Data:")
    for class_idx in np.unique(ytest):
        count = np.sum(ytest == class_idx)
        appliance_name = appliance_names.get(class_idx, f'class_{class_idx}')
        print(f"  {appliance_name}: {count} samples")
    
    # Sample from testing data for tuning (few-shot learning)
    Xtest_eval, X_tune, ytest_eval, y_tune = sample_from_large_categories(Xtest, ytest, n_samples=k_target)
    
    print(f"\nTraining and Evaluation:")
    print(f"  Training samples: {len(ytrain)}")
    print(f"  Tuning samples (few-shot): {len(y_tune)}")
    print(f"  Final testing samples: {len(ytest_eval)}")
    
    # Train source model
    print("  Training source model...")
    src_model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    src_model.fit(Xtrain, ytrain)
    
    # Create and update target model
    print("  Creating target model...")
    tgt_model = WeightedRandomForest(src_model, n_update=0.8, w_new=0.8, original_ser=False)
    tgt_model.update_forest(X_tune, y_tune)
    
    # Make predictions
    print("  Making predictions...")
    y_pred = tgt_model.predict(Xtest_eval)
    
    # Evaluate per-appliance performance
    print("  Calculating per-appliance metrics...")
    results = evaluate_per_appliance(ytest_eval, y_pred, appliance_names)
    
    # Add experiment details
    results['Experiment_Details'] = {
        'Description': 'Cross-house evaluation: Model trained on House 3, tested on House 1',
        'Training_samples': int(len(ytrain)),
        'Validation_samples': int(len(X_val)),
        'Tuning_samples': int(len(y_tune)),
        'Testing_samples': int(len(ytest_eval)),
        'k_target': k_target,
        'n_trees': n_trees,
        'Training_source': 'House 3',
        'Testing_source': 'House 1'
    }
    
    # Save results
    output_dir = "results/redd/kt_1_ori"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/PER_APPLIANCE_RESULTS.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4, separators=(",", ": "))
    
    print(f"\n=== Per-Appliance Results ===")
    print(f"Overall Performance:")
    print(f"  Accuracy: {results['Overall']['Accuracy']:.2f}%")
    print(f"  F1-macro: {results['Overall']['F1_macro']:.2f}%")
    print(f"  Precision: {results['Overall']['Precision']:.2f}%")
    print(f"  Recall: {results['Overall']['Recall']:.2f}%")
    
    print(f"\nPer-Appliance Performance:")
    for appliance, metrics in results['Per_Appliance'].items():
        print(f"  {appliance}:")
        print(f"    Accuracy: {metrics['Accuracy']:.2f}%")
        print(f"    Precision: {metrics['Precision']:.2f}%")
        print(f"    Recall: {metrics['Recall']:.2f}%")
        print(f"    F1-score: {metrics['F1_score']:.2f}%")
        print(f"    Samples: {metrics['True_samples']} (Correct: {metrics['Correct_predictions']})")
    
    print(f"\nResults saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    print("Starting per-appliance evaluation for REDD NILM...")
    try:
        results = run_per_appliance_evaluation(k_target=1, n_trees=9)
        print("\n" + "="*60)
        print("SUCCESS: Per-appliance evaluation completed!")
        print("="*60)
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
