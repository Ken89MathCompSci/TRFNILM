#!/usr/bin/env python3
"""
Simple test script to run REDD training and capture output
"""

import sys
import os
sys.path.append('src')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from SER.WeightedRF import WeightedRandomForest

def sample_from_large_categories(Xt, yt, n_samples=1):
    """Sample from large categories for tuning"""
    wanted_appl = np.unique(yt)
    idx = []
    for name in wanted_appl:
        appl_idx = np.where(yt == name)[0]
        for i in range(0, len(appl_idx), 10):
            if len(appl_idx[i:i+10]) >= n_samples:
                appl_idx_i = np.random.choice(appl_idx[i:i+10], n_samples, replace=False)
                idx.append(appl_idx_i)
    
    if len(idx) == 0:
        # Fallback: just take one sample per class
        for name in wanted_appl:
            appl_idx = np.where(yt == name)[0]
            if len(appl_idx) > 0:
                idx.append([appl_idx[0]])
    
    idx = np.array(idx).flatten()
    X_tune, y_tune = Xt[idx], yt[idx]
    X_test, y_test = np.delete(Xt, idx, axis=0), np.delete(yt, idx, axis=0)
    return X_test, X_tune, y_test, y_tune

def main():
    print("=== REDD Training Test ===")
    
    # Load REDD data
    try:
        data = np.load("src/data/redd/X.npy")
        labels = np.load("src/data/redd/Y.npy")
        house_labels = np.load("src/data/redd/house_labels.npy")
        
        print(f"Loaded data shape: {data.shape}")
        print(f"Loaded labels shape: {labels.shape}")
        print(f"Unique labels: {np.unique(labels)}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    try:
        # Simple train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Label encoding
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        
        print(f"Classes: {le.classes_}")
        
        # Split training set for validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
        )
        
        # Sample for tuning (few-shot adaptation)
        X_test_final, X_tune, y_test_final, y_tune = sample_from_large_categories(
            X_test, y_test_encoded, n_samples=1
        )
        
        print(f"Final splits - Train: {X_train_final.shape}, Val: {X_val.shape}, Test: {X_test_final.shape}, Tune: {X_tune.shape}")
        
        # Train source model
        print("Training source model...")
        src_model = RandomForestClassifier(n_estimators=50, random_state=42)
        src_model.fit(X_train_final, y_train_final)
        
        # Create target model with weighted RF
        print("Creating target model...")
        tgt_model = WeightedRandomForest(src_model, n_update=0.8, w_new=0.8, original_ser=False)
        tgt_model.update_forest(X_tune, y_tune)
        
        # Make predictions
        print("Making predictions...")
        y_pred = tgt_model.predict(X_test_final)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_final, y_pred) * 100
        f1 = f1_score(y_test_final, y_pred, average="macro") * 100
        precision = precision_score(y_test_final, y_pred, average="macro") * 100
        recall = recall_score(y_test_final, y_pred, average="macro") * 100
        
        results = {
            "Accuracy": accuracy,
            "F1_macro": f1,
            "Precision": precision,
            "Recall": recall,
            "Number of samples": len(y_test_final)
        }
        
        print("=== REDD Training Results ===")
        for metric, value in results.items():
            print(f"{metric}: {value:.2f}")
        
        # Save results
        import json
        os.makedirs("results/redd/kt_1_ori", exist_ok=True)
        with open("results/redd/kt_1_ori/WTRF_0.json", 'w') as f:
            json.dump({"Target": results}, f, indent=4)
        
        print("\nResults saved to results/redd/kt_1_ori/WTRF_0.json")
        print("=== Training completed successfully! ===")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
