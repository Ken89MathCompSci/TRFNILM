import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error
from SER.WeightedRF import WeightedRandomForest
import joblib  # Add this import

APPLIANCES = ["dish_washer", "refrigerator", "microwave", "washer_dryer"]
PREPROCESS_ROOT = "preprocess"

def train_trfnilm(X, y, n_trees=10, k_target=1):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    unique, counts = np.unique(y_enc, return_counts=True)
    if np.any(counts < 2):
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc)
    idx = np.random.choice(len(X_test), min(k_target, len(X_test)), replace=False)
    X_tune, y_tune = X_test[idx], y_test[idx]
    X_test_final = np.delete(X_test, idx, axis=0)
    y_test_final = np.delete(y_test, idx, axis=0)

    src_model = RandomForestClassifier(n_estimators=n_trees)
    src_model.fit(X_train, y_train)
    tgt_model = WeightedRandomForest(src_model, n_update=0.8, w_new=0.8, original_ser=False)
    tgt_model.update_forest(X_tune, y_tune)
    y_pred = tgt_model.predict(X_test_final)

    return {
        "accuracy": accuracy_score(y_test_final, y_pred),
        "f1_macro": f1_score(y_test_final, y_pred, average="macro"),
        "precision": precision_score(y_test_final, y_pred, average="macro"),
        "recall": recall_score(y_test_final, y_pred, average="macro"),
        "mae": mean_absolute_error(y_test_final, y_pred),
        "model": tgt_model  # Return the trained model
    }

def main():
    for appliance in APPLIANCES:
        print(f"Training TRFNILM for {appliance}...")
        X_path = os.path.join(PREPROCESS_ROOT, appliance, "X.npy")
        Y_path = os.path.join(PREPROCESS_ROOT, appliance, "Y.npy")
        if not os.path.exists(X_path) or not os.path.exists(Y_path):
            print(f"Missing data for {appliance}, skipping.")
            continue
        X = np.load(X_path)
        y = np.load(Y_path)
        results = train_trfnilm(X, y)
        print(f"Results for {appliance}: {{k: v for k, v in results.items() if k != 'model'}}")
        # Save the trained model
        model_path = os.path.join(PREPROCESS_ROOT, appliance, "trfnilm_model.joblib")
        joblib.dump(results["model"], model_path)
        print(f"Saved model to {model_path}")

if __name__ == "__main__":
    main()