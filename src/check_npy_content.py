
import os
import numpy as np

APPLIANCES = ["dish_washer", "refrigerator", "microwave", "washer_dryer"]
PREPROCESS_ROOT = "preprocess"

def check_npy(appliance):
    folder = os.path.join(PREPROCESS_ROOT, appliance)
    x_path = os.path.join(folder, "X.npy")
    y_path = os.path.join(folder, "Y.npy")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"{appliance}: Missing X.npy or Y.npy")
        return
    X = np.load(x_path)
    Y = np.load(y_path)
    print(f"{appliance}:")
    print(f"  X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"  First 5 X: {X[:5].flatten()}")
    print(f"  First 5 Y: {Y[:5]}")
    print("")

def main():
    for appliance in APPLIANCES:
        check_npy(appliance)

if __name__ == "__main__":
    main()