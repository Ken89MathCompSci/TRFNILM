import os
import numpy as np
import pandas as pd

APPLIANCES = ["dish_washer", "refrigerator", "microwave", "washer_dryer"]
PREPROCESS_ROOT = "preprocess"

def preprocess_appliance(appliance):
    folder = os.path.join(PREPROCESS_ROOT, appliance)
    if not os.path.exists(folder):
        print(f"Folder not found for {appliance}, skipping.")
        return

    # Find all CSV files in the appliance folder
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not csv_files:
        print(f"No CSV files found for {appliance}, skipping.")
        return

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(folder, csv_file), na_values=[''])
        df = df.dropna(subset=["power"])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Convert timestamp to seconds since epoch
        df['timestamp_seconds'] = df['timestamp'].astype(np.int64) // 10**9
        dfs.append(df)

    # Concatenate all dataframes
    if not dfs:
        print(f"No valid data for {appliance}, skipping.")
        return
    df_all = pd.concat(dfs, ignore_index=True)

    # X: timestamp as seconds since epoch, shape (n_samples, 1)
    X = df_all["timestamp_seconds"].values.reshape(-1, 1)
    # Y: power readings, shape (n_samples,)
    Y = df_all["power"].values.astype(float)

    np.save(os.path.join(folder, "X.npy"), X)
    np.save(os.path.join(folder, "Y.npy"), Y)
    print(f"Saved {appliance} to .npy.")

def main():
    for appliance in APPLIANCES:
        preprocess_appliance(appliance)

if __name__ == "__main__":
    main()