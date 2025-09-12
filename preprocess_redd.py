#!/usr/bin/env python3
"""
Script to preprocess REDD dataset using the existing NILM preprocessing pipeline
"""

import numpy as np
import os
import sys
from sklearn.preprocessing import LabelEncoder

# Add the src directory to the path
sys.path.append('src')
sys.path.append('.')

from dataset.load_redd_data import get_redd_data

def preprocess_and_save_redd():
    """
    Preprocess REDD data and save in the same format as other datasets
    """
    print("=== REDD Dataset Preprocessing ===")

    # Path to REDD data
    redd_path = "src/data/redd/redd.h5"

    if not os.path.exists(redd_path):
        print(f"Error: REDD file not found at {redd_path}")
        return

    # Target appliances (matching the CSV datasets)
    target_appliances = ['dishwasher', 'fridge', 'microwave', 'washingmachine']

    print(f"Preprocessing REDD data for appliances: {target_appliances}")

    try:
        # Preprocess the data
        features, labels, house_labels = get_redd_data(redd_path, target_appliances)

        if len(features) == 0:
            print("No data extracted. The REDD file might have a different structure.")
            return

        print("\nPreprocessing completed:")
        print(f"  - Total samples: {len(features)}")
        print(f"  - Feature dimensions: {features.shape}")
        print(f"  - Unique appliances: {np.unique(labels)}")
        print(f"  - Number of houses: {len(np.unique(house_labels))}")

        # Encode labels
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)

        print(f"  - Encoded labels: {le.classes_}")
        print(f"  - Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        # Save processed data
        output_dir = "src/data/redd"
        os.makedirs(output_dir, exist_ok=True)

        np.save(os.path.join(output_dir, "X.npy"), features)
        np.save(os.path.join(output_dir, "Y.npy"), encoded_labels)
        np.save(os.path.join(output_dir, "house_labels.npy"), house_labels)

        print("\nSaved processed data to:")
        print(f"  - {output_dir}/X.npy: Features {features.shape}")
        print(f"  - {output_dir}/Y.npy: Labels {encoded_labels.shape}")
        print(f"  - {output_dir}/house_labels.npy: House labels {house_labels.shape}")

        # Print sample statistics
        print("\nSample statistics:")
        for i, appliance in enumerate(le.classes_):
            count = np.sum(encoded_labels == i)
            print(f"  - {appliance}: {count} samples")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    preprocess_and_save_redd()
