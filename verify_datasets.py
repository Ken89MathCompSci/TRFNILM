import os
import numpy as np

def verify_dataset(dataset_name, dataset_dir):
    """Verify a dataset directory and its contents"""
    print(f"\n=== {dataset_name.upper()} DATASET ===")
    print(f"Directory: {dataset_dir}")
    
    if not os.path.exists(dataset_dir):
        print("❌ Dataset directory does not exist!")
        return False
    
    files = os.listdir(dataset_dir)
    print(f"Files: {files}")
    
    required_files = ['X.npy', 'Y.npy', 'house_labels.npy']
    missing_files = [f for f in required_files if f not in files]
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    # Load and verify data
    X = np.load(os.path.join(dataset_dir, 'X.npy'))
    Y = np.load(os.path.join(dataset_dir, 'Y.npy'))
    house_labels = np.load(os.path.join(dataset_dir, 'house_labels.npy'))
    
    print(f"✅ Features (X): {X.shape}")
    print(f"✅ Labels (Y): {Y.shape}")
    print(f"✅ House labels: {house_labels.shape}")
    print(f"✅ Unique labels: {np.unique(Y)}")
    print(f"✅ Unique houses: {np.unique(house_labels)}")
    print(f"✅ Label distribution: {np.bincount(Y)}")
    
    return True

def main():
    print("=== REDD DATASET VERIFICATION ===")
    
    datasets = {
        'training': 'src/data/redd/training_datasets',
        'validation': 'src/data/redd/validation_datasets', 
        'testing': 'src/data/redd/testing_datasets'
    }
    
    all_valid = True
    
    for dataset_name, dataset_dir in datasets.items():
        valid = verify_dataset(dataset_name, dataset_dir)
        if not valid:
            all_valid = False
    
    print(f"\n{'='*50}")
    if all_valid:
        print("✅ ALL DATASETS VERIFIED SUCCESSFULLY!")
        print("The REDD data extraction is complete and ready for the algorithm.")
    else:
        print("❌ SOME DATASETS FAILED VERIFICATION!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
