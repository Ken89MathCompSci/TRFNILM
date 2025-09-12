#!/usr/bin/env python3
"""
Load REDD data using NILMTK and extract it in the format needed by the algorithm
"""

from nilmtk.dataset import DataSet
import warnings
import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import LabelEncoder

# Add the src directory to the path
sys.path.append('src')
sys.path.append('.')

try:
    from data.feature_extract import statis_features
    REAL_FEATURES = True
except ImportError:
    print("Warning: Could not import statis_features. Using dummy feature extraction.")
    REAL_FEATURES = False
    def statis_features(voltage, current, fs):
        """Dummy feature extraction if the real one is not available"""
        return np.array([
            np.mean(current), np.std(current), np.max(current), np.min(current),
            np.mean(voltage), np.std(voltage), np.max(voltage), np.min(voltage),
            np.mean(current * voltage), np.std(current * voltage)
        ])

warnings.filterwarnings("ignore")

def robust_statis_features(voltage_sig, current_sig, fs):
    """Robust wrapper for statis_features that handles errors gracefully"""
    try:
        if REAL_FEATURES:
            # Real implementation expects voltage, current, fs
            features = statis_features(voltage_sig, current_sig, fs)
            return np.array(features)
        else:
            # Dummy implementation
            return statis_features(voltage_sig, current_sig, fs)
    except Exception as e:
        print(f"    Feature extraction failed: {e}, using basic features")
        # Fallback to basic statistical features
        return np.array([
            np.mean(current_sig), np.std(current_sig), np.max(current_sig), np.min(current_sig),
            np.mean(voltage_sig), np.std(voltage_sig), np.max(voltage_sig), np.min(voltage_sig),
            np.mean(current_sig * voltage_sig), np.std(current_sig * voltage_sig)
        ])

def load_redd_raw_data():
    """Load and display raw data samples from REDD dataset"""
    try:
        # Load the REDD dataset from the src/data/redd directory
        print("Loading REDD dataset...")
        dataset = DataSet('src/data/redd/redd.h5')
        
        # Get building 1
        building = dataset.buildings[1]
        
        # Set a small time window to get some sample data
        print("Setting time window for data loading...")
        dataset.set_window(start='2011-04-17', end='2011-04-18')  # Just one day
        
        print("\n" + "="*80)
        print("REDD DATASET RAW DATA SAMPLES")
        print("="*80)
        
        # Load mains data
        print("\n1. MAINS POWER DATA:")
        print("-" * 40)
        mains = building.elec.mains()
        try:
            mains_data = next(mains.load(physical_quantity='power', ac_type='active', sample_period=60))
            print(f"Shape: {mains_data.shape}")
            print(f"Columns: {mains_data.columns.tolist()}")
            print(f"Index range: {mains_data.index[0]} to {mains_data.index[-1]}")
            print("\nFirst 10 samples:")
            print(mains_data.head(10))
            print(f"\nBasic statistics:")
            print(mains_data.describe())
        except Exception as e:
            print(f"Error loading mains data: {e}")
        
        # Load fridge data
        print("\n\n2. FRIDGE POWER DATA:")
        print("-" * 40)
        fridge = building.elec['fridge']
        try:
            fridge_data = next(fridge.load(physical_quantity='power', ac_type='active', sample_period=60))
            print(f"Shape: {fridge_data.shape}")
            print(f"Columns: {fridge_data.columns.tolist()}")
            print(f"Index range: {fridge_data.index[0]} to {fridge_data.index[-1]}")
            print("\nFirst 10 samples:")
            print(fridge_data.head(10))
            print(f"\nBasic statistics:")
            print(fridge_data.describe())
        except Exception as e:
            print(f"Error loading fridge data: {e}")
        
        # Load microwave data
        print("\n\n3. MICROWAVE POWER DATA:")
        print("-" * 40)
        microwave = building.elec['microwave']
        try:
            microwave_data = next(microwave.load(physical_quantity='power', ac_type='active', sample_period=60))
            print(f"Shape: {microwave_data.shape}")
            print(f"Columns: {microwave_data.columns.tolist()}")
            print(f"Index range: {microwave_data.index[0]} to {microwave_data.index[-1]}")
            print("\nFirst 10 samples:")
            print(microwave_data.head(10))
            print(f"\nBasic statistics:")
            print(microwave_data.describe())
        except Exception as e:
            print(f"Error loading microwave data: {e}")
        
        # Load dishwasher data
        print("\n\n4. DISHWASHER POWER DATA:")
        print("-" * 40)
        dishwasher = building.elec['dish washer']
        try:
            dishwasher_data = next(dishwasher.load(physical_quantity='power', ac_type='active', sample_period=60))
            print(f"Shape: {dishwasher_data.shape}")
            print(f"Columns: {dishwasher_data.columns.tolist()}")
            print(f"Index range: {dishwasher_data.index[0]} to {dishwasher_data.index[-1]}")
            print("\nFirst 10 samples:")
            print(dishwasher_data.head(10))
            print(f"\nBasic statistics:")
            print(dishwasher_data.describe())
        except Exception as e:
            print(f"Error loading dishwasher data: {e}")
        
        print("\n" + "="*80)
        print("DATA LOADING COMPLETED")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

def extract_redd_data_nilmtk():
    """Extract REDD data using NILMTK and process for the algorithm"""
    print("\n" + "="*80)
    print("EXTRACTING REDD DATA FOR ALGORITHM")
    print("="*80)
    
    try:
        # Load the REDD dataset
        print("Loading REDD dataset...")
        dataset = DataSet('src/data/redd/redd.h5')
        
        # Target appliances matching the algorithm requirements
        target_appliances = {
            'fridge': 'fridge',
            'microwave': 'microwave', 
            'dish washer': 'dishwasher',
            'washing machine': 'washingmachine'
        }
        
        all_features = []
        all_labels = []
        all_house_labels = []
        
        # Process multiple buildings
        for building_id in [1, 2, 3]:  # Process buildings 1, 2, 3
            try:
                print(f"\nProcessing building {building_id}...")
                building = dataset.buildings[building_id]
                
                # Set a time window with data
                dataset.set_window(start='2011-04-17', end='2011-04-20')  # 3 days
                
                for nilmtk_name, algo_name in target_appliances.items():
                    try:
                        print(f"  Processing {nilmtk_name} -> {algo_name}")
                        
                        # Get the appliance
                        appliance = building.elec[nilmtk_name]
                        
                        # Load power data (active power)
                        power_data = next(appliance.load(physical_quantity='power', ac_type='active', sample_period=6))
                        
                        if power_data.empty:
                            print(f"    No data for {nilmtk_name}")
                            continue
                            
                        print(f"    Loaded {len(power_data)} samples")
                        
                        # Convert power to current (assuming 120V RMS)
                        voltage_rms = 120.0
                        current_rms = power_data.values.flatten() / voltage_rms
                        
                        # Create segments for feature extraction
                        segment_size = 1000  # Samples per segment
                        step_size = 500      # 50% overlap
                        
                        segments_processed = 0
                        max_segments = 50    # Limit segments per appliance per building
                        
                        for start_idx in range(0, len(current_rms) - segment_size, step_size):
                            if segments_processed >= max_segments:
                                break
                                
                            end_idx = start_idx + segment_size
                            current_segment = current_rms[start_idx:end_idx]
                            
                            # Only process segments with significant current (> 0.1A)
                            if np.max(np.abs(current_segment)) > 0.1:
                                
                                # Generate synthetic voltage (120V RMS, 60Hz)
                                t = np.linspace(0, len(current_segment)/30000, len(current_segment))
                                voltage_segment = 120 * np.sqrt(2) * np.sin(2 * np.pi * 60 * t)
                                
                                # Extract features
                                try:
                                    features = robust_statis_features(voltage_segment, current_segment, 30000)
                                    
                                    if len(features) > 0:
                                        all_features.append(features)
                                        all_labels.append(algo_name)
                                        all_house_labels.append(building_id - 1)  # 0-indexed
                                        segments_processed += 1
                                        
                                except Exception as e:
                                    print(f"      Error extracting features: {e}")
                                    continue
                        
                        print(f"    Extracted {segments_processed} segments")
                        
                    except Exception as e:
                        print(f"    Error processing {nilmtk_name}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error processing building {building_id}: {e}")
                continue
        
        if len(all_features) == 0:
            print("Error: No features could be extracted!")
            return False
        
        # Convert to arrays
        features = np.array(all_features)
        labels = np.array(all_labels)
        house_labels = np.array(all_house_labels)
        
        print(f"\nExtracted REDD data:")
        print(f"  - Total samples: {len(features)}")
        print(f"  - Feature dimensions: {features.shape}")
        print(f"  - Unique appliances: {np.unique(labels)}")
        print(f"  - Number of houses: {len(np.unique(house_labels))}")
        
        # Encode labels
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        
        print(f"\nLabel encoding:")
        for i, appliance in enumerate(le.classes_):
            count = np.sum(encoded_labels == i)
            print(f"  - {appliance} (label {i}): {count} samples")
        
        # Save processed data
        output_dir = "src/data/redd"
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, "X.npy"), features)
        np.save(os.path.join(output_dir, "Y.npy"), encoded_labels)
        np.save(os.path.join(output_dir, "house_labels.npy"), house_labels)
        
        print(f"\nSaved REDD data:")
        print(f"  - {output_dir}/X.npy: Features {features.shape}")
        print(f"  - {output_dir}/Y.npy: Labels {encoded_labels.shape}")
        print(f"  - {output_dir}/house_labels.npy: House labels {house_labels.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error extracting REDD data: {e}")
        return False

def main():
    """Main function"""
    print("=== REDD Data Loading with NILMTK ===")
    
    # First, try to load and display raw data
    if load_redd_raw_data():
        print("\n" + "="*80)
        
        # Then extract and process for the algorithm
        if extract_redd_data_nilmtk():
            print("\nREDD data extraction completed successfully!")
            print("Data is now available for the NILM algorithm.")
        else:
            print("\nREDD data extraction failed!")
    else:
        print("\nFailed to load raw REDD data!")

if __name__ == "__main__":
    main()
