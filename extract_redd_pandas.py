#!/usr/bin/env python3
"""
Extract REDD data using pandas HDFStore approach and process for the algorithm
Based on the provided REDD dataset extraction code
"""

import pandas as pd
import numpy as np
import os
import sys
import time
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

def load_meter_data_h5(h5_file_path, building, meter):
    """Load data from HDF5 file for specific building and meter using pandas HDFStore"""
    try:
        # Use pandas HDFStore to read the data
        with pd.HDFStore(h5_file_path, 'r') as store:
            key = f'/building{building}/elec/meter{meter}'
            df = store[key]
            
            # The dataframe should have a datetime index and power values
            # Reset index to make timestamp a column, then set it properly
            df = df.reset_index()
            
            # The columns might be named differently - let's be flexible
            if 'power' in df.columns:
                power_col = 'power'
            elif len(df.columns) > 1:
                # Assume the second column is power if no 'power' column
                power_col = df.columns[1]
            else:
                print(f"Warning: Could not identify power column for building{building}/meter{meter}")
                return pd.DataFrame()
            
            # Set up the dataframe with proper column names
            df = df.rename(columns={power_col: 'power'})
            df = df[['index', 'power']].copy()
            df['time'] = pd.to_datetime(df['index'])
            df = df[['time', 'power']].set_index('time')

            # Convert timezone-aware index to timezone-naive for easier comparison
            if df.index.tz is not None:
                df.index = df.index.tz_convert('UTC').tz_localize(None)

            # Flatten MultiIndex columns if they exist
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            return df
            
    except (KeyError, FileNotFoundError) as e:
        print(f"Warning: Could not find data for building{building}/meter{meter}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data for building{building}/meter{meter}: {e}")
        return pd.DataFrame()

def extract_redd_dataset(h5_file_path, dataset_type, building, time_start, time_end, output_dir):
    """Extract REDD dataset for a specific time period and building"""
    
    print(f"=== REDD {dataset_type.capitalize()} Data Extraction ===")
    
    # Define appliance mappings for different buildings
    appliance_mapping = {
        'fridge': {'meter': 5, 'name': 'fridge'},
        'microwave': {'meter': 11, 'name': 'microwave'},
        'dishwasher': {'meter': 6, 'name': 'dishwasher'},
        'washingmachine': {'meter': 20, 'name': 'washingmachine'}
    }
    
    all_features = []
    all_labels = []
    all_house_labels = []

    print(f"{dataset_type.capitalize()} period: {time_start} to {time_end}")
    print(f"Source: REDD House {building} (Building {building})")
    
    for appliance, config in appliance_mapping.items():
        print(f"\nProcessing {appliance}...")
        
        meter = config['meter']
        
        print(f"  Building {building}, Meter {meter}")
        
        try:
            # Load appliance data
            app_df = load_meter_data_h5(h5_file_path, building, meter)
            
            if app_df.empty:
                print(f"    No data found for {appliance} in building {building}")
                continue
            
            print(f"    Loaded {len(app_df)} total records")
            
            # Filter to specific time period with precise timestamps
            start_dt = pd.Timestamp(time_start)
            end_dt = pd.Timestamp(time_end)
            
            filtered_df = app_df[(app_df.index >= start_dt) & (app_df.index <= end_dt)]
            
            if filtered_df.empty:
                print(f"    No data in {dataset_type} period {time_start} to {time_end}")
                continue
            
            print(f"    Filtered to {len(filtered_df)} records in {dataset_type} period")
            
            # Convert power to current (assuming 120V RMS)
            voltage_rms = 120.0
            power_values = filtered_df['power'].values
            current_values = power_values / voltage_rms
            
            # Create segments for feature extraction
            segment_size = 1000  # Samples per segment
            step_size = 250      # 25% overlap for more training data
            max_segments = 200   # Allow more segments for training
            
            segments_processed = 0
            
            for start_idx in range(0, len(current_values) - segment_size, step_size):
                if segments_processed >= max_segments:
                    break
                    
                end_idx = start_idx + segment_size
                current_segment = current_values[start_idx:end_idx]
                
                # Use lower thresholds for appliances with low power consumption
                if appliance == 'dishwasher':
                    threshold = 0.01
                elif appliance == 'fridge':
                    threshold = 0.005  # Very low threshold for fridge
                else:
                    threshold = 0.05
                if np.max(np.abs(current_segment)) > threshold:
                    
                    # Generate synthetic voltage (120V RMS, 60Hz)
                    t = np.linspace(0, len(current_segment)/30000, len(current_segment))
                    voltage_segment = 120 * np.sqrt(2) * np.sin(2 * np.pi * 60 * t)
                    
                    # Extract features
                    try:
                        features = robust_statis_features(voltage_segment, current_segment, 30000)
                        
                        if len(features) > 0:
                            all_features.append(features)
                            all_labels.append(appliance)
                            all_house_labels.append(building - 1)  # Building index
                            segments_processed += 1
                            
                    except Exception as e:
                        print(f"      Error extracting features: {e}")
                        continue
            
            print(f"    Extracted {segments_processed} {dataset_type} segments for {appliance}")
                    
        except Exception as e:
            print(f"    Error processing {appliance} in building {building}: {e}")
            continue
    
    if len(all_features) == 0:
        print("Error: No features could be extracted!")
        return False
    
    # Convert to arrays
    features = np.array(all_features)
    labels = np.array(all_labels)
    house_labels = np.array(all_house_labels)
    
    print(f"\nExtraction Summary:")
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
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "X.npy"), features)
    np.save(os.path.join(output_dir, "Y.npy"), encoded_labels)
    np.save(os.path.join(output_dir, "house_labels.npy"), house_labels)
    
    print(f"\nSaved REDD {dataset_type} data:")
    print(f"  - {output_dir}/X.npy: Features {features.shape}")
    print(f"  - {output_dir}/Y.npy: Labels {encoded_labels.shape}")
    print(f"  - {output_dir}/house_labels.npy: House labels {house_labels.shape}")
    
    print(f"REDD {dataset_type} data extraction completed successfully!")
    
    return True

def extract_redd_algorithm_data(h5_file_path='src/data/redd/redd.h5'):
    """Extract REDD training data from House 3 for the NILM algorithm"""
    
    print("=== REDD Training Data Extraction from House 3 ===")
    start_time = time.time()
    
    # Define appliance mappings for REDD House 1 (Building 1)
    appliance_mapping = {
        'fridge': {
            'building': 1,
            'meter': 5,
            'name': 'fridge'
        },
        'microwave': {
            'building': 1,
            'meter': 11,
            'name': 'microwave'
        },
        'dishwasher': {
            'building': 1,
            'meter': 6,
            'name': 'dishwasher'
        },
        'washingmachine': {
            'building': 1,
            'meter': 20,
            'name': 'washingmachine'
        }
    }
    
    all_features = []
    all_labels = []
    all_house_labels = []
    
    # Specific testing time period for House 1
    train_start = "2011-04-18 09:22:12"
    train_end = "2011-05-23 09:21:51"

    print(f"Training period: {train_start} to {train_end}")
    print(f"Source: REDD House 1 (Building 1)")
    
    for appliance, config in appliance_mapping.items():
        print(f"\nProcessing {appliance}...")
        
        building = config['building']
        meter = config['meter']
        
        print(f"  Building {building}, Meter {meter}")
        
        try:
            # Load appliance data
            app_df = load_meter_data_h5(h5_file_path, building, meter)
            
            if app_df.empty:
                print(f"    No data found for {appliance} in building {building}")
                continue
            
            print(f"    Loaded {len(app_df)} total records")
            
            # Filter to specific training time period with precise timestamps
            start_dt = pd.Timestamp(train_start)
            end_dt = pd.Timestamp(train_end)
            
            filtered_df = app_df[(app_df.index >= start_dt) & (app_df.index <= end_dt)]
            
            if filtered_df.empty:
                print(f"    No data in training period {train_start} to {train_end}")
                continue
            
            print(f"    Filtered to {len(filtered_df)} records in training period")
            
            # Convert power to current (assuming 120V RMS)
            voltage_rms = 120.0
            power_values = filtered_df['power'].values
            current_values = power_values / voltage_rms
            
            # Create segments for feature extraction
            segment_size = 1000  # Samples per segment
            step_size = 250      # 25% overlap for more training data
            max_segments = 200   # Allow more segments for training
            
            segments_processed = 0
            
            for start_idx in range(0, len(current_values) - segment_size, step_size):
                if segments_processed >= max_segments:
                    break
                    
                end_idx = start_idx + segment_size
                current_segment = current_values[start_idx:end_idx]
                
                # Use lower threshold for dishwasher (low frequency data)
                threshold = 0.01 if appliance == 'dishwasher' else 0.05
                if np.max(np.abs(current_segment)) > threshold:
                    
                    # Generate synthetic voltage (120V RMS, 60Hz)
                    t = np.linspace(0, len(current_segment)/30000, len(current_segment))
                    voltage_segment = 120 * np.sqrt(2) * np.sin(2 * np.pi * 60 * t)
                    
                    # Extract features
                    try:
                        features = robust_statis_features(voltage_segment, current_segment, 30000)
                        
                        if len(features) > 0:
                            all_features.append(features)
                            all_labels.append(appliance)
                            all_house_labels.append(0)  # House 1 = index 0
                            segments_processed += 1
                            
                    except Exception as e:
                        print(f"      Error extracting features: {e}")
                        continue
            
            print(f"    Extracted {segments_processed} training segments for {appliance}")
                    
        except Exception as e:
            print(f"    Error processing {appliance} in building {building}: {e}")
            continue
    
    if len(all_features) == 0:
        print("Error: No features could be extracted!")
        return False
    
    # Convert to arrays
    features = np.array(all_features)
    labels = np.array(all_labels)
    house_labels = np.array(all_house_labels)
    
    print(f"\nExtraction Summary:")
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
    output_dir = "src/data/redd/testing_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "X.npy"), features)
    np.save(os.path.join(output_dir, "Y.npy"), encoded_labels)
    np.save(os.path.join(output_dir, "house_labels.npy"), house_labels)
    
    print(f"\nSaved REDD data:")
    print(f"  - {output_dir}/X.npy: Features {features.shape}")
    print(f"  - {output_dir}/Y.npy: Labels {encoded_labels.shape}")
    print(f"  - {output_dir}/house_labels.npy: House labels {house_labels.shape}")
    
    print(f"\nTotal processing time: {(time.time() - start_time) / 60:.2f} minutes")
    print("REDD data extraction completed successfully!")
    
    return True

def extract_all_redd_datasets(h5_file_path='src/data/redd/redd.h5'):
    """Extract all REDD datasets (training, validation, testing)"""
    
    print("=== REDD Complete Dataset Extraction ===")
    start_time = time.time()
    
    datasets = [
        {
            'type': 'training',
            'building': 3,
            'start': '2011-04-21 19:41:24',
            'end': '2011-04-22 19:41:21',
            'output': 'src/data/redd/training_datasets'
        },
        {
            'type': 'validation', 
            'building': 3,
            'start': '2011-05-23 10:31:24',
            'end': '2011-05-24 10:31:21', 
            'output': 'src/data/redd/validation_datasets'
        },
        {
            'type': 'testing',
            'building': 1,
            'start': '2011-04-18 09:22:12',
            'end': '2011-05-23 09:21:51',
            'output': 'src/data/redd/testing_datasets'
        }
    ]
    
    all_success = True
    
    for dataset_config in datasets:
        print(f"\n{'='*60}")
        try:
            success = extract_redd_dataset(
                h5_file_path=h5_file_path,
                dataset_type=dataset_config['type'],
                building=dataset_config['building'],
                time_start=dataset_config['start'],
                time_end=dataset_config['end'],
                output_dir=dataset_config['output']
            )
            if not success:
                all_success = False
        except Exception as e:
            print(f"Error extracting {dataset_config['type']} dataset: {e}")
            all_success = False
    
    print(f"\n{'='*60}")
    print(f"Total processing time: {(time.time() - start_time) / 60:.2f} minutes")
    
    return all_success

def main():
    """Main function"""
    print("=== REDD Data Extraction using Pandas HDFStore ===")
    
    h5_file_path = 'src/data/redd/redd.h5'
    
    if not os.path.exists(h5_file_path):
        print(f"Error: REDD file not found at {h5_file_path}")
        return
    
    success = extract_all_redd_datasets(h5_file_path)
    
    if success:
        print("\n" + "="*60)
        print("SUCCESS: All REDD datasets are now ready for the NILM algorithm!")
        print("  - Training datasets (House 3): src/data/redd/training_datasets/")
        print("  - Validation datasets (House 3): src/data/redd/validation_datasets/")  
        print("  - Testing datasets (House 1): src/data/redd/testing_datasets/")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("FAILED: Could not extract all REDD datasets")
        print("="*60)

if __name__ == "__main__":
    main()
