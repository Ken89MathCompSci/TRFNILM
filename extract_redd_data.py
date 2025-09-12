#!/usr/bin/env python3
"""
Improved REDD data extraction script that handles HDF5 structure properly
"""

import numpy as np
import h5py
import os
import sys
from sklearn.preprocessing import LabelEncoder

# Add the src directory to the path
sys.path.append('src')
sys.path.append('.')

try:
    from data.feature_extract import statis_features
except ImportError:
    print("Warning: Could not import statis_features. Using dummy feature extraction.")
    def statis_features(current, voltage, fs):
        """Dummy feature extraction if the real one is not available"""
        return np.array([
            np.mean(current), np.std(current), np.max(current), np.min(current),
            np.mean(voltage), np.std(voltage), np.max(voltage), np.min(voltage),
            np.mean(current * voltage), np.std(current * voltage)
        ])

def extract_redd_data_robust(filepath, max_samples_per_meter=10000):
    """
    Robust extraction of REDD data from HDF5 file
    
    Args:
        filepath (str): Path to the REDD HDF5 file
        max_samples_per_meter (int): Maximum samples to extract per meter
    
    Returns:
        dict: Dictionary containing extracted data for each house and meter
    """
    data = {}
    
    with h5py.File(filepath, 'r') as f:
        print(f"Root keys: {list(f.keys())}")
        
        # Get all building groups
        buildings = [key for key in f.keys() if key.startswith('building')]
        print(f"Found buildings: {buildings}")
        
        for building in buildings:
            building_group = f[building]
            print(f"\nProcessing {building}")
            
            if 'elec' not in building_group:
                print(f"  No 'elec' group in {building}")
                continue
                
            elec_group = building_group['elec']
            meters = [key for key in elec_group.keys() if key.startswith('meter')]
            print(f"  Found {len(meters)} meters")
            
            building_data = {}
            
            for meter in meters:
                meter_group = elec_group[meter]
                
                if 'table' not in meter_group:
                    print(f"    {meter}: No table found")
                    continue
                
                table = meter_group['table']
                print(f"    {meter}: shape={table.shape}, dtype={table.dtype}")
                
                try:
                    # Try different approaches to read the data
                    meter_data = extract_meter_data(table, max_samples_per_meter)
                    
                    if meter_data is not None and len(meter_data) > 0:
                        building_data[meter] = meter_data
                        print(f"      Successfully extracted {len(meter_data)} samples")
                    else:
                        print(f"      Failed to extract data from {meter}")
                        
                except Exception as e:
                    print(f"      Error extracting {meter}: {e}")
                    continue
            
            if building_data:
                data[building] = building_data
                print(f"  Successfully processed {building} with {len(building_data)} meters")
    
    return data

def extract_meter_data(table, max_samples):
    """
    Extract data from a meter table using different methods
    
    Args:
        table: HDF5 table/dataset
        max_samples (int): Maximum number of samples to extract
    
    Returns:
        np.array: Extracted current values
    """
    try:
        # Method 1: Try direct access to values_block_0
        if hasattr(table, 'dtype') and table.dtype.names:
            if 'values_block_0' in table.dtype.names:
                # Read a subset of the data first
                sample_size = min(max_samples, len(table))
                data_subset = table[:sample_size]
                values = data_subset['values_block_0']
                
                if len(values.shape) > 1 and values.shape[1] >= 1:
                    current = values[:, 0].flatten()
                else:
                    current = values.flatten()
                    
                # Filter out invalid values
                current = current[np.isfinite(current)]
                current = current[np.abs(current) < 1e6]  # Remove extreme outliers
                
                return current
    except Exception as e:
        print(f"        Method 1 failed: {e}")
    
    try:
        # Method 2: Try reading as regular array
        sample_size = min(max_samples, len(table))
        raw_data = table[:sample_size]
        
        if isinstance(raw_data, np.ndarray):
            if len(raw_data.shape) > 1:
                current = raw_data[:, 0] if raw_data.shape[1] > 0 else raw_data.flatten()
            else:
                current = raw_data.flatten()
                
            # Filter out invalid values
            current = current[np.isfinite(current)]
            current = current[np.abs(current) < 1e6]  # Remove extreme outliers
            
            return current
    except Exception as e:
        print(f"        Method 2 failed: {e}")
    
    try:
        # Method 3: Try chunk-by-chunk reading
        chunk_size = min(1000, max_samples)
        all_data = []
        
        for i in range(0, min(len(table), max_samples), chunk_size):
            end_idx = min(i + chunk_size, len(table), max_samples)
            chunk = table[i:end_idx]
            
            if hasattr(chunk, 'dtype') and chunk.dtype.names and 'values_block_0' in chunk.dtype.names:
                chunk_values = chunk['values_block_0']
                if len(chunk_values.shape) > 1:
                    all_data.extend(chunk_values[:, 0].flatten())
                else:
                    all_data.extend(chunk_values.flatten())
            else:
                if len(chunk.shape) > 1:
                    all_data.extend(chunk[:, 0].flatten())
                else:
                    all_data.extend(chunk.flatten())
        
        if all_data:
            current = np.array(all_data)
            current = current[np.isfinite(current)]
            current = current[np.abs(current) < 1e6]  # Remove extreme outliers
            return current
            
    except Exception as e:
        print(f"        Method 3 failed: {e}")
    
    return None

def process_extracted_data_for_algorithm(extracted_data, target_appliances=None):
    """
    Process the extracted raw data into the format needed by the algorithm
    
    Args:
        extracted_data (dict): Raw extracted data
        target_appliances (list): Target appliance names (optional)
    
    Returns:
        tuple: (features, labels, house_labels)
    """
    if target_appliances is None:
        target_appliances = ['dishwasher', 'fridge', 'microwave', 'washingmachine']
    
    # REDD meter mapping (approximate - may need adjustment based on actual data)
    meter_to_appliance = {
        'meter3': 'fridge',
        'meter4': 'dishwasher', 
        'meter5': 'microwave',
        'meter6': 'washingmachine',
        'meter7': 'fridge',
        'meter8': 'microwave',
        'meter9': 'dishwasher',
        'meter10': 'washingmachine',
        'meter11': 'microwave',
        'meter12': 'fridge',
        'meter13': 'dishwasher',
        'meter14': 'fridge',
        'meter15': 'microwave',
        'meter16': 'dishwasher',
        'meter17': 'washingmachine'
    }
    
    all_features = []
    all_labels = []
    all_house_labels = []
    
    house_counter = 0
    
    for building_name, building_data in extracted_data.items():
        print(f"Processing {building_name} for algorithm...")
        
        for meter_name, current_data in building_data.items():
            # Map meter to appliance
            appliance = meter_to_appliance.get(meter_name)
            if appliance and appliance in target_appliances and len(current_data) > 0:
                
                # Create segments for processing
                segment_size = 500  # ~16ms at 30kHz sampling rate
                step_size = 250     # 50% overlap
                
                segments_processed = 0
                max_segments = 20   # Limit segments per meter
                
                for start_idx in range(0, len(current_data) - segment_size, step_size):
                    if segments_processed >= max_segments:
                        break
                        
                    end_idx = start_idx + segment_size
                    segment = current_data[start_idx:end_idx]
                    
                    # Only process segments with significant activity
                    if np.max(np.abs(segment)) > 0.1:  # Threshold for activity
                        
                        # Create dummy voltage (constant 120V)
                        voltage_segment = np.ones_like(segment) * 120.0
                        
                        # Extract features
                        try:
                            features = statis_features(segment, voltage_segment, 30000)  # 30kHz sampling
                            
                            all_features.append(features)
                            all_labels.append(appliance)
                            all_house_labels.append(house_counter)
                            
                            segments_processed += 1
                            
                        except Exception as e:
                            print(f"      Error extracting features: {e}")
                            continue
                
                print(f"    {meter_name} -> {appliance}: {segments_processed} segments")
        
        house_counter += 1
    
    if len(all_features) == 0:
        print("Warning: No features extracted!")
        return np.array([]), np.array([]), np.array([])
    
    return np.array(all_features), np.array(all_labels), np.array(all_house_labels)

def main():
    """Main function to extract and process REDD data"""
    print("=== Enhanced REDD Data Extraction ===")
    
    redd_path = "src/data/redd/redd.h5"
    
    if not os.path.exists(redd_path):
        print(f"Error: REDD file not found at {redd_path}")
        return
    
    # Extract raw data
    print("Step 1: Extracting raw data from REDD.h5...")
    extracted_data = extract_redd_data_robust(redd_path, max_samples_per_meter=50000)
    
    if not extracted_data:
        print("Error: No data could be extracted from the REDD file")
        return
    
    print(f"Successfully extracted data from {len(extracted_data)} buildings")
    for building, meters in extracted_data.items():
        print(f"  {building}: {len(meters)} meters")
    
    # Process for algorithm
    print("\nStep 2: Processing data for algorithm...")
    target_appliances = ['dishwasher', 'fridge', 'microwave', 'washingmachine']
    features, labels, house_labels = process_extracted_data_for_algorithm(extracted_data, target_appliances)
    
    if len(features) == 0:
        print("Error: No processed data for algorithm")
        return
    
    print("Data processing completed:")
    print(f"  - Total samples: {len(features)}")
    print(f"  - Feature dimensions: {features.shape}")
    print(f"  - Unique appliances: {np.unique(labels)}")
    print(f"  - Number of houses: {len(np.unique(house_labels))}")
    
    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    print(f"  - Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Save processed data
    output_dir = "src/data/redd"
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "X.npy"), features)
    np.save(os.path.join(output_dir, "Y.npy"), encoded_labels)
    np.save(os.path.join(output_dir, "house_labels.npy"), house_labels)
    
    print(f"\nStep 3: Saved processed data:")
    print(f"  - {output_dir}/X.npy: Features {features.shape}")
    print(f"  - {output_dir}/Y.npy: Labels {encoded_labels.shape}")
    print(f"  - {output_dir}/house_labels.npy: House labels {house_labels.shape}")
    
    # Print sample statistics
    print("\nSample statistics:")
    for i, appliance in enumerate(le.classes_):
        count = np.sum(encoded_labels == i)
        print(f"  - {appliance}: {count} samples")

if __name__ == "__main__":
    main()
