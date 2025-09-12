import numpy as np
import h5py
import os
import sys
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from dataset.transform import *
from data.feature_extract import statis_features

def load_redd_h5_data(filepath):
    """
    Load REDD data from HDF5 file

    Args:
        filepath (str): Path to the REDD HDF5 file

    Returns:
        dict: Dictionary containing data for each house
    """
    data = {}

    with h5py.File(filepath, 'r') as f:
        # Get all house groups
        houses = [key for key in f.keys() if key.startswith('building')]
        print(f"Found houses: {houses}")

        for house in houses:
            house_data = {}
            house_group = f[house]
            print(f"Processing {house}, keys: {list(house_group.keys())}")

            # REDD data is organized as building/elec/meterX
            if 'elec' in house_group:
                elec_group = house_group['elec']
                print(f"  Elec group keys: {list(elec_group.keys())}")

                # Get all meters
                meters = [key for key in elec_group.keys() if key.startswith('meter')]
                print(f"  Found meters: {meters}")

                for meter in meters:
                    meter_data = {}
                    meter_group = elec_group[meter]
                    print(f"    {meter} keys: {list(meter_group.keys()) if hasattr(meter_group, 'keys') else 'No keys'}")

                    # Check if this meter has data
                    if hasattr(meter_group, 'keys') and 'table' in meter_group:
                        table = meter_group['table']
                        print(f"      {meter} table shape: {table.shape}")
                        print(f"      {meter} table dtype: {table.dtype}")

                        # Try different approaches to access the data
                        try:
                            # First try: direct access to structured array
                            if hasattr(table, 'dtype') and table.dtype.names:
                                print(f"      {meter} has structured array with fields: {table.dtype.names}")
                                if 'values_block_0' in table.dtype.names:
                                    # Try reading a small sample first
                                    sample_size = min(100, len(table))
                                    sample_data = table[:sample_size]
                                    print(f"      Sample data shape: {sample_data.shape}")
                                    print(f"      Sample values_block_0 shape: {sample_data['values_block_0'].shape}")

                                    # Extract the full data
                                    values = table['values_block_0']
                                    if values.shape[1] >= 1:
                                        meter_data['current'] = values[:, 0].flatten()
                                        print(f"      {meter} current shape: {meter_data['current'].shape}")
                                    if values.shape[1] >= 2:
                                        meter_data['voltage'] = values[:, 1].flatten()
                                        print(f"      {meter} voltage shape: {meter_data['voltage'].shape}")
                        except Exception as e:
                            print(f"      Error accessing {meter} structured data: {e}")
                            try:
                                # Second try: read as regular array with different chunking
                                print(f"      Trying alternative access for {meter}")
                                # Try reading in chunks to avoid memory issues
                                chunk_size = 10000
                                all_data = []
                                for i in range(0, len(table), chunk_size):
                                    end_idx = min(i + chunk_size, len(table))
                                    chunk = table[i:end_idx]
                                    if hasattr(chunk, 'dtype') and chunk.dtype.names and 'values_block_0' in chunk.dtype.names:
                                        all_data.append(chunk['values_block_0'][:, 0])
                                    else:
                                        all_data.append(np.array(chunk).flatten())

                                if all_data:
                                    meter_data['current'] = np.concatenate(all_data)
                                    print(f"      {meter} current shape (chunked): {meter_data['current'].shape}")
                            except Exception as e2:
                                print(f"      Failed chunked access for {meter}: {e2}")
                                try:
                                    # Third try: use HDF5 dataset properties
                                    print(f"      Trying HDF5 properties for {meter}")
                                    if hasattr(table, 'chunks') and table.chunks:
                                        print(f"      {meter} is chunked: {table.chunks}")
                                    if hasattr(table, 'compression'):
                                        print(f"      {meter} compression: {table.compression}")

                                    # Try reading with different options
                                    data = table[...]
                                    print(f"      Raw data shape: {data.shape}, dtype: {data.dtype}")
                                    meter_data['current'] = data.flatten() if len(data.shape) == 1 else data[:, 0]
                                    print(f"      {meter} current shape (raw): {meter_data['current'].shape}")
                                except Exception as e3:
                                    print(f"      All access methods failed for {meter}: {e3}")

                    if meter_data:
                        house_data[meter] = meter_data
                        print(f"    Added {meter} to house_data")

            if house_data:
                data[house] = house_data
                print(f"Added {house} to data with {len(house_data)} meters")

    print(f"Final data keys: {list(data.keys())}")
    return data

def extract_appliance_events(house_data, appliance_name, mains_voltage=None, window_size=10000, sampling_rate=30000):
    """
    Extract steady-state segments for a specific appliance

    Args:
        house_data (dict): Data for a specific house
        appliance_name (str): Name of the appliance
        mains_voltage (np.array): Mains voltage for alignment (optional)
        window_size (int): Size of the analysis window
        sampling_rate (int): Sampling rate in Hz

    Returns:
        list: List of current segments for the appliance
    """
    if appliance_name not in house_data:
        return []

    appliance_data = house_data[appliance_name]
    current = appliance_data.get('current', None)

    if current is None:
        return []

    # If we have mains voltage, use it for alignment
    if mains_voltage is None and 'mains' in house_data:
        mains_voltage = house_data['mains'].get('voltage', None)

    segments = []

    # Simple approach: divide into windows and extract segments
    # In practice, you'd want to detect on/off events
    step_size = window_size // 2  # 50% overlap

    for start_idx in range(0, len(current) - window_size, step_size):
        end_idx = start_idx + window_size
        segment = current[start_idx:end_idx]

        # Basic filtering: only keep segments with significant current
        if np.max(np.abs(segment)) > 0.1:  # Threshold to filter noise
            segments.append(segment)

    return segments

def get_trajectory_redd(current_segments, voltage_segments=None, fs=30000, f0=60):
    """
    Extract representative waveforms from REDD segments

    Args:
        current_segments (list): List of current segments
        voltage_segments (list): List of voltage segments (optional)
        fs (int): Sampling frequency
        f0 (int): Mains frequency

    Returns:
        tuple: (current_waveforms, voltage_waveforms)
    """
    if not current_segments:
        return np.array([]), np.array([])

    NS = int(fs / f0)  # Samples per period
    NP = 20  # Number of periods to average

    n_segments = len(current_segments)
    I = np.empty([n_segments, NS])
    V = np.empty([n_segments, NS]) if voltage_segments else None

    for i, current_seg in enumerate(current_segments):
        # Reshape into periods
        npts = len(current_seg)
        if npts < NS * NP:
            continue

        tempI = np.sum(np.reshape(current_seg[-NS*NP:], [NP, NS]), 0) / NP

        # Align current
        ix = np.argsort(np.abs(tempI))
        j = 0
        while j < len(ix) - 1:
            if ix[j] < NS - 1 and tempI[ix[j] + 1] > tempI[ix[j]]:
                real_ix = ix[j]
                break
            j += 1

        I[i] = np.hstack([tempI[real_ix:], tempI[:real_ix]])

        # Process voltage if available
        if voltage_segments and i < len(voltage_segments):
            voltage_seg = voltage_segments[i]
            if len(voltage_seg) >= NS * NP:
                tempV = np.sum(np.reshape(voltage_seg[-NS*NP:], [NP, NS]), 0) / NP
                V[i] = np.hstack([tempV[real_ix:], tempV[:real_ix]])

    if V is not None:
        return I, V
    else:
        return I, np.array([])

def preprocess_redd_data(filepath, target_appliances=None, sampling_rate=30000, window_size=10000):
    """
    Main function to preprocess REDD data

    Args:
        filepath (str): Path to REDD HDF5 file
        target_appliances (list): List of appliances to extract (optional)
        sampling_rate (int): Sampling rate
        window_size (int): Analysis window size

    Returns:
        tuple: (features, labels, house_labels)
    """
    print("Loading REDD data...")
    data = load_redd_h5_data(filepath)

    all_features = []
    all_labels = []
    all_house_labels = []

    # Default appliances if none specified
    if target_appliances is None:
        target_appliances = ['dishwasher', 'fridge', 'microwave', 'washingmachine']

    house_counter = 0

    for house_name, house_data in data.items():
        print(f"Processing {house_name}...")

        # Get mains voltage for alignment
        mains_voltage = None
        if 'mains' in house_data and 'voltage' in house_data['mains']:
            mains_voltage = house_data['mains']['voltage']

        for appliance in target_appliances:
            if appliance in house_data:
                print(f"  Extracting {appliance}...")

                # Extract current segments
                current_segments = extract_appliance_events(
                    house_data, appliance, mains_voltage, window_size, sampling_rate
                )

                # Extract voltage segments if available
                voltage_segments = None
                if 'voltage' in house_data[appliance]:
                    voltage_segments = [house_data[appliance]['voltage'][i:i+window_size]
                                      for i in range(0, len(house_data[appliance]['voltage']) - window_size, window_size//2)]

                if current_segments:
                    # Get representative waveforms
                    I_rep, V_rep = get_trajectory_redd(current_segments, voltage_segments, sampling_rate)

                    if len(I_rep) > 0:
                        # Extract features for each representative waveform
                        for i in range(len(I_rep)):
                            if V_rep.size > 0 and i < len(V_rep):
                                # Use both current and voltage
                                features = statis_features(I_rep[i], V_rep[i], sampling_rate)
                            else:
                                # Use current only (assume constant voltage)
                                constant_voltage = np.ones_like(I_rep[i]) * 120  # Assume 120V
                                features = statis_features(I_rep[i], constant_voltage, sampling_rate)

                            all_features.append(features)
                            all_labels.append(appliance)
                            all_house_labels.append(house_counter)

        house_counter += 1

    return np.array(all_features), np.array(all_labels), np.array(all_house_labels)

def get_redd_data(filepath, target_appliances=None):
    """
    Convenience function to get preprocessed REDD data

    Args:
        filepath (str): Path to REDD HDF5 file
        target_appliances (list): List of appliances to extract

    Returns:
        tuple: (features, labels, house_labels)
    """
    return preprocess_redd_data(filepath, target_appliances)

if __name__ == "__main__":
    # Example usage
    filepath = "../../data/redd/redd.h5"
    features, labels, house_labels = get_redd_data(filepath)

    print(f"Extracted {len(features)} samples")
    print(f"Unique appliances: {np.unique(labels)}")
    print(f"Number of houses: {len(np.unique(house_labels))}")
    print(f"Feature dimensions: {features.shape}")
