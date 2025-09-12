import os
import numpy as np

print('Checking extraction status...')
testing_dir = 'src/data/redd/testing_datasets'
print(f'Testing datasets dir exists: {os.path.exists(testing_dir)}')

if os.path.exists(testing_dir):
    files = os.listdir(testing_dir)
    print(f'Files: {files}')
    
    if 'X.npy' in files:
        X = np.load(f'{testing_dir}/X.npy')
        print(f'X.npy shape: {X.shape}')
        
    if 'Y.npy' in files:
        Y = np.load(f'{testing_dir}/Y.npy')
        print(f'Y.npy shape: {Y.shape}')
        print(f'Unique labels: {np.unique(Y)}')
        
    if 'house_labels.npy' in files:
        house_labels = np.load(f'{testing_dir}/house_labels.npy')
        print(f'House labels shape: {house_labels.shape}')
        print(f'Unique houses: {np.unique(house_labels)}')
