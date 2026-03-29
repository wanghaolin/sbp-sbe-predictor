"""Data loading module"""

import os
from typing import Dict

import pandas as pd


def load_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load data files
    
    Args:
        data_dir: Path to data directory
    
    Returns:
        Dictionary containing train, internal_test, and external_test dataframes
    """
    data_files = {
        'train': 'train.csv',
        'internal_test': 'internal_test.csv',
        'external_test': 'external_test.csv'
    }
    
    data = {}
    for key, file_name in data_files.items():
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            data[key] = pd.read_csv(file_path)
        else:
            raise FileNotFoundError(f"File not found: {file_path}")
    
    return data
