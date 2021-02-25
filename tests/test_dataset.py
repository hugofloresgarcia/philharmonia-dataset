import os
import logging

import numpy as np
from philharmonia_dataset import PhilharmoniaDataset


def test_dataset(root_dir):

    datasets = []
    for classes in (None, 'no-percussion', ['violin', 'cello']):
        datasets.append(PhilharmoniaDataset(
            root=root_dir,
            classes=classes, 
            download=True, 
            sample_rate=48000, 
            seed=0))
    
    for dataset in datasets:
        for idx in range(len(dataset)):
            item = dataset[idx]
            
            assert 'audio' in item
            assert isinstance(item['audio'], np.ndarray)
            assert item['audio'].ndim == 2
            assert 'instrument' in item
            assert item['instrument'] in dataset.classes
