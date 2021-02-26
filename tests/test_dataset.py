import os
import logging

import numpy as np
from philharmonia_dataset import PhilharmoniaDataset


def _validate_dataset_entry(item, dataset):
    assert 'audio' in item
    assert isinstance(item['audio'], np.ndarray)
    assert item['audio'].ndim == 2

    assert 'instrument' in item
    assert item['instrument'] in dataset.classes

    assert isinstance(item['pitch'], str)

    assert isinstance(item['one_hot'], np.ndarray)
    assert len(item['one_hot']) == len(dataset.classes)

    assert isinstance(item['articulation'], str)

    assert isinstance(item['dynamic'], str)

def test_dataset(root_dir):
    datasets = []
    for classes in (None, 'no-percussion', ['violin', 'cello']):
        datasets.append(PhilharmoniaDataset(
            root=root_dir,
            classes=classes, 
            download=True, 
            sample_rate=48000, 
            seed=0))
    
    dataset = datasets[0]
    for idx in range(len(dataset)):
        item = dataset[idx]

        _validate_dataset_entry(item, dataset)
