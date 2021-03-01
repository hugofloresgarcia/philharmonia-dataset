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
    dataset = PhilharmoniaDataset(
        root=root_dir,
        download=True, 
        sample_rate=48000, 
        seed=0)
    
    for idx in range(0, len(dataset), 2):
        item = dataset[idx]
        _validate_dataset_entry(item, dataset)

def test_dataset_onehots(root_dir):
    dataset = PhilharmoniaDataset(root=root_dir,
        download=True, 
        sample_rate=48000, 
        seed=0)

    for name in dataset.classes:
        example = dataset.get_example(name)
        assert int(np.argmax(example['one_hot'])) == dataset.classes.index(name)
        assert name == example['instrument']
