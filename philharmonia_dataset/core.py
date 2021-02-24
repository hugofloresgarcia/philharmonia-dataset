import os
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import audio_utils as au

from .download import download_dataset

def debatch(data):
    """
     convert batch size 1 to None
    """
    for key in data:
        if isinstance(data[key], list):
            assert len(data[key]) == 1, "can't debatch with batch size greater than 1" 
            data[key] = data[key][0]
    return data

class PhilharmoniaDataset(Dataset):
    def __init__(self, 
                 root: str = './data/philharmonia', 
                 classes: tuple = None,
                 download: bool = True, 
                 sample_rate: int = 48000):
        super().__init__()
        self.sample_rate = sample_rate

        # download if requested
        if download:
            download_dataset(root)

        self.root = Path(root)

        # generate a list of dicts from our dataframe
        self.records = pd.read_csv(self.root / 'all-samples' / 'metadata.csv', header=1).to_dict('records')

        # remove all the classes not specified, unless it was left as None
        if classes == 'no-percussion':
            self.classes = list("saxophone,flute,guitar,contrabassoon,bass-clarinet,"\
                                "trombone,cello,oboe,bassoon,banjo,mandolin,tuba,viola,"\
                                "french-horn,english-horn,violin,double-bass,trumpet,clarinet".split(','))
            self.records = [e for e in self.records if e['instrument'] in self.classes]
        elif classes is not None: 
            self.records = [e for e in self.records if e['instrument'] in classes]
            self.classes = list(set([e['instrument'] for e in self.records]))
        else:
            self.classes = list(set([e['instrument'] for e in self.records]))

        self.classes.sort()

    def _retrieve_entry(self, entry):
        path_to_audio = self.root / 'all-samples' / entry['parent'] / entry['filename']
        assert os.path.exists(path_to_audio), f"couldn't find {path_to_audio}"

        instrument = entry['instrument']

        data = {
            'one_hot': self.get_onehot(instrument),
            'label': np.argmax(self.get_onehot(instrument)), 
        }

        # add all the keys from the entryas well
        data.update(entry)

        # import our audio using torchaudio
        audio = au.io.load_audio_file(path_to_audio, self.sample_rate)
        data['audio'] = audio

        return data

    def __getitem__(self, index):
        def retrieve(index):
            return self._retrieve_entry(self.records[index])

        if isinstance(index, int):
            return retrieve(index)
        elif isinstance(index, slice):
            result = []
            start, stop, step = index.indices(len(self))
            for idx in range(start, stop, step):
                result.append(retrieve(idx))
            return result
        else:
            raise TypeError("index is neither an int or a slice")

    def __len__(self):
        return len(self.records)

    def get_onehot(self, label):
        assert label in self.classes, "couldn't find label in class list"
        return np.array([1 if label == l else 0 for l in self.classes])

    def get_example(self, class_name):
        """
        get a random example belonging to class_name from the dataset
        for demo purposes
        """
        subset = [e for e in self.records if e['instrument'] == class_name]
        # get a random index
        idx = torch.randint(0, high=len(subset), size=(1,)).item()

        entry = subset[idx]
        return self.retrieve_entry(entry)

