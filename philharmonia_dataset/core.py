import os
from pathlib import Path
from typing import Tuple

import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import audio_utils as au
import random
import sox
import warnings
warnings.simplefilter("ignore")

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
                 download: bool = True, 
                 sample_rate: int = 48000, 
                 classes: str = 'no-percussion',
                 seed: int = 0):
        r"""creates a PyTorch Dataset object for the Philharmonia Orchestra samples.
        https://philharmonia.co.uk/resources/sound-samples/

        indexing returns a dictionary with format:
        {
            audio (np.ndarray): audio array with shape (channels, samples)
            one_hot (np.ndarray): one hot encoding of label
            instrument (str): instrument name
            articulation (str): playing articulation (e.g 'pizz-normal') for pizzicato
            dynamic (str): playing dynamic (e.g. 'forte')
            pitch (str): pitch (e.g. 'Bb5'). If instrument is unpitched, will return 'nan'. 
        }

        Args:
            root (str, optional): path to dataset root. Defaults to './data/philharmonia'.
            classes (Tuple[str], optional) which classes to include, i.e. ['banjo', 'guitar'].
            download (bool, optional): whether to download the dataset. Defaults to True.
            sample_rate (int, optional): sample rate for loading audio. Defaults to 48000.
        """
        super().__init__()
        self.sample_rate = sample_rate

        # download if requested
        if download:
            download_dataset(root)

        self.root = Path(root)

        # generate a list of dicts from our dataframe
        self.records = pd.read_csv(self.root / 'all-samples' / 'metadata.csv').to_dict('records')

        non_percussion_classes = list("saxophone,flute,guitar,contrabassoon,bass-clarinet,"\
                                "trombone,cello,oboe,bassoon,banjo,mandolin,tuba,viola,"\
                                "french-horn,english-horn,violin,double-bass,trumpet,clarinet".split(','))
        # remove all the classes not specified, unless it was left as None
        # this branch uses a fixed classlist, so let's just set this to no-percussion
        if classes == 'no-percussion':
            self.classes = non_percussion_classes
            self.records = [e for e in self.records if e['instrument'] in self.classes]
        elif classes == 'percussion':
            self.classes = list(set([e['instrument'] for e in self.records]))
            self.classes = [c for c in self.classes if c not in non_percussion_classes]
            self.records = [e for e in self.records if e['instrument'] in classes]
        elif classes is not None: 
            self.records = [e for e in self.records if e['instrument'] in classes]
            self.classes = list(set([e['instrument'] for e in self.records]))
        else:
            self.classes = list(set([e['instrument'] for e in self.records]))

        self.classes.sort()

        random.seed(seed)
        random.shuffle(self.records)

    def _retrieve_entry(self, entry):
        path_to_audio = self.root / 'all-samples' / entry['path_relative_to_root'] / entry['filename']
        assert os.path.exists(path_to_audio), f"couldn't find {path_to_audio}"

        instrument = entry['instrument']

        data = {
            'one_hot': self.get_one_hot(instrument),
        }

        # add all the keys from the entryas well
        data.update(entry)

        # import our audio using sox
        audio = au.io.load_audio_file(str(path_to_audio), self.sample_rate)
        data['audio'] = audio

        def delete_key(d, key):
            if key in  d:
                del d[key]

        delete_key(data, 'Unnamed: 0')
        delete_key(data, 'note_length')
        delete_key(data, 'path_relative_to_root')
        delete_key(data, 'filename')
        
        data['articulation'] = data['articulation'].replace('.mp3', '')
        data['pitch'] = str(data['pitch'])
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

    def get_one_hot(self, label):
        assert label in self.classes, "couldn't find label in class list"
        return np.array([1 if label == l else 0 for l in self.classes])

    def _print_one_hot_table(self):
        """ prints a markdown table with indices and classnames"""
        out =  f'| Index | Label |\n'
        out += f'|-------|-------|\n'
        for idx, name in enumerate(self.classes):
            out += f'|{idx}|{name}|\n'
        print(out)

    def get_example(self, class_name):
        """
        get a random example belonging to class_name from the dataset
        for demo purposes
        """
        subset = [e for e in self.records if e['instrument'] == class_name]
        # get a random index
        idx = torch.randint(0, high=len(subset), size=(1,)).item()

        entry = subset[idx]
        return self._retrieve_entry(entry)


if __name__ == '__main__':
    from pprint import  pprint
    dataset = PhilharmoniaDataset('./data/philharmonia', classes='percussion',
                                  download=True, sample_rate=32000, seed=0)
    print(dataset.classes)
    # breakpoint()
    # dataset._print_one_hot_table()
    # print(dataset.get_example('clarinet'))