import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import os
import numpy as np

from .dl_dataset import download_dataset
import logging

def debatch(data):
    """
     convert batch size 1 to None
    """
    for key in data:
        if isinstance(data[key], list):
            assert len(data[key]) == 1, "can't debatch with batch size greater than 1" 
            data[key] = data[key][0]
    return data

class PhilharmoniaSet(Dataset):
    def __init__(self, 
                dataset_path: str = './data/philharmonia', 
                classes: tuple = None,
                download = True, 
                load_audio: bool = True):
        """
        create a PhilharmoniaSet object.
        params:
            path_to_csv (str): path to metadata.csv created upon downloading the dataset
            classes (tuple[str]): tuple with classnames to include in the dataset
            load_audio (bool): whether to load audio or pass the path to audio instead when retrieving an item
        """
        super().__init__()
        self.load_audio = load_audio

        # download if requested
        if download:
            download_dataset(dataset_path)

        path_to_csv = os.path.join(dataset_path ,'all-samples', 'metadata.csv')

        assert os.path.exists(path_to_csv), f"couldn't find metadata:{path_to_csv}"
        # generate a list of dicts from our dataframe
        self.metadata = pd.read_csv(path_to_csv).to_dict('records')

        # remove all the classes not specified, unless it was left as None
        if classes == 'no_percussion':
            self.classes = list("saxophone,flute,guitar,contrabassoon,bass-clarinet,trombone,cello,oboe,bassoon, \
                banjo,mandolin,tuba,viola,french-horn,english-horn,violin,double-bass,trumpet,clarinet".split(','))
            self.metadata = [e for e in self.metadata if e['instrument'] in self.classes]
        elif classes is not None: # if it's literally anything else lol (filter out unneeded metadata)
            self.metadata = [e for e in self.metadata if e['instrument'] in classes]
            self.classes = list(set([e['instrument'] for e in self.metadata]))
        else:
            self.classes = list(set([e['instrument'] for e in self.metadata]))

        self.classes.sort()
        
    def check_metadata(self):
        missing_files = []
        for entry in self.metadata:
            if not os.path.exists(entry['path_to_audio']):
                logging.warn(f'{entry["path_to_audio"]} is missing.')
                missing_files.append(entry['path_to_audio'])

        assert len(missing_files) == 0, 'some files were missing in the dataset.\
             delete metadata file and download again, or delete missing entries from metadata'

    def get_class_frequencies(self):
        """
        return a tuple with unique class names and their number of items
        """
        classes = []
        for c in self.classes:
            subset = [e for e in self.metadata if e['instrument'] == c]
            info = (c, len(subset))
            classes.append(info)

        return tuple(classes)

    def retrieve_entry(self, entry):
        path_to_audio = entry['path_to_audio']
        filename = path_to_audio.split('/')[-1]

        assert os.path.exists(path_to_audio), f"couldn't find {path_to_audio}"

        instrument = entry['instrument']
        pitch = entry['pitch']

        data = {
            'filename': filename,
            'one_hot': self.get_onehot(instrument),
            'numeric_label': np.argmax(self.get_onehot(instrument)), 
        }

        # add all the keys from the entry as well
        data.update(entry)

        if self.load_audio:
            # import our audio using torchaudio
            audio, sr = torchaudio.load(path_to_audio)
            data['audio'] = audio
            data['sr'] = sr
                
        return data

    def __getitem__(self, index):
        def retrieve(index):
            return self.retrieve_entry(self.metadata[index])

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
        return len(self.metadata)

    def get_onehot(self, label):
        assert label in self.classes, "couldn't find label in class list"
        return np.array([1 if label == l else 0 for l in self.classes])

    def get_example(self, class_name):
        """
        get a random example belonging to class_name from the dataset
        for demo purposes
        """
        subset = [e for e in self.metadata if e['instrument'] == class_name]
        # get a random index
        idx = torch.randint(0, high=len(subset), size=(1,)).item()

        entry = subset[idx]
        return self.retrieve_entry(entry)

