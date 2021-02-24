# philharmonia-dataset

This is a PyTorch Dataset implementation for 14,000 sound samples of the Philharmonia Orchestra, retrieved from their [website](https://philharmonia.co.uk/resources/sound-samples/)

usage (Python 3):
```python
from philharmonia_dataset import PhilharmoniaSet

# create a dataset object
dataset = PhilharmoniaDataset(
			root='./data/philharmonia', 
			classes='no-percussion', # dont load a bunch of percussion instruments 
			download=True, 
      sample_rate=SAMPLE_RATE,
)
```

sample output
```python
dataset[0]

{
 'articulation': 'arco-normal.mp3',
 'audio': array([[-1.6468803e-06, -2.8569633e-05, -7.0609335e-06, ...,
         -1.5821372e-06,  7.5128804e-07,  0.0000000e+00]], dtype=float32), # array with shape (channels, samples)
 'dynamic': 'piano',
 'filename': 'violin_A6_1_piano_arco-normal.mp3',
 'instrument': 'violin',
 'label': 18,
 'note_length': '1',
 'one_hot': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
 'parent': 'violin',
 'pitch': 'A6'}
```