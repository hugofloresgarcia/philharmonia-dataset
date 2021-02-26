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
```
dataset[0]

{
	audio (np.ndarray): audio array with shape (channels, samples)
	onehot (str): one hot encoding of label
	instrument (str): instrument name
	articulation (str): playing articulation (e.g 'pizz-normal') for pizzicato
	dynamic (str): playing dynamic (e.g. 'forte')
	pitch (str): pitch (e.g. 'B5'). If instrument is unpitched, will return 'nan'. 
}
```