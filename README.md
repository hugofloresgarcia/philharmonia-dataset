# philharmonia-dataset

This is a PyTorch Dataset implementation for 14,000 sound samples of the Philharmonia Orchestra, retrieved from their [website](https://philharmonia.co.uk/resources/sound-samples/)

## Installing

Clone the repo and install using pip (and install ffmpeg)
```bash
apt-get install ffmpeg
git clone https://github.com/hugofloresgarcia/philharmonia-dataset
cd philharmonia-dataset && pip install -e .
```

## Usage (Python 3):
```python
from philharmonia_dataset import PhilharmoniaSet

# create a dataset object
dataset = PhilharmoniaDataset(root='./data/philharmonia', 
							 download=True, 
							 sample_rate=48000,)
```

During the first run, calling `PhilharmoniaDataset` will download the audio files from [here](https://philharmonia.co.uk/resources/sound-samples/) and convert `mp3` files to `wav`, for faster loading. This will take approximately 5-10 minutes. 

sample output
```
dataset[0]

{
	audio (np.ndarray): audio array with shape (channels, samples)
	one_hot (np.ndarray): one hot encoding of label
	instrument (str): instrument name
	articulation (str): playing articulation (e.g 'pizz-normal') for pizzicato
	dynamic (str): playing dynamic (e.g. 'forte')
	pitch (str): pitch (e.g. 'B5'). If instrument is unpitched, will return 'nan'. 
}
```

## Labels

each example is assigned one of the following labels:

| Index | Label |
|-------|-------|
|0|banjo|
|1|bass-clarinet|
|2|bassoon|
|3|cello|
|4|clarinet|
|5|contrabassoon|
|6|double-bass|
|7|english-horn|
|8|flute|
|9|french-horn|
|10|guitar|
|11|mandolin|
|12|oboe|
|13|saxophone|
|14|trombone|
|15|trumpet|
|16|tuba|
|17|viola|
|18|violin|

for example, the one_hot encoding of `clarinet` would be 
```
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```
(index number 4 is 	`1`, while all other indices are `0`)