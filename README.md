# philharmonia-dataset

This is a PyTorch Dataset class implementation for 14,000 sound samples of the Philharmonia Orchestra, retrieved from their [website](https://philharmonia.co.uk/resources/sound-samples/)

usage (Python 3):
```
from philharmonia-dataset import PhilharmoniaSet,\
 train_test_split, debatch, download_dataset

# first, download the dataset 
save_path = './data/philharmonia'

# if the dataset is already downloaded, running this line will raise a warning
download_dataset(save_path)

# create a dataset object
dataset = PhilharmoniaSet(
			dataset_path=save_path, 
			classes=None, # imports all classes in dataset 
			load_audio=False
)

# split into train and test
trainloader, testloader = train_test_split(
					dataset
					batch_size=1, 
					val_split=0.2, 
					shuffle=True, 
					random_seed=0
)


# view the data
	for data in trainloader:
		print(data)

		# data will be a dictionary with keys:
		# 'filename': list(str) filename of audio file
		# 'instrument': list(str) instrument name 
		# 'pitch': list(str) pitch being played by instrument
		# 'label': list(str) integer encoded class
		# 'path_to_audio': list(str) relative path to audio file
		
		# if load_audio is set to true in the dataset object, 
		# the following keys will be available as well
		# 'audio': list(torch.Tensor) audio file with shape (channels, time)
		# 'sr': list(int)
		
```