"""
download philharmonia samples from
https://philharmonia.co.uk/resources/sound-samples/
"""
import urllib.request
import zipfile, re, os
import pandas as pd
import pydub
import glob
import logging
from pathlib import Path
import audio_utils as au
import librosa
import warnings
warnings.simplefilter("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def extract_nested_zip(zippedFile, toFolder):
    """ Extract a zip file including any nested zip files
        Delete the zip file(s) after extraction
    """
    with zipfile.ZipFile(zippedFile, 'r') as zfile:
        zfile.extractall(path=toFolder)
    os.remove(zippedFile)
    
    for root, dirs, files in os.walk(toFolder):
        dirs[:] = [d for d in dirs if not d[0] == '_']
        for filename in files:
            if re.search(r'\.zip$', filename):
                fileSpec = os.path.join(root, filename)
                logging.info(f'extracting: {fileSpec}')
                filename = filename[0:-4]
                extract_nested_zip(fileSpec, os.path.join(root, filename))

def create_entry(args):
    path, root_dir = args
    path = Path(path)
    filename = path.name
    logging.info(f'processing: {filename}')
    if 'viola_D6_05_piano_arco-normal.mp3' in filename or \
        'saxophone_Fs3_15_fortissimo_normal.mp3'  in filename or \
        "guitar_Gs4_very-long_forte_normal.mp3" in filename  or \
        "bass-clarinet_Gs3_025_piano_normal.mp3" in filename:
        os.remove(path)
        return
    if path.suffix  == '.mp3':
        # convert mp3 to wav
        src = path
        # dst = path.with_suffix('.wav')
        dst = src

        # convert wav to mp3
        # sr = librosa.core.get_samplerate(src)
        # sound = au.io.load_audio_file(src, sr)
        # au.io.write_audio_file(sound, dst, sr, 'wav', exist_ok=True)

        # os.remove(src)

        path = dst
        filename = path.name

        fsplit = filename.split('_')
        
        metadata = {
            'instrument': fsplit[0],
            'pitch': fsplit[1], 
            'path_relative_to_root': path.parent.relative_to(root_dir),
            'filename': filename, 
            'note_length': fsplit[2], 
            'dynamic': fsplit[3], 
            'articulation': fsplit[4]
        }
    return metadata
                
def generate_dataframe(root_dir):
    """
    generate a dictionary for metadata from our dataset
    """
    data = []
    root = Path(root_dir)
    mp3s = glob.glob(f'{root}/**/*.mp3', recursive=True)
    roots = [root_dir for r in mp3s]

    args = [(mp3, r) for mp3, r in zip(mp3s, roots)]

    from multiprocessing.pool import ThreadPool

    pool = ThreadPool()
    data = pool.map(create_entry, args)

    # data = process_map(create_entry, mp3s, roots)
    data = [d for d in data if isinstance(d, dict)]
    return pd.DataFrame(data)

def download_dataset(save_path = "./data/philharmonia"):
    """
    download audio samples for the philharmonia dataset
    a metadata frame will be saved to philharmonia/all-samples/metadata.csv
    
    params:
        save_path (str): path to save the dataset to
    returns:
        None
    """
    if os.path.exists(os.path.join(save_path, 'all-samples', 'metadata.csv')):
        logging.info(f"Philharmonia: looks like there is already a dataset in {save_path}. will not download. delete {save_path} and try again if you want to download")
        return 
    url = "https://philharmonia-assets.s3-eu-west-1.amazonaws.com/uploads/2020/02/12112005/all-samples.zip"
    
    # download the zip
    logging.info(f'downloading from {url}')
    logging.info('this may take a while ? :/ i think its like 250MB')
    urllib.request.urlretrieve(url, f'{save_path}.zip')
    
    # extract everything recursively
    extract_nested_zip(f'{save_path}.zip', save_path)
    
    logging.info('generating dataframe... (this will take around 5-7 min. go get a cup of coffee)')
    df_path = os.path.join(save_path, 'all-samples', 'metadata.csv')
    metadata = generate_dataframe(os.path.join(save_path, 'all-samples'))
    metadata.to_csv(df_path)
    logging.info(f'dataframe saved to {df_path}')
    logging.info('all done!')



if __name__ == "__main__":
    download_dataset()
    