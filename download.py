import kagglehub as kh
import pandas as pd 
import os
import logging as log

class DatasetDownloader:
    def __init__(self, datafolder = 'data'):
            self.api = kh.KaggleHub()
            self.datafolder = datafolder
            if not os.path.exists(self.datafolder):
                    os.makedirs(self.datafolder)
        
    def download_dataset(self, dataset_name):
            path = os.path.join(self.datafolder, dataset_name + ".csv")
            self.api.dataset_download_files(
                dataset_name, path=path, unzip=True)
            log.info(f"Dataset '{dataset_name}' downloaded and extracted to '{dataset_name}.csv'")
            return pd.read_csv(os.path.join(dataset_name + ".csv"))
    
    def load_dataset(self, dataset_name):
            path = os.path.join(self.datafolder, dataset_name + ".csv")
            if not os.path.exists(path):
                    log.info(f"Dataset '{dataset_name}' not found locally. Downloading...")
                    return self.download_dataset(dataset_name)
            else:
                    log.info(f"Loading dataset '{dataset_name}' from local storage.")
                    return pd.read_csv(os.path.join(dataset_name + ".csv"))



