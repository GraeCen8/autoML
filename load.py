import kagglehub as kh
import pandas as pd 
import os
import logging as log
from kagglehub import KaggleDatasetAdapter

class DatasetDownloader:
    def __init__(self, datafolder = 'data'):
           # self.api = kh.KaggleHub()
            self.datafolder = datafolder
            if not os.path.exists(self.datafolder):
                    os.makedirs(self.datafolder)
        
    #def download_dataset(self, dataset_id, dataset_name):
    #        os.system(f'bash ./download.sh {dataset_id} {dataset_name}')    
    
    def load_dataset(self, dataset_name):
            path = os.path.join(self.datafolder, f'{dataset_name}' + ".csv")
            return pd.read_csv(os.path.join(path))
            
                    
if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.download_dataset("shree1992/housedata?select=data.csv", "houses")
    df = downloader.load_dataset("houses")
    print(df.head())
