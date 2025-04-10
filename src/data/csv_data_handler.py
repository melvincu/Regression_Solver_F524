import pandas as pd

from data.data_handler import DataHandler

class CSVDataHandler(DataHandler):
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
    
    def get_data(self):
        df = pd.read_csv(self.file_path)
        # ...