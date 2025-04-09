from pandas import read_csv

from data.data_handler import DataHandler

class CSVDataHandler(DataHandler):
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
    
    def get_data():
        pass