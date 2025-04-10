from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from data.data_handler import DataHandler

class KaggleDataHandler(DataHandler):
    
    def __init__(self, fetched_df):
        super().__init__()
        self.df = fetched_df
        
    def get_data(self):
        
        # Get features and target
        X = self.df.data
        y = self.df.target
        
        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
            
        return X_train, X_test, y_train, y_test