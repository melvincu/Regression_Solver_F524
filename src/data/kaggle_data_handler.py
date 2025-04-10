from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from data.data_handler import DataHandler

class KaggleDataHandler(DataHandler):
    
    def get_data(self):
        # kaggle dataset
        housing = fetch_california_housing()
        
        # Get features and target
        X = housing.data
        y = housing.target
        
        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
            
        return X_train, X_test, y_train, y_test