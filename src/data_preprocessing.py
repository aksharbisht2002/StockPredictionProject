import pandas as pd
from sklearn.preprocessing import MinMaxScaler
def load_data():
    index_data = pd.read_csv('./data/indexData.csv')
    index_info = pd.read_csv('./data/indexInfo.csv')
    index_processed = pd.read_csv('./data/indexProcessed.csv')

    # Display first few rows
    print("Index Data:")
    print(index_data.head())
    
    print("Index Info:")
    print(index_info.head())
    
    print("Index Processed:")
    print(index_processed.head())
    
    return index_data, index_info, index_processed

def preprocess_data(data):
    # Ensure the column exists in the dataset
    if 'CloseUSD' not in data.columns:
        raise ValueError("Column 'CloseUSD' not found in the dataset.")
    
    # Normalize the CloseUSD column
    scaler = MinMaxScaler()
    data['Normalized_Column'] = scaler.fit_transform(data[['CloseUSD']])
    
    # Additional preprocessing steps (if needed)
    return data
