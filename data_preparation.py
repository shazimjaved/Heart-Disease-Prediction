import pandas as pd
import numpy as np
import requests
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def download_heart_disease_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open('data/heart_disease_raw.csv', 'w') as f:
            f.write(response.text)
        
        df = pd.read_csv('data/heart_disease_raw.csv', names=column_names)
        
        df = df.replace('?', np.nan)
        
        numeric_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        df = df.fillna(df.median())
        
        df['target'] = (df['target'] > 0).astype(int)
        
        df.to_csv('data/heart_disease_processed.csv', index=False)
        
        print(f"Dataset downloaded and processed successfully!")
        print(f"Shape: {df.shape}")
        print(f"Target distribution: {df['target'].value_counts()}")
        
        return df
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def prepare_data_for_training(df):
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    df = download_heart_disease_data()
    if df is not None:
        X_train, X_test, y_train, y_test, scaler = prepare_data_for_training(df)
        print("Data preparation completed successfully!")
