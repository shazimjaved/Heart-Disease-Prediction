import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import pickle
import os
from data_preparation import download_heart_disease_data, prepare_data_for_training

def train_models():
    
    df = download_heart_disease_data()
    if df is None:
        print("Failed to load data")
        return None
    
    X_train, X_test, y_train, y_test, scaler = prepare_data_for_training(df)
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    best_model = None
    best_score = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred))
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_model_name = name
            
            with open('models/best_model.pkl', 'wb') as f:
                pickle.dump(model, f)
    
    print(f"\nBest model: {best_model_name} with accuracy: {best_score:.4f}")
    
    feature_names = df.drop('target', axis=1).columns.tolist()
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    return best_model, best_model_name, best_score

def load_model():
    try:
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
            
        return model, scaler, feature_names
    except FileNotFoundError as e:
        print(f"Model files not found: {e}")
        return None, None, None

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    train_models()
