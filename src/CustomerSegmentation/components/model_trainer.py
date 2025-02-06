# Training the model

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from CustomerSegmentation import config
from CustomerSegmentation.logger import logging
from CustomerSegmentation.components.data_transformation import scale_data, save_data
from CustomerSegmentation.components.data_ingestion import load_data
import pickle

def train_model(df:pd.DataFrame)->pd.DataFrame:
    '''
    Train the model on the given dataframe
    '''
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(df)
    logging.info("Model trained on the data")   
    return kmeans

def save_model(model, file_path:str)->None:
    '''
    Save the model to the given file path
    '''
    pickle.dump(model, open(file_path, 'wb'))
    logging.info(f"Model saved to {file_path}")

def load_model(file_path:str)->pd.DataFrame:
    '''
    Load the model from the given file path
    '''
    model = pickle.load(open(file_path, 'rb'))
    logging.info(f"Model loaded from {file_path}")
    return model


if __name__ == '__main__':
    df = pd.read_csv(config.processed_data_path)
    model = train_model(df)
    save_model(model, config.model_path)
    print("Model trained and saved successfully")

