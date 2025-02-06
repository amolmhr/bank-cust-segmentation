# Building the pipeline for prediction
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from CustomerSegmentation.components.data_transformation import scale_data, rfm
from CustomerSegmentation.components.model_trainer import load_model
from CustomerSegmentation.logger import logging
from CustomerSegmentation import config
import pandas as pd
import numpy as np
import pickle

def predict(data:pd.DataFrame)->pd.DataFrame:
    '''
    Predict the clusters for the given data
    '''
    try:
        model = load_model(config.model_path)
        rfmdf = rfm(data)
        scaled_data = scale_data(rfmdf)
        clusters = model.predict(scaled_data)
        data['Cluster'] = clusters
        logging.info("Clusters predicted for the data")
        return data
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise Exception(f"An error occurred: {str(e)}")
    
# Building the pipeline
predict_pipeline = Pipeline([
    ('predict', predict)
])

# if __name__ == '__main__':
#     df = pd.read_csv(config.processed_data_path)
#     predict(df)
#     print("Prediction pipeline executed successfully")

