# Creating data ingestion 

import pandas as pd
import numpy as np
import os
from CustomerSegmentation.logger import logging
from CustomerSegmentation import config

def load_data(file_path:str)->pd.DataFrame:
    """
    Load data from the given file path
    Args:
        file_path (str): Path to the file
    Returns:
        pd.DataFrame: Dataframe containing the loaded data
    """
    try:
        df = pd.read_excel(file_path)
        logging.info(f"Data loaded from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error occured while loading data from {file_path} : {str(e)}")
        raise e

if __name__ == "__main__":
    df = load_data(config.raw_data_path)
    print(df.head())
