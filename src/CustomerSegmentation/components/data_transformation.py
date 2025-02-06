# Data Transformation
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from CustomerSegmentation import config
from CustomerSegmentation.logger import logging
from CustomerSegmentation.components.data_ingestion import load_data


def remove_columns(df:pd.DataFrame, columns:list)->pd.DataFrame:
    '''
    Remove the given columns from the dataframe
    ''' 

    df = df.drop(columns=columns)
    logging.info(f"Columns removed from the dataframe: {columns}")
    return df

def dropna_rows(df:pd.DataFrame, column:str)->pd.DataFrame:
    '''
    Remove the rows from the dataframe where the column has the given value
    ''' 

    df = df.dropna(subset=[column])
    logging.info(f"Rows removed from the dataframe where {column} has null values")
    return df
def rfm(df:pd.DataFrame)->pd.DataFrame:
    '''
    Calculate the Recency, Frequency, and Monetary values for the given dataframe
    '''
    latest_date = df['InvoiceDate'].max() + pd.DateOffset(days=1)
    rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
    'UnitPrice': lambda x: (x * df.loc[x.index, 'Quantity']).sum()  # Monetary
    })
    logging.info("RFM values calculated for the customers")

    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    # Reset index
    rfm.reset_index(inplace=True)

    return rfm

def scale_data(df:pd.DataFrame)->pd.DataFrame:
    '''
    Scale the data using MinMaxScaler
    '''
    scaler = MinMaxScaler()
    rfm_scaled = scaler.fit_transform(df[['Recency', 'Frequency', 'Monetary']])
    logging.info("Data scaled using MinMaxScaler")
    rfm_scaled = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])

    return rfm_scaled

def save_data(df:pd.DataFrame, file_path:str)->None:
    '''
    Save the dataframe to the given file path
    '''
    df.to_csv(file_path, index=False)
    logging.info(f"Data saved to {file_path}")

if __name__ == "__main__":
    df = load_data(config.raw_data_path)
    df = dropna_rows(df, 'CustomerID')
    rfmdf = rfm(df)
    rfmdf = scale_data(rfmdf)
    print(rfmdf.head())
    save_data(rfmdf, config.processed_data_path)
