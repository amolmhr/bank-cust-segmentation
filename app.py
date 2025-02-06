# Configuring the FAST API app for running th customer segmentation model
from fastapi import FastAPI
import pandas as pd
import pickle

app = FastAPI()
model = pickle.load(open('models/model.pkl', 'rb'))


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict/")
async def predict_cluster(data:dict):
    '''
    Predict the cluster for the given data
    '''
    try:
        # Convert the data to a dataframe
        data = pd.DataFrame(data)
        # Predict the cluster
        cluster = model.predict(data)
        return {"cluster": cluster}
    except Exception as e:
        return {"error": str(e)}