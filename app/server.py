from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import joblib
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import os
import uvicorn

port = int(os.getenv("PORT", 8000))

app = FastAPI()

try:
    kmeans_model = joblib.load('app/models/kmeans_model.joblib')
    preprocessor = joblib.load('app/models/preprocessor.joblib')
    selector = joblib.load('app/models/selector.joblib')
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Modelos no encontrados. Asegúrate de que los archivos .joblib estén en la carpeta 'models'.")

class UserData(BaseModel):
    fullVisitorId: str
    channelGrouping: str
    weekend_prop: float
    hour: float
    sessionId: int
    device_browser: str = Field(..., alias="device.browser")
    device_deviceCategory: str = Field(..., alias="device.deviceCategory")
    device_isMobile: float = Field(..., alias="device.isMobile")
    device_operatingSystem: str = Field(..., alias="device.operatingSystem")
    totals_hits: float = Field(..., alias="totals.hits")
    totals_pageviews: float = Field(..., alias="totals.pageviews")
    bounce_prop: float
    trafficSource_medium: str = Field(..., alias="trafficSource.medium")

class PredictionResponse(BaseModel):
    cluster: int


def hour_bins(hour):
    hour = int(hour)
    if 0 <= hour < 3:
        return '00:00-02:59'
    elif 3 <= hour < 6:
        return '03:00-05:59'
    elif 6 <= hour < 9:
        return '06:00-08:59'
    elif 9 <= hour < 12:
        return '09:00-11:59'
    elif 12 <= hour < 15:
        return '12:00-14:59'
    elif 15 <= hour < 18:
        return '15:00-17:59'
    elif 18 <= hour < 21:
        return '18:00-20:59'
    else:
        return '21:00-23:59'

def process_data(df):
    rename = {
        'fullVisitorId':'UserId',
        'device.operatingSystem':'OS',
        'device.browser':'Browser',
        'device.deviceCategory':'DeviceType',
        'channelGrouping':'Channel',
        'weekend_prop':'%Weekend',
        'device.isMobile':'isMobile',
        'totals.hits':'Events',
        'totals.pageviews':'PageViews',
        'bounce_prop':'%Bounce',
        'trafficSource.medium':'Source'
        }

    df = df.rename(columns=rename)
    df['HourBins'] = df['hour'].apply(hour_bins)

    categories = {
        'Browser': 'category',
        'Channel': 'category',
        'OS': 'category',
        'DeviceType': 'category',
        'Source': 'category',
        'HourBins': 'category'
    }
    df = df.astype(categories)

    try:
        X = preprocessor.transform(df)
        X = X.toarray()
        X_selected = selector.transform(X)
        return X_selected
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar los datos: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(user_data: UserData):
    """
    Predice el cluster para un único datapoint basado en los datos del usuario.

    Los datos deben ser proporcionados exactamente como aparecen en el CSV original.

    Ejemplo para pegar en FastAPI docs:
    ```json
    {
        "fullVisitorId": "0213131142648941",
        "channelGrouping": "Direct",
        "weekend_prop": 0.0,
        "hour": 22.0,
        "sessionId": 1,
        "device.browser": "Chrome",
        "device.deviceCategory": "desktop",
        "device.isMobile": 0.0,
        "device.operatingSystem": "Macintosh",
        "totals.hits": 14.0,
        "totals.pageviews": 13.0,
        "bounce_prop": 0.0,
        "trafficSource.medium": "(none)"
    }
    ```
    """
    df = pd.DataFrame([user_data.dict(by_alias=True)])
    X_selected = process_data(df)
    cluster = kmeans_model.predict(X_selected)[0]
    return {"cluster": int(cluster)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="0.0.0.0", port=port, reload=True)