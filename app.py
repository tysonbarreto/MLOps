from enum import Enum

import joblib
import pandas as pd
from fastapi import FastAPI
from xgboost import XGBRegressor
from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from bodyfat import Config
from bodyfat.features import create_new_features


class Sex(str,Enum):
    male:str = "M"
    female:str = "F"


class BodyfarPredictionRequest(BaseModel):
    hip: float
    abdomen: float
    age: int
    weight: float
    height: float
    sex:Sex
    

class BodyfarPrediction(BaseModel):
    bodyfat: float
    
preprocessor = joblib.load(Config.Path.MODELS_DIR / "preprocessor.joblib")

model = XGBRegressor()
model.load_model(Config.Path.MODELS_DIR / "model.json")

app = FastAPI(
    title="Bodyfat Prediction API",
    description="Rest API to predict bodyfat percentage based on provided measurements",
)

@app.post("/predict",response_model=BodyfarPrediction)
def make_prediction(input_data: BodyfarPredictionRequest):
    input_df = pd.DataFrame([input_data.model_dump()])
    input_df = create_new_features(input_df)
    preprocessed_data = preprocessor.transform(input_df)
    prediction = model.predict(preprocessed_data)[0]
    return BodyfarPrediction(bodyfat=prediction)