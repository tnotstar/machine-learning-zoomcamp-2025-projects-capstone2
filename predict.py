import os
import pickle
import pprint as pp

import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel


class StrokePredictionRequest(BaseModel):
    gender: str
    age: float
    hypertension: int
    ever_married: int
    heart_disease: int
    work_type: str
    residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str


class StrokePredictionResponse(BaseModel):
    stroke_probability: float
    has_stroke: bool


app = FastAPI(title="Stroke Prediction Service")

with open("pipeline_v1.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)


def predict_single(input: dict) -> tuple[float, bool]:
    X = dv.transform([input])
    probability = model.predict_proba(X)[0, 1]
    clazz = model.predict(X)[0]
    print(clazz)
    return float(probability), bool(clazz)


input = {
    "gender": "Male",
    "age": 67.0,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": 1,
    "work_type": "Private",
    "residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked",
}

print(f"Canary prediction probability: {predict_single(input)}")


@app.get("/")
async def docs_redirect():
    return RedirectResponse(url="/docs")


@app.post("/predict")
async def predict(input: StrokePredictionRequest) -> StrokePredictionResponse:
    pp.pprint(input)
    tx = input.model_dump()
    pp.pprint(tx)
    prob, clazz = predict_single(tx)
    output = StrokePredictionResponse(stroke_probability=prob, has_stroke=clazz)
    pp.pprint(output)
    return output


if __name__ == "__main__":
    IP_ADDRESS = os.environ.get("IP_ADDRESS", "0.0.0.0")
    PORT = os.environ.get("PORT", 8080)
    uvicorn.run(app, host=IP_ADDRESS, port=PORT)
