from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

base_path = Path(__file__).resolve().parents[1]

model_path = base_path / "src" / "models" / "churn_pipeline.pkl"

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = joblib.load(model_path)
        print("Modelo carregado com sucesso!")
    except FileNotFoundError:
        print(f"Modelo não encontrado em: {model_path}")
        print("Certifique-se de que o modelo foi treinado e salvo.")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")

    yield

    print("Aplicação finalizando...")


app = FastAPI(title="Churn Prediction API", lifespan=lifespan)


class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def predict_churn(data: CustomerInput):
    global model

    if model is None:
        return {"error": "Modelo não foi carregado corretamente"}

    try:
        df = pd.DataFrame([data.dict()])
        proba = model.predict_proba(df)[0, 1]
        return {"churn_probability": float(proba)}
    except Exception as e:
        return {"error": f"Erro na predição: {str(e)}"}


@app.get("/")
def read_root():
    return {"message": "Churn Prediction API is running!"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

# uvicorn api.main:app --reload

# http://127.0.0.1:8000/docs#/default/predict_churn_predict_post

# teste
# {
#   "gender": "Female",
#   "SeniorCitizen": 0,
#   "Partner": "Yes",
#   "Dependents": "No",
#   "tenure": 12,
#   "PhoneService": "Yes",
#   "MultipleLines": "No",
#   "InternetService": "Fiber optic",
#   "OnlineSecurity": "No",
#   "OnlineBackup": "Yes",
#   "DeviceProtection": "No",
#   "TechSupport": "No",
#   "StreamingTV": "Yes",
#   "StreamingMovies": "Yes",
#   "Contract": "Month-to-month",
#   "PaperlessBilling": "Yes",
#   "PaymentMethod": "Electronic check",
#   "MonthlyCharges": 85.5,
#   "TotalCharges": 1024.3
# }