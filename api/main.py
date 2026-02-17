from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path
import time
from api.monitoring import monitor

base_path = Path(__file__).resolve().parents[1]
model_path = base_path / "src" / "models" / "churn_pipeline.pkl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        pipeline = joblib.load(model_path)
        app.state.pipeline = pipeline
        app.state.preprocessor = pipeline.named_steps["preprocess"]
        app.state.model = pipeline.named_steps["model"]
        app.state.explainer = shap.TreeExplainer(app.state.model)
        print("Modelo e SHAP Explainer carregados com sucesso!")
    except FileNotFoundError:
        print(f"Modelo não encontrado em: {model_path}")
        app.state.pipeline = None
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        app.state.pipeline = None

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


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_level: str
    top_features: list[dict]


def _get_risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    return "LOW"

def _get_shap_explanation(app_state, df: pd.DataFrame, top_n: int = 5) -> list[dict]:
    X_transformed = app_state.preprocessor.transform(df)
    feature_names = app_state.preprocessor.get_feature_names_out()

    shap_values = app_state.explainer.shap_values(X_transformed)

    #classificação binária, pegar os valores da classe positiva (churn=1)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_row = shap_values[0]  #apenas 1 amostra

    feature_impact = sorted(zip(feature_names, shap_row),key=lambda x: abs(x[1]),reverse=True,)[:top_n]

    return [
        {
            "feature": name,
            "shap_value": round(float(value), 4),
            "direction": "aumenta churn" if value > 0 else "reduz churn",
        }
        for name, value in feature_impact
    ]


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(data: CustomerInput):
    if app.state.pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo não foi carregado corretamente")

    start_time = time.perf_counter()
    try:
        df = pd.DataFrame([data.model_dump()])
        proba = app.state.pipeline.predict_proba(df)[0, 1]
        explanation = _get_shap_explanation(app.state, df)

        latency_ms = (time.perf_counter() - start_time) * 1000
        monitor.record_prediction(float(proba), latency_ms)

        return PredictionResponse(
            churn_probability=round(float(proba), 4),
            risk_level=_get_risk_level(proba),
            top_features=explanation,
        )
    except Exception as e:
        monitor.record_error()
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "Churn Prediction API is running!"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": app.state.pipeline is not None,
    }


@app.get("/metrics")
def get_model_metrics():
    return monitor.get_metrics()