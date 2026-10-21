from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "advertising.csv"
FEATURES = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"

app = FastAPI(
    title="Sales Prediction API",
    description="Sales prediction from TV, Radio, and Newspaper advertising budgets.",
    version="1.0.0",
)


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "dataset/advertising.csv is missing. Keep the existing dataset folder "
            "beside app.py when applying this patch."
        )
    df = pd.read_csv(DATASET_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    # Normalize common alternate column names.
    aliases = {
        "tv": "TV", "tv ad budget ($)": "TV",
        "radio": "Radio", "radio ad budget ($)": "Radio",
        "newspaper": "Newspaper", "newspaper ad budget ($)": "Newspaper",
        "sales": "Sales", "sales ($)": "Sales",
    }
    df = df.rename(columns={c: aliases.get(c.lower(), c) for c in df.columns})
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing columns: {missing}. Expected: {FEATURES + [TARGET]}"
        )
    df = df[FEATURES + [TARGET]].copy()
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
    if len(df) < 10:
        raise ValueError("At least 10 valid rows are required to train the models.")
    return df


@lru_cache(maxsize=1)
def train_models():
    df = load_dataset()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    candidates = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=1
        ),
    }
    models, metrics = {}, {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        models[name] = pipeline
        metrics[name] = {
            "MAE": round(float(mean_absolute_error(y_test, pred)), 4),
            "MSE": round(float(mse), 4),
            "RMSE": round(float(np.sqrt(mse)), 4),
            "R2 Score": round(float(r2_score(y_test, pred)), 4),
        }
    best_name = max(metrics, key=lambda name: metrics[name]["R2 Score"])
    return df, models, metrics, best_name


class PredictionRequest(BaseModel):
    tv: float = Field(ge=0, le=1000)
    radio: float = Field(ge=0, le=1000)
    newspaper: float = Field(ge=0, le=1000)
    model_name: str | None = None


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api/health")
def health():
    try:
        df, _, _, best_name = train_models()
        return {
            "status": "ok",
            "dataset_rows": int(len(df)),
            "best_model": best_name,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/summary")
def summary():
    try:
        df, _, metrics, best_name = train_models()
        return {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "summary": {
                col: {
                    "mean": round(float(df[col].mean()), 3),
                    "min": round(float(df[col].min()), 3),
                    "max": round(float(df[col].max()), 3),
                }
                for col in df.columns
            },
            "model_metrics": metrics,
            "best_model": best_name,
            "sample": df.head(10).to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/predict")
def predict(request: PredictionRequest):
    try:
        _, models, metrics, best_name = train_models()
        chosen = request.model_name or best_name
        if chosen not in models:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model. Choose one of: {list(models)}",
            )
        row = pd.DataFrame(
            [[request.tv, request.radio, request.newspaper]],
            columns=FEATURES,
        )
        value = float(models[chosen].predict(row)[0])
        return {
            "prediction": round(value, 3),
            "model_name": chosen,
            "metrics": metrics[chosen],
            "inputs": {
                "TV": request.tv,
                "Radio": request.radio,
                "Newspaper": request.newspaper,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
