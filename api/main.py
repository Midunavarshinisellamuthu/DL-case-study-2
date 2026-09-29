import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "model" / "fish_mortality_lstm_model.pth"
SCALER_PATH = PROJECT_DIR / "model" / "scaler.pkl"
DATASET_PATH = PROJECT_DIR / "dataset" / "aquaponics_5min_complete.csv"
WINDOW_SIZE = 144
FEATURE_NAMES = ["DO", "temperature", "pH", "turbidity", "ammonia", "fish_weight"]
RISK_LABELS = {0: "Low", 1: "Moderate", 2: "High"}


class FishMortalityLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 64
        self.lstm = nn.LSTM(
            input_size=6,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),
        )

    def forward(self, values):
        output, _ = self.lstm(values)
        forward_state = output[:, -1, :self.hidden_size]
        backward_state = output[:, 0, self.hidden_size:]
        return self.head(torch.cat([forward_state, backward_state], dim=1))


class Reading(BaseModel):
    DO: float
    temperature: float
    pH: float
    turbidity: float
    ammonia: float
    fish_weight: float


class PredictionRequest(BaseModel):
    readings: List[Reading]


class PredictionResponse(BaseModel):
    risk_label: str
    model_label: str
    biological_label: str
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float
    input_mode: str


model = None
scaler = None


@asynccontextmanager
async def lifespan(_app):
    global model, scaler
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Feature scaler not found: {SCALER_PATH}")

    with SCALER_PATH.open("rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    model = FishMortalityLSTM()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    yield


app = FastAPI(title="AquaGuard 5-minute API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


def _biological_risk(readings: np.ndarray) -> int:
    latest = readings[-1]
    previous_hour = readings[max(0, len(readings) - 13)]
    row = pd.Series({
        "DO": latest[0],
        "temperature": latest[1],
        "pH": latest[2],
        "turbidity": latest[3],
        "ammonia": latest[4],
        "do_slope": latest[0] - previous_hour[0],
    })

    if latest[0] < 2.5 or latest[2] < 5.5 or latest[2] > 9.5:
        return 2
    if latest[4] > 2.5 or latest[1] > 35.0:
        return 2

    score = 3 * (2 if latest[0] < 2.5 else 1 if latest[0] < 4.0 else 0)
    score += 2 * (2 if latest[3] > 100 else 1 if latest[3] > 50 else 0)
    score += 1 * (2 if latest[2] < 5.5 or latest[2] > 9.5 else 1 if latest[2] < 6.5 or latest[2] > 8.5 else 0)
    score += 1 * (2 if latest[1] > 35 else 1 if latest[1] > 31 else 0)
    score += 2 * (2 if latest[4] > 2.5 else 1 if latest[4] > 0.5 else 0)
    label = 0 if score <= 3 else 1 if score <= 7 else 2
    if latest[0] - previous_hour[0] <= -0.75 and label == 0:
        label = 1
    return label


def predict_readings(readings: np.ndarray) -> dict:
    if model is None or scaler is None:
        raise RuntimeError("Model assets are not loaded")
    if readings.shape not in ((1, 6), (WINDOW_SIZE, 6)):
        raise ValueError("Provide one reading or exactly 144 readings with six features each")

    input_mode = "single-reading preview" if len(readings) == 1 else "144-reading sequence"
    if len(readings) == 1:
        readings = np.repeat(readings, WINDOW_SIZE, axis=0)

    started = time.perf_counter()
    scaled = scaler.transform(readings)
    input_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(input_tensor), dim=1)[0]

    model_class = int(torch.argmax(probabilities).item())
    model_confidence = float(probabilities[model_class].item())
    bio_class = _biological_risk(readings)
    if bio_class == 2:
        final_class = 2
        confidence = max(model_confidence if model_class == 2 else 0.95, 0.95)
    elif bio_class == 1 and model_class == 0:
        final_class = 1
        confidence = max(float(probabilities[1].item()), 0.75)
    else:
        final_class = model_class
        confidence = model_confidence

    return {
        "risk_label": RISK_LABELS[final_class],
        "model_label": RISK_LABELS[model_class],
        "biological_label": RISK_LABELS[bio_class],
        "confidence": confidence,
        "probabilities": {
            RISK_LABELS[index]: float(probabilities[index].item())
            for index in range(3)
        },
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "input_mode": input_mode,
    }


@app.get("/api/health")
async def health():
    return {"status": "ready" if model is not None and scaler is not None else "loading"}


@app.get("/api/dataset/summary")
async def dataset_summary():
    if not DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail="Bundled 5-minute dataset is missing")
    dataset = pd.read_csv(DATASET_PATH, usecols=["split"])
    counts = dataset["split"].value_counts().to_dict()
    return {
        "filename": DATASET_PATH.name,
        "rows": int(len(dataset)),
        "splits": {str(name): int(count) for name, count in counts.items()},
        "sampling_interval": "5 minutes",
        "window_steps": WINDOW_SIZE,
        "forecast_horizon": "12 hours",
    }


@app.get("/api/dataset/sample")
async def dataset_sample(
    split: str = Query(default="TEST", pattern="^(TRAIN|VAL|TEST)$"),
    offset: int = Query(default=0, ge=0),
):
    if not DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail="Bundled 5-minute dataset is missing")
    dataset = pd.read_csv(DATASET_PATH)
    subset = dataset.loc[dataset["split"] == split]
    pond_windows = []
    for pond_name, pond_rows in subset.groupby("pond", sort=True):
        pond_rows = pond_rows.assign(
            _timestamp=pd.to_datetime(pond_rows["datetime"], utc=True)
        ).sort_values("_timestamp").reset_index(drop=True)
        run_ids = pond_rows["_timestamp"].diff().ne(pd.Timedelta(minutes=5)).cumsum()
        for _, continuous_rows in pond_rows.groupby(run_ids, sort=False):
            window_count = len(continuous_rows) - WINDOW_SIZE + 1
            if window_count > 0:
                pond_windows.append((str(pond_name), continuous_rows, window_count))
    total_windows = sum(window_count for _, _, window_count in pond_windows)
    if total_windows == 0:
        raise HTTPException(status_code=422, detail=f"Not enough {split} rows for a full window")

    selected = offset % total_windows
    pond_name = ""
    start = 0
    for candidate_pond, pond_rows, window_count in pond_windows:
        if selected < window_count:
            pond_name = candidate_pond
            start = selected
            window = pond_rows.iloc[start:start + WINDOW_SIZE]
            break
        selected -= window_count

    readings = window[FEATURE_NAMES].astype(float).to_dict(orient="records")
    last_row = window.iloc[-1]
    return {
        "split": split,
        "pond": pond_name,
        "offset": start,
        "start_time": window.iloc[0]["_timestamp"].isoformat(),
        "end_time": last_row["_timestamp"].isoformat(),
        "expected_label": str(last_row.get("target_risk_category_12h", "Unknown")),
        "readings": readings,
    }


@app.get("/api/dataset/download")
async def download_dataset():
    if not DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail="Bundled 5-minute dataset is missing")
    return FileResponse(DATASET_PATH, media_type="text/csv", filename=DATASET_PATH.name)


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model assets are not loaded")
    if len(payload.readings) not in (1, WINDOW_SIZE):
        raise HTTPException(status_code=422, detail="Send one reading or exactly 144 five-minute readings")

    rows = np.asarray(
        [[getattr(reading, feature) for feature in FEATURE_NAMES] for reading in payload.readings],
        dtype=np.float32,
    )
    try:
        return predict_readings(rows)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}") from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8005, reload=True)