# AquaGuard 5-minute project

Standalone 5-minute BiLSTM risk-prediction app. This folder contains only the 5-minute model, its feature scaler, the 5-minute dataset exports, and an independent React dashboard/API.

## Model and data

- Sampling interval: 5 minutes
- Input window: 144 readings (12 hours)
- Forecast horizon: 144 steps (12 hours)
- Features, in order: DO, temperature, pH, turbidity, ammonia, fish weight
- Dataset: `dataset/aquaponics_5min_complete.csv`, with train, validation, and test CSV splits alongside it
- Model and scaler: `model/`

The exported CSVs are included for inspection, download, and test-window inference. The original `.npy` sliding-window arrays are deliberately not duplicated because the app does not require them at runtime; they remain available in the parent repository's `archive_5min/data_processed/` if retraining is needed.

## Run locally

Requires Python 3.10+ and Node.js 20.19+ (or 22.12+ for Vite 8).

Start the API in one terminal:

```powershell
cd aquaguard-5min\api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

Start the dashboard in another terminal:

```powershell
cd aquaguard-5min\frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal (normally `http://localhost:5173`). The API is available at `http://127.0.0.1:8005`.

## Smoke test

After installing the API requirements, run:

```powershell
cd aquaguard-5min\api
python -m unittest smoke_test.py
```

## GitHub as a separate repository

From the project root, initialize and push this folder as its own repository after creating an empty GitHub repository:

```powershell
cd aquaguard-5min
git init
git add .
git commit -m "Create standalone AquaGuard 5-minute project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```