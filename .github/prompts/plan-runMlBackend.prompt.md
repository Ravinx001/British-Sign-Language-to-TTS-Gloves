# Running the ML Model & Backend Together

The backend already loads and wraps the ML models on startup. One command runs everything:

```powershell
cd c:\laragon\www\sign_gloves_ML_models
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

This:
1. Loads the XGBoost static model + CNN-LSTM dynamic model + scaler via `PredictionService`
2. Initializes TTS, WebSocket manager, and all API routes
3. Serves the dashboard at **http://127.0.0.1:8000**

## Available Interfaces

| Interface | URL |
|-----------|-----|
| Dashboard SPA | `http://127.0.0.1:8000` |
| API Docs (Swagger) | `http://127.0.0.1:8000/docs` |
| REST Prediction | `POST /api/predict/raw` or `/api/predict/features` |
| Batch Prediction | `POST /api/predict/batch` |
| WebSocket (ESP32) | `ws://127.0.0.1:8000/ws/predict` |
| WebSocket (Dashboard) | `ws://127.0.0.1:8000/ws/dashboard` |
| Settings | `GET/PUT /api/settings` |

## Expose on Local Network (for ESP32 Gloves)

```powershell
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Then ESP32 gloves connect to `ws://<your-PC-IP>:8000/ws/predict`.
