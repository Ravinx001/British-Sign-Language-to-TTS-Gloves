# Plan: BSL Gloves FastAPI Backend

Build a FastAPI backend wrapping the existing ML pipeline (XGBoost static + CNN-LSTM dynamic) to accept ESP32 WebSocket streams, manual REST input, and batch file uploads — with text + voice output and a real-time dashboard. Six phases, from scaffolding to full SPA.

## Current State

- ML pipeline **complete**: `SignPredictor`, feature engineering, segmentation, config — all in `ml/`
- Models **trained**: XGBoost (97.38% acc), CNN-LSTM (100% synthetic), scaler fitted
- **No `backend/` directory** — building from scratch
- **No `requirements.txt`** at project root
- **Python 3.14** — some packages (`pyttsx3`) may need compatibility checks
- `SignPredictor.predict()` returns `(str, float)` **tuple**, not dict — service wrapper needed
- Thresholds are **module-level constants**, not instance attributes — wrapper needed for runtime config
- No `predict_batch()` on `SignPredictor` — must build in service layer

---

## Phase 1: Project Scaffolding & Core Service Layer

**Goal**: Directory structure, dependencies, `PredictionService` wrapping `SignPredictor`.

1. Create `backend/` structure: `main.py`, `routers/`, `services/`, `models/`, `static/`, `static/audio/`
2. Create `backend/requirements.txt` — `fastapi`, `uvicorn[standard]`, `python-multipart`, `websockets`, `jinja2`, `aiofiles`, `numpy`, `torch`, `xgboost`, `scikit-learn`, `scipy` (TTS deferred to Phase 4)
3. Create `backend/services/predictor.py` — `PredictionService`:
   - Init wraps `SignPredictor` from `ml/predict.py`
   - `predict_from_features(features, accel)` → wraps `predictor.predict()` tuple into dict `{sign, confidence, latency_ms, route}`
   - `predict_from_raw(right_packet, left_packet)` → converts raw sensor dicts to 36-feature vector using `extract_features()` from `ml/feature_engineering.py`, then predicts
   - `predict_batch(X, y_true)` → iterate rows, compute accuracy if `y` provided
   - Track metrics: total predictions, latency sum, confidence accumulator
   - **Key**: `SignPredictor.predict()` returns `(sign, confidence)` — must wrap. Route detection requires checking segmenter state.
4. Create `backend/models/schemas.py` — Pydantic models: `RawSensorInput`, `FeatureInput`, `PredictionResponse`
5. Create minimal `backend/main.py` — FastAPI app with lifespan loading `PredictionService`, `sys.path` patched to import `ml` package

**Verification**: Server starts without errors; `PredictionService.predict_from_features()` works with a sample from `data/processed/test.npz`

### Relevant Files

- `ml/predict.py` — `SignPredictor.__init__()`, `predict(feature_vec, accel_xyz)` returns `(str, float)`
- `ml/feature_engineering.py` — `extract_features()`, `normalize_flex()`, `normalize_fsr()`, `quaternion_to_euler()`
- `ml/config.py` — `FEATURE_NAMES`, `IDX_TO_LETTER`, all thresholds, `PATHS`, `NUM_FEATURES`
- `ml/segmentation.py` — `MotionSegmenter`, `get_current_energy()`

---

## Phase 2: REST Prediction Endpoints

**Goal**: Three REST prediction routes + metrics/reset.

*Depends on Phase 1*

1. Create `backend/routers/predict.py`:
   - `POST /api/predict/raw` — accepts `RawSensorInput`, calls `service.predict_from_raw()`
   - `POST /api/predict/features` — accepts `FeatureInput`, calls `service.predict_from_features()`
   - `GET /api/metrics` — returns `service.get_metrics()`
   - `POST /api/reset` — calls `service.reset()`
2. Create `backend/routers/batch.py`:
   - `POST /api/predict/batch` — file upload (`.npz` or `.csv`), validate 36 columns
   - Return `{total_samples, accuracy, avg_latency_ms, predictions[]}`
3. Register routers in `main.py`

**Verification**: Swagger at `/docs` shows all endpoints; `POST /api/predict/batch` with `test.npz` returns ~97% accuracy

### Relevant Files

- `backend/services/predictor.py` — `predict_from_raw()`, `predict_from_features()`, `predict_batch()`
- `backend/models/schemas.py` — `RawSensorInput`, `FeatureInput`, `PredictionResponse`
- `data/processed/test.npz` — test data for verification

---

## Phase 3: WebSocket — ESP32 Streaming & Dashboard Broadcast

**Goal**: Real-time WebSocket for ESP32 gloves + dashboard broadcast channel.

*Depends on Phase 1*

1. Create `backend/services/connection_manager.py`:
   - `ESP32Client` dataclass: `client_id`, `hand_id`, `ip`, timestamps, `packet_count`, `avg_latency_ms`, `battery_pct`
   - `ConnectionManager`: `connect_esp32()`, `disconnect_esp32()`, `update_esp32()`, `broadcast_to_dashboards()`, `broadcast_prediction()`
2. Create `backend/routers/websocket.py`:
   - `WS /ws/predict` — ESP32 endpoint: accept JSON at 50Hz, pair R/L frames by timestamp, predict, respond, broadcast
   - `WS /ws/dashboard` — browser endpoint: send `init` message, then stream prediction/connection events
3. Add `ConnectionManager` to `app.state` in `main.py` lifespan

**Verification**: `wscat` to `/ws/predict` + send JSON → get prediction back; second WS to `/ws/dashboard` receives broadcast

### Relevant Files

- `ml/segmentation.py` — `MotionSegmenter` (used per-connection for motion tracking)
- `ml/predict.py` — `SignPredictor.predict(feature_vec, accel_xyz)`
- `backend/services/predictor.py` — `predict_from_raw()`

---

## Phase 4: Text-to-Speech & Audio Serving

**Goal**: Server-side TTS + audio file endpoint.

*Parallel with Phase 3*

1. Create `backend/services/tts_service.py`:
   - `TTSService` using `pyttsx3` (offline, thread-safe with `_lock`)
   - `speak(sign)` → generates `static/audio/latest.wav` in background thread
   - `set_rate()`, `set_volume()`, `set_speaker_enabled()`, `get_settings()`
   - **Fallback**: If `pyttsx3` doesn't work on Python 3.14, use `gtts` (requires internet) or defer (browser Web Speech API is primary)
2. Create `backend/routers/audio.py`:
   - `GET /api/audio/latest` → `FileResponse` for latest `.wav`/`.mp3`
3. Add TTS calls to prediction endpoints in `routers/predict.py`
4. Add `pyttsx3` (and optionally `gtts`) to `requirements.txt`

**Verification**: `POST /api/predict/features` → `audio_url` in response; `GET /api/audio/latest` returns audio file

### Relevant Files

- `backend/static/audio/` — output directory for generated audio files
- `backend/routers/predict.py` — add TTS call after prediction

---

## Phase 5: Settings & Connection Management API

**Goal**: Runtime-configurable thresholds, TTS settings, connection management.

*Depends on Phases 3 + 4*

1. Create `backend/routers/settings.py`:
   - `GET /api/settings` — current thresholds, TTS config, model info
   - `PUT /api/settings` — accept partial updates, apply to `PredictionService` and `TTSService`
   - **Key adaptation**: `SignPredictor` uses module constants, not instance attrs. `PredictionService` must store runtime overrides and apply them during prediction.
   - Broadcast `settings_changed` to all dashboard WS clients
2. Add connection endpoints:
   - `GET /api/connections` — all ESP32 clients from `ConnectionManager`
   - `POST /api/connections/{client_id}/disconnect` — force-close a glove
3. Add `SettingsUpdate`, `SettingsResponse`, `ConnectionInfo`, `ConnectionsResponse` to `schemas.py`

**Verification**: `PUT /api/settings {static_confidence_threshold: 0.5}` → subsequent predictions use new threshold; change broadcasts to `/ws/dashboard`

### Relevant Files

- `ml/config.py` — default threshold values (read-only reference)
- `backend/services/predictor.py` — must support overridable thresholds
- `backend/services/connection_manager.py` — `get_all_connections()`

---

## Phase 6: Web Dashboard SPA

**Goal**: Single-page dashboard with 5 tabs — vanilla HTML/JS, Chart.js (CDN).

*Depends on all prior phases*

1. Create `backend/static/index.html` — full SPA:
   - **Live Feed**: current sign, confidence line chart (Chart.js, last 100 pts), letter frequency bar, motion energy gauge, prediction history, TTS toggle, metrics bar
   - **Manual Input**: raw sensors form (pre-filled defaults), 36-feature textarea, result card
   - **Batch Upload**: file upload (.npz/.csv), results card
   - **Settings**: threshold sliders, TTS rate/volume, speaker toggle, model info table, Apply/Reset buttons
   - **Connections**: glove table (hand, IP, latency, battery, disconnect button), pairing instructions
2. WebSocket client: connect `/ws/dashboard`, handle all message types, auto-reconnect (3s)
3. Chart.js (CDN): confidence line + frequency bar, no-animation mode for perf
4. Web Speech API for browser TTS (primary, zero-latency) with server fallback
5. Mount static files and serve `index.html` at `/` in `main.py`

**Verification**: Dashboard at `http://localhost:8000` loads all 5 tabs; manual input returns predictions; batch upload works; settings round-trip; charts render

### Relevant Files

- All `backend/routers/*.py` — endpoints consumed by dashboard JS
- `backend/services/connection_manager.py` — WebSocket message protocol
- `docs/new-docs/guide-backend-integration.md` — full reference architecture & HTML template

---

## Dependency Graph

```
Phase 1 ──→ Phase 2 ──────────→ Phase 5 ──→ Phase 6
   │                                ↑
   ├──→ Phase 3 ───────────────────┘
   │
   └──→ Phase 4 (parallel with 3) ─┘
```

---

## Decisions

- **Vanilla HTML/JS** (no React build step) — simpler deployment, matches blueprint
- **In-memory state only** — no database, metrics reset on restart
- **Settings are ephemeral** — revert to `ml/config.py` defaults on restart
- **No authentication** — local network only (ESP32 + dev machine)
- **Browser Web Speech API is primary TTS** — server TTS is fallback only

## Further Considerations

1. **`pyttsx3` on Python 3.14** — may not be compatible. Test early in Phase 4; if it fails, use `gtts` (online) or skip server TTS entirely since browser Web Speech API is the primary TTS path.
2. **`scaler.pkl` existence** — config references `data/processed/scaler.pkl` but it wasn't visible in directory listing. Verify it exists before Phase 1 testing; if missing, re-run `ml/train_static.py` to regenerate.

## Excluded

- Production deployment (Docker, HTTPS, auth)
- Database persistence for prediction history
- Multi-worker scaling
- ESP32 firmware (reference sketch only in guide)
