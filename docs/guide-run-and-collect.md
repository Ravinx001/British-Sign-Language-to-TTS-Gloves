# Running the App & Collecting Data — Step-by-Step Guide

This guide walks you from a fresh checkout to a clean, retrainable dataset. It assumes the backend is at `c:\laragon\www\sign_gloves_ML_models` and you have two ESP32 gloves flashed with the matching firmware.

---

## 1. One-time setup

### 1.1 Create the Python venv

```powershell
cd c:\laragon\www\sign_gloves_ML_models
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

### 1.2 Find your PC's LAN IP

The ESP32 gloves connect to your PC over Wi-Fi. They need the PC's LAN IP — `127.0.0.1` and `localhost` will not work.

```powershell
ipconfig | Select-String "IPv4"
```

Pick the address on the same Wi-Fi network as the gloves (typically `192.168.x.x` or `10.0.x.x`). Write it down.

### 1.3 Open the firewall port

Run **once** as Administrator:

```powershell
New-NetFirewallRule -DisplayName "BSL Gloves uvicorn" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

### 1.4 Flash the gloves

In your ESP32 firmware, set:

```cpp
const char* WS_HOST = "192.168.x.x";   // your PC's LAN IP from 1.2
const uint16_t WS_PORT = 8000;
const char* WS_PATH = "/ws/collect";   // for collection
// or "/ws/predict" for live recognition
```

Right glove must send `hand_id: 1`, left glove `hand_id: 2`.

---

## 2. Start the backend

```powershell
cd c:\laragon\www\sign_gloves_ML_models
.venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

> ⚠ **`--host 0.0.0.0` is required** — without it, the gloves can't reach the server. `127.0.0.1` only listens on localhost.

You should see startup logs ending in:

```
Collection data: ...\data\raw\real (N user(s) in users.json)
LAN IPs for ESP32: 192.168.x.x
Models loaded. Server ready.
```

Open in your browser:
- **Data collection:** http://localhost:8000/collection
- **Live recognition:** http://localhost:8000/

Both pages have R/L glove pills in the top-right — **click either pill** to open the diagnostics modal.

---

## 3. Verify gloves are connected (use the diagnostics modal)

This is the most important sanity check. Before recording anything, click a glove pill on the `/collection` page.

What to look for:

| Section | Healthy values |
|---|---|
| **WebSocket URL** | `ws://<your-LAN-IP>:8000/ws/collect` |
| **Server LAN IPs** | Lists at least the IP your gloves are pointed at |
| **Right Glove · Hz** | ~50 Hz (green). 25–45 = yellow (warn). <25 = red (bad) |
| **Left Glove · Hz** | Same |
| **Last seen** | <100 ms ago for both |
| **Schema** | ✓ counter climbing, ✗ at 0 |

The **Live Packet Viewer** at the bottom shows raw frames as they arrive. Each line looks like:

```
[R] ts=104523 flex=[1850,820,3680,3740,3700] ✓
[L] ts=104530 flex=[800,800,800,800,800] ✓
```

Use **Pause** to freeze the log so you can read it. Use the hand filter to isolate one glove.

### If a glove won't connect

Walk down the troubleshooting checklist in the modal:

1. Same Wi-Fi network on PC + gloves.
2. uvicorn started with `--host 0.0.0.0`.
3. ESP32 firmware `WS_HOST` matches a LAN IP shown in the modal.
4. Windows Firewall allows TCP 8000.
5. Glove powered on, ESP32 serial monitor says "WS connected".
6. If Hz is low (< 25) — power issue (USB drop, low battery).
7. If `✗` count is climbing — schema mismatch. Inspect a packet in the live viewer and compare field names with what the backend expects (`flex`, `fsr`, `quaternion`, `accel`, `gyro`, `timestamp_ms`, `hand_id`).

---

## 4. Create users

On the `/collection` page, top-left **Recorder** card:

1. Click **+ New**.
2. Type a full name (e.g. "Madura C").
3. Click **Add**.

The user is saved and a folder gets created on disk:

```
data/raw/real/madura-c-a1b2c3d4/
```

The slug-prefixed folder name lets you find that user's data by browsing the disk. Existing users from before this update are auto-renamed on the next backend restart.

---

## 5. Record sessions

1. Pick the user from the dropdown.
2. Click a letter in the **Select Sign** grid (or press the keyboard letter).
3. Click **▶ Start Recording** — there's a 3-2-1 countdown.
4. Hold the sign for ~2 seconds.
5. Click **■ Stop**.

The session is saved to:

```
data/raw/real/<slug-uuid>/<LETTER>/<LETTER>_<uuid>_<timestamp>.jsonl
```

The right panel updates immediately:
- The progress tile for that letter increments.
- The **Sessions** table below the grid shows the new session with **Preview / Trim / Delete** buttons.

### Recording targets

The default goal is **50 sessions per sign**. Tile colors:

- ⚫ none — 0
- 🔴 low — 1–14
- 🟡 mid — 15–29
- 🟢 good — 30–49
- 💚 full — 50+

Adjust the goal slider if you want a different target.

### Dynamic signs (J, Z)

Marked with ★. These need **motion** during the 2-second window — sign the actual letter motion, don't hold a static pose.

---

## 6. Clean up bad data

This is what makes the difference between a usable training set and a noisy one. Three levels of delete:

### 6.1 Delete one session

In the Sessions table, click 🗑 next to a row. Confirms, then deletes the file and decrements the count.

### 6.2 Delete all sessions for one sign (per user)

Hover over a progress tile → click the **⋯** kebab in the top-right → **🗑 Delete all (N)**. Confirms with the count and user name.

### 6.3 Delete an entire user

Select the user → click the **⚙** gear next to the dropdown → **🗑 Delete user**. You'll be asked to type the user's name to confirm. This wipes the user's folder and removes them from `users.json`.

> All deletes are blocked while that user/sign/session is actively recording — you'll get a 409 error. Stop recording first.

---

## 7. Trim the start/end of a session

The first ~0.3 s and last ~0.5 s of every recording are usually noise — your hand reaching for the Stop button, etc. To trim:

1. In the Sessions table, click **✂ Trim** on the row.
2. The preview modal shows a sparkline of the right (blue) and left (green) flex averages over time.
3. Drag the **Start** and **End** sliders. Yellow marker lines appear on the sparkline; the trimmed regions are shaded red.
4. Watch the "Keeping frames N – M (X frames)" readout below the sliders.
5. Click **✂ Apply Trim**. Confirms, then atomically rewrites the JSONL.

The session count in the manifest is **not** affected — it's still one session, just shorter. Run trim multiple times if you want to nibble more off the ends.

---

## 8. Recommended collection workflow

Practical sequence to build a good dataset:

1. Open `/collection`, verify both gloves green in the diagnostics modal.
2. Create your user.
3. Pick letter **A**. Record 5 sessions back-to-back without removing the gloves.
4. Click **A** in the progress grid → review the 5 sessions table → preview each one to spot-check signal quality. Trim or delete obvious junk.
5. Repeat for B, C, D, ... Z.
6. Once you have ~30 sessions per static sign and ~40 per dynamic sign (J, Z), you're ready to retrain.
7. Have a second person create their own user and repeat — diverse recorders make the model generalize better.

### Per-session quality heuristics

In the preview sparkline, a healthy 2 s session looks like:
- A clear "settle" plateau in the middle with low jitter.
- For static signs: flat lines for ~80% of the duration.
- For dynamic signs (J/Z): a clear motion arc, then a settle.

If the lines are flatlined at the same value for the entire window, the glove probably wasn't streaming — check the diagnostics modal and re-record.

---

## 9. Retrain the models

After you've collected enough data, the existing trainers in `ml/` work without changes. The on-disk format is identical to before.

```powershell
cd c:\laragon\www\sign_gloves_ML_models
.venv\Scripts\Activate.ps1
python -m ml.train_static    # XGBoost / RF / SVM for 24 static letters
python -m ml.train_dynamic   # CNN-LSTM for J, Z
```

Restart the backend after training so the new model files are loaded:

```
Ctrl+C (in the uvicorn terminal)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 10. Live recognition

Once retrained, point your firmware to `WS_PATH = "/ws/predict"` and open http://localhost:8000/. Sign with the gloves; predictions appear in the **Live Feed** tab.

Both pages share the same `/ws/diagnostics` data source, so the gloves diagnostics modal works identically on the live page — click the R/L pills any time to see Hz, schema validation, and the live packet stream.

---

## Quick reference

| Task | Where |
|---|---|
| Verify gloves are connected | Click R/L pill on either page |
| Add a user | `/collection` → Recorder → + New |
| Delete a user (and all their data) | `/collection` → Recorder → ⚙ → Delete user |
| Delete all of one sign for a user | Hover progress tile → ⋯ → Delete all |
| Delete one session | Sessions table → 🗑 |
| Preview a session | Sessions table → 👁 Preview |
| Trim a session | Sessions table → ✂ Trim |
| See packet rate / schema errors | Diagnostics modal |
| Live raw packets | Diagnostics modal → Live Packet Viewer |

---

## Where files live

```
data/raw/real/
├── users.json              # user list (now includes folder_name field)
├── manifest.json           # session counts: {user_id: {label: count}}
└── <slug-uuid>/            # one folder per user
    ├── A/
    │   ├── A_a1b2c3d4_1715234567890.jsonl
    │   └── ...
    ├── B/
    └── ...
```

Each `.jsonl` line is one frame from one hand. Right (hand_id=1) and left (hand_id=2) frames alternate, paired by timestamp on the server.
