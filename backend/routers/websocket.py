"""WebSocket endpoints for ESP32 glove streaming and dashboard broadcast."""

import asyncio
import json
import time
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import numpy as np

router = APIRouter()

# Frame buffer to pair right + left hand packets by timestamp. The two ESP32s
# do not always arrive on the server in the same event-loop tick, so prediction
# must wait briefly for the opposite hand instead of immediately fabricating a
# neutral missing-hand packet.
frame_buffer: dict[int, dict] = {}
frame_received: dict[int, float] = {}  # server receive time per timestamp key
FRAME_PAIR_TOLERANCE_MS = 300
FRAME_BUFFER_MAX_AGE_MS = 500  # Discard unpaired frames older than 500ms

# Serializes ML inference calls across all WS handlers. SignPredictor has
# internal state (motion segmenter buffer, debounce history, dynamic latch)
# that is NOT thread-safe — but inference itself (especially CNN-LSTM at
# 20-50ms on CPU) blocks the asyncio loop if called synchronously, causing
# ESP32 WebSocket disconnects on dynamic predictions. The lock +
# asyncio.to_thread combo runs each call on a worker thread (event loop
# stays responsive for heartbeats and other gloves' incoming frames) while
# the lock guarantees only one call mutates predictor state at a time.
_inference_lock = asyncio.Lock()


async def _predict_async(service, *args):
    """Run service.predict_from_raw in a thread under the inference lock."""
    async with _inference_lock:
        return await asyncio.to_thread(service.predict_from_raw, *args)


def _normalize_hand_id(value) -> int | None:
    """Return hand_id as 1/2, accepting numeric strings from firmware."""
    try:
        hand_id = int(value)
    except (TypeError, ValueError):
        return None
    return hand_id if hand_id in (1, 2) else None


def _coerce_timestamp_ms(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(time.time() * 1000)


def _find_pair_key(timestamp_ms: int, hand_id: int) -> int | None:
    for existing_ts, hands in frame_buffer.items():
        if hand_id not in hands and abs(existing_ts - timestamp_ms) <= FRAME_PAIR_TOLERANCE_MS:
            return existing_ts
    return None


def _esp32_prediction_payload(result: dict, motion_energy: float) -> dict:
    """Small response for ESP32 firmware; dashboard gets the full payload."""
    confidence = result.get("confidence")
    return {
        "type": "prediction",
        "sign": result.get("sign") or "Unknown",
        "confidence": confidence if confidence is not None else 0.0,
        "route": result.get("route") or "unknown",
        "motion_energy": round(motion_energy, 3),
    }


@router.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """Accept WebSocket connection from ESP32 gloves.

    Expects JSON packets at 50 Hz per glove. Pairs packets by timestamp,
    runs prediction, sends results back, and broadcasts to dashboard clients.

    Packet format:
    {
        "timestamp_ms": 104523,
        "hand_id": 1,
        "flex": [5 values],
        "fsr": [3 values],
        "contact_bitmask": int,
        "quaternion": [w, x, y, z],
        "accel": [x, y, z],
        "gyro": [x, y, z],
        "battery_pct": 87.5
    }
    """
    await websocket.accept()

    service = websocket.app.state.prediction_service
    cm = websocket.app.state.connection_manager
    diagnostics = websocket.app.state.prediction_diagnostics

    socket_id = str(uuid.uuid4())[:8]
    registered_hands: dict[int, str] = {}

    # ESP32 dedupe: only forward to the device when the emitted sign actually
    # changes. Avoids spamming the ESP32's RX queue during held letters and
    # the dynamic-latch window (12 identical dynamic frames in 240ms). The
    # dashboard still gets every frame via cm.broadcast_prediction.
    last_emitted_sign: str | None = None
    # Heartbeat: track when we last sent any data to this ESP32. During the
    # ~2.2s dynamic collection window all predictions are Unknown — nothing is sent
    # to the device. If we're silent for >1s, push a tiny keepalive so the TCP
    # connection stays active and the ESP32 does not time out and reconnect.
    last_sent_ms: float = 0.0

    try:
        while True:
            raw = await websocket.receive_text()
            packet = json.loads(raw)

            hand_id = _normalize_hand_id(packet.get("hand_id", 1))
            if hand_id is None:
                diagnostics.ingest_packet(packet)
                continue
            packet["hand_id"] = hand_id
            diagnostics.ingest_packet(packet)

            timestamp_ms = _coerce_timestamp_ms(
                packet.get("timestamp_ms", int(time.time() * 1000))
            )

            if hand_id not in registered_hands:
                client_id = f"{socket_id}-{hand_id}"
                await cm.connect_esp32(websocket, client_id, hand_id, endpoint="predict")
                registered_hands[hand_id] = client_id

            cm.update_esp32(registered_hands[hand_id], packet)

            # Clean old frames based on server receive time (not ESP32 clock)
            now = time.time()
            stale = [
                ts for ts in frame_buffer
                if now - frame_received.get(ts, 0) > FRAME_BUFFER_MAX_AGE_MS / 1000
            ]
            for ts in stale:
                frame_buffer.pop(ts, None)
                frame_received.pop(ts, None)

            # Pair with the opposite hand using timestamp tolerance. This mirrors
            # collection-mode pairing and prevents the live model from seeing a
            # real glove plus a fake neutral glove on every packet.
            pair_key = _find_pair_key(timestamp_ms, hand_id)
            if pair_key is None:
                pair_key = timestamp_ms
                frame_buffer[pair_key] = {hand_id: packet}
                frame_received[pair_key] = now
            else:
                frame_buffer[pair_key][hand_id] = packet

            # Check if we have both hands for this timestamp window.
            frame = frame_buffer.get(pair_key, {})
            if 1 in frame and 2 in frame:
                right_pkt = frame[1]
                left_pkt = frame[2]
                frame_buffer.pop(pair_key, None)
                frame_received.pop(pair_key, None)

                result = await _predict_async(service, right_pkt, left_pkt)

                # Compute motion energy for dashboard
                r_accel = np.asarray(right_pkt.get("accel", [0, 0, 9.8]))
                me = float(np.linalg.norm(r_accel - np.array([0, 0, 9.81])))

                now_ms = time.time() * 1000
                sign = result.get("sign")
                if sign in (None, "Unknown"):
                    # Reset so the next non-Unknown sign re-emits even if same letter.
                    last_emitted_sign = None
                elif sign != last_emitted_sign:
                    await websocket.send_json(_esp32_prediction_payload(result, me))
                    last_emitted_sign = sign
                    last_sent_ms = now_ms
                # else: same letter being held — skip ESP32 send.

                # Heartbeat: keep TCP alive during dynamic collection window (all Unknown)
                if now_ms - last_sent_ms > 1000:
                    try:
                        await websocket.send_json({"type": "collecting"})
                        last_sent_ms = now_ms
                    except Exception:
                        pass

                await cm.broadcast_prediction(result, me)

            # Otherwise wait for the other glove. Stale one-hand frames are
            # discarded by the cleanup above; these deployed models are
            # dual-glove models, so a fake neutral hand is more harmful than
            # silence.

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error ({socket_id}): {e}")
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        for client_id in registered_hands.values():
            await cm.disconnect_esp32(client_id)


@router.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """Browser dashboard WebSocket — receives live prediction broadcasts.

    On connect, receives current state:
    {"type": "init", "connections": [...]}

    Then receives streaming messages:
    {"type": "prediction", "sign": "A", "confidence": 0.94, ...}
    {"type": "connection", "event": "connected"|"disconnected", "client": {...}}
    {"type": "settings_changed", "settings": {...}}
    """
    cm = websocket.app.state.connection_manager
    await cm.connect_dashboard(websocket)

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            action = msg.get("action")

            if action == "get_connections":
                await websocket.send_json({
                    "type": "connections",
                    "connections": cm.get_all_connections(),
                })
            elif action == "get_metrics":
                service = websocket.app.state.prediction_service
                metrics = service.get_metrics()
                metrics["type"] = "metrics"
                await websocket.send_json(metrics)

    except WebSocketDisconnect:
        cm.disconnect_dashboard(websocket)
    except Exception:
        cm.disconnect_dashboard(websocket)
