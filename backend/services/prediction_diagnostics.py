"""Diagnostics telemetry for prediction-mode ESP32 packets."""

import asyncio
import time
from collections import deque
from typing import Optional


_RATE_WINDOW_S = 2.0


class PredictionDiagnostics:
    """Tracks packet rate, schema health, liveness, and recent packets."""

    def __init__(self):
        self._recent_packets: deque = deque(maxlen=200)
        self._packet_rate_window: dict[int, deque] = {1: deque(), 2: deque()}
        self._schema_stats: dict[int, dict] = {
            1: {"valid": 0, "invalid": 0, "last_error": None},
            2: {"valid": 0, "invalid": 0, "last_error": None},
        }
        self._last_seen: dict[int, float] = {}
        self._subscribers: set[asyncio.Queue] = set()

    def _validate_packet(self, packet: dict) -> tuple[bool, Optional[str]]:
        if not isinstance(packet, dict):
            return False, "not a dict"
        if packet.get("hand_id") not in (1, 2):
            return False, f"invalid hand_id: {packet.get('hand_id')}"
        if not isinstance(packet.get("timestamp_ms"), (int, float)):
            return False, "timestamp_ms missing or not numeric"
        flex = packet.get("flex")
        if not (isinstance(flex, list) and len(flex) == 5 and all(isinstance(v, (int, float)) for v in flex)):
            return False, f"flex must be list[5] numbers, got: {flex!r}"
        fsr = packet.get("fsr")
        if not (isinstance(fsr, list) and len(fsr) == 3 and all(isinstance(v, (int, float)) for v in fsr)):
            return False, f"fsr must be list[3] numbers, got: {fsr!r}"
        quat = packet.get("quaternion")
        if not (isinstance(quat, list) and len(quat) == 4 and all(isinstance(v, (int, float)) for v in quat)):
            return False, f"quaternion must be list[4] numbers, got: {quat!r}"
        return True, None

    def ingest_packet(self, packet: dict) -> tuple[bool, Optional[str]]:
        """Record one incoming prediction packet and publish diagnostics."""
        now = time.time()
        hand_id = packet.get("hand_id") if isinstance(packet, dict) else None
        valid, error = self._validate_packet(packet)

        if hand_id in self._schema_stats:
            if valid:
                self._schema_stats[hand_id]["valid"] += 1
            else:
                self._schema_stats[hand_id]["invalid"] += 1
                self._schema_stats[hand_id]["last_error"] = error
            self._last_seen[hand_id] = now
            win = self._packet_rate_window[hand_id]
            win.append(now)
            while win and now - win[0] > _RATE_WINDOW_S:
                win.popleft()

        msg = {"type": "packet", "packet": packet, "valid": valid, "error": error, "ts": now}
        self._recent_packets.append(msg)
        for q in list(self._subscribers):
            try:
                q.put_nowait(msg)
            except Exception:
                pass
        return valid, error

    def get_diagnostics(self, server_lan_ips: list[str] = None) -> dict:
        now = time.time()
        observed_hz = {}
        last_seen_age_ms = {}
        for hand_id in (1, 2):
            win = self._packet_rate_window.get(hand_id, deque())
            valid_ts = [t for t in win if now - t <= _RATE_WINDOW_S]
            observed_hz[hand_id] = round(len(valid_ts) / _RATE_WINDOW_S, 1)
            last = self._last_seen.get(hand_id)
            last_seen_age_ms[hand_id] = round((now - last) * 1000) if last else None
        return {
            "endpoint": "predict",
            "ws_url_hint": "ws://<LAN-IP>:8000/ws/predict",
            "server_lan_ips": server_lan_ips or [],
            "expected_hz": 50,
            "observed_hz": observed_hz,
            "last_seen_age_ms": last_seen_age_ms,
            "schema": {
                str(hid): dict(stats) for hid, stats in self._schema_stats.items()
            },
            "active_session": None,
            "server_time_ms": int(now * 1000),
        }

    def add_subscriber(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._subscribers.add(q)
        return q

    def remove_subscriber(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

