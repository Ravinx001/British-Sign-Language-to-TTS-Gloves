"""Data collection service: session management, JSONL writing, frame pairing, and diagnostics."""

import asyncio
import json
import logging
import math
import os
import re
import shutil
import time
import unicodedata
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ml.config import REST_LABELS, REST_POSES

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "real"
USERS_FILE = DATA_DIR / "users.json"
MANIFEST_FILE = DATA_DIR / "manifest.json"

FRAME_BUFFER_MAX_AGE_S = 0.30

BSL_SIGNS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
REST_SIGNS = list(REST_LABELS)
COLLECTION_LABELS = BSL_SIGNS + REST_SIGNS

# Packet rate measurement window in seconds
_RATE_WINDOW_S = 2.0


def _slugify(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^\w\s-]", "", name.lower())
    name = re.sub(r"[\s_]+", "-", name).strip("-")
    return name[:30] or "user"


def _vector_magnitude(values) -> float:
    if not isinstance(values, list):
        return 0.0
    nums = []
    for value in values[:3]:
        if isinstance(value, (int, float)):
            nums.append(float(value))
    if not nums:
        return 0.0
    return round(math.sqrt(sum(value * value for value in nums)), 3)


class CollectionService:
    """Manages recording sessions, frame pairing, JSONL persistence, and diagnostics."""

    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._session_lock = asyncio.Lock()

        # Active session
        self._session_id: Optional[str] = None
        self._session_user_id: Optional[str] = None
        self._session_label: Optional[str] = None
        self._session_file = None
        self._session_frame_count: int = 0
        self._session_start_time: Optional[float] = None

        # Frame pairing buffer
        self._frame_buffer: dict[int, dict] = {}
        self._frame_received: dict[int, float] = {}

        # Glove liveness
        self._glove_last_seen: dict[int, float] = {}

        # Diagnostics telemetry
        self._recent_packets: deque = deque(maxlen=200)
        self._packet_rate_window: dict[int, deque] = {1: deque(), 2: deque()}
        self._schema_stats: dict[int, dict] = {
            1: {"valid": 0, "invalid": 0, "last_error": None},
            2: {"valid": 0, "invalid": 0, "last_error": None},
        }
        self._diag_subscribers: set = set()

        # Persisted state
        self._users: dict[str, dict] = {}
        self._manifest: dict[str, dict[str, int]] = {}
        self._load_users()
        self._load_manifest()
        self._migrate_user_folders()

    # ------------------------------------------------------------------ #
    # Persistence helpers                                                  #
    # ------------------------------------------------------------------ #

    def _load_users(self):
        try:
            if USERS_FILE.exists():
                self._users = json.loads(USERS_FILE.read_text(encoding="utf-8"))
        except Exception:
            self._users = {}

    def _save_users(self):
        try:
            USERS_FILE.write_text(json.dumps(self._users, indent=2), encoding="utf-8")
        except OSError as e:
            raise RuntimeError(f"Could not write users file: {e}") from e

    def _load_manifest(self):
        try:
            if MANIFEST_FILE.exists():
                self._manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        except Exception:
            self._manifest = {}

    def _save_manifest(self):
        MANIFEST_FILE.write_text(json.dumps(self._manifest, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Folder naming                                                        #
    # ------------------------------------------------------------------ #

    def _folder_for(self, user_id: str) -> Path:
        user = self._users.get(user_id)
        if user and user.get("folder_name"):
            return DATA_DIR / user["folder_name"]
        return DATA_DIR / user_id

    def _migrate_user_folders(self):
        changed = False
        for user_id, user in list(self._users.items()):
            if user.get("folder_name"):
                continue
            folder_name = f"{_slugify(user.get('name', 'user'))}-{user_id}"
            old_path = DATA_DIR / user_id
            new_path = DATA_DIR / folder_name
            if old_path.exists() and not new_path.exists():
                try:
                    old_path.rename(new_path)
                    log.info("Migrated folder %s -> %s", old_path.name, new_path.name)
                except Exception as e:
                    log.warning("Could not migrate folder %s: %s", old_path, e)
            user["folder_name"] = folder_name
            changed = True
        if changed:
            try:
                self._save_users()
            except Exception as e:
                log.warning("Could not save users after migration: %s", e)

    # ------------------------------------------------------------------ #
    # Users                                                                #
    # ------------------------------------------------------------------ #

    def get_users(self) -> list[dict]:
        return sorted(self._users.values(), key=lambda u: u.get("name", ""))

    def get_user(self, user_id: str) -> Optional[dict]:
        return self._users.get(user_id)

    def add_user(self, name: str) -> dict:
        import uuid
        user_id = str(uuid.uuid4())[:8]
        folder_name = f"{_slugify(name)}-{user_id}"
        user = {
            "user_id": user_id,
            "name": name.strip(),
            "folder_name": folder_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._users[user_id] = user
        try:
            self._save_users()
        except RuntimeError:
            self._users.pop(user_id, None)
            raise
        return user

    def delete_user(self, user_id: str):
        if self._session_user_id == user_id and self._session_id is not None:
            raise RuntimeError(f"User '{user_id}' has an active recording session")
        user = self._users.get(user_id)
        if not user:
            raise FileNotFoundError(f"User '{user_id}' not found")
        folder = self._folder_for(user_id)
        if folder.exists():
            shutil.rmtree(folder)
        self._users.pop(user_id, None)
        self._manifest.pop(user_id, None)
        self._save_users()
        self._save_manifest()

    # ------------------------------------------------------------------ #
    # Session management                                                   #
    # ------------------------------------------------------------------ #

    async def start_session(self, user_id: str, label: str) -> dict:
        async with self._session_lock:
            if self._session_id is not None:
                await self._do_finalize()

            session_id = f"{label}_{user_id}_{int(time.time() * 1000)}"
            out_dir = self._folder_for(user_id) / label
            out_dir.mkdir(parents=True, exist_ok=True)

            self._session_id = session_id
            self._session_user_id = user_id
            self._session_label = label
            self._session_file = (out_dir / f"{session_id}.jsonl").open("w", encoding="utf-8")
            self._session_frame_count = 0
            self._session_start_time = time.time()

            return {
                "session_id": session_id,
                "user_id": user_id,
                "label": label,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }

    async def stop_session(self) -> dict:
        async with self._session_lock:
            if self._session_id is None:
                return {"error": "No active session"}
            return await self._do_finalize()

    async def _do_finalize(self) -> dict:
        if self._session_file:
            self._session_file.close()
            self._session_file = None

        result = {
            "session_id": self._session_id,
            "user_id": self._session_user_id,
            "label": self._session_label,
            "paired_frames": self._session_frame_count,
            "duration_s": round(time.time() - (self._session_start_time or time.time()), 1),
        }

        uid, lbl = self._session_user_id, self._session_label
        if uid and lbl:
            self._manifest.setdefault(uid, {})[lbl] = (
                self._manifest.get(uid, {}).get(lbl, 0) + 1
            )
            self._save_manifest()

        self._session_id = None
        self._session_user_id = None
        self._session_label = None
        self._session_frame_count = 0
        self._session_start_time = None

        return result

    def get_session_status(self) -> dict:
        return {
            "active": self._session_id is not None,
            "session_id": self._session_id,
            "user_id": self._session_user_id,
            "label": self._session_label,
            "frame_count": self._session_frame_count,
            "duration_s": (
                round(time.time() - self._session_start_time, 1)
                if self._session_start_time
                else 0
            ),
        }

    # ------------------------------------------------------------------ #
    # Sessions CRUD                                                        #
    # ------------------------------------------------------------------ #

    def list_sessions(self, user_id: str, label: str) -> list[dict]:
        label_dir = self._folder_for(user_id) / label
        if not label_dir.exists():
            return []
        sessions = []
        for path in sorted(label_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
                lines = [l for l in lines if l.strip()]
                frame_count = len(lines) // 2
                started_at = None
                duration_s = 0.0
                if lines:
                    try:
                        first = json.loads(lines[0])
                        started_at = first.get("recorded_at")
                    except Exception:
                        pass
                if len(lines) >= 2:
                    try:
                        last = json.loads(lines[-1])
                        last_time = last.get("recorded_at")
                        if started_at and last_time:
                            t0 = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                            t1 = datetime.fromisoformat(last_time.replace("Z", "+00:00"))
                            duration_s = round((t1 - t0).total_seconds(), 1)
                    except Exception:
                        pass
                sessions.append({
                    "session_id": path.stem,
                    "started_at": started_at,
                    "frame_count": frame_count,
                    "duration_s": duration_s,
                    "size_bytes": path.stat().st_size,
                    "path": str(path),
                })
            except Exception as e:
                log.warning("Could not read session %s: %s", path, e)
        return sessions

    def get_session_preview(self, user_id: str, label: str, session_id: str, stride: int = 2) -> dict:
        path = self._folder_for(user_id) / label / f"{session_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {session_id}")
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        total_pairs = len(lines) // 2
        sample = []
        started_at = None
        duration_s = 0.0
        try:
            if lines:
                first = json.loads(lines[0])
                started_at = first.get("recorded_at")
            if len(lines) >= 2:
                last = json.loads(lines[-1])
                last_time = last.get("recorded_at")
                if started_at and last_time:
                    t0 = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                    t1 = datetime.fromisoformat(last_time.replace("Z", "+00:00"))
                    duration_s = round((t1 - t0).total_seconds(), 1)
        except Exception:
            pass

        # Sample pairs at stride for sparkline
        for pair_idx in range(0, total_pairs, max(1, stride)):
            right_line = lines[pair_idx * 2] if pair_idx * 2 < len(lines) else None
            left_line = lines[pair_idx * 2 + 1] if pair_idx * 2 + 1 < len(lines) else None
            try:
                r = json.loads(right_line) if right_line else {}
                l = json.loads(left_line) if left_line else {}
                right_flex = r.get("flex", [])
                left_flex = l.get("flex", [])
                right_fsr = r.get("fsr", [])
                left_fsr = l.get("fsr", [])
                r_flex_avg = round(sum(right_flex) / len(right_flex), 1) if right_flex else 0
                l_flex_avg = round(sum(left_flex) / len(left_flex), 1) if left_flex else 0
                r_fsr_avg = round(sum(right_fsr) / len(right_fsr), 1) if right_fsr else 0
                l_fsr_avg = round(sum(left_fsr) / len(left_fsr), 1) if left_fsr else 0
                r_accel_mag = _vector_magnitude(r.get("accel", []))
                l_accel_mag = _vector_magnitude(l.get("accel", []))
                r_gyro_mag = _vector_magnitude(r.get("gyro", []))
                l_gyro_mag = _vector_magnitude(l.get("gyro", []))
                r_qw = r.get("quaternion", [1, 0, 0, 0])
                l_qw = l.get("quaternion", [1, 0, 0, 0])
                ts = r.get("timestamp_ms", pair_idx * 20)
                sample.append({
                    "t_ms": ts,
                    "right_flex_avg": r_flex_avg,
                    "left_flex_avg": l_flex_avg,
                    "right_fsr_avg": r_fsr_avg,
                    "left_fsr_avg": l_fsr_avg,
                    "right_accel_mag": r_accel_mag,
                    "left_accel_mag": l_accel_mag,
                    "right_gyro_mag": r_gyro_mag,
                    "left_gyro_mag": l_gyro_mag,
                    "right_quat_w": round(r_qw[0], 3) if r_qw else 1.0,
                    "left_quat_w": round(l_qw[0], 3) if l_qw else 1.0,
                })
            except Exception:
                pass
        return {
            "session_id": session_id,
            "frames": total_pairs,
            "duration_s": duration_s,
            "started_at": started_at,
            "sample": sample,
        }

    def delete_session(self, user_id: str, label: str, session_id: str) -> int:
        if self._session_id == session_id:
            raise RuntimeError("Cannot delete the currently active session")
        path = self._folder_for(user_id) / label / f"{session_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        path.unlink()
        new_count = max(0, self._manifest.get(user_id, {}).get(label, 1) - 1)
        self._manifest.setdefault(user_id, {})[label] = new_count
        self._save_manifest()
        return new_count

    def delete_user_label(self, user_id: str, label: str) -> int:
        if self._session_user_id == user_id and self._session_label == label and self._session_id:
            raise RuntimeError("Cannot delete a label with an active recording session")
        label_dir = self._folder_for(user_id) / label
        deleted = 0
        if label_dir.exists():
            deleted = sum(1 for _ in label_dir.glob("*.jsonl"))
            shutil.rmtree(label_dir)
        self._manifest.setdefault(user_id, {})[label] = 0
        self._save_manifest()
        return deleted

    def trim_session(self, user_id: str, label: str, session_id: str, start_frame: int, end_frame: int) -> dict:
        if self._session_id == session_id:
            raise RuntimeError("Cannot trim the currently active session")
        path = self._folder_for(user_id) / label / f"{session_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        total_pairs = len(lines) // 2
        start_frame = max(0, start_frame)
        end_frame = min(total_pairs, end_frame)
        if start_frame >= end_frame:
            raise ValueError("start_frame must be less than end_frame")

        kept_lines = lines[start_frame * 2: end_frame * 2]
        tmp_path = path.with_suffix(".jsonl.tmp")
        tmp_path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
        os.replace(str(tmp_path), str(path))

        kept_pairs = end_frame - start_frame
        duration_s = 0.0
        if len(kept_lines) >= 2:
            try:
                t0 = datetime.fromisoformat(
                    json.loads(kept_lines[0]).get("recorded_at", "").replace("Z", "+00:00")
                )
                t1 = datetime.fromisoformat(
                    json.loads(kept_lines[-1]).get("recorded_at", "").replace("Z", "+00:00")
                )
                duration_s = round((t1 - t0).total_seconds(), 1)
            except Exception:
                pass
        return {
            "session_id": session_id,
            "frames": kept_pairs,
            "duration_s": duration_s,
            "size_bytes": path.stat().st_size,
        }

    # ------------------------------------------------------------------ #
    # Frame ingestion                                                      #
    # ------------------------------------------------------------------ #

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
        return True, None

    def _publish_diagnostic(self, packet: dict, valid: bool, error: Optional[str]):
        msg = {"type": "packet", "packet": packet, "valid": valid, "error": error, "ts": time.time()}
        self._recent_packets.append(msg)
        for q in list(self._diag_subscribers):
            try:
                q.put_nowait(msg)
            except Exception:
                pass

    def ingest_packet(self, packet: dict) -> bool:
        hand_id = packet.get("hand_id", 1)
        ts: int = packet.get("timestamp_ms", int(time.time() * 1000))
        now = time.time()

        # Diagnostics telemetry
        valid, error = self._validate_packet(packet)
        if hand_id in self._schema_stats:
            if valid:
                self._schema_stats[hand_id]["valid"] += 1
            else:
                self._schema_stats[hand_id]["invalid"] += 1
                self._schema_stats[hand_id]["last_error"] = error
        if hand_id in self._packet_rate_window:
            win = self._packet_rate_window[hand_id]
            win.append(now)
            while win and now - win[0] > _RATE_WINDOW_S:
                win.popleft()
        self._publish_diagnostic(packet, valid, error)

        self._glove_last_seen[hand_id] = now

        matched_ts: Optional[int] = None
        for existing_ts, hands in self._frame_buffer.items():
            if hand_id not in hands and abs(existing_ts - ts) <= 300:
                matched_ts = existing_ts
                break

        if matched_ts is not None:
            self._frame_buffer[matched_ts][hand_id] = packet
            ts = matched_ts
        else:
            self._frame_buffer[ts] = {hand_id: packet}
            self._frame_received[ts] = now

        stale = [k for k, t in list(self._frame_received.items()) if now - t > FRAME_BUFFER_MAX_AGE_S]
        for k in stale:
            self._frame_buffer.pop(k, None)
            self._frame_received.pop(k, None)

        frame = self._frame_buffer.get(ts, {})
        if 1 not in frame or 2 not in frame:
            return False

        right = frame[1]
        left = frame[2]
        self._frame_buffer.pop(ts, None)
        self._frame_received.pop(ts, None)

        if self._session_id is None or self._session_file is None:
            return False

        now_iso = datetime.now(timezone.utc).isoformat()
        extra = {
            "label": self._session_label,
            "user_id": self._session_user_id,
            "session_id": self._session_id,
            "recorded_at": now_iso,
        }
        for pkt in (right, left):
            self._session_file.write(json.dumps({**pkt, **extra}) + "\n")
        self._session_file.flush()
        self._session_frame_count += 1
        return True

    def disconnect_glove(self, hand_id: int):
        self._glove_last_seen.pop(hand_id, None)

    def get_glove_status(self) -> list[dict]:
        now = time.time()
        result = []
        for hand_id, label in ((1, "right"), (2, "left")):
            last = self._glove_last_seen.get(hand_id)
            age_ms = round((now - last) * 1000) if last else None
            result.append({
                "hand_id": hand_id,
                "hand": label,
                "connected": age_ms is not None and age_ms < 2000,
                "last_seen_age_ms": age_ms,
            })
        return result

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def get_diagnostics(self, server_lan_ips: list[str] = None) -> dict:
        now = time.time()
        observed_hz = {}
        last_seen_age_ms = {}
        for hand_id in (1, 2):
            win = self._packet_rate_window.get(hand_id, deque())
            valid_ts = [t for t in win if now - t <= _RATE_WINDOW_S]
            observed_hz[hand_id] = round(len(valid_ts) / _RATE_WINDOW_S, 1)
            last = self._glove_last_seen.get(hand_id)
            last_seen_age_ms[hand_id] = round((now - last) * 1000) if last else None
        return {
            "endpoint": "collect",
            "ws_url_hint": "ws://<LAN-IP>:8000/ws/collect",
            "server_lan_ips": server_lan_ips or [],
            "expected_hz": 50,
            "observed_hz": observed_hz,
            "last_seen_age_ms": last_seen_age_ms,
            "schema": {
                str(hid): dict(s) for hid, s in self._schema_stats.items()
            },
            "active_session": self._session_id,
            "server_time_ms": int(now * 1000),
        }

    def get_recent_packets(self, hand_id: Optional[int] = None, since_ts_ms: Optional[float] = None, limit: int = 50) -> list:
        pkts = list(self._recent_packets)
        if hand_id is not None:
            pkts = [p for p in pkts if p.get("packet", {}).get("hand_id") == hand_id]
        if since_ts_ms is not None:
            since_s = since_ts_ms / 1000.0
            pkts = [p for p in pkts if p.get("ts", 0) > since_s]
        return pkts[-limit:]

    def add_diag_subscriber(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._diag_subscribers.add(q)
        return q

    def remove_diag_subscriber(self, q: asyncio.Queue):
        self._diag_subscribers.discard(q)

    # ------------------------------------------------------------------ #
    # Progress                                                             #
    # ------------------------------------------------------------------ #

    def get_all_progress(self) -> dict:
        return {uid: self._manifest.get(uid, {}) for uid in self._users}

    def get_user_progress(self, user_id: str) -> dict:
        return self._manifest.get(user_id, {})

    def get_global_progress(self) -> dict:
        totals: dict[str, int] = {}
        for uid_counts in self._manifest.values():
            for lbl, cnt in uid_counts.items():
                totals[lbl] = totals.get(lbl, 0) + cnt
        return totals
