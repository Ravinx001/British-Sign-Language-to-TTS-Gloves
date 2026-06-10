"""Collection endpoints: REST session/user/progress APIs, diagnostics, and WebSocket ingest."""

import asyncio
import json
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from starlette.responses import JSONResponse

from backend.services.collection_service import BSL_SIGNS, COLLECTION_LABELS, REST_POSES
from ml.config import DYNAMIC_LETTERS

router = APIRouter(tags=["collection"])

_NO_STORE = {"Cache-Control": "no-store, must-revalidate"}


def _normalize_hand_id(value) -> int | None:
    """Return hand_id as 1/2, accepting numeric strings from firmware."""
    try:
        hand_id = int(value)
    except (TypeError, ValueError):
        return None
    return hand_id if hand_id in (1, 2) else None


# ------------------------------------------------------------------ #
# Pydantic request models                                             #
# ------------------------------------------------------------------ #


class UserCreate(BaseModel):
    name: str


class SessionStart(BaseModel):
    user_id: str
    label: str


class TrimRequest(BaseModel):
    start_frame: int
    end_frame: int


# ------------------------------------------------------------------ #
# Users                                                               #
# ------------------------------------------------------------------ #


@router.get("/api/collection/users")
async def list_users(request: Request):
    data = request.app.state.collection_service.get_users()
    return JSONResponse(content=data, headers=_NO_STORE)


@router.post("/api/collection/users", status_code=201)
async def create_user(body: UserCreate, request: Request):
    if not body.name.strip():
        raise HTTPException(400, "name must not be empty")
    try:
        return request.app.state.collection_service.add_user(body.name)
    except RuntimeError as e:
        raise HTTPException(500, str(e)) from e


@router.delete("/api/collection/users/{user_id}")
async def delete_user(user_id: str, request: Request):
    svc = request.app.state.collection_service
    try:
        svc.delete_user(user_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"deleted": True}


# ------------------------------------------------------------------ #
# Signs                                                               #
# ------------------------------------------------------------------ #


@router.get("/api/collection/signs")
async def list_signs():
    return {
        "signs": BSL_SIGNS,
        "dynamic": sorted(DYNAMIC_LETTERS),
        "rest": REST_POSES,
        "labels": COLLECTION_LABELS,
    }


# ------------------------------------------------------------------ #
# Session                                                             #
# ------------------------------------------------------------------ #


@router.post("/api/collection/session/start")
async def start_session(body: SessionStart, request: Request):
    svc = request.app.state.collection_service
    if body.label not in COLLECTION_LABELS:
        raise HTTPException(400, f"Unknown sign label '{body.label}'")
    if svc.get_user(body.user_id) is None:
        raise HTTPException(400, f"Unknown user_id '{body.user_id}'")
    return await svc.start_session(body.user_id, body.label)


@router.post("/api/collection/session/stop")
async def stop_session(request: Request):
    return await request.app.state.collection_service.stop_session()


@router.get("/api/collection/session/status")
async def session_status(request: Request):
    data = request.app.state.collection_service.get_session_status()
    return JSONResponse(content=data, headers=_NO_STORE)


# ------------------------------------------------------------------ #
# Sessions CRUD                                                       #
# ------------------------------------------------------------------ #


@router.get("/api/collection/sessions/{user_id}/{label}")
async def list_sessions(user_id: str, label: str, request: Request):
    svc = request.app.state.collection_service
    if svc.get_user(user_id) is None:
        raise HTTPException(404, f"User '{user_id}' not found")
    if label not in COLLECTION_LABELS:
        raise HTTPException(400, f"Unknown sign label '{label}'")
    sessions = svc.list_sessions(user_id, label)
    return JSONResponse(content={"sessions": sessions}, headers=_NO_STORE)


@router.get("/api/collection/sessions/{user_id}/{label}/{session_id}/preview")
async def session_preview(
    user_id: str,
    label: str,
    session_id: str,
    request: Request,
    stride: int = Query(default=2, ge=1, le=20),
):
    svc = request.app.state.collection_service
    if svc.get_user(user_id) is None:
        raise HTTPException(404, f"User '{user_id}' not found")
    try:
        data = svc.get_session_preview(user_id, label, session_id, stride=stride)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return JSONResponse(content=data, headers=_NO_STORE)


@router.delete("/api/collection/sessions/{user_id}/{label}/{session_id}")
async def delete_session(user_id: str, label: str, session_id: str, request: Request):
    svc = request.app.state.collection_service
    if svc.get_user(user_id) is None:
        raise HTTPException(404, f"User '{user_id}' not found")
    try:
        manifest_count = svc.delete_session(user_id, label, session_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"deleted": True, "manifest_count": manifest_count}


@router.post("/api/collection/sessions/{user_id}/{label}/{session_id}/trim")
async def trim_session(user_id: str, label: str, session_id: str, body: TrimRequest, request: Request):
    svc = request.app.state.collection_service
    if svc.get_user(user_id) is None:
        raise HTTPException(404, f"User '{user_id}' not found")
    try:
        result = svc.trim_session(user_id, label, session_id, body.start_frame, body.end_frame)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return result


# ------------------------------------------------------------------ #
# Progress                                                            #
# ------------------------------------------------------------------ #


@router.get("/api/collection/progress")
async def get_all_progress(request: Request):
    svc = request.app.state.collection_service
    data = {
        "by_user": svc.get_all_progress(),
        "global": svc.get_global_progress(),
    }
    return JSONResponse(content=data, headers=_NO_STORE)


@router.get("/api/collection/progress/{user_id}")
async def get_user_progress(user_id: str, request: Request):
    svc = request.app.state.collection_service
    if svc.get_user(user_id) is None:
        raise HTTPException(404, f"User '{user_id}' not found")
    data = {"user_id": user_id, "counts": svc.get_user_progress(user_id)}
    return JSONResponse(content=data, headers=_NO_STORE)


@router.delete("/api/collection/progress/{user_id}/{label}")
async def delete_user_label(user_id: str, label: str, request: Request):
    svc = request.app.state.collection_service
    if svc.get_user(user_id) is None:
        raise HTTPException(404, f"User '{user_id}' not found")
    if label not in COLLECTION_LABELS:
        raise HTTPException(400, f"Unknown sign label '{label}'")
    try:
        deleted_count = svc.delete_user_label(user_id, label)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"deleted_sessions": deleted_count, "label": label}


# ------------------------------------------------------------------ #
# Glove status                                                        #
# ------------------------------------------------------------------ #


@router.get("/api/collection/gloves")
async def glove_status(request: Request):
    data = {"gloves": request.app.state.collection_service.get_glove_status()}
    return JSONResponse(content=data, headers=_NO_STORE)


# ------------------------------------------------------------------ #
# Diagnostics                                                         #
# ------------------------------------------------------------------ #


@router.get("/api/collection/diagnostics")
async def get_diagnostics(request: Request):
    svc = request.app.state.collection_service
    lan_ips = getattr(request.app.state, "lan_ips", [])
    data = svc.get_diagnostics(server_lan_ips=lan_ips)
    return JSONResponse(content=data, headers=_NO_STORE)


@router.get("/api/collection/packets/recent")
async def recent_packets(
    request: Request,
    hand_id: Optional[int] = Query(default=None),
    since_ts_ms: Optional[float] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    svc = request.app.state.collection_service
    pkts = svc.get_recent_packets(hand_id=hand_id, since_ts_ms=since_ts_ms, limit=limit)
    return JSONResponse(content={"packets": pkts}, headers=_NO_STORE)


@router.websocket("/ws/diagnostics")
async def websocket_diagnostics(websocket: WebSocket):
    """Stream live packet events and periodic stats to browser clients."""
    await websocket.accept()
    endpoint = websocket.query_params.get("endpoint", "collect")
    if endpoint not in {"collect", "predict"}:
        endpoint = "collect"

    if endpoint == "predict":
        svc = websocket.app.state.prediction_diagnostics
        q = svc.add_subscriber()
    else:
        svc = websocket.app.state.collection_service
        q = svc.add_diag_subscriber()

    async def stats_sender():
        while True:
            await asyncio.sleep(1.0)
            lan_ips = getattr(websocket.app.state, "lan_ips", [])
            stats = svc.get_diagnostics(server_lan_ips=lan_ips)
            stats["type"] = "stats"
            try:
                await websocket.send_json(stats)
            except Exception:
                break

    stats_task = asyncio.create_task(stats_sender())
    try:
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=0.1)
                await websocket.send_json(msg)
            except asyncio.TimeoutError:
                pass
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        stats_task.cancel()
        if endpoint == "predict":
            svc.remove_subscriber(q)
        else:
            svc.remove_diag_subscriber(q)


# ------------------------------------------------------------------ #
# WebSocket — ESP32 ingest                                            #
# ------------------------------------------------------------------ #


@router.websocket("/ws/collect")
async def websocket_collect(websocket: WebSocket):
    """Accept JSON sensor packets from ESP32 gloves at 50 Hz."""
    await websocket.accept()
    svc = websocket.app.state.collection_service
    cm = websocket.app.state.connection_manager
    socket_id = str(uuid.uuid4())[:8]
    seen_hand_ids: set[int] = set()
    registered_hands: dict[int, str] = {}

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                packet = json.loads(raw)
            except json.JSONDecodeError:
                continue

            hand_id = _normalize_hand_id(packet.get("hand_id"))
            if hand_id is not None:
                packet["hand_id"] = hand_id
                seen_hand_ids.add(hand_id)
                if hand_id not in registered_hands:
                    client_id = f"{socket_id}-{hand_id}"
                    await cm.connect_esp32(websocket, client_id, hand_id, endpoint="collect")
                    registered_hands[hand_id] = client_id
                cm.update_esp32(registered_hands[hand_id], packet)

            paired = svc.ingest_packet(packet)
            status = svc.get_session_status()

            await websocket.send_json({
                "type": "ack",
                "paired": paired,
                "recording": status["active"],
                "label": status.get("label"),
                "frame_count": status["frame_count"],
            })

    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        for hand_id in seen_hand_ids:
            svc.disconnect_glove(hand_id)
        for client_id in registered_hands.values():
            await cm.disconnect_esp32(client_id)
