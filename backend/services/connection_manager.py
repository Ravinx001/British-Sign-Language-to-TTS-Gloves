"""WebSocket state management for ESP32 glove connections and browser dashboards."""

import time
from dataclasses import dataclass

from fastapi import WebSocket


@dataclass
class ESP32Client:
    """State for a connected ESP32 glove."""

    client_id: str
    websocket: WebSocket
    hand_id: int  # 1=right, 2=left
    endpoint: str = "predict"
    ip: str = ""
    connected_at: float = 0.0
    last_seen: float = 0.0
    packet_count: int = 0
    avg_latency_ms: float = 0.0
    battery_pct: float | None = None

    def update(self, packet: dict):
        """Update state from incoming sensor packet."""
        now = time.time()
        if self.last_seen > 0:
            interval_ms = (now - self.last_seen) * 1000
            # Exponential moving average of packet interval
            alpha = 0.1
            self.avg_latency_ms = alpha * interval_ms + (1 - alpha) * self.avg_latency_ms
        self.last_seen = now
        self.packet_count += 1
        if "battery_pct" in packet:
            self.battery_pct = packet["battery_pct"]

    def to_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "endpoint": self.endpoint,
            "hand_id": self.hand_id,
            "hand": "right" if self.hand_id == 1 else "left",
            "ip": self.ip,
            "status": "active" if (time.time() - self.last_seen) < 2.0 else "stale",
            "packet_count": self.packet_count,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "battery_pct": self.battery_pct,
            "connected_at": self.connected_at,
            "last_seen": self.last_seen,
        }


class ConnectionManager:
    """Manage ESP32 glove connections and browser dashboard subscriptions."""

    def __init__(self):
        self.esp32_clients: dict[str, ESP32Client] = {}
        self.dashboard_clients: list[WebSocket] = []

    # --- ESP32 glove connections ---

    async def connect_esp32(self, websocket: WebSocket, client_id: str, hand_id: int, endpoint: str = "predict"):
        """Register a new ESP32 glove connection."""
        client = ESP32Client(
            client_id=client_id,
            websocket=websocket,
            hand_id=hand_id,
            endpoint=endpoint,
            ip=websocket.client.host if websocket.client else "",
            connected_at=time.time(),
            last_seen=time.time(),
        )
        self.esp32_clients[client_id] = client
        await self.broadcast_to_dashboards({
            "type": "connection",
            "event": "connected",
            "client": client.to_dict(),
        })

    async def disconnect_esp32(self, client_id: str):
        """Remove an ESP32 glove connection."""
        client = self.esp32_clients.pop(client_id, None)
        if client:
            await self.broadcast_to_dashboards({
                "type": "connection",
                "event": "disconnected",
                "client": client.to_dict(),
            })

    def update_esp32(self, client_id: str, packet: dict):
        """Update ESP32 client state from incoming packet."""
        client = self.esp32_clients.get(client_id)
        if client:
            client.update(packet)

    def get_all_connections(self) -> list[dict]:
        """Get status of all ESP32 clients."""
        return [c.to_dict() for c in self.esp32_clients.values()]

    # --- Browser dashboard subscriptions ---

    async def connect_dashboard(self, websocket: WebSocket):
        """Register a browser dashboard WebSocket."""
        await websocket.accept()
        self.dashboard_clients.append(websocket)
        # Send current state on connect
        await websocket.send_json({
            "type": "init",
            "connections": self.get_all_connections(),
        })

    def disconnect_dashboard(self, websocket: WebSocket):
        """Remove a browser dashboard subscription."""
        if websocket in self.dashboard_clients:
            self.dashboard_clients.remove(websocket)

    async def broadcast_to_dashboards(self, message: dict):
        """Send a message to all connected browser dashboards."""
        disconnected = []
        for ws in self.dashboard_clients:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect_dashboard(ws)

    async def broadcast_prediction(self, result: dict, motion_energy: float = 0.0):
        """Broadcast a prediction result to all dashboard clients."""
        await self.broadcast_to_dashboards({
            "type": "prediction",
            "sign": result.get("sign"),
            "display_label": result.get("display_label"),
            "is_rest": result.get("is_rest", False),
            "rest_pose": result.get("rest_pose"),
            "confidence": result.get("confidence"),
            "route": result.get("route"),
            "latency_ms": result.get("latency_ms"),
            "top_k": result.get("top_k", []),
            "calibration": result.get("calibration"),
            "feature_debug": result.get("feature_debug"),
            "motion_debug": result.get("motion_debug"),
            "motion_energy": round(motion_energy, 3),
            "timestamp": time.time(),
        })
