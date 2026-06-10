"""Smoke-test collection REST + WebSocket (run server separately)."""

import asyncio
import json
import sys
import urllib.error
import urllib.request


BASE = "http://127.0.0.1:8000"


def http_json(method: str, path: str, body: dict | None = None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


async def ws_pair_once(uid: str):
    import websockets

    http_json("POST", "/api/collection/session/start", {"user_id": uid, "label": "X"})
    ts = 1700000000123
    pkt_r = {
        "timestamp_ms": ts,
        "hand_id": 1,
        "contact_bitmask": 0,
        "flex": [810, 820, 830, 840, 850],
        "fsr": [100, 200, 300],
        "accel": [-0.4, 3.2, 9.1],
        "gyro": [1.0, -1.0, 0.5],
        "quaternion": [0.92, 0.08, -0.37, 0.05],
    }
    pkt_l = {**pkt_r, "hand_id": 2}
    uri = "ws://127.0.0.1:8000/ws/collect"
    async with websockets.connect(uri) as w1, websockets.connect(uri) as w2:
        await w1.send(json.dumps(pkt_r))
        await w2.send(json.dumps(pkt_l))
        ack1 = json.loads(await w1.recv())
        ack2 = json.loads(await w2.recv())
    stop = http_json("POST", "/api/collection/session/stop")
    return ack1, ack2, stop


def main():
    try:
        signs = http_json("GET", "/api/collection/signs")
    except urllib.error.URLError as e:
        print("FAIL: server not reachable:", e)
        sys.exit(2)

    assert len(signs["signs"]) == 26
    user = http_json("POST", "/api/collection/users", {"name": "pytest_ws"})
    uid = user["user_id"]

    try:
        ack1, ack2, stop = asyncio.run(ws_pair_once(uid))
    except ImportError:
        print("SKIP: install websockets for WS test")
        sys.exit(0)

    assert ack1.get("paired") is True or ack2.get("paired") is True
    assert stop["paired_frames"] >= 1
    print("OK collection WS paired_frames=", stop["paired_frames"])
    sys.exit(0)


if __name__ == "__main__":
    main()
