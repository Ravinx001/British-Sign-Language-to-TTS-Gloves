"""Phase 3 verification — test WebSocket endpoints."""
import asyncio
import json
import sys

import websockets


async def run_tests():
    passed = 0
    failed = 0

    def report(name, ok, msg=""):
        nonlocal passed, failed
        if ok:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name} — {msg}")
            failed += 1

    # Test 1: Dashboard WebSocket connects and receives init
    try:
        async with websockets.connect("ws://127.0.0.1:8000/ws/dashboard") as dash:
            init_raw = await asyncio.wait_for(dash.recv(), timeout=3)
            init_msg = json.loads(init_raw)
            ok = init_msg.get("type") == "init" and "connections" in init_msg
            report("Dashboard WS init message", ok, f"got: {init_msg}")
    except Exception as e:
        report("Dashboard WS init message", False, str(e))

    # Test 2: ESP32 WS sends a packet and gets a prediction back
    try:
        async with websockets.connect("ws://127.0.0.1:8000/ws/predict") as esp:
            packet = {
                "timestamp_ms": 100000,
                "hand_id": 1,
                "flex": [1850, 820, 3680, 3740, 3700],
                "fsr": [2650, 2890, 180],
                "contact_bitmask": 8,
                "quaternion": [0.92, 0.08, -0.37, 0.05],
                "accel": [0.1, -0.2, 9.8],
            }
            await esp.send(json.dumps(packet))
            resp_raw = await asyncio.wait_for(esp.recv(), timeout=3)
            resp = json.loads(resp_raw)
            ok = "latency_ms" in resp and "route" in resp
            report("ESP32 WS prediction response", ok, f"got: {resp}")
    except Exception as e:
        report("ESP32 WS prediction response", False, str(e))

    # Test 3: Dashboard receives prediction broadcast from ESP32
    try:
        async with websockets.connect("ws://127.0.0.1:8000/ws/dashboard") as dash:
            # Consume init
            await asyncio.wait_for(dash.recv(), timeout=3)

            async with websockets.connect("ws://127.0.0.1:8000/ws/predict") as esp:
                # Send a prediction packet (this also triggers registration)
                packet = {
                    "timestamp_ms": 200000,
                    "hand_id": 1,
                    "flex": [1850, 820, 3680, 3740, 3700],
                    "fsr": [2650, 2890, 180],
                    "contact_bitmask": 8,
                    "quaternion": [0.92, 0.08, -0.37, 0.05],
                    "accel": [0, 0, 9.8],
                }
                await esp.send(json.dumps(packet))
                # ESP gets prediction
                await asyncio.wait_for(esp.recv(), timeout=3)

                # Dashboard should receive connection event then prediction
                # (registration happens on first packet, not WS connect)
                msgs = []
                for _ in range(2):
                    raw = await asyncio.wait_for(dash.recv(), timeout=3)
                    msgs.append(json.loads(raw))

                types = {m["type"] for m in msgs}
                ok_conn = any(
                    m.get("type") == "connection" and m.get("event") == "connected"
                    for m in msgs
                )
                ok_pred = any(
                    m.get("type") == "prediction" and "timestamp" in m
                    for m in msgs
                )
                report("Dashboard receives connection event", ok_conn,
                       f"types={types}")
                report("Dashboard receives prediction broadcast", ok_pred,
                       f"types={types}")

            # ESP32 disconnects — dashboard should get disconnect event
            disc_raw = await asyncio.wait_for(dash.recv(), timeout=3)
            disc_msg = json.loads(disc_raw)
            ok3 = disc_msg.get("type") == "connection" and disc_msg.get("event") == "disconnected"
            report("Dashboard receives disconnect event", ok3, f"got: {disc_msg}")
    except Exception as e:
        report("Dashboard broadcast flow", False, str(e))

    # Test 4: Dashboard get_metrics action
    try:
        async with websockets.connect("ws://127.0.0.1:8000/ws/dashboard") as dash:
            await asyncio.wait_for(dash.recv(), timeout=3)  # init
            await dash.send(json.dumps({"action": "get_metrics"}))
            metrics_raw = await asyncio.wait_for(dash.recv(), timeout=3)
            metrics = json.loads(metrics_raw)
            ok = metrics.get("type") == "metrics" and "total_predictions" in metrics
            report("Dashboard get_metrics action", ok, f"got: {metrics}")
    except Exception as e:
        report("Dashboard get_metrics action", False, str(e))

    print(f"\n=== Phase 3 Results: {passed} passed, {failed} failed ===")
    return failed


if __name__ == "__main__":
    failed = asyncio.run(run_tests())
    sys.exit(1 if failed else 0)
