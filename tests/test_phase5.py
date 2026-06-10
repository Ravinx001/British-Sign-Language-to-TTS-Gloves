"""Phase 5 verification — Settings & connection management API."""

import json
import sys
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"
passed = 0
failed = 0


def report(name, ok, msg=""):
    global passed, failed
    if ok:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} — {msg}")
        failed += 1


def get_json(path):
    with urllib.request.urlopen(f"{BASE}{path}") as resp:
        return json.loads(resp.read())


def put_json(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}", data=data,
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def post_json(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}", data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


# Test 1: GET /api/settings returns all expected fields
try:
    settings = get_json("/api/settings")
    expected = {
        "static_confidence_threshold", "dynamic_confidence_threshold",
        "motion_energy_threshold", "tts_enabled", "tts_engine",
        "tts_rate", "tts_volume",
    }
    ok = expected.issubset(set(settings.keys()))
    report("GET /api/settings returns all fields", ok,
           f"keys={list(settings.keys())}")
except Exception as e:
    report("GET /api/settings", False, str(e))

# Test 2: PUT /api/settings updates threshold
try:
    put_json("/api/settings", {"static_confidence_threshold": 0.5})
    settings = get_json("/api/settings")
    ok = settings["static_confidence_threshold"] == 0.5
    report("PUT /api/settings updates threshold", ok,
           f"threshold={settings['static_confidence_threshold']}")
    # Restore default
    put_json("/api/settings", {"static_confidence_threshold": 0.85})
except Exception as e:
    report("PUT /api/settings threshold", False, str(e))

# Test 3: Lower threshold → more predictions pass
try:
    # First predict with default high threshold
    features_input = {"features": [0.5] * 36}
    result_high = post_json("/api/predict/features", features_input)
    # Now lower the threshold drastically
    put_json("/api/settings", {"static_confidence_threshold": 0.01})
    result_low = post_json("/api/predict/features", features_input)
    # With very low threshold, we should always get a sign
    ok = result_low.get("sign") is not None
    report("Lower threshold produces prediction", ok,
           f"high={result_high.get('sign')}, low={result_low.get('sign')}")
    # Restore
    put_json("/api/settings", {"static_confidence_threshold": 0.85})
except Exception as e:
    report("Threshold affects prediction", False, str(e))

# Test 4: PUT /api/settings TTS settings
try:
    put_json("/api/settings", {"tts_enabled": False, "tts_rate": 200})
    settings = get_json("/api/settings")
    ok = settings["tts_enabled"] is False and settings["tts_rate"] == 200
    report("PUT /api/settings TTS config", ok, f"settings={settings}")
    # Restore
    put_json("/api/settings", {"tts_enabled": True, "tts_rate": 150})
except Exception as e:
    report("PUT /api/settings TTS", False, str(e))

# Test 5: GET /api/connections returns list
try:
    conns = get_json("/api/connections")
    ok = "connections" in conns and "total" in conns
    report("GET /api/connections", ok, f"keys={list(conns.keys())}")
except Exception as e:
    report("GET /api/connections", False, str(e))

# Test 6: POST /api/connections/{id}/disconnect with invalid id
try:
    result = post_json("/api/connections/nonexistent/disconnect", {})
    ok = result.get("status") == "not_found"
    report("Disconnect nonexistent client returns not_found", ok, f"result={result}")
except Exception as e:
    report("Disconnect nonexistent", False, str(e))


print(f"\n=== Phase 5 Results: {passed} passed, {failed} failed ===")
sys.exit(1 if failed else 0)
