"""Phase 2 verification — test all REST endpoints."""
import io
import json
import sys

import numpy as np
import urllib.request

BASE = "http://127.0.0.1:8000"
passed = 0
failed = 0


def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS: {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {name} — {e}")
        failed += 1


def post_json(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}", data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read()), r.status


def get_json(path):
    with urllib.request.urlopen(f"{BASE}{path}") as r:
        return json.loads(r.read()), r.status


# --- Test /api/predict/features ---
def test_predict_features():
    data = np.load("data/processed/test.npz")
    sample = data["X"][1].tolist()  # sample 1 has high confidence
    body = {"features": sample}
    result, status = post_json("/api/predict/features", body)
    assert status == 200, f"status={status}"
    assert "sign" in result, f"missing 'sign' key"
    assert "latency_ms" in result, f"missing 'latency_ms'"
    assert result["route"] == "static", f"route={result['route']}"


# --- Test /api/predict/raw ---
def test_predict_raw():
    body = {
        "right_flex": [1850, 820, 3680, 3740, 3700],
        "right_fsr": [2650, 2890, 180],
        "right_bitmask": 8,
        "right_quaternion": [0.92, 0.08, -0.37, 0.05],
        "left_flex": [810, 815, 820, 808, 830],
        "left_fsr": [100, 90, 50],
        "left_bitmask": 0,
        "left_quaternion": [0.98, 0.02, -0.15, 0.01],
    }
    result, status = post_json("/api/predict/raw", body)
    assert status == 200, f"status={status}"
    assert "latency_ms" in result


# --- Test /api/predict/batch with NPZ ---
def test_predict_batch_npz():
    data = np.load("data/processed/test.npz")
    buf = io.BytesIO()
    np.savez(buf, X=data["X"][:50], y=data["y"][:50])
    buf.seek(0)
    boundary = b"----TestBoundary123"
    body = (
        b"------TestBoundary123\r\n"
        b'Content-Disposition: form-data; name="file"; filename="test.npz"\r\n'
        b"Content-Type: application/octet-stream\r\n\r\n"
        + buf.read()
        + b"\r\n------TestBoundary123--\r\n"
    )
    req = urllib.request.Request(
        f"{BASE}/api/predict/batch", data=body,
        headers={"Content-Type": "multipart/form-data; boundary=----TestBoundary123"},
    )
    with urllib.request.urlopen(req) as r:
        result = json.loads(r.read())
        assert r.status == 200
    assert result["total_samples"] == 50, f"total={result['total_samples']}"
    assert "avg_latency_ms" in result
    assert "accuracy" in result
    print(f"    batch accuracy={result['accuracy']}, avg_latency={result['avg_latency_ms']}ms")


# --- Test /api/metrics ---
def test_metrics():
    result, status = get_json("/api/metrics")
    assert status == 200
    assert "total_predictions" in result
    assert result["total_predictions"] > 0, "should have predictions after above tests"


# --- Test /api/reset ---
def test_reset():
    req = urllib.request.Request(f"{BASE}/api/reset", data=b"", method="POST")
    with urllib.request.urlopen(req) as r:
        result = json.loads(r.read())
    assert result["status"] == "reset"
    metrics, _ = get_json("/api/metrics")
    assert metrics["total_predictions"] == 0, "predictions should be 0 after reset"


# --- Test /docs ---
def test_docs():
    with urllib.request.urlopen(f"{BASE}/docs") as r:
        assert r.status == 200


# --- Test validation error (wrong feature count) ---
def test_validation_error():
    body = {"features": [0.0] * 10}  # only 10, need 36
    try:
        post_json("/api/predict/features", body)
        assert False, "should have raised"
    except urllib.error.HTTPError as e:
        assert e.code == 422, f"expected 422, got {e.code}"


print("=== Phase 2 Tests ===")
test("POST /api/predict/features", test_predict_features)
test("POST /api/predict/raw", test_predict_raw)
test("POST /api/predict/batch (npz)", test_predict_batch_npz)
test("GET /api/metrics", test_metrics)
test("POST /api/reset", test_reset)
test("GET /docs", test_docs)
test("Validation error (422)", test_validation_error)
print(f"\n=== Results: {passed} passed, {failed} failed ===")
sys.exit(1 if failed else 0)
