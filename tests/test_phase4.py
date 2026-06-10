"""Phase 4 verification — TTS service & audio endpoint."""

import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# Ensure project root is on path for direct imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def post_json(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}", data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


# Test 1: TTS settings appear in metrics-like response
# (We check indirectly that TTS service loaded via prediction response)
try:
    features = [0.5] * 36
    result = post_json("/api/predict/features", {"features": features})
    # If a sign was predicted, audio_url should be present
    if result.get("sign"):
        ok = result.get("audio_url") is not None
        report("Prediction includes audio_url when sign predicted", ok,
               f"sign={result.get('sign')}, audio_url={result.get('audio_url')}")
    else:
        # No sign predicted (low confidence) — audio_url should be null
        ok = result.get("audio_url") is None
        report("Prediction has no audio_url when sign is null", ok,
               f"result={result}")
except Exception as e:
    report("Prediction with TTS", False, str(e))

# Test 2: Audio endpoint returns 404 when no audio exists yet (or returns file)
try:
    req = urllib.request.Request(f"{BASE}/api/audio/latest")
    try:
        with urllib.request.urlopen(req) as resp:
            content_type = resp.headers.get("Content-Type", "")
            ok = "audio" in content_type
            report("Audio endpoint returns audio file", ok, f"content-type={content_type}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            report("Audio endpoint returns 404 when no audio", True)
        else:
            report("Audio endpoint", False, f"HTTP {e.code}")
except Exception as e:
    report("Audio endpoint", False, str(e))

# Test 3: TTS Service initializes without crash
try:
    from backend.services.tts_service import TTSService
    tts = TTSService()
    settings = tts.get_settings()
    ok = "tts_enabled" in settings and "tts_engine" in settings
    report("TTSService initializes and returns settings", ok, f"settings={settings}")
except Exception as e:
    report("TTSService initialization", False, str(e))

# Test 4: TTS Service enable/disable
try:
    from backend.services.tts_service import TTSService
    tts = TTSService()
    tts.set_enabled(False)
    result = tts.speak("hello")
    ok = result is None  # Should return None when disabled
    report("TTSService disabled returns None", ok, f"got: {result}")
    tts.set_enabled(True)
except Exception as e:
    report("TTSService enable/disable", False, str(e))


print(f"\n=== Phase 4 Results: {passed} passed, {failed} failed ===")
sys.exit(1 if failed else 0)
