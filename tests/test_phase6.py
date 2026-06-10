"""Phase 6 verification — Dashboard SPA serves and works."""

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


# Test 1: GET / returns HTML with dashboard content
try:
    with urllib.request.urlopen(f"{BASE}/") as resp:
        html = resp.read().decode()
        ok = "BSL Sign Language Gloves" in html and "chart-conf" in html
        report("Dashboard HTML loads at /", ok,
               f"length={len(html)}, has_title={'BSL' in html}")
except Exception as e:
    report("Dashboard HTML loads at /", False, str(e))

# Test 2: Dashboard has all 5 tabs
try:
    with urllib.request.urlopen(f"{BASE}/") as resp:
        html = resp.read().decode()
        tabs = ["tab-live", "tab-manual", "tab-batch", "tab-settings", "tab-connections"]
        missing = [t for t in tabs if t not in html]
        ok = len(missing) == 0
        report("Dashboard has all 5 tabs", ok, f"missing={missing}")
except Exception as e:
    report("Dashboard has all 5 tabs", False, str(e))

# Test 3: Chart.js CDN referenced
try:
    with urllib.request.urlopen(f"{BASE}/") as resp:
        html = resp.read().decode()
        ok = "chart.js" in html.lower() or "chart.umd" in html
        report("Chart.js included", ok)
except Exception as e:
    report("Chart.js included", False, str(e))

# Test 4: WebSocket connection URL present
try:
    with urllib.request.urlopen(f"{BASE}/") as resp:
        html = resp.read().decode()
        ok = "ws/dashboard" in html
        report("WebSocket dashboard URL in HTML", ok)
except Exception as e:
    report("WebSocket dashboard URL in HTML", False, str(e))

# Test 5: Web Speech API TTS code present
try:
    with urllib.request.urlopen(f"{BASE}/") as resp:
        html = resp.read().decode()
        ok = "speechSynthesis" in html
        report("Web Speech API TTS in HTML", ok)
except Exception as e:
    report("Web Speech API TTS in HTML", False, str(e))

# Test 6: All API endpoints still work (regression)
try:
    # Settings
    with urllib.request.urlopen(f"{BASE}/api/settings") as resp:
        s = json.loads(resp.read())
    ok1 = "static_confidence_threshold" in s

    # Connections
    with urllib.request.urlopen(f"{BASE}/api/connections") as resp:
        c = json.loads(resp.read())
    ok2 = "connections" in c

    # Metrics
    with urllib.request.urlopen(f"{BASE}/api/metrics") as resp:
        m = json.loads(resp.read())
    ok3 = "total_predictions" in m

    ok = ok1 and ok2 and ok3
    report("All Phase 2-5 API endpoints still work", ok,
           f"settings={ok1}, conns={ok2}, metrics={ok3}")
except Exception as e:
    report("API regression check", False, str(e))


print(f"\n=== Phase 6 Results: {passed} passed, {failed} failed ===")
sys.exit(1 if failed else 0)
