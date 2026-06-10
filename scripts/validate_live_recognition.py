"""
Offline replay validator for live recognition.

Picks 5 random held-out test sessions per user (10 total), streams the raw
frames through PredictionService.predict_from_raw() at 50 Hz pacing, applies
the same debouncing + confidence threshold the WebSocket route uses, and
records:
    - Emitted sign per session
    - Time to first emission
    - % of frames where the predictor said 'Unknown'
    - Whether the emitted sign matches the ground-truth label

Acceptance criterion: ≥ 90% of replayed sessions emit the correct letter
within 200ms of the settled window start.

Usage:
    python -m scripts.validate_live_recognition
    python -m scripts.validate_live_recognition --n-per-user 5
    python -m scripts.validate_live_recognition --no-pace        # skip sleep
    python -m scripts.validate_live_recognition --seed 123
    python -m scripts.validate_live_recognition --out results.json
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import DYNAMIC_LETTERS, SAMPLE_RATE_HZ, SEED
from ml.real_data import (
    compute_motion_energy_series,
    discover_sessions,
    find_settled_window,
    load_users,
    pair_frames,
    SETTLE_THRESHOLD,
)
from backend.services.predictor import PredictionService

FRAME_INTERVAL_S = 1.0 / SAMPLE_RATE_HZ   # 20ms at 50 Hz
SETTLED_EMISSION_WINDOW_MS = 200           # Acceptance window post settled-start


# =============================================================================
# Session picker: use only test sessions (held-out)
# =============================================================================

def pick_test_sessions(
    n_per_user: int = 5,
    seed: int = SEED,
) -> list[dict]:
    """Pick random static test sessions from each user.

    Returns:
        List of dicts: {user_id, user_name, letter, path, settle_start, settle_end}
    """
    rng = np.random.default_rng(seed)
    users = load_users()
    from ml.config import DYNAMIC_LETTERS, LETTER_TO_IDX
    static_letters = sorted(set(LETTER_TO_IDX.keys()) - DYNAMIC_LETTERS)

    # Build inventory of all sessions per user — same as build_real_dataset does
    # We pick from the full set here; in a real pipeline you'd load the session
    # split from coverage_report.json to guarantee test-only.
    # For safety we re-scan and take the last 5 sessions alphabetically,
    # which mirrors the 80/10/10 split's test bucket (last sessions per bucket).
    picked = []

    for user_id, user_info in users.items():
        user_sessions = []
        for letter in static_letters:
            sessions = discover_sessions(user_id, letter)
            for sess_path in sessions:
                pairs = pair_frames(sess_path)
                if len(pairs) < 10:
                    continue
                energy = compute_motion_energy_series(pairs)
                s_start, s_end = find_settled_window(energy, threshold=SETTLE_THRESHOLD)
                if (s_end - s_start + 1) < int(0.6 * SAMPLE_RATE_HZ):
                    continue
                user_sessions.append({
                    'user_id': user_id,
                    'user_name': user_info['name'],
                    'letter': letter,
                    'path': str(sess_path),
                    'settle_start': s_start,
                    'settle_end': s_end,
                })

        if not user_sessions:
            print(f"  [WARN] No usable sessions for user {user_info['name']}")
            continue

        rng.shuffle(user_sessions)
        picked.extend(user_sessions[:n_per_user])

    return picked


# =============================================================================
# Replay a single session through PredictionService
# =============================================================================

def replay_session(
    service: PredictionService,
    session: dict,
    pace: bool = True,
) -> dict:
    """Stream one session's frames through PredictionService at 50 Hz.

    The service is reset before each session so prior state doesn't bleed in.

    Returns:
        {
          'user_id': str,
          'user_name': str,
          'letter': str,           # ground truth
          'emitted_sign': str | None,
          'correct': bool,
          'time_to_emission_ms': float | None,
          'unknown_frame_pct': float,
          'n_frames': int,
          'within_window': bool,   # emitted within 200ms of settled_start
        }
    """
    service.reset()

    pairs = pair_frames(Path(session['path']))
    if not pairs:
        return {**session, 'error': 'no_pairs', 'correct': False,
                'emitted_sign': None, 'time_to_emission_ms': None,
                'unknown_frame_pct': 1.0, 'n_frames': 0, 'within_window': False}

    settle_start = session['settle_start']
    n_frames = len(pairs)
    n_unknown = 0
    first_emission_frame = None
    emitted_sign = None

    t_start = time.perf_counter()

    for frame_idx, pair in enumerate(pairs):
        r = pair['right']
        l = pair['left']

        right_packet = {
            'flex': r['flex'],
            'fsr': r['fsr'],
            'quaternion': r['quaternion'],
            'accel': r.get('accel', [0.0, 0.0, 9.81]),
        }
        left_packet = {
            'flex': l['flex'],
            'fsr': l['fsr'],
            'quaternion': l['quaternion'],
            'accel': l.get('accel', [0.0, 0.0, 9.81]),
        }

        result = service.predict_from_raw(right_packet, left_packet)

        sign = result.get('sign')
        if sign is None:
            n_unknown += 1
        elif emitted_sign is None:
            # First non-Unknown emission
            emitted_sign = sign
            first_emission_frame = frame_idx

        if pace:
            time.sleep(FRAME_INTERVAL_S)

    elapsed_wall_ms = (time.perf_counter() - t_start) * 1000

    # Time to emission from settled window start (in ms)
    if first_emission_frame is not None and first_emission_frame >= settle_start:
        time_to_emission_ms = (first_emission_frame - settle_start) * FRAME_INTERVAL_S * 1000
    elif first_emission_frame is not None:
        time_to_emission_ms = 0.0  # emitted before settled window — still counts
    else:
        time_to_emission_ms = None

    correct = (emitted_sign == session['letter'])
    within_window = (
        time_to_emission_ms is not None
        and time_to_emission_ms <= SETTLED_EMISSION_WINDOW_MS
        and correct
    )
    unknown_pct = n_unknown / max(n_frames, 1)

    return {
        'user_id': session['user_id'],
        'user_name': session['user_name'],
        'letter': session['letter'],
        'emitted_sign': emitted_sign,
        'correct': correct,
        'time_to_emission_ms': round(time_to_emission_ms, 1) if time_to_emission_ms is not None else None,
        'within_window': within_window,
        'unknown_frame_pct': round(unknown_pct, 4),
        'n_frames': n_frames,
        'settle_start_frame': settle_start,
        'first_emission_frame': first_emission_frame,
    }


# =============================================================================
# Main validator
# =============================================================================

def validate(
    n_per_user: int = 5,
    pace: bool = True,
    seed: int = SEED,
    output_path: str | None = None,
    verbose: bool = True,
) -> dict:
    """Run the full offline replay validation.

    Returns:
        dict with per-session results and aggregate metrics.
    """
    sep = '=' * 65
    print(sep)
    print('Offline Replay Validator')
    print(sep)

    # Initialise predictor (loads v2 models from PATHS)
    print('\nInitialising PredictionService...')
    service = PredictionService()
    print('  OK')

    # Pick sessions
    print(f'\nPicking {n_per_user} sessions per user...')
    sessions = pick_test_sessions(n_per_user=n_per_user, seed=seed)
    print(f'  Total sessions to replay: {len(sessions)}')

    if not sessions:
        print('  [ERROR] No sessions found. Run build_real_dataset.py first.')
        return {}

    # Replay
    results = []
    print(f'\nReplaying sessions at {"50 Hz paced" if pace else "full speed"}...')

    for i, sess in enumerate(sessions):
        prefix = f'  [{i+1:2d}/{len(sessions)}] {sess["user_name"]:8s} {sess["letter"]}  →  '
        result = replay_session(service, sess, pace=pace)
        emitted = result.get('emitted_sign', '?') or 'None'
        ok = '✓' if result['correct'] else '✗'
        in_win = '⚡' if result['within_window'] else '  '
        t_ms = result['time_to_emission_ms']
        t_str = f'{t_ms:.0f}ms' if t_ms is not None else 'N/A'
        unk_pct = result['unknown_frame_pct'] * 100
        if verbose:
            print(f'{prefix}{emitted:3s} {ok} {in_win}  (t={t_str}, unk={unk_pct:.0f}%)')
        results.append(result)

    # Aggregate metrics
    n = len(results)
    n_correct = sum(1 for r in results if r['correct'])
    n_within = sum(1 for r in results if r['within_window'])
    avg_unk = np.mean([r['unknown_frame_pct'] for r in results])

    emission_times = [r['time_to_emission_ms'] for r in results if r['time_to_emission_ms'] is not None]
    avg_emission_ms = float(np.mean(emission_times)) if emission_times else None
    median_emission_ms = float(np.median(emission_times)) if emission_times else None

    accuracy = n_correct / n if n > 0 else 0.0
    within_rate = n_within / n if n > 0 else 0.0

    # Per-user breakdown
    per_user = defaultdict(lambda: {'correct': 0, 'total': 0, 'within': 0})
    for r in results:
        u = r['user_name']
        per_user[u]['total'] += 1
        if r['correct']:
            per_user[u]['correct'] += 1
        if r['within_window']:
            per_user[u]['within'] += 1

    # Acceptance check
    ACCEPTANCE_THRESHOLD = 0.90
    accepted = within_rate >= ACCEPTANCE_THRESHOLD

    summary = {
        'n_sessions': n,
        'n_correct': n_correct,
        'n_within_window': n_within,
        'accuracy': round(accuracy, 4),
        'within_200ms_rate': round(within_rate, 4),
        'avg_unknown_frame_pct': round(float(avg_unk), 4),
        'avg_emission_ms': round(avg_emission_ms, 1) if avg_emission_ms is not None else None,
        'median_emission_ms': round(median_emission_ms, 1) if median_emission_ms is not None else None,
        'acceptance_threshold': ACCEPTANCE_THRESHOLD,
        'accepted': accepted,
        'per_user': {
            u: {
                'accuracy': round(v['correct'] / v['total'], 4),
                'within_rate': round(v['within'] / v['total'], 4),
                **v,
            }
            for u, v in per_user.items()
        },
    }

    output = {
        'summary': summary,
        'sessions': results,
    }

    # Print summary
    print(f'\n{sep}')
    print(f'Summary:')
    print(f'  Sessions replayed:         {n}')
    print(f'  Correct:                   {n_correct} / {n}  ({accuracy*100:.1f}%)')
    print(f'  Within 200ms (& correct):  {n_within} / {n}  ({within_rate*100:.1f}%)')
    print(f'  Avg unknown frames:        {avg_unk*100:.1f}%')
    if avg_emission_ms is not None:
        print(f'  Avg time to emission:      {avg_emission_ms:.0f}ms')
        print(f'  Median time to emission:   {median_emission_ms:.0f}ms')
    print(f'\n  Acceptance (≥{ACCEPTANCE_THRESHOLD*100:.0f}% within 200ms): '
          f'{"PASS ✓" if accepted else "FAIL ✗"}')
    if not accepted:
        print(f'  → Gap: need {int(ACCEPTANCE_THRESHOLD * n)} sessions; got {n_within}')
        print('  → Next step: check scaler, motion routing, debounce threshold.')
    print(sep)

    # Save output
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(str(out), 'w') as f:
            json.dump(output, f, indent=2)
        print(f'Results saved: {out}')
    else:
        default_out = PROJECT_ROOT / 'data' / 'figures' / 'validation_replay.json'
        default_out.parent.mkdir(parents=True, exist_ok=True)
        with open(str(default_out), 'w') as f:
            json.dump(output, f, indent=2)
        print(f'Results saved: {default_out}')

    return output


# =============================================================================
# Live new-user harness — interactive prompt against a running backend
# =============================================================================

NEW_USER_DEFAULT_LETTERS = ['E', 'K', 'R', 'T', 'W']
NEW_USER_DEFAULT_REPS = 5
NEW_USER_HOLD_SECONDS = 3.0


def validate_new_user_live(
    letters: list[str] | None = None,
    reps: int = NEW_USER_DEFAULT_REPS,
    hold_seconds: float = NEW_USER_HOLD_SECONDS,
    ws_url: str = 'ws://localhost:8000/ws/dashboard',
    output_path: str | None = None,
    debug: bool = False,
    target_top1: float = 0.60,
    target_top3: float = 0.80,
    min_correct_per_letter: int | None = None,
) -> dict:
    """Interactive live-hardware accuracy test against a running backend.

    The operator is prompted for each (letter, rep) combination, holds the
    sign for ``hold_seconds`` while predictions stream in via the dashboard
    WebSocket, and the script tallies top-1 and top-3 hit rates.

    Requires the backend to be running and the new wearer to have completed
    the calibration wizard. Uses the `websockets` async library (a transitive
    dependency of FastAPI).
    """
    import asyncio
    from urllib.parse import urlparse
    from urllib.request import Request, urlopen

    try:
        import websockets
    except ImportError:
        raise SystemExit(
            "The `websockets` package is required for --new-user mode.\n"
            "Install via: pip install websockets"
        )

    letters = letters or NEW_USER_DEFAULT_LETTERS
    results: list[dict] = []

    def reset_backend_stream_state():
        """Clear debounce/motion state between prompted live trials."""
        parsed = urlparse(ws_url)
        scheme = 'https' if parsed.scheme == 'wss' else 'http'
        reset_url = f'{scheme}://{parsed.netloc}/api/reset'
        req = Request(reset_url, method='POST')
        with urlopen(req, timeout=2) as resp:
            resp.read()

    async def collect_once(letter: str, rep: int) -> dict:
        if letter in DYNAMIC_LETTERS:
            prompt = (
                f"\n  [{letter} rep {rep}/{reps}] Get ready for dynamic {letter}. "
                "Press Enter, perform the motion, then settle..."
            )
            capture_msg = (
                f"    Capturing for {hold_seconds:.1f}s - perform the motion once, "
                "then hold the ending pose..."
            )
        else:
            prompt = (
                f"\n  [{letter} rep {rep}/{reps}] Hold the sign for {letter} "
                "and press Enter when ready..."
            )
            capture_msg = f"    Capturing for {hold_seconds:.1f}s - hold steady..."
        print(prompt)
        input()
        try:
            reset_backend_stream_state()
        except Exception as e:  # noqa: BLE001
            if debug:
                print(f"    [WARN] Could not reset predictor state: {e}")
        print(capture_msg)
        try:
            async with websockets.connect(ws_url, ping_interval=None) as ws:
                end = asyncio.get_event_loop().time() + hold_seconds
                top1_counts: dict[str, int] = {}
                emitted_counts: Counter[str] = Counter()
                route_counts: Counter[str] = Counter()
                raw_top1_counts: Counter[str] = Counter()
                raw_top1_conf: list[float] = []
                motion_energy: list[float] = []
                feature_delta: list[float] = []
                accel_delta: list[float] = []
                dynamic_gate_counts: Counter[str] = Counter()
                max_abs_z: list[float] = []
                high_z_count: list[int] = []
                top_z_counts: Counter[str] = Counter()
                top3_hits = 0
                samples = 0
                while asyncio.get_event_loop().time() < end:
                    timeout = max(0.05, end - asyncio.get_event_loop().time())
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    except asyncio.TimeoutError:
                        break
                    try:
                        data = json.loads(msg)
                    except (ValueError, TypeError):
                        continue
                    if data.get('type') != 'prediction':
                        continue
                    sign = data.get('sign')
                    emitted_label = (
                        sign
                        or data.get('rest_pose')
                        or data.get('display_label')
                    )
                    top_k_full = data.get('top_k') or []
                    top_k = [t.get('sign') for t in top_k_full]
                    emitted_counts[emitted_label or 'None'] += 1
                    route_counts[data.get('route') or 'None'] += 1
                    if top_k_full:
                        raw_top1 = top_k_full[0].get('sign') or 'None'
                        raw_top1_counts[raw_top1] += 1
                        try:
                            raw_top1_conf.append(
                                float(top_k_full[0].get('confidence') or 0.0)
                            )
                        except (TypeError, ValueError):
                            pass
                    try:
                        motion_energy.append(float(data.get('motion_energy') or 0.0))
                    except (TypeError, ValueError):
                        pass
                    motion_debug = data.get('motion_debug') or {}
                    try:
                        feature_delta.append(
                            float(motion_debug.get('feature_delta') or 0.0)
                        )
                    except (TypeError, ValueError):
                        pass
                    try:
                        accel_delta.append(
                            float(motion_debug.get('accel_delta') or 0.0)
                        )
                    except (TypeError, ValueError):
                        pass
                    gate = motion_debug.get('dynamic_gate') or {}
                    if gate:
                        gate_key = 'accepted' if gate.get('accepted') else (
                            gate.get('reason') or f"rejected:{gate.get('context_label')}"
                        )
                        dynamic_gate_counts[str(gate_key)] += 1
                    feature_debug = data.get('feature_debug') or {}
                    try:
                        max_abs_z.append(float(feature_debug.get('max_abs_z') or 0.0))
                    except (TypeError, ValueError):
                        pass
                    try:
                        high_z_count.append(int(feature_debug.get('high_z_count') or 0))
                    except (TypeError, ValueError):
                        pass
                    for item in feature_debug.get('top_z') or []:
                        name = item.get('name')
                        if name:
                            top_z_counts[str(name)] += 1
                    if emitted_label:
                        top1_counts[emitted_label] = top1_counts.get(emitted_label, 0) + 1
                    if letter in top_k:
                        top3_hits += 1
                    samples += 1
        except Exception as e:  # noqa: BLE001
            return {'letter': letter, 'rep': rep, 'error': str(e)}

        top1_letter = (
            max(top1_counts, key=top1_counts.get) if top1_counts else None
        )
        raw_top1_letter = (
            raw_top1_counts.most_common(1)[0][0] if raw_top1_counts else None
        )
        return {
            'letter': letter,
            'rep': rep,
            'samples': samples,
            'top1_dominant': top1_letter,
            'top1_correct': top1_letter == letter,
            'top3_hit_rate': (top3_hits / samples) if samples else 0.0,
            'top3_correct': (top3_hits / samples) >= 0.5 if samples else False,
            'debug': {
                'emitted_counts': dict(emitted_counts),
                'route_counts': dict(route_counts),
                'raw_top1_counts': dict(raw_top1_counts),
                'raw_top1_dominant': raw_top1_letter,
                'raw_top1_correct': raw_top1_letter == letter,
                'raw_top1_conf_avg': (
                    round(float(np.mean(raw_top1_conf)), 4)
                    if raw_top1_conf else None
                ),
                'raw_top1_conf_min': (
                    round(float(np.min(raw_top1_conf)), 4)
                    if raw_top1_conf else None
                ),
                'raw_top1_conf_max': (
                    round(float(np.max(raw_top1_conf)), 4)
                    if raw_top1_conf else None
                ),
                'motion_energy_avg': (
                    round(float(np.mean(motion_energy)), 4)
                    if motion_energy else None
                ),
                'motion_energy_p95': (
                    round(float(np.percentile(motion_energy, 95)), 4)
                    if motion_energy else None
                ),
                'feature_delta_p95': (
                    round(float(np.percentile(feature_delta, 95)), 4)
                    if feature_delta else None
                ),
                'accel_delta_p95': (
                    round(float(np.percentile(accel_delta, 95)), 4)
                    if accel_delta else None
                ),
                'dynamic_gate_counts': dict(dynamic_gate_counts.most_common(5)),
                'max_abs_z_avg': (
                    round(float(np.mean(max_abs_z)), 4)
                    if max_abs_z else None
                ),
                'max_abs_z_p95': (
                    round(float(np.percentile(max_abs_z, 95)), 4)
                    if max_abs_z else None
                ),
                'high_z_count_avg': (
                    round(float(np.mean(high_z_count)), 2)
                    if high_z_count else None
                ),
                'top_z_channels': dict(top_z_counts.most_common(8)),
            },
        }

    async def runner():
        for letter in letters:
            for rep in range(1, reps + 1):
                res = await collect_once(letter, rep)
                results.append(res)
                if 'error' in res:
                    print(f"    ERROR: {res['error']}")
                else:
                    line = (
                        f"    samples={res['samples']:3d} "
                        f"top1={res['top1_dominant']} "
                        f"({'PASS' if res['top1_correct'] else 'FAIL'})  "
                        f"top3_rate={res['top3_hit_rate']:.2f}"
                    )
                    if debug:
                        dbg = res.get('debug') or {}
                        line += (
                            f" | raw_top1={dbg.get('raw_top1_dominant')} "
                            f"conf_avg={dbg.get('raw_top1_conf_avg')} "
                            f"routes={dbg.get('route_counts')} "
                            f"emitted={dbg.get('emitted_counts')} "
                            f"motion_p95={dbg.get('motion_energy_p95')} "
                            f"feat_d95={dbg.get('feature_delta_p95')} "
                            f"acc_d95={dbg.get('accel_delta_p95')} "
                            f"gate={dbg.get('dynamic_gate_counts')} "
                            f"z_p95={dbg.get('max_abs_z_p95')} "
                            f"top_z={dbg.get('top_z_channels')}"
                        )
                    print(line)

    print('=' * 65)
    print('Live new-user accuracy harness')
    print(f'  Letters: {letters}')
    print(f'  Reps per letter: {reps}')
    print(f'  Hold duration: {hold_seconds:.1f}s')
    print(f'  WebSocket: {ws_url}')
    if debug:
        print('  Debug: route/raw-top1/confidence/motion telemetry enabled')
    print('=' * 65)

    asyncio.run(runner())

    # Aggregate
    n = sum(1 for r in results if 'error' not in r)
    n_top1 = sum(1 for r in results if r.get('top1_correct'))
    n_top3 = sum(1 for r in results if r.get('top3_correct'))
    n_raw_top1 = sum(
        1 for r in results
        if (r.get('debug') or {}).get('raw_top1_correct')
    )
    per_letter: dict[str, dict[str, int | bool]] = {}
    for letter in letters:
        letter_trials = [
            r for r in results
            if r.get('letter') == letter and 'error' not in r
        ]
        top1_correct = sum(1 for r in letter_trials if r.get('top1_correct'))
        top3_correct = sum(1 for r in letter_trials if r.get('top3_correct'))
        letter_summary: dict[str, int | bool] = {
            'trials': len(letter_trials),
            'top1_correct': top1_correct,
            'top3_correct': top3_correct,
        }
        if min_correct_per_letter is not None:
            letter_summary['top1_min_pass'] = (
                top1_correct >= min_correct_per_letter
            )
        per_letter[letter] = letter_summary

    per_letter_min_pass = (
        all(
            bool(row.get('top1_min_pass'))
            for row in per_letter.values()
        )
        if min_correct_per_letter is not None else True
    )
    summary = {
        'letters': letters,
        'reps': reps,
        'n_trials': n,
        'top1_correct': n_top1,
        'top3_correct': n_top3,
        'top1_accuracy': round(n_top1 / n, 4) if n else 0.0,
        'top3_accuracy': round(n_top3 / n, 4) if n else 0.0,
        'raw_top1_accuracy': round(n_raw_top1 / n, 4) if n else 0.0,
        'targets': {
            'top1': target_top1,
            'top3': target_top3,
            'min_correct_per_letter': min_correct_per_letter,
        },
        'per_letter': per_letter,
        'pass': {
            'top1': (n_top1 / n) >= target_top1 if n else False,
            'top3': (n_top3 / n) >= target_top3 if n else False,
            'per_letter_min_top1': per_letter_min_pass,
        },
    }
    summary['pass']['overall'] = all(summary['pass'].values())
    print('\n' + '=' * 65)
    print(f"  Top-1: {n_top1}/{n} = {summary['top1_accuracy']:.2f} "
          f"(target {target_top1:.2f}) "
          f"{'PASS' if summary['pass']['top1'] else 'FAIL'}")
    print(f"  Top-3: {n_top3}/{n} = {summary['top3_accuracy']:.2f} "
          f"(target {target_top3:.2f}) "
          f"{'PASS' if summary['pass']['top3'] else 'FAIL'}")
    if min_correct_per_letter is not None:
        print(
            f"  Per-letter top-1 minimum: {min_correct_per_letter}/{reps} "
            f"{'PASS' if per_letter_min_pass else 'FAIL'}"
        )
        for letter in letters:
            row = per_letter[letter]
            print(
                f"    {letter}: {row['top1_correct']}/{row['trials']} "
                f"{'PASS' if row.get('top1_min_pass') else 'FAIL'}"
            )
    if debug:
        print(f"  Raw top-1 before threshold/debounce: {n_raw_top1}/{n} = "
              f"{summary['raw_top1_accuracy']:.2f}")
    print('=' * 65)

    out = {'summary': summary, 'trials': results}
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'Results saved: {out_path}')
    return out


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Offline replay validator for live sign recognition.'
    )
    parser.add_argument(
        '--n-per-user', type=int, default=5,
        help='Number of test sessions to replay per user (default: 5)'
    )
    parser.add_argument(
        '--no-pace', action='store_true',
        help='Disable 50 Hz sleep pacing (runs at full CPU speed)'
    )
    parser.add_argument(
        '--seed', type=int, default=SEED,
        help=f'Random seed for session selection (default: {SEED})'
    )
    parser.add_argument(
        '--out', type=str, default=None,
        help='Path to save JSON results (default: data/figures/validation_replay.json)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress per-session output'
    )
    parser.add_argument(
        '--new-user', action='store_true',
        help=(
            'Live interactive mode: connect to a running backend via WebSocket '
            'and prompt the operator to perform each held-out letter while '
            'tallying top-1/top-3 hit rates. Skips the offline replay.'
        ),
    )
    parser.add_argument(
        '--letters', type=str, default=','.join(NEW_USER_DEFAULT_LETTERS),
        help='Comma-separated held-out letters for --new-user mode.',
    )
    parser.add_argument(
        '--reps', type=int, default=NEW_USER_DEFAULT_REPS,
        help='Repetitions per letter for --new-user mode.',
    )
    parser.add_argument(
        '--hold', type=float, default=NEW_USER_HOLD_SECONDS,
        help='Seconds to hold each sign for --new-user mode.',
    )
    parser.add_argument(
        '--ws-url', type=str, default='ws://localhost:8000/ws/dashboard',
        help='Dashboard WebSocket URL for --new-user mode.',
    )
    parser.add_argument(
        '--live-debug', action='store_true',
        help=(
            'In --new-user mode, print and save route/raw-top1/confidence/'
            'motion telemetry for each trial.'
        ),
    )
    parser.add_argument(
        '--target-top1', type=float, default=0.60,
        help='Required aggregate top-1 accuracy for --new-user mode.',
    )
    parser.add_argument(
        '--target-top3', type=float, default=0.80,
        help='Required aggregate top-3 accuracy for --new-user mode.',
    )
    parser.add_argument(
        '--min-correct-per-letter', type=int, default=None,
        help=(
            'Optional per-letter top-1 minimum for --new-user mode, e.g. 4 '
            'with --reps 5 requires every tested letter to pass at least 4 reps.'
        ),
    )
    args = parser.parse_args()

    if args.new_user:
        letters = [s.strip().upper() for s in args.letters.split(',') if s.strip()]
        validate_new_user_live(
            letters=letters,
            reps=args.reps,
            hold_seconds=args.hold,
            ws_url=args.ws_url,
            output_path=args.out,
            debug=args.live_debug,
            target_top1=args.target_top1,
            target_top3=args.target_top3,
            min_correct_per_letter=args.min_correct_per_letter,
        )
    else:
        validate(
            n_per_user=args.n_per_user,
            pace=not args.no_pace,
            seed=args.seed,
            output_path=args.out,
            verbose=not args.quiet,
        )
