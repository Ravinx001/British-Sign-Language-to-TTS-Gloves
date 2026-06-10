"""ML prediction service wrapping SignPredictor with metrics tracking."""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the project root is importable so `ml` package resolves
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.predict import SignPredictor
from ml.feature_engineering import (
    apply_sensor_mask,
    extract_features,
    recompute_derived,
)
from ml.config import CALIBRATION, FEATURE_NAMES, NUM_FEATURES, PATHS, REST_LABELS, STATIC_IDX_TO_LABEL

_OFFSET_MASK = np.asarray(CALIBRATION['offset_channel_mask'], dtype=bool)
_CALIB_DIR = PROJECT_ROOT / PATHS['calibrations_dir']
_SCALER_STATS_PATH = PROJECT_ROOT / PATHS['scaler_stats']
_LABEL_RE = re.compile(r'^[A-Za-z0-9_\-]{1,64}$')
_REST_LABELS = set(REST_LABELS)


def _wrap_display_label(label: str | None, confidence: float | None) -> dict:
    is_rest = label in _REST_LABELS
    is_letter = bool(label) and not is_rest and label != 'Unknown'
    return {
        'sign': label if is_letter else None,
        'display_label': label if (is_letter or is_rest) else None,
        'is_rest': bool(is_rest),
        'rest_pose': label if is_rest else None,
        'confidence': round(confidence, 4) if (label and label != 'Unknown') else None,
    }


class PredictionService:
    """Wraps SignPredictor with feature engineering, metrics, and dict output."""

    def __init__(self):
        self.predictor = SignPredictor()
        self._total_predictions = 0
        self._latency_sum = 0.0
        self._conf_sum = 0.0
        self._conf_count = 0
        self._start_time = time.time()
        # Thresholds live on self.predictor; properties below proxy them so the
        # settings router and dashboard see one source of truth.

        # --- Per-user calibration state ---
        # self._calibration is either None or a dict with keys:
        #   offset: ndarray(NUM_FEATURES,) — additive correction applied BEFORE
        #     the StandardScaler. Masked to flex + Euler + openness channels;
        #     FSRs and inter-hand deltas stay 0.
        #   scale: ndarray(NUM_FEATURES,) — multiplicative correction (only flex
        #     channels diverge from 1.0; the rest stay 1.0). Populated by the
        #     optional reference-letter step.
        #   label: str | None — profile name (used by save/load).
        #   neutral_done: bool
        #   refs_done: list[str] — letters whose amplitude has been captured.
        self._calibration: dict | None = None
        self._calibrating: bool = False
        self._calib_state: str | None = None     # 'neutral' or 'ref:<LETTER>'
        self._calib_buffer: list[np.ndarray] = []
        self._calib_target_frames: int = 100
        # Lazy-loaded training stats sidecar (mean/scale/neutral_baseline/etc).
        self._scaler_stats: dict | None = None

    # ------------------------------------------------------------------
    # Threshold properties — proxy to predictor + segmenter
    # ------------------------------------------------------------------

    @property
    def static_confidence_threshold(self) -> float:
        return self.predictor.static_confidence_threshold

    @static_confidence_threshold.setter
    def static_confidence_threshold(self, value: float):
        self.predictor.static_confidence_threshold = float(value)

    @property
    def dynamic_confidence_threshold(self) -> float:
        return self.predictor.dynamic_confidence_threshold

    @dynamic_confidence_threshold.setter
    def dynamic_confidence_threshold(self, value: float):
        self.predictor.dynamic_confidence_threshold = float(value)

    @property
    def motion_energy_threshold(self) -> float:
        return self.predictor.motion_energy_threshold

    @motion_energy_threshold.setter
    def motion_energy_threshold(self, value: float):
        v = float(value)
        self.predictor.motion_energy_threshold = v
        # MotionSegmenter has its own copy used in the accel-routed path
        self.predictor.segmenter.threshold = v

    def predict_from_raw(self, right_packet: dict, left_packet: dict) -> dict:
        """Predict from raw sensor packets (ESP32 format).

        Each packet contains: flex, fsr, contact_bitmask, quaternion, accel.
        Converts to a NUM_FEATURES vector via extract_features(), then predicts.
        """
        raw_frame = {
            'R_flex': right_packet['flex'],
            'L_flex': left_packet['flex'],
            'R_fsr': right_packet['fsr'],
            'L_fsr': left_packet['fsr'],
            'R_quaternion': right_packet['quaternion'],
            'L_quaternion': left_packet['quaternion'],
        }
        feature_vec = extract_features(raw_frame)

        # Use accelerometer if available (average of both hands)
        r_accel = np.asarray(right_packet.get('accel', [0, 0, 9.8]), dtype=np.float64)
        l_accel = np.asarray(left_packet.get('accel', [0, 0, 9.8]), dtype=np.float64)
        accel = (r_accel + l_accel) / 2.0

        return self._predict_and_wrap(feature_vec, accel)

    def predict_from_features(self, features: list, accel: list | None = None) -> dict:
        """Predict from a pre-computed NUM_FEATURES vector."""
        feature_vec = np.asarray(features, dtype=np.float64)
        accel_arr = np.asarray(accel, dtype=np.float64) if accel else None
        return self._predict_and_wrap(feature_vec, accel_arr)

    def predict_oneshot_from_raw(self, right_packet: dict, left_packet: dict) -> dict:
        """One-shot prediction for REST callers (Manual Input form, single-frame
        external clients). Bypasses the streaming pipeline's debounce filter
        and motion routing — both of which require multiple consecutive frames
        to emit a stable prediction. Always uses the static classifier and
        returns the model's raw top-1 (subject to the confidence threshold).
        """
        raw_frame = {
            'R_flex': right_packet['flex'],
            'L_flex': left_packet['flex'],
            'R_fsr': right_packet['fsr'],
            'L_fsr': left_packet['fsr'],
            'R_quaternion': right_packet['quaternion'],
            'L_quaternion': left_packet['quaternion'],
        }
        feature_vec = extract_features(raw_frame)
        return self._predict_oneshot(feature_vec)

    def predict_oneshot_from_features(self, features: list) -> dict:
        """One-shot prediction from a pre-computed NUM_FEATURES vector."""
        feature_vec = np.asarray(features, dtype=np.float64)
        return self._predict_oneshot(feature_vec)

    def _predict_oneshot(self, feature_vec: np.ndarray) -> dict:
        """Shared one-shot path: calibration -> scaler -> static model -> top-K."""
        feature_vec = self._apply_calibration(feature_vec)

        t0 = time.perf_counter()
        sign, confidence = self._predict_raw_static(feature_vec)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Apply the same threshold the streaming path uses.
        if confidence < self.static_confidence_threshold:
            display = _wrap_display_label(None, None)
        else:
            display = _wrap_display_label(sign, confidence)

        # Re-build top-K against the same classifier used in _predict_raw_static.
        top_k = self.predictor._build_static_topk(
            self.predictor.static_model.predict_proba(
                self.predictor.scaler.transform(feature_vec.reshape(1, -1))
            )[0],
            self.predictor._topk_k,
        )

        return {
            **display,
            'latency_ms': round(elapsed_ms, 3),
            'route': 'static',
            'top_k': [{'sign': s, 'confidence': round(c, 4)} for s, c in top_k],
            'calibration': self.get_calibration_status(),
        }

    def predict_batch(self, X: np.ndarray, y_true: np.ndarray | None = None) -> dict:
        """Run batch predictions on an array of feature vectors.

        Uses the static classifier directly (no debounce) since batch samples
        are independent, not a streaming time series.

        Args:
            X: np.ndarray of shape (n_samples, 36).
            y_true: optional np.ndarray of integer labels for accuracy calc.

        Returns:
            dict with total_samples, predictions, accuracy (if y provided),
            avg_latency_ms, total_time_ms.
        """
        predictions = []
        correct = 0
        batch_start = time.perf_counter()

        for i in range(X.shape[0]):
            t0 = time.perf_counter()
            sign, confidence = self._predict_raw_static(X[i])
            if confidence < self.static_confidence_threshold:
                sign = 'Unknown'
            display = _wrap_display_label(sign, confidence if sign != 'Unknown' else None)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            pred = {
                **display,
                'latency_ms': round(elapsed_ms, 3),
            }
            predictions.append(pred)

            if y_true is not None and sign != 'Unknown':
                true_idx = int(y_true[i])
                true_label = self.predictor._static_label_names.get(
                    true_idx,
                    self.predictor._static_label_names.get(
                        str(true_idx),
                        STATIC_IDX_TO_LABEL.get(true_idx, str(true_idx)),
                    ),
                )
                if sign == true_label:
                    correct += 1

        total_time = (time.perf_counter() - batch_start) * 1000
        latencies = [p['latency_ms'] for p in predictions]

        output = {
            'total_samples': X.shape[0],
            'predictions': predictions,
            'avg_latency_ms': round(sum(latencies) / len(latencies), 3) if latencies else 0,
            'total_time_ms': round(total_time, 1),
        }
        if y_true is not None:
            output['accuracy'] = round(correct / X.shape[0], 4) if X.shape[0] > 0 else 0

        return output

    def get_metrics(self) -> dict:
        """Get accumulated performance metrics."""
        uptime = time.time() - self._start_time
        return {
            'total_predictions': self._total_predictions,
            'avg_latency_ms': round(
                self._latency_sum / self._total_predictions, 3
            ) if self._total_predictions > 0 else 0,
            'avg_confidence': round(
                self._conf_sum / self._conf_count, 4
            ) if self._conf_count > 0 else 0,
            'uptime_seconds': round(uptime, 1),
        }

    def get_thresholds(self) -> dict:
        """Get current runtime thresholds."""
        return {
            'static_confidence_threshold': self.static_confidence_threshold,
            'dynamic_confidence_threshold': self.dynamic_confidence_threshold,
            'motion_energy_threshold': self.motion_energy_threshold,
        }

    def reset(self):
        """Reset predictor state and accumulated metrics."""
        self.predictor.reset()
        self._total_predictions = 0
        self._latency_sum = 0.0
        self._conf_sum = 0.0
        self._conf_count = 0
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _predict_and_wrap(self, feature_vec: np.ndarray, accel: np.ndarray | None) -> dict:
        """Run streaming prediction + wrap into response dict + update metrics.

        Delegates to SignPredictor.predict() so motion routing, debouncing,
        and dynamic-segment classification all take effect. Thresholds live
        on the predictor (see properties above) and are applied internally.
        Applies the calibration offset (if set) before passing to the model.
        """
        # During an active calibration session, capture raw features into the
        # buffer; once we have enough, freeze the step's contribution.
        if self._calibrating:
            self._calib_buffer.append(feature_vec.copy())
            if len(self._calib_buffer) >= self._calib_target_frames:
                self._finalize_calibration_step()

        # Apply offset + scale + re-derive + re-mask BEFORE the StandardScaler.
        feature_vec = self._apply_calibration(feature_vec)
        feature_debug = self._feature_debug(feature_vec)

        t0 = time.perf_counter()
        sign, confidence = self.predictor.predict(feature_vec, accel_xyz=accel)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        route = self._detect_route(accel)
        top_k = getattr(self.predictor, '_last_topk', []) or []

        display = _wrap_display_label(
            sign if sign != 'Unknown' else None,
            confidence if sign != 'Unknown' else None,
        )

        result = {
            **display,
            'latency_ms': round(elapsed_ms, 3),
            'route': route,
            'top_k': [
                {'sign': s, 'confidence': round(c, 4)} for s, c in top_k
            ],
            'calibration': self.get_calibration_status(),
            'feature_debug': feature_debug,
            'motion_debug': self.predictor.motion_debug(),
        }

        # Update metrics
        self._total_predictions += 1
        self._latency_sum += elapsed_ms
        if sign != 'Unknown':
            self._conf_sum += confidence
            self._conf_count += 1

        return result

    def _feature_debug(self, feature_vec: np.ndarray, top_n: int = 5) -> dict:
        """Summarize how far a calibrated live frame is from train distribution."""
        scaled = self.predictor.scaler.transform(feature_vec.reshape(1, -1))[0]
        abs_scaled = np.abs(scaled)
        order = np.argsort(abs_scaled)[::-1][:top_n]
        mean = np.asarray(self.predictor.scaler.mean_, dtype=np.float64)
        scale = np.asarray(self.predictor.scaler.scale_, dtype=np.float64)
        return {
            'max_abs_z': round(float(abs_scaled.max()), 4),
            'mean_abs_z': round(float(abs_scaled.mean()), 4),
            'high_z_count': int(np.sum(abs_scaled > 3.0)),
            'top_z': [
                {
                    'idx': int(i),
                    'name': FEATURE_NAMES[int(i)],
                    'z': round(float(scaled[int(i)]), 4),
                    'value': round(float(feature_vec[int(i)]), 4),
                    'train_mean': round(float(mean[int(i)]), 4),
                    'train_scale': round(float(scale[int(i)]), 4),
                }
                for i in order
            ],
        }

    # ------------------------------------------------------------------
    # Per-user calibration
    # ------------------------------------------------------------------

    def _load_scaler_stats(self) -> dict:
        """Lazy-load the JSON sidecar containing training distribution stats.

        Returns {} when the sidecar is missing — callers must fall back to
        scaler.mean_ in that case.
        """
        if self._scaler_stats is not None:
            return self._scaler_stats
        try:
            with open(_SCALER_STATS_PATH, 'r') as f:
                self._scaler_stats = json.load(f)
        except (OSError, ValueError):
            self._scaler_stats = {}
        return self._scaler_stats

    def _train_neutral_baseline(self) -> np.ndarray:
        """Return the training-neutral baseline vector.

        Prefers `train_neutral_baseline` from scaler_stats.json (mean of frames
        with both hands open). Falls back to scaler.mean_ if the sidecar isn't
        present — same as the previous behaviour.
        """
        stats = self._load_scaler_stats()
        baseline = stats.get('train_neutral_baseline')
        if baseline is None:
            return np.asarray(self.predictor.scaler.mean_, dtype=np.float64)
        return np.asarray(baseline, dtype=np.float64)

    def _empty_calibration(self) -> dict:
        return {
            'offset': np.zeros(NUM_FEATURES, dtype=np.float64),
            'scale':  np.ones(NUM_FEATURES, dtype=np.float64),
            'label': None,
            'neutral_done': False,
            'refs_done': [],
            'captured_at': None,
        }

    def _apply_calibration(self, feature_vec: np.ndarray) -> np.ndarray:
        """Apply offset + scale, then re-derive + re-mask. NO-op when None."""
        cal = self._calibration
        if cal is None:
            return feature_vec
        fv = (feature_vec - cal['offset']) * cal['scale']
        # Re-derive indices 22-25 from the corrected source channels and
        # re-mask broken sensors so masked channels stay exactly 0.
        recompute_derived(fv)
        apply_sensor_mask(fv)
        return fv

    def start_calibration(self, n_frames: int | None = None) -> dict:
        """Begin a neutral-pose capture window.

        The next ``n_frames`` incoming feature vectors are buffered. When the
        buffer fills, the mean is compared against ``train_neutral_baseline``
        from scaler_stats.json to produce a per-channel offset (masked to flex
        / Euler / openness channels only — FSRs and inter-hand deltas stay 0).

        The user should hold both hands flat on the table, palms down, fingers
        straight, for ~3 seconds.
        """
        if self._calibration is None:
            self._calibration = self._empty_calibration()
        self._calibrating = True
        self._calib_state = 'neutral'
        self._calib_buffer = []
        target = n_frames if n_frames is not None else CALIBRATION['neutral_frames']
        self._calib_target_frames = max(10, int(target))
        return self.get_calibration_status()

    def start_reference_capture(self, letter: str) -> dict:
        """Begin capturing a reference letter pose for amplitude scaling.

        Requires a prior neutral capture (so the offset is already known).
        Reads ``frames_per_reference`` from config.
        """
        letter = letter.strip().upper()
        if self._calibration is None or not self._calibration['neutral_done']:
            raise ValueError("Run /api/calibrate/start (neutral) first.")
        if letter not in CALIBRATION['reference_letters']:
            allowed = ", ".join(CALIBRATION['reference_letters'])
            raise ValueError(f"Reference letter must be one of: {allowed}")
        self._calibrating = True
        self._calib_state = f'ref:{letter}'
        self._calib_buffer = []
        self._calib_target_frames = int(CALIBRATION['frames_per_reference'])
        return self.get_calibration_status()

    def _finalize_calibration_step(self) -> None:
        """Compute the offset (neutral step) or scale (reference step)."""
        buf = np.stack(self._calib_buffer, axis=0)
        if self._calibration is None:
            self._calibration = self._empty_calibration()

        if self._calib_state == 'neutral':
            neutral_mean = buf.mean(axis=0)
            baseline = self._train_neutral_baseline()
            offset = neutral_mean - baseline
            # Apply mask: only flex/euler/openness channels are offset-corrected.
            offset = np.where(_OFFSET_MASK, offset, 0.0)
            self._calibration['offset'] = offset.astype(np.float64)
            self._calibration['neutral_done'] = True
            self._calibration['captured_at'] = time.time()
        elif self._calib_state and self._calib_state.startswith('ref:'):
            letter = self._calib_state.split(':', 1)[1]
            scale = self._refine_scale_with_reference(buf, letter)
            self._calibration['scale'] = scale.astype(np.float64)
            if letter not in self._calibration['refs_done']:
                self._calibration['refs_done'].append(letter)
        # else: unknown state — silently end.

        self._calibrating = False
        self._calib_state = None
        self._calib_buffer = []

    def _refine_scale_with_reference(self, buf: np.ndarray, letter: str) -> np.ndarray:
        """Compute a per-flex amplitude scale from a reference-letter capture.

        After offset correction the user's neutral pose ≈ train_neutral_baseline.
        Comparing the user's letter amplitude against the training letter
        amplitude (both measured as deviation from train_neutral_baseline) gives
        the per-flex scale needed to inflate / deflate the user's range into
        the training distribution.

          corrected      = buf - offset          # user post-offset for this letter
          user_amp[flex] = mean(corrected)[flex] - train_neutral_baseline[flex]
          train_amp[flex]= stats.letter_flex_amplitudes[letter]
          scale[flex]    = clip(train_amp / user_amp, 0.7, 1.4)

        Only flex channels get a non-1.0 scale. Falls back to the previous
        scale if the sidecar isn't present.
        """
        cal = self._calibration
        prev_scale = np.array(cal['scale'], dtype=np.float64)
        stats = self._load_scaler_stats()
        amps = (stats.get('letter_flex_amplitudes') or {}).get(letter)
        flex_idx = stats.get('flex_indices') or list(range(0, 10))
        if not amps:
            return prev_scale  # sidecar missing; leave scale unchanged

        baseline = self._train_neutral_baseline()
        offset = cal['offset']
        corrected_mean = (buf - offset).mean(axis=0)
        user_amp = corrected_mean[flex_idx] - baseline[flex_idx]
        train_amp = np.asarray(amps, dtype=np.float64)

        lo, hi = CALIBRATION['amplitude_scale_clip']
        with np.errstate(divide='ignore', invalid='ignore'):
            raw = np.where(
                np.abs(user_amp) > 1e-6,
                train_amp / user_amp,
                1.0,
            )
        clipped = np.clip(raw, lo, hi)

        # Smooth across multiple reference letters with a geometric mean.
        scale = np.array(prev_scale, dtype=np.float64)
        already_refined = bool(cal['refs_done'])
        for i, idx in enumerate(flex_idx):
            scale[idx] = float(
                np.sqrt(prev_scale[idx] * clipped[i]) if already_refined
                else clipped[i]
            )
        return scale

    def get_calibration_status(self) -> dict:
        cal = self._calibration
        return {
            'active': self._calibrating,
            'collected': len(self._calib_buffer),
            'target': self._calib_target_frames,
            'state': self._calib_state,
            'offset_applied': cal is not None and cal['neutral_done'],
            'label': cal['label'] if cal else None,
            'refs_done': list(cal['refs_done']) if cal else [],
        }

    def clear_calibration(self) -> dict:
        """Discard any captured calibration and stop an in-progress capture."""
        self._calibration = None
        self._calibrating = False
        self._calib_state = None
        self._calib_buffer = []
        return self.get_calibration_status()

    def reset_state(self) -> dict:
        """Reset predictor state (debounce / motion / dynamic latch) WITHOUT
        clearing calibration. Wired to a frontend "Reset" button so the user
        can clear a stuck prediction between letters.
        """
        self.predictor.reset()
        return {'reset': True, 'calibration': self.get_calibration_status()}

    # ------------------------------------------------------------------
    # Calibration profile persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_label(label: str) -> str:
        label = (label or '').strip()
        if not _LABEL_RE.match(label):
            raise ValueError(
                "Profile label must be 1-64 chars of [A-Za-z0-9_-]."
            )
        return label

    def save_calibration(self, label: str) -> dict:
        """Persist the current calibration to data/calibrations/<label>.json."""
        label = self._validate_label(label)
        cal = self._calibration
        if cal is None or not cal['neutral_done']:
            raise ValueError("No active calibration to save. Run /start first.")
        _CALIB_DIR.mkdir(parents=True, exist_ok=True)
        out = _CALIB_DIR / f'{label}.json'
        payload = {
            'label': label,
            'offset': cal['offset'].tolist(),
            'scale':  cal['scale'].tolist(),
            'neutral_done': cal['neutral_done'],
            'refs_done': list(cal['refs_done']),
            'captured_at': cal.get('captured_at'),
        }
        with open(out, 'w') as f:
            json.dump(payload, f, indent=2)
        cal['label'] = label
        return {'saved': True, **self.get_calibration_status()}

    def load_calibration(self, label: str) -> dict:
        """Load a previously-saved profile and apply it to subsequent frames."""
        label = self._validate_label(label)
        path = _CALIB_DIR / f'{label}.json'
        if not path.exists():
            raise FileNotFoundError(f"No saved profile named '{label}'.")
        with open(path, 'r') as f:
            payload = json.load(f)
        self._calibration = {
            'offset': np.asarray(payload['offset'], dtype=np.float64),
            'scale':  np.asarray(payload['scale'],  dtype=np.float64),
            'label':  payload.get('label', label),
            'neutral_done': bool(payload.get('neutral_done', True)),
            'refs_done': list(payload.get('refs_done', [])),
            'captured_at': payload.get('captured_at'),
        }
        self._calibrating = False
        self._calib_state = None
        self._calib_buffer = []
        return {'loaded': True, **self.get_calibration_status()}

    def list_calibration_profiles(self) -> list[dict]:
        """Enumerate saved profiles by reading data/calibrations/*.json."""
        if not _CALIB_DIR.exists():
            return []
        out: list[dict] = []
        for path in sorted(_CALIB_DIR.glob('*.json')):
            try:
                with open(path, 'r') as f:
                    payload = json.load(f)
            except (OSError, ValueError):
                continue
            out.append({
                'label': payload.get('label', path.stem),
                'refs_done': list(payload.get('refs_done', [])),
                'captured_at': payload.get('captured_at'),
            })
        return out

    def _predict_raw_static(self, feature_vec: np.ndarray) -> tuple[str, float]:
        """Run static classifier and return (sign, confidence) without threshold."""
        X = self.predictor.scaler.transform(feature_vec.reshape(1, -1))
        proba = self.predictor.static_model.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        confidence = float(proba[idx])
        # Apply inv_label_map to convert contiguous training label → global idx
        raw_label = int(self.predictor.static_model.classes_[idx])
        sign = self.predictor._decode_static_label(raw_label)
        return sign, confidence

    def _detect_route(self, accel: np.ndarray | None) -> str:
        """Return the route taken by the MOST RECENT predict() call.

        Uses SignPredictor._last_route, which is set by each branch of
        predict(). Checking segmenter.in_motion here is unreliable because
        the flag resets to False on the same frame the dynamic segment is
        extracted and emitted — dynamic predictions would appear as 'static'.
        """
        return getattr(self.predictor, '_last_route', 'static')
