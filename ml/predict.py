"""
Unified two-stage BSL sign prediction service.

Routes sensor frames to either the static (XGBoost) or dynamic (CNN-LSTM)
classifier based on motion energy, applies confidence thresholds, and
debounces output for stable predictions.

Architecture:
    predict(frame) → (sign, confidence)
    │
    ├─ Motion Detection (gravity-compensated accel or Euler-angle proxy)
    │  └─ ME ≤ 2.5 m/s² → STATIC route (XGBoost)
    │  └─ ME > 2.5 m/s² for 3+ frames → buffer 100 frames → DYNAMIC route
    │
    ├─ Thresholding
    │  ├─ Static: confidence ≥ 0.85
    │  └─ Dynamic: confidence ≥ 0.60
    │
    └─ Debouncing: 3 consecutive identical predictions → emit

Usage:
    from ml.predict import SignPredictor
    predictor = SignPredictor()
    for feature_vec, accel_xyz in sensor_stream:
        sign, confidence = predictor.predict(feature_vec, accel_xyz)
"""

import json
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import torch

from ml.config import (
    DEBOUNCE_FRAMES,
    DYNAMIC_CONFIDENCE_THRESHOLD,
    DYNAMIC_LETTERS,
    DYNAMIC_WINDOW_FRAMES,
    FEATURE_WEIGHTS,
    IDX_TO_LETTER,
    MOTION_ENERGY_THRESHOLD,
    MOTION_ONSET_FRAMES,
    NUM_FEATURES,
    PATHS,
    REST_LABELS,
    STATIC_CONFIDENCE_THRESHOLD,
)
from ml.segmentation import MotionSegmenter, extract_segment
from ml.train_dynamic import load_cnn_lstm_model
from ml.train_static import load_static_model

_W_ARR = np.asarray(FEATURE_WEIGHTS, dtype=np.float64)
_FEATURE_MOTION_THRESHOLD = 0.12
_ACCEL_DELTA_THRESHOLD = 1.20
_FEATURE_MOTION_ONSET_FRAMES = 2
_FEATURE_MOTION_OFFSET_FRAMES = 12
_FEATURE_MOTION_MAX_BUFFER = 500
_USE_GRAVITY_SEGMENTER_ROUTE = False
_DYNAMIC_CONFIDENCE_FLOOR = 0.90
_DYNAMIC_CONTEXT_LABELS = {
    # H ends in a two-finger/sideways shape that the static model commonly
    # sees as M/V/T because H itself is not in the static classifier.
    'H': {'M', 'T', 'V'},
    # J starts/ends near the I family in this dataset.
    'J': {'I', 'Y', 'K'},
}
_DYNAMIC_CONTEXT_MIN_CONFIDENCE = 0.20

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_best_static_model_path() -> Path:
    """Pick the deployed static model by reading training_report_real.json.

    Falls back to PATHS['rf_model'] if the report is missing or doesn't name
    a valid model. (RF was previously the best on real data.)
    """
    report_rel = PATHS.get('training_report_real')
    if report_rel:
        report_path = PROJECT_ROOT / report_rel
        if report_path.exists():
            try:
                with open(report_path) as f:
                    report = json.load(f)
                best = report.get('best_model')
                key_map = {
                    'xgb': 'xgboost_model',
                    'rf':  'rf_model',
                    'svm': 'svm_model',
                    'mlp': 'mlp_model',
                }
                if best in key_map:
                    return PROJECT_ROOT / PATHS[key_map[best]]
            except (OSError, ValueError):
                pass
    return PROJECT_ROOT / PATHS['rf_model']


class SignPredictor:
    """Unified BSL sign language prediction service.

    Maintains internal state for motion detection, frame buffering,
    and prediction debouncing. Call predict() once per sensor frame.
    """

    def __init__(self, static_model_path=None, dynamic_model_path=None):
        self._load_scaler()
        self._load_static_model(static_model_path)
        self._load_dynamic_model(dynamic_model_path)
        self._prewarm_dynamic_model()

        # Motion segmenter (for accel-based routing)
        self.segmenter = MotionSegmenter()

        # Runtime-overridable thresholds (defaults from ml/config.py).
        # The backend PredictionService syncs these from its own attributes
        # so the dashboard sliders take effect without reloading the model.
        self.static_confidence_threshold = STATIC_CONFIDENCE_THRESHOLD
        self.dynamic_confidence_threshold = DYNAMIC_CONFIDENCE_THRESHOLD
        self.motion_energy_threshold = MOTION_ENERGY_THRESHOLD

        # Euler-proxy motion state (for when accel_xyz is unavailable)
        self._prev_euler = None
        self._euler_above_count = 0
        self._euler_in_motion = False
        self._euler_buffer = []

        # Feature-delta motion state. The accelerometer route can read high
        # during a steady tilted pose; frame-to-frame feature change is better
        # for spotting actual H/J gesture motion without hurting static holds.
        self._prev_feature_vec = None
        self._prev_accel_vec = None
        self._feature_above_count = 0
        self._feature_below_count = 0
        self._feature_in_motion = False
        self._feature_buffer = []
        self._feature_segment_start_idx = 0
        self._feature_frame_idx = 0
        self._last_feature_delta_energy = 0.0
        self._last_accel_delta_energy = 0.0

        # Debounce state. Use a short rolling vote instead of requiring every
        # frame to be identical; live paired-glove packets can still jitter for
        # an occasional frame even when the hand pose is steady.
        self._pred_history = deque(maxlen=max(DEBOUNCE_FRAMES + 2, 6))
        self._last_emitted = ("Unknown", 0.0)
        self._unknown_frames = 0

        # Dynamic latch: hold dynamic result for several frames after segment
        self._dynamic_latch_remaining = 0
        self._dynamic_latch_pred = None

        # Diagnostics: top-K predictions from the most recent route invocation
        # (list of (sign, confidence) tuples, highest first). Cleared per call.
        self._last_topk: list[tuple[str, float]] = []
        self._topk_k = 3
        # Route used by the MOST RECENT predict() call. Set to "dynamic" when
        # _route_dynamic actually fires (segment complete OR during latch);
        # "static" otherwise. The segmenter's in_motion flag is unreliable
        # here because it resets to False on the same frame the segment is
        # extracted and dynamic prediction is emitted.
        self._last_route: str = "static"
        self._last_dynamic_gate: dict = {}

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_scaler(self):
        p = Path(PATHS['scaler'])
        with open(p, 'rb') as f:
            self.scaler = pickle.load(f)

    def _load_static_model(self, path=None):
        p = Path(path) if path else _resolve_best_static_model_path()
        (
            self.static_model,
            self._static_inv_label_map,
            self._static_weighted,
            self._static_label_names,
        ) = load_static_model(p, include_label_names=True)
        self.static_model_path = str(p)

    def _decode_static_label(self, raw_label):
        global_label = self._static_inv_label_map.get(int(raw_label), int(raw_label))
        if global_label in self._static_label_names:
            return str(self._static_label_names[global_label])
        if str(global_label) in self._static_label_names:
            return str(self._static_label_names[str(global_label)])
        return IDX_TO_LETTER.get(global_label, str(global_label))

    @staticmethod
    def is_rest_label(label):
        return str(label) in set(REST_LABELS)

    def _load_dynamic_model(self, path=None):
        try:
            self.dynamic_model, self.dynamic_class_names, _ = load_cnn_lstm_model()
        except FileNotFoundError:
            self.dynamic_model = None
            self.dynamic_class_names = np.array([], dtype=object)
        self._dynamic_bsl_letters = set(DYNAMIC_LETTERS)

    def _prewarm_dynamic_model(self):
        """Run one silent dummy inference to trigger PyTorch JIT compilation.

        Eliminates the 200-800ms cold-start spike on the first real dynamic detection.
        """
        if self.dynamic_model is None:
            return
        try:
            dummy = np.zeros((DYNAMIC_WINDOW_FRAMES, NUM_FEATURES), dtype=np.float32)
            self._route_dynamic(dummy)
        except Exception:
            pass  # Never block startup if prewarm fails

    # ------------------------------------------------------------------
    # Classification routes
    # ------------------------------------------------------------------

    def _route_static(self, feature_vec):
        """Classify a single frame with the deployed static model (RF/XGB/SVM/MLP).

        Order:
          1. scaler.transform (always)
          2. if model was trained on weighted features (MLP), multiply by
             FEATURE_WEIGHTS — matches what train_mlp did to its inputs.
        """
        X = self.scaler.transform(feature_vec.reshape(1, -1))
        if getattr(self, '_static_weighted', False):
            X = X * _W_ARR
        proba = self.static_model.predict_proba(X)[0]
        # Record top-K for diagnostics regardless of threshold gating
        self._last_topk = self._build_static_topk(proba, self._topk_k)

        idx = int(np.argmax(proba))
        confidence = float(proba[idx])
        # classes_[idx] is the contiguous label (0..N-1); inv_label_map converts
        # it back to the global LETTER_TO_IDX value so IDX_TO_LETTER works.
        raw_label = int(self.static_model.classes_[idx])
        sign = self._decode_static_label(raw_label)

        if confidence < self.static_confidence_threshold:
            return ("Unknown", confidence)
        return (sign, confidence)

    def _build_static_topk(self, proba, k):
        order = np.argsort(proba)[::-1][:k]
        out = []
        for i in order:
            raw = int(self.static_model.classes_[int(i)])
            out.append((self._decode_static_label(raw), float(proba[int(i)])))
        return out

    def _static_top1_no_diag(self, feature_vec):
        """Return static top-1 without overwriting live top-k diagnostics."""
        X = self.scaler.transform(feature_vec.reshape(1, -1))
        if getattr(self, '_static_weighted', False):
            X = X * _W_ARR
        proba = self.static_model.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        raw_label = int(self.static_model.classes_[idx])
        return self._decode_static_label(raw_label), float(proba[idx])

    def _dynamic_context_gate(self, class_name, segment):
        """Reject H/J motion unless the hand shape also looks plausible."""
        n_tail = min(15, len(segment))
        context_frame = np.median(segment[-n_tail:], axis=0)
        context_label, context_conf = self._static_top1_no_diag(context_frame)
        allowed = _DYNAMIC_CONTEXT_LABELS.get(class_name, set())
        ok = (
            context_label in allowed
            and context_conf >= _DYNAMIC_CONTEXT_MIN_CONFIDENCE
        )
        self._last_dynamic_gate = {
            'accepted': bool(ok),
            'class_name': class_name,
            'context_label': context_label,
            'context_confidence': round(float(context_conf), 4),
            'allowed': sorted(allowed),
        }
        return ok

    def _route_dynamic(self, segment):
        """Classify a 100-frame segment with the CNN-LSTM model."""
        if self.dynamic_model is None:
            self._last_topk = []
            return ("Unknown", 0.0)

        # Scale each frame with the fitted scaler
        n_frames = segment.shape[0]
        scaled = self.scaler.transform(segment.reshape(-1, NUM_FEATURES))
        scaled = scaled.reshape(n_frames, NUM_FEATURES)
        x = torch.FloatTensor(scaled).unsqueeze(0)
        with torch.no_grad():
            logits = self.dynamic_model(x)
            proba = torch.softmax(logits, dim=1)[0]

        proba_np = proba.cpu().numpy()
        # Record top-K for diagnostics
        order = np.argsort(proba_np)[::-1][:self._topk_k]
        self._last_topk = [
            (str(self.dynamic_class_names[int(i)]), float(proba_np[int(i)]))
            for i in order
        ]

        idx = int(torch.argmax(proba))
        confidence = float(proba[idx])
        class_name = str(self.dynamic_class_names[idx])

        # Only configured dynamic BSL letters are emitted; distractors -> Unknown.
        if class_name not in self._dynamic_bsl_letters:
            self._last_dynamic_gate = {
                'accepted': False,
                'class_name': class_name,
                'reason': 'not_dynamic_letter',
            }
            return ("Unknown", confidence)
        if confidence < max(self.dynamic_confidence_threshold, _DYNAMIC_CONFIDENCE_FLOOR):
            self._last_dynamic_gate = {
                'accepted': False,
                'class_name': class_name,
                'confidence': round(float(confidence), 4),
                'reason': 'low_dynamic_confidence',
            }
            return ("Unknown", confidence)
        if not self._dynamic_context_gate(class_name, segment):
            return ("Unknown", confidence)
        return (class_name, confidence)

    # ------------------------------------------------------------------
    # Debouncing
    # ------------------------------------------------------------------

    def _debounce(self, prediction):
        """Apply a rolling-vote debounce over recent thresholded predictions."""
        self._pred_history.append(prediction)

        sign, _ = prediction
        if sign == "Unknown":
            self._unknown_frames += 1
            if self._unknown_frames >= self._pred_history.maxlen:
                self._last_emitted = ("Unknown", 0.0)
            return ("Unknown", 0.0)

        self._unknown_frames = 0
        recent = [(s, c) for s, c in self._pred_history if s != "Unknown"]
        if len(recent) < DEBOUNCE_FRAMES:
            return ("Unknown", 0.0)

        counts: dict[str, int] = {}
        score_sums: dict[str, float] = {}
        for s, c in recent:
            counts[s] = counts.get(s, 0) + 1
            score_sums[s] = score_sums.get(s, 0.0) + float(c)

        best = max(score_sums, key=lambda s: (score_sums[s], counts[s]))
        if counts[best] >= DEBOUNCE_FRAMES:
            self._last_emitted = (best, score_sums[best] / counts[best])
            return self._last_emitted

        return ("Unknown", 0.0)

    # ------------------------------------------------------------------
    # Motion energy (Euler-angle proxy)
    # ------------------------------------------------------------------

    def _euler_motion_energy(self, feature_vec):
        """Estimate motion energy from frame-to-frame Euler angle change."""
        euler = feature_vec[16:22].copy()
        if self._prev_euler is None:
            self._prev_euler = euler
            return 0.0

        delta = np.abs(euler - self._prev_euler)
        # Handle angle wraparound (e.g. 359° → 1° = 2° change, not 358°)
        delta = np.minimum(delta, 360.0 - delta)
        self._prev_euler = euler

        # Heuristic scaling: sum of angle deltas (°/frame) → approx m/s²
        # ~5°/frame total across 6 axes ≈ 2.5 m/s² threshold
        return float(np.sum(delta)) * 0.5

    def _feature_delta_energy(self, feature_vec):
        """Frame-to-frame movement score over the engineered feature vector."""
        if self._prev_feature_vec is None:
            self._prev_feature_vec = feature_vec.copy()
            self._last_feature_delta_energy = 0.0
            return 0.0

        delta = np.abs(feature_vec - self._prev_feature_vec)
        # Euler-angle channels wrap at 360 degrees.
        delta[16:22] = np.minimum(delta[16:22], 360.0 - delta[16:22])
        self._prev_feature_vec = feature_vec.copy()
        energy = float(np.mean(delta))
        self._last_feature_delta_energy = energy
        return energy

    def _accel_delta_energy(self, accel_xyz):
        """Frame-to-frame movement score from raw accelerometer change."""
        if accel_xyz is None:
            self._last_accel_delta_energy = 0.0
            return 0.0

        accel = np.asarray(accel_xyz, dtype=np.float64)
        if self._prev_accel_vec is None:
            self._prev_accel_vec = accel.copy()
            self._last_accel_delta_energy = 0.0
            return 0.0

        energy = float(np.linalg.norm(accel - self._prev_accel_vec))
        self._prev_accel_vec = accel.copy()
        self._last_accel_delta_energy = energy
        return energy

    def _feed_feature_motion(self, feature_vec, accel_xyz=None):
        """Return a dynamic segment when live movement starts and settles."""
        feature_energy = self._feature_delta_energy(feature_vec)
        accel_energy = self._accel_delta_energy(accel_xyz)
        moving = (
            feature_energy >= _FEATURE_MOTION_THRESHOLD
            or accel_energy >= _ACCEL_DELTA_THRESHOLD
        )
        self._feature_buffer.append(feature_vec.copy())

        result = None
        if not self._feature_in_motion:
            if moving:
                self._feature_above_count += 1
                if self._feature_above_count >= _FEATURE_MOTION_ONSET_FRAMES:
                    self._feature_in_motion = True
                    self._feature_segment_start_idx = (
                        self._feature_frame_idx - _FEATURE_MOTION_ONSET_FRAMES + 1
                    )
                    self._feature_below_count = 0
            else:
                self._feature_above_count = 0
        else:
            if not moving:
                self._feature_below_count += 1
                if self._feature_below_count >= _FEATURE_MOTION_OFFSET_FRAMES:
                    end_idx = self._feature_frame_idx - _FEATURE_MOTION_OFFSET_FRAMES + 1
                    buf = np.array(self._feature_buffer)
                    start = max(0, self._feature_segment_start_idx)
                    end = min(end_idx, len(buf) - 1)
                    result = extract_segment(buf, start, end, DYNAMIC_WINDOW_FRAMES)
                    self._feature_in_motion = False
                    self._feature_above_count = 0
                    self._feature_below_count = 0
            else:
                self._feature_below_count = 0

        self._feature_frame_idx += 1
        if len(self._feature_buffer) > _FEATURE_MOTION_MAX_BUFFER:
            trim = len(self._feature_buffer) - _FEATURE_MOTION_MAX_BUFFER
            self._feature_buffer = self._feature_buffer[trim:]
            self._feature_segment_start_idx = max(0, self._feature_segment_start_idx - trim)
            self._feature_frame_idx -= trim

        return result

    # ------------------------------------------------------------------
    # Main predict entry point
    # ------------------------------------------------------------------

    def predict(self, feature_vec, accel_xyz=None):
        """Process one sensor frame and return a debounced prediction.

        Args:
            feature_vec: np.ndarray (NUM_FEATURES,) — extracted feature vector.
            accel_xyz: np.ndarray (3,) — optional raw accelerometer (m/s²).
                       When provided, uses gravity-compensated motion energy.
                       When None, uses Euler angle velocity as a proxy.

        Returns:
            (sign: str, confidence: float) — thresholded, debounced prediction.
            Returns ("Unknown", 0.0) if below confidence or not yet stabilized.
        """
        feature_vec = np.asarray(feature_vec, dtype=np.float64)
        # Avoid reporting stale top-k diagnostics while motion buffering returns
        # Unknown or a latched/debounced prediction without running a classifier.
        self._last_topk = []

        if accel_xyz is not None:
            return self._predict_with_accel(feature_vec, accel_xyz)
        return self._predict_euler_proxy(feature_vec)

    _DYNAMIC_LATCH_FRAMES = 12  # Hold dynamic result for 240ms at 50Hz

    def _force_emit(self, prediction):
        """Force-emit a prediction, bypassing debounce."""
        self._pred_history.clear()
        for _ in range(self._pred_history.maxlen):
            self._pred_history.append(prediction)
        self._last_emitted = prediction
        self._unknown_frames = 0

    def _predict_with_accel(self, feature_vec, accel_xyz):
        """Predict using raw accelerometer data for motion routing."""
        # If latching a dynamic result, keep emitting it
        if self._dynamic_latch_remaining > 0:
            self._dynamic_latch_remaining -= 1
            # Still feed the segmenter to keep its state updated
            accel = np.asarray(accel_xyz, dtype=np.float64)
            self.segmenter.feed(accel, feature_vec)
            self._feed_feature_motion(feature_vec, accel)
            self._last_route = "dynamic"
            return self._dynamic_latch_pred

        accel = np.asarray(accel_xyz, dtype=np.float64)
        if self.dynamic_model is None:
            raw_pred = self._route_static(feature_vec)
            self._last_route = "static"
            return self._debounce(raw_pred)

        feature_segment = self._feed_feature_motion(feature_vec, accel)
        if feature_segment is not None:
            raw_pred = self._route_dynamic(feature_segment)
            self._force_emit(raw_pred)
            self._dynamic_latch_pred = raw_pred
            self._dynamic_latch_remaining = self._DYNAMIC_LATCH_FRAMES
            self._last_route = "dynamic"
            return raw_pred

        segment = None
        if _USE_GRAVITY_SEGMENTER_ROUTE:
            segment = self.segmenter.feed(accel, feature_vec)
        else:
            # Keep the legacy segmenter warm for diagnostics/settings, but do
            # not let gravity-energy drift pin steady static signs as motion.
            self.segmenter.feed(accel, feature_vec)

        # Segmenter returned a completed dynamic segment
        if segment is not None:
            raw_pred = self._route_dynamic(segment)
            self._force_emit(raw_pred)
            self._dynamic_latch_pred = raw_pred
            self._dynamic_latch_remaining = self._DYNAMIC_LATCH_FRAMES
            self._last_route = "dynamic"
            return raw_pred

        # Not in motion → route to static classifier
        segmenter_blocks_static = (
            _USE_GRAVITY_SEGMENTER_ROUTE and self.segmenter.in_motion
        )
        if not segmenter_blocks_static and not self._feature_in_motion:
            raw_pred = self._route_static(feature_vec)
            self._last_route = "static"
            return self._debounce(raw_pred)

        # In motion, still buffering → output Unknown
        self._last_route = "dynamic"
        return ("Unknown", 0.0)

    def _predict_euler_proxy(self, feature_vec):
        """Predict using Euler angle velocity as motion proxy."""
        if self.dynamic_model is None:
            raw_pred = self._route_static(feature_vec)
            self._last_route = "static"
            return self._debounce(raw_pred)

        energy = self._euler_motion_energy(feature_vec)

        if not self._euler_in_motion:
            if energy >= self.motion_energy_threshold:
                self._euler_above_count += 1
                if self._euler_above_count >= MOTION_ONSET_FRAMES:
                    self._euler_in_motion = True
                    self._euler_buffer = [feature_vec.copy()]
                    self._last_route = "dynamic"
                    return self._last_emitted
            else:
                self._euler_above_count = 0
                # Static frame
                raw_pred = self._route_static(feature_vec)
                self._last_route = "static"
                return self._debounce(raw_pred)

        # Currently buffering dynamic frames
        if self._euler_in_motion:
            self._euler_buffer.append(feature_vec.copy())

            if len(self._euler_buffer) >= DYNAMIC_WINDOW_FRAMES:
                segment = np.array(self._euler_buffer[:DYNAMIC_WINDOW_FRAMES])
                self._euler_in_motion = False
                self._euler_above_count = 0
                self._euler_buffer = []

                raw_pred = self._route_dynamic(segment)
                self._last_route = "dynamic"
                return self._debounce(raw_pred)

        self._last_route = "dynamic"
        return self._last_emitted

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all internal state for a new recognition session."""
        self.segmenter.reset()
        self._prev_euler = None
        self._euler_above_count = 0
        self._euler_in_motion = False
        self._euler_buffer = []
        self._prev_feature_vec = None
        self._prev_accel_vec = None
        self._feature_above_count = 0
        self._feature_below_count = 0
        self._feature_in_motion = False
        self._feature_buffer = []
        self._feature_segment_start_idx = 0
        self._feature_frame_idx = 0
        self._last_feature_delta_energy = 0.0
        self._last_accel_delta_energy = 0.0
        self._pred_history.clear()
        self._last_emitted = ("Unknown", 0.0)
        self._unknown_frames = 0
        self._dynamic_latch_remaining = 0
        self._dynamic_latch_pred = None
        self._last_dynamic_gate = {}

    def motion_debug(self):
        """Small live-routing diagnostic used by the dashboard/test harness."""
        return {
            'feature_delta': round(float(self._last_feature_delta_energy), 4),
            'accel_delta': round(float(self._last_accel_delta_energy), 4),
            'feature_in_motion': bool(self._feature_in_motion),
            'feature_threshold': _FEATURE_MOTION_THRESHOLD,
            'accel_threshold': _ACCEL_DELTA_THRESHOLD,
            'gravity_route_enabled': bool(_USE_GRAVITY_SEGMENTER_ROUTE),
            'dynamic_gate': dict(self._last_dynamic_gate),
        }
