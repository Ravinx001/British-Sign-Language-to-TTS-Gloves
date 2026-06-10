"""
Motion segmentation: detect dynamic gesture onset/offset from sensor stream.

Uses gravity-compensated linear acceleration (motion energy) to determine
when the hand is moving. A low-pass filter estimates gravity; subtracting it
from raw accelerometer readings gives linear acceleration. The magnitude of
that vector is the motion energy signal.

Key features:
  - Gravity estimation via exponential low-pass filter (α=0.1)
  - Motion onset: 3+ consecutive frames above threshold (2.5 m/s²)
  - Motion offset: 25 consecutive frames below threshold (500ms at 50Hz)
  - Segment extraction: pad/truncate to exactly 100 frames
  - Stateful buffer for streaming frame-by-frame input

Usage:
    from ml.segmentation import MotionSegmenter
    seg = MotionSegmenter()
    for frame in sensor_stream:
        result = seg.feed(accel_xyz)
        if result is not None:
            segment = result  # (100, 36) array ready for dynamic classifier
"""

import numpy as np

from ml.config import (
    DYNAMIC_WINDOW_FRAMES,
    GRAVITY_LPF_ALPHA,
    MOTION_ENERGY_THRESHOLD,
    MOTION_OFFSET_FRAMES,
    MOTION_ONSET_FRAMES,
    NUM_FEATURES,
)


def motion_energy(accel_xyz, gravity_estimate):
    """Compute gravity-compensated motion energy from raw accelerometer.

    Args:
        accel_xyz: np.ndarray of shape (3,) — raw accelerometer [x, y, z] in m/s².
        gravity_estimate: np.ndarray of shape (3,) — current gravity estimate.

    Returns:
        float: magnitude of linear acceleration (m/s²).
    """
    linear_accel = accel_xyz - gravity_estimate
    return float(np.linalg.norm(linear_accel))


def update_gravity_estimate(gravity, accel_xyz, alpha=GRAVITY_LPF_ALPHA):
    """Update gravity estimate with exponential low-pass filter.

    g_new = α * accel + (1 - α) * g_old

    This converges to the gravity component when the sensor is stationary.

    Args:
        gravity: np.ndarray of shape (3,) — previous gravity estimate.
        accel_xyz: np.ndarray of shape (3,) — raw accel reading.
        alpha: filter coefficient (0.1 → slow-tracking, stable).

    Returns:
        np.ndarray of shape (3,) — updated gravity estimate.
    """
    return alpha * accel_xyz + (1.0 - alpha) * gravity


def detect_motion(energy_signal, threshold=MOTION_ENERGY_THRESHOLD,
                  onset_frames=MOTION_ONSET_FRAMES,
                  offset_frames=MOTION_OFFSET_FRAMES):
    """Detect motion segments from an energy signal.

    Rules:
      - Onset: energy above threshold for onset_frames consecutive frames.
      - Offset: energy below threshold for offset_frames consecutive frames.

    Args:
        energy_signal: np.ndarray of shape (N,) — per-frame motion energy.
        threshold: energy threshold in m/s².
        onset_frames: consecutive frames above threshold to trigger onset.
        offset_frames: consecutive frames below threshold to trigger offset.

    Returns:
        list of (start_idx, end_idx) tuples for each detected segment.
    """
    segments = []
    in_motion = False
    above_count = 0
    below_count = 0
    segment_start = 0

    for i, e in enumerate(energy_signal):
        if not in_motion:
            if e >= threshold:
                above_count += 1
                if above_count >= onset_frames:
                    in_motion = True
                    # Start is onset_frames back from current
                    segment_start = i - onset_frames + 1
                    below_count = 0
            else:
                above_count = 0
        else:
            if e < threshold:
                below_count += 1
                if below_count >= offset_frames:
                    in_motion = False
                    # End was offset_frames back from current
                    end = i - offset_frames + 1
                    segments.append((segment_start, end))
                    above_count = 0
                    below_count = 0
            else:
                below_count = 0

    # If still in motion at end of signal, close the segment
    if in_motion:
        segments.append((segment_start, len(energy_signal) - 1))

    return segments


def extract_segment(feature_buffer, start, end, target_len=DYNAMIC_WINDOW_FRAMES):
    """Extract and pad/truncate a segment to exactly target_len frames.

    If the segment is shorter than target_len, it is center-padded with the
    edge values repeated. If longer, it is center-cropped.

    Args:
        feature_buffer: np.ndarray of shape (N, n_features) — full buffer.
        start: start index (inclusive).
        end: end index (inclusive).
        target_len: output length (default 100 frames).

    Returns:
        np.ndarray of shape (target_len, n_features).
    """
    segment = feature_buffer[start:end + 1]
    seg_len = len(segment)

    if seg_len == target_len:
        return segment.copy()

    if seg_len > target_len:
        # Center crop
        offset = (seg_len - target_len) // 2
        return segment[offset:offset + target_len].copy()

    # Pad: center the segment within the target window
    n_features = segment.shape[1] if segment.ndim == 2 else NUM_FEATURES
    result = np.zeros((target_len, n_features), dtype=np.float64)

    pad_before = (target_len - seg_len) // 2
    result[pad_before:pad_before + seg_len] = segment

    # Replicate edges into padding regions
    if pad_before > 0:
        result[:pad_before] = segment[0]
    pad_after_start = pad_before + seg_len
    if pad_after_start < target_len:
        result[pad_after_start:] = segment[-1]

    return result


class MotionSegmenter:
    """Stateful motion segmenter for streaming frame-by-frame input.

    Maintains internal buffers for gravity estimation, energy tracking,
    and feature accumulation. Call `feed()` with each new frame.

    Typical usage:
        seg = MotionSegmenter()
        for accel_xyz, feature_vec in sensor_stream:
            segment = seg.feed(accel_xyz, feature_vec)
            if segment is not None:
                # segment is (100, 36), ready for dynamic classifier
                prediction = dynamic_model.predict(segment)
    """

    def __init__(self, threshold=MOTION_ENERGY_THRESHOLD,
                 onset_frames=MOTION_ONSET_FRAMES,
                 offset_frames=MOTION_OFFSET_FRAMES,
                 target_len=DYNAMIC_WINDOW_FRAMES):
        self.threshold = threshold
        self.onset_frames = onset_frames
        self.offset_frames = offset_frames
        self.target_len = target_len

        # Gravity estimate (initialized to [0, 0, 9.81] — sensor at rest)
        self.gravity = np.array([0.0, 0.0, 9.81], dtype=np.float64)

        # State machine
        self.in_motion = False
        self.above_count = 0
        self.below_count = 0

        # Buffers
        self.feature_buffer = []
        self.segment_start_idx = 0
        self.frame_idx = 0

    def reset(self):
        """Reset all internal state."""
        self.gravity = np.array([0.0, 0.0, 9.81], dtype=np.float64)
        self.in_motion = False
        self.above_count = 0
        self.below_count = 0
        self.feature_buffer = []
        self.segment_start_idx = 0
        self.frame_idx = 0

    def feed(self, accel_xyz, feature_vec):
        """Process one frame of sensor data.

        Args:
            accel_xyz: np.ndarray of shape (3,) — raw accelerometer m/s².
            feature_vec: np.ndarray of shape (36,) — extracted feature vector.

        Returns:
            np.ndarray of shape (target_len, n_features) if a segment was
            completed, otherwise None.
        """
        # Only update gravity when truly stationary — not in motion and no 
        # onset pending. Prevents gravity from drifting toward sustained
        # motion acceleration, which would suppress motion energy.
        accel = np.asarray(accel_xyz, dtype=np.float64)
        if not self.in_motion and self.above_count == 0:
            self.gravity = update_gravity_estimate(self.gravity, accel)

        # Compute motion energy
        energy = motion_energy(accel, self.gravity)

        # Store feature
        self.feature_buffer.append(np.asarray(feature_vec, dtype=np.float64))

        result = None

        if not self.in_motion:
            if energy >= self.threshold:
                self.above_count += 1
                if self.above_count >= self.onset_frames:
                    self.in_motion = True
                    self.segment_start_idx = self.frame_idx - self.onset_frames + 1
                    self.below_count = 0
            else:
                self.above_count = 0
        else:
            if energy < self.threshold:
                self.below_count += 1
                if self.below_count >= self.offset_frames:
                    # Motion ended — extract segment
                    end_idx = self.frame_idx - self.offset_frames + 1
                    buf = np.array(self.feature_buffer)
                    start = max(0, self.segment_start_idx)
                    end = min(end_idx, len(buf) - 1)
                    result = extract_segment(buf, start, end, self.target_len)

                    # Reset motion state
                    self.in_motion = False
                    self.above_count = 0
                    self.below_count = 0
            else:
                self.below_count = 0

        self.frame_idx += 1

        # Limit buffer growth (keep last 500 frames)
        max_buf = 500
        if len(self.feature_buffer) > max_buf:
            trim = len(self.feature_buffer) - max_buf
            self.feature_buffer = self.feature_buffer[trim:]
            self.segment_start_idx = max(0, self.segment_start_idx - trim)
            self.frame_idx -= trim

        return result

    def get_current_energy(self, accel_xyz):
        """Peek at current motion energy without advancing state.

        Useful for external monitoring / UI display.
        """
        accel = np.asarray(accel_xyz, dtype=np.float64)
        return motion_energy(accel, self.gravity)


def segment_batch(energy_signals, feature_sequences, **kwargs):
    """Batch segment detection for pre-recorded data.

    Args:
        energy_signals: list of np.ndarray energy traces.
        feature_sequences: list of np.ndarray (N, 36) feature sequences.

    Returns:
        list of extracted segments (each target_len × 36).
    """
    all_segments = []
    target_len = kwargs.get('target_len', DYNAMIC_WINDOW_FRAMES)

    for energy, features in zip(energy_signals, feature_sequences):
        segs = detect_motion(energy, **{k: v for k, v in kwargs.items()
                                        if k != 'target_len'})
        for start, end in segs:
            segment = extract_segment(features, start, end, target_len)
            all_segments.append(segment)

    return all_segments
