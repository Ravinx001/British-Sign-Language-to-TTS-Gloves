"""
BSL fingerspelling sensor signature definitions for all 26 letters.

Each letter is defined as a 26-feature target vector (normalized) with per-feature
variance, a static/dynamic flag, and metadata for confused pairs. Trajectory
keyframe arrays are provided for the configured dynamic letters (currently H and J).

Feature order matches ml/config.py FEATURE_NAMES exactly (26 features, touch removed).
Cross-references: docs/plan-phase1MlPipelineCombined.md §3 (BSL SignBank table),
                  docs/ML-dev-plan.md §2 (26-feature spec, v1.2).

Flex values:  0.0 = straight/extended, 1.0 = fully curled
FSR values:   0.0 = no pressure, 1.0 = maximum pressure
Orientation:  Euler angles in degrees (roll, pitch, yaw)
Hand openness: 0.0 = all fingers open, 1.0 = all fingers curled (fist)
Inter-hand delta: absolute degree difference
"""

import numpy as np

from ml.config import (
    FEATURE_NAMES,
    NUM_FEATURES,
    CONFUSED_PAIRS,
    DYNAMIC_LETTERS,
    LETTERS,
)

# =============================================================================
# Helper: Feature index lookups
# =============================================================================
_FI = {name: idx for idx, name in enumerate(FEATURE_NAMES)}


def _make_vector(
    # Right flex (thumb, index, middle, ring, pinky)
    r_flex=(0.05, 0.05, 0.05, 0.05, 0.05),
    # Left flex
    l_flex=(0.05, 0.05, 0.05, 0.05, 0.05),
    # Right FSR (thumb, index, pinky)
    r_fsr=(0.02, 0.02, 0.02),
    # Left FSR
    l_fsr=(0.02, 0.02, 0.02),
    # Right euler (roll, pitch, yaw) degrees
    r_euler=(0.0, 0.0, 0.0),
    # Left euler
    l_euler=(0.0, 0.0, 0.0),
    # Deprecated touch params silently ignored (hardware removed, v1.2)
    **_ignored,
):
    """Build a 26-feature vector from grouped sensor values."""
    vec = np.zeros(NUM_FEATURES, dtype=np.float64)

    # Flex (0–9)
    vec[0:5] = r_flex
    vec[5:10] = l_flex

    # FSR (10–15)
    vec[10:13] = r_fsr
    vec[13:16] = l_fsr

    # Orientation (16–21)
    vec[16:19] = r_euler
    vec[19:22] = l_euler

    # Derived: hand openness (22–23)
    vec[22] = np.mean(r_flex)
    vec[23] = np.mean(l_flex)

    # Derived: inter-hand deltas (24–25)
    vec[24] = abs(r_euler[0] - l_euler[0])  # delta roll
    vec[25] = abs(r_euler[1] - l_euler[1])  # delta pitch

    return vec


# =============================================================================
# Per-feature variance (σ) used by synthetic data generator.
# Higher variance on features that are less precisely controlled.
# =============================================================================
DEFAULT_FLEX_VAR = 0.06       # Flex sensors: moderate variance
DEFAULT_FSR_VAR = 0.08        # FSR: higher variance (pressure is noisy)
DEFAULT_EULER_VAR = 8.0       # Degrees — natural orientation variation
DEFAULT_DERIVED_VAR = 0.04    # Derived features inherit from flex


def _default_variance():
    """Return default per-feature variance vector."""
    var = np.zeros(NUM_FEATURES, dtype=np.float64)
    var[0:10] = DEFAULT_FLEX_VAR
    var[10:16] = DEFAULT_FSR_VAR
    var[16:22] = DEFAULT_EULER_VAR
    var[22:24] = DEFAULT_DERIVED_VAR
    var[24:26] = DEFAULT_EULER_VAR  # delta inherits euler variance
    return var


# =============================================================================
# Non-dominant (left) hand common poses
# =============================================================================
# Flat open, palm up — most common L pose for BSL fingerspelling
_L_FLAT_OPEN = dict(
    l_flex=(0.05, 0.05, 0.05, 0.05, 0.05),
    l_euler=(0.0, 0.0, 0.0),  # palm up
)

# Fist, thumb tucked
_L_FIST_TUCKED = dict(
    l_flex=(0.85, 0.90, 0.90, 0.90, 0.90),
    l_euler=(0.0, 0.0, 0.0),
)

# Flat, fingers extended (similar to flat open but palm more vertical)
_L_FLAT_EXTENDED = dict(
    l_flex=(0.05, 0.05, 0.05, 0.05, 0.05),
    l_euler=(0.0, 45.0, 0.0),
)

# Passive / relaxed
_L_PASSIVE = dict(
    l_flex=(0.15, 0.15, 0.15, 0.15, 0.15),
    l_euler=(0.0, 20.0, 0.0),
)

# Fist, thumb up
_L_FIST_THUMB_UP = dict(
    l_flex=(0.20, 0.90, 0.90, 0.90, 0.90),
    l_euler=(0.0, 0.0, 0.0),
)


# =============================================================================
# 26 Letter Definitions
#
# Each entry: {
#   'vector': np.ndarray[36] — target feature vector (normalized),
#   'variance': np.ndarray[36] — per-feature σ for data gen,
#   'dynamic': bool - True for configured dynamic letters,
#   'description': str — brief BSL description,
#   'key_signals': str — primary discriminating features,
# }
# =============================================================================

SIGN_DEFINITIONS = {}


# --- A: Index extended, others curled → L: flat open, palm up ---
SIGN_DEFINITIONS['A'] = {
    'vector': _make_vector(
        r_flex=(0.85, 0.05, 0.90, 0.90, 0.90),
        r_fsr=(0.60, 0.65, 0.05),
        r_touch=(0, 1, 0, 0, 0),
        r_euler=(0.0, 10.0, 0.0),
        l_fsr=(0.05, 0.05, 0.25),
        l_touch=(1, 0, 0, 0, 0),
        **_L_FLAT_OPEN,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R index extended pointing into L open palm, others curled',
    'key_signals': 'R_flex_index low, others high; L_all low; FSR high',
}

# --- B: Both hands flat, pressed together ---
SIGN_DEFINITIONS['B'] = {
    'vector': _make_vector(
        r_flex=(0.05, 0.05, 0.05, 0.05, 0.05),
        r_fsr=(0.15, 0.20, 0.70),
        r_touch=(0, 0, 0, 0, 0),
        r_euler=(0.0, 0.0, 0.0),
        l_fsr=(0.15, 0.20, 0.65),
        l_touch=(0, 0, 0, 0, 0),
        **_L_FLAT_OPEN,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R flat hand pressed against L flat open palm',
    'key_signals': 'All flex low both hands; FSR pad high (contact)',
}

# --- C: Curved C-shape → L: passive ---
SIGN_DEFINITIONS['C'] = {
    'vector': _make_vector(
        r_flex=(0.40, 0.50, 0.50, 0.50, 0.50),
        r_fsr=(0.02, 0.02, 0.02),
        r_touch=(0, 0, 0, 0, 0),
        r_euler=(0.0, 30.0, -20.0),
        l_fsr=(0.02, 0.02, 0.02),
        l_touch=(0, 0, 0, 0, 0),
        **_L_PASSIVE,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R hand curved in C-shape, L passive',
    'key_signals': 'R_flex mid-range (~0.5); L relaxed',
}

# --- D: R index curved, thumb at L index base ---
SIGN_DEFINITIONS['D'] = {
    'vector': _make_vector(
        r_flex=(0.30, 0.45, 0.85, 0.85, 0.85),
        r_fsr=(0.50, 0.10, 0.02),
        r_touch=(1, 0, 0, 0, 0),
        r_euler=(0.0, 15.0, 0.0),
        l_fsr=(0.05, 0.35, 0.02),
        l_touch=(0, 1, 0, 0, 0),
        **_L_FLAT_EXTENDED,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R index curved, R thumb touching L index base area',
    'key_signals': 'R_index mid; Touch R_thumb + L_index',
}

# --- E: R index touches L index fingertip ---
SIGN_DEFINITIONS['E'] = {
    'vector': _make_vector(
        r_flex=(0.80, 0.08, 0.85, 0.85, 0.85),
        r_fsr=(0.05, 0.55, 0.02),
        r_touch=(0, 1, 0, 0, 0),
        r_euler=(0.0, 10.0, 0.0),
        l_fsr=(0.05, 0.40, 0.02),
        l_touch=(0, 1, 0, 0, 0),
        **_L_FLAT_OPEN,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R index straight, touches L index fingertip',
    'key_signals': 'R_index low; Touch R_index → L_index',
}

# --- F: Thumb+index circle, others spread ---
SIGN_DEFINITIONS['F'] = {
    'vector': _make_vector(
        r_flex=(0.45, 0.50, 0.08, 0.08, 0.08),
        r_fsr=(0.55, 0.50, 0.02),
        r_touch=(1, 1, 0, 0, 0),
        r_euler=(0.0, 20.0, 0.0),
        l_fsr=(0.02, 0.02, 0.02),
        l_touch=(0, 0, 0, 0, 0),
        **_L_FLAT_OPEN,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R thumb+index form circle (OK shape), other fingers spread',
    'key_signals': 'R_thumb + R_index mid; others low',
}

# --- G: R index points at L palm side → L: fist, thumb up ---
SIGN_DEFINITIONS['G'] = {
    'vector': _make_vector(
        r_flex=(0.85, 0.05, 0.90, 0.90, 0.90),
        r_fsr=(0.05, 0.45, 0.02),
        r_touch=(0, 1, 0, 0, 0),
        r_euler=(0.0, 0.0, -30.0),
        l_fsr=(0.02, 0.02, 0.15),
        l_touch=(0, 0, 0, 0, 0),
        **_L_FIST_THUMB_UP,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R index pointing at L fist (thumb up) palm side',
    'key_signals': 'R_index low, others high; L_all high (fist)',
}

# --- H: Index+middle extended sideways + motion (DYNAMIC) ---
SIGN_DEFINITIONS['H'] = {
    'vector': _make_vector(
        r_flex=(0.80, 0.05, 0.05, 0.85, 0.85),
        r_fsr=(0.02, 0.02, 0.02),
        r_touch=(0, 0, 0, 0, 0),
        r_euler=(0.0, 0.0, -45.0),
        l_fsr=(0.02, 0.02, 0.02),
        l_touch=(0, 0, 0, 0, 0),
        **_L_FLAT_OPEN,
    ),
    'variance': _default_variance(),
    'dynamic': True,
    'description': 'R index+middle extended sideways with short motion',
    'key_signals': 'R_index + R_middle low; others high; short lateral motion',
}

# --- I: R index touches L middle fingertip ---
SIGN_DEFINITIONS['I'] = {
    'vector': _make_vector(
        r_flex=(0.80, 0.08, 0.85, 0.85, 0.85),
        r_fsr=(0.05, 0.50, 0.02),
        r_touch=(0, 1, 0, 0, 0),
        r_euler=(0.0, 10.0, 0.0),
        l_fsr=(0.02, 0.02, 0.02),
        l_touch=(0, 0, 1, 0, 0),
        **_L_FLAT_OPEN,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R index touches L middle fingertip',
    'key_signals': 'R_index low; Touch at L_middle',
}

# --- J: Like I + downward motion (DYNAMIC) ---
# Preparatory wrist rotation distinguishes J from I in static domain:
# wrist is tilted ~25° roll and ~-15° pitch to begin the downward sweep,
# and slightly looser grip produces different FSR signature.
SIGN_DEFINITIONS['J'] = {
    'vector': _make_vector(
        r_flex=(0.75, 0.10, 0.80, 0.82, 0.82),
        r_fsr=(0.08, 0.35, 0.04),
        r_touch=(0, 1, 0, 0, 0),
        r_euler=(25.0, -15.0, 10.0),
        l_fsr=(0.02, 0.02, 0.02),
        l_touch=(0, 0, 1, 0, 0),
        **_L_FLAT_OPEN,
    ),
    'variance': _default_variance(),
    'dynamic': True,
    'description': 'Like I (R index at L middle) + downward sweeping motion',
    'key_signals': 'Same as I but R_euler tilted for sweep prep + gyro_Y spike (dynamic)',
}

# --- K: R index bent, touches L index ---
SIGN_DEFINITIONS['K'] = {
    'vector': _make_vector(
        r_flex=(0.80, 0.45, 0.85, 0.85, 0.85),
        r_fsr=(0.05, 0.45, 0.02),
        r_touch=(0, 1, 0, 0, 0),
        r_euler=(0.0, 15.0, 0.0),
        l_fsr=(0.05, 0.30, 0.02),
        l_touch=(0, 1, 0, 0, 0),
        **_L_FLAT_EXTENDED,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R index bent, touching L index',
    'key_signals': 'R_index mid; Touch R_index → L_index',
}

# --- L: R thumb+index form L shape ---
SIGN_DEFINITIONS['L'] = {
    'vector': _make_vector(
        r_flex=(0.05, 0.05, 0.90, 0.90, 0.90),
        r_fsr=(0.02, 0.02, 0.02),
        r_touch=(0, 0, 0, 0, 0),
        r_euler=(0.0, 0.0, 0.0),
        l_fsr=(0.02, 0.02, 0.02),
        l_touch=(0, 0, 0, 0, 0),
        **_L_PASSIVE,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R thumb+index extended forming L, others curled',
    'key_signals': 'R_thumb + R_index low; others high',
}

# --- M: 3 fingers draped over L fist ---
SIGN_DEFINITIONS['M'] = {
    'vector': _make_vector(
        r_flex=(0.80, 0.10, 0.10, 0.10, 0.85),
        r_fsr=(0.05, 0.35, 0.02),
        r_touch=(0, 1, 1, 1, 0),
        r_euler=(0.0, -10.0, 0.0),
        l_fsr=(0.02, 0.02, 0.10),
        l_touch=(1, 1, 1, 0, 0),
        **_L_FIST_TUCKED,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R index+middle+ring draped over L fist',
    'key_signals': 'R_index/mid/ring low; L_all high (fist)',
}

# --- N: 2 fingers draped over L fist ---
SIGN_DEFINITIONS['N'] = {
    'vector': _make_vector(
        r_flex=(0.80, 0.10, 0.10, 0.85, 0.85),
        r_fsr=(0.05, 0.35, 0.02),
        r_touch=(0, 1, 1, 0, 0),
        r_euler=(0.0, -10.0, 0.0),
        l_fsr=(0.02, 0.02, 0.10),
        l_touch=(1, 1, 0, 0, 0),
        **_L_FIST_TUCKED,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R index+middle draped over L fist',
    'key_signals': 'R_index/mid low (ring/pinky high); L_all high (fist)',
}

# --- O: R index touches L ring fingertip ---
SIGN_DEFINITIONS['O'] = {
    'vector': _make_vector(
        r_flex=(0.80, 0.08, 0.85, 0.85, 0.85),
        r_fsr=(0.05, 0.50, 0.02),
        r_touch=(0, 1, 0, 0, 0),
        r_euler=(0.0, 10.0, 0.0),
        l_fsr=(0.02, 0.02, 0.02),
        l_touch=(0, 0, 0, 1, 0),
        **_L_FLAT_OPEN,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R index touches L ring fingertip',
    'key_signals': 'R_index low; Touch at L_ring',
}

# --- P: R index points down at L palm ---
SIGN_DEFINITIONS['P'] = {
    'vector': _make_vector(
        r_flex=(0.80, 0.08, 0.85, 0.85, 0.85),
        r_fsr=(0.05, 0.50, 0.02),
        r_touch=(0, 1, 0, 0, 0),
        r_euler=(0.0, -65.0, 0.0),
        l_fsr=(0.02, 0.15, 0.15),
        l_touch=(0, 0, 0, 0, 0),
        **_L_FLAT_OPEN,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R index pointing down at L open palm',
    'key_signals': 'R_index low; R_pitch inverted (pointing down)',
}

# --- Q: Like P but twisted ---
SIGN_DEFINITIONS['Q'] = {
    'vector': _make_vector(
        r_flex=(0.80, 0.08, 0.85, 0.85, 0.85),
        r_fsr=(0.05, 0.50, 0.02),
        r_touch=(0, 1, 0, 0, 0),
        r_euler=(45.0, -65.0, 0.0),
        l_fsr=(0.02, 0.15, 0.15),
        l_touch=(0, 0, 0, 0, 0),
        **_L_FLAT_OPEN,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'Like P but with wrist rotation (twisted)',
    'key_signals': 'Similar to P + R_roll shifted (~45°)',
}

# --- R: Index+middle crossed ---
SIGN_DEFINITIONS['R'] = {
    'vector': _make_vector(
        r_flex=(0.80, 0.08, 0.20, 0.85, 0.85),
        r_fsr=(0.02, 0.02, 0.02),
        r_touch=(0, 1, 1, 0, 0),
        r_euler=(0.0, 15.0, 0.0),
        l_fsr=(0.02, 0.02, 0.02),
        l_touch=(0, 0, 0, 0, 0),
        **_L_FLAT_OPEN,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R index+middle crossed (index over middle)',
    'key_signals': 'R_index low, R_middle slightly higher',
}

# --- S: Fist pressed on L palm ---
SIGN_DEFINITIONS['S'] = {
    'vector': _make_vector(
        r_flex=(0.90, 0.90, 0.90, 0.90, 0.90),
        r_fsr=(0.30, 0.35, 0.70),
        r_touch=(0, 0, 0, 0, 0),
        r_euler=(0.0, -10.0, 0.0),
        l_fsr=(0.10, 0.10, 0.60),
        l_touch=(0, 0, 0, 0, 0),
        **_L_FLAT_OPEN,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R fist pressed down on L open palm',
    'key_signals': 'R_all high (fist); L_all low; FSR pad high (contact)',
}

# --- T: Thumb between index+middle ---
SIGN_DEFINITIONS['T'] = {
    'vector': _make_vector(
        r_flex=(0.35, 0.60, 0.55, 0.85, 0.85),
        r_fsr=(0.45, 0.02, 0.02),
        r_touch=(1, 1, 1, 0, 0),
        r_euler=(0.0, 10.0, 0.0),
        l_fsr=(0.02, 0.02, 0.02),
        l_touch=(0, 0, 0, 0, 0),
        **_L_FLAT_EXTENDED,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R thumb tucked between index and middle fingers',
    'key_signals': 'Touch R_thumb + R_index + R_middle; thumb FSR',
}

# --- U: R index touches L pinky fingertip ---
SIGN_DEFINITIONS['U'] = {
    'vector': _make_vector(
        r_flex=(0.80, 0.08, 0.85, 0.85, 0.85),
        r_fsr=(0.05, 0.50, 0.02),
        r_touch=(0, 1, 0, 0, 0),
        r_euler=(0.0, 10.0, 0.0),
        l_fsr=(0.02, 0.02, 0.02),
        l_touch=(0, 0, 0, 0, 1),
        **_L_FLAT_OPEN,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R index touches L pinky fingertip',
    'key_signals': 'R_index low; Touch at L_pinky',
}

# --- V: Index+middle extended (V sign) ---
SIGN_DEFINITIONS['V'] = {
    'vector': _make_vector(
        r_flex=(0.80, 0.05, 0.05, 0.90, 0.90),
        r_fsr=(0.02, 0.02, 0.02),
        r_touch=(0, 0, 0, 0, 0),
        r_euler=(0.0, 20.0, 0.0),
        l_fsr=(0.02, 0.02, 0.02),
        l_touch=(0, 0, 0, 0, 0),
        **_L_PASSIVE,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R index+middle extended in V, others curled',
    'key_signals': 'R_index + R_middle low; others high',
}

# --- W: Index+middle+ring extended ---
SIGN_DEFINITIONS['W'] = {
    'vector': _make_vector(
        r_flex=(0.85, 0.05, 0.05, 0.05, 0.90),
        r_fsr=(0.02, 0.02, 0.02),
        r_touch=(0, 0, 0, 0, 0),
        r_euler=(0.0, 20.0, 0.0),
        l_fsr=(0.02, 0.02, 0.02),
        l_touch=(0, 0, 0, 0, 0),
        **_L_PASSIVE,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R index+middle+ring extended, others curled',
    'key_signals': 'R_index/mid/ring low; others high',
}

# --- X: Hooked index finger ---
SIGN_DEFINITIONS['X'] = {
    'vector': _make_vector(
        r_flex=(0.80, 0.45, 0.85, 0.85, 0.85),
        r_fsr=(0.02, 0.02, 0.02),
        r_touch=(0, 0, 0, 0, 0),
        r_euler=(0.0, 10.0, 0.0),
        l_fsr=(0.02, 0.02, 0.02),
        l_touch=(0, 0, 0, 0, 0),
        **_L_FLAT_OPEN,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R index finger hooked (bent at knuckle), others curled',
    'key_signals': 'R_index mid-range (~0.45); bent hook shape',
}

# --- Y: Thumb+pinky extended ---
SIGN_DEFINITIONS['Y'] = {
    'vector': _make_vector(
        r_flex=(0.05, 0.90, 0.90, 0.90, 0.05),
        r_fsr=(0.02, 0.02, 0.02),
        r_touch=(0, 0, 0, 0, 0),
        r_euler=(0.0, 20.0, 0.0),
        l_fsr=(0.02, 0.02, 0.02),
        l_touch=(0, 0, 0, 0, 0),
        **_L_PASSIVE,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R thumb+pinky extended, middle three curled',
    'key_signals': 'R_thumb + R_pinky low; others high',
}

# --- Z: R index extended, static in this hardware/user model ---
# Z is treated as a static sign for this project; only H and J are dynamic.
SIGN_DEFINITIONS['Z'] = {
    'vector': _make_vector(
        r_flex=(0.85, 0.05, 0.92, 0.92, 0.92),
        r_fsr=(0.02, 0.15, 0.02),
        r_touch=(0, 0, 0, 0, 0),
        r_euler=(10.0, 30.0, -15.0),
        l_fsr=(0.02, 0.02, 0.02),
        l_touch=(0, 0, 0, 0, 0),
        **_L_PASSIVE,
    ),
    'variance': _default_variance(),
    'dynamic': False,
    'description': 'R index extended, arm raised, wrist horizontal',
    'key_signals': 'R_index low; R_pitch elevated; static sign in live model',
}


# =============================================================================
# Trajectory Keyframes for Dynamic Letters (H, J)
#
# Each keyframe: (time_fraction, gyro_x, gyro_y, gyro_z) in °/s
# Applied on top of the base static pose during sequence generation.
# time_fraction is 0.0 → 1.0 over the 100-frame window.
# =============================================================================

DYNAMIC_TRAJECTORIES = {
    # H: Index+middle extended sideways, then a short lateral sweep
    'H': [
        (0.00, 0.0, 0.0, 0.0),       # Start: hold H-position
        (0.18, 0.0, 0.0, 0.0),       # Still holding
        (0.30, 90.0, -10.0, 20.0),   # Begin lateral sweep
        (0.45, 150.0, -15.0, 25.0),  # Peak motion
        (0.58, 100.0, -8.0, 15.0),   # Continuing sweep
        (0.70, 30.0, -3.0, 5.0),     # Decelerating
        (0.85, 0.0, 0.0, 0.0),       # Stopped
        (1.00, 0.0, 0.0, 0.0),       # Rest
    ],

    # J: Like I position, then a downward sweep (negative Y gyro spike)
    'J': [
        (0.00, 0.0, 0.0, 0.0),       # Start: hold I-position
        (0.20, 0.0, 0.0, 0.0),       # Still holding
        (0.30, 5.0, -80.0, 10.0),    # Begin downward sweep
        (0.45, 8.0, -150.0, 15.0),   # Peak downward motion
        (0.55, 10.0, -120.0, -20.0), # Curve at bottom
        (0.65, 5.0, -40.0, -30.0),   # Decelerating
        (0.75, 2.0, -10.0, -10.0),   # Slowing
        (0.85, 0.0, 0.0, 0.0),       # Stopped
        (1.00, 0.0, 0.0, 0.0),       # Rest
    ],
}


# =============================================================================
# Distractor gesture base definitions (for dynamic classifier training)
# These prevent CNN-LSTM 2-class overfitting on the configured dynamic signs.
# =============================================================================

DISTRACTOR_DEFINITIONS = {
    'wave_horizontal': {
        'description': 'Generic hand wave side-to-side',
        'base_flex': (0.10, 0.10, 0.10, 0.10, 0.10),
        'trajectory': [
            (0.0, 0.0, 0.0, 0.0),
            (0.2, 80.0, 5.0, 30.0),
            (0.4, -80.0, -5.0, -30.0),
            (0.6, 80.0, 5.0, 30.0),
            (0.8, -80.0, -5.0, -30.0),
            (1.0, 0.0, 0.0, 0.0),
        ],
    },
    'wave_vertical': {
        'description': 'Hand waving up and down',
        'base_flex': (0.10, 0.10, 0.10, 0.10, 0.10),
        'trajectory': [
            (0.0, 0.0, 0.0, 0.0),
            (0.2, 5.0, 100.0, 5.0),
            (0.4, -5.0, -100.0, -5.0),
            (0.6, 5.0, 100.0, 5.0),
            (0.8, -5.0, -100.0, -5.0),
            (1.0, 0.0, 0.0, 0.0),
        ],
    },
    'scratch': {
        'description': 'Scratching motion with curled fingers',
        'base_flex': (0.60, 0.70, 0.70, 0.70, 0.70),
        'trajectory': [
            (0.0, 0.0, 0.0, 0.0),
            (0.15, 20.0, -40.0, 10.0),
            (0.30, -20.0, 40.0, -10.0),
            (0.45, 20.0, -40.0, 10.0),
            (0.60, -20.0, 40.0, -10.0),
            (0.75, 20.0, -40.0, 10.0),
            (0.90, -10.0, 20.0, -5.0),
            (1.0, 0.0, 0.0, 0.0),
        ],
    },
    'adjust_glove': {
        'description': 'Adjusting/fidgeting with the glove',
        'base_flex': (0.40, 0.50, 0.45, 0.50, 0.50),
        'trajectory': [
            (0.0, 0.0, 0.0, 0.0),
            (0.2, 30.0, 20.0, -15.0),
            (0.4, -15.0, -10.0, 25.0),
            (0.6, 10.0, 30.0, -20.0),
            (0.8, -20.0, -15.0, 10.0),
            (1.0, 0.0, 0.0, 0.0),
        ],
    },
    'random_gesture': {
        'description': 'Random meaningless hand movement',
        'base_flex': (0.30, 0.30, 0.30, 0.30, 0.30),
        'trajectory': [
            (0.0, 0.0, 0.0, 0.0),
            (0.15, 50.0, 50.0, 50.0),
            (0.35, -60.0, 30.0, -40.0),
            (0.55, 40.0, -70.0, 60.0),
            (0.75, -30.0, 40.0, -30.0),
            (1.0, 0.0, 0.0, 0.0),
        ],
    },
}


# =============================================================================
# Validation
# =============================================================================

def validate_definitions():
    """Verify all 26 letters are defined with correct vector shapes."""
    for letter in LETTERS:
        assert letter in SIGN_DEFINITIONS, f"Missing definition for '{letter}'"
        defn = SIGN_DEFINITIONS[letter]
        assert defn['vector'].shape == (NUM_FEATURES,), \
            f"Letter '{letter}' vector shape {defn['vector'].shape} != ({NUM_FEATURES},)"
        assert defn['variance'].shape == (NUM_FEATURES,), \
            f"Letter '{letter}' variance shape {defn['variance'].shape} != ({NUM_FEATURES},)"
        assert defn['dynamic'] == (letter in DYNAMIC_LETTERS), \
            f"Letter '{letter}' dynamic flag mismatch"

    # Verify dynamic trajectories exist for dynamic letters
    for letter in DYNAMIC_LETTERS:
        assert letter in DYNAMIC_TRAJECTORIES, \
            f"Missing trajectory keyframes for dynamic letter '{letter}'"

    # Verify distractor count
    assert len(DISTRACTOR_DEFINITIONS) >= 4, \
        f"Need ≥4 distractor classes, got {len(DISTRACTOR_DEFINITIONS)}"


# Run validation on import
validate_definitions()


# =============================================================================
# Convenience accessors
# =============================================================================

def get_static_definitions():
    """Return definitions for the 24 static letters only."""
    return {k: v for k, v in SIGN_DEFINITIONS.items() if not v['dynamic']}


def get_dynamic_definitions():
    """Return definitions for the 2 dynamic letters only."""
    return {k: v for k, v in SIGN_DEFINITIONS.items() if v['dynamic']}


def get_target_matrix():
    """Return (26, 36) matrix of target vectors, ordered A–Z."""
    return np.array([SIGN_DEFINITIONS[letter]['vector'] for letter in LETTERS])


def get_variance_matrix():
    """Return (26, 36) matrix of per-feature variances, ordered A–Z."""
    return np.array([SIGN_DEFINITIONS[letter]['variance'] for letter in LETTERS])
