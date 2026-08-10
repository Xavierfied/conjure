"""
hand_gesture_engine.py

Temporal hand-gesture detection engine for MediaPipe Hands.

Features:
- Scale-normalized pinch detection
- Smoothed distance
- Timestamp-based velocity
- Acceleration
- Contact hysteresis
- Pinch state machine
- One-shot PINCH / SNAP / RELEASE events
- Snap cooldown
- Horizontal / vertical wave detection
- Movement-noise rejection
- Confidence scores
- Per-hand reset support

Expected MediaPipe landmark format:

    landmarks.shape == (21, 3)

where each row is:

    [x, y, z]

MediaPipe landmark indices used:

    0  = Wrist
    4  = Thumb tip
    5  = Index MCP
    9  = Middle MCP
    12 = Middle tip
    13 = Ring MCP
"""


from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


# ============================================================
# ENUMS
# ============================================================

class PinchPhase(Enum):
    """
    Current continuous pinch state.
    """

    OPEN = auto()
    APPROACHING = auto()
    ARMED = auto()
    CONTACT = auto()
    HOLDING = auto()


class PinchEvent(Enum):
    """
    One-shot pinch events.

    These are different from PinchPhase.

    Example:

        phase = HOLDING
        event = NONE

    Or on the frame where a snap occurs:

        phase = CONTACT
        event = SNAP
    """

    NONE = auto()
    PINCH_START = auto()
    SNAP = auto()
    PINCH_RELEASE = auto()


class WaveState(Enum):
    """
    Current wave classification.
    """

    CALIBRATING = auto()
    NONE = auto()
    HORIZONTAL = auto()
    VERTICAL = auto()


class GestureEvent(Enum):
    """
    High-level events intended for the application layer.
    """

    NONE = auto()
    PINCH_START = auto()
    PINCH_SNAP = auto()
    PINCH_RELEASE = auto()
    HORIZONTAL_WAVE = auto()
    VERTICAL_WAVE = auto()


# ============================================================
# HELPERS
# ============================================================

def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """
    Restrict a value to [minimum, maximum].
    """
    return max(minimum, min(maximum, value))


class EMAFilter:
    """
    Exponential Moving Average filter.

    alpha closer to 1.0:
        More responsive
        Less smoothing

    alpha closer to 0.0:
        More smoothing
        More latency
    """

    def __init__(self, alpha: float = 0.45):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(
                "alpha must be > 0 and <= 1"
            )

        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, value: float) -> float:
        if self.value is None:
            self.value = value
        else:
            self.value = (
                self.alpha * value
                + (1.0 - self.alpha) * self.value
            )

        return self.value

    def reset(self) -> None:
        self.value = None


# ============================================================
# PINCH RESULT
# ============================================================

@dataclass
class PinchResult:
    """
    Output from ContextualPinchDetector.
    """

    phase: PinchPhase
    event: PinchEvent

    distance: float
    velocity: float
    acceleration: float

    in_contact: bool
    confidence: float


# ============================================================
# PINCH DETECTOR
# ============================================================

class ContextualPinchDetector:
    """
    Detects thumb-middle pinch / snap gestures.

    State flow:

        OPEN
          |
          | closing motion
          v
        APPROACHING
          |
          | strong approach
          v
        ARMED
          |
          | contact
          v
        CONTACT
          |
          v
        HOLDING
          |
          | release
          v
        OPEN


    A SNAP is generated as a one-shot event when an
    intentional fast approach reaches contact.

    This detector uses:

        - normalized distance
        - smoothing
        - velocity
        - acceleration
        - hysteresis
        - temporal state
        - cooldown
    """

    def __init__(
        self,

        # ----------------------------------------------------
        # Contact thresholds
        # ----------------------------------------------------

        contact_enter: float = 0.18,
        contact_exit: float = 0.24,

        # ----------------------------------------------------
        # Motion thresholds
        # ----------------------------------------------------

        approach_velocity: float = 0.10,
        arm_velocity: float = 0.13,
        snap_velocity: float = 0.07,

        # ----------------------------------------------------
        # History / smoothing
        # ----------------------------------------------------

        history_size: int = 8,
        smoothing_alpha: float = 0.45,

        # ----------------------------------------------------
        # Temporal requirements
        # ----------------------------------------------------

        min_approach_frames: int = 2,

        # Prevent repeated snaps.
        snap_cooldown: float = 0.18,
    ):
        self.contact_enter = contact_enter
        self.contact_exit = contact_exit

        self.approach_velocity = approach_velocity
        self.arm_velocity = arm_velocity
        self.snap_velocity = snap_velocity

        self.min_approach_frames = (
            min_approach_frames
        )

        self.snap_cooldown = snap_cooldown

        self.distance_history = deque(
            maxlen=history_size
        )

        self.velocity_history = deque(
            maxlen=history_size
        )

        self.distance_filter = EMAFilter(
            smoothing_alpha
        )

        self.phase = PinchPhase.OPEN

        self.last_timestamp: Optional[float] = None
        self.last_distance: Optional[float] = None

        self.last_velocity = 0.0

        self.approach_frames = 0

        self.last_snap_timestamp: Optional[float] = None

        self.was_contact = False

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:
        """
        Clear all temporal state.

        Call this when MediaPipe loses the hand.
        """

        self.distance_history.clear()
        self.velocity_history.clear()

        self.distance_filter.reset()

        self.phase = PinchPhase.OPEN

        self.last_timestamp = None
        self.last_distance = None

        self.last_velocity = 0.0

        self.approach_frames = 0

        self.last_snap_timestamp = None

        self.was_contact = False

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        landmarks: np.ndarray,
        palm_scale: float,
        timestamp: Optional[float] = None,
    ) -> PinchResult:

        if palm_scale <= 0:
            return PinchResult(
                phase=self.phase,
                event=PinchEvent.NONE,
                distance=1.0,
                velocity=0.0,
                acceleration=0.0,
                in_contact=False,
                confidence=0.0,
            )

        # ----------------------------------------------------
        # Validate landmarks
        # ----------------------------------------------------

        if landmarks.shape[0] < 21:
            raise ValueError(
                "Expected 21 MediaPipe hand landmarks."
            )

        # ----------------------------------------------------
        # Extract thumb + middle fingertips
        # ----------------------------------------------------

        thumb_tip = landmarks[4]
        middle_tip = landmarks[12]

        # ----------------------------------------------------
        # Scale-normalized distance
        # ----------------------------------------------------

        raw_distance = (
            np.linalg.norm(
                thumb_tip - middle_tip
            )
            / palm_scale
        )

        distance = self.distance_filter.update(
            float(raw_distance)
        )

        self.distance_history.append(
            distance
        )

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        if timestamp is None:

            if self.last_timestamp is None:
                timestamp = 0.0
            else:
                # Assume ~30 FPS when timestamp isn't supplied.
                timestamp = (
                    self.last_timestamp
                    + (1.0 / 30.0)
                )

        if self.last_timestamp is None:
            dt = 1.0 / 30.0
        else:
            dt = timestamp - self.last_timestamp

            # Protect against bad timestamps.
            dt = max(
                0.001,
                min(dt, 0.2),
            )

        self.last_timestamp = timestamp

        # ----------------------------------------------------
        # Velocity
        # ----------------------------------------------------

        if self.last_distance is None:
            velocity = 0.0
        else:
            # Positive velocity means fingers are
            # moving toward each other.
            velocity = (
                -(distance - self.last_distance)
                / dt
            )

        self.last_distance = distance

        # ----------------------------------------------------
        # Smooth velocity
        # ----------------------------------------------------

        self.velocity_history.append(
            velocity
        )

        if len(self.velocity_history) >= 2:
            velocity = float(
                np.mean(
                    list(
                        self.velocity_history
                    )[-2:]
                )
            )

        # ----------------------------------------------------
        # Acceleration
        # ----------------------------------------------------

        acceleration = (
            (velocity - self.last_velocity)
            / dt
        )

        self.last_velocity = velocity

        # ----------------------------------------------------
        # Contact hysteresis
        # ----------------------------------------------------

        if self.was_contact:

            # Once contact has been established,
            # don't release until we move past the
            # larger exit threshold.
            in_contact = (
                distance < self.contact_exit
            )

        else:

            # Enter contact using the smaller threshold.
            in_contact = (
                distance < self.contact_enter
            )

        self.was_contact = in_contact

        # ----------------------------------------------------
        # Approach detection
        # ----------------------------------------------------

        is_approaching = (
            velocity > self.approach_velocity
        )

        is_fast_approach = (
            velocity > self.arm_velocity
        )

        if is_approaching:
            self.approach_frames += 1
        else:
            self.approach_frames = 0

        sustained_approach = (
            self.approach_frames
            >= self.min_approach_frames
        )

        # ----------------------------------------------------
        # Snap cooldown
        # ----------------------------------------------------

        cooldown_active = False

        if self.last_snap_timestamp is not None:

            cooldown_active = (
                timestamp
                - self.last_snap_timestamp
                < self.snap_cooldown
            )

        # ----------------------------------------------------
        # Default event
        # ----------------------------------------------------

        event = PinchEvent.NONE

        # ====================================================
        # STATE MACHINE
        # ====================================================

        # ----------------------------------------------------
        # OPEN
        # ----------------------------------------------------

        if self.phase == PinchPhase.OPEN:

            if sustained_approach:

                self.phase = (
                    PinchPhase.APPROACHING
                )

        # ----------------------------------------------------
        # APPROACHING
        # ----------------------------------------------------

        elif self.phase == PinchPhase.APPROACHING:

            if in_contact:

                self.phase = PinchPhase.CONTACT

                if not cooldown_active:

                    event = (
                        PinchEvent.PINCH_START
                    )

                    if (
                        is_fast_approach
                        or velocity
                        > self.snap_velocity
                    ):
                        event = (
                            PinchEvent.SNAP
                        )

                        self.last_snap_timestamp = (
                            timestamp
                        )

            elif not is_approaching:

                self.phase = PinchPhase.OPEN

            elif is_fast_approach:

                self.phase = PinchPhase.ARMED

        # ----------------------------------------------------
        # ARMED
        # ----------------------------------------------------

        elif self.phase == PinchPhase.ARMED:

            if in_contact:

                self.phase = PinchPhase.CONTACT

                if not cooldown_active:

                    event = (
                        PinchEvent.SNAP
                    )

                    self.last_snap_timestamp = (
                        timestamp
                    )

            elif not is_approaching:

                self.phase = PinchPhase.OPEN

        # ----------------------------------------------------
        # CONTACT
        # ----------------------------------------------------

        elif self.phase == PinchPhase.CONTACT:

            if not in_contact:

                self.phase = PinchPhase.OPEN

                event = (
                    PinchEvent.PINCH_RELEASE
                )

            else:

                self.phase = (
                    PinchPhase.HOLDING
                )

        # ----------------------------------------------------
        # HOLDING
        # ----------------------------------------------------

        elif self.phase == PinchPhase.HOLDING:

            if not in_contact:

                self.phase = PinchPhase.OPEN

                event = (
                    PinchEvent.PINCH_RELEASE
                )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = (
            self._calculate_confidence(
                distance=distance,
                velocity=velocity,
                in_contact=in_contact,
            )
        )

        return PinchResult(
            phase=self.phase,
            event=event,
            distance=distance,
            velocity=velocity,
            acceleration=acceleration,
            in_contact=in_contact,
            confidence=confidence,
        )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    def _calculate_confidence(
        self,
        distance: float,
        velocity: float,
        in_contact: bool,
    ) -> float:

        if in_contact:

            contact_score = clamp(
                1.0
                - (
                    distance
                    / max(
                        self.contact_exit,
                        1e-6,
                    )
                )
            )

        else:
            contact_score = 0.0

        velocity_score = clamp(
            velocity
            / max(
                self.arm_velocity,
                1e-6,
            )
        )

        confidence = (
            0.65 * contact_score
            + 0.35 * velocity_score
        )

        return clamp(confidence)


# ============================================================
# WAVE RESULT
# ============================================================

@dataclass
class WaveResult:
    """
    Output from ContextualWaveDetector.
    """

    state: WaveState

    x_range: float
    y_range: float

    x_swings: int
    y_swings: int

    confidence: float


# ============================================================
# WAVE DETECTOR
# ============================================================

class ContextualWaveDetector:
    """
    Detects horizontal / vertical waves using wrist trajectory.

    The detector:

        1. Smooths wrist position.
        2. Stores a temporal window.
        3. Calculates movement range.
        4. Calculates meaningful direction changes.
        5. Rejects tiny tracking noise.
        6. Checks axis dominance.
        7. Produces a confidence score.
    """

    def __init__(
        self,

        # Number of frames in trajectory window.
        window_size: int = 20,

        # Minimum normalized movement range.
        range_threshold: float = 0.45,

        # Minimum meaningful direction changes.
        swing_threshold: int = 2,

        # Dominant axis must exceed the other axis
        # by this factor.
        axis_dominance: float = 1.30,

        # Ignore movements smaller than this.
        movement_epsilon: float = 0.015,

        # Position smoothing.
        smoothing_alpha: float = 0.40,
    ):

        self.window_size = window_size

        self.range_threshold = (
            range_threshold
        )

        self.swing_threshold = (
            swing_threshold
        )

        self.axis_dominance = (
            axis_dominance
        )

        self.movement_epsilon = (
            movement_epsilon
        )

        self.position_filter_x = (
            EMAFilter(smoothing_alpha)
        )

        self.position_filter_y = (
            EMAFilter(smoothing_alpha)
        )

        self.wrist_history = deque(
            maxlen=window_size
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:

        self.wrist_history.clear()

        self.position_filter_x.reset()
        self.position_filter_y.reset()

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        landmarks: np.ndarray,
        palm_scale: float,
    ) -> WaveResult:

        if palm_scale <= 0:

            return WaveResult(
                state=WaveState.CALIBRATING,
                x_range=0.0,
                y_range=0.0,
                x_swings=0,
                y_swings=0,
                confidence=0.0,
            )

        if landmarks.shape[0] < 21:

            raise ValueError(
                "Expected 21 MediaPipe hand landmarks."
            )

        # ----------------------------------------------------
        # Wrist
        # ----------------------------------------------------

        wrist = landmarks[0][:2]

        x = self.position_filter_x.update(
            float(wrist[0])
        )

        y = self.position_filter_y.update(
            float(wrist[1])
        )

        # ----------------------------------------------------
        # Normalize by hand scale
        # ----------------------------------------------------

        normalized_point = np.array(
            [
                x / palm_scale,
                y / palm_scale,
            ],
            dtype=np.float32,
        )

        self.wrist_history.append(
            normalized_point
        )

        # ----------------------------------------------------
        # Calibration
        # ----------------------------------------------------

        if (
            len(self.wrist_history)
            < self.window_size
        ):

            return WaveResult(
                state=WaveState.CALIBRATING,
                x_range=0.0,
                y_range=0.0,
                x_swings=0,
                y_swings=0,
                confidence=0.0,
            )

        pts = np.asarray(
            self.wrist_history
        )

        x_values = pts[:, 0]
        y_values = pts[:, 1]

        # ----------------------------------------------------
        # Movement range
        # ----------------------------------------------------

        x_range = float(
            np.ptp(x_values)
        )

        y_range = float(
            np.ptp(y_values)
        )

        # ----------------------------------------------------
        # Frame-to-frame movement
        # ----------------------------------------------------

        dx = np.diff(x_values)
        dy = np.diff(y_values)

        # Ignore MediaPipe jitter.
        dx[
            np.abs(dx)
            < self.movement_epsilon
        ] = 0.0

        dy[
            np.abs(dy)
            < self.movement_epsilon
        ] = 0.0

        # ----------------------------------------------------
        # Direction changes
        # ----------------------------------------------------

        x_swings = (
            self._count_direction_changes(dx)
        )

        y_swings = (
            self._count_direction_changes(dy)
        )

        # ====================================================
        # HORIZONTAL
        # ====================================================

        horizontal_possible = (
            x_range
            > self.range_threshold

            and x_swings
            >= self.swing_threshold

            and x_range
            > (
                self.axis_dominance
                * max(
                    y_range,
                    1e-6,
                )
            )
        )

        # ====================================================
        # VERTICAL
        # ====================================================

        vertical_possible = (
            y_range
            > self.range_threshold

            and y_swings
            >= self.swing_threshold

            and y_range
            > (
                self.axis_dominance
                * max(
                    x_range,
                    1e-6,
                )
            )
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        if horizontal_possible:

            confidence = (
                self._wave_confidence(
                    dominant_range=x_range,
                    secondary_range=y_range,
                    swings=x_swings,
                )
            )

            return WaveResult(
                state=WaveState.HORIZONTAL,
                x_range=x_range,
                y_range=y_range,
                x_swings=x_swings,
                y_swings=y_swings,
                confidence=confidence,
            )

        if vertical_possible:

            confidence = (
                self._wave_confidence(
                    dominant_range=y_range,
                    secondary_range=x_range,
                    swings=y_swings,
                )
            )

            return WaveResult(
                state=WaveState.VERTICAL,
                x_range=x_range,
                y_range=y_range,
                x_swings=x_swings,
                y_swings=y_swings,
                confidence=confidence,
            )

        return WaveResult(
            state=WaveState.NONE,
            x_range=x_range,
            y_range=y_range,
            x_swings=x_swings,
            y_swings=y_swings,
            confidence=0.0,
        )

    # ========================================================
    # DIRECTION CHANGES
    # ========================================================

    @staticmethod
    def _count_direction_changes(
        velocity: np.ndarray,
    ) -> int:

        # Ignore zero values caused by noise filtering.
        non_zero = velocity[
            velocity != 0
        ]

        if len(non_zero) < 2:
            return 0

        signs = np.sign(
            non_zero
        )

        return int(
            np.count_nonzero(
                np.diff(signs)
            )
        )

    # ========================================================
    # WAVE CONFIDENCE
    # ========================================================

    def _wave_confidence(
        self,
        dominant_range: float,
        secondary_range: float,
        swings: int,
    ) -> float:

        # Amplitude.
        amplitude_score = clamp(
            dominant_range
            / (
                self.range_threshold
                * 2.0
            )
        )

        # Axis dominance.
        dominance_score = clamp(
            (
                dominant_range
                / max(
                    secondary_range,
                    1e-6,
                )
            )
            / self.axis_dominance
        )

        # Swing count.
        swing_score = clamp(
            swings / 4.0
        )

        confidence = (
            0.45 * amplitude_score
            + 0.35 * dominance_score
            + 0.20 * swing_score
        )

        return clamp(
            confidence
        )


# ============================================================
# COMPLETE HAND RESULT
# ============================================================

@dataclass
class HandGestureResult:
    """
    Complete result from HandGestureEngine.
    """

    palm_scale: float

    pinch: PinchResult
    wave: WaveResult

    events: list[GestureEvent] = field(
        default_factory=list
    )


# ============================================================
# HAND GESTURE ENGINE
# ============================================================

class HandGestureEngine:
    """
    Main per-hand gesture engine.

    Create ONE instance for each tracked hand.

    Example:

        left_engine = HandGestureEngine()
        right_engine = HandGestureEngine()

    Do not share temporal state between hands.
    """

    def __init__(self):

        self.pinch_detector = (
            ContextualPinchDetector()
        )

        self.wave_detector = (
            ContextualWaveDetector()
        )

        self.last_wave_state = (
            WaveState.NONE
        )

    # ========================================================
    # PALM SCALE
    # ========================================================

    @staticmethod
    def compute_palm_scale(
        landmarks: np.ndarray,
    ) -> float:
        """
        Estimate hand scale using multiple palm distances.

        Using multiple MCP points is more stable than relying
        only on wrist -> middle MCP.
        """

        wrist = landmarks[0]

        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]

        distances = [
            np.linalg.norm(
                wrist - index_mcp
            ),

            np.linalg.norm(
                wrist - middle_mcp
            ),

            np.linalg.norm(
                wrist - ring_mcp
            ),
        ]

        scale = float(
            np.mean(distances)
        )

        return max(
            scale,
            1e-6,
        )

    # ========================================================
    # PROCESS
    # ========================================================

    def process(
        self,
        landmarks: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> HandGestureResult:
        """
        Process one MediaPipe hand for one frame.

        Parameters
        ----------
        landmarks:
            NumPy array of shape (21, 3).

        timestamp:
            Timestamp in seconds.

            Prefer:

                time.monotonic()

            from your webcam loop.
        """

        if landmarks.shape[0] < 21:

            raise ValueError(
                "Expected 21 MediaPipe hand landmarks."
            )

        # ----------------------------------------------------
        # Palm scale
        # ----------------------------------------------------

        palm_scale = (
            self.compute_palm_scale(
                landmarks
            )
        )

        # ----------------------------------------------------
        # Pinch
        # ----------------------------------------------------

        pinch = (
            self.pinch_detector.update(
                landmarks=landmarks,
                palm_scale=palm_scale,
                timestamp=timestamp,
            )
        )

        # ----------------------------------------------------
        # Wave
        # ----------------------------------------------------

        wave = (
            self.wave_detector.update(
                landmarks=landmarks,
                palm_scale=palm_scale,
            )
        )

        # ----------------------------------------------------
        # High-level events
        # ----------------------------------------------------

        events: list[GestureEvent] = []

        # ====================================================
        # PINCH EVENTS
        # ====================================================

        if (
            pinch.event
            == PinchEvent.PINCH_START
        ):

            events.append(
                GestureEvent.PINCH_START
            )

        elif (
            pinch.event
            == PinchEvent.SNAP
        ):

            events.append(
                GestureEvent.PINCH_SNAP
            )

        elif (
            pinch.event
            == PinchEvent.PINCH_RELEASE
        ):

            events.append(
                GestureEvent.PINCH_RELEASE
            )

        # ====================================================
        # WAVE EVENTS
        # ====================================================

        # Only generate an event when entering
        # a wave state.

        if (
            wave.state
            == WaveState.HORIZONTAL
        ):

            if (
                self.last_wave_state
                != WaveState.HORIZONTAL
            ):

                events.append(
                    GestureEvent.HORIZONTAL_WAVE
                )

        elif (
            wave.state
            == WaveState.VERTICAL
        ):

            if (
                self.last_wave_state
                != WaveState.VERTICAL
            ):

                events.append(
                    GestureEvent.VERTICAL_WAVE
                )

        self.last_wave_state = (
            wave.state
        )

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        return HandGestureResult(
            palm_scale=palm_scale,
            pinch=pinch,
            wave=wave,
            events=events,
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:
        """
        Reset all temporal state.

        Call this when the corresponding hand disappears
        from the MediaPipe tracking result.
        """

        self.pinch_detector.reset()
        self.wave_detector.reset()

        self.last_wave_state = (
            WaveState.NONE
        )


# ============================================================
# OPTIONAL SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    """
    This does NOT use a webcam.

    It simply verifies that the engine can be constructed
    and accepts a 21x3 landmark array.
    """

    engine = HandGestureEngine()

    # Fake MediaPipe-style landmarks.
    fake_landmarks = np.zeros(
        (21, 3),
        dtype=np.float32,
    )

    # Give the palm some non-zero dimensions.
    fake_landmarks[0] = [0.50, 0.70, 0.00]
    fake_landmarks[5] = [0.40, 0.50, 0.00]
    fake_landmarks[9] = [0.50, 0.45, 0.00]
    fake_landmarks[13] = [0.60, 0.50, 0.00]

    # Thumb tip.
    fake_landmarks[4] = [
        0.40,
        0.40,
        0.00,
    ]

    # Middle tip.
    fake_landmarks[12] = [
        0.60,
        0.30,
        0.00,
    ]

    result = engine.process(
        fake_landmarks,
        timestamp=0.0,
    )

    print("Gesture engine initialized successfully.")
    print("Palm scale:", result.palm_scale)
    print("Pinch phase:", result.pinch.phase.name)
    print("Pinch event:", result.pinch.event.name)
    print("Wave state:", result.wave.state.name)

