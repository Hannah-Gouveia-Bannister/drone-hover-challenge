"""
Vision scaffold for reading two camera feeds and estimating drone position.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import cv2
import numpy as np

from controller import PositionEstimate


@dataclass
class Detection2D:
    x: float
    y: float
    confidence: float


@dataclass
class StereoDetection:
    camera_a: Optional[Detection2D]
    camera_b: Optional[Detection2D]
    timestamp: float


@dataclass
class AttitudeEstimate:
    pitch: float
    roll: float


@dataclass
class StereoObservation:
    detection: StereoDetection
    attitude: Optional[AttitudeEstimate]


class DroneTracker(Protocol):
    def detect(self, frame: cv2.typing.MatLike) -> Optional[Detection2D]:
        ...


class AttitudeProvider(Protocol):
    def estimate(
        self,
        frame_a: cv2.typing.MatLike,
        frame_b: cv2.typing.MatLike,
        detection_a: Optional[Detection2D],
        detection_b: Optional[Detection2D],
    ) -> Optional[AttitudeEstimate]:
        ...


class SupportsLevelCalibration(Protocol):
    def recalibrate_level(self) -> None:
        ...


@dataclass
class HSVRange:
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


@dataclass
class MarkerPairDetection:
    front: Detection2D
    rear: Detection2D


class ColorBlobTracker:
    def __init__(
        self,
        hsv_ranges: list[HSVRange],
        min_area: float = 150.0,
        blur_size: int = 5,
        morph_kernel_size: int = 5,
    ) -> None:
        self.hsv_ranges = hsv_ranges
        self.min_area = min_area
        self.blur_size = blur_size
        self.kernel = np.ones(
            (morph_kernel_size, morph_kernel_size), dtype=np.uint8
        )

    def detect(self, frame: cv2.typing.MatLike) -> Optional[Detection2D]:
        contour = self.extract_contour(frame)
        if contour is None:
            return None

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None

        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
        frame_area = float(frame.shape[0] * frame.shape[1])
        area = cv2.contourArea(contour)
        confidence = min(1.0, area / max(frame_area * 0.05, 1.0))

        return Detection2D(x=center_x, y=center_y, confidence=confidence)

    def extract_contour(
        self, frame: cv2.typing.MatLike
    ) -> Optional[cv2.typing.MatLike]:
        mask = self.build_mask(frame)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < self.min_area:
            return None

        return contour

    def build_mask(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        blurred = cv2.GaussianBlur(frame, (self.blur_size, self.blur_size), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for hsv_range in self.hsv_ranges:
            partial_mask = cv2.inRange(
                hsv,
                np.array(hsv_range.lower, dtype=np.uint8),
                np.array(hsv_range.upper, dtype=np.uint8),
            )
            mask = cv2.bitwise_or(mask, partial_mask)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    def annotate(
        self, frame: cv2.typing.MatLike, detection: Optional[Detection2D]
    ) -> cv2.typing.MatLike:
        annotated = frame.copy()
        if detection is None:
            return annotated

        center = (int(detection.x), int(detection.y))
        cv2.circle(annotated, center, 10, (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{detection.confidence:.2f}",
            (center[0] + 12, center[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        return annotated


class NullTracker(ColorBlobTracker):
    def __init__(self) -> None:
        super().__init__(hsv_ranges=[])

    def detect(self, frame: cv2.typing.MatLike) -> Optional[Detection2D]:
        return None


class NullAttitudeProvider:
    def estimate(
        self,
        frame_a: cv2.typing.MatLike,
        frame_b: cv2.typing.MatLike,
        detection_a: Optional[Detection2D],
        detection_b: Optional[Detection2D],
    ) -> Optional[AttitudeEstimate]:
        return None


class CompositeTracker:
    def __init__(self, trackers: list[DroneTracker]) -> None:
        self.trackers = trackers

    def detect(self, frame: cv2.typing.MatLike) -> Optional[Detection2D]:
        for tracker in self.trackers:
            detection = tracker.detect(frame)
            if detection is not None:
                return detection
        return None


class CompositeAttitudeProvider:
    def __init__(self, providers: list[AttitudeProvider]) -> None:
        self.providers = providers

    def estimate(
        self,
        frame_a: cv2.typing.MatLike,
        frame_b: cv2.typing.MatLike,
        detection_a: Optional[Detection2D],
        detection_b: Optional[Detection2D],
    ) -> Optional[AttitudeEstimate]:
        for provider in self.providers:
            estimate = provider.estimate(
                frame_a,
                frame_b,
                detection_a,
                detection_b,
            )
            if estimate is not None:
                return estimate
        return None

    def recalibrate_level(self) -> None:
        for provider in self.providers:
            recalibrate = getattr(provider, "recalibrate_level", None)
            if callable(recalibrate):
                recalibrate()


class TwoColorMarkerTracker:
    def __init__(
        self,
        front_tracker: ColorBlobTracker,
        rear_tracker: ColorBlobTracker,
    ) -> None:
        self.front_tracker = front_tracker
        self.rear_tracker = rear_tracker

    def detect_pair(
        self, frame: cv2.typing.MatLike
    ) -> Optional[MarkerPairDetection]:
        front = self.front_tracker.detect(frame)
        rear = self.rear_tracker.detect(frame)
        if front is None or rear is None:
            return None
        return MarkerPairDetection(front=front, rear=rear)

    def detect(self, frame: cv2.typing.MatLike) -> Optional[Detection2D]:
        pair = self.detect_pair(frame)
        if pair is None:
            return None

        center_x = (pair.front.x + pair.rear.x) / 2.0
        center_y = (pair.front.y + pair.rear.y) / 2.0
        confidence = min(pair.front.confidence, pair.rear.confidence)
        return Detection2D(x=center_x, y=center_y, confidence=confidence)


class TwoPointMarkerAttitudeProvider:
    def __init__(
        self,
        tracker: TwoColorMarkerTracker,
        camera_a_axis: str = "roll",
        camera_b_axis: str = "pitch",
        camera_a_sign: float = 1.0,
        camera_b_sign: float = 1.0,
        angle_smoothing: float = 0.25,
        min_marker_length: float = 8.0,
        max_angle_degrees: float = 20.0,
    ) -> None:
        self.tracker = tracker
        self.camera_a_axis = camera_a_axis
        self.camera_b_axis = camera_b_axis
        self.camera_a_sign = camera_a_sign
        self.camera_b_sign = camera_b_sign
        self.angle_smoothing = angle_smoothing
        self.min_marker_length = min_marker_length
        self.max_angle_degrees = max_angle_degrees
        self._zero_angle_a: Optional[float] = None
        self._zero_angle_b: Optional[float] = None
        self._smoothed_pitch = 0.0
        self._smoothed_roll = 0.0

    def estimate(
        self,
        frame_a: cv2.typing.MatLike,
        frame_b: cv2.typing.MatLike,
        detection_a: Optional[Detection2D],
        detection_b: Optional[Detection2D],
    ) -> Optional[AttitudeEstimate]:
        pair_a = self.tracker.detect_pair(frame_a)
        pair_b = self.tracker.detect_pair(frame_b)
        if pair_a is None or pair_b is None:
            return None

        angle_a = self._pair_angle(pair_a)
        angle_b = self._pair_angle(pair_b)
        if angle_a is None or angle_b is None:
            return None

        if self._zero_angle_a is None:
            self._zero_angle_a = angle_a
        if self._zero_angle_b is None:
            self._zero_angle_b = angle_b

        camera_a_value = self.camera_a_sign * self._wrap_angle(
            angle_a - self._zero_angle_a
        )
        camera_b_value = self.camera_b_sign * self._wrap_angle(
            angle_b - self._zero_angle_b
        )

        pitch_measurement, roll_measurement = self._map_axes(
            camera_a_value,
            camera_b_value,
        )

        self._smoothed_pitch = self._smooth(self._smoothed_pitch, pitch_measurement)
        self._smoothed_roll = self._smooth(self._smoothed_roll, roll_measurement)

        return AttitudeEstimate(
            pitch=self._clamp_angle(self._smoothed_pitch),
            roll=self._clamp_angle(self._smoothed_roll),
        )

    def recalibrate_level(self) -> None:
        self._zero_angle_a = None
        self._zero_angle_b = None
        self._smoothed_pitch = 0.0
        self._smoothed_roll = 0.0

    def _pair_angle(self, pair: MarkerPairDetection) -> Optional[float]:
        dx = pair.front.x - pair.rear.x
        dy = pair.front.y - pair.rear.y
        if (dx * dx + dy * dy) ** 0.5 < self.min_marker_length:
            return None
        return self._wrap_angle(np.degrees(np.arctan2(dy, dx)))

    def _map_axes(
        self,
        camera_a_value: float,
        camera_b_value: float,
    ) -> tuple[float, float]:
        pitch_measurement = 0.0
        roll_measurement = 0.0

        if self.camera_a_axis == "pitch":
            pitch_measurement = camera_a_value
        else:
            roll_measurement = camera_a_value

        if self.camera_b_axis == "pitch":
            pitch_measurement = camera_b_value
        else:
            roll_measurement = camera_b_value

        return pitch_measurement, roll_measurement

    def _smooth(self, current: float, measurement: float) -> float:
        alpha = self.angle_smoothing
        return (1.0 - alpha) * current + alpha * measurement

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        wrapped = (angle + 180.0) % 360.0 - 180.0
        if wrapped > 90.0:
            wrapped -= 180.0
        if wrapped < -90.0:
            wrapped += 180.0
        return wrapped

    def _clamp_angle(self, angle: float) -> float:
        return max(-self.max_angle_degrees, min(self.max_angle_degrees, angle))


class ArucoMarkerTracker:
    def __init__(
        self,
        dictionary_name: str = "DICT_4X4_50",
        marker_id: Optional[int] = None,
    ) -> None:
        self.marker_id = marker_id
        self._dictionary = None
        self._detector = None
        if hasattr(cv2, "aruco"):
            dictionary_id = getattr(cv2.aruco, dictionary_name, None)
            if dictionary_id is not None:
                self._dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
                if hasattr(cv2.aruco, "ArucoDetector"):
                    self._detector = cv2.aruco.ArucoDetector(self._dictionary)

    def detect(self, frame: cv2.typing.MatLike) -> Optional[Detection2D]:
        marker = self.detect_marker(frame)
        if marker is None:
            return None
        return marker[0]

    def detect_marker(
        self, frame: cv2.typing.MatLike
    ) -> Optional[tuple[Detection2D, np.ndarray]]:
        if self._dictionary is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self._dictionary)

        if ids is None or len(corners) == 0:
            return None

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            if self.marker_id is not None and int(marker_id) != self.marker_id:
                continue

            corner_array = marker_corners.reshape(4, 2)
            center = corner_array.mean(axis=0)
            perimeter = cv2.arcLength(corner_array.astype(np.float32), True)
            confidence = min(1.0, perimeter / max(frame.shape[1] * 0.2, 1.0))
            detection = Detection2D(
                x=float(center[0]),
                y=float(center[1]),
                confidence=confidence,
            )
            return detection, corner_array

        return None


class ArucoAttitudeProvider:
    def __init__(
        self,
        tracker: ArucoMarkerTracker,
        camera_a_axis: str = "roll",
        camera_b_axis: str = "pitch",
        camera_a_sign: float = 1.0,
        camera_b_sign: float = 1.0,
        angle_smoothing: float = 0.2,
        max_angle_degrees: float = 20.0,
    ) -> None:
        self.tracker = tracker
        self.camera_a_axis = camera_a_axis
        self.camera_b_axis = camera_b_axis
        self.camera_a_sign = camera_a_sign
        self.camera_b_sign = camera_b_sign
        self.angle_smoothing = angle_smoothing
        self.max_angle_degrees = max_angle_degrees
        self._zero_angle_a: Optional[float] = None
        self._zero_angle_b: Optional[float] = None
        self._smoothed_pitch = 0.0
        self._smoothed_roll = 0.0

    def estimate(
        self,
        frame_a: cv2.typing.MatLike,
        frame_b: cv2.typing.MatLike,
        detection_a: Optional[Detection2D],
        detection_b: Optional[Detection2D],
    ) -> Optional[AttitudeEstimate]:
        marker_a = self.tracker.detect_marker(frame_a)
        marker_b = self.tracker.detect_marker(frame_b)
        if marker_a is None or marker_b is None:
            return None

        angle_a = self._marker_angle(marker_a[1])
        angle_b = self._marker_angle(marker_b[1])

        if self._zero_angle_a is None:
            self._zero_angle_a = angle_a
        if self._zero_angle_b is None:
            self._zero_angle_b = angle_b

        camera_a_value = self.camera_a_sign * self._wrap_angle(
            angle_a - self._zero_angle_a
        )
        camera_b_value = self.camera_b_sign * self._wrap_angle(
            angle_b - self._zero_angle_b
        )

        pitch_measurement = (
            camera_a_value if self.camera_a_axis == "pitch" else camera_b_value
        )
        roll_measurement = (
            camera_a_value if self.camera_a_axis == "roll" else camera_b_value
        )

        self._smoothed_pitch = self._smooth(self._smoothed_pitch, pitch_measurement)
        self._smoothed_roll = self._smooth(self._smoothed_roll, roll_measurement)

        return AttitudeEstimate(
            pitch=self._clamp_angle(self._smoothed_pitch),
            roll=self._clamp_angle(self._smoothed_roll),
        )

    def recalibrate_level(self) -> None:
        self._zero_angle_a = None
        self._zero_angle_b = None
        self._smoothed_pitch = 0.0
        self._smoothed_roll = 0.0

    @staticmethod
    def _marker_angle(corners: np.ndarray) -> float:
        top_left = corners[0]
        top_right = corners[1]
        dx = top_right[0] - top_left[0]
        dy = top_right[1] - top_left[1]
        wrapped = np.degrees(np.arctan2(dy, dx))
        return ((wrapped + 180.0) % 360.0) - 180.0

    def _smooth(self, current: float, measurement: float) -> float:
        alpha = self.angle_smoothing
        return (1.0 - alpha) * current + alpha * measurement

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        wrapped = (angle + 180.0) % 360.0 - 180.0
        if wrapped > 90.0:
            wrapped -= 180.0
        if wrapped < -90.0:
            wrapped += 180.0
        return wrapped

    def _clamp_angle(self, angle: float) -> float:
        return max(-self.max_angle_degrees, min(self.max_angle_degrees, angle))


class ContourAngleAttitudeProvider:
    def __init__(
        self,
        tracker: ColorBlobTracker,
        camera_a_axis: str = "roll",
        camera_b_axis: str = "pitch",
        camera_a_sign: float = 1.0,
        camera_b_sign: float = 1.0,
        angle_smoothing: float = 0.3,
        min_aspect_ratio: float = 1.2,
        max_angle_degrees: float = 20.0,
    ) -> None:
        self.tracker = tracker
        self.camera_a_axis = camera_a_axis
        self.camera_b_axis = camera_b_axis
        self.camera_a_sign = camera_a_sign
        self.camera_b_sign = camera_b_sign
        self.angle_smoothing = angle_smoothing
        self.min_aspect_ratio = min_aspect_ratio
        self.max_angle_degrees = max_angle_degrees
        self._zero_angle_a: Optional[float] = None
        self._zero_angle_b: Optional[float] = None
        self._smoothed_pitch = 0.0
        self._smoothed_roll = 0.0

    def estimate(
        self,
        frame_a: cv2.typing.MatLike,
        frame_b: cv2.typing.MatLike,
        detection_a: Optional[Detection2D],
        detection_b: Optional[Detection2D],
    ) -> Optional[AttitudeEstimate]:
        if detection_a is None or detection_b is None:
            return None

        angle_a = self._estimate_marker_angle(frame_a)
        angle_b = self._estimate_marker_angle(frame_b)
        if angle_a is None or angle_b is None:
            return None

        if self._zero_angle_a is None:
            self._zero_angle_a = angle_a
        if self._zero_angle_b is None:
            self._zero_angle_b = angle_b

        camera_a_value = self.camera_a_sign * self._wrap_angle(
            angle_a - self._zero_angle_a
        )
        camera_b_value = self.camera_b_sign * self._wrap_angle(
            angle_b - self._zero_angle_b
        )

        pitch_measurement = 0.0
        roll_measurement = 0.0

        if self.camera_a_axis == "pitch":
            pitch_measurement = camera_a_value
        else:
            roll_measurement = camera_a_value

        if self.camera_b_axis == "pitch":
            pitch_measurement = camera_b_value
        else:
            roll_measurement = camera_b_value

        self._smoothed_pitch = self._smooth(self._smoothed_pitch, pitch_measurement)
        self._smoothed_roll = self._smooth(self._smoothed_roll, roll_measurement)

        return AttitudeEstimate(
            pitch=self._clamp_angle(self._smoothed_pitch),
            roll=self._clamp_angle(self._smoothed_roll),
        )

    def recalibrate_level(self) -> None:
        self._zero_angle_a = None
        self._zero_angle_b = None
        self._smoothed_pitch = 0.0
        self._smoothed_roll = 0.0

    def _estimate_marker_angle(
        self, frame: cv2.typing.MatLike
    ) -> Optional[float]:
        contour = self.tracker.extract_contour(frame)
        if contour is None or len(contour) < 3:
            return None

        rect = cv2.minAreaRect(contour)
        (_, _), (width, height), raw_angle = rect
        if width <= 0 or height <= 0:
            return None

        aspect_ratio = max(width, height) / max(min(width, height), 1e-6)
        if aspect_ratio < self.min_aspect_ratio:
            return None

        angle = raw_angle
        if width < height:
            angle += 90.0

        return self._wrap_angle(angle)

    def _smooth(self, current: float, measurement: float) -> float:
        alpha = self.angle_smoothing
        return (1.0 - alpha) * current + alpha * measurement

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        wrapped = (angle + 180.0) % 360.0 - 180.0
        if wrapped > 90.0:
            wrapped -= 180.0
        if wrapped < -90.0:
            wrapped += 180.0
        return wrapped

    def _clamp_angle(self, angle: float) -> float:
        return max(-self.max_angle_degrees, min(self.max_angle_degrees, angle))


def default_tracker() -> ColorBlobTracker:
    # Assumption: the drone or attached marker has a bright red feature.
    return ColorBlobTracker(
        hsv_ranges=[
            HSVRange(lower=(0, 120, 70), upper=(10, 255, 255)),
            HSVRange(lower=(170, 120, 70), upper=(180, 255, 255)),
        ],
        min_area=120.0,
    )


def default_two_color_tracker() -> TwoColorMarkerTracker:
    front_tracker = ColorBlobTracker(
        hsv_ranges=[
            HSVRange(lower=(0, 120, 70), upper=(10, 255, 255)),
            HSVRange(lower=(170, 120, 70), upper=(180, 255, 255)),
        ],
        min_area=60.0,
    )
    rear_tracker = ColorBlobTracker(
        hsv_ranges=[
            HSVRange(lower=(90, 100, 60), upper=(130, 255, 255)),
        ],
        min_area=60.0,
    )
    return TwoColorMarkerTracker(front_tracker=front_tracker, rear_tracker=rear_tracker)


def default_hybrid_tracker() -> CompositeTracker:
    trackers: list[DroneTracker] = []

    aruco_tracker = ArucoMarkerTracker()
    if aruco_tracker._dictionary is not None:
        trackers.append(aruco_tracker)

    trackers.append(default_two_color_tracker())
    trackers.append(default_tracker())
    return CompositeTracker(trackers)


def default_attitude_provider(
    tracker: Optional[DroneTracker] = None,
) -> AttitudeProvider:
    providers: list[AttitudeProvider] = []

    aruco_tracker = ArucoMarkerTracker()
    if aruco_tracker._dictionary is not None:
        providers.append(
            ArucoAttitudeProvider(
                tracker=aruco_tracker,
                camera_a_axis="roll",
                camera_b_axis="pitch",
                camera_a_sign=1.0,
                camera_b_sign=1.0,
            )
        )

    two_color_tracker = default_two_color_tracker()
    providers.append(
        TwoPointMarkerAttitudeProvider(
            tracker=two_color_tracker,
            camera_a_axis="roll",
            camera_b_axis="pitch",
            camera_a_sign=1.0,
            camera_b_sign=1.0,
        )
    )

    contour_tracker = tracker if isinstance(tracker, ColorBlobTracker) else default_tracker()
    providers.append(
        ContourAngleAttitudeProvider(
            tracker=contour_tracker,
            camera_a_axis="roll",
            camera_b_axis="pitch",
            camera_a_sign=1.0,
            camera_b_sign=1.0,
            angle_smoothing=0.25,
            min_aspect_ratio=1.25,
            max_angle_degrees=20.0,
        )
    )
    return CompositeAttitudeProvider(providers)


class StereoVisionPositionProvider:
    def __init__(
        self,
        camera_a_index: int = 0,
        camera_b_index: int = 1,
        tracker: Optional[DroneTracker] = None,
        attitude_provider: Optional[AttitudeProvider] = None,
    ) -> None:
        self.camera_a_index = camera_a_index
        self.camera_b_index = camera_b_index
        self.tracker = tracker or default_hybrid_tracker()
        self.attitude_provider = attitude_provider or default_attitude_provider(
            self.tracker
        )
        self.cap_a: Optional[cv2.VideoCapture] = None
        self.cap_b: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        if self.cap_a is None:
            self.cap_a = cv2.VideoCapture(self.camera_a_index)
        if self.cap_b is None:
            self.cap_b = cv2.VideoCapture(self.camera_b_index)

        if not self.cap_a.isOpened() or not self.cap_b.isOpened():
            self.stop()
            raise RuntimeError("Unable to open both camera feeds")

    def stop(self) -> None:
        if self.cap_a is not None:
            self.cap_a.release()
            self.cap_a = None
        if self.cap_b is not None:
            self.cap_b.release()
            self.cap_b = None

    def recalibrate_level(self) -> None:
        recalibrate = getattr(self.attitude_provider, "recalibrate_level", None)
        if callable(recalibrate):
            recalibrate()

    def __enter__(self) -> "StereoVisionPositionProvider":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def get_position_estimate(self) -> Optional[PositionEstimate]:
        stereo_observation = self.get_stereo_observation()
        if stereo_observation is None:
            return None
        return self.triangulate(stereo_observation)

    def get_stereo_detection(self) -> Optional[StereoDetection]:
        stereo_observation = self.get_stereo_observation()
        if stereo_observation is None:
            return None
        return stereo_observation.detection

    def get_stereo_observation(self) -> Optional["StereoObservation"]:
        cap_a = self._require_camera(self.cap_a, "A")
        cap_b = self._require_camera(self.cap_b, "B")

        ok_a, frame_a = cap_a.read()
        ok_b, frame_b = cap_b.read()
        if not ok_a or not ok_b:
            return None

        detection_a = self.tracker.detect(frame_a)
        detection_b = self.tracker.detect(frame_b)
        if detection_a is None or detection_b is None:
            return None

        timestamp = time.monotonic()
        attitude = self.attitude_provider.estimate(
            frame_a,
            frame_b,
            detection_a,
            detection_b,
        )
        detection = StereoDetection(
            camera_a=detection_a,
            camera_b=detection_b,
            timestamp=timestamp,
        )
        return StereoObservation(
            detection=detection,
            attitude=attitude,
        )

    def triangulate(self, observation: "StereoObservation") -> PositionEstimate:
        detection = observation.detection
        if detection.camera_a is None or detection.camera_b is None:
            raise ValueError("Stereo detection requires both camera observations")

        # Placeholder geometry until the cage/camera calibration is known.
        x = (detection.camera_a.x + detection.camera_b.x) / 2.0
        y = (detection.camera_a.y + detection.camera_b.y) / 2.0
        disparity = abs(detection.camera_a.x - detection.camera_b.x)
        z = self._estimate_depth_from_disparity(disparity)

        actual_pitch = None
        actual_roll = None
        if observation.attitude is not None:
            actual_pitch = observation.attitude.pitch
            actual_roll = observation.attitude.roll

        return PositionEstimate(
            x=x,
            y=y,
            z=z,
            timestamp=detection.timestamp,
            actual_pitch=actual_pitch,
            actual_roll=actual_roll,
        )

    @staticmethod
    def _estimate_depth_from_disparity(disparity: float) -> float:
        if disparity <= 1e-6:
            return 0.0

        # Temporary stand-in until camera intrinsics and baseline are measured.
        return min(1.0, 0.2 + 100.0 / (disparity + 1.0))

    @staticmethod
    def _require_camera(
        camera: Optional[cv2.VideoCapture], label: str
    ) -> cv2.VideoCapture:
        if camera is None:
            raise RuntimeError(f"Camera {label} is not started")
        return camera


def main() -> None:
    with StereoVisionPositionProvider() as provider:
        while True:
            stereo_observation = provider.get_stereo_observation()
            if stereo_observation is None:
                print("No detection")
                time.sleep(0.1)
                continue

            estimate = provider.triangulate(stereo_observation)
            print(estimate)
            time.sleep(0.1)


if __name__ == "__main__":
    main()
