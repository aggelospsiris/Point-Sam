from __future__ import annotations

import argparse
import importlib
import math
import os
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip().lower() for item in raw.replace(",", " ").split() if item.strip()]


def env_int_list(name: str, default: str) -> list[int]:
    raw = os.getenv(name, default)
    values: list[int] = []
    for item in raw.replace(",", " ").split():
        try:
            values.append(int(item.strip()))
        except ValueError:
            continue
    return values


@dataclass
class PointingRay:
    origin_3d: np.ndarray
    direction_3d: np.ndarray
    start_2d: tuple[int, int]
    end_2d: tuple[int, int]
    quality: float = 0.0


@dataclass
class PointPrompt:
    pixel: tuple[int, int]
    point_3d: np.ndarray | None
    forward_m: float
    ray_offset_m: float
    source: str


@dataclass
class SegmentationResult:
    mask: np.ndarray
    score: float
    backend: str
    elapsed_ms: float


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.status = "starting"
        self.metrics: dict[str, float | int | str] = {}
        self.running = True

    def set_frame(self, image: np.ndarray, status: str, metrics: dict[str, float | int | str] | None = None) -> None:
        encode_start = time.perf_counter()
        quality = int(np.clip(env_int("JPEG_QUALITY", 90), 50, 100))
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return
        encode_ms = (time.perf_counter() - encode_start) * 1000.0
        with self.lock:
            self.jpeg = encoded.tobytes()
            self.status = status
            if metrics is not None:
                self.metrics = {**metrics, "jpeg_ms": round(encode_ms, 1)}

    def get_frame(self) -> tuple[bytes | None, str]:
        with self.lock:
            return self.jpeg, self.status

    def get_metrics(self) -> dict[str, float | int | str]:
        with self.lock:
            return dict(self.metrics)


class RealSenseCamera:
    def __init__(self) -> None:
        self.width = env_int("FRAME_WIDTH", 1280)
        self.height = env_int("FRAME_HEIGHT", 720)
        self.fps = env_int("FRAME_FPS", 30)
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.depth_scale = 0.001
        self.intrinsics: rs.intrinsics | None = None

    def start(self) -> None:
        requested = (self.width, self.height, self.fps)
        candidates = [
            requested,
            (1280, 720, min(self.fps, 30)),
            (848, 480, min(self.fps, 30)),
            (640, 480, min(self.fps, 30)),
        ]
        seen: set[tuple[int, int, int]] = set()
        last_error: Exception | None = None
        for width, height, fps in candidates:
            if (width, height, fps) in seen:
                continue
            seen.add((width, height, fps))
            config = rs.config()
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            try:
                profile = self.pipeline.start(config)
            except Exception as exc:
                last_error = exc
                continue

            self.width = width
            self.height = height
            self.fps = fps
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.intrinsics = color_profile.get_intrinsics()
            return

        raise RuntimeError(f"RealSense stream unavailable: {last_error}")

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("missing RealSense color/depth frame")
        color = np.asanyarray(color_frame.get_data())
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
        return color, depth_m

    def stop(self) -> None:
        self.pipeline.stop()


class RealSenseBag:
    """Finite RealSense bag source with the same aligned RGB-D output as the camera."""

    def __init__(self, bag_path: str | Path) -> None:
        self.path = Path(bag_path).expanduser().resolve()
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.playback: Any | None = None
        self.depth_scale = 0.001
        self.intrinsics: rs.intrinsics | None = None
        self.width = 0
        self.height = 0
        self.fps = 30
        self.last_timestamp_s: float | None = None
        self._finished = False

    def start(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"RealSense bag not found: {self.path}")

        config = rs.config()
        config.enable_device_from_file(str(self.path), repeat_playback=False)
        profile = self.pipeline.start(config)
        self.playback = profile.get_device().as_playback()
        # Rendering is driven by inference speed, not wall-clock playback speed.
        self.playback.set_real_time(False)

        try:
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
        except Exception as exc:
            raise RuntimeError("The bag does not contain a usable RealSense depth stream") from exc

        try:
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.intrinsics = color_profile.get_intrinsics()
            self.width = color_profile.width()
            self.height = color_profile.height()
            self.fps = max(1, int(color_profile.fps()))
        except Exception as exc:
            raise RuntimeError("The bag does not contain a usable RealSense color stream") from exc

    def read(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._finished or self.is_finished():
            self._finished = True
            return None

        try:
            frames = self.pipeline.wait_for_frames(5_000)
        except RuntimeError:
            if self.is_finished():
                self._finished = True
                return None
            raise

        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            if self.is_finished():
                self._finished = True
                return None
            raise RuntimeError("bag frame is missing aligned RealSense color or depth data")

        color = self.color_to_bgr(color_frame)
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
        self.last_timestamp_s = float(color_frame.get_timestamp()) / 1_000.0
        return color, depth_m

    def color_to_bgr(self, color_frame: Any) -> np.ndarray:
        color = np.asanyarray(color_frame.get_data())
        color_format = color_frame.profile.format()
        if color_format == rs.format.bgr8:
            return color
        if color_format == rs.format.rgb8:
            return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        if color_format == rs.format.bgra8:
            return cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
        if color_format == rs.format.rgba8:
            return cv2.cvtColor(color, cv2.COLOR_RGBA2BGR)
        if color_format == rs.format.yuyv:
            return cv2.cvtColor(color, cv2.COLOR_YUV2BGR_YUY2)
        raise RuntimeError(f"unsupported bag color format: {color_format}")

    def is_finished(self) -> bool:
        return self.playback is not None and self.playback.current_status() == rs.playback_status.stopped

    def stop(self) -> None:
        self.pipeline.stop()


class PointingEstimator:
    def __init__(self) -> None:
        self.backend = os.getenv("HAND_BACKEND", "mediapipe").strip().lower()
        self.smoothing = env_float("POINTING_SMOOTHING", 0.35)
        self.min_hand_ray_length_m = env_float("MIN_HAND_RAY_LENGTH_M", 0.035)
        self.keypoint_confidence = env_float("HAND_KEYPOINT_CONFIDENCE", 0.25)
        self.hand_keypoint_visibility = env_float("HAND_KEYPOINT_VISIBILITY", self.keypoint_confidence)
        self.pointing_start_landmark = env_int("POINTING_START_LANDMARK", 5)
        self.pointing_end_landmark = env_int("POINTING_END_LANDMARK", 8)
        self.finger_landmarks = env_int_list("POINTING_FINGER_LANDMARKS", "5,6,7,8")
        if len(self.finger_landmarks) < 2:
            self.finger_landmarks = [self.pointing_start_landmark, self.pointing_end_landmark]
        self.hand_depth_patch_radius = env_int("HAND_DEPTH_PATCH_RADIUS", 5)
        self.hand_depth_percentile = env_float("HAND_DEPTH_PERCENTILE", 25.0)
        self.hand_depth_cluster_m = env_float("HAND_DEPTH_CLUSTER_M", 0.08)
        self.ray_origin_advance_m = env_float("POINTING_RAY_ORIGIN_ADVANCE_M", 0.015)
        self.max_ray_jump_px = env_float("POINTING_MAX_JUMP_PX", 220.0)
        self.max_ray_angle_jump_deg = env_float("POINTING_MAX_ANGLE_JUMP_DEG", 35.0)
        self.smoothed_origin_3d: np.ndarray | None = None
        self.smoothed_direction_3d: np.ndarray | None = None
        self.smoothed_end_2d: tuple[int, int] | None = None
        self.last_hand_confidence = 0.0
        self.last_hand_candidates = 0
        self.last_ray_quality = 0.0

        if self.backend == "yolo":
            self.init_yolo()
        elif self.backend == "mediapipe":
            self.init_mediapipe()
        else:
            raise RuntimeError(f"unsupported HAND_BACKEND={self.backend!r}; use yolo or mediapipe")

    def init_yolo(self) -> None:
        from ultralytics import YOLO

        self.hand_model_path = os.getenv("HAND_MODEL_PATH", "models/yolo26_hand_pose_fp16.onnx")
        self.hand_confidence = env_float("HAND_CONFIDENCE", 0.10)
        self.hand_keypoint_visibility = env_float("HAND_KEYPOINT_VISIBILITY", 0.05)
        self.hand_imgsz = env_int("HAND_IMGSZ", 640)
        self.max_hands = env_int("MAX_HANDS", 1)
        self.hand_device = os.getenv("HAND_DEVICE", "0")
        self.hand_model_kind = os.path.splitext(self.hand_model_path)[1].lower()

        if not os.path.exists(self.hand_model_path):
            raise RuntimeError(f"YOLO hand model not found: {self.hand_model_path}")

        if self.hand_model_kind == ".pt":
            self.hand_model = YOLO(self.hand_model_path)
            return

        try:
            import onnxruntime as ort
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("ONNXRuntime GPU is not installed. Rebuild after updating requirements.txt.") from exc

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.hand_session = ort.InferenceSession(self.hand_model_path, providers=providers)
        self.hand_input = self.hand_session.get_inputs()[0]
        self.hand_input_name = self.hand_input.name
        self.hand_input_float16 = "float16" in self.hand_input.type

    def init_mediapipe(self) -> None:
        self.tracking_confidence = env_float("HAND_TRACKING_CONFIDENCE", self.keypoint_confidence)
        self.max_hands = env_int("MAX_HANDS", 1)
        self.hand_process_width = env_int("HAND_PROCESS_WIDTH", 640)
        self.hand_model_complexity = env_int("HAND_MODEL_COMPLEXITY", 1)

        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is not installed. Rebuild after updating requirements.txt.") from exc

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            model_complexity=self.hand_model_complexity,
            min_detection_confidence=self.keypoint_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )

    def estimate(self, image_bgr: np.ndarray, depth_m: np.ndarray, intrinsics: rs.intrinsics) -> PointingRay | None:
        if self.backend == "yolo":
            return self.estimate_yolo(image_bgr, depth_m, intrinsics)
        return self.estimate_mediapipe(image_bgr, depth_m, intrinsics)

    def estimate_yolo(self, image_bgr: np.ndarray, depth_m: np.ndarray, intrinsics: rs.intrinsics) -> PointingRay | None:
        if self.hand_model_kind == ".pt":
            return self.estimate_yolo_pt(image_bgr, depth_m, intrinsics)
        return self.estimate_yolo_onnx(image_bgr, depth_m, intrinsics)

    def estimate_yolo_pt(self, image_bgr: np.ndarray, depth_m: np.ndarray, intrinsics: rs.intrinsics) -> PointingRay | None:
        result = self.hand_model.predict(
            image_bgr,
            conf=self.hand_confidence,
            imgsz=self.hand_imgsz,
            device=self.hand_device,
            verbose=False,
        )[0]
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            self.reset_smoothing()
            self.last_hand_confidence = 0.0
            self.last_hand_candidates = 0
            return None

        boxes_conf = result.boxes.conf.detach().cpu().numpy()
        keypoints_xy = result.keypoints.xy.detach().cpu().numpy()
        if result.keypoints.conf is None:
            keypoints_conf = np.ones(keypoints_xy.shape[:2], dtype=np.float32)
        else:
            keypoints_conf = result.keypoints.conf.detach().cpu().numpy()

        self.last_hand_confidence = float(np.max(boxes_conf)) if len(boxes_conf) else 0.0
        self.last_hand_candidates = int(np.count_nonzero(boxes_conf >= self.hand_confidence))
        best_ray: PointingRay | None = None
        best_score = -1.0
        selected = 0

        for hand_index in np.argsort(-boxes_conf):
            if selected >= max(1, self.max_hands):
                break
            confidence = float(boxes_conf[hand_index])
            if confidence < self.hand_confidence:
                continue
            finger_points = self.yolo_finger_points(
                keypoints_xy[hand_index],
                keypoints_conf[hand_index],
                image_bgr.shape[1],
                image_bgr.shape[0],
            )
            ray = self.ray_from_finger_points(finger_points, depth_m, intrinsics)
            if ray is None:
                continue
            score = confidence + 0.35 * ray.quality
            selected += 1
            if score > best_score:
                best_score = score
                best_ray = ray

        if best_ray is None:
            self.reset_smoothing()
            return None
        return self.smooth_ray(best_ray)

    def estimate_yolo_onnx(self, image_bgr: np.ndarray, depth_m: np.ndarray, intrinsics: rs.intrinsics) -> PointingRay | None:
        height, width = image_bgr.shape[:2]
        tensor, scale, pad_x, pad_y = self.preprocess_yolo(image_bgr)
        output = self.hand_session.run(None, {self.hand_input_name: tensor})[0]
        detections = np.asarray(output).reshape(-1, output.shape[-1])
        if len(detections) == 0:
            self.reset_smoothing()
            self.last_hand_confidence = 0.0
            self.last_hand_candidates = 0
            return None

        confidences = detections[:, 4]
        self.last_hand_confidence = float(np.max(confidences))
        self.last_hand_candidates = int(np.count_nonzero(confidences >= self.hand_confidence))
        best_ray: PointingRay | None = None
        best_score = -1.0
        selected = 0

        for detection in detections[np.argsort(-detections[:, 4])]:
            if selected >= max(1, self.max_hands):
                break
            confidence = float(detection[4])
            if confidence < self.hand_confidence:
                continue
            keypoints = detection[6:].reshape(21, 3)
            finger_points = self.yolo_onnx_finger_points(keypoints, scale, pad_x, pad_y, width, height)
            ray = self.ray_from_finger_points(finger_points, depth_m, intrinsics)
            if ray is None:
                continue
            score = confidence + 0.35 * ray.quality
            selected += 1
            if score > best_score:
                best_score = score
                best_ray = ray

        if best_ray is None:
            self.reset_smoothing()
            return None
        return self.smooth_ray(best_ray)

    def preprocess_yolo(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        height, width = image_bgr.shape[:2]
        size = self.hand_imgsz
        scale = min(size / width, size / height)
        resized_w = max(1, int(round(width * scale)))
        resized_h = max(1, int(round(height * scale)))
        resized = cv2.resize(image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_x = (size - resized_w) / 2.0
        pad_y = (size - resized_h) / 2.0
        x0 = int(round(pad_x - 0.1))
        y0 = int(round(pad_y - 0.1))
        padded[y0:y0 + resized_h, x0:x0 + resized_w] = resized
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32) / 255.0
        if self.hand_input_float16:
            tensor = tensor.astype(np.float16)
        return tensor, scale, float(x0), float(y0)

    def yolo_keypoint_to_pixel(
        self,
        keypoint: np.ndarray,
        scale: float,
        pad_x: float,
        pad_y: float,
        width: int,
        height: int,
    ) -> tuple[int, int]:
        x = (float(keypoint[0]) - pad_x) / scale
        y = (float(keypoint[1]) - pad_y) / scale
        return int(np.clip(round(x), 0, width - 1)), int(np.clip(round(y), 0, height - 1))

    def yolo_finger_points(
        self,
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray,
        width: int,
        height: int,
    ) -> list[tuple[int, tuple[int, int], float]]:
        points: list[tuple[int, tuple[int, int], float]] = []
        for landmark_id in self.finger_landmarks:
            if landmark_id >= keypoints_xy.shape[0] or landmark_id >= keypoints_conf.shape[0]:
                continue
            confidence = float(keypoints_conf[landmark_id])
            if confidence < self.hand_keypoint_visibility:
                continue
            points.append((landmark_id, keypoint_to_pixel(keypoints_xy[landmark_id], width, height), confidence))
        return points

    def yolo_onnx_finger_points(
        self,
        keypoints: np.ndarray,
        scale: float,
        pad_x: float,
        pad_y: float,
        width: int,
        height: int,
    ) -> list[tuple[int, tuple[int, int], float]]:
        points: list[tuple[int, tuple[int, int], float]] = []
        for landmark_id in self.finger_landmarks:
            if landmark_id >= len(keypoints):
                continue
            confidence = float(keypoints[landmark_id, 2])
            if confidence < self.hand_keypoint_visibility:
                continue
            pixel = self.yolo_keypoint_to_pixel(keypoints[landmark_id], scale, pad_x, pad_y, width, height)
            points.append((landmark_id, pixel, confidence))
        return points

    def estimate_mediapipe(self, image_bgr: np.ndarray, depth_m: np.ndarray, intrinsics: rs.intrinsics) -> PointingRay | None:
        height, width = image_bgr.shape[:2]
        process_image = image_bgr
        if self.hand_process_width > 0 and width > self.hand_process_width:
            process_height = max(1, round(height * self.hand_process_width / width))
            process_image = cv2.resize(image_bgr, (self.hand_process_width, process_height), interpolation=cv2.INTER_AREA)

        image_rgb = cv2.cvtColor(process_image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        result = self.hands.process(image_rgb)
        if not result.multi_hand_landmarks:
            self.reset_smoothing()
            self.last_hand_confidence = 0.0
            self.last_hand_candidates = 0
            return None

        self.last_hand_confidence = 1.0
        self.last_hand_candidates = len(result.multi_hand_landmarks)
        best_ray: PointingRay | None = None
        best_score = -1.0
        handedness_scores = []
        if result.multi_handedness:
            handedness_scores = [float(item.classification[0].score) for item in result.multi_handedness]

        for hand_index, hand_landmarks in enumerate(result.multi_hand_landmarks):
            landmarks = hand_landmarks.landmark
            finger_points: list[tuple[int, tuple[int, int], float]] = []
            for landmark_id in self.finger_landmarks:
                if landmark_id >= len(landmarks):
                    continue
                confidence = mediapipe_landmark_confidence(landmarks[landmark_id])
                if confidence < self.hand_keypoint_visibility:
                    continue
                finger_points.append((landmark_id, landmark_to_pixel(landmarks[landmark_id], width, height), confidence))
            ray = self.ray_from_finger_points(finger_points, depth_m, intrinsics)
            if ray is None:
                continue
            score = handedness_scores[hand_index] if hand_index < len(handedness_scores) else 1.0
            score += 0.35 * ray.quality
            if score > best_score:
                best_score = score
                best_ray = ray

        if best_ray is None:
            self.reset_smoothing()
            return None
        return self.smooth_ray(best_ray)

    def ray_from_finger_points(
        self,
        finger_points: list[tuple[int, tuple[int, int], float]],
        depth_m: np.ndarray,
        intrinsics: rs.intrinsics,
    ) -> PointingRay | None:
        if len(finger_points) < 2:
            return None

        order = {landmark_id: index for index, landmark_id in enumerate(self.finger_landmarks)}
        finger_points = sorted(finger_points, key=lambda item: order.get(item[0], item[0]))
        samples: list[tuple[int, tuple[int, int], np.ndarray, float]] = []
        for landmark_id, pixel, confidence in finger_points:
            patch_radius = self.finger_depth_patch_radius(landmark_id)
            point_3d = deproject_pixel_with_patch_depth(
                pixel,
                depth_m,
                intrinsics,
                patch_radius=patch_radius,
                depth_percentile=self.hand_depth_percentile,
                depth_cluster_m=self.hand_depth_cluster_m,
            )
            if point_3d is not None:
                samples.append((landmark_id, pixel, point_3d, confidence))

        if len(samples) < 2:
            return self.ray_from_legacy_finger_points(finger_points, depth_m, intrinsics)

        points_3d = [sample[2] for sample in samples]
        overall = points_3d[-1] - points_3d[0]
        overall_length = float(np.linalg.norm(overall))
        if overall_length < self.min_hand_ray_length_m:
            return None

        overall_direction = overall / overall_length
        segment_directions: list[np.ndarray] = []
        segment_weights: list[float] = []
        path_length = 0.0
        segment_agreement = 0.0
        agreement_weight = 0.0
        for index in range(len(samples) - 1):
            start = samples[index]
            end = samples[index + 1]
            vector = end[2] - start[2]
            length = float(np.linalg.norm(vector))
            if length < 0.008:
                continue
            direction = vector / length
            agreement = float(np.dot(direction, overall_direction))
            path_length += length
            if agreement < math.cos(math.radians(65.0)):
                continue
            distal_weight = 1.0 + 0.25 * index
            confidence_weight = max(0.05, min(start[3], end[3]))
            weight = length * distal_weight * confidence_weight
            segment_directions.append(direction)
            segment_weights.append(weight)
            segment_agreement += max(0.0, agreement) * weight
            agreement_weight += weight

        if segment_directions:
            weighted_direction = normalize(np.sum(np.stack(segment_directions) * np.array(segment_weights)[:, None], axis=0))
            segment_weight = float(np.clip(env_float("POINTING_SEGMENT_DIRECTION_WEIGHT", 0.0), 0.0, 0.5))
            direction = normalize((1.0 - segment_weight) * overall_direction + segment_weight * weighted_direction)
        else:
            direction = overall_direction

        if float(np.linalg.norm(direction)) == 0.0:
            return None

        if path_length <= 0.0:
            path_length = overall_length
        straightness = float(np.clip(overall_length / max(path_length, 1e-6), 0.0, 1.0))
        coverage = float(np.clip(len(samples) / max(2, len(self.finger_landmarks)), 0.0, 1.0))
        agreement = segment_agreement / agreement_weight if agreement_weight > 0.0 else straightness
        quality = float(np.clip(coverage * (0.45 + 0.55 * straightness) * (0.35 + 0.65 * agreement), 0.0, 1.0))

        origin = points_3d[-1] + direction * max(0.0, self.ray_origin_advance_m)
        return PointingRay(
            origin.astype(np.float32),
            direction.astype(np.float32),
            finger_points[0][1],
            finger_points[-1][1],
            quality,
        )

    def finger_depth_patch_radius(self, landmark_id: int) -> int:
        if landmark_id == self.finger_landmarks[-1]:
            return max(2, self.hand_depth_patch_radius - 2)
        if len(self.finger_landmarks) >= 2 and landmark_id == self.finger_landmarks[-2]:
            return max(3, self.hand_depth_patch_radius - 1)
        return max(3, self.hand_depth_patch_radius)

    def ray_from_legacy_finger_points(
        self,
        finger_points: list[tuple[int, tuple[int, int], float]],
        depth_m: np.ndarray,
        intrinsics: rs.intrinsics,
    ) -> PointingRay | None:
        by_id = {landmark_id: pixel for landmark_id, pixel, _ in finger_points}
        start_px = by_id.get(self.pointing_start_landmark)
        end_px = by_id.get(self.pointing_end_landmark)
        if start_px is None or end_px is None:
            if len(finger_points) < 2:
                return None
            start_px = finger_points[0][1]
            end_px = finger_points[-1][1]
        ray = self.ray_from_pixels(start_px, end_px, depth_m, intrinsics)
        if ray is None:
            return None
        return PointingRay(ray.origin_3d, ray.direction_3d, ray.start_2d, ray.end_2d, 0.35)

    def ray_from_pixels(
        self,
        start_px: tuple[int, int],
        end_px: tuple[int, int],
        depth_m: np.ndarray,
        intrinsics: rs.intrinsics,
    ) -> PointingRay | None:
        start_3d = deproject_pixel_with_patch_depth(start_px, depth_m, intrinsics, patch_radius=6)
        end_3d = deproject_pixel_with_patch_depth(end_px, depth_m, intrinsics, patch_radius=4)
        if start_3d is None or end_3d is None:
            return None
        vector = end_3d - start_3d
        length = float(np.linalg.norm(vector))
        if length < self.min_hand_ray_length_m:
            return None
        return PointingRay(start_3d, vector / length, start_px, end_px)

    def smooth_ray(self, ray: PointingRay) -> PointingRay:
        alpha = float(np.clip(self.smoothing, 0.0, 1.0))
        if self.smoothed_direction_3d is not None:
            dot = float(np.clip(np.dot(ray.direction_3d, self.smoothed_direction_3d), -1.0, 1.0))
            if self.max_ray_angle_jump_deg > 0.0 and dot < math.cos(math.radians(self.max_ray_angle_jump_deg)):
                self.reset_smoothing()
        if self.smoothed_end_2d is not None and self.max_ray_jump_px > 0.0:
            jump_px = math.hypot(ray.end_2d[0] - self.smoothed_end_2d[0], ray.end_2d[1] - self.smoothed_end_2d[1])
            if jump_px > self.max_ray_jump_px:
                self.reset_smoothing()

        if self.smoothed_origin_3d is None or self.smoothed_direction_3d is None or alpha <= 0.0:
            self.smoothed_origin_3d = ray.origin_3d
            self.smoothed_direction_3d = ray.direction_3d
            self.smoothed_end_2d = ray.end_2d
            self.last_ray_quality = ray.quality
            return ray

        direction = ray.direction_3d
        direction_dot = float(np.clip(np.dot(direction, self.smoothed_direction_3d), -1.0, 1.0))
        direction_change_deg = math.degrees(math.acos(direction_dot))
        angle_deadband_deg = env_float("POINTING_ANGLE_DEADBAND_DEG", 2.0)
        moving_alpha = float(np.clip(env_float("POINTING_MOVING_SMOOTHING", 0.65), alpha, 1.0))
        direction_alpha = alpha
        if direction_change_deg <= angle_deadband_deg:
            direction = self.smoothed_direction_3d
        else:
            # Follow an intentional finger movement quickly; the low alpha remains
            # active only while the hand is nearly stationary.
            direction_alpha = moving_alpha

        raw_end = np.asarray(ray.end_2d, dtype=np.float32)
        previous_end = np.asarray(self.smoothed_end_2d, dtype=np.float32)
        endpoint_movement_px = float(np.linalg.norm(raw_end - previous_end))
        if endpoint_movement_px <= env_float("POINTING_ENDPOINT_DEADBAND_PX", 8.0):
            smoothed_end = previous_end
            endpoint_alpha = alpha
        else:
            endpoint_alpha = moving_alpha
            smoothed_end = endpoint_alpha * raw_end + (1.0 - endpoint_alpha) * previous_end
        end_2d = (int(round(float(smoothed_end[0]))), int(round(float(smoothed_end[1]))))

        origin_alpha = max(alpha, direction_alpha)
        origin = origin_alpha * ray.origin_3d + (1.0 - origin_alpha) * self.smoothed_origin_3d
        direction = normalize(direction_alpha * direction + (1.0 - direction_alpha) * self.smoothed_direction_3d)
        self.smoothed_origin_3d = origin
        self.smoothed_direction_3d = direction
        self.smoothed_end_2d = end_2d
        self.last_ray_quality = ray.quality
        return PointingRay(origin, direction, ray.start_2d, end_2d, ray.quality)

    def reset_smoothing(self) -> None:
        self.smoothed_origin_3d = None
        self.smoothed_direction_3d = None
        self.smoothed_end_2d = None
        self.last_ray_quality = 0.0


class PointHitEstimator:
    def __init__(self) -> None:
        self.last_pixel: np.ndarray | None = None
        self.last_source = ""
        self.stable_prompt: PointPrompt | None = None
        self.pending_prompt: PointPrompt | None = None
        self.pending_prompt_frames = 0
        self.pending_switch_frames = 0

    def estimate(self, ray: PointingRay, depth_m: np.ndarray, intrinsics: rs.intrinsics) -> PointPrompt | None:
        prompt = self.ray_march_hit(ray, depth_m, intrinsics)
        if prompt is None and env_bool("POINT_CLOUD_HIT_FALLBACK", False):
            prompt = self.surface_hit(ray, depth_m, intrinsics)
        if prompt is None and env_bool("POINT_ALLOW_FALLBACK", False):
            prompt = self.fallback_prompt(ray, intrinsics, depth_m.shape[1], depth_m.shape[0])
        if prompt is None and env_bool("POINT_SHOW_PROJECTED_WHEN_NO_DEPTH", True):
            prompt = self.projected_prompt(ray, intrinsics, depth_m.shape[1], depth_m.shape[0])
        if prompt is None:
            self.reset_smoothing()
            return None
        prompt = self.stabilize_prompt(prompt)
        return self.smooth_prompt(prompt, depth_m.shape[1], depth_m.shape[0])

    def stabilize_prompt(self, prompt: PointPrompt) -> PointPrompt:
        """Debounce switches between two competing ray/depth intersections."""
        if self.stable_prompt is None:
            self.stable_prompt = prompt
            self.pending_prompt = None
            self.pending_prompt_frames = 0
            self.pending_switch_frames = 0
            return prompt

        # A newly detected foreground surface should win immediately. Moving to a
        # farther surface still requires the configured stability confirmation.
        closer_switch_margin_m = max(0.0, env_float("POINT_CLOSER_SWITCH_MARGIN_M", 0.05))
        if prompt.source == "depth_hit" and (
            self.stable_prompt.source != "depth_hit"
            or prompt.forward_m + closer_switch_margin_m < self.stable_prompt.forward_m
        ):
            self.stable_prompt = prompt
            self.pending_prompt = None
            self.pending_prompt_frames = 0
            self.pending_switch_frames = 0
            return prompt

        stable_radius_px = max(0.0, env_float("POINT_HIT_STABLE_RADIUS_PX", 18.0))
        stable_depth_m = max(0.0, env_float("POINT_HIT_STABLE_DEPTH_M", 0.12))
        if self.prompts_match(prompt, self.stable_prompt, stable_radius_px, stable_depth_m):
            self.stable_prompt = prompt
            self.pending_prompt = None
            self.pending_prompt_frames = 0
            self.pending_switch_frames = 0
            return prompt

        pending_radius_px = max(stable_radius_px, env_float("POINT_HIT_PENDING_RADIUS_PX", 28.0))
        pending_depth_m = max(stable_depth_m, env_float("POINT_HIT_PENDING_DEPTH_M", 0.18))
        if self.pending_prompt is not None and self.prompts_match(
            prompt,
            self.pending_prompt,
            pending_radius_px,
            pending_depth_m,
        ):
            self.pending_prompt = prompt
            self.pending_prompt_frames += 1
        else:
            self.pending_prompt = prompt
            self.pending_prompt_frames = 1

        self.pending_switch_frames += 1
        confirm_frames = max(1, env_int("POINT_HIT_SWITCH_CONFIRM_FRAMES", 3))
        max_hold_frames = max(confirm_frames, env_int("POINT_HIT_MAX_HOLD_FRAMES", 5))
        if self.pending_prompt_frames >= confirm_frames or self.pending_switch_frames >= max_hold_frames:
            self.stable_prompt = prompt
            self.pending_prompt = None
            self.pending_prompt_frames = 0
            self.pending_switch_frames = 0
        return self.stable_prompt

    @staticmethod
    def prompts_match(first: PointPrompt, second: PointPrompt, radius_px: float, depth_m: float) -> bool:
        pixel_distance = math.hypot(first.pixel[0] - second.pixel[0], first.pixel[1] - second.pixel[1])
        if pixel_distance > radius_px:
            return False
        if np.isfinite(first.forward_m) and np.isfinite(second.forward_m):
            if abs(first.forward_m - second.forward_m) > depth_m:
                return False
        return True

    def ray_march_hit(self, ray: PointingRay, depth_m: np.ndarray, intrinsics: rs.intrinsics) -> PointPrompt | None:
        min_forward = env_float("POINT_MIN_FORWARD_M", 0.20)
        max_forward = env_float("POINT_MAX_FORWARD_M", 4.0)
        samples = max(8, env_int("POINT_RAY_MARCH_SAMPLES", 96))
        patch_radius = max(0, env_int("POINT_RAY_MARCH_PATCH_RADIUS", 2))
        candidates: list[tuple[float, float, float, float, tuple[int, int], np.ndarray]] = []
        width = depth_m.shape[1]
        height = depth_m.shape[0]

        near_field_m = float(np.clip(env_float("POINT_NEAR_FIELD_M", 0.75), min_forward, max_forward))
        near_fraction = float(np.clip(env_float("POINT_NEAR_SAMPLE_FRACTION", 0.65), 0.25, 0.90))
        if min_forward < near_field_m < max_forward:
            near_count = int(np.clip(round(samples * near_fraction), 4, samples - 4))
            far_count = samples - near_count
            distances_m = np.concatenate(
                [
                    np.linspace(min_forward, near_field_m, near_count, endpoint=False),
                    np.linspace(near_field_m, max_forward, far_count),
                ]
            )
        else:
            distances_m = np.linspace(min_forward, max_forward, samples)

        last_pixel: tuple[int, int] | None = None
        for distance_m in distances_m:
            expected = ray.origin_3d + ray.direction_3d * float(distance_m)
            pixel = project_point_3d(expected, intrinsics)
            if pixel is None:
                continue
            x = int(np.clip(pixel[0], 0, width - 1))
            y = int(np.clip(pixel[1], 0, height - 1))
            if last_pixel == (x, y):
                continue
            last_pixel = (x, y)

            point_3d = deproject_pixel_with_patch_depth(
                (x, y),
                depth_m,
                intrinsics,
                patch_radius=patch_radius,
                depth_percentile=env_float("POINT_RAY_MARCH_DEPTH_PERCENTILE", 50.0),
                depth_cluster_m=env_float("POINT_RAY_MARCH_DEPTH_CLUSTER_M", 0.06),
            )
            if point_3d is None:
                continue

            support_radius = max(0, env_int("POINT_HIT_SUPPORT_RADIUS_PX", 3))
            min_support = max(1, env_int("POINT_HIT_MIN_DEPTH_SUPPORT", 4))
            if support_radius > 0 and min_support > 1:
                y0 = max(0, y - support_radius)
                y1 = min(height, y + support_radius + 1)
                x0 = max(0, x - support_radius)
                x1 = min(width, x + support_radius + 1)
                support_window = depth_m[y0:y1, x0:x1]
                support_tolerance_m = max(0.005, env_float("POINT_HIT_SUPPORT_DEPTH_TOLERANCE_M", 0.05))
                supported = valid_depth(support_window) & (np.abs(support_window - float(point_3d[2])) <= support_tolerance_m)
                if int(np.count_nonzero(supported)) < min_support:
                    continue

            vector = point_3d - ray.origin_3d
            forward = float(vector @ ray.direction_3d)
            if forward < min_forward or forward > max_forward:
                continue
            offset = float(np.linalg.norm(np.cross(vector, ray.direction_3d)))
            allowed_offset = self.allowed_ray_offset(forward)
            if offset > allowed_offset:
                continue

            normalized_offset = offset / max(allowed_offset, 1e-6)
            distance_from_finger = float(np.linalg.norm(vector))
            candidates.append((distance_from_finger, forward, normalized_offset, offset, (x, y), point_3d))

        if not candidates:
            return None

        # Distance from the fingertip is the primary decision. Alignment only
        # chooses between samples belonging to the same nearest surface.
        nearest_distance = min(item[0] for item in candidates)
        surface_band_m = max(0.01, env_float("POINT_FRONT_SURFACE_WINDOW_M", 0.06))
        nearest_surface = [item for item in candidates if item[0] <= nearest_distance + surface_band_m]
        alignment_tiebreak_m = max(0.0, env_float("POINT_ALIGNMENT_TIEBREAK_M", 0.01))
        _, forward, _, offset, pixel, point_3d = min(
            nearest_surface,
            # Alignment may break a near-distance tie, but it can never pull the
            # target several centimetres onto a background surface.
            key=lambda item: item[0] + alignment_tiebreak_m * item[2],
        )
        pixel = self.refine_prompt_pixel(pixel, point_3d, depth_m)
        return PointPrompt(
            pixel=pixel,
            point_3d=point_3d.astype(np.float32),
            forward_m=float(forward),
            ray_offset_m=float(offset),
            source="depth_hit",
        )

    def allowed_ray_offset(self, forward_m: float) -> float:
        max_offset = env_float("POINT_HIT_MAX_OFFSET_M", 0.10)
        min_offset = env_float("POINT_HIT_MIN_OFFSET_M", 0.025)
        cone_deg = env_float("POINT_HIT_CONE_DEG", 3.0)
        if cone_deg <= 0.0:
            return max_offset
        return float(min(max_offset, max(min_offset, forward_m * math.tan(math.radians(cone_deg)))))

    def surface_hit(self, ray: PointingRay, depth_m: np.ndarray, intrinsics: rs.intrinsics) -> PointPrompt | None:
        stride = max(1, env_int("POINT_HIT_STRIDE", 4))
        valid = valid_depth(depth_m)
        valid_sample = valid[::stride, ::stride]
        sample_ys, sample_xs = np.where(valid_sample)
        if len(sample_xs) == 0:
            return None
        xs = (sample_xs * stride).astype(np.float32)
        ys = (sample_ys * stride).astype(np.float32)
        z = depth_m[ys.astype(np.int32), xs.astype(np.int32)]
        points = deproject_arrays(xs, ys, z, intrinsics)
        vectors = points - ray.origin_3d
        forward = vectors @ ray.direction_3d
        min_forward = env_float("POINT_MIN_FORWARD_M", 0.20)
        max_forward = env_float("POINT_MAX_FORWARD_M", 4.0)
        forward_mask = (forward >= min_forward) & (forward <= max_forward)
        if not np.any(forward_mask):
            return None

        points = points[forward_mask]
        xs = xs[forward_mask]
        ys = ys[forward_mask]
        forward = forward[forward_mask]
        vectors = vectors[forward_mask]
        offsets = np.linalg.norm(np.cross(vectors, ray.direction_3d), axis=1)
        min_offset = env_float("POINT_HIT_MIN_OFFSET_M", 0.025)
        allowed_offsets = np.array([self.allowed_ray_offset(float(value)) for value in forward], dtype=np.float32)
        close_mask = offsets <= allowed_offsets
        if not np.any(close_mask):
            return None

        offset_weight = env_float("POINT_HIT_OFFSET_WEIGHT", 4.0)
        core_fraction = float(np.clip(env_float("POINT_HIT_CORE_FRACTION", 0.45), 0.05, 1.0))
        core_offsets = np.maximum(min_offset, allowed_offsets * core_fraction)
        core_mask = close_mask & (offsets <= core_offsets)
        if np.any(core_mask):
            candidate_indices = np.where(core_mask)[0]
            scores = forward[candidate_indices] + offset_weight * offsets[candidate_indices]
        else:
            candidate_indices = np.where(close_mask)[0]
            forward_weight = env_float("POINT_HIT_FORWARD_WEIGHT", 0.08)
            normalized_offsets = offsets[candidate_indices] / np.maximum(allowed_offsets[candidate_indices], 1e-6)
            scores = offset_weight * normalized_offsets + forward_weight * forward[candidate_indices]
        best = int(candidate_indices[int(np.argmin(scores))])
        pixel = (
            int(np.clip(round(float(xs[best])), 0, depth_m.shape[1] - 1)),
            int(np.clip(round(float(ys[best])), 0, depth_m.shape[0] - 1)),
        )
        pixel = self.refine_prompt_pixel(pixel, points[best], depth_m)
        return PointPrompt(
            pixel=pixel,
            point_3d=points[best].astype(np.float32),
            forward_m=float(forward[best]),
            ray_offset_m=float(offsets[best]),
            source="depth_hit",
        )

    def fallback_prompt(self, ray: PointingRay, intrinsics: rs.intrinsics, width: int, height: int) -> PointPrompt | None:
        prompt = self.projected_prompt(ray, intrinsics, width, height)
        if prompt is None:
            return None
        return PointPrompt(prompt.pixel, prompt.point_3d, prompt.forward_m, prompt.ray_offset_m, "fallback")

    def projected_prompt(self, ray: PointingRay, intrinsics: rs.intrinsics, width: int, height: int) -> PointPrompt | None:
        distance_m = env_float("POINT_FALLBACK_DISTANCE_M", 2.0)
        point = ray.origin_3d + ray.direction_3d * distance_m
        pixel = project_point_3d(point, intrinsics)
        if pixel is None:
            pixel = project_ray_endpoint_2d(ray, intrinsics, width, height, max_distance_m=distance_m)
        if pixel is None:
            return None
        return PointPrompt(
            pixel=(int(np.clip(pixel[0], 0, width - 1)), int(np.clip(pixel[1], 0, height - 1))),
            point_3d=point.astype(np.float32),
            forward_m=distance_m,
            ray_offset_m=float("inf"),
            source="ray_projected",
        )

    def smooth_prompt(self, prompt: PointPrompt, width: int, height: int) -> PointPrompt:
        alpha = float(np.clip(env_float("POINT_PIXEL_SMOOTHING", 0.45), 0.0, 1.0))
        current = np.array(prompt.pixel, dtype=np.float32)
        max_jump_px = env_float("POINT_PIXEL_MAX_JUMP_PX", 160.0)
        reset = False
        if self.last_pixel is not None and max_jump_px > 0.0:
            reset = float(np.linalg.norm(current - self.last_pixel)) > max_jump_px
        if prompt.source != self.last_source:
            reset = True
        if self.last_pixel is None or alpha <= 0.0 or reset:
            self.last_pixel = current
            self.last_source = prompt.source
            return prompt
        deadband_px = max(0.0, env_float("POINT_PIXEL_DEADBAND_PX", 10.0))
        if float(np.linalg.norm(current - self.last_pixel)) <= deadband_px:
            pixel = (
                int(np.clip(round(float(self.last_pixel[0])), 0, width - 1)),
                int(np.clip(round(float(self.last_pixel[1])), 0, height - 1)),
            )
            self.last_source = prompt.source
            return PointPrompt(pixel, prompt.point_3d, prompt.forward_m, prompt.ray_offset_m, prompt.source)
        smoothed = alpha * current + (1.0 - alpha) * self.last_pixel
        self.last_pixel = smoothed
        self.last_source = prompt.source
        pixel = (
            int(np.clip(round(float(smoothed[0])), 0, width - 1)),
            int(np.clip(round(float(smoothed[1])), 0, height - 1)),
        )
        return PointPrompt(pixel, prompt.point_3d, prompt.forward_m, prompt.ray_offset_m, prompt.source)

    def refine_prompt_pixel(self, pixel: tuple[int, int], point_3d: np.ndarray, depth_m: np.ndarray) -> tuple[int, int]:
        radius = env_int("POINT_PROMPT_REFINE_RADIUS_PX", 14)
        if radius <= 0 or point_3d is None:
            return pixel
        target_z = float(point_3d[2])
        if not np.isfinite(target_z):
            return pixel

        x, y = pixel
        y0 = max(0, y - radius)
        y1 = min(depth_m.shape[0], y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(depth_m.shape[1], x + radius + 1)
        window = depth_m[y0:y1, x0:x1]
        if window.size == 0:
            return pixel

        tolerance = env_float("POINT_PROMPT_DEPTH_TOLERANCE_M", 0.07)
        close_depth = valid_depth(window) & (np.abs(window - target_z) <= tolerance)
        if not np.any(close_depth):
            return pixel

        labels_count, labels, _, _ = cv2.connectedComponentsWithStats(close_depth.astype(np.uint8), 8)
        if labels_count <= 1:
            return pixel

        local_x = int(np.clip(x - x0, 0, labels.shape[1] - 1))
        local_y = int(np.clip(y - y0, 0, labels.shape[0] - 1))
        label = int(labels[local_y, local_x])
        if label <= 0:
            candidates = np.argwhere(close_depth)
            distances = (candidates[:, 1] - local_x) ** 2 + (candidates[:, 0] - local_y) ** 2
            nearest_y, nearest_x = candidates[int(np.argmin(distances))]
            label = int(labels[int(nearest_y), int(nearest_x)])
        if label <= 0:
            return pixel

        component = labels == label
        if int(np.count_nonzero(component)) < 4:
            return pixel
        if component[0, :].any() or component[-1, :].any() or component[:, 0].any() or component[:, -1].any():
            return pixel
        distances = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 3)
        refined_y, refined_x = np.unravel_index(int(np.argmax(distances)), distances.shape)
        return (
            int(np.clip(x0 + refined_x, 0, depth_m.shape[1] - 1)),
            int(np.clip(y0 + refined_y, 0, depth_m.shape[0] - 1)),
        )

    def reset_smoothing(self) -> None:
        self.last_pixel = None
        self.last_source = ""
        self.stable_prompt = None
        self.pending_prompt = None
        self.pending_prompt_frames = 0
        self.pending_switch_frames = 0


class SAMPointSegmenter:
    def __init__(self) -> None:
        self.requested_backend = os.getenv("SAM_BACKEND", "auto").strip().lower()
        self.device = os.getenv("SAM_DEVICE", "cuda")
        self.predictor: Any | None = None
        self.backend_name = ""
        self.backend_kind = ""
        self.load_error = ""

    def load(self) -> None:
        if self.predictor is not None:
            return
        backends = [self.requested_backend]
        if self.requested_backend == "auto":
            backends = env_list("SAM_BACKENDS", "sam3,sam2,sam1,transformers")

        errors: list[str] = []
        for backend in backends:
            try:
                if backend == "sam3":
                    self.load_sam3()
                elif backend == "sam2":
                    self.load_sam2()
                elif backend == "sam1":
                    self.load_sam1()
                elif backend == "transformers":
                    self.load_transformers()
                else:
                    raise RuntimeError(f"unknown SAM backend {backend!r}")
                self.load_error = ""
                return
            except Exception as exc:
                errors.append(f"{backend}: {exc}")
                self.predictor = None
                self.backend_name = ""
                self.backend_kind = ""

        self.load_error = "; ".join(errors)
        raise RuntimeError(f"No SAM backend loaded. {self.load_error}")

    def load_sam3(self) -> None:
        factory_spec = os.getenv("SAM3_FACTORY", "").strip()
        if factory_spec:
            module_name, function_name = factory_spec.split(":", 1)
            factory = getattr(importlib.import_module(module_name), function_name)
            checkpoint = os.getenv("SAM_CHECKPOINT", "").strip() or None
            model_id = os.getenv("SAM_MODEL_ID", "").strip() or None
            try:
                predictor = factory(checkpoint=checkpoint, model_id=model_id, device=self.device)
            except TypeError:
                predictor = factory()
            self.predictor = predictor
            self.backend_name = "sam3"
            self.backend_kind = "sam_like"
            return

        import torch
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        model_id = os.getenv("SAM3_MODEL_ID", "").strip() or os.getenv("SAM_MODEL_ID", "").strip() or "facebook/sam3"
        dtype = resolve_torch_dtype(torch)
        model = Sam3TrackerModel.from_pretrained(model_id, torch_dtype=dtype)
        if self.device:
            model.to(self.device)
        model.eval()
        processor = Sam3TrackerProcessor.from_pretrained(model_id)
        self.predictor = (model, processor)
        self.backend_name = "sam3-tracker"
        self.backend_kind = "sam3_tracker"

    def load_sam2(self) -> None:
        checkpoint = os.getenv("SAM_CHECKPOINT", "").strip() or os.getenv("SAM2_CHECKPOINT", "").strip()
        config = os.getenv("SAM2_CONFIG", "").strip()
        if not checkpoint or not config:
            raise RuntimeError("SAM2_CONFIG and SAM_CHECKPOINT are required")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(config, checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(model)
        self.backend_name = "sam2"
        self.backend_kind = "sam_like"

    def load_sam1(self) -> None:
        checkpoint = os.getenv("SAM_CHECKPOINT", "").strip()
        if not checkpoint:
            raise RuntimeError("SAM_CHECKPOINT is required")
        if not os.path.exists(checkpoint):
            raise RuntimeError(f"SAM_CHECKPOINT not found: {checkpoint}")
        from segment_anything import SamPredictor, sam_model_registry

        model_type = os.getenv("SAM_MODEL_TYPE", "vit_h").strip()
        if model_type not in sam_model_registry:
            raise RuntimeError(f"unsupported SAM_MODEL_TYPE={model_type!r}")
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        self.backend_name = "sam1"
        self.backend_kind = "sam_like"

    def load_transformers(self) -> None:
        model_id = os.getenv("SAM_MODEL_ID", "").strip()
        if not model_id:
            raise RuntimeError("SAM_MODEL_ID is required for transformers backend")
        import torch
        from transformers import SamModel, SamProcessor

        dtype = resolve_torch_dtype(torch)
        model = SamModel.from_pretrained(model_id, torch_dtype=dtype)
        if self.device:
            model.to(self.device)
        processor = SamProcessor.from_pretrained(model_id)
        self.predictor = (model, processor)
        self.backend_name = "transformers"
        self.backend_kind = "transformers"

    def predict(self, image_bgr: np.ndarray, prompt: PointPrompt) -> SegmentationResult | None:
        self.load()
        start = time.perf_counter()
        if self.backend_kind == "sam3_tracker":
            result = self.predict_sam3_tracker(image_bgr, prompt)
        elif self.backend_kind == "transformers":
            result = self.predict_transformers(image_bgr, prompt)
        else:
            result = self.predict_sam_like(image_bgr, prompt)
        if result is None:
            return None
        result.elapsed_ms = (time.perf_counter() - start) * 1000.0
        return result

    def predict_sam_like(self, image_bgr: np.ndarray, prompt: PointPrompt) -> SegmentationResult | None:
        if self.predictor is None:
            return None
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if hasattr(self.predictor, "set_image"):
            self.predictor.set_image(image_rgb)
        point_coords = np.array([prompt.pixel], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        multimask = env_bool("SAM_MULTIMASK", True)
        prediction = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask,
        )
        if isinstance(prediction, dict):
            masks = first_present(prediction, "masks", "pred_masks")
            scores = first_present(prediction, "scores", "iou_predictions", "iou_scores")
        else:
            masks = prediction[0]
            scores = prediction[1] if len(prediction) > 1 else None
        return self.pick_mask(masks, scores, prompt)

    def predict_transformers(self, image_bgr: np.ndarray, prompt: PointPrompt) -> SegmentationResult | None:
        if self.predictor is None:
            return None
        import torch
        from PIL import Image

        model, processor = self.predictor
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        inputs = processor(image, input_points=[[list(prompt.pixel)]], return_tensors="pt")
        inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        dtype = resolve_torch_dtype(torch)
        autocast_enabled = env_bool("SAM_AUTOCAST", True) and str(model.device).startswith("cuda") and dtype != torch.float32
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            outputs = model(**inputs)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.detach().float().cpu(),
            inputs["original_sizes"].detach().cpu(),
            inputs["reshaped_input_sizes"].detach().cpu(),
        )[0]
        scores = outputs.iou_scores.detach().float().cpu().numpy().reshape(-1)
        return self.pick_mask(masks.numpy(), scores, prompt)

    def predict_sam3_tracker(self, image_bgr: np.ndarray, prompt: PointPrompt) -> SegmentationResult | None:
        if self.predictor is None:
            return None
        import torch
        from PIL import Image

        model, processor = self.predictor
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        x, y = prompt.pixel
        input_points = [[[[float(x), float(y)]]]]
        input_labels = [[[1]]]
        inputs = processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(model.device)
        dtype = resolve_torch_dtype(torch)
        autocast_enabled = env_bool("SAM_AUTOCAST", True) and str(model.device).startswith("cuda") and dtype != torch.float32
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            outputs = model(**inputs, multimask_output=env_bool("SAM_MULTIMASK", True))
        masks = processor.post_process_masks(outputs.pred_masks.detach().float().cpu(), inputs["original_sizes"].detach().cpu())[0]
        scores = first_present_output(outputs, "iou_scores", "predicted_iou", "scores")
        if hasattr(scores, "detach"):
            scores = scores.detach().float().cpu()
        return self.pick_mask(masks, scores, prompt)

    def pick_mask(self, masks: Any, scores: Any, prompt: PointPrompt | None = None) -> SegmentationResult | None:
        masks_np = to_numpy(masks)
        if masks_np is None:
            return None
        while masks_np.ndim > 3:
            masks_np = np.squeeze(masks_np, axis=0)
        if masks_np.ndim == 2:
            masks_np = masks_np[None]
        if masks_np.ndim != 3 or masks_np.shape[0] == 0:
            return None

        scores_np = to_numpy(scores)
        if scores_np is None:
            model_scores = np.zeros(masks_np.shape[0], dtype=np.float32)
        else:
            model_scores = np.asarray(scores_np, dtype=np.float32).reshape(-1)
            if len(model_scores) < masks_np.shape[0]:
                model_scores = np.pad(model_scores, (0, masks_np.shape[0] - len(model_scores)))
            model_scores = model_scores[: masks_np.shape[0]]

        mask_stack = masks_np > 0.0
        adjusted_scores = model_scores.astype(np.float32, copy=True)
        area_ratios = mask_stack.reshape(mask_stack.shape[0], -1).mean(axis=1)
        contains_prompt = np.zeros(mask_stack.shape[0], dtype=bool)
        if prompt is not None:
            x, y = prompt.pixel
            if 0 <= y < mask_stack.shape[1] and 0 <= x < mask_stack.shape[2]:
                contains_bonus = env_float("SAM_PROMPT_CONTAINS_BONUS", 0.25)
                miss_penalty = env_float("SAM_PROMPT_MISS_PENALTY", 0.75)
                contains_prompt = mask_stack[:, y, x]
                adjusted_scores += np.where(contains_prompt, contains_bonus, -miss_penalty)

        min_area = env_float("SAM_MASK_MIN_AREA_RATIO", 0.0004)
        max_area = env_float("SAM_MASK_MAX_AREA_RATIO", 0.65)
        adjusted_scores -= np.where(area_ratios < min_area, 0.35, 0.0)
        adjusted_scores -= np.where(area_ratios > max_area, 0.35, 0.0)

        detail_area = env_float("SAM_MASK_DETAIL_AREA_RATIO", 0.012)
        detail_penalty = env_float("SAM_MASK_DETAIL_PENALTY", 0.65)
        if detail_area > 0.0 and detail_penalty > 0.0:
            detail_scale = np.clip((detail_area - area_ratios) / detail_area, 0.0, 1.0)
            adjusted_scores -= detail_penalty * detail_scale

        best = int(np.argmax(adjusted_scores))
        if prompt is not None and env_bool("SAM_PREFER_ENCLOSING_MASK", True):
            enclosing_min_area = env_float("SAM_ENCLOSING_MIN_AREA_RATIO", detail_area)
            enclosing_max_area = env_float("SAM_ENCLOSING_MAX_AREA_RATIO", min(max_area, 0.55))
            enclosing_candidates = contains_prompt & (area_ratios >= enclosing_min_area) & (area_ratios <= enclosing_max_area)
            if np.any(enclosing_candidates):
                margin = env_float("SAM_ENCLOSING_SCORE_MARGIN", 0.45)
                area_weight = env_float("SAM_ENCLOSING_AREA_WEIGHT", 0.50)
                best_score = float(np.max(adjusted_scores))
                candidate_indices = np.where(enclosing_candidates & (adjusted_scores >= best_score - margin))[0]
                if len(candidate_indices) > 0:
                    scaled_area = np.sqrt(np.clip(area_ratios[candidate_indices] / max(enclosing_min_area, 1e-6), 0.0, 4.0))
                    enclosing_scores = adjusted_scores[candidate_indices] + area_weight * scaled_area
                    best = int(candidate_indices[int(np.argmax(enclosing_scores))])

        mask = mask_stack[best]
        return SegmentationResult(mask=mask, score=float(adjusted_scores[best]), backend=self.backend_name, elapsed_ms=0.0)



class AsyncSegmenter:
    """Runs expensive SAM inference without blocking the live pointing loop."""

    def __init__(self, segmenter: SAMPointSegmenter) -> None:
        self.segmenter = segmenter
        self.condition = threading.Condition()
        self.pending: tuple[np.ndarray, PointPrompt, int] | None = None
        self.latest_result: SegmentationResult | None = None
        self.latest_error = ""
        self.generation = 0
        self.busy = False
        self.running = True
        self.thread = threading.Thread(target=self._run, name="sam-inference", daemon=True)
        self.thread.start()

    def submit(self, image_bgr: np.ndarray, prompt: PointPrompt) -> None:
        with self.condition:
            # Keep only the newest request if SAM is still processing an older frame.
            self.pending = (image_bgr.copy(), prompt, self.generation)
            self.condition.notify()

    def snapshot(self) -> tuple[SegmentationResult | None, str, bool]:
        with self.condition:
            return self.latest_result, self.latest_error, self.busy or self.pending is not None

    def clear(self) -> None:
        with self.condition:
            self.generation += 1
            self.pending = None
            self.latest_result = None
            self.latest_error = ""

    def stop(self) -> None:
        with self.condition:
            self.running = False
            self.pending = None
            self.condition.notify()
        self.thread.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            with self.condition:
                while self.running and self.pending is None:
                    self.condition.wait()
                if not self.running:
                    return
                image_bgr, prompt, generation = self.pending
                self.pending = None
                self.busy = True

            try:
                result = self.segmenter.predict(image_bgr, prompt)
                error = ""
            except Exception as exc:
                result = None
                error = str(exc)

            with self.condition:
                self.busy = False
                if generation == self.generation:
                    self.latest_result = result
                    self.latest_error = error
                self.condition.notify()

def to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
        if str(getattr(value, "dtype", "")) == "torch.bfloat16":
            value = value.float()
        value = value.cpu().numpy()
    return np.asarray(value)


def resolve_torch_dtype(torch_module: Any) -> Any:
    raw = os.getenv("SAM_DTYPE", "float32").strip().lower()
    if raw in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if raw in {"fp16", "float16", "half"}:
        return torch_module.float16
    return torch_module.float32


def first_present(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def first_present_output(output: Any, *names: str) -> Any:
    for name in names:
        value = getattr(output, name, None)
        if value is not None:
            return value
    return None


def landmark_to_pixel(landmark: Any, width: int, height: int) -> tuple[int, int]:
    x = int(np.clip(landmark.x * width, 0, width - 1))
    y = int(np.clip(landmark.y * height, 0, height - 1))
    return x, y


def keypoint_to_pixel(keypoint: np.ndarray, width: int, height: int) -> tuple[int, int]:
    x = int(np.clip(round(float(keypoint[0])), 0, width - 1))
    y = int(np.clip(round(float(keypoint[1])), 0, height - 1))
    return x, y


def mediapipe_landmark_confidence(landmark: Any) -> float:
    values = []
    for name in ("visibility", "presence"):
        value = getattr(landmark, name, None)
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value) and value > 0.0:
            values.append(value)
    return min(values) if values else 1.0


def valid_depth(depth_m: np.ndarray) -> np.ndarray:
    depth_min = env_float("DEPTH_MIN_M", 0.15)
    depth_max = env_float("DEPTH_MAX_M", 4.0)
    return np.isfinite(depth_m) & (depth_m >= depth_min) & (depth_m <= depth_max)


def deproject_arrays(xs: np.ndarray, ys: np.ndarray, z: np.ndarray, intrinsics: rs.intrinsics) -> np.ndarray:
    x3 = (xs - intrinsics.ppx) / intrinsics.fx * z
    y3 = (ys - intrinsics.ppy) / intrinsics.fy * z
    return np.stack([x3, y3, z], axis=1).astype(np.float32)


def deproject_pixel_with_patch_depth(
    pixel: tuple[int, int],
    depth_m: np.ndarray,
    intrinsics: rs.intrinsics,
    patch_radius: int = 4,
    depth_percentile: float = 50.0,
    depth_cluster_m: float = 0.0,
) -> np.ndarray | None:
    z = patch_depth_value(pixel, depth_m, patch_radius, depth_percentile, depth_cluster_m)
    if z is None:
        return None
    x, y = pixel
    try:
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [float(x), float(y)], float(z))
        return np.asarray(point, dtype=np.float32)
    except Exception:
        return np.array(
            [
                (x - intrinsics.ppx) / intrinsics.fx * z,
                (y - intrinsics.ppy) / intrinsics.fy * z,
                z,
            ],
            dtype=np.float32,
        )


def patch_depth_value(
    pixel: tuple[int, int],
    depth_m: np.ndarray,
    patch_radius: int,
    depth_percentile: float = 50.0,
    depth_cluster_m: float = 0.0,
) -> float | None:
    x, y = pixel
    y0 = max(0, y - patch_radius)
    y1 = min(depth_m.shape[0], y + patch_radius + 1)
    x0 = max(0, x - patch_radius)
    x1 = min(depth_m.shape[1], x + patch_radius + 1)
    patch = depth_m[y0:y1, x0:x1]
    values = patch[valid_depth(patch)]
    if len(values) == 0:
        return None
    percentile = float(np.clip(depth_percentile, 0.0, 100.0))
    anchor = float(np.percentile(values, percentile))
    if depth_cluster_m > 0.0:
        clustered = values[np.abs(values - anchor) <= depth_cluster_m]
        if len(clustered) >= max(3, int(0.08 * len(values))):
            values = clustered
    return float(np.median(values))


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def project_point_3d(point: np.ndarray, intrinsics: rs.intrinsics) -> tuple[int, int] | None:
    x, y, z = [float(value) for value in point]
    if not np.isfinite(z) or z <= 0.03:
        return None
    try:
        pixel_x, pixel_y = rs.rs2_project_point_to_pixel(intrinsics, [x, y, z])
        return int(round(pixel_x)), int(round(pixel_y))
    except Exception:
        pixel_x = int(round((x / z) * intrinsics.fx + intrinsics.ppx))
        pixel_y = int(round((y / z) * intrinsics.fy + intrinsics.ppy))
        return pixel_x, pixel_y


def project_ray_endpoint_2d(
    ray: PointingRay,
    intrinsics: rs.intrinsics,
    width: int,
    height: int,
    max_distance_m: float = 4.0,
) -> tuple[int, int] | None:
    far_pixel: tuple[int, int] | None = None
    for distance_m in np.linspace(0.15, max_distance_m, 32):
        point = ray.origin_3d + ray.direction_3d * float(distance_m)
        projected = project_point_3d(point, intrinsics)
        if projected is not None:
            far_pixel = projected
    if far_pixel is None:
        return ray.end_2d
    clipped = cv2.clipLine((0, 0, width, height), ray.start_2d, far_pixel)
    if not clipped[0]:
        return int(np.clip(far_pixel[0], 0, width - 1)), int(np.clip(far_pixel[1], 0, height - 1))
    return clipped[2]


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (20, 20, 20),
    background: tuple[int, int, int] = (245, 245, 245),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    x = int(np.clip(x, 4, max(4, image.shape[1] - text_w - 8)))
    y = int(np.clip(y, text_h + 8, max(text_h + 8, image.shape[0] - baseline - 6)))
    cv2.rectangle(image, (x - 4, y - text_h - 6), (x + text_w + 4, y + baseline + 4), background, -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_segmentation_contours(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image, contours, -1, color, 2)


def draw_overlay(
    image: np.ndarray,
    ray: PointingRay | None,
    prompt: PointPrompt | None,
    result: SegmentationResult | None,
    status: str,
) -> np.ndarray:
    output = image.copy()
    if result is not None and result.mask.shape[:2] == output.shape[:2]:
        tint = np.zeros_like(output)
        tint[result.mask] = (0, 220, 0)
        output = cv2.addWeighted(output, 1.0, tint, 0.42, 0.0)
        draw_segmentation_contours(output, result.mask, (0, 255, 0))

    if ray is not None:
        line_end = prompt.pixel if prompt is not None else ray.end_2d
        cv2.line(output, ray.start_2d, line_end, (0, 255, 255), 2)
        cv2.circle(output, ray.start_2d, 5, (255, 255, 0), -1)
        cv2.circle(output, ray.end_2d, 5, (0, 180, 255), -1)

    if prompt is not None:
        x, y = prompt.pixel
        cv2.drawMarker(output, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 24, 3)
        cv2.circle(output, (x, y), 8, (0, 255, 255), 2)

    draw_label(output, status, (12, 28), (20, 20, 20), background=(245, 245, 245))
    if result is not None:
        draw_label(
            output,
            f"{result.backend} mask score={result.score:.3f}",
            (12, 56),
            (20, 20, 20),
            background=(200, 255, 205),
        )
    elif prompt is not None:
        offset = "inf" if not np.isfinite(prompt.ray_offset_m) else f"{prompt.ray_offset_m:.3f}m"
        draw_label(
            output,
            f"point {prompt.pixel} {prompt.source} offset={offset}",
            (12, 56),
            (20, 20, 20),
            background=(255, 238, 170),
        )
    return output


def blank_frame(message: str, width: int = 960, height: int = 540) -> np.ndarray:
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    draw_label(image, message, (24, 48), (20, 20, 20), background=(245, 245, 245))
    return image


def worker(state: SharedState) -> None:
    camera = RealSenseCamera()
    pointing: PointingEstimator | None = None
    hit_estimator = PointHitEstimator()
    segmenter = SAMPointSegmenter()
    async_segmenter: AsyncSegmenter | None = None
    last_result: SegmentationResult | None = None
    last_prompt_pixel: tuple[int, int] | None = None
    last_prompt_source = ""
    last_sam_submit_time = 0.0
    frame_count = 0

    try:
        state.set_frame(blank_frame("Starting RealSense..."), "starting RealSense")
        camera.start()
        if camera.intrinsics is None:
            raise RuntimeError("RealSense color intrinsics unavailable")

        state.set_frame(blank_frame("Loading hand tracker..."), "loading hand tracker")
        pointing = PointingEstimator()

        state.set_frame(blank_frame("Loading SAM..."), "loading SAM")
        segmenter.load()
        async_segmenter = AsyncSegmenter(segmenter)

        while state.running:
            frame_start = time.perf_counter()

            start = time.perf_counter()
            color, depth = camera.read()
            read_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            ray = pointing.estimate(color, depth, camera.intrinsics)
            hand_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            if ray is not None:
                prompt = hit_estimator.estimate(ray, depth, camera.intrinsics)
            else:
                hit_estimator.reset_smoothing()
                prompt = None
            hit_ms = (time.perf_counter() - start) * 1000.0

            async_result, async_error, sam_busy = async_segmenter.snapshot()
            if async_error:
                segmenter.load_error = async_error
                last_result = None
            elif async_result is not None:
                segmenter.load_error = ""
                last_result = async_result

            if prompt is None:
                if last_prompt_pixel is not None or last_result is not None or sam_busy:
                    async_segmenter.clear()
                last_result = None
                last_prompt_pixel = None
                last_prompt_source = ""
            else:
                prompt_jump_px = 0.0
                if last_prompt_pixel is not None:
                    prompt_jump_px = math.hypot(prompt.pixel[0] - last_prompt_pixel[0], prompt.pixel[1] - last_prompt_pixel[1])
                if prompt.source != last_prompt_source or prompt_jump_px > env_float("SAM_RESET_PROMPT_JUMP_PX", 90.0):
                    last_result = None
                    async_segmenter.clear()
                last_prompt_pixel = prompt.pixel
                last_prompt_source = prompt.source

            now = time.monotonic()
            sam_allowed = prompt is not None and (prompt.source != "ray_projected" or env_bool("SAM_ON_PROJECTED_PROMPT", False))
            sam_interval_s = max(0.1, env_float("SAM_INTERVAL_S", 0.75))
            interval_due = last_sam_submit_time <= 0.0 or now - last_sam_submit_time >= sam_interval_s

            if sam_allowed and prompt is not None and interval_due:
                async_segmenter.submit(color, prompt)
                last_sam_submit_time = now
                sam_busy = True
            elif not sam_allowed and (last_result is not None or sam_busy):
                async_segmenter.clear()
                last_result = None
                sam_busy = False

            sam_ms = last_result.elapsed_ms if last_result is not None else 0.0
            total_ms = (time.perf_counter() - frame_start) * 1000.0
            fps = 1000.0 / total_ms if total_ms > 0 else 0.0
            if ray is None:
                status = "ray=no point=no mask=no"
            elif prompt is None:
                status = "ray=yes point=no mask=no"
            else:
                mask_status = "updating" if sam_busy else ("yes" if last_result else "no")
                status = f"ray=yes point={prompt.pixel} mask={mask_status}"
            if segmenter.load_error:
                status = f"SAM error: {segmenter.load_error[:120]}"

            overlay = draw_overlay(color, ray, prompt, last_result, status)
            metrics: dict[str, float | int | str] = {
                "frame": frame_count,
                "fps": round(fps, 2),
                "total_ms": round(total_ms, 1),
                "read_ms": round(read_ms, 1),
                "hand_ms": round(hand_ms, 1),
                "hit_ms": round(hit_ms, 1),
                "sam_ms": round(sam_ms, 1),
                "sam_busy": int(sam_busy),
                "resolution": f"{color.shape[1]}x{color.shape[0]}",
                "sam_backend": segmenter.backend_name or segmenter.requested_backend,
                "hand_backend": pointing.backend if pointing is not None else "none",
                "hand_conf": round(pointing.last_hand_confidence, 3) if pointing is not None else 0.0,
                "hand_candidates": pointing.last_hand_candidates if pointing is not None else 0,
                "ray_quality": round(pointing.last_ray_quality, 3) if pointing is not None else 0.0,
                "point_source": prompt.source if prompt is not None else "none",
                "point_forward_m": round(prompt.forward_m, 3) if prompt is not None else 0.0,
                "point_ray_offset_m": round(prompt.ray_offset_m, 3) if prompt is not None and np.isfinite(prompt.ray_offset_m) else -1.0,
            }
            state.set_frame(overlay, status, metrics)
            frame_count += 1
    except Exception as exc:
        traceback.print_exc()
        state.set_frame(blank_frame(f"Error: {exc}"), f"error: {exc}")
        while state.running:
            time.sleep(0.5)
    finally:
        if async_segmenter is not None:
            async_segmenter.stop()
        try:
            camera.stop()
        except Exception:
            pass


def default_bag_output_path(bag_path: Path) -> Path:
    return bag_path.with_name(f"{bag_path.stem}_pointing_sam.mp4")


def open_mp4_writer(output_path: Path, width: int, height: int, fps: int, codec: str) -> cv2.VideoWriter:
    codec = codec.strip()
    if len(codec) != 4:
        raise ValueError(f"MP4 codec must be a four-character code, got {codec!r}")
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"could not open MP4 output {output_path} with codec {codec!r}; "
            "try --codec avc1 or --codec mp4v"
        )
    return writer


def render_rosbag(
    bag_path: str | Path,
    output_path: str | Path | None = None,
    *,
    overwrite: bool = False,
    output_fps: int = 0,
    codec: str = "mp4v",
) -> Path:
    """Render a finite RealSense recording through the live pointing/SAM pipeline."""
    source = RealSenseBag(bag_path)
    resolved_bag_path = source.path
    target = Path(output_path).expanduser().resolve() if output_path else default_bag_output_path(resolved_bag_path)
    if target.suffix.lower() != ".mp4":
        raise ValueError(f"output must be an .mp4 file, got {target}")
    if target.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {target} (use --overwrite to replace it)")
    target.parent.mkdir(parents=True, exist_ok=True)

    pointing: PointingEstimator | None = None
    segmenter = SAMPointSegmenter()
    hit_estimator = PointHitEstimator()
    writer: cv2.VideoWriter | None = None
    last_result: SegmentationResult | None = None
    last_prompt_pixel: tuple[int, int] | None = None
    last_prompt_source = ""
    last_sam_time_s: float | None = None
    first_timestamp_s: float | None = None
    frame_count = 0

    try:
        print(f"[bag] opening {resolved_bag_path}")
        source.start()
        if source.intrinsics is None:
            raise RuntimeError("RealSense bag color intrinsics unavailable")

        print("[bag] loading hand tracker")
        pointing = PointingEstimator()
        print("[bag] loading SAM")
        segmenter.load()
        fps = output_fps if output_fps > 0 else source.fps
        print(f"[bag] rendering {source.width}x{source.height} at {fps} fps to {target}")

        while True:
            frame_start = time.perf_counter()
            frame = source.read()
            if frame is None:
                break
            color, depth = frame

            if writer is None:
                writer = open_mp4_writer(target, color.shape[1], color.shape[0], fps, codec)
            elif (color.shape[1], color.shape[0]) != (source.width, source.height):
                raise RuntimeError("bag color resolution changed during playback; MP4 output requires a fixed resolution")

            if source.last_timestamp_s is not None:
                if first_timestamp_s is None:
                    first_timestamp_s = source.last_timestamp_s
                video_time_s = max(0.0, source.last_timestamp_s - first_timestamp_s)
            else:
                video_time_s = frame_count / float(fps)

            start = time.perf_counter()
            ray = pointing.estimate(color, depth, source.intrinsics)
            hand_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            if ray is not None:
                prompt = hit_estimator.estimate(ray, depth, source.intrinsics)
            else:
                hit_estimator.reset_smoothing()
                prompt = None
            hit_ms = (time.perf_counter() - start) * 1000.0

            if prompt is None:
                last_result = None
                last_prompt_pixel = None
                last_prompt_source = ""
                last_sam_time_s = None
            else:
                prompt_jump_px = 0.0
                if last_prompt_pixel is not None:
                    prompt_jump_px = math.hypot(
                        prompt.pixel[0] - last_prompt_pixel[0],
                        prompt.pixel[1] - last_prompt_pixel[1],
                    )
                if prompt.source != last_prompt_source or prompt_jump_px > env_float("SAM_RESET_PROMPT_JUMP_PX", 90.0):
                    last_result = None
                    last_sam_time_s = None
                last_prompt_pixel = prompt.pixel
                last_prompt_source = prompt.source

            sam_ms = 0.0
            sam_allowed = prompt is not None and (
                prompt.source != "ray_projected" or env_bool("SAM_ON_PROJECTED_PROMPT", False)
            )
            sam_interval_s = max(0.1, env_float("SAM_INTERVAL_S", 0.75))
            interval_due = last_sam_time_s is None or video_time_s - last_sam_time_s >= sam_interval_s
            if sam_allowed and prompt is not None and interval_due:
                try:
                    result = segmenter.predict(color, prompt)
                    segmenter.load_error = ""
                    last_result = result
                    sam_ms = result.elapsed_ms if result is not None else 0.0
                except Exception as exc:
                    segmenter.load_error = str(exc)
                    last_result = None
                last_sam_time_s = video_time_s
            elif not sam_allowed:
                last_result = None
                last_sam_time_s = None

            total_ms = (time.perf_counter() - frame_start) * 1000.0
            fps_actual = 1000.0 / total_ms if total_ms > 0.0 else 0.0
            if ray is None:
                status = "ray=no point=no mask=no"
            elif prompt is None:
                status = "ray=yes point=no mask=no"
            else:
                status = f"ray=yes point={prompt.pixel} mask={'yes' if last_result is not None else 'no'}"
            if segmenter.load_error:
                status = f"SAM error: {segmenter.load_error[:120]}"

            overlay = draw_overlay(color, ray, prompt, last_result, status)
            writer.write(overlay)
            frame_count += 1

            if frame_count % 100 == 0:
                print(
                    f"[bag] rendered {frame_count} frames "
                    f"({fps_actual:.1f} processing fps, hand={hand_ms:.1f}ms, hit={hit_ms:.1f}ms, SAM={sam_ms:.1f}ms)"
                )
    finally:
        if writer is not None:
            writer.release()
        try:
            source.stop()
        except Exception:
            pass

    if frame_count == 0:
        raise RuntimeError(f"no RGB-D frames were read from {resolved_bag_path}")
    print(f"[bag] complete: {frame_count} frames written to {target}")
    return target


app = FastAPI()
state = SharedState()


@app.on_event("startup")
def start_worker() -> None:
    thread = threading.Thread(target=worker, args=(state,), daemon=True)
    thread.start()


@app.on_event("shutdown")
def stop_worker() -> None:
    state.running = False


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html>
  <head>
    <title>RealSense SAM Pointing</title>
    <style>
      body { margin: 0; font-family: system-ui, sans-serif; background: #101114; color: #f5f5f5; }
      main { min-height: 100vh; display: grid; grid-template-rows: auto 1fr; }
      header { display: flex; align-items: center; gap: 14px; padding: 12px 16px; background: #1b1d22; }
      h1 { margin: 0; font-size: 16px; font-weight: 650; }
      .status { margin-left: auto; opacity: .78; font-size: 14px; }
      img { width: 100%; height: calc(100vh - 49px); object-fit: contain; background: #050505; }
    </style>
  </head>
  <body>
    <main>
      <header>
        <h1>RealSense SAM Pointing</h1>
        <div id="status" class="status">starting</div>
      </header>
      <img src="/stream" />
      <script>
        async function refreshStatus() {
          try {
            const response = await fetch('/status', { cache: 'no-store' });
            const data = await response.json();
            const metrics = data.metrics || {};
            document.getElementById('status').textContent =
              `${data.status} | ${metrics.fps || 0} fps | ${metrics.sam_ms || 0}ms SAM`;
          } catch (error) {
            document.getElementById('status').textContent = 'status unavailable';
          }
        }
        setInterval(refreshStatus, 750);
        refreshStatus();
      </script>
    </main>
  </body>
</html>
"""


@app.get("/status")
def status() -> dict[str, object]:
    _, current = state.get_frame()
    return {
        "status": current,
        "metrics": state.get_metrics(),
    }


def stream_frames():
    while True:
        jpeg, _ = state.get_frame()
        if jpeg is None:
            image = blank_frame("Waiting for first frame...")
            ok, encoded = cv2.imencode(".jpg", image)
            jpeg = encoded.tobytes() if ok else b""
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(1.0 / 20.0)


@app.get("/stream")
def stream() -> StreamingResponse:
    return StreamingResponse(stream_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the live RealSense pointing demo or render a RealSense .bag recording to MP4."
    )
    parser.add_argument("bag_path", nargs="?", help="RealSense .bag recording to render")
    parser.add_argument("--bag", dest="bag_option", help="RealSense .bag recording to render")
    parser.add_argument("--output", help="MP4 output path (defaults beside the bag)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing MP4 output")
    parser.add_argument("--fps", type=int, default=0, help="output FPS; defaults to the recorded color stream FPS")
    parser.add_argument(
        "--codec",
        default=os.getenv("BAG_VIDEO_CODEC", "mp4v"),
        help="four-character OpenCV MP4 codec (default: mp4v)",
    )
    args = parser.parse_args(argv)
    if args.bag_path and args.bag_option:
        parser.error("provide the bag either positionally or with --bag, not both")
    args.bag = args.bag_option or args.bag_path
    if (args.output or args.overwrite or args.fps or args.codec != os.getenv("BAG_VIDEO_CODEC", "mp4v")) and not args.bag:
        parser.error("--output, --overwrite, --fps, and --codec require a bag path")
    if args.fps < 0:
        parser.error("--fps must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bag:
        render_rosbag(
            args.bag,
            args.output or os.getenv("BAG_OUTPUT_PATH") or None,
            overwrite=args.overwrite or env_bool("BAG_OVERWRITE", False),
            output_fps=args.fps,
            codec=args.codec,
        )
        return 0

    uvicorn.run(app, host="0.0.0.0", port=8000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
