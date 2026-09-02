from __future__ import annotations

import importlib
import math
import os
import threading
import time
import traceback
from dataclasses import dataclass
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


@dataclass
class PointingRay:
    origin_3d: np.ndarray
    direction_3d: np.ndarray
    start_2d: tuple[int, int]
    end_2d: tuple[int, int]


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


class PointingEstimator:
    def __init__(self) -> None:
        self.backend = os.getenv("HAND_BACKEND", "mediapipe").strip().lower()
        self.smoothing = env_float("POINTING_SMOOTHING", 0.35)
        self.min_hand_ray_length_m = env_float("MIN_HAND_RAY_LENGTH_M", 0.035)
        self.keypoint_confidence = env_float("HAND_KEYPOINT_CONFIDENCE", 0.25)
        self.pointing_start_landmark = env_int("POINTING_START_LANDMARK", 5)
        self.pointing_end_landmark = env_int("POINTING_END_LANDMARK", 8)
        self.smoothed_origin_3d: np.ndarray | None = None
        self.smoothed_direction_3d: np.ndarray | None = None
        self.last_hand_confidence = 0.0
        self.last_hand_candidates = 0

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
            if keypoints_xy.shape[1] <= max(self.pointing_start_landmark, self.pointing_end_landmark):
                continue
            start_vis = float(keypoints_conf[hand_index, self.pointing_start_landmark])
            end_vis = float(keypoints_conf[hand_index, self.pointing_end_landmark])
            if start_vis < self.hand_keypoint_visibility or end_vis < self.hand_keypoint_visibility:
                continue
            start_px = keypoint_to_pixel(keypoints_xy[hand_index, self.pointing_start_landmark], image_bgr.shape[1], image_bgr.shape[0])
            end_px = keypoint_to_pixel(keypoints_xy[hand_index, self.pointing_end_landmark], image_bgr.shape[1], image_bgr.shape[0])
            ray = self.ray_from_pixels(start_px, end_px, depth_m, intrinsics)
            if ray is None:
                continue
            score = confidence + 0.20 * min(start_vis, end_vis)
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
            if len(keypoints) <= max(self.pointing_start_landmark, self.pointing_end_landmark):
                continue
            start_vis = float(keypoints[self.pointing_start_landmark, 2])
            end_vis = float(keypoints[self.pointing_end_landmark, 2])
            if start_vis < self.hand_keypoint_visibility or end_vis < self.hand_keypoint_visibility:
                continue
            start_px = self.yolo_keypoint_to_pixel(keypoints[self.pointing_start_landmark], scale, pad_x, pad_y, width, height)
            end_px = self.yolo_keypoint_to_pixel(keypoints[self.pointing_end_landmark], scale, pad_x, pad_y, width, height)
            ray = self.ray_from_pixels(start_px, end_px, depth_m, intrinsics)
            if ray is None:
                continue
            score = confidence + 0.20 * min(start_vis, end_vis)
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
            if len(landmarks) <= max(self.pointing_start_landmark, self.pointing_end_landmark):
                continue
            start_px = landmark_to_pixel(landmarks[self.pointing_start_landmark], width, height)
            end_px = landmark_to_pixel(landmarks[self.pointing_end_landmark], width, height)
            ray = self.ray_from_pixels(start_px, end_px, depth_m, intrinsics)
            if ray is None:
                continue
            score = handedness_scores[hand_index] if hand_index < len(handedness_scores) else 1.0
            if score > best_score:
                best_score = score
                best_ray = ray

        if best_ray is None:
            self.reset_smoothing()
            return None
        return self.smooth_ray(best_ray)

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
        if self.smoothed_origin_3d is None or self.smoothed_direction_3d is None or alpha <= 0.0:
            self.smoothed_origin_3d = ray.origin_3d
            self.smoothed_direction_3d = ray.direction_3d
            return ray

        origin = alpha * ray.origin_3d + (1.0 - alpha) * self.smoothed_origin_3d
        direction = normalize(alpha * ray.direction_3d + (1.0 - alpha) * self.smoothed_direction_3d)
        self.smoothed_origin_3d = origin
        self.smoothed_direction_3d = direction
        return PointingRay(origin, direction, ray.start_2d, ray.end_2d)

    def reset_smoothing(self) -> None:
        self.smoothed_origin_3d = None
        self.smoothed_direction_3d = None


class PointHitEstimator:
    def __init__(self) -> None:
        self.last_pixel: np.ndarray | None = None

    def estimate(self, ray: PointingRay, depth_m: np.ndarray, intrinsics: rs.intrinsics) -> PointPrompt | None:
        prompt = self.surface_hit(ray, depth_m, intrinsics)
        if prompt is None:
            prompt = self.fallback_prompt(ray, intrinsics, depth_m.shape[1], depth_m.shape[0])
        if prompt is None:
            self.last_pixel = None
            return None
        return self.smooth_prompt(prompt, depth_m.shape[1], depth_m.shape[0])

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
        max_offset = env_float("POINT_HIT_MAX_OFFSET_M", 0.10)
        close_mask = offsets <= max_offset
        if not np.any(close_mask):
            return None

        close_indices = np.where(close_mask)[0]
        offset_weight = env_float("POINT_HIT_OFFSET_WEIGHT", 4.0)
        scores = forward[close_indices] + offset_weight * offsets[close_indices]
        best = int(close_indices[int(np.argmin(scores))])
        pixel = (
            int(np.clip(round(float(xs[best])), 0, depth_m.shape[1] - 1)),
            int(np.clip(round(float(ys[best])), 0, depth_m.shape[0] - 1)),
        )
        return PointPrompt(
            pixel=pixel,
            point_3d=points[best].astype(np.float32),
            forward_m=float(forward[best]),
            ray_offset_m=float(offsets[best]),
            source="depth_hit",
        )

    def fallback_prompt(self, ray: PointingRay, intrinsics: rs.intrinsics, width: int, height: int) -> PointPrompt | None:
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
            source="fallback",
        )

    def smooth_prompt(self, prompt: PointPrompt, width: int, height: int) -> PointPrompt:
        alpha = float(np.clip(env_float("POINT_PIXEL_SMOOTHING", 0.45), 0.0, 1.0))
        current = np.array(prompt.pixel, dtype=np.float32)
        if self.last_pixel is None or alpha <= 0.0:
            self.last_pixel = current
            return prompt
        smoothed = alpha * current + (1.0 - alpha) * self.last_pixel
        self.last_pixel = smoothed
        pixel = (
            int(np.clip(round(float(smoothed[0])), 0, width - 1)),
            int(np.clip(round(float(smoothed[1])), 0, height - 1)),
        )
        return PointPrompt(pixel, prompt.point_3d, prompt.forward_m, prompt.ray_offset_m, prompt.source)


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
        return self.pick_mask(masks, scores)

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
        with torch.no_grad():
            outputs = model(**inputs)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.detach().float().cpu(),
            inputs["original_sizes"].detach().cpu(),
            inputs["reshaped_input_sizes"].detach().cpu(),
        )[0]
        scores = outputs.iou_scores.detach().float().cpu().numpy().reshape(-1)
        return self.pick_mask(masks.numpy(), scores)

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
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=env_bool("SAM_MULTIMASK", True))
        masks = processor.post_process_masks(outputs.pred_masks.detach().float().cpu(), inputs["original_sizes"].detach().cpu())[0]
        scores = first_present_output(outputs, "iou_scores", "predicted_iou", "scores")
        if hasattr(scores, "detach"):
            scores = scores.detach().float().cpu()
        return self.pick_mask(masks, scores)

    def pick_mask(self, masks: Any, scores: Any) -> SegmentationResult | None:
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
            best = 0
            score = 0.0
        else:
            scores_np = np.asarray(scores_np, dtype=np.float32).reshape(-1)
            best = int(np.argmax(scores_np[: masks_np.shape[0]]))
            score = float(scores_np[best])
        mask = masks_np[best] > 0.0
        return SegmentationResult(mask=mask, score=score, backend=self.backend_name, elapsed_ms=0.0)


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
) -> np.ndarray | None:
    x, y = pixel
    y0 = max(0, y - patch_radius)
    y1 = min(depth_m.shape[0], y + patch_radius + 1)
    x0 = max(0, x - patch_radius)
    x1 = min(depth_m.shape[1], x + patch_radius + 1)
    patch = depth_m[y0:y1, x0:x1]
    values = patch[valid_depth(patch)]
    if len(values) == 0:
        return None
    z = float(np.median(values))
    return np.array(
        [
            (x - intrinsics.ppx) / intrinsics.fx * z,
            (y - intrinsics.ppy) / intrinsics.fy * z,
            z,
        ],
        dtype=np.float32,
    )


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def project_point_3d(point: np.ndarray, intrinsics: rs.intrinsics) -> tuple[int, int] | None:
    x, y, z = [float(value) for value in point]
    if not np.isfinite(z) or z <= 0.03:
        return None
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
    last_result: SegmentationResult | None = None
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

        while state.running:
            frame_start = time.perf_counter()

            start = time.perf_counter()
            color, depth = camera.read()
            read_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            ray = pointing.estimate(color, depth, camera.intrinsics)
            hand_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            prompt = hit_estimator.estimate(ray, depth, camera.intrinsics) if ray is not None else None
            hit_ms = (time.perf_counter() - start) * 1000.0

            sam_ms = 0.0
            if prompt is not None and frame_count % max(1, env_int("SAM_EVERY_N", 1)) == 0:
                try:
                    result = segmenter.predict(color, prompt)
                    if result is not None:
                        last_result = result
                        sam_ms = result.elapsed_ms
                except Exception as exc:
                    last_result = None
                    segmenter.load_error = str(exc)

            if prompt is None:
                last_result = None

            total_ms = (time.perf_counter() - frame_start) * 1000.0
            fps = 1000.0 / total_ms if total_ms > 0 else 0.0
            if ray is None:
                status = "ray=no point=no mask=no"
            elif prompt is None:
                status = "ray=yes point=no mask=no"
            else:
                status = f"ray=yes point={prompt.pixel} mask={'yes' if last_result else 'no'}"
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
                "resolution": f"{color.shape[1]}x{color.shape[0]}",
                "sam_backend": segmenter.backend_name or segmenter.requested_backend,
                "hand_backend": pointing.backend if pointing is not None else "none",
                "hand_conf": round(pointing.last_hand_confidence, 3) if pointing is not None else 0.0,
                "hand_candidates": pointing.last_hand_candidates if pointing is not None else 0,
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
        try:
            camera.stop()
        except Exception:
            pass


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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
