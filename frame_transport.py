"""Wire format for aligned RealSense color and depth frames.

Each HTTP body is a four-byte big-endian JSON header length, the JSON header,
one JPEG color image, and one lossless 16-bit PNG depth image.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass

import cv2
import numpy as np


MAX_FRAME_BYTES = 24 * 1024 * 1024
MAX_HEADER_BYTES = 4096
MAX_WIDTH = 4096
MAX_HEIGHT = 2160


@dataclass
class ReceivedFrame:
    color: np.ndarray
    depth_m: np.ndarray
    intrinsics: dict[str, object]
    camera_id: str


def encode_frame(
    color: np.ndarray,
    depth_raw: np.ndarray,
    intrinsics: object,
    depth_scale: float,
    camera_id: str,
    jpeg_quality: int = 85,
) -> bytes:
    if color.ndim != 3 or color.shape[2] != 3 or depth_raw.shape != color.shape[:2] or depth_raw.dtype != np.uint16:
        raise ValueError("color must be BGR8 and aligned depth must be uint16")
    height, width = depth_raw.shape
    if not (0 < width <= MAX_WIDTH and 0 < height <= MAX_HEIGHT):
        raise ValueError("frame dimensions are out of range")
    quality = max(50, min(100, jpeg_quality))
    color_ok, color_jpeg = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, quality])
    depth_ok, depth_png = cv2.imencode(".png", depth_raw)
    if not color_ok or not depth_ok:
        raise ValueError("could not encode color or depth")
    header = {
        "width": width,
        "height": height,
        "color_length": len(color_jpeg),
        "depth_length": len(depth_png),
        "depth_scale": float(depth_scale),
        "camera_id": camera_id,
        "intrinsics": {
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "ppx": float(intrinsics.ppx),
            "ppy": float(intrinsics.ppy),
            "model": str(intrinsics.model).split(".")[-1],
            "coeffs": [float(value) for value in intrinsics.coeffs],
        },
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise ValueError("frame header is too large")
    packet = struct.pack("!I", len(header_bytes)) + header_bytes + color_jpeg.tobytes() + depth_png.tobytes()
    if len(packet) > MAX_FRAME_BYTES:
        raise ValueError("frame is too large")
    return packet


def decode_frame(packet: bytes) -> ReceivedFrame:
    if len(packet) < 4 or len(packet) > MAX_FRAME_BYTES:
        raise ValueError("invalid frame size")
    header_length = struct.unpack("!I", packet[:4])[0]
    if not (0 < header_length <= MAX_HEADER_BYTES) or len(packet) < 4 + header_length:
        raise ValueError("invalid frame header length")
    try:
        header = json.loads(packet[4 : 4 + header_length])
        width, height = int(header["width"]), int(header["height"])
        color_length, depth_length = int(header["color_length"]), int(header["depth_length"])
        scale = float(header["depth_scale"])
        camera_id = header["camera_id"]
        intrinsics = header["intrinsics"]
        camera_values = [float(intrinsics[name]) for name in ("fx", "fy", "ppx", "ppy")]
        coeffs = [float(value) for value in intrinsics["coeffs"]]
        model = intrinsics["model"]
    except (ValueError, OverflowError, TypeError, KeyError, IndexError) as exc:
        raise ValueError("invalid frame metadata") from exc
    if not (0 < width <= MAX_WIDTH and 0 < height <= MAX_HEIGHT):
        raise ValueError("frame dimensions are out of range")
    if color_length <= 0 or depth_length <= 0 or 4 + header_length + color_length + depth_length != len(packet):
        raise ValueError("frame payload lengths do not match")
    if not (math.isfinite(scale) and 0 < scale < 1):
        raise ValueError("invalid depth scale")
    if not isinstance(camera_id, str) or len(camera_id) > 128:
        raise ValueError("invalid camera ID")
    if not isinstance(model, str) or len(model) > 64 or len(coeffs) != 5:
        raise ValueError("invalid intrinsics")
    if not all(math.isfinite(value) for value in camera_values + coeffs) or camera_values[0] <= 0 or camera_values[1] <= 0:
        raise ValueError("invalid intrinsics")
    color_start = 4 + header_length
    depth_start = color_start + color_length
    try:
        color = cv2.imdecode(np.frombuffer(packet[color_start:depth_start], dtype=np.uint8), cv2.IMREAD_COLOR)
        depth_raw = cv2.imdecode(np.frombuffer(packet[depth_start:], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        raise ValueError("invalid image encoding") from exc
    if color is None or color.shape != (height, width, 3):
        raise ValueError("invalid color image")
    if depth_raw is None or depth_raw.shape != (height, width) or depth_raw.dtype != np.uint16:
        raise ValueError("invalid depth image")
    return ReceivedFrame(
        color=color,
        depth_m=depth_raw.astype(np.float32) * scale,
        intrinsics={
            "width": width,
            "height": height,
            "fx": camera_values[0],
            "fy": camera_values[1],
            "ppx": camera_values[2],
            "ppy": camera_values[3],
            "model": model,
            "coeffs": coeffs,
        },
        camera_id=camera_id,
    )
