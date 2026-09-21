"""Send aligned RealSense RGB-D frames to sam_pointing_demo --api."""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request

import numpy as np
import pyrealsense2 as rs

from frame_transport import encode_frame


def start_camera(width: int, height: int, fps: int, serial: str | None) -> tuple[rs.pipeline, rs.pipeline_profile]:
    pipeline = rs.pipeline()
    candidates = [(width, height, fps), (1280, 720, min(fps, 30)), (848, 480, min(fps, 30)), (640, 480, min(fps, 30))]
    last_error: Exception | None = None
    for candidate_width, candidate_height, candidate_fps in dict.fromkeys(candidates):
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, candidate_width, candidate_height, rs.format.bgr8, candidate_fps)
        config.enable_stream(rs.stream.depth, candidate_width, candidate_height, rs.format.z16, candidate_fps)
        try:
            return pipeline, pipeline.start(config)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"RealSense stream unavailable: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Send aligned RealSense frames to a SAM pointing API server")
    parser.add_argument("--url", default="http://127.0.0.1:8001/api/frames", help="frame upload URL, usually an SSH tunnel")
    parser.add_argument("--serial", help="RealSense serial number to open")
    parser.add_argument("--camera-id", help="source ID shown by the server (defaults to device serial)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--send-fps", type=float, default=10.0)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    args = parser.parse_args()
    if args.send_fps <= 0 or args.camera_fps <= 0:
        parser.error("frame rates must be positive")
    if not args.url.startswith(("http://", "https://")):
        parser.error("--url must be an HTTP URL")

    pipeline, profile = start_camera(args.width, args.height, args.camera_fps, args.serial)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    serial = profile.get_device().get_info(rs.camera_info.serial_number)
    camera_id = args.camera_id or serial
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_profile.get_intrinsics()
    print(f"Sending {intrinsics.width}x{intrinsics.height} from camera {camera_id} to {args.url}", flush=True)
    next_send = 0.0
    last_error_log = 0.0
    sent = 0
    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            now = time.monotonic()
            if now < next_send:
                continue
            next_send = now + 1.0 / args.send_fps
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            packet = encode_frame(color, depth, intrinsics, depth_scale, camera_id, args.jpeg_quality)
            request = urllib.request.Request(
                args.url,
                data=packet,
                headers={"Content-Type": "application/octet-stream"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    if response.status != 202:
                        raise RuntimeError(f"server returned HTTP {response.status}")
                sent += 1
                if sent % 100 == 0:
                    print(f"Sent {sent} frames", flush=True)
            except urllib.error.HTTPError as exc:
                if exc.code in {404, 409, 415}:
                    print(f"Server rejected camera {camera_id}: HTTP {exc.code} {exc.reason}", flush=True)
                    return
                if time.monotonic() - last_error_log >= 2.0:
                    print(f"Upload failed: HTTP {exc.code} {exc.reason}", flush=True)
                    last_error_log = time.monotonic()
            except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                if time.monotonic() - last_error_log >= 2.0:
                    print(f"Upload failed: {exc}", flush=True)
                    last_error_log = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
