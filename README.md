# RealSense Pointing SAM Segmentation Demo

This is the same Docker-style layout as `my_point`, but the runtime path is simpler:

```text
RealSense aligned RGB-D
-> hand keypoints create a 3D pointing ray
-> the ray is intersected with the visible depth cloud
-> that 2D hit coordinate is sent to SAM with the full RGB frame
-> SAM returns the mask for the object under the point
```

There is no object detector, no object classes, no candidate boxes, and no VLM selection.

## Run

First accept the gated SAM3 model terms on Hugging Face and export a token, then build:

```bash
cd /home/sgeorgiou/point/sam_point
export HF_TOKEN=hf_your_token_here
docker compose build --no-cache
docker compose up
```

Open:

```text
http://localhost:8001
```

## SAM3 Checkpoint

This project now uses the official local SAM3 checkpoint repo:

```text
facebook/sam3
```

The point-click path uses the SAM3 Tracker classes from Transformers:

```text
Sam3TrackerModel + Sam3TrackerProcessor
```

That is the SAM3 mode that supports one positive point click and returns masks for the object at that point. The model is gated on Hugging Face, so you must first open this page, log in, and accept the access terms:

```text
https://huggingface.co/facebook/sam3
```

Then create a Hugging Face token and build with it:

```bash
cd /home/sgeorgiou/point/sam_point
export HF_TOKEN=hf_your_token_here
docker compose build --no-cache
docker compose up
```

During the image build, Docker downloads the checkpoint into:

```text
/opt/models/facebook-sam3
```

At runtime the app is forced to use SAM3 only:

```yaml
SAM_BACKEND: sam3
SAM3_MODEL_ID: /opt/models/facebook-sam3
```

If the token is missing, expired, or you have not accepted the model terms, the build will fail while downloading the checkpoint, and the app will not fall back to SAM1. That is intentional so you can tell immediately whether real SAM3 is running.

## Remote RealSense over SSH

Run the SAM server on the GPU machine in API mode. This mode waits for camera frames and does not open a local RealSense:

~~~bash
HOST_BIND=127.0.0.1 APP_FLAGS=--api docker compose up
~~~

The UI and upload API share port 8001. On the PC connected to the RealSense, create a local SSH tunnel to the server:

~~~bash
ssh -N -L 8001:127.0.0.1:8001 user@server
~~~

In another terminal on that camera PC, install the sender dependencies and start sending aligned RGB-D frames:

~~~bash
python -m pip install -r sender-requirements.txt
python remote_realsense_sender.py
~~~

The sender uses http://127.0.0.1:8001/api/frames by default, sends up to 10 frames per second, and uses the RealSense serial as its camera ID. Copy remote_realsense_sender.py, frame_transport.py, and sender-requirements.txt to the camera PC if this repository is not there. The server runs hand tracking, pointing, and SAM, then publishes the result to the existing UI at http://127.0.0.1:8001.

To view the UI from another PC, make the same SSH tunnel on that PC and open http://127.0.0.1:8001 in its browser. If you view it on the camera PC, its existing tunnel is enough.

To accept only one named camera, start the server with APP_FLAGS="--api --camera-id CAMERA_ID" and use --camera-id CAMERA_ID on the sender. The camera ID also appears in /status metrics. The upload endpoint accepts a binary POST containing a JPEG color frame, lossless 16-bit PNG depth frame, depth scale, and color intrinsics. Both frames must already be aligned; the included sender handles that. The upload endpoint is available only with --api.

Without Docker, start the server with python -m sam_pointing_demo --api --host 127.0.0.1 --port 8001 after installing the server dependencies and model. For a different sender URL, pass --url http://127.0.0.1:PORT/api/frames.
