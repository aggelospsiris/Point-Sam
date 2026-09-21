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
cd /path/to/sam_point
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
cd /path/to/sam_point
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

### 1. Start the server on pc2

After building the image as described above, start API mode on the GPU machine:

~~~bash
HOST_BIND=127.0.0.1 APP_FLAGS=--api docker compose up
~~~

API mode waits for uploaded frames instead of opening a local RealSense. In a second terminal on pc2, check that it is listening:

~~~bash
curl http://127.0.0.1:8001/status
~~~

The UI and frame upload API share port 8001. Docker Compose binds that port to pc2's loopback address, so a camera on another PC needs an SSH tunnel.

### 2. Connect the camera PC

If the RealSense and sender are on pc2, skip the SSH tunnel. Otherwise, run this on the PC connected to the RealSense:

~~~bash
ssh -fN -o ExitOnForwardFailure=yes -L 8001:127.0.0.1:8001 user@pc2
curl http://127.0.0.1:8001/status
~~~

Replace user@pc2 with your SSH login. The -fN options keep the tunnel running in the background. If you use ssh -N without -f, leave that terminal open and run the sender in a different terminal. Pressing Ctrl+C closes a foreground tunnel.

### 3. Send aligned color and depth

On the camera PC, in a checkout of this repository, install the small sender environment and start the sender:

~~~bash
python3 -m venv ~/realsense-sender-venv
. ~/realsense-sender-venv/bin/activate
python -m pip install -r sender-requirements.txt
python remote_realsense_sender.py
~~~

If the repository is not on the camera PC, copy remote_realsense_sender.py, frame_transport.py, and sender-requirements.txt there first. The sender uses http://127.0.0.1:8001/api/frames by default and sends up to 10 frames per second. It aligns depth to color before uploading and uses the RealSense serial as its camera ID.

Open http://127.0.0.1:8001 in a browser on the camera PC to view the processed stream. On another viewing PC, create the same SSH tunnel there and open that address. The server runs hand tracking, pointing, and SAM, then publishes the overlay to the UI.

To accept only one named camera, start the server with APP_FLAGS="--api --camera-id CAMERA_ID" and use --camera-id CAMERA_ID on the sender. The camera ID appears in the UI and /status metrics. The upload endpoint accepts a binary POST containing a JPEG color frame, lossless 16-bit PNG depth frame, depth scale, and color intrinsics. It is available only in --api mode.

If the sender reports "Connection refused", check /status on pc2 first, then on the camera PC. The latter must work before starting the sender. For a different local tunnel port, pass --url http://127.0.0.1:PORT/api/frames to the sender.

Without Docker, start the server with python -m sam_pointing_demo --api --host 127.0.0.1 --port 8001 after installing the server dependencies and model.
