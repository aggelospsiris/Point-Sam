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

## Render a recorded RealSense bag to MP4

Pass a `.bag` recording to the included helper. It starts the finite
`sam-pointing-video` container, mounts only the bag's parent directory, and
exits when rendering is complete:

```bash
./run_rosbag.sh /absolute/path/to/capture.bag
```

The output is written beside the recording as:

```text
/absolute/path/to/capture_pointing_sam.mp4
```

To choose the filename or intentionally replace an existing output:

```bash
./run_rosbag.sh /absolute/path/to/capture.bag annotated.mp4 --overwrite
```

The batch service reuses the live service's hand tracking, aligned RGB-D
pointing ray, SAM configuration, and overlay drawing. It reads the color and
depth streams directly from the bag, keeps the recording's color FPS, and
does not need a physical RealSense camera attached. The first run builds the
same gated SAM3 image, so export `HF_TOKEN` as in the live setup.

If you prefer to invoke Compose directly, mount the directory containing the
bag and use container paths for the input and output:

```bash
ROSBAG_DIR=/absolute/path/to \
ROSBAG_PATH=/recording/capture.bag \
VIDEO_OUTPUT=/recording/capture_pointing_sam.mp4 \
docker compose --profile video run --rm --build sam-pointing-video
```

`mp4v` is the default MP4 codec. If a downstream player requires H.264 and
your OpenCV build supports it, set `BAG_VIDEO_CODEC=avc1` before running the
helper or Compose command.

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
