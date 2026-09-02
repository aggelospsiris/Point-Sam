FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface \
    TORCH_HOME=/workspace/models/torch \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /workspace

ARG HF_TOKEN=""
ARG SAM3_MODEL_ID="facebook/sam3"
ARG SAM3_LOCAL_DIR="/opt/models/facebook-sam3"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install \
        --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.7.1+cu126 \
        torchvision==0.22.1+cu126 \
    && pip install -r requirements.txt

RUN python -c "from huggingface_hub import snapshot_download; token=\"${HF_TOKEN}\".strip(); model=\"${SAM3_MODEL_ID}\"; out=\"${SAM3_LOCAL_DIR}\"; __import__(\"sys\").exit(\"HF_TOKEN build arg is required for gated facebook/sam3 checkpoint download\") if not token else None; print(f\"Downloading {model} to {out}\"); snapshot_download(repo_id=model, local_dir=out, token=token)"

COPY . .

EXPOSE 8000

CMD ["python", "-m", "sam_pointing_demo"]
