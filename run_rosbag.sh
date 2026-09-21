#!/usr/bin/env bash
# Render a recorded RealSense RGB-D bag with the same pointing/SAM overlay as live mode.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

usage() {
  echo "Usage: $0 /absolute/or/relative/capture.bag [output.mp4] [--overwrite]" >&2
  echo "The MP4 is written beside the bag unless an output filename is supplied." >&2
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage
  exit 2
fi

bag_input=$1
shift
bag_dir=$(cd -- "$(dirname -- "$bag_input")" && pwd -P)
bag_filename=$(basename -- "$bag_input")
bag_path="$bag_dir/$bag_filename"

if [[ ! -f "$bag_path" ]]; then
  echo "RealSense bag not found: $bag_path" >&2
  exit 2
fi

output_filename="${bag_filename%.*}_pointing_sam.mp4"
overwrite=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --overwrite)
      overwrite=1
      ;;
    *.mp4)
      if [[ "$1" == */* ]]; then
        echo "Output must be a filename; it is written beside the bag." >&2
        exit 2
      fi
      output_filename=$1
      ;;
    *)
      usage
      exit 2
      ;;
  esac
  shift
done

output_path="$bag_dir/$output_filename"
if [[ -e "$output_path" && $overwrite -ne 1 ]]; then
  echo "Output already exists: $output_path (pass --overwrite to replace it)" >&2
  exit 2
fi

echo "Rendering $bag_path"
echo "Output:    $output_path"
ROSBAG_DIR="$bag_dir" \
ROSBAG_PATH="/recording/$bag_filename" \
VIDEO_OUTPUT="/recording/$output_filename" \
BAG_OVERWRITE="$overwrite" \
docker compose \
  --project-directory "$script_dir" \
  -f "$script_dir/docker-compose.yml" \
  --profile video run --rm --build sam-pointing-video
