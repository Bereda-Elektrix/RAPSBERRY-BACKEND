#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -d ".venv" ]; then
  . ".venv/bin/activate"
elif [ -d "/home/adi/Desktop/.venv" ]; then
  . "/home/adi/Desktop/.venv/bin/activate"
elif [ -d "/home/adi/.venv" ]; then
  . "/home/adi/.venv/bin/activate"
fi

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
CAMERA_FPS="${CAMERA_FPS:-15}"
CAMERA_COMMAND_FILE="${CAMERA_COMMAND_FILE:-/tmp/raspberry_camera_commands.json}"
RADAR_COMMAND_FILE="${RADAR_COMMAND_FILE:-/tmp/raspberry_radar_commands.json}"
AUTO_START_CAMERA="${AUTO_START_CAMERA:-1}"
AUTO_START_RADAR="${AUTO_START_RADAR:-1}"

rm -f "${CAMERA_COMMAND_FILE}" "${RADAR_COMMAND_FILE}"

export CAMERA_FPS
export CAMERA_COMMAND_FILE
export RADAR_COMMAND_FILE
export AUTO_START_CAMERA
export AUTO_START_RADAR

python -m uvicorn main_api:app --host "${HOST}" --port "${PORT}"
