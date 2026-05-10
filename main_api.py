#!/usr/bin/env python3

import asyncio
import base64
import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


BASE_DIR = Path(__file__).resolve().parent
CAMERA_SCRIPT = BASE_DIR / "s.py"
RADAR_SCRIPT = BASE_DIR / "radar_servo_scan.py"
CAMERA_COMMAND_FILE = os.environ.get(
    "CAMERA_COMMAND_FILE",
    "/tmp/raspberry_camera_commands.json",
)
RADAR_COMMAND_FILE = os.environ.get(
    "RADAR_COMMAND_FILE",
    "/tmp/raspberry_radar_commands.json",
)


@dataclass
class BackendState:
    clients: set[WebSocket] = field(default_factory=set)
    live_clients: set[WebSocket] = field(default_factory=set)
    radar_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    camera_process: asyncio.subprocess.Process | None = None
    camera_fps: int = 15
    camera_exit_code: int | None = None
    radar_process: asyncio.subprocess.Process | None = None
    radar_exit_code: int | None = None
    radar_scan_active: bool = False
    radar_tracking_active: bool = False
    radar_config: dict[str, Any] = field(
        default_factory=lambda: {
            "start_angle": 100,
            "end_angle": 180,
            "step_angle": 10,
            "start_mm": 300,
            "end_mm": 3000,
        }
    )
    last_radar: dict[str, Any] = field(default_factory=dict)
    last_camera_frame: dict[str, Any] = field(default_factory=dict)
    simulator_task: asyncio.Task | None = None
    simulator_fps: int = 5


state = BackendState()


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    if os.environ.get("AUTO_START_CAMERA", "1") == "1":
        await start_camera(int(os.environ.get("CAMERA_FPS", "15")))
    if os.environ.get("AUTO_START_RADAR", "1") == "1":
        await start_radar()

    try:
        yield
    finally:
        await stop_simulation()
        await stop_camera()
        await stop_radar()


app = FastAPI(title="raspberry-backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_status() -> dict[str, Any]:
    return {
        "camera_running": state.camera_process is not None
        and state.camera_process.returncode is None,
        "camera_fps": state.camera_fps,
        "camera_exit_code": state.camera_exit_code,
        "radar_running": state.radar_process is not None
        and state.radar_process.returncode is None,
        "radar_exit_code": state.radar_exit_code,
        "radar_scan_active": state.radar_scan_active,
        "radar_tracking_active": state.radar_tracking_active,
        "radar_config": state.radar_config,
        "simulator_running": state.simulator_task is not None
        and not state.simulator_task.done(),
        "simulator_fps": state.simulator_fps,
        "last_radar": state.last_radar or None,
        "clients": len(state.clients) + len(state.live_clients),
    }


async def send_json_safe(websocket: WebSocket, payload: dict[str, Any]) -> bool:
    try:
        await websocket.send_json(payload)
        return True
    except Exception:
        return False


def _round_float(value: Any, digits: int = 2, default: float = 0.0) -> float:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return default


def _build_live_data_payload() -> dict[str, Any]:
    camera = state.last_camera_frame or {}
    radar = state.last_radar or {}
    thermal = camera.get("thermal") if isinstance(camera.get("thermal"), dict) else {}
    stats = camera.get("stats") if isinstance(camera.get("stats"), dict) else {}
    target = camera.get("target") if isinstance(camera.get("target"), dict) else {}
    fusion = camera.get("fusion") if isinstance(camera.get("fusion"), dict) else {}
    orientation = (
        camera.get("orientation")
        if isinstance(camera.get("orientation"), dict)
        else {}
    )
    reading = radar.get("reading") if isinstance(radar.get("reading"), dict) else {}
    servo = camera.get("servo") if isinstance(camera.get("servo"), dict) else {}
    camera_running = state.camera_process is not None and state.camera_process.returncode is None

    distance_mm = reading.get("distance_mm")
    distance_m = (
        _round_float(distance_mm / 1000.0, 3)
        if isinstance(distance_mm, (int, float))
        else 0.0
    )
    radar_angle_deg = int(reading.get("angle_deg", 0))
    target_detected = bool(target)
    tracking_enabled = bool(camera.get("tracking_enabled", False))
    motion_detected = target_detected or tracking_enabled
    radar_following = bool(fusion.get("radar_following", False))
    target_correlated = target_detected and bool(reading) and distance_m > 0 and radar_following
    correlated_target = (
        {
            **target,
            "distance_m": distance_m,
            "distance_mm": distance_mm,
            "radar_angle_deg": radar_angle_deg,
            "correlated": target_correlated,
        }
        if target_detected
        else None
    )

    return {
        "type": "live_data",
        "camera": {
            "running": camera_running,
            "format": camera.get("format", "jpeg"),
            "image": camera.get("image"),
            "orientation": {
                "rotation": int(orientation.get("rotation", 270)),
                "flip_vertical": bool(orientation.get("flip_vertical", False)),
            },
            "stats": {
                "fps": int(stats.get("fps", state.camera_fps)),
                "center_temp": _round_float(stats.get("center_temp")),
                "hotspot_temp": _round_float(stats.get("hotspot_temp")),
                "min_temp": _round_float(stats.get("min_temp")),
                "max_temp": _round_float(stats.get("max_temp")),
            },
        },
        "thermal": {
            "width": int(thermal.get("width", 80)),
            "height": int(thermal.get("height", 62)),
            "min": _round_float(thermal.get("min", stats.get("min_temp"))),
            "max": _round_float(thermal.get("max", stats.get("max_temp"))),
            "avg": _round_float(thermal.get("avg")),
            "frame": thermal.get("frame", []),
        },
        "radar": {
            "presence": bool(reading) and distance_m > 0,
            "motion": motion_detected,
            "distance_m": distance_m,
            "distance_mm": distance_mm,
            "angle_deg": radar_angle_deg,
            "correlated": target_correlated,
        },
        "target": correlated_target,
        "servo": {
            "x_angle": int(servo.get("x_angle", 90)),
            "y_angle": int(servo.get("y_angle", 90)),
            "tracking_enabled": tracking_enabled,
        },
        "fusion": {
            "enabled": bool(fusion.get("enabled", False)),
            "target_locked": bool(fusion.get("target_locked", False)),
            "radar_following": radar_following,
            "target_correlated": target_correlated,
        },
        "timestamp": camera.get("timestamp") or radar.get("timestamp") or time.time(),
        "source": "raspberry_pi",
    }


def _latest_camera_jpeg_bytes() -> bytes | None:
    image_base64 = state.last_camera_frame.get("image")
    if not isinstance(image_base64, str) or not image_base64:
        return None

    try:
        return base64.b64decode(image_base64)
    except Exception:
        return None


def _update_cached_state(payload: dict[str, Any]) -> None:
    payload_type = payload.get("type")
    if payload_type == "camera_frame":
        state.last_camera_frame = payload
        fusion = payload.get("fusion")
        if isinstance(fusion, dict):
            if fusion.get("enabled"):
                state.radar_scan_active = not bool(fusion.get("radar_following", False))
                state.radar_tracking_active = bool(fusion.get("radar_following", False))
            else:
                state.radar_tracking_active = False
    elif payload_type == "radar_read":
        state.last_radar = payload


async def _iter_stream_lines(
    stream: asyncio.StreamReader,
    *,
    chunk_size: int = 16384,
    max_buffer_size: int = 262144,
):
    buffer = bytearray()

    while True:
        chunk = await stream.read(chunk_size)
        if not chunk:
            break

        buffer.extend(chunk)
        while True:
            newline_index = buffer.find(b"\n")
            if newline_index == -1:
                break

            line = bytes(buffer[:newline_index])
            del buffer[: newline_index + 1]
            yield line

        if len(buffer) >= max_buffer_size:
            yield bytes(buffer)
            buffer.clear()

    if buffer:
        yield bytes(buffer)


async def _wait_for_camera_frame(timeout_seconds: float = 2.0) -> bytes | None:
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        jpeg_bytes = _latest_camera_jpeg_bytes()
        if jpeg_bytes is not None:
            return jpeg_bytes
        await asyncio.sleep(0.05)

    return _latest_camera_jpeg_bytes()


async def _broadcast_live_payload() -> None:
    if not state.live_clients:
        return

    payload = _build_live_data_payload()
    stale_clients: list[WebSocket] = []
    for websocket in list(state.live_clients):
        if not await send_json_safe(websocket, payload):
            stale_clients.append(websocket)
    for websocket in stale_clients:
        state.live_clients.discard(websocket)


async def broadcast(payload: dict[str, Any]) -> None:
    _update_cached_state(payload)
    stale_clients: list[WebSocket] = []
    for websocket in list(state.clients):
        if not await send_json_safe(websocket, payload):
            stale_clients.append(websocket)
    for websocket in stale_clients:
        state.clients.discard(websocket)
    await _broadcast_live_payload()


async def _watch_camera_process(process: asyncio.subprocess.Process) -> None:
    return_code = await process.wait()
    if state.camera_process is process:
        state.camera_process = None
        state.camera_exit_code = return_code
        await broadcast(
            {
                "type": "camera_exit",
                "return_code": return_code,
                "status": build_status(),
            }
        )


async def _watch_radar_process(process: asyncio.subprocess.Process) -> None:
    return_code = await process.wait()
    if state.radar_process is process:
        state.radar_process = None
        state.radar_exit_code = return_code
        await broadcast(
            {
                "type": "radar_exit",
                "return_code": return_code,
                "status": build_status(),
            }
        )


async def start_camera(fps: int = 15) -> dict[str, Any]:
    if state.camera_process is not None and state.camera_process.returncode is None:
        return {
            "ok": True,
            "message": "camera already running",
            "status": build_status(),
        }

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(CAMERA_SCRIPT),
        str(fps),
        cwd=str(BASE_DIR),
        env={
            **os.environ,
            "STREAM_WS": "1",
            "CAMERA_COMMAND_FILE": CAMERA_COMMAND_FILE,
        },
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    state.camera_process = process
    state.camera_fps = fps
    state.camera_exit_code = None
    asyncio.create_task(_stream_process_stdout(process, "camera"))
    asyncio.create_task(_stream_process_stderr(process, "camera"))
    asyncio.create_task(_watch_camera_process(process))
    return {
        "ok": True,
        "message": "camera started",
        "pid": process.pid,
        "status": build_status(),
    }


async def start_radar() -> dict[str, Any]:
    if state.radar_process is not None and state.radar_process.returncode is None:
        return {
            "ok": True,
            "message": "radar already running",
            "status": build_status(),
        }

    config = state.radar_config
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(RADAR_SCRIPT),
        cwd=str(BASE_DIR),
        env={
            **os.environ,
            "RADAR_COMMAND_FILE": RADAR_COMMAND_FILE,
            "RADAR_START_ANGLE": str(config["start_angle"]),
            "RADAR_END_ANGLE": str(config["end_angle"]),
            "RADAR_STEP_ANGLE": str(config["step_angle"]),
            "RADAR_START_MM": str(config["start_mm"]),
            "RADAR_END_MM": str(config["end_mm"]),
        },
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    state.radar_process = process
    state.radar_exit_code = None
    state.radar_scan_active = True
    asyncio.create_task(_stream_process_stdout(process, "radar"))
    asyncio.create_task(_stream_process_stderr(process, "radar"))
    asyncio.create_task(_watch_radar_process(process))
    return {
        "ok": True,
        "message": "radar started",
        "pid": process.pid,
        "status": build_status(),
    }


async def _stream_process_stdout(
    process: asyncio.subprocess.Process,
    source: str,
) -> None:
    if process.stdout is None:
        return

    async for line in _iter_stream_lines(process.stdout):
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            continue

        if source == "camera":
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                await broadcast({"type": "camera_log", "message": text})
                continue

            if isinstance(payload, dict):
                await broadcast(payload)
            continue

        payload = _parse_radar_log_line(text)
        if payload is None:
            await broadcast({"type": "radar_log", "message": text})
        else:
            await broadcast(payload)


async def _stream_process_stderr(
    process: asyncio.subprocess.Process,
    source: str,
) -> None:
    if process.stderr is None:
        return

    log_type = "camera_log" if source == "camera" else "radar_log"

    async for line in _iter_stream_lines(process.stderr):
        text = line.decode("utf-8", errors="replace").strip()
        if text:
            await broadcast({"type": log_type, "message": text})


def _parse_radar_log_line(text: str) -> dict[str, Any] | None:
    if "|" not in text or "kat=" not in text:
        return None

    reading: dict[str, Any] = {
        "angle_deg": 0,
        "distance_mm": None,
        "strength": 0.0,
        "temperature_c": 0.0,
        "num_distances": 0,
    }

    try:
        for raw_part in text.split("|"):
            part = raw_part.strip()
            if part.startswith("kat="):
                reading["angle_deg"] = int(part.split("=", 1)[1].split()[0])
            elif part.startswith("odleglosc="):
                reading["distance_mm"] = int(part.split("=", 1)[1].split()[0])
            elif part.startswith("sila="):
                reading["strength"] = float(part.split("=", 1)[1].strip())
            elif part.startswith("temp="):
                reading["temperature_c"] = float(part.split("=", 1)[1].split()[0])
            elif part.startswith("peaks="):
                reading["num_distances"] = int(part.split("=", 1)[1].strip())
            elif "brak obiektu" in part:
                reading["distance_mm"] = None
    except Exception:
        return None

    payload = {
        "type": "radar_read",
        "ok": True,
        "reading": reading,
        "start_mm": state.radar_config["start_mm"],
        "end_mm": state.radar_config["end_mm"],
        "timestamp": time.time(),
    }
    state.last_radar = payload
    return payload


async def read_radar_once(start_mm: int, end_mm: int) -> dict[str, Any]:
    async with state.radar_lock:
        last = state.last_radar
        if last:
            return last

        return {
            "ok": False,
            "start_mm": start_mm,
            "end_mm": end_mm,
            "error": "No radar data is available yet from the worker process.",
            "timestamp": time.time(),
        }


async def stop_camera() -> dict[str, Any]:
    process = state.camera_process
    if process is None or process.returncode is not None:
        state.camera_process = None
        return {
            "ok": True,
            "message": "camera already stopped",
            "status": build_status(),
        }

    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()

    state.camera_process = None
    state.camera_exit_code = process.returncode
    return {
        "ok": True,
        "message": "camera stopped",
        "return_code": process.returncode,
        "status": build_status(),
    }


async def stop_radar() -> dict[str, Any]:
    process = state.radar_process
    if process is None or process.returncode is not None:
        state.radar_process = None
        return {
            "ok": True,
            "message": "radar already stopped",
            "status": build_status(),
        }

    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()

    state.radar_process = None
    state.radar_exit_code = process.returncode
    state.radar_scan_active = False
    return {
        "ok": True,
        "message": "radar stopped",
        "return_code": process.returncode,
        "status": build_status(),
    }


def normalize_ws_message(raw_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    return {"action": raw_text.strip()}


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    os.replace(temp_path, path)


def _write_camera_command(command: dict[str, Any]) -> None:
    _atomic_write_json(CAMERA_COMMAND_FILE, command)


def _write_radar_command(command: dict[str, Any]) -> None:
    _atomic_write_json(RADAR_COMMAND_FILE, command)


async def handle_ws_action(payload: dict[str, Any]) -> dict[str, Any]:
    action = str(payload.get("action", "")).strip().lower()

    if action == "ping":
        return {"type": "pong", "status": build_status()}

    if action == "status":
        return {"type": "status", "status": build_status()}

    if action == "camera_start":
        fps = int(payload.get("fps", 15))
        return {"type": "camera_start", **await start_camera(fps)}

    if action == "camera_stop":
        return {"type": "camera_stop", **await stop_camera()}

    if action == "radar_start":
        return {"type": "radar_start", **await start_radar()}

    if action == "radar_stop":
        return {"type": "radar_stop", **await stop_radar()}

    if action == "radar_read":
        start_mm = int(payload.get("start_mm", state.radar_config["start_mm"]))
        end_mm = int(payload.get("end_mm", state.radar_config["end_mm"]))
        return {"type": "radar_read", **await read_radar_once(start_mm, end_mm)}

    if action == "simulate_start":
        fps = int(payload.get("fps", 5))
        return {"type": "simulate_start", **await start_simulation(fps)}

    if action == "simulate_stop":
        return {"type": "simulate_stop", **await stop_simulation()}

    return {
        "type": "error",
        "ok": False,
        "error": (
            "unknown action; use ping, status, camera_start, camera_stop, "
            "radar_start, radar_stop, radar_read, simulate_start, or simulate_stop"
        ),
    }


async def handle_live_ws_command(payload: dict[str, Any]) -> dict[str, Any]:
    command = str(payload.get("command", "")).strip().lower()

    if not command:
        return _build_live_data_payload()

    if command == "set_camera_servo":
        if state.camera_process is None or state.camera_process.returncode is not None:
            await start_camera(state.camera_fps)
        x_angle = int(payload.get("x_angle", 90))
        y_angle = int(payload.get("y_angle", 90))
        await asyncio.to_thread(
            _write_camera_command,
            {
                "command": command,
                "x_angle": x_angle,
                "y_angle": y_angle,
                "timestamp": time.time(),
            },
        )
        if not state.last_camera_frame:
            state.last_camera_frame = {}
        state.last_camera_frame["servo"] = {"x_angle": x_angle, "y_angle": y_angle}
        state.last_camera_frame["tracking_enabled"] = False
        return _build_live_data_payload()

    if command == "set_camera_orientation":
        if state.camera_process is None or state.camera_process.returncode is not None:
            await start_camera(state.camera_fps)
        rotation = int(payload.get("rotation", 180))
        flip_vertical = bool(payload.get("flip_vertical", False))
        await asyncio.to_thread(
            _write_camera_command,
            {
                "command": command,
                "rotation": rotation,
                "flip_vertical": flip_vertical,
                "timestamp": time.time(),
            },
        )
        if not state.last_camera_frame:
            state.last_camera_frame = {}
        state.last_camera_frame["orientation"] = {
            "rotation": rotation,
            "flip_vertical": flip_vertical,
        }
        return _build_live_data_payload()

    if command in {"start_tracking", "stop_tracking", "start_fusion", "stop_fusion"}:
        if state.camera_process is None or state.camera_process.returncode is not None:
            await start_camera(state.camera_fps)
        await asyncio.to_thread(
            _write_camera_command,
            {
                "command": command,
                "timestamp": time.time(),
            },
        )
        if not state.last_camera_frame:
            state.last_camera_frame = {}
        if command in {"start_tracking", "stop_tracking"}:
            tracking_enabled = command == "start_tracking"
            state.last_camera_frame["tracking_enabled"] = tracking_enabled
            state.radar_tracking_active = tracking_enabled
        else:
            fusion_enabled = command == "start_fusion"
            state.last_camera_frame["fusion"] = {
                "enabled": fusion_enabled,
                "target_locked": False,
                "radar_following": False,
            }
            state.radar_tracking_active = False
            if not fusion_enabled:
                state.radar_scan_active = True
        return _build_live_data_payload()

    if command == "set_radar_servo":
        if state.radar_process is None or state.radar_process.returncode is not None:
            await start_radar()
        angle = int(
            payload.get(
                "angle",
                payload.get("x_angle", payload.get("servo_angle", 90)),
            )
        )
        await asyncio.to_thread(
            _write_radar_command,
            {
                "command": command,
                "angle": angle,
                "timestamp": time.time(),
            },
        )
        state.radar_scan_active = False
        return _build_live_data_payload()

    if command == "configure_radar_scan":
        if state.radar_process is None or state.radar_process.returncode is not None:
            await start_radar()
        state.radar_config = {
            "start_angle": int(payload.get("start_angle", 100)),
            "end_angle": int(payload.get("end_angle", 180)),
            "step_angle": int(payload.get("step_angle", 10)),
            "start_mm": int(payload.get("start_mm", 300)),
            "end_mm": int(payload.get("end_mm", 3000)),
        }
        await asyncio.to_thread(
            _write_radar_command,
            {
                "command": command,
                **state.radar_config,
                "timestamp": time.time(),
            },
        )
        return _build_live_data_payload()

    if command == "start_radar_scan":
        if state.radar_process is None or state.radar_process.returncode is not None:
            await start_radar()
        await asyncio.to_thread(
            _write_radar_command,
            {
                "command": command,
                "timestamp": time.time(),
            },
        )
        state.radar_scan_active = True
        return _build_live_data_payload()

    if command == "stop_radar_scan":
        if state.radar_process is None or state.radar_process.returncode is not None:
            await start_radar()
        await asyncio.to_thread(
            _write_radar_command,
            {
                "command": command,
                "timestamp": time.time(),
            },
        )
        state.radar_scan_active = False
        return _build_live_data_payload()

    if command == "radar_selftest":
        if state.radar_process is None or state.radar_process.returncode is not None:
            await start_radar()
        await asyncio.to_thread(
            _write_radar_command,
            {
                "command": command,
                "timestamp": time.time(),
            },
        )
        state.radar_scan_active = False
        return _build_live_data_payload()

    if command == "ping":
        return _build_live_data_payload()

    return {
        "error": f"unsupported command: {command}",
        **_build_live_data_payload(),
    }


async def start_simulation(fps: int = 5) -> dict[str, Any]:
    if state.simulator_task is not None and not state.simulator_task.done():
        return {
            "ok": True,
            "message": "simulation already running",
            "status": build_status(),
        }

    state.simulator_fps = fps
    state.simulator_task = asyncio.create_task(_simulation_loop(fps))
    return {
        "ok": True,
        "message": "simulation started",
        "status": build_status(),
    }


async def stop_simulation() -> dict[str, Any]:
    task = state.simulator_task
    if task is None or task.done():
        state.simulator_task = None
        return {
            "ok": True,
            "message": "simulation already stopped",
            "status": build_status(),
        }

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    state.simulator_task = None
    return {
        "ok": True,
        "message": "simulation stopped",
        "status": build_status(),
    }


async def _simulation_loop(fps: int) -> None:
    try:
        while True:
            await asyncio.sleep(1.0 / max(1, fps))
    except asyncio.CancelledError:
        raise


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "status": build_status()}


@app.get("/status")
async def status() -> dict[str, Any]:
    return {"ok": True, "status": build_status()}


@app.get("/live")
async def live() -> dict[str, Any]:
    if state.camera_process is None or state.camera_process.returncode is not None:
        await start_camera(state.camera_fps)
    return _build_live_data_payload()


@app.get("/video_feed")
async def video_feed() -> Response:
    if state.camera_process is None or state.camera_process.returncode is not None:
        await start_camera(state.camera_fps)

    jpeg_bytes = await _wait_for_camera_frame()
    if jpeg_bytes is None:
        raise HTTPException(
            status_code=503,
            detail="No camera frame is available yet.",
        )

    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.get("/camera/frame.jpg")
async def camera_frame_jpg() -> Response:
    return await video_feed()


@app.post("/camera/servo")
async def camera_servo(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        x_angle = int(payload["x_angle"])
        y_angle = int(payload["y_angle"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="x_angle and y_angle are required integers",
        ) from exc

    response = await handle_live_ws_command(
        {
            "command": "set_camera_servo",
            "x_angle": x_angle,
            "y_angle": y_angle,
        }
    )
    return {"ok": True, "liveData": response}


@app.post("/camera/orientation")
async def camera_orientation(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        rotation = int(payload.get("rotation", 180))
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="rotation must be one of: 0, 90, 180, 270",
        ) from exc

    if rotation not in {0, 90, 180, 270}:
        raise HTTPException(
            status_code=400,
            detail="rotation must be one of: 0, 90, 180, 270",
        )

    flip_vertical = bool(payload.get("flip_vertical", False))
    response = await handle_live_ws_command(
        {
            "command": "set_camera_orientation",
            "rotation": rotation,
            "flip_vertical": flip_vertical,
        }
    )
    return {"ok": True, "liveData": response}


@app.post("/radar/servo")
async def radar_servo(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        angle = int(
            payload.get(
                "angle",
                payload.get("x_angle", payload.get("servo_angle")),
            )
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="angle or x_angle is required as integer",
        ) from exc

    response = await handle_live_ws_command(
        {
            "command": "set_radar_servo",
            "angle": angle,
        }
    )
    return {"ok": True, "liveData": response}


@app.post("/radar/scan/configure")
async def radar_scan_configure(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        start_angle = int(payload["start_angle"])
        end_angle = int(payload["end_angle"])
        step_angle = int(payload["step_angle"])
        start_mm = int(payload["start_mm"])
        end_mm = int(payload["end_mm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "start_angle, end_angle, step_angle, start_mm and end_mm "
                "are required integers"
            ),
        ) from exc

    response = await handle_live_ws_command(
        {
            "command": "configure_radar_scan",
            "start_angle": start_angle,
            "end_angle": end_angle,
            "step_angle": step_angle,
            "start_mm": start_mm,
            "end_mm": end_mm,
        }
    )
    return {"ok": True, "liveData": response}


@app.post("/radar/scan/start")
async def radar_scan_start() -> dict[str, Any]:
    response = await handle_live_ws_command({"command": "start_radar_scan"})
    return {"ok": True, "liveData": response}


@app.post("/radar/scan/stop")
async def radar_scan_stop() -> dict[str, Any]:
    response = await handle_live_ws_command({"command": "stop_radar_scan"})
    return {"ok": True, "liveData": response}


@app.post("/radar/selftest")
async def radar_selftest() -> dict[str, Any]:
    response = await handle_live_ws_command({"command": "radar_selftest"})
    return {"ok": True, "liveData": response}


@app.post("/tracking/start")
async def tracking_start() -> dict[str, Any]:
    response = await handle_live_ws_command({"command": "start_tracking"})
    return {"ok": True, "liveData": response}


@app.post("/tracking/stop")
async def tracking_stop() -> dict[str, Any]:
    response = await handle_live_ws_command({"command": "stop_tracking"})
    return {"ok": True, "liveData": response}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    state.clients.add(websocket)
    await send_json_safe(
        websocket,
        {
            "type": "connected",
            "message": "websocket ready",
            "status": build_status(),
        },
    )

    try:
        while True:
            raw_text = await websocket.receive_text()
            payload = normalize_ws_message(raw_text)
            response = await handle_ws_action(payload)
            await send_json_safe(websocket, response)
    except WebSocketDisconnect:
        pass
    finally:
        state.clients.discard(websocket)


@app.websocket("/ws/live")
async def websocket_live_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    state.live_clients.add(websocket)
    if state.camera_process is None or state.camera_process.returncode is not None:
        await start_camera(state.camera_fps)
    await send_json_safe(websocket, _build_live_data_payload())

    try:
        while True:
            raw_text = await websocket.receive_text()
            payload = normalize_ws_message(raw_text)
            response = await handle_live_ws_command(payload)
            await send_json_safe(websocket, response)
    except WebSocketDisconnect:
        pass
    finally:
        state.live_clients.discard(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=False)
