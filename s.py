#!/usr/bin/env python3
# Prosty podglad MI48 z automatycznym sledzeniem cieplego poruszajacego sie obiektu przez 2 serwa.
# Dodany tryb skanowania po utracie celu oraz predictive follow.

import atexit
import base64
import logging
import os
import json
import signal
import sys
import time


def ensure_system_dist_packages() -> None:
    dist_packages = "/usr/lib/python3/dist-packages"
    if dist_packages not in sys.path and os.path.isdir(dist_packages):
        sys.path.append(dist_packages)


ensure_system_dist_packages()

import numpy as np

if os.environ.get("XDG_SESSION_TYPE") == "wayland" and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

try:
    import cv2 as cv
except Exception:
    print("Please install OpenCV (or link existing installation)")
    sys.exit(1)

try:
    import lgpio
except Exception as exc:
    print(f"Brak biblioteki lgpio: {exc}")
    sys.exit(1)

from senxor.utils import RollingAverageFilter, connect_senxor, cv_filter, data_to_frame, remap
from senxor.mi48 import format_framestats, format_header


logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

mi48 = None
servo_ctrl = None
cleanup_done = False

SERVO_GORA_DOL = 17
SERVO_LEWO_PRAWO = 27
SERVO_FREQ = 50
PAN_DIRECTION = 1
TILT_DIRECTION = 1
WINDOW_NAME = "MI48 Thermal Motion Tracking + Servo"
STREAM_WS = os.environ.get("STREAM_WS") == "1"
CAMERA_COMMAND_FILE = os.environ.get("CAMERA_COMMAND_FILE", "/tmp/raspberry_camera_commands.json")
RADAR_COMMAND_FILE = os.environ.get("RADAR_COMMAND_FILE", "/tmp/raspberry_radar_commands.json")
STREAM_JPEG_QUALITY = int(os.environ.get("STREAM_JPEG_QUALITY", "55"))
STREAM_MAX_WIDTH = int(os.environ.get("STREAM_MAX_WIDTH", "320"))
STREAM_INCLUDE_THERMAL_FRAME = os.environ.get("STREAM_INCLUDE_THERMAL_FRAME", "0") == "1"
STREAM_THERMAL_STRIDE = max(1, int(os.environ.get("STREAM_THERMAL_STRIDE", "2")))
CAMERA_ROTATION_MODE = int(os.environ.get("CAMERA_ROTATION_MODE", "270"))
CAMERA_FLIP_VERTICAL = os.environ.get("CAMERA_FLIP_VERTICAL", "0") == "1"


def normalize_rotation_mode(value):
    return value if value in {0, 90, 180, 270} else 180

COLORMAPS = [
    ("turbo", cv.COLORMAP_TURBO),
    ("inferno", cv.COLORMAP_INFERNO),
    ("magma", cv.COLORMAP_MAGMA),
    ("plasma", cv.COLORMAP_PLASMA),
    ("jet", cv.COLORMAP_JET),
]


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


class PipelineState:
    def __init__(self):
        self.blur_ks = 1
        self.d = 3
        self.sigma_color = 20
        self.sigma_space = 20
        self.alpha = 0.25
        self.scale = 5
        self.show_hotspot = True
        self.show_target = True
        self.show_mask = False
        self.show_motion = False
        self.show_crosshair = True
        self.show_text = True
        self.rotation_mode = normalize_rotation_mode(CAMERA_ROTATION_MODE)
        self.flip_vertical = CAMERA_FLIP_VERTICAL
        self.colormap_index = 0
        self.temp_margin = 1.8
        self.min_area = 18
        self.frame_avg = None
        self.prev_frame_smooth = None
        self.dminav = RollingAverageFilter(N=15)
        self.dmaxav = RollingAverageFilter(N=15)
        self.clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

        self.servo_tracking = True
        self.scan_enabled = True
        self.stop_scan_on_detection = True
        self.resume_scan_when_target_lost = True
        self.scan_paused_for_target = False

        self.motion_threshold = 0.45
        self.motion_min_ratio = 0.08
        self.human_min_mean_delta = 1.0
        self.human_min_peak_delta = 1.8
        self.human_max_peak_temp = 40.0
        self.human_max_peak_over_mean = 6.5
        self.min_fill_ratio = 0.22
        self.max_fill_ratio = 1.0
        self.min_aspect_ratio = 0.20
        self.max_aspect_ratio = 1.35
        self.human_min_width = 5
        self.human_min_height = 8

        self.target_beta = 0.45
        self.last_target = None
        self.target_lock_enabled = True
        self.target_locked = False
        self.target_alarm_active = False
        self.target_missing_frames = 0
        self.target_missing_limit = 8
        self.target_lock_margin = 10
        self.fusion_tracking_enabled = True
        self.radar_follow_deadband = 2
        self.radar_follow_interval = 0.18
        self.last_radar_follow_angle = None
        self.last_radar_follow_ts = 0.0
        self.last_radar_scan_command = None

        self.last_target_centroid = None
        self.target_velocity = (0.0, 0.0)
        self.predictive_follow_frames = 0
        self.predictive_gain = 1.4

        self.validation_reason = "brak"
        self.last_command_mtime = 0.0

    @property
    def colormap_name(self):
        return COLORMAPS[self.colormap_index][0]

    @property
    def colormap_value(self):
        return COLORMAPS[self.colormap_index][1]

    def next_colormap(self):
        self.colormap_index = (self.colormap_index + 1) % len(COLORMAPS)

    def build_filter_params(self):
        return {
            "blur_ks": self.blur_ks,
            "d": self.d,
            "sigmaColor": self.sigma_color,
            "sigmaSpace": self.sigma_space,
        }


class AutoServoControl:
    def __init__(
        self,
        pin_ud=SERVO_GORA_DOL,
        pin_lr=SERVO_LEWO_PRAWO,
        min_angle=30,
        max_angle=150,
        step=1,
        update_interval=0.20,
    ):
        self.pin_ud = pin_ud
        self.pin_lr = pin_lr
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.step = step
        self.update_interval = update_interval
        self.last_update = 0.0
        self.enabled = True
        self.cleaned = False

        self.scan_last_update = 0.0
        self.scan_interval = 0.18
        self.scan_pan_step = 2
        self.scan_tilt_step = 2
        self.scan_pan_dir = 1
        self.scan_tilt_dir = 1
        self.scan_tilt_min = max(self.min_angle, 65)
        self.scan_tilt_max = min(self.max_angle, 115)

        self.chip = lgpio.gpiochip_open(0)
        self.claimed_ud = False
        self.claimed_lr = False

        lgpio.gpio_claim_output(self.chip, self.pin_ud)
        self.claimed_ud = True
        lgpio.gpio_claim_output(self.chip, self.pin_lr)
        self.claimed_lr = True

        self.servo_ud_angle = 90
        self.servo_lr_angle = 90
        self._center()

    def _clamp(self, angle):
        return max(self.min_angle, min(self.max_angle, int(angle)))

    def _clamp_scan_tilt(self, angle):
        return max(self.scan_tilt_min, min(self.scan_tilt_max, int(angle)))

    def _angle_to_duty(self, angle):
        angle = self._clamp(angle)
        return 2.5 + (angle / 180.0) * 10.0

    def _apply(self):
        if not self.enabled:
            return
        duty_ud = self._angle_to_duty(self.servo_ud_angle)
        duty_lr = self._angle_to_duty(self.servo_lr_angle)
        lgpio.tx_pwm(self.chip, self.pin_ud, SERVO_FREQ, duty_ud)
        lgpio.tx_pwm(self.chip, self.pin_lr, SERVO_FREQ, duty_lr)

    def _center(self):
        self.servo_ud_angle = 90
        self.servo_lr_angle = 90
        self.enabled = True
        self._apply()
        logger.info("Serwa ustawione na centrum (90, 90).")

    def center(self):
        self._center()

    def set_angles(self, x_angle, y_angle):
        self.servo_lr_angle = self._clamp(x_angle)
        self.servo_ud_angle = self._clamp(y_angle)
        self.enabled = True
        self._apply()

    def detach(self):
        self.enabled = False
        try:
            lgpio.tx_pwm(self.chip, self.pin_ud, 0, 0)
        except Exception:
            pass
        try:
            lgpio.tx_pwm(self.chip, self.pin_lr, 0, 0)
        except Exception:
            pass

    def update_from_target(self, cx, cy, frame_w, frame_h):
        if not self.enabled:
            return

        now = time.monotonic()
        if now - self.last_update < self.update_interval:
            return
        self.last_update = now

        center_x = frame_w / 2.0
        center_y = frame_h / 2.0
        dx = cx - center_x
        dy = cy - center_y

        thr_x = frame_w * 0.07
        thr_y = frame_h * 0.07

        pan_delta = 0
        tilt_delta = 0

        if abs(dx) > thr_x:
            gain_x = int(np.clip(abs(dx) / (frame_w * 0.12), 1, 4))
            pan_delta = gain_x * PAN_DIRECTION if dx > 0 else -gain_x * PAN_DIRECTION

        if abs(dy) > thr_y:
            gain_y = int(np.clip(abs(dy) / (frame_h * 0.12), 1, 4))
            tilt_delta = gain_y * TILT_DIRECTION if dy > 0 else -gain_y * TILT_DIRECTION

        new_lr_angle = self._clamp(self.servo_lr_angle + pan_delta)
        new_ud_angle = self._clamp(self.servo_ud_angle + tilt_delta)

        changed = new_lr_angle != self.servo_lr_angle or new_ud_angle != self.servo_ud_angle

        self.servo_lr_angle = new_lr_angle
        self.servo_ud_angle = new_ud_angle

        if changed:
            self.enabled = True
            self._apply()

    def predictive_step(self, vx, vy, gain=1.0):
        if not self.enabled:
            return

        now = time.monotonic()
        if now - self.scan_last_update < self.scan_interval:
            return
        self.scan_last_update = now

        pan_delta = 0
        tilt_delta = 0

        if abs(vx) > 0.2:
            pan_delta = int(np.clip(abs(vx) * gain, 1, 4))
            if vx < 0:
                pan_delta = -pan_delta
            pan_delta *= PAN_DIRECTION

        if abs(vy) > 0.2:
            tilt_delta = int(np.clip(abs(vy) * gain, 1, 3))
            if vy < 0:
                tilt_delta = -tilt_delta
            tilt_delta *= TILT_DIRECTION

        if pan_delta == 0 and tilt_delta == 0:
            return

        self.servo_lr_angle = self._clamp(self.servo_lr_angle + pan_delta)
        self.servo_ud_angle = self._clamp(self.servo_ud_angle + tilt_delta)
        self._apply()

    def scan_step(self):
        if not self.enabled:
            return

        now = time.monotonic()
        if now - self.scan_last_update < self.scan_interval:
            return
        self.scan_last_update = now

        hit_edge = False

        new_lr = self.servo_lr_angle + self.scan_pan_dir * self.scan_pan_step
        if new_lr >= self.max_angle:
            new_lr = self.max_angle
            self.scan_pan_dir = -1
            hit_edge = True
        elif new_lr <= self.min_angle:
            new_lr = self.min_angle
            self.scan_pan_dir = 1
            hit_edge = True

        self.servo_lr_angle = self._clamp(new_lr)

        if hit_edge:
            new_ud = self.servo_ud_angle + self.scan_tilt_dir * self.scan_tilt_step
            if new_ud >= self.scan_tilt_max:
                new_ud = self.scan_tilt_max
                self.scan_tilt_dir = -1
            elif new_ud <= self.scan_tilt_min:
                new_ud = self.scan_tilt_min
                self.scan_tilt_dir = 1
            self.servo_ud_angle = self._clamp_scan_tilt(new_ud)

        self._apply()

    def cleanup(self):
        if self.cleaned:
            return
        self.cleaned = True

        self.detach()

        try:
            if self.claimed_ud:
                lgpio.gpio_free(self.chip, self.pin_ud)
        except Exception:
            pass

        try:
            if self.claimed_lr:
                lgpio.gpio_free(self.chip, self.pin_lr)
        except Exception:
            pass

        try:
            lgpio.gpiochip_close(self.chip)
        except Exception:
            pass

        logger.info("lgpio: serwa wylaczone, GPIO zwolnione, gpiochip zamkniety.")


def cleanup():
    global mi48, servo_ctrl, cleanup_done
    if cleanup_done:
        return
    cleanup_done = True
    try:
        if mi48 is not None:
            mi48.stop()
    except Exception:
        pass
    try:
        cv.destroyAllWindows()
    except Exception:
        pass
    try:
        if servo_ctrl is not None:
            servo_ctrl.cleanup()
    except Exception:
        pass


def cleanup_and_exit(code=0):
    cleanup()
    raise SystemExit(code)


def _atomic_write_json(path, payload):
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    os.replace(temp_path, path)


def write_radar_command(command, **extra):
    if not RADAR_COMMAND_FILE:
        return
    payload = {
        "command": command,
        "timestamp": time.time(),
        **extra,
    }
    try:
        _atomic_write_json(RADAR_COMMAND_FILE, payload)
    except Exception as exc:
        logger.warning("Nie udalo sie zapisac komendy radaru: %s", exc)


def emit_stream_event(event_type, **payload):
    if not STREAM_WS:
        return
    print(
        json.dumps(
            {
                "type": event_type,
                "timestamp": time.time(),
                **payload,
            },
            default=json_default,
            separators=(",", ":"),
        ),
        flush=True,
    )


def signal_handler(sig, frame):
    logger.info("Exiting due to signal %s", sig)
    cleanup()
    os._exit(0)


def apply_temporal_smoothing(frame, state):
    if state.frame_avg is None:
        state.frame_avg = frame.astype(np.float32)
    else:
        state.frame_avg = (
            state.alpha * frame.astype(np.float32)
            + (1.0 - state.alpha) * state.frame_avg
        )
    return state.frame_avg.astype(np.float32)


def detect_motion_mask(frame_smooth, state):
    if state.prev_frame_smooth is None:
        state.prev_frame_smooth = frame_smooth.copy()
        return np.zeros(frame_smooth.shape, dtype=np.uint8)

    diff = cv.absdiff(frame_smooth.astype(np.float32), state.prev_frame_smooth.astype(np.float32))
    state.prev_frame_smooth = frame_smooth.copy()

    motion_mask = (diff >= state.motion_threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    motion_mask = cv.morphologyEx(motion_mask, cv.MORPH_OPEN, kernel)
    motion_mask = cv.morphologyEx(motion_mask, cv.MORPH_DILATE, kernel)
    return motion_mask


def detect_thermal_target(frame_smooth, min_temp, max_temp, state):
    dynamic_threshold = min(max_temp - 0.8, min_temp + state.temp_margin)
    thermal_mask = (frame_smooth >= dynamic_threshold).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    thermal_mask = cv.morphologyEx(thermal_mask, cv.MORPH_OPEN, kernel)
    thermal_mask = cv.morphologyEx(thermal_mask, cv.MORPH_CLOSE, kernel)
    return thermal_mask


def detect_best_human_target(frame_smooth, thermal_mask, motion_mask, min_temp, state):
    combined_mask = cv.bitwise_and(thermal_mask, motion_mask)

    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_DILATE, kernel)

    contours, _ = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    best = None
    overheated_rejections = 0

    for contour in contours:
        area = cv.contourArea(contour)
        if area < state.min_area:
            continue

        x, y, w, h = cv.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        if w < state.human_min_width or h < state.human_min_height:
            continue
        if y <= 1 and h <= max(10, frame_smooth.shape[0] // 5):
            continue

        aspect_ratio = w / float(h)
        if aspect_ratio < state.min_aspect_ratio or aspect_ratio > state.max_aspect_ratio:
            continue

        roi_temp = frame_smooth[y:y + h, x:x + w]
        roi_motion = motion_mask[y:y + h, x:x + w]

        mean_temp = float(roi_temp.mean())
        peak_temp = float(roi_temp.max())
        mean_delta = mean_temp - float(min_temp)
        peak_delta = peak_temp - float(min_temp)

        if mean_delta < state.human_min_mean_delta:
            continue
        if peak_delta < state.human_min_peak_delta:
            continue
        if peak_temp > state.human_max_peak_temp:
            overheated_rejections += 1
            continue
        if peak_temp - mean_temp > state.human_max_peak_over_mean:
            continue

        fill_ratio = float(area) / max(1.0, float(w * h))
        if fill_ratio < state.min_fill_ratio or fill_ratio > state.max_fill_ratio:
            continue

        motion_ratio = float(np.count_nonzero(roi_motion)) / max(1.0, float(w * h))
        if motion_ratio < state.motion_min_ratio:
            continue

        motion_score = float(np.mean(roi_motion > 0))
        score = 0.8 * mean_temp + 0.18 * area + 16.0 * motion_score + 6.0 * fill_ratio

        moments = cv.moments(contour)
        if moments["m00"] == 0:
            cx = x + w // 2
            cy = y + h // 2
        else:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

        candidate = {
            "score": score,
            "bbox": (x, y, w, h),
            "centroid": (cx, cy),
            "area": area,
            "fill_ratio": fill_ratio,
            "motion_ratio": motion_ratio,
            "mean_temp": mean_temp,
            "peak_temp": peak_temp,
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    if best is None:
        if overheated_rejections:
            return None, combined_mask, "odrzucono zbyt goracy obiekt"
        return None, combined_mask, "brak cieplego ruchomego celu"

    return best, combined_mask, "cieply ruchomy obiekt"


def detect_locked_target(frame_smooth, thermal_mask, motion_mask, min_temp, state):
    if not state.target_lock_enabled or state.last_target is None:
        return None, None, None

    x, y, w, h = state.last_target["bbox"]
    margin = state.target_lock_margin
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame_smooth.shape[1], x + w + margin)
    y2 = min(frame_smooth.shape[0], y + h + margin)

    if x2 <= x1 or y2 <= y1:
        return None, None, None

    roi_frame = frame_smooth[y1:y2, x1:x2]
    roi_thermal = thermal_mask[y1:y2, x1:x2]
    roi_motion = motion_mask[y1:y2, x1:x2]

    target, combined_mask, reason = detect_best_human_target(
        roi_frame,
        roi_thermal,
        roi_motion,
        min_temp,
        state,
    )
    if target is None:
        return None, combined_mask, reason

    tx, ty, tw, th = target["bbox"]
    cx, cy = target["centroid"]
    locked_target = {
        **target,
        "bbox": (tx + x1, ty + y1, tw, th),
        "centroid": (cx + x1, cy + y1),
    }
    return locked_target, combined_mask, "sledzenie zablokowanego celu"


def update_tracked_target(current_target, state):
    if current_target is None:
        state.target_missing_frames += 1
        if state.target_missing_frames > state.target_missing_limit:
            state.last_target = None
            state.target_locked = False
            if state.scan_paused_for_target:
                state.scan_paused_for_target = False
            if state.target_alarm_active:
                emit_stream_event(
                    "target_alarm",
                    event="lost",
                    message="cel zniknal z pola widzenia",
                )
                state.target_alarm_active = False
        return state.last_target

    state.target_missing_frames = 0
    state.target_locked = state.target_lock_enabled
    if not state.target_alarm_active:
        emit_stream_event(
            "target_alarm",
            event="detected",
            message="wykryto cel",
        )
        state.target_alarm_active = True
        if state.stop_scan_on_detection and not state.scan_paused_for_target:
            state.scan_paused_for_target = True

    x, y, w, h = current_target["bbox"]
    cx, cy = current_target["centroid"]

    if state.last_target is None:
        state.last_target = {
            **current_target,
            "bbox": (x, y, w, h),
            "centroid": (cx, cy),
        }
        return state.last_target

    lx, ly, lw, lh = state.last_target["bbox"]
    lcx, lcy = state.last_target["centroid"]
    beta = state.target_beta

    smoothed = {
        **current_target,
        "bbox": (
            int(beta * x + (1.0 - beta) * lx),
            int(beta * y + (1.0 - beta) * ly),
            int(beta * w + (1.0 - beta) * lw),
            int(beta * h + (1.0 - beta) * lh),
        ),
        "centroid": (
            int(beta * cx + (1.0 - beta) * lcx),
            int(beta * cy + (1.0 - beta) * lcy),
        ),
    }

    state.last_target = smoothed
    return state.last_target


def update_target_motion_model(tracked_target, state):
    if tracked_target is None:
        return

    cx, cy = tracked_target["centroid"]

    if state.last_target_centroid is not None:
        lx, ly = state.last_target_centroid
        vx = cx - lx
        vy = cy - ly

        old_vx, old_vy = state.target_velocity
        beta = 0.5
        state.target_velocity = (
            beta * vx + (1.0 - beta) * old_vx,
            beta * vy + (1.0 - beta) * old_vy,
        )

    state.last_target_centroid = (cx, cy)


def draw_overlay(image, state, stats, hotspot_xy, target):
    h, w = image.shape[:2]

    if state.show_crosshair:
        cx = w // 2
        cy = h // 2
        cv.line(image, (cx - 12, cy), (cx + 12, cy), (255, 255, 255), 1, cv.LINE_AA)
        cv.line(image, (cx, cy - 12), (cx, cy + 12), (255, 255, 255), 1, cv.LINE_AA)

    if state.show_hotspot and hotspot_xy is not None:
        hx, hy = hotspot_xy
        cv.circle(image, (hx, hy), 8, (255, 255, 255), 1, cv.LINE_AA)
        cv.circle(image, (hx, hy), 2, (255, 255, 255), -1, cv.LINE_AA)

    if state.show_target and target is not None:
        x1, y1, x2, y2 = target["bbox_disp"]
        cx, cy = target["centroid_disp"]
        cv.rectangle(image, (x1, y1), (x2, y2), (80, 255, 80), 2, cv.LINE_AA)
        cv.circle(image, (cx, cy), 4, (80, 255, 80), -1, cv.LINE_AA)

    if state.show_text:
        target_text = "Target: no"
        if target is not None:
            target_text = (
                f"Target: yes   Mean {target['mean_temp']:.1f} C   "
                f"Peak {target['peak_temp']:.1f} C   Area {target['area']:.0f}   "
                f"Motion {target['motion_ratio']:.2f}"
            )

        servo_text = "Servo: off"
        if servo_ctrl is not None:
            vx, vy = state.target_velocity
            servo_text = (
                f"Servo: {'tracking' if state.servo_tracking else 'hold'}   "
                f"Lock: {'on' if state.target_locked else 'search'}   "
                f"Fusion: {'on' if state.fusion_tracking_enabled else 'off'}   "
                f"Scan: {'on' if state.scan_enabled else 'off'}   "
                f"UD {servo_ctrl.servo_ud_angle}   LR {servo_ctrl.servo_lr_angle}   "
                f"Vx {vx:.2f} Vy {vy:.2f}"
            )

        lines = [
            f"Min {stats['min_temp']:.1f} C   Max {stats['max_temp']:.1f} C   Ctr {stats['center_temp']:.1f} C",
            f"Hot {stats['hotspot_temp']:.1f} C   FPS {stats['fps']}   Map {state.colormap_name}",
            target_text,
            f"Thermal-motion gate: {state.validation_reason}",
            servo_text,
            "Q quit  M map  H hotspot  B target  N mask  J motion  X crosshair  T text  SPACE detach",
            "C center servo  S tracking  K scan  R rotate  V flip  +/- alpha  [ ] scale  ,/. temp  ;/' area",
        ]
        y = 22
        for line in lines:
            cv.putText(
                image,
                line,
                (10, y),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
            y += 22


def transform_for_display(image, state):
    if state.rotation_mode == 90:
        image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    elif state.rotation_mode == 180:
        image = cv.rotate(image, cv.ROTATE_180)
    elif state.rotation_mode == 270:
        image = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)

    if state.flip_vertical:
        image = cv.flip(image, 0)
    return image


def build_tracking_gray(filt_uint8, state):
    if state.scale == 1:
        return filt_uint8.copy()
    return cv.resize(
        filt_uint8,
        (filt_uint8.shape[1] * state.scale, filt_uint8.shape[0] * state.scale),
        interpolation=cv.INTER_CUBIC,
    )


def build_tracking_mask(mask, state):
    if state.scale == 1:
        return mask.copy()
    return cv.resize(
        mask,
        (mask.shape[1] * state.scale, mask.shape[0] * state.scale),
        interpolation=cv.INTER_NEAREST,
    )


def map_point_for_display(x, y, base_w, base_h, state):
    if state.rotation_mode == 90:
        rx = base_h - 1 - y
        ry = x
        out_w, out_h = base_h, base_w
    elif state.rotation_mode == 180:
        rx = base_w - 1 - x
        ry = base_h - 1 - y
        out_w, out_h = base_w, base_h
    elif state.rotation_mode == 270:
        rx = y
        ry = base_w - 1 - x
        out_w, out_h = base_h, base_w
    else:
        rx = x
        ry = y
        out_w, out_h = base_w, base_h

    if state.flip_vertical:
        ry = out_h - 1 - ry

    return int(rx), int(ry)


def map_bbox_for_display(x, y, w, h, base_w, base_h, state):
    corners = [
        map_point_for_display(x, y, base_w, base_h, state),
        map_point_for_display(x + w - 1, y, base_w, base_h, state),
        map_point_for_display(x, y + h - 1, base_w, base_h, state),
        map_point_for_display(x + w - 1, y + h - 1, base_w, base_h, state),
    ]
    xs = [pt[0] for pt in corners]
    ys = [pt[1] for pt in corners]
    return min(xs), min(ys), max(xs), max(ys)


def map_raw_point_to_display(x, y, raw_w, raw_h, state):
    return map_point_for_display(
        int(x * state.scale),
        int(y * state.scale),
        raw_w * state.scale,
        raw_h * state.scale,
        state,
    )


def map_raw_bbox_to_display(x, y, w, h, raw_w, raw_h, state):
    return map_bbox_for_display(
        int(x * state.scale),
        int(y * state.scale),
        max(1, int(w * state.scale)),
        max(1, int(h * state.scale)),
        raw_w * state.scale,
        raw_h * state.scale,
        state,
    )


def handle_key(key, state):
    if key == ord("q"):
        return False
    if key == ord("m"):
        state.next_colormap()
    elif key == ord("h"):
        state.show_hotspot = not state.show_hotspot
    elif key == ord("b"):
        state.show_target = not state.show_target
    elif key == ord("n"):
        state.show_mask = not state.show_mask
    elif key == ord("j"):
        state.show_motion = not state.show_motion
    elif key == ord("x"):
        state.show_crosshair = not state.show_crosshair
    elif key == ord("t"):
        state.show_text = not state.show_text
    elif key == ord("c"):
        if servo_ctrl is not None:
            servo_ctrl.center()
    elif key == ord("s"):
        state.servo_tracking = not state.servo_tracking
    elif key == ord("k"):
        state.scan_enabled = not state.scan_enabled
    elif key == ord("r"):
        rotations = [0, 90, 180, 270]
        idx = rotations.index(state.rotation_mode)
        state.rotation_mode = rotations[(idx + 1) % len(rotations)]
    elif key == ord("v"):
        state.flip_vertical = not state.flip_vertical
    elif key in (ord("+"), ord("=")):
        state.alpha = min(0.90, state.alpha + 0.05)
    elif key == ord("-"):
        state.alpha = max(0.05, state.alpha - 0.05)
    elif key == ord("]"):
        state.scale = min(10, state.scale + 1)
    elif key == ord("["):
        state.scale = max(1, state.scale - 1)
    elif key == ord("."):
        state.temp_margin = min(8.0, state.temp_margin + 0.1)
    elif key == ord(","):
        state.temp_margin = max(0.3, state.temp_margin - 0.1)
    elif key == ord("'"):
        state.min_area = min(500, state.min_area + 2)
    elif key == ord(";"):
        state.min_area = max(2, state.min_area - 2)
    elif key == ord(" "):
        if servo_ctrl is not None:
            servo_ctrl.detach()
    return True


def apply_external_command(state):
    if not CAMERA_COMMAND_FILE or not os.path.exists(CAMERA_COMMAND_FILE):
        return

    try:
        mtime = os.path.getmtime(CAMERA_COMMAND_FILE)
        if mtime <= state.last_command_mtime:
            return

        with open(CAMERA_COMMAND_FILE, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logger.warning("Nie udalo sie odczytac komendy kamery: %s", exc)
        return

    state.last_command_mtime = mtime
    command = str(payload.get("command", "")).strip().lower()

    if command == "set_camera_servo":
        if servo_ctrl is None:
            return
        x_angle = int(payload.get("x_angle", servo_ctrl.servo_lr_angle))
        y_angle = int(payload.get("y_angle", servo_ctrl.servo_ud_angle))
        servo_ctrl.set_angles(x_angle, y_angle)
        state.servo_tracking = False
        logger.info("Ustawiono serwa recznie: x=%s y=%s", x_angle, y_angle)
    elif command == "start_tracking":
        state.servo_tracking = True
        logger.info("Wlaczono tracking z aplikacji.")
    elif command == "stop_tracking":
        state.servo_tracking = False
        logger.info("Wylaczono tracking z aplikacji.")
    elif command == "set_camera_orientation":
        state.rotation_mode = normalize_rotation_mode(int(payload.get("rotation", state.rotation_mode)))
        state.flip_vertical = bool(payload.get("flip_vertical", state.flip_vertical))
        logger.info(
            "Ustawiono orientacje kamery: rotation=%s flip_vertical=%s",
            state.rotation_mode,
            state.flip_vertical,
        )
    elif command == "start_fusion":
        state.fusion_tracking_enabled = True
        logger.info("Wlaczono fusion camera-radar.")
    elif command == "stop_fusion":
        state.fusion_tracking_enabled = False
        state.last_radar_follow_angle = None
        state.last_radar_scan_command = None
        write_radar_command("start_radar_scan")
        logger.info("Wylaczono fusion camera-radar.")


def build_stream_image(vis):
    if STREAM_MAX_WIDTH > 0 and vis.shape[1] > STREAM_MAX_WIDTH:
        scale = STREAM_MAX_WIDTH / float(vis.shape[1])
        new_size = (
            max(1, int(vis.shape[1] * scale)),
            max(1, int(vis.shape[0] * scale)),
        )
        return cv.resize(vis, new_size, interpolation=cv.INTER_AREA)
    return vis


def build_thermal_payload(frame_smooth, avg_temp):
    thermal_payload = {
        "width": int(frame_smooth.shape[1]),
        "height": int(frame_smooth.shape[0]),
        "min": float(np.min(frame_smooth)),
        "max": float(np.max(frame_smooth)),
        "avg": avg_temp,
    }

    if STREAM_INCLUDE_THERMAL_FRAME:
        reduced = frame_smooth[::STREAM_THERMAL_STRIDE, ::STREAM_THERMAL_STRIDE]
        thermal_payload.update(
            {
                "width": int(reduced.shape[1]),
                "height": int(reduced.shape[0]),
                "frame": np.round(reduced.reshape(-1), 2).tolist(),
                "stride": STREAM_THERMAL_STRIDE,
            }
        )
    else:
        thermal_payload["frame"] = []

    return thermal_payload


def sync_radar_to_camera(state, force_resume=False):
    if not state.fusion_tracking_enabled:
        return
    if servo_ctrl is None:
        return

    if state.scan_paused_for_target:
        if state.last_radar_scan_command != "stop_radar_scan":
            write_radar_command("stop_radar_scan")
            state.last_radar_scan_command = "stop_radar_scan"
        state.last_radar_follow_angle = None
        return

    now = time.monotonic()
    should_follow = (
        state.servo_tracking
        and (
            state.target_locked
            or state.target_missing_frames <= state.predictive_follow_frames
        )
    )

    if force_resume or not should_follow:
        if state.last_radar_scan_command != "start_radar_scan":
            write_radar_command("start_radar_scan")
            state.last_radar_scan_command = "start_radar_scan"
        state.last_radar_follow_angle = None
        return

    desired_angle = int(servo_ctrl.servo_lr_angle)
    if state.last_radar_scan_command != "stop_radar_scan":
        write_radar_command("stop_radar_scan")
        state.last_radar_scan_command = "stop_radar_scan"

    angle_changed = (
        state.last_radar_follow_angle is None
        or abs(desired_angle - state.last_radar_follow_angle) >= state.radar_follow_deadband
    )
    if not angle_changed:
        return
    if now - state.last_radar_follow_ts < state.radar_follow_interval:
        return

    write_radar_command("set_radar_servo", angle=desired_angle)
    state.last_radar_follow_angle = desired_angle
    state.last_radar_follow_ts = now


def main():
    global mi48, servo_ctrl

    atexit.register(cleanup)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal_handler)

    state = PipelineState()

    mi48, connected_port, port_names = connect_senxor()
    if mi48 is None:
        if port_names:
            logger.error(
                "Nie udalo sie polaczyc z kamera MI48. Wykryte porty: %s. "
                "Port moze byc zajety albo kamera nie odpowiada.",
                ", ".join(port_names),
            )
        else:
            logger.error(
                "Nie wykryto kamery MI48. Sprawdz polaczenie USB i zasilanie."
            )
        cleanup_and_exit(1)

    logger.info("Camera info:")
    logger.info(mi48.camera_info)

    if len(sys.argv) == 2:
        stream_fps = int(sys.argv[1])
    else:
        stream_fps = 15

    mi48.set_fps(stream_fps)
    mi48.disable_filter(f1=True, f2=True, f3=True)
    mi48.set_filter_1(85)
    mi48.enable_filter(f1=True, f2=False, f3=False, f3_ks_5=False)
    mi48.set_offset_corr(0.0)
    mi48.set_sens_factor(100)
    mi48.get_sens_factor()
    mi48.start(stream=True, with_header=True)

    servo_ctrl = AutoServoControl()
    if not STREAM_WS:
        cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)

    try:
        while True:
            apply_external_command(state)
            data, header = mi48.read()
            if data is None:
                logger.critical("NONE data received instead of GFRA")
                cleanup_and_exit(1)

            p2 = np.percentile(data, 2)
            p98 = np.percentile(data, 98)

            min_temp = state.dminav(p2)
            max_temp = state.dmaxav(p98)

            frame = data_to_frame(data, (80, 62), hflip=False)
            frame = np.clip(frame, min_temp, max_temp)
            frame_smooth = apply_temporal_smoothing(frame, state)

            ch = frame_smooth.shape[0] // 2
            cw = frame_smooth.shape[1] // 2
            center_temp = float(frame_smooth[ch, cw])

            hot_index = np.unravel_index(np.argmax(frame_smooth), frame_smooth.shape)
            hot_y, hot_x = hot_index
            hotspot_temp = float(frame_smooth[hot_y, hot_x])
            avg_temp = float(np.mean(frame_smooth))

            thermal_mask = detect_thermal_target(frame_smooth, min_temp, max_temp, state)
            motion_mask = detect_motion_mask(frame_smooth, state)
            target_raw = None
            combined_mask = None
            validation_reason = "brak cieplego ruchomego celu"

            if state.target_locked and state.last_target is not None:
                target_raw, combined_mask, validation_reason = detect_locked_target(
                    frame_smooth,
                    thermal_mask,
                    motion_mask,
                    min_temp,
                    state,
                )
            elif target_raw is None:
                target_raw, combined_mask, validation_reason = detect_best_human_target(
                    frame_smooth,
                    thermal_mask,
                    motion_mask,
                    min_temp,
                    state,
                )

            state.validation_reason = validation_reason
            tracked_target = update_tracked_target(target_raw, state)
            update_target_motion_model(tracked_target, state)

            img_u8 = remap(frame_smooth)
            img_u8 = state.clahe.apply(img_u8)
            filt_uint8 = cv_filter(
                img_u8,
                state.build_filter_params(),
                use_median=True,
                use_bilat=True,
                use_nlm=False,
            )
            filt_uint8 = cv.GaussianBlur(filt_uint8, (3, 3), 0)

            tracking_gray = build_tracking_gray(filt_uint8, state)

            if state.show_mask:
                vis_base = build_tracking_mask(combined_mask, state)
                vis_base = cv.cvtColor(vis_base, cv.COLOR_GRAY2BGR)
            elif state.show_motion:
                vis_base = build_tracking_mask(motion_mask, state)
                vis_base = cv.cvtColor(vis_base, cv.COLOR_GRAY2BGR)
            else:
                vis_base = cv.applyColorMap(tracking_gray, state.colormap_value)

            vis = transform_for_display(vis_base, state)
            hotspot_xy = map_raw_point_to_display(hot_x, hot_y, 80, 62, state)

            target = None
            if tracked_target is not None:
                tx, ty, tw, th = tracked_target["bbox"]
                target = {
                    **tracked_target,
                    "bbox_disp": map_raw_bbox_to_display(tx, ty, tw, th, 80, 62, state),
                    "centroid_disp": map_raw_point_to_display(
                        tracked_target["centroid"][0],
                        tracked_target["centroid"][1],
                        80,
                        62,
                        state,
                    ),
                }

            if (
                tracked_target is not None
                and state.servo_tracking
                and not state.scan_paused_for_target
            ):
                servo_ctrl.update_from_target(
                    tracked_target["centroid"][0] * state.scale,
                    tracked_target["centroid"][1] * state.scale,
                    tracking_gray.shape[1],
                    tracking_gray.shape[0],
                )
            elif (
                tracked_target is None
                and state.servo_tracking
                and not state.scan_paused_for_target
            ):
                if (
                    state.predictive_follow_frames > 0
                    and state.target_missing_frames <= state.predictive_follow_frames
                ):
                    vx, vy = state.target_velocity
                    servo_ctrl.predictive_step(vx, vy, gain=state.predictive_gain)
                elif state.scan_enabled:
                    servo_ctrl.scan_step()

            sync_radar_to_camera(
                state,
                force_resume=tracked_target is None
                and state.target_missing_frames > state.predictive_follow_frames,
            )

            stats = {
                "min_temp": min_temp,
                "max_temp": max_temp,
                "center_temp": center_temp,
                "hotspot_temp": hotspot_temp,
                "fps": stream_fps,
            }

            draw_overlay(
                vis,
                state,
                stats,
                hotspot_xy,
                target,
            )

            if STREAM_WS:
                stream_vis = build_stream_image(vis)
                ok, encoded = cv.imencode(
                    ".jpg",
                    stream_vis,
                    [int(cv.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY],
                )
                if ok:
                    print(
                        json.dumps(
                            {
                                "type": "camera_frame",
                                "format": "jpeg",
                                "image": base64.b64encode(encoded).decode("ascii"),
                                "thermal": build_thermal_payload(frame_smooth, avg_temp),
                                "stats": stats,
                                "target": target,
                                "tracking_enabled": state.servo_tracking,
                                "fusion": {
                                    "enabled": state.fusion_tracking_enabled,
                                    "target_locked": state.target_locked,
                                    "radar_following": (
                                        state.fusion_tracking_enabled
                                        and state.last_radar_scan_command == "stop_radar_scan"
                                    ),
                                },
                                "servo": {
                                    "x_angle": servo_ctrl.servo_lr_angle if servo_ctrl is not None else 90,
                                    "y_angle": servo_ctrl.servo_ud_angle if servo_ctrl is not None else 90,
                                },
                                "timestamp": time.time(),
                            },
                            default=json_default,
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )

            if header is not None:
                logger.debug("  ".join([format_header(header), format_framestats(data)]))
            else:
                logger.debug(format_framestats(data))

            if not STREAM_WS:
                cv.imshow(WINDOW_NAME, vis)
                key = cv.waitKey(1) & 0xFF
                if not handle_key(key, state):
                    break

                try:
                    if cv.getWindowProperty(WINDOW_NAME, cv.WND_PROP_VISIBLE) < 1:
                        break
                except cv.error:
                    break
    finally:
        cleanup()


if __name__ == "__main__":
    main()
