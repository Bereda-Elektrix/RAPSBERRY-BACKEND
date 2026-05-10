#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import time


def ensure_system_dist_packages() -> None:
    dist_packages = "/usr/lib/python3/dist-packages"
    if dist_packages not in sys.path and os.path.isdir(dist_packages):
        sys.path.append(dist_packages)


ensure_system_dist_packages()

from gpiozero import Device, OutputDevice, Servo

try:
    from .radar_distance import XM125Distance
except ImportError:
    from radar_distance import XM125Distance


SERVO_PIN = 22
OTHER_SERVO_PINS = ()
START_ANGLE = int(os.environ.get("RADAR_START_ANGLE", "100"))
END_ANGLE = int(os.environ.get("RADAR_END_ANGLE", "180"))
STEP_ANGLE = int(os.environ.get("RADAR_STEP_ANGLE", "10"))
STEP_DELAY_S = 0.01
START_MM = int(os.environ.get("RADAR_START_MM", "300"))
END_MM = int(os.environ.get("RADAR_END_MM", "3000"))
REST_TIME = 0.5
RADAR_RECOVERY_DELAY_S = 0.5
RADAR_MAX_RECOVERY_ATTEMPTS = 3
SERVO_SELFTEST_DELAY_S = 1.2
SERVO_SETTLE_DELAY_S = 0.15
RADAR_COMMAND_FILE = os.environ.get(
    "RADAR_COMMAND_FILE",
    "/tmp/raspberry_radar_commands.json",
)


def try_enable_pigpio():
    try:
        subprocess.run(
            ["pgrep", "-x", "pigpiod"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return False

    try:
        from gpiozero.pins.pigpio import PiGPIOFactory

        Device.pin_factory = PiGPIOFactory()
        return True
    except Exception as exc:
        print(f"[WARN] pigpio jest uruchomione, ale nie udalo sie go uzyc: {exc}")
        return False


def angle_to_value(angle):
    return max(-1.0, min(1.0, (angle - 90.0) / 90.0))


def clamp_angle(angle):
    return max(0, min(180, angle))


def move_servo(servo, angle):
    servo.value = angle_to_value(clamp_angle(angle))


def servo_selftest(servo):
    print("[INFO] Test serwa: 90 -> 40 -> 140 -> 90")
    for angle in (90, START_ANGLE, END_ANGLE, 90):
        move_servo(servo, angle)
        print(f"[INFO] Ustawiam serwo na {angle} stopni")
        time.sleep(SERVO_SELFTEST_DELAY_S)


def build_quiet_outputs(active_servo_pin):
    quiet_outputs = []
    for pin in OTHER_SERVO_PINS:
        if pin == active_servo_pin:
            continue
        quiet_outputs.append(OutputDevice(pin, active_high=True, initial_value=False))
    return quiet_outputs


def build_sweep_angles(start_angle, end_angle, step_angle):
    if step_angle <= 0:
        raise ValueError("Krok serwa musi byc dodatni.")
    if start_angle >= end_angle:
        raise ValueError("Kat poczatkowy musi byc mniejszy niz koncowy.")

    forward = list(range(start_angle, end_angle + 1, step_angle))
    backward = list(range(end_angle - step_angle, start_angle, -step_angle))
    return forward + backward


def setup_radar(radar, start_mm, end_mm):
    radar.distance_setup(start_mm=start_mm, end_mm=end_mm)
    print(f"Radar gotowy. Zakres: {start_mm} mm - {end_mm} mm")


def recover_radar(radar, start_mm, end_mm, error):
    print(f"[WARN] Blad radaru: {error}")
    for attempt in range(1, RADAR_MAX_RECOVERY_ATTEMPTS + 1):
        try:
            print(
                f"[INFO] Proba ponownej inicjalizacji radaru "
                f"{attempt}/{RADAR_MAX_RECOVERY_ATTEMPTS}..."
            )
            time.sleep(RADAR_RECOVERY_DELAY_S)
            setup_radar(radar, start_mm, end_mm)
            print("[INFO] Radar wznowil prace.")
            return True
        except RuntimeError as recovery_error:
            print(f"[WARN] Nieudana proba inicjalizacji: {recovery_error}")
    return False


class CommandState:
    def __init__(self, start_mm, end_mm):
        self.start_angle = START_ANGLE
        self.end_angle = END_ANGLE
        self.step_angle = STEP_ANGLE
        self.start_mm = start_mm
        self.end_mm = end_mm
        self.scan_active = True
        self.current_angle = 90
        self.last_command_mtime = 0.0
        self.sweep_angles = build_sweep_angles(
            self.start_angle,
            self.end_angle,
            self.step_angle,
        )

    def update_scan(self, start_angle, end_angle, step_angle, start_mm, end_mm):
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.step_angle = step_angle
        self.start_mm = start_mm
        self.end_mm = end_mm
        self.sweep_angles = build_sweep_angles(start_angle, end_angle, step_angle)


def apply_external_command(state, servo, radar):
    if not RADAR_COMMAND_FILE or not os.path.exists(RADAR_COMMAND_FILE):
        return

    try:
        mtime = os.path.getmtime(RADAR_COMMAND_FILE)
        if mtime <= state.last_command_mtime:
            return

        with open(RADAR_COMMAND_FILE, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"[WARN] Nie udalo sie odczytac komendy radaru: {exc}")
        return

    state.last_command_mtime = mtime
    command = str(payload.get("command", "")).strip().lower()

    if command == "set_radar_servo":
        angle = int(
            payload.get(
                "angle",
                payload.get("x_angle", payload.get("servo_angle", 90)),
            )
        )
        move_servo(servo, angle)
        state.current_angle = clamp_angle(angle)
        state.scan_active = False
        print(f"[INFO] Ustawiono serwo radaru na {angle} stopni")
        return

    if command == "configure_radar_scan":
        start_angle = int(payload.get("start_angle", state.start_angle))
        end_angle = int(payload.get("end_angle", state.end_angle))
        step_angle = int(payload.get("step_angle", state.step_angle))
        start_mm = int(payload.get("start_mm", state.start_mm))
        end_mm = int(payload.get("end_mm", state.end_mm))
        state.update_scan(start_angle, end_angle, step_angle, start_mm, end_mm)
        setup_radar(radar, start_mm, end_mm)
        print(
            "[INFO] Zmieniono konfiguracje skanu "
            f"{start_angle}-{end_angle} stopni, krok {step_angle}, "
            f"zakres {start_mm}-{end_mm} mm"
        )
        return

    if command == "start_radar_scan":
        state.scan_active = True
        print("[INFO] Wlaczono scan radaru z aplikacji")
        return

    if command == "stop_radar_scan":
        state.scan_active = False
        print("[INFO] Wylaczono scan radaru z aplikacji")
        return

    if command == "radar_selftest":
        state.scan_active = False
        servo_selftest(servo)
        print("[INFO] Uruchomiono self-test radaru z aplikacji")


def main():
    servo_pin = SERVO_PIN
    start_mm = START_MM
    end_mm = END_MM

    if len(sys.argv) == 2:
        servo_pin = int(sys.argv[1])
    elif len(sys.argv) == 4:
        servo_pin = int(sys.argv[1])
        start_mm = int(sys.argv[2])
        end_mm = int(sys.argv[3])
    elif len(sys.argv) not in (1, 2, 4):
        print("Uzycie: python3 radar_servo_scan.py [servo_gpio [start_mm end_mm]]")
        return 1

    print("Test: radar mierzy odleglosc, a serwo skanuje lewo-prawo.")
    print(f"GPIO serwa: {servo_pin}")
    print(f"Zakres radaru: {start_mm} mm - {end_mm} mm")
    print("Jesli radar korzysta u Ciebie z GPIO 22, nie podlaczaj tam serwa.")
    print("Zasilanie serwa podaj z osobnego 5V i polacz mase z GND Raspberry Pi.")
    print(f"Pozostale piny serw trzymane nisko: {', '.join(map(str, OTHER_SERVO_PINS))}")

    pigpio_active = try_enable_pigpio()
    print("Tryb PWM: pigpio" if pigpio_active else "Tryb PWM: gpiozero/lgpio")

    radar = XM125Distance()
    try:
        servo = Servo(
            servo_pin,
            min_pulse_width=0.0005,
            max_pulse_width=0.0025,
            frame_width=0.02,
        )
        quiet_outputs = build_quiet_outputs(servo_pin)
    except Exception as error:
        radar.close()
        print(f"[ERROR] Nie moge przejac GPIO {servo_pin}: {error}")
        return 1

    command_state = CommandState(start_mm, end_mm)

    try:
        servo_selftest(servo)
        setup_radar(radar, start_mm, end_mm)
        move_servo(servo, command_state.current_angle)
        print("Start skanowania. Oczekiwanie na komendy z backendu.")

        while True:
            apply_external_command(command_state, servo, radar)

            if not command_state.scan_active:
                try:
                    reading = radar.measure_once()
                except RuntimeError as error:
                    if recover_radar(
                        radar,
                        command_state.start_mm,
                        command_state.end_mm,
                        error,
                    ):
                        continue
                    print("[ERROR] Radar nie wznowil pracy. Koniec testu.")
                    return 1

                if reading["distance_mm"] is None:
                    print(
                        f"kat={command_state.current_angle:3d} stopni | brak obiektu | "
                        f"temp={reading['temperature_c']} C"
                    )
                else:
                    print(
                        f"kat={command_state.current_angle:3d} stopni | "
                        f"odleglosc={reading['distance_mm']} mm | "
                        f"sila={reading['strength']} | "
                        f"temp={reading['temperature_c']} C | "
                        f"peaks={reading['num_distances']}"
                    )
                time.sleep(0.08)
                continue

            for angle in command_state.sweep_angles:
                apply_external_command(command_state, servo, radar)
                if not command_state.scan_active:
                    break

                move_servo(servo, angle)
                command_state.current_angle = clamp_angle(angle)
                time.sleep(SERVO_SETTLE_DELAY_S)
                time.sleep(STEP_DELAY_S)

                try:
                    reading = radar.measure_once()
                except RuntimeError as error:
                    if recover_radar(
                        radar,
                        command_state.start_mm,
                        command_state.end_mm,
                        error,
                    ):
                        continue
                    print("[ERROR] Radar nie wznowil pracy. Koniec testu.")
                    return 1

                if reading["distance_mm"] is None:
                    print(
                        f"kat={angle:3d} stopni | brak obiektu | "
                        f"temp={reading['temperature_c']} C"
                    )
                else:
                    print(
                        f"kat={angle:3d} stopni | "
                        f"odleglosc={reading['distance_mm']} mm | "
                        f"sila={reading['strength']} | "
                        f"temp={reading['temperature_c']} C | "
                        f"peaks={reading['num_distances']}"
                    )
    except KeyboardInterrupt:
        print("\nKoniec testu.")
        return 0
    finally:
        servo.detach()
        for output in quiet_outputs:
            output.off()
            output.close()
        time.sleep(REST_TIME)
        radar.close()


if __name__ == "__main__":
    sys.exit(main())
