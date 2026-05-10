import sys
import time
import subprocess
from pathlib import Path

from smbus2 import SMBus, i2c_msg


I2C_BUS = 1
I2C_ADDR = 0x52

REG_PROTOCOL_STATUS = 0x0001
REG_DETECTOR_STATUS = 0x0003
REG_RESULT = 0x0010
REG_PEAK0_DISTANCE = 0x0011
REG_PEAK0_STRENGTH = 0x001B
REG_START = 0x0040
REG_END = 0x0041
REG_COMMAND = 0x0100

CMD_APPLY_CONFIGURATION = 1
CMD_START_DETECTOR = 2
CMD_RECALIBRATE = 5
CMD_RESET_MODULE = 0x52535421

RESULT_NUM_DISTANCES_MASK = 0x0000000F
RESULT_CALIBRATION_NEEDED_MASK = 0x00000200
RESULT_MEASURE_DISTANCE_ERROR_MASK = 0x00000400
RESULT_TEMPERATURE_MASK = 0xFFFF0000

I2C_RETRIES = 8
I2C_RETRY_DELAY = 0.05
DEFAULT_DETECTION_MAX_MM = 2000
ALERT_COOLDOWN_S = 1.5
DEFAULT_SOUND_PATH = Path("/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga")


class XM125Distance:
    def __init__(self, bus_id=I2C_BUS, addr=I2C_ADDR):
        self.bus = SMBus(bus_id)
        self.addr = addr

    def close(self):
        self.bus.close()

    def read_reg(self, reg):
        last_error = None
        for _ in range(I2C_RETRIES):
            try:
                write = i2c_msg.write(self.addr, [(reg >> 8) & 0xFF, reg & 0xFF])
                read = i2c_msg.read(self.addr, 4)
                self.bus.i2c_rdwr(write, read)
                return int.from_bytes(bytes(read), "big")
            except OSError as exc:
                last_error = exc
                time.sleep(I2C_RETRY_DELAY)
        raise RuntimeError(f"I2C read error on reg 0x{reg:04X}: {last_error}")

    def write_reg(self, reg, value):
        last_error = None
        payload = [(reg >> 8) & 0xFF, reg & 0xFF]
        payload.extend(int(value & 0xFFFFFFFF).to_bytes(4, "big"))
        for _ in range(I2C_RETRIES):
            try:
                self.bus.i2c_rdwr(i2c_msg.write(self.addr, payload))
                return
            except OSError as exc:
                last_error = exc
                time.sleep(I2C_RETRY_DELAY)
        raise RuntimeError(f"I2C write error on reg 0x{reg:04X}: {last_error}")

    def detector_error_status(self):
        reg_val = self.read_reg(REG_DETECTOR_STATUS)
        if reg_val & 0x00010000:
            return 1
        if reg_val & 0x00020000:
            return 2
        if reg_val & 0x00040000:
            return 3
        if reg_val & 0x00080000:
            return 5
        if reg_val & 0x00100000:
            return 6
        if reg_val & 0x00200000:
            return 7
        if reg_val & 0x00400000:
            return 8
        if reg_val & 0x00800000:
            return 9
        if reg_val & 0x01000000:
            return 10
        if reg_val & 0x02000000:
            return 11
        if reg_val & 0x10000000:
            return 12
        if reg_val & 0x80000000:
            return 13
        return 0

    def distance_setup(self, start_mm=300, end_mm=5000):
        self.write_reg(REG_COMMAND, CMD_RESET_MODULE)
        time.sleep(0.2)

        if self.detector_error_status() != 0:
            raise RuntimeError("XM125 detector error after reset")

        self.write_reg(REG_START, start_mm)
        time.sleep(0.05)
        self.write_reg(REG_END, end_mm)
        time.sleep(0.05)
        self.write_reg(REG_COMMAND, CMD_APPLY_CONFIGURATION)
        time.sleep(0.25)

        err = self.detector_error_status()
        if err != 0:
            raise RuntimeError(f"XM125 detector configuration error: {err}")

    def read_result(self):
        result = self.read_reg(REG_RESULT)
        num_distances = result & RESULT_NUM_DISTANCES_MASK
        calibration_needed = 1 if (result & RESULT_CALIBRATION_NEEDED_MASK) else 0
        measure_error = 1 if (result & RESULT_MEASURE_DISTANCE_ERROR_MASK) else 0
        temperature_c = (result & RESULT_TEMPERATURE_MASK) >> 16
        return {
            "raw": result,
            "num_distances": num_distances,
            "calibration_needed": calibration_needed,
            "measure_error": measure_error,
            "temperature_c": temperature_c,
        }

    def measure_once(self):
        if self.detector_error_status() != 0:
            raise RuntimeError("XM125 detector status error before measurement")

        self.write_reg(REG_COMMAND, CMD_START_DETECTOR)
        time.sleep(0.25)

        result = self.read_result()

        if result["calibration_needed"]:
            self.write_reg(REG_COMMAND, CMD_RECALIBRATE)
            time.sleep(0.3)
            result = self.read_result()

        if result["measure_error"]:
            raise RuntimeError("XM125 measurement error")

        if result["num_distances"] == 0:
            return {
                "distance_mm": None,
                "strength": None,
                "temperature_c": result["temperature_c"],
                "num_distances": 0,
            }

        distance_mm = self.read_reg(REG_PEAK0_DISTANCE)
        strength_raw = self.read_reg(REG_PEAK0_STRENGTH)
        strength = int.from_bytes(strength_raw.to_bytes(4, "big"), "big", signed=True)

        return {
            "distance_mm": distance_mm,
            "strength": strength,
            "temperature_c": result["temperature_c"],
            "num_distances": result["num_distances"],
        }


def play_detection_sound():
    if DEFAULT_SOUND_PATH.exists():
        try:
            subprocess.Popen(
                ["paplay", str(DEFAULT_SOUND_PATH)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except OSError:
            pass

    # Fallback, gdy systemowy dzwiek nie moze zostac odtworzony.
    print("\a", end="", flush=True)


def main():
    start_mm = 300
    end_mm = 5000
    detection_max_mm = DEFAULT_DETECTION_MAX_MM

    if len(sys.argv) == 3:
        start_mm = int(sys.argv[1])
        end_mm = int(sys.argv[2])
    elif len(sys.argv) == 4:
        start_mm = int(sys.argv[1])
        end_mm = int(sys.argv[2])
        detection_max_mm = int(sys.argv[3])
    elif len(sys.argv) not in (1, 3, 4):
        print(
            "Uzycie: python3 radar_distance.py [start_mm end_mm [alarm_max_mm]]"
        )
        sys.exit(1)

    radar = XM125Distance()
    was_detected = False
    last_alert_time = 0.0

    try:
        radar.distance_setup(start_mm=start_mm, end_mm=end_mm)
        print(f"XM125 gotowy. Zakres: {start_mm} mm - {end_mm} mm")
        print(f"Alarm aktywny dla obiektu do: {detection_max_mm} mm")
        print("Ctrl+C aby zakonczyc.")

        while True:
            reading = radar.measure_once()
            if reading["distance_mm"] is None:
                was_detected = False
                print(f"Brak obiektu | temp={reading['temperature_c']} C")
            else:
                distance_mm = reading["distance_mm"]
                detected_now = distance_mm <= detection_max_mm

                if detected_now and (
                    not was_detected or time.monotonic() - last_alert_time >= ALERT_COOLDOWN_S
                ):
                    play_detection_sound()
                    last_alert_time = time.monotonic()

                status = "WYKRYTO CIEBIE" if detected_now else "obiekt poza strefa alarmu"
                was_detected = detected_now

                print(
                    f"{status} | "
                    f"Odleglosc: {distance_mm} mm | "
                    f"{distance_mm / 10.0:.1f} cm | "
                    f"{distance_mm / 1000.0:.3f} m | "
                    f"sila={reading['strength']} | "
                    f"temp={reading['temperature_c']} C | "
                    f"peaks={reading['num_distances']}"
                )
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nKoniec.")
    finally:
        radar.close()


if __name__ == "__main__":
    main()
