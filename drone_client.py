"""
Drone socket client for sending commands and reading telemetry.
"""

from __future__ import annotations

import socket
from typing import Optional


class DroneClient:
    def __init__(
        self,
        host: str = "192.168.4.1",
        port: int = 8080,
        timeout: float = 2.0,
        expect_greeting: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.expect_greeting = expect_greeting
        self._socket: Optional[socket.socket] = None

    def connect(self) -> None:
        if self._socket is not None:
            return

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect((self.host, self.port))

        if self.expect_greeting:
            try:
                greeting = self._read_line(sock)
            except socket.timeout as exc:
                sock.close()
                raise ConnectionError(
                    "Connected to the drone network, but no greeting was received on "
                    f"{self.host}:{self.port}. The ESP32 may be using a different "
                    "protocol, port, or startup message."
                ) from exc

            if greeting != "connected":
                sock.close()
                raise ConnectionError(
                    f"Unexpected greeting from drone: {greeting!r}"
                )

        self._socket = sock

    def close(self) -> None:
        if self._socket is None:
            return

        try:
            self.emergency_stop()
        except (OSError, ConnectionError):
            pass
        finally:
            self._socket.close()
            self._socket = None

    def __enter__(self) -> "DroneClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _require_socket(self) -> socket.socket:
        if self._socket is None:
            raise RuntimeError("Drone socket is not connected")
        return self._socket

    def _read_line(self, sock: socket.socket) -> str:
        rx = b""
        while not rx.endswith(b"\n"):
            data = sock.recv(1024)
            if not data:
                raise ConnectionError("Drone disconnected")
            rx += data
        return rx.decode("ascii").strip()

    def send_command(self, command: str) -> Optional[str]:
        sock = self._require_socket()

        try:
            sock.sendall((command + "\n").encode("ascii"))
            return self._read_line(sock)
        except socket.timeout:
            print("Drone response timeout")
            return None
        except (ConnectionError, OSError) as exc:
            print("Communication error:", exc)
            return None

    @staticmethod
    def _clamp_thrust(value: float) -> int:
        return max(0, min(250, int(value)))

    @staticmethod
    def _parse_float_response(resp: Optional[str], label: str) -> Optional[float]:
        if resp is None:
            return None

        try:
            return float(resp)
        except ValueError:
            print(f"Invalid {label} response:", resp)
            return None

    def emergency_stop(self) -> Optional[str]:
        return self.send_command("mode0")

    def set_mode(self, mode: int) -> Optional[str]:
        return self.send_command(f"mode{int(mode)}")

    def get_mode(self) -> Optional[str]:
        return self.send_command("gMode")

    def manual_thrusts(self, a: float, b: float, c: float, d: float) -> Optional[str]:
        thrusts = [self._clamp_thrust(value) for value in (a, b, c, d)]
        return self.send_command(f"manT\n{thrusts[0]},{thrusts[1]},{thrusts[2]},{thrusts[3]}")

    def increment_thrusts(self, a: float, b: float, c: float, d: float) -> Optional[str]:
        values = [int(value) for value in (a, b, c, d)]
        return self.send_command(f"incT\n{values[0]},{values[1]},{values[2]},{values[3]}")

    def get_pitch(self) -> Optional[float]:
        value = self._parse_float_response(self.send_command("angX"), "pitch")
        return value / 16 if value is not None else None

    def get_roll(self) -> Optional[float]:
        value = self._parse_float_response(self.send_command("angY"), "roll")
        return value / 16 if value is not None else None

    def get_gyro_pitch(self) -> Optional[float]:
        return self._parse_float_response(self.send_command("gyroX"), "gyro pitch")

    def get_gyro_roll(self) -> Optional[float]:
        return self._parse_float_response(self.send_command("gyroY"), "gyro roll")

    def set_pitch(self, pitch: float) -> Optional[str]:
        return self.send_command(f"gx{pitch}")

    def set_roll(self, roll: float) -> Optional[str]:
        return self.send_command(f"gy{roll}")

    def set_p_gain(self, gain: float) -> Optional[str]:
        return self.send_command(f"gainP{gain}")

    def set_i_gain(self, gain: float) -> Optional[str]:
        return self.send_command(f"gainI{gain}")

    def set_d_gain(self, gain: float) -> Optional[str]:
        return self.send_command(f"gainD{gain}")

    def red_led(self, value: int) -> Optional[str]:
        return self.send_command(f"lr{int(value)}")

    def blue_led(self, value: int) -> Optional[str]:
        return self.send_command(f"lb{int(value)}")

    def green_led(self, value: int) -> Optional[str]:
        return self.send_command(f"lg{int(value)}")

    def reset_integral(self) -> Optional[str]:
        return self.send_command("irst")

    def get_i_values(self) -> Optional[list[float]]:
        resp = self.send_command("geti")
        if resp is None:
            return None

        values = resp.split(",")
        if len(values) != 2:
            print("Invalid integral response:", resp)
            return None

        try:
            return [float(values[0]), float(values[1])]
        except ValueError:
            print("Invalid integral response:", resp)
            return None

    def set_yaw(self, yaw: float) -> Optional[str]:
        return self.send_command(f"yaw{yaw}")
