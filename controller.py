"""
Control loop scaffold for hover stabilization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Protocol

from drone_client import DroneClient


@dataclass
class PositionEstimate:
    x: float
    y: float
    z: float
    timestamp: float
    actual_pitch: Optional[float] = None
    actual_roll: Optional[float] = None


@dataclass
class ControlCommand:
    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0
    baseline_thrust: Optional[int] = None


@dataclass
class HoverTarget:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.5


class PositionProvider(Protocol):
    def get_position_estimate(self) -> Optional[PositionEstimate]:
        ...


class HoverController:
    def __init__(
        self,
        drone: DroneClient,
        position_provider: PositionProvider,
        target: HoverTarget = HoverTarget(),
        loop_hz: float = 20.0,
        attitude_correction_hz: float = 4.0,
        attitude_correction_alpha: float = 0.35,
    ) -> None:
        self.drone = drone
        self.position_provider = position_provider
        self.target = target
        self.loop_hz = loop_hz
        self._period = 1.0 / loop_hz
        self._attitude_correction_period = 1.0 / attitude_correction_hz
        self._attitude_correction_alpha = attitude_correction_alpha
        self._last_command = ControlCommand()
        self._pitch_bias = 0.0
        self._roll_bias = 0.0
        self._last_attitude_correction_time = 0.0

    def configure_drone(self) -> None:
        self.drone.set_mode(2)
        self.drone.set_pitch(0)
        self.drone.set_roll(0)
        self.drone.set_p_gain(0.1)
        self.drone.set_i_gain(0.00001)
        self.drone.set_d_gain(1.0)

    def run(self, duration_seconds: Optional[float] = None) -> None:
        start_time = time.monotonic()
        next_tick = start_time

        self.configure_drone()

        while True:
            now = time.monotonic()
            if duration_seconds is not None and now - start_time >= duration_seconds:
                break

            estimate = self.position_provider.get_position_estimate()
            command = self.compute_control_command(estimate)
            self.apply_control(command)

            next_tick += self._period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()

    def compute_control_command(
        self, estimate: Optional[PositionEstimate]
    ) -> ControlCommand:
        if estimate is None:
            return ControlCommand(
                pitch=self._pitch_bias,
                roll=self._roll_bias,
                yaw=0.0,
            )

        x_error = self.target.x - estimate.x
        y_error = self.target.y - estimate.y
        z_error = self.target.z - estimate.z

        self._maybe_update_attitude_bias(estimate)

        # Positive y error means the drone is forward of target and should pitch back.
        desired_pitch = self._clamp_axis(y_error * 10.0)
        desired_roll = self._clamp_axis(x_error * 10.0)
        baseline_thrust = self._compute_baseline_thrust(z_error)

        return ControlCommand(
            pitch=self._clamp_axis(desired_pitch + self._pitch_bias),
            roll=self._clamp_axis(desired_roll + self._roll_bias),
            yaw=0.0,
            baseline_thrust=baseline_thrust,
        )

    def apply_control(self, command: ControlCommand) -> None:
        self.drone.set_pitch(command.pitch)
        self.drone.set_roll(command.roll)
        self.drone.set_yaw(command.yaw)

        if command.baseline_thrust is not None:
            thrust = command.baseline_thrust
            self.drone.manual_thrusts(thrust, thrust, thrust, thrust)

        self._last_command = command

    @staticmethod
    def _clamp_axis(value: float, limit: float = 15.0) -> float:
        return max(-limit, min(limit, value))

    def _maybe_update_attitude_bias(self, estimate: PositionEstimate) -> None:
        if estimate.actual_pitch is None or estimate.actual_roll is None:
            return

        if (
            estimate.timestamp - self._last_attitude_correction_time
            < self._attitude_correction_period
        ):
            return

        gyro_pitch = self.drone.get_pitch()
        gyro_roll = self.drone.get_roll()
        if gyro_pitch is None or gyro_roll is None:
            return

        measured_pitch_bias = gyro_pitch - estimate.actual_pitch
        measured_roll_bias = gyro_roll - estimate.actual_roll

        alpha = self._attitude_correction_alpha
        self._pitch_bias = (1.0 - alpha) * self._pitch_bias + alpha * measured_pitch_bias
        self._roll_bias = (1.0 - alpha) * self._roll_bias + alpha * measured_roll_bias
        self._last_attitude_correction_time = estimate.timestamp

    @staticmethod
    def _compute_baseline_thrust(z_error: float) -> int:
        hover_thrust = 120
        thrust_adjustment = int(z_error * 40.0)
        return max(0, min(250, hover_thrust + thrust_adjustment))


class NullPositionProvider:
    def get_position_estimate(self) -> Optional[PositionEstimate]:
        return None


def main() -> None:
    with DroneClient() as drone:
        controller = HoverController(drone=drone, position_provider=NullPositionProvider())
        controller.run(duration_seconds=5.0)


if __name__ == "__main__":
    main()
