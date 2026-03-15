"""
Simple hardware tests for motors and baseline thrust.

Use these tests carefully in a clear, secured area with the drone restrained
unless you are intentionally performing a supervised lift test.
"""

from __future__ import annotations

import argparse
import time

from drone_client import DroneClient


def ramp_manual_thrust(
    drone: DroneClient,
    target: int,
    hold_seconds: float,
    step: int,
    delay: float,
) -> None:
    drone.set_mode(1)

    current = 0
    while current < target:
        current = min(target, current + step)
        drone.manual_thrusts(current, current, current, current)
        print(f"Manual thrust: {current}")
        time.sleep(delay)

    print(f"Holding manual thrust at {target} for {hold_seconds:.1f}s")
    time.sleep(hold_seconds)

    while current > 0:
        current = max(0, current - step)
        drone.manual_thrusts(current, current, current, current)
        print(f"Manual thrust: {current}")
        time.sleep(delay)


def test_single_motor(
    drone: DroneClient,
    motor_index: int,
    target: int,
    hold_seconds: float,
    step: int,
    delay: float,
) -> None:
    drone.set_mode(1)

    current = 0
    while current < target:
        current = min(target, current + step)
        thrusts = [0, 0, 0, 0]
        thrusts[motor_index] = current
        drone.manual_thrusts(*thrusts)
        print(f"Motor {motor_index + 1} thrust: {current}")
        time.sleep(delay)

    print(
        f"Holding motor {motor_index + 1} at thrust {target} for {hold_seconds:.1f}s"
    )
    time.sleep(hold_seconds)

    while current > 0:
        current = max(0, current - step)
        thrusts = [0, 0, 0, 0]
        thrusts[motor_index] = current
        drone.manual_thrusts(*thrusts)
        print(f"Motor {motor_index + 1} thrust: {current}")
        time.sleep(delay)


def baseline_pid_test(
    drone: DroneClient,
    baseline_thrust: int,
    hold_seconds: float,
) -> None:
    drone.set_mode(2)
    drone.set_pitch(0)
    drone.set_roll(0)
    drone.set_p_gain(0.1)
    drone.set_i_gain(0.00001)
    drone.set_d_gain(1.0)
    drone.manual_thrusts(
        baseline_thrust,
        baseline_thrust,
        baseline_thrust,
        baseline_thrust,
    )

    print(
        f"PID mode baseline thrust {baseline_thrust} applied for {hold_seconds:.1f}s"
    )
    time.sleep(hold_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bench tests for the drone")
    parser.add_argument(
        "--mode",
        choices=["all-motors", "single-motor", "pid-baseline"],
        default="all-motors",
        help="Test to run",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=80,
        help="Target thrust for all-motors or single-motor mode",
    )
    parser.add_argument(
        "--baseline-thrust",
        type=int,
        default=120,
        help="Baseline thrust for pid-baseline mode",
    )
    parser.add_argument(
        "--motor",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Motor number for single-motor mode",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=2.0,
        help="How long to hold the target thrust",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Thrust increment during ramp-up/ramp-down",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Delay between ramp steps in seconds",
    )
    parser.add_argument(
        "--host",
        default="192.168.4.1",
        help="Drone host address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Drone TCP port",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Socket timeout in seconds",
    )
    parser.add_argument(
        "--no-greeting",
        action="store_true",
        help="Skip waiting for the startup greeting message",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with DroneClient(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        expect_greeting=not args.no_greeting,
    ) as drone:
        try:
            if args.mode == "all-motors":
                ramp_manual_thrust(
                    drone=drone,
                    target=args.target,
                    hold_seconds=args.hold_seconds,
                    step=args.step,
                    delay=args.delay,
                )
            elif args.mode == "single-motor":
                test_single_motor(
                    drone=drone,
                    motor_index=args.motor - 1,
                    target=args.target,
                    hold_seconds=args.hold_seconds,
                    step=args.step,
                    delay=args.delay,
                )
            else:
                baseline_pid_test(
                    drone=drone,
                    baseline_thrust=args.baseline_thrust,
                    hold_seconds=args.hold_seconds,
                )
        finally:
            drone.emergency_stop()


if __name__ == "__main__":
    main()
