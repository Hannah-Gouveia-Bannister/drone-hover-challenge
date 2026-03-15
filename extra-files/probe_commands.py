"""
Send individual commands to the drone and print any responses.
"""

from __future__ import annotations

import argparse

from drone_client import DroneClient


DEFAULT_COMMANDS = [
    "gMode",
    "angX",
    "angY",
    "gyroX",
    "gyroY",
    "geti",
    "mode1",
    "gMode",
    "mode0",
]


def decode_command(command: str) -> str:
    return command.encode("utf-8").decode("unicode_escape")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe drone commands")
    parser.add_argument(
        "commands",
        nargs="*",
        help="Commands to send. If omitted, a default probe set is used.",
    )
    parser.add_argument("--host", default="192.168.4.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument(
        "--expect-greeting",
        action="store_true",
        help="Wait for the startup greeting before probing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = args.commands or DEFAULT_COMMANDS

    with DroneClient(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        expect_greeting=args.expect_greeting,
    ) as drone:
        for command in commands:
            decoded_command = decode_command(command)
            response = drone.send_command(decoded_command)
            print(f"{decoded_command!r} -> {response!r}")


if __name__ == "__main__":
    main()