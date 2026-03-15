"""
The following functions can be used to communicate with the drone

general advice:
do not have constant high-bandwidth communications with the drone,
because processing time doing wifi stuff is processing time not spent updating the gyroscope,
which will lead to increased drift
"""

import socket
import select
import time

s = None


def _require_socket():
    if s is None:
        raise RuntimeError("Drone socket is not connected")
    return s


def empty_socket(sock):
    input_ready, _, _ = select.select([sock], [], [], 0.0)
    while input_ready:
        data = sock.recv(1)
        if not data:
            break
        input_ready, _, _ = select.select([sock], [], [], 0.0)


def _read_line(sock):
    rx = b""
    while not rx.endswith(b"\n"):
        data = sock.recv(1024)
        if not data:
            raise ConnectionError("Drone disconnected")
        rx += data
    return rx.decode("ascii").strip()


def _parse_float_response(resp, label):
    if resp is None:
        return None

    try:
        return float(resp)
    except ValueError:
        print(f"Invalid {label} response:", resp)
        return None


def msg(tx, clear_buffer=True):
    sock = _require_socket()

    try:
        if clear_buffer:
            empty_socket(sock)

        sock.sendall((tx + "\n").encode("ascii"))
        return _read_line(sock)

    except socket.timeout:
        print("Drone response timeout")
        return None

    except (ConnectionError, OSError) as e:
        print("Communication error:", e)
        return None


def clamp_thrust(val):
    return max(0, min(250, int(val)))


def emergency_stop():
    return msg("mode0")


def e():
    return emergency_stop()


# mode 0: off
# mode 1: full manual motor control
# mode 2: PID control for pitch and roll
def set_mode(m):
    return msg("mode" + str(m))


def get_mode():
    return msg("gMode")


# always between 0 and 250
# in mode 2 sets baseline value in PID results are added to
def manual_thrusts(A, B, C, D):
    A = clamp_thrust(A)
    B = clamp_thrust(B)
    C = clamp_thrust(C)
    D = clamp_thrust(D)
    return msg(f"manT\n{A},{B},{C},{D}")


# same as prev function, but increments last value instead of overwriting
def increment_thrusts(A, B, C, D):
    A = int(A)
    B = int(B)
    C = int(C)
    D = int(D)
    return msg(f"incT\n{A},{B},{C},{D}")


def get_pitch():  # unit close-ish to degrees, but not exact
    resp = msg("angX")
    value = _parse_float_response(resp, "pitch")
    return value / 16 if value is not None else None


def get_roll():  # unit close-ish to degrees, but not exact
    resp = msg("angY")
    value = _parse_float_response(resp, "roll")
    return value / 16 if value is not None else None


def get_gyro_pitch():  # pitch rate in degree/sec
    resp = msg("gyroX")
    return _parse_float_response(resp, "gyro pitch")


def get_gyro_roll():  # roll rate in degree/sec
    resp = msg("gyroY")
    return _parse_float_response(resp, "gyro roll")


# target pitch to aim for in mode 2
# same unit as get_pitch()
def set_pitch(r):
    return msg("gx" + str(r))


# target roll to aim for in mode 2
# same unit as get_roll()
def set_roll(r):
    return msg("gy" + str(r))


def set_p_gain(p):  # approx 0 - 0.5
    return msg("gainP" + str(p))


def set_i_gain(i):  # below 0.00003
    return msg("gainI" + str(i))


def set_d_gain(d):  # approx 0 - 10
    return msg("gainD" + str(d))


def red_LED(val):  # controls LED light. 1 for on, 0 for off
    return msg("lr" + str(val))


def blue_LED(val):
    return msg("lb" + str(val))


def green_LED(val):
    return msg("lg" + str(val))


def reset_integral():  # resets the value of integrands in the PID loops to 0
    return msg("irst")


# returns [I_x, I_y] the integrands from the pitch and roll pid loops
def get_i_values():
    resp = msg("geti")
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


def set_yaw(y):  # directly sets motor difference for yaw control
    return msg("yaw" + str(y))


def get_firmware_version():
    return msg("vers")


# the following functions only work if firmware 1.2 or higher is installed on the drone
# if you want to use this, please make sure by running msg("vers")

# use at start of code if you want to use the drone outside of the cage. Overrides all mode changes
def lock_props():
    return msg("lck")


# recalibrates the gyroscope.
# Do not communicate with the drone for 15 seconds after calling this
def recalibrate():
    return msg("rst")


def main():
    global s

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("192.168.4.1", 8080))
        s.settimeout(5)

        # Initial handshake: don't clear the buffer here
        if _read_line(s) != "connected":
            blue_LED(1)
        else:
            blue_LED(0)

        recalibrate()
        time.sleep(15)

        set_mode(2)
        set_pitch(0)
        set_roll(0)
        set_p_gain(0.1)
        set_i_gain(0.00001)
        set_d_gain(1)

        while True:
            pitch = get_pitch()
            roll = get_roll()
            print("pitch:", pitch, "roll:", roll)

    finally:
        if s is not None:
            try:
                emergency_stop()
            except (RuntimeError, OSError, ConnectionError):
                pass

            try:
                s.close()
            finally:
                s = None


if __name__ == "__main__":
    main()                                  