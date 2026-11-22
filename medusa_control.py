"""Medusa real-time control process: owns /dev/ser1 and keeps the Arduino heartbeat at 50 Hz.

Start it at a higher priority than the AI process (QNX):

    on -p 21 python3 -u medusa_control.py --port /dev/ser1

``on -p 21`` sets the process's priority; threads Python starts here (the
HTTP server, the stress timer) inherit it, and the 50 Hz loop runs in the main
thread. The AI process stays at the default priority 10 and sends targets with
``qnx_ai_service.py --control-port 8768``. See medusa/control.py.

The servo defaults (65..115, home 90) predate the current turret. Pass the
firmware's ranges (``--servo-min 15 --servo-max 165 --servo-home 90``, tilt 90),
or the firmware rejects every park line.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading

from medusa.control import (
    DEFAULT_ACTUAL_PORT, DEFAULT_HTTP_PORT, DEFAULT_TARGET_PORT, PERIOD_S, STRESS_MAX_S, TARGET_TIMEOUT_S,
    ControlHTTPServer, ControlLoop, SerialPort, StressTest, _udp_socket,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", default="/dev/ser1", help="serial device wired to the Arduino")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--target-port", type=int, default=DEFAULT_TARGET_PORT,
                        help="localhost UDP port for medusa-target packets from the AI process")
    parser.add_argument("--actual-port", type=int, default=DEFAULT_ACTUAL_PORT,
                        help="localhost UDP port the AI process listens on for medusa-actual; 0 disables")
    parser.add_argument("--http-port", type=int, default=DEFAULT_HTTP_PORT, help="0.0.0.0 port for /control.json")
    parser.add_argument("--period-ms", type=float, default=PERIOD_S * 1000)
    parser.add_argument("--target-timeout", type=float, default=TARGET_TIMEOUT_S,
                        help="seconds an AI target stays fresh; older targets park the turret")
    parser.add_argument("--servo-min", type=int, default=65)
    parser.add_argument("--servo-max", type=int, default=115)
    parser.add_argument("--servo-home", type=int, default=90)
    parser.add_argument("--tilt-min", type=int, default=65)
    parser.add_argument("--tilt-max", type=int, default=115)
    parser.add_argument("--tilt-home", type=int, default=90)
    parser.add_argument("--stress-workers", type=int, default=4, help="busy-loop processes for POST /stress")
    parser.add_argument("--stress-priority", type=int, default=10,
                        help="QNX priority for stress workers (started with `on -p`)")
    parser.add_argument("--switch-interval", type=float, default=0.001,
                        help="sys.setswitchinterval: how long another thread may hold the GIL")
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("servo", "tilt"):
        low, high, home = (getattr(args, f"{name}_{k}") for k in ("min", "max", "home"))
        if not 0 <= low <= home <= high <= 180:
            parser.error(f"--{name}-min <= --{name}-home <= --{name}-max must hold within 0..180")
    if args.period_ms <= 0 or args.target_timeout <= 0 or args.stress_workers < 1:
        parser.error("--period-ms, --target-timeout and --stress-workers must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    sys.setswitchinterval(args.switch_interval)
    try:
        serial = SerialPort.open(args.port, args.baud)
    except (OSError, AttributeError) as error:
        print(f"medusa_control cannot start: serial {args.port}: {error}", file=sys.stderr)
        return 2
    sockets, http = [], None
    stress = StressTest(args.stress_workers, args.stress_priority)
    loop = None
    try:
        target_sock = _udp_socket(args.target_port)
        sockets.append(target_sock)
        actual_sock = _udp_socket() if args.actual_port else None
        if actual_sock:
            sockets.append(actual_sock)
        loop = ControlLoop(
            serial, target_sock=target_sock, actual_sock=actual_sock,
            actual_addr=("127.0.0.1", args.actual_port) if args.actual_port else None,
            period_s=args.period_ms / 1000, target_timeout=args.target_timeout,
            yaw_range=(args.servo_min, args.servo_max), pitch_range=(args.tilt_min, args.tilt_max),
            home=(args.servo_home, args.tilt_home), stress=stress,
        )
        http = ControlHTTPServer(loop, args.http_port)
        http.start()
        stop = threading.Event()
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: stop.set())
        print(f"medusa_control: {args.port}, targets on 127.0.0.1:{args.target_port}, "
              f"status on :{http.port}/control.json, {1000 / args.period_ms:.0f} Hz", flush=True)
        loop.run(stop)
    except OSError as error:
        print(f"medusa_control cannot start: {error}", file=sys.stderr)
        return 2
    finally:
        stress.stop()
        if loop is not None:
            loop.shutdown()  # Park lines: home, LED off; the firmware disarms 300 ms later.
        serial.close()
        if http is not None:
            http.stop()
        for sock in sockets:
            sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
