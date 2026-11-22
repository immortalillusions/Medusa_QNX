"""Medusa Pi usage monitor: samples CPU, memory, temperature and uptime every 2 s.

Run it BELOW the control process's priority (QNX):

    on -p 10 python3 -u medusa_sysmon.py --out /tmp/medusa_system.json

It rewrites --out atomically each sample; medusa_control.py's /control.json reads that
file into its ``system`` block (null fields when it is missing or older than 10 s).
Sampling lives here, not in the control process, so the 50 Hz loop never pays for it.
See medusa/sysmon.py.

It also runs the CPU stress test: POST :8082/stress on the control process sends a UDP
datagram to --stress-port, and this process spawns the busy-loop workers (``on -p
--stress-priority``) and writes their state to --stress-file for control.json. The control
process never forks or execs (spawning from it stalled the loop ~150 ms). See medusa/stress.py.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading

from medusa.stress import DEFAULT_STRESS_FILE, DEFAULT_STRESS_PORT, StressRunner, StressService
from medusa.sysmon import DEFAULT_SYSTEM_FILE, SYSTEM_PERIOD_S, SystemSampler


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_SYSTEM_FILE, help="JSON file the control process reads")
    parser.add_argument("--period", type=float, default=SYSTEM_PERIOD_S, help="seconds between samples")
    parser.add_argument("--stress-port", type=int, default=DEFAULT_STRESS_PORT,
                        help="localhost UDP port for stress requests from the control process; 0 disables")
    parser.add_argument("--stress-file", default=DEFAULT_STRESS_FILE, help="stress status JSON the control process reads")
    parser.add_argument("--stress-workers", type=int, default=4, help="busy-loop processes per stress run")
    parser.add_argument("--stress-priority", type=int, default=10,
                        help="QNX priority for stress workers (started with `on -p`)")
    parser.add_argument("--stress-max", type=float, default=60, help="longest stress run in seconds")
    args = parser.parse_args(argv)
    if args.period <= 0 or args.stress_workers < 1:
        parser.error("--period and --stress-workers must be positive")
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    print(f"medusa_sysmon: writing {args.out} every {args.period:g} s", flush=True)
    stress_thread = None
    if args.stress_port:
        try:
            service = StressService(StressRunner(args.stress_workers, args.stress_priority), args.stress_port,
                                    args.stress_file, args.stress_max)
        except OSError as error:
            print(f"medusa_sysmon: stress runner disabled: {error}", file=sys.stderr)
        else:
            stress_thread = threading.Thread(target=service.serve, args=(stop,), daemon=True, name="stress")
            stress_thread.start()
            print(f"medusa_sysmon: stress requests on 127.0.0.1:{service.port}, status in {args.stress_file}",
                  flush=True)
    try:
        SystemSampler(period_s=args.period).run_forever(args.out, stop)
    except OSError as error:
        print(f"medusa_sysmon: {error}", file=sys.stderr)
        return 1
    finally:
        stop.set()
        if stress_thread is not None:
            stress_thread.join(timeout=3)  # kills any running workers
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
