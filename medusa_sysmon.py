"""Medusa Pi usage monitor: samples CPU, memory, temperature and uptime every 2 s.

Run it BELOW the control process's priority (QNX):

    on -p 10 python3 -u medusa_sysmon.py --out /tmp/medusa_system.json

It rewrites --out atomically each sample; medusa_control.py's /control.json reads that
file into its ``system`` block (null fields when it is missing or older than 10 s).
Sampling lives here, not in the control process, so the 50 Hz loop never pays for it.
See medusa/sysmon.py.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading

from medusa.sysmon import DEFAULT_SYSTEM_FILE, SYSTEM_PERIOD_S, SystemSampler


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_SYSTEM_FILE, help="JSON file the control process reads")
    parser.add_argument("--period", type=float, default=SYSTEM_PERIOD_S, help="seconds between samples")
    args = parser.parse_args(argv)
    if args.period <= 0:
        parser.error("--period must be positive")
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    print(f"medusa_sysmon: writing {args.out} every {args.period:g} s", flush=True)
    try:
        SystemSampler(period_s=args.period).run_forever(args.out, stop)
    except OSError as error:
        print(f"medusa_sysmon: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
