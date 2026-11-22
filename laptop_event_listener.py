"""Receives a QNX Medusa decision and triggers the laptop privacy action."""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
from ipaddress import IPv4Address

from medusa.actions import PrivacyAction
from medusa.windows_desktop import TASKBAR_SLOTS, DesktopSwitchTrigger, SnooperSignal


_LOG = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Medusa QNX event listener")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--switch-desktop", action="store_true",
                        help="activate a pinned Windows app on a confirmed Pi snooper event")
    parser.add_argument("--taskbar-slot", type=int, choices=TASKBAR_SLOTS, default=1,
                        help="pinned app position (default: 1)")
    parser.add_argument("--pi-ip", type=IPv4Address, help="IPv4 address of the Pi sending events")
    args = parser.parse_args(argv)

    if args.switch_desktop and args.pi_ip is None:
        parser.error("--switch-desktop requires --pi-ip to select the Pi sending confirmations")
    if args.switch_desktop and sys.platform != "win32":
        parser.error("--switch-desktop requires Windows")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    pi_ip = str(args.pi_ip) if args.pi_ip is not None else None
    action = PrivacyAction(enabled=not args.no_browser and not args.switch_desktop)
    # A Pi clear message already represents a streak of clear camera frames.
    switcher = DesktopSwitchTrigger(args.taskbar_slot, clear_frames=1) if args.switch_desktop else None
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind(("", args.port))
            if switcher is not None:
                _LOG.info("Listening for Pi %s on UDP port %d. Taskbar slot %d.",
                          pi_ip, args.port, args.taskbar_slot)
            while True:
                data, address = sock.recvfrom(1024)
                if pi_ip is not None and address[0] != pi_ip:
                    continue
                try:
                    event = json.loads(data)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(event, dict) or event.get("type") != "medusa-event":
                    continue
                active = event.get("active")
                if not isinstance(active, bool):
                    continue
                reason = event.get("reason")
                if not isinstance(reason, str) or not reason:
                    reason = "a Pi decision"
                if switcher is None:
                    action.update(active)
                    print(f"QNX {address[0]}: active={active}; {reason}")
                    continue
                if active:
                    signal = SnooperSignal.CONFIRMED
                elif event.get("scene_clear") is True:
                    signal = SnooperSignal.CLEAR
                else:
                    signal = SnooperSignal.UNCONFIRMED
                switcher.update(signal, reason)
    except KeyboardInterrupt:
        return 0
    except OSError as error:
        parser.exit(1, f"Medusa UDP listener on port {args.port} failed: {error}\n")
    finally:
        if switcher is not None:
            switcher.close()


if __name__ == "__main__":
    raise SystemExit(main())

