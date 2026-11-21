"""Receives a QNX Medusa decision and triggers the laptop privacy action."""

from __future__ import annotations

import argparse
import json
import socket

from medusa.actions import PrivacyAction


def main() -> int:
    parser = argparse.ArgumentParser(description="Medusa QNX event listener")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    action = PrivacyAction(enabled=not args.no_browser)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("", args.port))
        while True:
            data, address = sock.recvfrom(1024)
            try:
                event = json.loads(data)
                if event.get("type") != "medusa-event":
                    continue
                active = bool(event.get("active"))
                action.update(active)
                print(f"QNX {address[0]}: active={active}; {event.get('reason', '')}")
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue


if __name__ == "__main__":
    main()

