"""Read-only account and market commands. Credentials never belong in arguments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ahl_api.client import AHL, AhlError, DEFAULT_BASE_URL, read_dotenv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["portfolio", "balance", "accounts", "ticker", "market-status"])
    parser.add_argument("symbol", nargs="?")
    parser.add_argument("--env", type=Path, help="Optional local .env credential file")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args(argv)
    if args.command == "ticker" and not args.symbol:
        parser.error("ticker requires a symbol")
    env = read_dotenv(args.env) if args.env else {}
    config = {
        "user": os.environ.get("AHL_USERNAME") or env.get("user"),
        "pass": os.environ.get("AHL_PASSWORD") or env.get("pass"),
        "pin": os.environ.get("AHL_PIN") or env.get("pin"),
    }
    try:
        with AHL(config, base_url=args.base_url) as client:
            if args.command == "ticker":
                result = client.fetch_ticker(args.symbol)
            else:
                result = getattr(client, "fetch_" + args.command.replace("-", "_"))()
            print(json.dumps(result, indent=2, default=str))
    except (AhlError, OSError) as exc:
        print(f"ahl: {exc}", file=sys.stderr)
        return 1
    return 0
