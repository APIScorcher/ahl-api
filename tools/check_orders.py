from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ahl_api.client import AHL, read_dotenv


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only order status checks.")
    parser.add_argument("order_ids", nargs="*", help="Specific broker/client order ids to query.")
    args = parser.parse_args()

    env = read_dotenv()
    ahl = AHL({"user": env.get("user"), "pass": env.get("pass")}, audit_enabled=True)
    ahl.login()

    print("OPEN_ORDERS")
    print(json.dumps(ahl.fetch_open_orders(), indent=2, default=str))

    print("CLOSED_ORDERS")
    print(json.dumps(ahl.fetch_closed_orders(), indent=2, default=str))

    for order_id in args.order_ids:
        print(f"ORDER {order_id}")
        try:
            print(json.dumps(ahl.fetch_order(order_id), indent=2, default=str))
        except Exception as exc:
            print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
