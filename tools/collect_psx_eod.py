from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ahl_api.datasets import collect_eod_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a checksummed, read-only PSX EOD snapshot for explicit symbols."
    )
    parser.add_argument("--symbols", nargs="+", required=True, help="Explicit symbols or index identifiers.")
    parser.add_argument("--output-dir", required=True, help="New snapshot directory; it must not already contain a manifest.")
    parser.add_argument("--years", type=int, default=20, help="Requested history; the public endpoint currently returns about five years.")
    parser.add_argument("--request-delay", type=float, default=1.0, help="Seconds between requests.")
    parser.add_argument(
        "--acknowledge-data-terms",
        action="store_true",
        help="Confirm you reviewed PSX/AHL data terms. This is not a licence grant.",
    )
    parser.add_argument(
        "--systematic-use-authorized",
        action="store_true",
        help="Record that separate written authorization/licensing has been obtained.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = collect_eod_snapshot(
        args.symbols,
        Path(args.output_dir),
        years=args.years,
        acknowledge_data_terms=args.acknowledge_data_terms,
        systematic_use_authorized=args.systematic_use_authorized,
        request_delay_seconds=args.request_delay,
    )
    print(f"snapshot: {Path(args.output_dir).resolve()}")
    print(f"snapshot id: {manifest.snapshot_id}")
    print(f"symbols: {len(manifest.symbols)}")
    print("No authentication was used and no trading endpoint was called.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
