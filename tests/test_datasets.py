from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ahl_api.datasets import collect_eod_snapshot, load_eod_snapshot


class FakeClient:
    def fetch_historical_daily(self, symbol: str, *, years=None):
        return [
            {"date": "2026-01-01", "open": 100, "close": 101, "volume": 1000},
            {"date": "2026-01-02", "open": 102, "close": 103, "volume": 2000},
        ]


class DatasetTests(unittest.TestCase):
    def test_collection_requires_terms_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(PermissionError):
                collect_eod_snapshot(["AAA"], directory, client=FakeClient())

    def test_snapshot_round_trip_and_checksum_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = collect_eod_snapshot(
                ["aaa", "BBB"],
                directory,
                acknowledge_data_terms=True,
                request_delay_seconds=0,
                client=FakeClient(),
            )
            data, loaded_manifest = load_eod_snapshot(directory)

            self.assertEqual(manifest.snapshot_id, loaded_manifest.snapshot_id)
            self.assertEqual(set(data), {"AAA", "BBB"})
            self.assertFalse(manifest.adjusted_for_corporate_actions)
            metadata = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema_version"], 1)

            with (Path(directory) / "AAA.csv").open("a", encoding="utf-8") as handle:
                handle.write("tampered\n")
            with self.assertRaises(ValueError):
                load_eod_snapshot(directory)


if __name__ == "__main__":
    unittest.main()
