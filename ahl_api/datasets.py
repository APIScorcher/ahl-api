"""Reproducible, explicit-symbol PSX EOD snapshots.

The collector is deliberately opt-in because PSX data terms restrict automated
retrieval and storage. A terms acknowledgement is not a data licence; the
manifest records authorization separately so research reports cannot silently
claim licensed provenance.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ahl_api.backtest import normalize_bars
from ahl_api.client import AHL


@dataclass(frozen=True)
class SnapshotManifest:
    schema_version: int
    snapshot_id: str
    collected_at_utc: str
    source: str
    requested_years: int | None
    symbols: tuple[str, ...]
    files: dict[str, dict[str, Any]]
    adjusted_for_corporate_actions: bool
    includes_cash_dividends: bool
    point_in_time_universe: bool
    historical_membership_complete: bool
    historical_board_lots_complete: bool
    historical_sessions_complete: bool
    circuit_and_halt_data_complete: bool
    systematic_use_authorized: bool
    limitations: tuple[str, ...]

    @classmethod
    def from_json(cls, path: str | Path) -> "SnapshotManifest":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        values["symbols"] = tuple(values["symbols"])
        values["limitations"] = tuple(values["limitations"])
        return cls(**values)


def collect_eod_snapshot(
    symbols: Sequence[str],
    output_dir: str | Path,
    *,
    years: int | None = None,
    acknowledge_data_terms: bool = False,
    systematic_use_authorized: bool = False,
    request_delay_seconds: float = 1.0,
    client: Any | None = None,
) -> SnapshotManifest:
    """Collect one immutable snapshot for explicitly supplied symbols."""

    if not acknowledge_data_terms:
        raise PermissionError(
            "PSX data terms must be reviewed and acknowledged before collection; "
            "acknowledgement does not grant a systematic-use licence"
        )
    normalized = tuple(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()))
    if not normalized:
        raise ValueError("at least one explicit symbol is required")
    if request_delay_seconds < 0:
        raise ValueError("request_delay_seconds cannot be negative")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"snapshot already exists: {manifest_path}")

    public_client = client or AHL(dry_run=True, audit_enabled=False)
    files: dict[str, dict[str, Any]] = {}
    for index, symbol in enumerate(normalized):
        rows = public_client.fetch_historical_daily(symbol, years=years)
        bars = normalize_bars(symbol, rows)
        if bars.empty:
            raise ValueError(f"no EOD rows returned for {symbol}")
        path = destination / f"{symbol}.csv"
        bars.to_csv(path, index=False, lineterminator="\n")
        files[symbol] = {
            "path": path.name,
            "sha256": _sha256(path),
            "rows": int(len(bars)),
            "start_date": str(pd.Timestamp(bars["date"].min()).date()),
            "end_date": str(pd.Timestamp(bars["date"].max()).date()),
        }
        if index < len(normalized) - 1 and request_delay_seconds:
            time.sleep(request_delay_seconds)

    collected_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    snapshot_id = hashlib.sha256(
        (collected_at + "|" + "|".join(f"{symbol}:{files[symbol]['sha256']}" for symbol in normalized)).encode("ascii")
    ).hexdigest()[:16]
    manifest = SnapshotManifest(
        schema_version=1,
        snapshot_id=snapshot_id,
        collected_at_utc=collected_at,
        source="AHL client via PSX DPS /timeseries/eod/{symbol}",
        requested_years=years,
        symbols=normalized,
        files=files,
        adjusted_for_corporate_actions=False,
        includes_cash_dividends=False,
        point_in_time_universe=False,
        historical_membership_complete=False,
        historical_board_lots_complete=False,
        historical_sessions_complete=False,
        circuit_and_halt_data_complete=False,
        systematic_use_authorized=bool(systematic_use_authorized),
        limitations=(
            "Rolling history observed at approximately five years regardless of longer requests.",
            "Rows contain open, close, and volume; high and low are unavailable.",
            "Corporate-action, dividend, symbol-history, and adjustment semantics are unverified.",
            "The symbol list is not a point-in-time historical KSE-100 universe.",
            "Portal data is delayed and is unsuitable for historical intraday execution research.",
        ),
    )
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_eod_snapshot(
    snapshot_dir: str | Path,
    *,
    symbols: Sequence[str] | None = None,
    verify_checksums: bool = True,
) -> tuple[dict[str, pd.DataFrame], SnapshotManifest]:
    directory = Path(snapshot_dir)
    manifest = SnapshotManifest.from_json(directory / "manifest.json")
    selected = tuple(symbol.upper() for symbol in symbols) if symbols is not None else manifest.symbols
    unknown = set(selected).difference(manifest.files)
    if unknown:
        raise ValueError(f"symbols are not present in snapshot: {sorted(unknown)}")

    data: dict[str, pd.DataFrame] = {}
    for symbol in selected:
        metadata = manifest.files[symbol]
        path = directory / str(metadata["path"])
        if verify_checksums and _sha256(path) != metadata["sha256"]:
            raise ValueError(f"snapshot checksum mismatch: {symbol}")
        frame = pd.read_csv(path)
        normalized = normalize_bars(symbol, frame)
        if len(normalized) != int(metadata["rows"]):
            raise ValueError(f"snapshot row-count mismatch: {symbol}")
        data[symbol] = normalized
    return data, manifest


def manifest_provenance(manifest: SnapshotManifest) -> dict[str, Any]:
    return {
        "source": manifest.source,
        "adjusted_for_corporate_actions": manifest.adjusted_for_corporate_actions,
        "includes_cash_dividends": manifest.includes_cash_dividends,
        "point_in_time_universe": manifest.point_in_time_universe,
        "historical_membership_complete": manifest.historical_membership_complete,
        "historical_board_lots_complete": manifest.historical_board_lots_complete,
        "historical_sessions_complete": manifest.historical_sessions_complete,
        "circuit_and_halt_data_complete": manifest.circuit_and_halt_data_complete,
        "systematic_use_authorized": manifest.systematic_use_authorized,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "SnapshotManifest",
    "collect_eod_snapshot",
    "load_eod_snapshot",
    "manifest_provenance",
]
