"""Data auditing and walk-forward validation for PSX strategy research."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from ahl_api.backtest import BacktestConfig, BacktestEngine, BacktestResult


@dataclass(frozen=True)
class DataProvenance:
    source: str = "unspecified"
    adjusted_for_corporate_actions: bool = False
    includes_cash_dividends: bool = False
    point_in_time_universe: bool = False
    historical_membership_complete: bool = False
    historical_board_lots_complete: bool = False
    historical_sessions_complete: bool = False
    circuit_and_halt_data_complete: bool = False
    systematic_use_authorized: bool = False


@dataclass(frozen=True)
class DataAuditResult:
    summary: pd.DataFrame
    issues: pd.DataFrame
    provenance: DataProvenance

    @property
    def has_blockers(self) -> bool:
        return not self.issues.empty and bool((self.issues["severity"] == "blocker").any())

    def write_report(self, output_dir: str | Path, *, prefix: str = "data_audit") -> dict[str, Path]:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "summary": directory / f"{prefix}_summary.csv",
            "issues": directory / f"{prefix}_issues.csv",
            "metadata": directory / f"{prefix}_metadata.json",
        }
        self.summary.to_csv(paths["summary"], index=False)
        self.issues.to_csv(paths["issues"], index=False)
        metadata = {**self.provenance.__dict__, "has_blockers": self.has_blockers}
        paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        return paths


@dataclass(frozen=True)
class MembershipSchedule:
    """Point-in-time symbol membership using inclusive effective dates."""

    rows: pd.DataFrame

    @classmethod
    def from_csv(cls, path: str | Path) -> "MembershipSchedule":
        return cls.from_frame(pd.read_csv(path))

    @classmethod
    def from_frame(cls, rows: pd.DataFrame) -> "MembershipSchedule":
        required = {"symbol", "effective_from", "effective_until"}
        missing = required.difference(rows.columns)
        if missing:
            raise ValueError(f"membership rows missing columns: {sorted(missing)}")
        frame = rows.copy()
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        frame["effective_from"] = pd.to_datetime(frame["effective_from"]).dt.normalize()
        frame["effective_until"] = pd.to_datetime(frame["effective_until"], errors="coerce").dt.normalize()
        return cls(frame.sort_values(["effective_from", "symbol"]).reset_index(drop=True))

    def symbols_on(self, value: Any) -> set[str]:
        current = pd.Timestamp(value).normalize()
        active = self.rows[
            (self.rows["effective_from"] <= current)
            & (self.rows["effective_until"].isna() | (self.rows["effective_until"] >= current))
        ]
        return set(active["symbol"])


@dataclass(frozen=True)
class StrategySpec:
    name: str
    factory: Callable[..., Any]
    parameter_grid: Mapping[str, Sequence[Any]]


@dataclass(frozen=True)
class WalkForwardConfig:
    train_months: int = 24
    test_months: int = 6
    step_months: int = 6
    mode: str = "rolling"
    embargo_sessions: int = 1
    inner_train_months: int = 12
    inner_test_months: int = 3
    inner_step_months: int = 3
    minimum_inner_folds: int = 2
    minimum_round_trips: int = 5
    continue_on_interrupt: bool = True


@dataclass(frozen=True)
class StressScenario:
    name: str
    config_overrides: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame
    selection_runs: pd.DataFrame
    leaderboard: pd.DataFrame
    stitched_leaderboard: pd.DataFrame
    results: dict[str, BacktestResult] = field(default_factory=dict, repr=False)

    def write_report(self, output_dir: str | Path, *, prefix: str = "walk_forward") -> dict[str, Path]:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "folds": directory / f"{prefix}_folds.csv",
            "selection_runs": directory / f"{prefix}_selection_runs.csv",
            "leaderboard": directory / f"{prefix}_leaderboard.csv",
            "stitched_leaderboard": directory / f"{prefix}_stitched_leaderboard.csv",
            "json": directory / f"{prefix}.json",
        }
        self.folds.to_csv(paths["folds"], index=False)
        self.selection_runs.to_csv(paths["selection_runs"], index=False)
        self.leaderboard.to_csv(paths["leaderboard"], index=False)
        self.stitched_leaderboard.to_csv(paths["stitched_leaderboard"], index=False)
        paths["json"].write_text(
            json.dumps(
                {
                    "folds": _records(self.folds),
                    "selection_runs": _records(self.selection_runs),
                    "leaderboard": _records(self.leaderboard),
                    "stitched_leaderboard": _records(self.stitched_leaderboard),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        for run_id, result in self.results.items():
            result_paths = result.write_report(directory / "runs" / run_id, prefix=run_id)
            for key, path in result_paths.items():
                paths[f"{run_id}.{key}"] = path
        return paths


def audit_market_data(
    data: Mapping[str, pd.DataFrame],
    *,
    provenance: DataProvenance | None = None,
    minimum_rows: int = 252,
    minimum_coverage: float = 0.95,
    maximum_stale_sessions: int = 5,
    suspicious_return_threshold: float = 0.35,
) -> DataAuditResult:
    """Audit bars without silently converting missing data into usable evidence."""

    if not data:
        raise ValueError("data must include at least one symbol")
    source = provenance or DataProvenance()
    normalized_dates: dict[str, pd.Series] = {}
    for symbol, frame in data.items():
        normalized_dates[symbol.upper()] = pd.to_datetime(frame.get("date", pd.Series(dtype=object)), errors="coerce")
    all_dates = sorted({value.normalize() for dates in normalized_dates.values() for value in dates.dropna()})
    latest_date = pd.Timestamp(all_dates[-1]) if all_dates else pd.NaT
    maximum_rows = max((len(frame) for frame in data.values()), default=0)

    summary_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    required = {"date", "open", "close", "volume"}
    for raw_symbol, frame in sorted(data.items()):
        symbol = raw_symbol.upper()
        missing_columns = sorted(required.difference(frame.columns))
        if missing_columns:
            _issue(issue_rows, symbol, "blocker", "missing_columns", f"Missing required columns: {', '.join(missing_columns)}")
            continue

        dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        opens = pd.to_numeric(frame["open"], errors="coerce")
        closes = pd.to_numeric(frame["close"], errors="coerce")
        volumes = pd.to_numeric(frame["volume"], errors="coerce")
        highs = pd.to_numeric(frame["high"], errors="coerce") if "high" in frame else pd.Series(pd.NA, index=frame.index, dtype="Float64")
        lows = pd.to_numeric(frame["low"], errors="coerce") if "low" in frame else pd.Series(pd.NA, index=frame.index, dtype="Float64")
        valid_dates = dates.dropna()
        duplicate_dates = int(valid_dates.duplicated().sum())
        nonpositive_prices = int(((opens <= 0) | (closes <= 0)).fillna(False).sum())
        zero_volume_rows = int((volumes <= 0).fillna(True).sum())
        missing_core_rows = int((dates.isna() | opens.isna() | closes.isna() | volumes.isna()).sum())
        close_returns = closes.pct_change(fill_method=None)
        suspicious_returns = int((close_returns.abs() >= suspicious_return_threshold).fillna(False).sum())
        opening_gaps = opens / closes.shift(1) - 1.0
        suspicious_gaps = int((opening_gaps.abs() >= suspicious_return_threshold).fillna(False).sum())
        high_low_rows = int((highs.notna() & lows.notna()).sum())
        invalid_ohlc = int(
            (
                (highs < lows)
                | (highs < pd.concat([opens, closes], axis=1).max(axis=1))
                | (lows > pd.concat([opens, closes], axis=1).min(axis=1))
            ).fillna(False).sum()
        )
        coverage = len(frame) / maximum_rows if maximum_rows else 0.0
        stale_sessions = None
        if not valid_dates.empty and not pd.isna(latest_date):
            last = pd.Timestamp(valid_dates.max())
            stale_sessions = len([value for value in all_dates if last < value <= latest_date])

        if len(frame) < minimum_rows:
            _issue(issue_rows, symbol, "blocker", "insufficient_history", f"Only {len(frame)} rows; need at least {minimum_rows}.")
        if coverage < minimum_coverage:
            _issue(issue_rows, symbol, "warning", "low_coverage", f"Coverage is {coverage:.1%} of the fullest symbol.")
        if stale_sessions is not None and stale_sessions > maximum_stale_sessions:
            _issue(issue_rows, symbol, "warning", "stale_series", f"Last bar is {stale_sessions} market sessions stale.")
        if duplicate_dates:
            _issue(issue_rows, symbol, "blocker", "duplicate_dates", f"Found {duplicate_dates} duplicate dates.")
        if missing_core_rows:
            _issue(issue_rows, symbol, "blocker", "missing_core_values", f"Found {missing_core_rows} rows with missing core values.")
        if nonpositive_prices:
            _issue(issue_rows, symbol, "blocker", "nonpositive_prices", f"Found {nonpositive_prices} rows with nonpositive prices.")
        if zero_volume_rows:
            _issue(issue_rows, symbol, "warning", "nonpositive_volume", f"Found {zero_volume_rows} rows with missing or nonpositive volume.")
        gap_fraction = suspicious_gaps / max(len(frame) - 1, 1)
        if gap_fraction > 0.05:
            _issue(
                issue_rows,
                symbol,
                "blocker",
                "open_close_scale_mismatch",
                f"{suspicious_gaps} opening gaps ({gap_fraction:.1%}) exceed the discontinuity threshold; open and adjusted close scales appear inconsistent.",
            )
        elif suspicious_returns or suspicious_gaps:
            _issue(
                issue_rows,
                symbol,
                "warning",
                "suspicious_discontinuity",
                f"Found {suspicious_returns} close jumps and {suspicious_gaps} opening gaps above {suspicious_return_threshold:.0%}.",
            )
        if high_low_rows == 0:
            _issue(issue_rows, symbol, "warning", "high_low_unavailable", "Canonical ATR, Donchian, and intraday stop tests are blocked.")
        elif invalid_ohlc:
            _issue(issue_rows, symbol, "blocker", "invalid_ohlc", f"Found {invalid_ohlc} inconsistent OHLC rows.")

        summary_rows.append(
            {
                "symbol": symbol,
                "rows": int(len(frame)),
                "start_date": valid_dates.min() if not valid_dates.empty else None,
                "end_date": valid_dates.max() if not valid_dates.empty else None,
                "coverage": coverage,
                "stale_sessions": stale_sessions,
                "duplicate_dates": duplicate_dates,
                "missing_core_rows": missing_core_rows,
                "nonpositive_prices": nonpositive_prices,
                "nonpositive_volume_rows": zero_volume_rows,
                "high_low_rows": high_low_rows,
                "suspicious_close_returns": suspicious_returns,
                "suspicious_opening_gaps": suspicious_gaps,
                "suspicious_opening_gap_fraction": gap_fraction,
            }
        )

    if not source.adjusted_for_corporate_actions:
        _issue(issue_rows, "*", "blocker", "corporate_actions_unverified", "Prices are not verified for splits, bonus shares, rights, and symbol changes.")
    if not source.includes_cash_dividends:
        _issue(issue_rows, "*", "blocker", "dividends_missing", "Stock returns exclude cash dividends and dividend withholding tax.")
    if not source.point_in_time_universe or not source.historical_membership_complete:
        _issue(issue_rows, "*", "blocker", "survivorship_bias", "Historical index membership is incomplete or not enforced point in time.")
    if not source.historical_board_lots_complete:
        _issue(issue_rows, "*", "blocker", "historical_board_lots_missing", "Date- and symbol-specific historical board lots are unavailable.")
    if not source.historical_sessions_complete:
        _issue(issue_rows, "*", "blocker", "historical_sessions_missing", "Historical regular, Friday, Ramadan, and exceptional session schedules are incomplete.")
    if not source.circuit_and_halt_data_complete:
        _issue(issue_rows, "*", "blocker", "circuits_and_halts_missing", "Historical circuit locks, executable liquidity, and halts are unavailable.")
    if not source.systematic_use_authorized:
        _issue(issue_rows, "*", "warning", "data_license_unconfirmed", "Authorization for systematic collection/use has not been confirmed.")

    return DataAuditResult(pd.DataFrame(summary_rows), pd.DataFrame(issue_rows), source)


def generate_walk_forward_folds(calendar: Sequence[Any], config: WalkForwardConfig) -> list[dict[str, Any]]:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(list(calendar)).normalize().unique()))
    if dates.empty:
        return []
    if config.mode not in {"rolling", "anchored"}:
        raise ValueError("walk-forward mode must be rolling or anchored")
    if min(config.train_months, config.test_months, config.step_months) <= 0:
        raise ValueError("walk-forward month values must be positive")
    if config.embargo_sessions < 0:
        raise ValueError("embargo_sessions cannot be negative")

    folds: list[dict[str, Any]] = []
    first = dates.min()
    anchor = first
    while True:
        train_start_target = first if config.mode == "anchored" else anchor
        train_end_target = anchor + pd.DateOffset(months=config.train_months)
        train_dates = dates[(dates >= train_start_target) & (dates < train_end_target)]
        if train_dates.empty:
            break
        later = dates[dates > train_dates.max()]
        if len(later) <= config.embargo_sessions:
            break
        test_start = pd.Timestamp(later[config.embargo_sessions])
        test_end_target = test_start + pd.DateOffset(months=config.test_months)
        test_dates = dates[(dates >= test_start) & (dates < test_end_target)]
        if test_dates.empty:
            break
        folds.append(
            {
                "fold": len(folds),
                "train_start": pd.Timestamp(train_dates.min()),
                "train_end": pd.Timestamp(train_dates.max()),
                "test_start": pd.Timestamp(test_dates.min()),
                "test_end": pd.Timestamp(test_dates.max()),
            }
        )
        anchor = anchor + pd.DateOffset(months=config.step_months)
        if anchor >= dates.max():
            break
    return folds


def run_walk_forward_validation(
    data: Mapping[str, pd.DataFrame],
    specs: Sequence[StrategySpec],
    *,
    backtest_config: BacktestConfig,
    validation_config: WalkForwardConfig | None = None,
    stress_scenarios: Sequence[StressScenario] | None = None,
) -> WalkForwardResult:
    """Nested parameter selection, untouched outer tests, and continuous OOS stitching."""

    validation = validation_config or WalkForwardConfig()
    if min(
        validation.inner_train_months,
        validation.inner_test_months,
        validation.inner_step_months,
        validation.minimum_inner_folds,
    ) <= 0:
        raise ValueError("inner walk-forward values must be positive")
    calendar = sorted({pd.Timestamp(value).normalize() for frame in data.values() for value in pd.to_datetime(frame["date"])})
    folds = generate_walk_forward_folds(calendar, validation)
    scenarios = list(stress_scenarios or [StressScenario("base")])
    if not scenarios or scenarios[0].name != "base":
        scenarios.insert(0, StressScenario("base"))

    selection_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    stitched_rows: list[dict[str, Any]] = []
    retained_results: dict[str, BacktestResult] = {}
    selected_segments: dict[str, list[dict[str, Any]]] = {spec.name: [] for spec in specs}

    for fold in folds:
        outer_train_calendar = [
            value
            for value in calendar
            if fold["train_start"] <= value <= fold["train_end"]
        ]
        inner_config = WalkForwardConfig(
            train_months=validation.inner_train_months,
            test_months=validation.inner_test_months,
            step_months=validation.inner_step_months,
            mode=validation.mode,
            embargo_sessions=validation.embargo_sessions,
            inner_train_months=validation.inner_train_months,
            inner_test_months=validation.inner_test_months,
            inner_step_months=validation.inner_step_months,
            minimum_inner_folds=validation.minimum_inner_folds,
            minimum_round_trips=validation.minimum_round_trips,
            continue_on_interrupt=validation.continue_on_interrupt,
        )
        inner_folds = [
            inner
            for inner in generate_walk_forward_folds(outer_train_calendar, inner_config)
            if inner["test_end"] <= fold["train_end"]
        ]
        for spec in specs:
            candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
            interrupted = False
            if len(inner_folds) < validation.minimum_inner_folds:
                fold_rows.append(
                    {
                        **fold,
                        "run_id": f"fold_{fold['fold']:02d}_{spec.name}_insufficient_inner_history",
                        "strategy": spec.name,
                        "scenario": "base",
                        "status": "insufficient_inner_history",
                        "parameters": None,
                        "inner_folds": len(inner_folds),
                    }
                )
                continue
            for run_index, parameters in enumerate(_parameter_combinations(spec.parameter_grid)):
                inner_metrics: list[dict[str, Any]] = []
                for inner in inner_folds:
                    inner_test_config = replace(
                        backtest_config,
                        start_date=inner["test_start"],
                        end_date=inner["test_end"],
                    )
                    try:
                        result = BacktestEngine(data, config=inner_test_config).run(spec.factory(**parameters))
                    except KeyboardInterrupt:
                        if not validation.continue_on_interrupt:
                            raise
                        interrupted = True
                        break
                    inner_metrics.append(result.metrics)
                    selection_rows.append(
                        {
                            **fold,
                            "selection_level": "inner_fold",
                            "inner_fold": inner["fold"],
                            "inner_train_start": inner["train_start"],
                            "inner_train_end": inner["train_end"],
                            "inner_test_start": inner["test_start"],
                            "inner_test_end": inner["test_end"],
                            "strategy": spec.name,
                            "run_index": run_index,
                            "parameters": json.dumps(parameters, sort_keys=True),
                            "selection_score": _selection_score(result.metrics, validation.minimum_round_trips),
                            **result.metrics,
                        }
                    )
                if interrupted:
                    break
                aggregate = _aggregate_inner_selection(inner_metrics, validation.minimum_round_trips)
                selection_rows.append(
                    {
                        **fold,
                        "selection_level": "aggregate",
                        "inner_fold": None,
                        "strategy": spec.name,
                        "run_index": run_index,
                        "parameters": json.dumps(parameters, sort_keys=True),
                        **aggregate,
                    }
                )
                candidates.append((float(aggregate["selection_score"]), parameters, aggregate))
            if not candidates:
                continue
            selected_score, selected_parameters, selected_aggregate = max(
                candidates,
                key=lambda item: (item[0], -len(json.dumps(item[1], sort_keys=True))),
            )
            selected_segments[spec.name].append(
                {
                    "start": fold["test_start"],
                    "end": fold["test_end"],
                    "factory": spec.factory,
                    "parameters": dict(selected_parameters),
                    "fold": fold["fold"],
                }
            )

            for scenario in scenarios:
                test_config = replace(
                    backtest_config,
                    start_date=fold["test_start"],
                    end_date=fold["test_end"],
                    **dict(scenario.config_overrides),
                )
                run_id = f"fold_{fold['fold']:02d}_{spec.name}_{scenario.name}"
                try:
                    test_result = BacktestEngine(data, config=test_config).run(spec.factory(**selected_parameters))
                    status = "partial_selection" if interrupted else "complete"
                    retained_results[run_id] = test_result
                    fold_rows.append(
                        {
                            **fold,
                            "run_id": run_id,
                            "strategy": spec.name,
                            "scenario": scenario.name,
                            "status": status,
                            "parameters": json.dumps(selected_parameters, sort_keys=True),
                            "inner_selection_score": selected_score,
                            "inner_folds": selected_aggregate["inner_folds"],
                            **test_result.metrics,
                        }
                    )
                except KeyboardInterrupt:
                    if not validation.continue_on_interrupt:
                        raise
                    fold_rows.append(
                        {
                            **fold,
                            "run_id": run_id,
                            "strategy": spec.name,
                            "scenario": scenario.name,
                            "status": "interrupted",
                            "parameters": json.dumps(selected_parameters, sort_keys=True),
                        }
                    )

    for spec in specs:
        segments = selected_segments.get(spec.name, [])
        if not segments:
            continue
        start = min(pd.Timestamp(segment["start"]) for segment in segments)
        end = max(pd.Timestamp(segment["end"]) for segment in segments)
        for scenario in scenarios:
            run_id = f"stitched_{spec.name}_{scenario.name}"
            scheduled = _ScheduledStrategy(
                [
                    (
                        pd.Timestamp(segment["start"]),
                        pd.Timestamp(segment["end"]),
                        spec.factory(**segment["parameters"]),
                    )
                    for segment in segments
                ]
            )
            stitched_config = replace(
                backtest_config,
                start_date=start,
                end_date=end,
                **dict(scenario.config_overrides),
            )
            try:
                result = BacktestEngine(data, config=stitched_config).run(scheduled)
                retained_results[run_id] = result
                stitched_rows.append(
                    {
                        "run_id": run_id,
                        "strategy": spec.name,
                        "scenario": scenario.name,
                        "status": "complete",
                        "start_date": start,
                        "end_date": end,
                        "outer_folds": len(segments),
                        **result.metrics,
                    }
                )
            except KeyboardInterrupt:
                if not validation.continue_on_interrupt:
                    raise
                stitched_rows.append(
                    {
                        "run_id": run_id,
                        "strategy": spec.name,
                        "scenario": scenario.name,
                        "status": "interrupted",
                        "start_date": start,
                        "end_date": end,
                        "outer_folds": len(segments),
                    }
                )

    fold_frame = pd.DataFrame(fold_rows)
    leaderboard = _walk_forward_leaderboard(fold_frame)
    stitched_frame = pd.DataFrame(stitched_rows)
    return WalkForwardResult(
        fold_frame,
        pd.DataFrame(selection_rows),
        leaderboard,
        stitched_frame,
        retained_results,
    )


class _ScheduledStrategy:
    def __init__(self, segments: Sequence[tuple[pd.Timestamp, pd.Timestamp, Any]]) -> None:
        self.segments = list(segments)

    def target_weights(self, context: Any) -> Mapping[str, float]:
        strategy = self._active(context.date)
        method = getattr(strategy, "target_weights", None) if strategy is not None else None
        return dict(method(context) or {}) if callable(method) else {}

    def needs_history(self, date: pd.Timestamp) -> bool:
        strategy = self._active(date)
        method = getattr(strategy, "needs_history", None) if strategy is not None else None
        return bool(method(date)) if callable(method) else strategy is not None

    def orders(self, context: Any) -> Sequence[Any]:
        strategy = self._active(context.date)
        method = getattr(strategy, "orders", None) if strategy is not None else None
        return list(method(context) or []) if callable(method) else []

    def _active(self, value: Any) -> Any | None:
        date = pd.Timestamp(value)
        for start, end, strategy in self.segments:
            if start <= date <= end:
                return strategy
        return None


def _aggregate_inner_selection(metrics: Sequence[Mapping[str, Any]], minimum_round_trips: int) -> dict[str, Any]:
    returns = [float(item.get("total_return") or 0.0) for item in metrics]
    drawdowns = [float(item.get("max_drawdown") or 0.0) for item in metrics]
    turnovers = [float(item.get("annualized_gross_turnover") or 0.0) for item in metrics]
    round_trips = sum(int(item.get("round_trip_count") or 0) for item in metrics)
    compound_return = math.prod(1.0 + value for value in returns) - 1.0 if returns else 0.0
    worst_drawdown = min(drawdowns, default=0.0)
    median_return = float(pd.Series(returns).median()) if returns else 0.0
    trade_penalty = max(0, minimum_round_trips - round_trips) * 0.05
    selection_score = median_return + 0.25 * compound_return - 0.50 * abs(worst_drawdown) - 0.001 * sum(turnovers) - trade_penalty
    return {
        "selection_score": float(selection_score),
        "inner_folds": len(metrics),
        "positive_inner_folds": sum(value > 0 for value in returns),
        "total_return": float(compound_return),
        "median_inner_return": median_return,
        "max_drawdown": worst_drawdown,
        "round_trip_count": round_trips,
        "annualized_gross_turnover": float(pd.Series(turnovers).median()) if turnovers else 0.0,
    }


def price_index_benchmark(frame: pd.DataFrame, *, name: str, initial_capital: float = 50_000.0) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = frame.copy()
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    rows["close"] = pd.to_numeric(rows["close"], errors="coerce")
    rows = rows.dropna(subset=["date", "close"]).sort_values("date")
    if rows.empty:
        raise ValueError(f"benchmark {name} has no valid rows")
    rows["daily_return"] = rows["close"].pct_change().fillna(0.0)
    rows["equity"] = initial_capital * (1.0 + rows["daily_return"]).cumprod()
    peak = rows["equity"].cummax()
    rows["drawdown"] = rows["equity"] / peak - 1.0
    years = max((rows["date"].iloc[-1] - rows["date"].iloc[0]).days / 365.25, 0.0)
    total_return = float(rows["equity"].iloc[-1] / initial_capital - 1.0)
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    volatility = float(rows["daily_return"].iloc[1:].std(ddof=1)) if len(rows) > 2 else 0.0
    metrics = {
        "name": name,
        "initial_capital": initial_capital,
        "final_equity": float(rows["equity"].iloc[-1]),
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": float(rows["drawdown"].min()),
        "annualized_volatility": volatility * math.sqrt(252.0),
        "sharpe": float(rows["daily_return"].iloc[1:].mean() / volatility * math.sqrt(252.0)) if volatility > 0 else None,
    }
    return rows[["date", "close", "equity", "daily_return", "drawdown"]], metrics


def default_stress_scenarios(config: BacktestConfig) -> list[StressScenario]:
    return [
        StressScenario("base"),
        StressScenario(
            "double_costs",
            {
                "brokerage_bps": config.brokerage_bps * 2.0,
                "brokerage_per_share": config.brokerage_per_share * 2.0,
                "other_fees_bps": config.other_fees_bps * 2.0,
                "slippage_bps": max(config.slippage_bps * 2.0, 50.0),
            },
        ),
        StressScenario("delayed_entry", {"signal_delay_sessions": config.signal_delay_sessions + 1}),
        StressScenario("missed_fills", {"missed_fill_probability": 0.10}),
        StressScenario("adverse_slippage", {"slippage_bps": max(config.slippage_bps, 100.0)}),
    ]


def _parameter_combinations(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid)
    values = [list(grid[key]) for key in keys]
    if any(not options for options in values):
        raise ValueError("parameter grid values cannot be empty")
    return [dict(zip(keys, combination)) for combination in product(*values)]


def _selection_score(metrics: Mapping[str, Any], minimum_round_trips: int) -> float:
    cagr = float(metrics.get("cagr") or 0.0)
    drawdown = abs(float(metrics.get("max_drawdown") or 0.0))
    turnover = float(metrics.get("turnover") or 0.0)
    round_trips = int(metrics.get("round_trip_count") or 0)
    trade_penalty = max(0, minimum_round_trips - round_trips) * 0.05
    return cagr - 0.50 * drawdown - 0.001 * turnover - trade_penalty


def _walk_forward_leaderboard(folds: pd.DataFrame) -> pd.DataFrame:
    if folds.empty:
        return pd.DataFrame()
    complete = folds[folds["status"].isin(["complete", "partial_selection"])].copy()
    if complete.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (strategy, scenario), group in complete.groupby(["strategy", "scenario"], sort=True):
        returns = pd.to_numeric(group["total_return"], errors="coerce").dropna()
        rows.append(
            {
                "strategy": strategy,
                "scenario": scenario,
                "folds": int(len(group)),
                "positive_folds": int((returns > 0).sum()),
                "compound_test_return": float((1.0 + returns).prod() - 1.0) if not returns.empty else None,
                "median_test_return": float(returns.median()) if not returns.empty else None,
                "median_cagr": float(pd.to_numeric(group["cagr"], errors="coerce").median()),
                "worst_max_drawdown": float(pd.to_numeric(group["max_drawdown"], errors="coerce").min()),
                "total_round_trips": int(pd.to_numeric(group["round_trip_count"], errors="coerce").fillna(0).sum()),
                "median_sharpe": float(pd.to_numeric(group["sharpe"], errors="coerce").median()),
                "median_turnover": float(pd.to_numeric(group["turnover"], errors="coerce").median()),
                "maximum_best_symbol_profit_fraction": float(pd.to_numeric(group["best_symbol_profit_fraction"], errors="coerce").max()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["scenario", "compound_test_return", "worst_max_drawdown"],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)


def _issue(rows: list[dict[str, Any]], symbol: str, severity: str, code: str, message: str) -> None:
    rows.append({"symbol": symbol, "severity": severity, "code": code, "message": message})


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    cleaned = frame.astype(object).where(pd.notna(frame), None)
    records = cleaned.to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if isinstance(value, (pd.Timestamp, pd.Period)):
                record[key] = str(value)
            elif hasattr(value, "item"):
                try:
                    record[key] = value.item()
                except (TypeError, ValueError):
                    pass
    return records


__all__ = [
    "DataAuditResult",
    "DataProvenance",
    "MembershipSchedule",
    "StrategySpec",
    "StressScenario",
    "WalkForwardConfig",
    "WalkForwardResult",
    "audit_market_data",
    "default_stress_scenarios",
    "generate_walk_forward_folds",
    "price_index_benchmark",
    "run_walk_forward_validation",
]
