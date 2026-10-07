from __future__ import annotations

import argparse
import sys
from itertools import product
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ahl_api.backtest import BacktestConfig, BacktestEngine
from ahl_api.client import AHL
from ahl_api.strategies import (
    LiquidityFilteredMomentumStrategy,
    MomentumTrendFilterStrategy,
    PullbackDcaStrategy,
)


DEFAULT_SYMBOLS = [
    "OGDC",
    "FFC",
    "HBL",
    "MCB",
    "UBL",
    "MEBL",
    "LUCK",
    "HUBC",
    "ENGRO",
    "MARI",
    "PPL",
    "PSO",
    "SYS",
    "BAFL",
    "BAHL",
    "EFERT",
    "POL",
    "NBP",
    "DGKC",
    "CHCC",
    "MTL",
    "INDU",
    "SNGP",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest built-in AHL strategies on PSX daily history.")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--initial-capital", type=float, default=1_000_000)
    parser.add_argument("--min-avg-traded-value", type=float, default=0.0)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    parser.add_argument("--benchmark-symbols", nargs="*", default=None)
    parser.add_argument("--allow-stale-symbols", action="store_true")
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--min-rows", type=int, default=120)
    parser.add_argument("--run-sweeps", action="store_true")
    parser.add_argument("--sweep-mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--sweep-strategies", nargs="*", choices=["momentum", "trend", "dca"], default=["momentum", "trend", "dca"])
    parser.add_argument("--sweep-details", action="store_true")
    parser.add_argument("--continue-on-interrupt", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--walk-forward-strategy", choices=["momentum", "trend", "dca"], default="dca")
    parser.add_argument("--train-months", type=int, default=36)
    parser.add_argument("--test-months", type=int, default=12)
    parser.add_argument("--step-months", type=int, default=12)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


class BuyAndHoldStrategy:
    def __init__(self, weights: dict[str, float]):
        self.weights = {symbol.upper(): weight for symbol, weight in weights.items()}
        self.sent = False

    def target_weights(self, context):
        if self.sent:
            return {}
        self.sent = True
        return self.weights


class StartDateStrategy:
    def __init__(self, strategy, start_date: pd.Timestamp):
        self.strategy = strategy
        self.start_date = pd.Timestamp(start_date)

    def target_weights(self, context):
        if context.date < self.start_date:
            return {}
        method = getattr(self.strategy, "target_weights", None)
        if not callable(method):
            return {}
        return method(context)

    def orders(self, context):
        if context.date < self.start_date:
            return []
        method = getattr(self.strategy, "orders", None)
        if not callable(method):
            return []
        return method(context)


def fetch_data(
    symbols: list[str],
    years: int,
    *,
    min_rows: int,
    min_coverage: float,
    allow_stale_symbols: bool,
) -> dict[str, pd.DataFrame]:
    client = AHL(audit_enabled=False)
    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            rows = client.fetch_historical_daily(symbol, years=years)
        except Exception as exc:
            print(f"skip {symbol}: fetch failed: {exc}")
            continue
        if len(rows) < min_rows:
            print(f"skip {symbol}: only {len(rows)} rows")
            continue
        data[symbol.upper()] = pd.DataFrame(rows)
        first = data[symbol.upper()]["date"].iloc[0]
        last = data[symbol.upper()]["date"].iloc[-1]
        print(f"loaded {symbol.upper()}: {len(rows)} rows ({first} to {last})")
    if allow_stale_symbols or not data:
        return data

    latest = max(pd.to_datetime(frame["date"]).max() for frame in data.values())
    max_rows = max(len(frame) for frame in data.values())
    filtered: dict[str, pd.DataFrame] = {}
    for symbol, frame in data.items():
        last = pd.to_datetime(frame["date"]).max()
        coverage = len(frame) / max_rows if max_rows else 0
        if last != latest:
            print(f"drop {symbol}: stale last date {last.date()} vs {latest.date()}")
            continue
        if coverage < min_coverage:
            print(f"drop {symbol}: coverage {coverage:.1%} below {min_coverage:.1%}")
            continue
        filtered[symbol] = frame
    return filtered


def summarize(name: str, result, *, category: str, params: str = "", status: str = "complete", disclaimer: str = "") -> dict[str, object]:
    metrics = result.metrics
    return {
        "strategy": name,
        "category": category,
        "status": status,
        "disclaimer": disclaimer,
        "params": params,
        "final_equity": metrics.get("final_equity"),
        "total_return_pct": _pct(metrics.get("total_return")),
        "cagr_pct": _pct(metrics.get("cagr")),
        "max_drawdown_pct": _pct(metrics.get("max_drawdown")),
        "trade_count": metrics.get("trade_count"),
        "win_rate_pct": _pct(metrics.get("win_rate")),
        "turnover": metrics.get("turnover"),
        "cash_utilization_pct": _pct(metrics.get("cash_utilization")),
    }


def _pct(value):
    if value is None:
        return None
    return float(value) * 100


def equal_weight(symbols: list[str], exposure: float = 1.0) -> dict[str, float]:
    symbols = [symbol.upper() for symbol in symbols]
    if not symbols:
        return {}
    return {symbol: exposure / len(symbols) for symbol in symbols}


def interrupted_summary(name: str, *, category: str, params: str = "", disclaimer: str) -> dict[str, object]:
    return {
        "strategy": name,
        "category": category,
        "status": "interrupted",
        "disclaimer": disclaimer,
        "params": params,
        "final_equity": None,
        "total_return_pct": None,
        "cagr_pct": None,
        "max_drawdown_pct": None,
        "trade_count": None,
        "win_rate_pct": None,
        "turnover": None,
        "cash_utilization_pct": None,
    }


def run_strategy(
    engine: BacktestEngine,
    output_dir: Path,
    summaries: list[dict[str, object]],
    name: str,
    strategy,
    *,
    category: str,
    params: str = "",
    continue_on_interrupt: bool = False,
) -> None:
    print(f"running {name}...")
    try:
        result = engine.run(strategy)
    except KeyboardInterrupt:
        if not continue_on_interrupt:
            raise
        disclaimer = "Interrupted by Ctrl+C; this single strategy result is incomplete and has no partial equity curve."
        summaries.append(interrupted_summary(name, category=category, params=params, disclaimer=disclaimer))
        print(f"interrupted {name}; continuing because --continue-on-interrupt is enabled")
        return
    result.write_report(output_dir / name, prefix=name)
    summaries.append(summarize(name, result, category=category, params=params))


def run_sweeps(
    engine: BacktestEngine,
    config: BacktestConfig,
    output_dir: Path,
    min_avg_traded_value: float,
    summaries: list[dict[str, object]],
    *,
    mode: str,
    selected: set[str],
    include_details: bool,
    continue_on_interrupt: bool,
) -> None:
    sweep_dir = output_dir / "sweeps"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    if mode == "quick":
        sweep_specs = [
            (
                "momentum",
                "sweep_liquidity_momentum",
                LiquidityFilteredMomentumStrategy,
                {
                    "top_n": [3, 5],
                    "momentum_lookback": [120, 252],
                    "liquidity_lookback": [20],
                    "min_avg_traded_value": [min_avg_traded_value],
                    "rebalance": ["monthly"],
                },
            ),
            (
                "trend",
                "sweep_momentum_trend",
                MomentumTrendFilterStrategy,
                {
                    "top_n": [3, 5],
                    "momentum_lookback": [120, 252],
                    "liquidity_lookback": [20],
                    "min_avg_traded_value": [min_avg_traded_value],
                    "trend_sma": [100, 200],
                    "rebalance": ["monthly"],
                },
            ),
            (
                "dca",
                "sweep_pullback_dca",
                PullbackDcaStrategy,
                {
                    "top_n": [3, 5],
                    "momentum_lookback": [120],
                    "liquidity_lookback": [20],
                    "min_avg_traded_value": [min_avg_traded_value],
                    "trend_sma": [100],
                    "dip_ema": [20, 50],
                    "dip_thresholds": [(0.03, 0.06, 0.09), (0.05, 0.10)],
                    "take_profit_above_ema": [0.06],
                    "max_position_weight": [0.20, 0.30],
                    "universe_rebalance": ["monthly"],
                },
            ),
        ]
    else:
        sweep_specs = [
            (
                "momentum",
                "sweep_liquidity_momentum",
                LiquidityFilteredMomentumStrategy,
                {
                    "top_n": [3, 5],
                    "momentum_lookback": [60, 120, 252],
                    "liquidity_lookback": [20],
                    "min_avg_traded_value": [min_avg_traded_value],
                    "rebalance": ["weekly", "monthly"],
                },
            ),
            (
                "trend",
                "sweep_momentum_trend",
                MomentumTrendFilterStrategy,
                {
                    "top_n": [3, 5],
                    "momentum_lookback": [60, 120, 252],
                    "liquidity_lookback": [20],
                    "min_avg_traded_value": [min_avg_traded_value],
                    "trend_sma": [50, 100, 200],
                    "rebalance": ["weekly", "monthly"],
                },
            ),
            (
                "dca",
                "sweep_pullback_dca",
                PullbackDcaStrategy,
                {
                    "top_n": [3, 5],
                    "momentum_lookback": [60, 120],
                    "liquidity_lookback": [20],
                    "min_avg_traded_value": [min_avg_traded_value],
                    "trend_sma": [100, 200],
                    "dip_ema": [20, 50],
                    "dip_thresholds": [(0.03, 0.06, 0.09), (0.05, 0.10), (0.02, 0.04, 0.06)],
                    "take_profit_above_ema": [0.04, 0.06, 0.10],
                    "max_position_weight": [0.20, 0.30],
                    "universe_rebalance": ["monthly"],
                },
            ),
        ]

    for key, name, factory, grid in sweep_specs:
        if key not in selected:
            continue
        run_count = _grid_size(grid)
        print(f"running {name} ({run_count} runs, mode={mode})...")
        sweep_output_dir = sweep_dir / name
        sweep_output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = sweep_output_dir / f"{name}_summary.csv"
        ranked_path = sweep_output_dir / f"{name}_ranked.csv"
        summary_rows: list[dict[str, object]] = []
        interrupted = False

        combinations = _parameter_combinations(grid)
        for run_index, strategy_kwargs in enumerate(combinations):
            run_id = f"run_{run_index:04d}"
            print(f"  {name} {run_index + 1}/{run_count}: {_compact_params(strategy_kwargs)}")
            try:
                strategy = factory(**strategy_kwargs)
                result = BacktestEngine(engine.data, config=config).run(strategy)
            except KeyboardInterrupt:
                if not continue_on_interrupt:
                    raise
                interrupted = True
                print(f"interrupted {name} at {run_id}; keeping {len(summary_rows)} completed runs and moving on")
                break

            if include_details:
                result.write_report(sweep_output_dir / run_id, prefix=run_id)
            summary_rows.append(
                {
                    "run_index": run_index,
                    "run_id": run_id,
                    **strategy_kwargs,
                    **result.metrics,
                    "status": "complete",
                    "disclaimer": "",
                }
            )
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

        if not summary_rows:
            disclaimer = "Interrupted by Ctrl+C before any sweep run completed."
            summaries.append(interrupted_summary(name, category="sweep_best", disclaimer=disclaimer))
            continue

        summary_frame = pd.DataFrame(summary_rows)
        ranked = summary_frame.sort_values(["final_equity", "max_drawdown"], ascending=[False, False])
        summary_frame.to_csv(summary_path, index=False)
        ranked.to_csv(sweep_dir / name / f"{name}_ranked.csv", index=False)
        best = ranked.iloc[0].to_dict()
        status = "partial" if interrupted else "complete"
        disclaimer = "Interrupted by Ctrl+C; best result is only among completed sweep runs." if interrupted else ""
        summaries.append(
            {
                "strategy": name,
                "category": "sweep_best",
                "status": status,
                "disclaimer": disclaimer,
                "params": _format_params(best),
                "final_equity": best.get("final_equity"),
                "total_return_pct": _pct(best.get("total_return")),
                "cagr_pct": _pct(best.get("cagr")),
                "max_drawdown_pct": _pct(best.get("max_drawdown")),
                "trade_count": best.get("trade_count"),
                "win_rate_pct": _pct(best.get("win_rate")),
                "turnover": best.get("turnover"),
                "cash_utilization_pct": _pct(best.get("cash_utilization")),
            }
        )


def run_walk_forward(
    data: dict[str, pd.DataFrame],
    config: BacktestConfig,
    output_dir: Path,
    *,
    strategy_key: str,
    mode: str,
    min_avg_traded_value: float,
    train_months: int,
    test_months: int,
    step_months: int,
    continue_on_interrupt: bool,
) -> pd.DataFrame:
    if train_months <= 0 or test_months <= 0 or step_months <= 0:
        raise ValueError("train-months, test-months, and step-months must be positive")

    specs = sweep_specs_for_mode(mode, min_avg_traded_value)
    spec_by_key = {key: (name, factory, grid) for key, name, factory, grid in specs}
    name, factory, grid = spec_by_key[strategy_key]
    calendar = combined_calendar(data)
    folds = walk_forward_folds(calendar, train_months=train_months, test_months=test_months, step_months=step_months)
    output = output_dir / "walk_forward" / name
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for fold_index, fold in enumerate(folds):
        print(
            f"walk-forward {name} fold {fold_index + 1}/{len(folds)}: "
            f"train {fold['train_start'].date()}..{fold['train_end'].date()}, "
            f"test {fold['test_start'].date()}..{fold['test_end'].date()}"
        )
        train_data = slice_data(data, fold["train_start"], fold["train_end"])
        test_data = slice_data(data, fold["test_start"], fold["test_end"])
        train_summary = run_grid(
            train_data,
            config,
            factory,
            grid,
            output / f"fold_{fold_index:02d}_train",
            f"fold_{fold_index:02d}_train",
            continue_on_interrupt=continue_on_interrupt,
        )
        if train_summary.empty:
            rows.append({**fold, "fold": fold_index, "status": "interrupted", "disclaimer": "No training runs completed."})
            continue

        train_ranked = train_summary.sort_values(["final_equity", "max_drawdown"], ascending=[False, False])
        train_ranked.to_csv(output / f"fold_{fold_index:02d}_train_ranked.csv", index=False)
        best = train_ranked.iloc[0].to_dict()
        params = {key: best[key] for key in grid if key in best}
        params = normalize_grid_params(params, grid)
        status = "partial" if (train_summary["status"] == "partial").any() else "complete"
        disclaimer = "Training sweep was interrupted; chosen params are best among completed runs." if status == "partial" else ""

        print(f"  best train params: {_compact_params(params)}")
        try:
            evaluation_data = slice_data(data, fold["train_start"], fold["test_end"])
            test_result = BacktestEngine(evaluation_data, config=config).run(StartDateStrategy(factory(**params), fold["test_start"]))
        except KeyboardInterrupt:
            if not continue_on_interrupt:
                raise
            rows.append(
                {
                    **fold,
                    "fold": fold_index,
                    "status": "interrupted",
                    "disclaimer": "Interrupted during test evaluation.",
                    "params": _compact_params(params),
                }
            )
            continue

        test_result.write_report(output / f"fold_{fold_index:02d}_test", prefix=f"fold_{fold_index:02d}_test")
        test_metrics = metrics_for_window(test_result, fold["test_start"], fold["test_end"], config.initial_capital)
        rows.append(
            {
                **fold,
                "fold": fold_index,
                "status": status,
                "disclaimer": disclaimer,
                "params": _compact_params(params),
                "train_final_equity": best.get("final_equity"),
                "train_total_return_pct": _pct(best.get("total_return")),
                "train_max_drawdown_pct": _pct(best.get("max_drawdown")),
                "test_final_equity": test_metrics.get("final_equity"),
                "test_total_return_pct": _pct(test_metrics.get("total_return")),
                "test_cagr_pct": _pct(test_metrics.get("cagr")),
                "test_max_drawdown_pct": _pct(test_metrics.get("max_drawdown")),
                "test_trade_count": test_metrics.get("trade_count"),
                "test_win_rate_pct": _pct(test_metrics.get("win_rate")),
                "test_cash_utilization_pct": _pct(test_metrics.get("cash_utilization")),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(output / f"{name}_walk_forward.csv", index=False)
    return result


def sweep_specs_for_mode(mode: str, min_avg_traded_value: float):
    if mode == "quick":
        return [
            (
                "momentum",
                "sweep_liquidity_momentum",
                LiquidityFilteredMomentumStrategy,
                {
                    "top_n": [3, 5],
                    "momentum_lookback": [120, 252],
                    "liquidity_lookback": [20],
                    "min_avg_traded_value": [min_avg_traded_value],
                    "rebalance": ["monthly"],
                },
            ),
            (
                "trend",
                "sweep_momentum_trend",
                MomentumTrendFilterStrategy,
                {
                    "top_n": [3, 5],
                    "momentum_lookback": [120, 252],
                    "liquidity_lookback": [20],
                    "min_avg_traded_value": [min_avg_traded_value],
                    "trend_sma": [100, 200],
                    "rebalance": ["monthly"],
                },
            ),
            (
                "dca",
                "sweep_pullback_dca",
                PullbackDcaStrategy,
                {
                    "top_n": [3, 5],
                    "momentum_lookback": [120],
                    "liquidity_lookback": [20],
                    "min_avg_traded_value": [min_avg_traded_value],
                    "trend_sma": [100],
                    "dip_ema": [20, 50],
                    "dip_thresholds": [(0.03, 0.06, 0.09), (0.05, 0.10)],
                    "take_profit_above_ema": [0.06],
                    "max_position_weight": [0.20, 0.30],
                    "universe_rebalance": ["monthly"],
                },
            ),
        ]
    return [
        (
            "momentum",
            "sweep_liquidity_momentum",
            LiquidityFilteredMomentumStrategy,
            {
                "top_n": [3, 5],
                "momentum_lookback": [60, 120, 252],
                "liquidity_lookback": [20],
                "min_avg_traded_value": [min_avg_traded_value],
                "rebalance": ["weekly", "monthly"],
            },
        ),
        (
            "trend",
            "sweep_momentum_trend",
            MomentumTrendFilterStrategy,
            {
                "top_n": [3, 5],
                "momentum_lookback": [60, 120, 252],
                "liquidity_lookback": [20],
                "min_avg_traded_value": [min_avg_traded_value],
                "trend_sma": [50, 100, 200],
                "rebalance": ["weekly", "monthly"],
            },
        ),
        (
            "dca",
            "sweep_pullback_dca",
            PullbackDcaStrategy,
            {
                "top_n": [3, 5],
                "momentum_lookback": [60, 120],
                "liquidity_lookback": [20],
                "min_avg_traded_value": [min_avg_traded_value],
                "trend_sma": [100, 200],
                "dip_ema": [20, 50],
                "dip_thresholds": [(0.03, 0.06, 0.09), (0.05, 0.10), (0.02, 0.04, 0.06)],
                "take_profit_above_ema": [0.04, 0.06, 0.10],
                "max_position_weight": [0.20, 0.30],
                "universe_rebalance": ["monthly"],
            },
        ),
    ]


def run_grid(
    data: dict[str, pd.DataFrame],
    config: BacktestConfig,
    factory,
    grid: dict[str, list[object]],
    output_dir: Path,
    prefix: str,
    *,
    continue_on_interrupt: bool,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    combinations = _parameter_combinations(grid)
    for run_index, strategy_kwargs in enumerate(combinations):
        print(f"    train run {run_index + 1}/{len(combinations)}: {_compact_params(strategy_kwargs)}")
        try:
            result = BacktestEngine(data, config=config).run(factory(**strategy_kwargs))
        except KeyboardInterrupt:
            if not continue_on_interrupt:
                raise
            break
        rows.append(
            {
                "run_index": run_index,
                "run_id": f"run_{run_index:04d}",
                **strategy_kwargs,
                **result.metrics,
                "status": "complete",
                "disclaimer": "",
            }
        )
        pd.DataFrame(rows).to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    return pd.DataFrame(rows)


def combined_calendar(data: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted({date for frame in data.values() for date in pd.to_datetime(frame["date"])}))


def walk_forward_folds(calendar: pd.DatetimeIndex, *, train_months: int, test_months: int, step_months: int) -> list[dict[str, pd.Timestamp]]:
    folds: list[dict[str, pd.Timestamp]] = []
    if calendar.empty:
        return folds
    anchor = calendar.min()
    last = calendar.max()
    while True:
        train_start = anchor
        train_end_target = train_start + pd.DateOffset(months=train_months)
        test_start_target = train_end_target
        test_end_target = test_start_target + pd.DateOffset(months=test_months)
        if test_start_target > last:
            break
        train_dates = calendar[(calendar >= train_start) & (calendar < train_end_target)]
        test_dates = calendar[(calendar >= test_start_target) & (calendar < test_end_target)]
        if not train_dates.empty and not test_dates.empty:
            folds.append(
                {
                    "train_start": pd.Timestamp(train_dates.min()),
                    "train_end": pd.Timestamp(train_dates.max()),
                    "test_start": pd.Timestamp(test_dates.min()),
                    "test_end": pd.Timestamp(test_dates.max()),
                }
            )
        anchor = anchor + pd.DateOffset(months=step_months)
        if anchor >= last:
            break
    return folds


def slice_data(data: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    sliced: dict[str, pd.DataFrame] = {}
    for symbol, frame in data.items():
        dates = pd.to_datetime(frame["date"])
        subset = frame[(dates >= start) & (dates <= end)].copy()
        if not subset.empty:
            sliced[symbol] = subset
    return sliced


def normalize_grid_params(params: dict[str, object], grid: dict[str, list[object]]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in params.items():
        for candidate in grid[key]:
            if str(candidate) == str(value):
                normalized[key] = candidate
                break
        else:
            normalized[key] = value
    return normalized


def metrics_for_window(result, start: pd.Timestamp, end: pd.Timestamp, initial_capital: float) -> dict[str, object]:
    curve = result.equity_curve.copy()
    curve["date"] = pd.to_datetime(curve["date"])
    window = curve[(curve["date"] >= start) & (curve["date"] <= end)].copy()
    if window.empty:
        return {}

    final_equity = float(window["equity"].iloc[-1])
    total_return = final_equity / initial_capital - 1.0
    years = max((pd.Timestamp(window["date"].iloc[-1]) - pd.Timestamp(window["date"].iloc[0])).days / 365.25, 0.0)
    cagr = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 else 0.0
    peak = window["equity"].cummax()
    max_drawdown = float((window["equity"] / peak - 1.0).min())
    cash_utilization = float((window["invested_value"] / window["equity"].replace(0, pd.NA)).fillna(0.0).mean())

    trades = result.trades.copy()
    trade_count = 0
    win_rate = None
    if not trades.empty:
        trades["fill_date"] = pd.to_datetime(trades["fill_date"])
        trades = trades[(trades["fill_date"] >= start) & (trades["fill_date"] <= end)]
        trade_count = int(len(trades))
        sells = trades[trades["side"] == "sell"]
        if not sells.empty:
            win_rate = float((sells["realized_pnl"] > 0).mean())

    return {
        "final_equity": final_equity,
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "trade_count": trade_count,
        "win_rate": win_rate,
        "cash_utilization": cash_utilization,
    }


def _parameter_combinations(grid: dict[str, list[object]]) -> list[dict[str, object]]:
    keys = list(grid)
    values = [list(grid[key]) for key in keys]
    return [dict(zip(keys, combination)) for combination in product(*values)]


def _grid_size(grid: dict[str, list[object]]) -> int:
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def _compact_params(params: dict[str, object]) -> str:
    return ", ".join(f"{key}={value}" for key, value in params.items())


def _format_params(row: dict[str, object]) -> str:
    excluded = {
        "run_index",
        "run_id",
        "initial_capital",
        "final_equity",
        "total_return",
        "cagr",
        "max_drawdown",
        "trade_count",
        "win_rate",
        "turnover",
        "cash_utilization",
        "status",
        "disclaimer",
    }
    parts = []
    for key, value in row.items():
        if key in excluded:
            continue
        if _is_missing(value):
            continue
        parts.append(f"{key}={value}")
    return "; ".join(parts)


def _is_missing(value) -> bool:
    if isinstance(value, (tuple, list, dict)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def print_console_summary(summary: pd.DataFrame) -> None:
    compact_columns = [
        "strategy",
        "category",
        "status",
        "final_equity",
        "total_return_pct",
        "cagr_pct",
        "max_drawdown_pct",
        "trade_count",
        "win_rate_pct",
        "cash_utilization_pct",
    ]
    available = [column for column in compact_columns if column in summary.columns]
    display = summary[available].copy()
    for column in [
        "final_equity",
        "total_return_pct",
        "cagr_pct",
        "max_drawdown_pct",
        "win_rate_pct",
        "cash_utilization_pct",
    ]:
        if column in display.columns:
            display[column] = display[column].map(_format_number)
    print(display.to_string(index=False, max_colwidth=32))

    if "params" in summary.columns:
        sweep_params = summary[(summary.get("category") == "sweep_best") & summary["params"].notna()]
        if not sweep_params.empty:
            print()
            print("Best sweep parameters:")
            for _, row in sweep_params.iterrows():
                print(f"- {row['strategy']}: {row['params']}")

    if "disclaimer" in summary.columns:
        disclaimers = summary[summary["disclaimer"].fillna("").astype(str) != ""]
        if not disclaimers.empty:
            print()
            print("Disclaimers:")
            for _, row in disclaimers.iterrows():
                print(f"- {row['strategy']}: {row['disclaimer']}")


def print_walk_forward_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        print("No walk-forward folds were generated.")
        return
    display_columns = [
        "fold",
        "status",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "test_final_equity",
        "test_total_return_pct",
        "test_cagr_pct",
        "test_max_drawdown_pct",
        "test_trade_count",
        "test_cash_utilization_pct",
    ]
    available = [column for column in display_columns if column in summary.columns]
    display = summary[available].copy()
    for column in [
        "test_final_equity",
        "test_total_return_pct",
        "test_cagr_pct",
        "test_max_drawdown_pct",
        "test_cash_utilization_pct",
    ]:
        if column in display.columns:
            display[column] = display[column].map(_format_number)
    for column in ["train_start", "train_end", "test_start", "test_end"]:
        if column in display.columns:
            display[column] = pd.to_datetime(display[column]).dt.date.astype(str)
    print(display.to_string(index=False, max_colwidth=32))

    if "params" in summary.columns:
        print()
        print("Walk-forward chosen parameters:")
        for _, row in summary.iterrows():
            print(f"- fold {row.get('fold')}: {row.get('params', '')}")

    if "test_total_return_pct" in summary.columns:
        valid = pd.to_numeric(summary["test_total_return_pct"], errors="coerce").dropna()
        drawdown = pd.to_numeric(summary.get("test_max_drawdown_pct"), errors="coerce").dropna()
        if not valid.empty:
            print()
            print(f"Average test return: {valid.mean():,.2f}%")
            print(f"Median test return: {valid.median():,.2f}%")
        if not drawdown.empty:
            print(f"Worst test drawdown: {drawdown.min():,.2f}%")


def _format_number(value) -> str:
    if _is_missing(value):
        return ""
    return f"{float(value):,.2f}"


def main() -> int:
    args = parse_args()
    selected_symbols = [symbol.upper() for symbol in args.symbols]
    selected_benchmarks = [symbol.upper() for symbol in (args.benchmark_symbols or selected_symbols)]
    fetch_symbols = list(dict.fromkeys(selected_symbols + selected_benchmarks))
    data = fetch_data(
        fetch_symbols,
        args.years,
        min_rows=args.min_rows,
        min_coverage=args.min_coverage,
        allow_stale_symbols=args.allow_stale_symbols,
    )
    if len(data) < 2:
        raise SystemExit("not enough symbols loaded to run multi-symbol strategy backtests")

    config = BacktestConfig(initial_capital=args.initial_capital)
    engine = BacktestEngine(data, config=config)
    strategies = {
        "liquidity_momentum": LiquidityFilteredMomentumStrategy(
            top_n=args.top_n,
            momentum_lookback=120,
            liquidity_lookback=20,
            min_avg_traded_value=args.min_avg_traded_value,
            rebalance="monthly",
        ),
        "momentum_trend_sma100": MomentumTrendFilterStrategy(
            top_n=args.top_n,
            momentum_lookback=120,
            liquidity_lookback=20,
            min_avg_traded_value=args.min_avg_traded_value,
            trend_sma=100,
            rebalance="monthly",
        ),
        "momentum_trend_sma200": MomentumTrendFilterStrategy(
            top_n=args.top_n,
            momentum_lookback=120,
            liquidity_lookback=20,
            min_avg_traded_value=args.min_avg_traded_value,
            trend_sma=200,
            rebalance="monthly",
        ),
        "pullback_dca": PullbackDcaStrategy(
            top_n=args.top_n,
            momentum_lookback=120,
            liquidity_lookback=20,
            min_avg_traded_value=args.min_avg_traded_value,
            trend_sma=100,
            dip_ema=20,
            dip_thresholds=(0.03, 0.06, 0.09),
            take_profit_above_ema=0.06,
            max_position_weight=0.20,
            universe_rebalance="monthly",
        ),
    }

    output_dir = Path(args.output_dir or Path("artifacts") / "reports" / "backtests" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    benchmark_symbols = [symbol for symbol in selected_benchmarks if symbol in data]
    if benchmark_symbols:
        run_strategy(
            engine,
            output_dir,
            summaries,
            "benchmark_equal_weight_buy_hold",
            BuyAndHoldStrategy(equal_weight(benchmark_symbols)),
            category="benchmark",
            params=f"symbols={','.join(benchmark_symbols)}",
            continue_on_interrupt=args.continue_on_interrupt,
        )
    for symbol in benchmark_symbols:
        run_strategy(
            engine,
            output_dir,
            summaries,
            f"benchmark_buy_hold_{symbol}",
            BuyAndHoldStrategy({symbol: 1.0}),
            category="benchmark",
            params=f"symbol={symbol}",
            continue_on_interrupt=args.continue_on_interrupt,
        )
    for name, strategy in strategies.items():
        run_strategy(engine, output_dir, summaries, name, strategy, category="strategy", continue_on_interrupt=args.continue_on_interrupt)

    if args.run_sweeps:
        run_sweeps(
            engine,
            config,
            output_dir,
            args.min_avg_traded_value,
            summaries,
            mode=args.sweep_mode,
            selected=set(args.sweep_strategies),
            include_details=args.sweep_details,
            continue_on_interrupt=args.continue_on_interrupt,
        )

    if args.walk_forward:
        wf = run_walk_forward(
            data,
            config,
            output_dir,
            strategy_key=args.walk_forward_strategy,
            mode=args.sweep_mode,
            min_avg_traded_value=args.min_avg_traded_value,
            train_months=args.train_months,
            test_months=args.test_months,
            step_months=args.step_months,
            continue_on_interrupt=args.continue_on_interrupt,
        )
        print()
        print("Walk-forward validation:")
        print_walk_forward_summary(wf)

    summary = pd.DataFrame(summaries).sort_values("final_equity", ascending=False, na_position="last")
    summary_path = output_dir / "strategy_summary.csv"
    summary.to_csv(summary_path, index=False)
    print()
    print_console_summary(summary)
    print()
    print(f"reports: {output_dir.resolve()}")
    print(f"summary: {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
