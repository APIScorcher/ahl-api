"""Statistical diagnostics and charts for backtest research."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ahl_api.backtest import BacktestConfig, BacktestEngine, BacktestResult


def block_bootstrap_returns(
    equity_curve: pd.DataFrame,
    *,
    simulations: int = 2_000,
    block_size: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Moving-block bootstrap of observed daily strategy returns."""

    if simulations <= 0 or block_size <= 0:
        raise ValueError("simulations and block_size must be positive")
    returns = pd.to_numeric(equity_curve.get("daily_return"), errors="coerce").dropna().to_numpy(dtype=float)
    if len(returns) > 1:
        returns = returns[1:]
    if len(returns) < block_size:
        raise ValueError("equity curve is too short for the requested bootstrap block")

    rng = np.random.default_rng(seed)
    starts = np.arange(0, len(returns) - block_size + 1)
    blocks_needed = int(math.ceil(len(returns) / block_size))
    rows: list[dict[str, float | int]] = []
    for simulation in range(simulations):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sampled = np.concatenate([returns[start : start + block_size] for start in chosen])[: len(returns)]
        path = np.cumprod(1.0 + sampled)
        peaks = np.maximum.accumulate(path)
        drawdowns = path / peaks - 1.0
        rows.append(
            {
                "simulation": simulation,
                "terminal_return": float(path[-1] - 1.0),
                "max_drawdown": float(drawdowns.min()),
                "worst_day": float(sampled.min()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_summary(paths: pd.DataFrame) -> dict[str, float]:
    if paths.empty:
        return {}
    terminal = pd.to_numeric(paths["terminal_return"], errors="coerce").dropna()
    drawdown = pd.to_numeric(paths["max_drawdown"], errors="coerce").dropna()
    return {
        "simulations": int(len(paths)),
        "probability_positive": float((terminal > 0).mean()),
        "terminal_return_p05": float(terminal.quantile(0.05)),
        "terminal_return_median": float(terminal.median()),
        "terminal_return_p95": float(terminal.quantile(0.95)),
        "max_drawdown_p05": float(drawdown.quantile(0.05)),
        "max_drawdown_median": float(drawdown.median()),
    }


def performance_by_market_regime(
    equity_curve: pd.DataFrame,
    market_frame: pd.DataFrame,
    *,
    sma_window: int = 200,
) -> pd.DataFrame:
    """Split strategy returns by whether the market closed above its trailing SMA."""

    if sma_window <= 0:
        raise ValueError("sma_window must be positive")
    market = market_frame[["date", "close"]].copy()
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    market["market_close"] = pd.to_numeric(market["close"], errors="coerce")
    market["market_sma"] = market["market_close"].rolling(sma_window, min_periods=sma_window).mean()
    market["regime"] = np.where(market["market_close"] > market["market_sma"], "risk_on", "risk_off")

    strategy = equity_curve[["date", "daily_return"]].copy()
    strategy["date"] = pd.to_datetime(strategy["date"]).dt.normalize()
    merged = strategy.merge(market[["date", "market_close", "market_sma", "regime"]], on="date", how="inner")
    merged = merged.dropna(subset=["daily_return", "market_sma"])
    rows: list[dict[str, Any]] = []
    for regime, group in merged.groupby("regime", sort=True):
        returns = pd.to_numeric(group["daily_return"], errors="coerce").dropna()
        volatility = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
        rows.append(
            {
                "regime": regime,
                "sessions": int(len(returns)),
                "compound_return": float((1.0 + returns).prod() - 1.0),
                "annualized_return": float((1.0 + returns.mean()) ** 252 - 1.0) if not returns.empty else None,
                "annualized_volatility": volatility * math.sqrt(252.0),
                "sharpe_zero_rate": float(returns.mean() / volatility * math.sqrt(252.0)) if volatility > 0 else None,
                "positive_session_rate": float((returns > 0).mean()) if not returns.empty else None,
            }
        )
    return pd.DataFrame(rows)


def parameter_stability_table(selection_runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate train scores by exact parameter set across folds."""

    if selection_runs.empty:
        return pd.DataFrame()
    if "selection_level" in selection_runs:
        aggregates = selection_runs[selection_runs["selection_level"] == "aggregate"]
        if not aggregates.empty:
            selection_runs = aggregates
    rows: list[dict[str, Any]] = []
    for (strategy, parameters), group in selection_runs.groupby(["strategy", "parameters"], sort=True):
        scores = pd.to_numeric(group["selection_score"], errors="coerce").dropna()
        returns = pd.to_numeric(group.get("total_return"), errors="coerce").dropna()
        rows.append(
            {
                "strategy": strategy,
                "parameters": parameters,
                "folds_evaluated": int(len(group)),
                "mean_selection_score": float(scores.mean()) if not scores.empty else None,
                "median_selection_score": float(scores.median()) if not scores.empty else None,
                "worst_selection_score": float(scores.min()) if not scores.empty else None,
                "positive_train_folds": int((returns > 0).sum()),
                "mean_train_return": float(returns.mean()) if not returns.empty else None,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["strategy", "median_selection_score", "worst_selection_score"],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)


def run_symbol_exclusion_tests(
    data: Mapping[str, pd.DataFrame],
    strategy_factory: Callable[..., Any],
    parameters: Mapping[str, Any],
    *,
    config: BacktestConfig,
    protected_symbols: Sequence[str] = (),
) -> pd.DataFrame:
    """Rerun a fixed strategy after removing each tradable symbol in turn."""

    protected = {symbol.upper() for symbol in protected_symbols}
    rows: list[dict[str, Any]] = []
    for excluded in sorted(set(data).difference(protected)):
        subset = {symbol: frame for symbol, frame in data.items() if symbol != excluded}
        result = BacktestEngine(subset, config=replace(config)).run(strategy_factory(**dict(parameters)))
        rows.append({"excluded_symbol": excluded, **result.metrics})
    return pd.DataFrame(rows)


def write_research_charts(
    output_dir: str | Path,
    *,
    results: Mapping[str, BacktestResult] | None = None,
    benchmarks: Mapping[str, pd.DataFrame] | None = None,
    folds: pd.DataFrame | None = None,
    stability: pd.DataFrame | None = None,
    bootstrap_paths: pd.DataFrame | None = None,
) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    result_map = dict(results or {})
    benchmark_map = dict(benchmarks or {})

    if result_map or benchmark_map:
        path = directory / "equity_comparison.png"
        fig, axis = plt.subplots(figsize=(11, 6))
        for name, result in result_map.items():
            curve = result.equity_curve
            axis.plot(pd.to_datetime(curve["date"]), curve["strategy_index"], label=name, linewidth=1.5)
        for name, curve in benchmark_map.items():
            values = pd.to_numeric(curve["equity"], errors="coerce")
            axis.plot(pd.to_datetime(curve["date"]), values / values.iloc[0], label=name, linewidth=1.2, linestyle="--")
        axis.set_title("Normalized Equity")
        axis.set_ylabel("Growth of PKR 1")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths["equity"] = path

        path = directory / "drawdown_comparison.png"
        fig, axis = plt.subplots(figsize=(11, 5))
        for name, result in result_map.items():
            curve = result.equity_curve
            axis.plot(pd.to_datetime(curve["date"]), curve["drawdown"], label=name, linewidth=1.4)
        for name, curve in benchmark_map.items():
            axis.plot(pd.to_datetime(curve["date"]), curve["drawdown"], label=name, linewidth=1.1, linestyle="--")
        axis.set_title("Drawdown")
        axis.set_ylabel("Drawdown")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths["drawdown"] = path

    trades = [result.trades.assign(strategy=name) for name, result in result_map.items() if not result.trades.empty]
    if trades:
        trade_frame = pd.concat(trades, ignore_index=True)
        realized = trade_frame[trade_frame["side"] == "sell"]
        if not realized.empty:
            path = directory / "trade_distribution.png"
            fig, axis = plt.subplots(figsize=(9, 5))
            for name, group in realized.groupby("strategy"):
                axis.hist(group["realized_pnl"], bins=25, alpha=0.45, label=name)
            axis.axvline(0, color="black", linewidth=0.8)
            axis.set_title("Realized P&L Distribution")
            axis.set_xlabel("PKR per sell fill")
            axis.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths["trades"] = path

    if folds is not None and not folds.empty and "total_return" in folds:
        base = folds[folds["scenario"] == "base"].copy() if "scenario" in folds else folds.copy()
        if not base.empty:
            path = directory / "fold_comparison.png"
            pivot = base.pivot_table(index="fold", columns="strategy", values="total_return", aggfunc="first")
            fig, axis = plt.subplots(figsize=(11, 5))
            pivot.plot(kind="bar", ax=axis)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title("Out-of-Sample Return by Walk-Forward Fold")
            axis.set_ylabel("Fold return")
            axis.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths["folds"] = path

    if stability is not None and not stability.empty:
        path = directory / "parameter_stability.png"
        display = stability.groupby("strategy", sort=True).head(10).copy()
        display["label"] = display.groupby("strategy").cumcount().map(lambda value: f"rank {value + 1}")
        fig, axis = plt.subplots(figsize=(11, 5))
        for strategy, group in display.groupby("strategy"):
            axis.plot(group["label"], group["median_selection_score"], marker="o", label=strategy)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title("Parameter Stability Across Training Folds")
        axis.set_ylabel("Median selection score")
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths["stability"] = path

    if bootstrap_paths is not None and not bootstrap_paths.empty:
        path = directory / "bootstrap_terminal_returns.png"
        fig, axis = plt.subplots(figsize=(9, 5))
        axis.hist(bootstrap_paths["terminal_return"], bins=40, color="#3274a1", alpha=0.85)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title("Block-Bootstrap Terminal Returns")
        axis.set_xlabel("Terminal return")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths["bootstrap"] = path

    return paths


def write_json(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    return destination


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    raise TypeError(f"cannot serialize {type(value).__name__}")


__all__ = [
    "block_bootstrap_returns",
    "bootstrap_summary",
    "parameter_stability_table",
    "performance_by_market_regime",
    "run_symbol_exclusion_tests",
    "write_json",
    "write_research_charts",
]
