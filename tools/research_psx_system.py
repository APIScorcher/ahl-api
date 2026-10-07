from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ahl_api.backtest import BacktestConfig, BacktestEngine, BacktestResult, StrategyContext
from ahl_api.datasets import load_eod_snapshot, manifest_provenance
from ahl_api.research import (
    DataProvenance,
    StrategySpec,
    WalkForwardConfig,
    audit_market_data,
    default_stress_scenarios,
    price_index_benchmark,
    run_walk_forward_validation,
)
from ahl_api.research_reporting import (
    block_bootstrap_returns,
    bootstrap_summary,
    parameter_stability_table,
    performance_by_market_regime,
    run_symbol_exclusion_tests,
    write_json,
    write_research_charts,
)
from ahl_api.research_strategies import (
    BollingerTrendReversionStrategy,
    CloseChannelBreakoutStrategy,
    CrossSectionalMomentumStrategy,
    RsiTrendPullbackStrategy,
    TechnicalEnsembleStrategy,
)
from ahl_api.strategies import LiquidityFilteredMomentumStrategy, MomentumTrendFilterStrategy


INDEX_SYMBOLS = {"KSE100", "KSE100PR"}
SOURCE_LINKS = {
    "psx_indices": "https://www.psx.com.pk/psx/product-and-services/indices",
    "kse_methodology": "https://www.psx.com.pk/psx/themes/psx/uploads/KSE-100-and-KSE100-PR-Brochure-Jun-2025-Updated.pdf",
    "psx_data_terms": "https://www.psx.com.pk/psx/terms-of-use",
    "psx_data_services": "https://www.psx.com.pk/psx/product-and-services/data-services-vending",
    "psx_charges": "https://www.psx.com.pk/psx/resources-and-tools/investors/investor-awareness-guide",
    "t1": "https://www.secp.gov.pk/media-center/event-gallery/secp-introduces-strategic-roadmap-to-transition-to-t1-settlement-cycle/",
    "cgt": "https://www.nccpl.com.pk/cgt",
    "js_momentum": "https://www.psx.com.pk/psx/themes/psx/uploads/JSMFI_brochure_Updated_January_7_2022_by_JS-PDF.pdf",
    "momentum_original": "https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1993.tb04702.x",
    "emerging_momentum": "https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00151",
    "psx_momentum_mixed": "https://doi.org/10.1177/0972150921991506",
    "psx_microstructure": "https://doi.org/10.1016/j.jfineco.2004.06.014",
}


class CashStrategy:
    def needs_history(self, date: pd.Timestamp) -> bool:
        return False

    def target_weights(self, context: StrategyContext) -> dict[str, float]:
        return {}


class BuyAndHoldStrategy:
    def __init__(self, weights: Mapping[str, float]) -> None:
        self.weights = {symbol.upper(): float(weight) for symbol, weight in weights.items()}
        self.sent = False

    def needs_history(self, date: pd.Timestamp) -> bool:
        return False

    def target_weights(self, context: StrategyContext) -> dict[str, float]:
        if self.sent:
            return {}
        self.sent = True
        return dict(self.weights)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run data-audited, nested walk-forward PSX technical-system research."
    )
    parser.add_argument("--snapshot-dir", required=True, help="Checksummed snapshot created by collect_psx_eod.py.")
    parser.add_argument("--output-dir", help="Report directory. Defaults to artifacts/research/<timestamp>.")
    parser.add_argument("--mode", choices=["smoke", "quick", "full"], default="quick")
    parser.add_argument("--initial-capital", type=float, default=50_000.0)
    parser.add_argument("--slippage-bps", type=float, default=25.0)
    parser.add_argument("--other-fees-bps", type=float, default=0.0)
    parser.add_argument("--cgt-rate", type=float, default=0.15)
    parser.add_argument("--max-volume-participation", type=float, default=0.005)
    parser.add_argument("--minimum-trade-value", type=float, default=1_000.0)
    parser.add_argument("--min-traded-value", type=float, default=5_000_000.0)
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--step-months", type=int, default=6)
    parser.add_argument("--inner-train-months", type=int, default=12)
    parser.add_argument("--inner-test-months", type=int, default=3)
    parser.add_argument("--inner-step-months", type=int, default=3)
    parser.add_argument("--minimum-inner-folds", type=int, default=2)
    parser.add_argument("--walk-forward-mode", choices=["rolling", "anchored", "both"], default="both")
    parser.add_argument("--bootstrap-simulations", type=int, help="Defaults by mode: smoke 100, quick 2,000, full 10,000.")
    parser.add_argument("--skip-symbol-exclusion", action="store_true")
    parser.add_argument("--fail-on-interrupt", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true", help="Regenerate final reports from completed CSV/JSON outputs.")
    return parser.parse_args()


def research_config(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        initial_capital=args.initial_capital,
        brokerage_bps=15.0,
        brokerage_per_share=0.03,
        sales_tax_rate=0.15,
        slippage_bps=args.slippage_bps,
        other_fees_bps=args.other_fees_bps,
        capital_gains_tax_rate=args.cgt_rate,
        settlement_lag_days=2,
        settlement_lag_schedule=(("2026-02-09", 1),),
        signal_delay_sessions=1,
        target_intent_expiry_sessions=3,
        max_volume_participation=args.max_volume_participation,
        minimum_trade_value=args.minimum_trade_value,
        board_lot_size=1,
    )


def strategy_specs(mode: str, min_traded_value: float) -> list[StrategySpec]:
    fixed_momentum = {
        "momentum_skip": [21],
        "liquidity_lookback": [60],
        "min_median_traded_value": [min_traded_value],
        "min_price": [20.0],
        "max_zero_volume_fraction": [0.05],
        "volatility_lookback": [63],
        "max_gross_exposure": [0.90],
        "max_position_weight": [0.40],
        "rebalance": ["monthly"],
        "market_symbol": ["KSE100PR"],
        "market_sma": [200],
    }
    fixed_breakout = {
        "liquidity_lookback": [60],
        "min_median_traded_value": [min_traded_value],
        "min_price": [20.0],
        "max_gross_exposure": [0.90],
        "max_position_weight": [0.40],
        "market_symbol": ["KSE100PR"],
        "market_sma": [200],
    }
    fixed_pullback = {
        "recovery_ceiling_rsi": [50.0],
        "oversold_window": [3],
        "exit_rsi": [70.0],
        "trend_slope_lookback": [20],
        "exit_ema": [20],
        "max_positions": [2],
        "max_position_weight": [0.45],
        "cash_reserve": [0.10],
        "liquidity_lookback": [60],
        "min_median_traded_value": [min_traded_value],
        "min_price": [20.0],
        "market_symbol": ["KSE100PR"],
        "market_sma": [200],
    }

    if mode == "smoke":
        return [
            StrategySpec(
                "cross_sectional_momentum",
                CrossSectionalMomentumStrategy,
                {**fixed_momentum, "top_n": [3], "momentum_lookback": [126], "trend_sma": [200], "sizing": ["equal"]},
            ),
            StrategySpec(
                "close_channel_breakout",
                CloseChannelBreakoutStrategy,
                {**fixed_breakout, "top_n": [3], "entry_lookback": [63], "exit_lookback": [20], "trend_sma": [150]},
            ),
            StrategySpec(
                "rsi_trend_pullback",
                RsiTrendPullbackStrategy,
                {**fixed_pullback, "rsi_period": [5], "oversold_rsi": [30.0], "trend_sma": [150]},
            ),
            StrategySpec(
                "technical_ensemble",
                TechnicalEnsembleStrategy,
                {
                    "top_n": [3], "momentum_lookback": [126], "trend_sma": [200], "breakout_lookback": [63],
                    "rsi_period": [14], "min_votes": [3], "liquidity_lookback": [60],
                    "min_median_traded_value": [min_traded_value], "min_price": [20.0],
                    "max_gross_exposure": [0.90], "max_position_weight": [0.40],
                    "market_symbol": ["KSE100PR"], "market_sma": [200], "rebalance": ["monthly"],
                },
            ),
        ]

    momentum_top = [2, 3] if mode == "quick" else [2, 3, 5]
    momentum_lookbacks = [126, 252] if mode == "quick" else [63, 126, 189, 252]
    trends = [150, 200]
    breakout_entries = [63, 126] if mode == "quick" else [42, 63, 126]
    rsi_periods = [3, 5] if mode == "quick" else [2, 3, 5, 7]
    band_stds = [1.5, 2.0] if mode == "quick" else [1.5, 2.0, 2.5]
    return [
        StrategySpec(
            "cross_sectional_momentum",
            CrossSectionalMomentumStrategy,
            {
                **fixed_momentum,
                "top_n": momentum_top,
                "momentum_lookback": momentum_lookbacks,
                "trend_sma": trends,
                "sizing": ["equal"] if mode == "quick" else ["equal", "inverse_volatility"],
            },
        ),
        StrategySpec(
            "close_channel_breakout",
            CloseChannelBreakoutStrategy,
            {
                **fixed_breakout,
                "top_n": momentum_top,
                "entry_lookback": breakout_entries,
                "exit_lookback": [20],
                "trend_sma": trends,
            },
        ),
        StrategySpec(
            "rsi_trend_pullback",
            RsiTrendPullbackStrategy,
            {
                **fixed_pullback,
                "rsi_period": rsi_periods,
                "oversold_rsi": [25.0, 30.0],
                "trend_sma": [150],
            },
        ),
        StrategySpec(
            "bollinger_trend_reversion",
            BollingerTrendReversionStrategy,
            {
                "band_window": [20], "band_std": band_stds, "trend_sma": [150], "max_positions": [2],
                "max_position_weight": [0.45], "cash_reserve": [0.10], "liquidity_lookback": [60],
                "min_median_traded_value": [min_traded_value], "min_price": [20.0],
                "market_symbol": ["KSE100PR"], "market_sma": [200],
            },
        ),
        StrategySpec(
            "technical_ensemble",
            TechnicalEnsembleStrategy,
            {
                "top_n": momentum_top, "momentum_lookback": [126], "trend_sma": trends,
                "breakout_lookback": [63], "rsi_period": [14], "min_votes": [3, 4],
                "liquidity_lookback": [60], "min_median_traded_value": [min_traded_value],
                "min_price": [20.0], "max_gross_exposure": [0.90], "max_position_weight": [0.40],
                "market_symbol": ["KSE100PR"], "market_sma": [200], "rebalance": ["monthly"],
            },
        ),
    ]


def first_parameters(spec: StrategySpec) -> dict[str, Any]:
    return {name: list(values)[0] for name, values in spec.parameter_grid.items()}


def run_exploration(
    data: Mapping[str, pd.DataFrame],
    specs: list[StrategySpec],
    config: BacktestConfig,
    output: Path,
    *,
    continue_on_interrupt: bool,
) -> tuple[dict[str, BacktestResult], pd.DataFrame]:
    results: dict[str, BacktestResult] = {}
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[str, Any, str]] = [
        (spec.name, spec.factory(**first_parameters(spec)), json.dumps(first_parameters(spec), sort_keys=True))
        for spec in specs
    ]
    candidates.extend(
        [
            (
                "legacy_liquidity_momentum",
                LiquidityFilteredMomentumStrategy(top_n=3, momentum_lookback=126, liquidity_lookback=20, min_avg_traded_value=5_000_000, rebalance="monthly"),
                "fixed legacy baseline",
            ),
            (
                "legacy_momentum_sma200",
                MomentumTrendFilterStrategy(top_n=3, momentum_lookback=126, liquidity_lookback=20, min_avg_traded_value=5_000_000, trend_sma=200, rebalance="monthly"),
                "fixed legacy baseline",
            ),
        ]
    )
    for name, strategy, parameters in candidates:
        print(f"exploration: {name}", flush=True)
        try:
            result = BacktestEngine(data, config=config).run(strategy)
        except KeyboardInterrupt:
            if not continue_on_interrupt:
                raise
            rows.append({"strategy": name, "status": "interrupted", "parameters": parameters})
            pd.DataFrame(rows).to_csv(output / "in_sample_leaderboard.csv", index=False)
            continue
        result.write_report(output / "in_sample" / name, prefix=name)
        results[name] = result
        rows.append({"strategy": name, "status": "complete", "parameters": parameters, **result.metrics})
        pd.DataFrame(rows).to_csv(output / "in_sample_leaderboard.csv", index=False)
    rows.append(
        {
            "strategy": "legacy_pullback_dca",
            "status": "rejected_without_rerun",
            "parameters": "order thresholds are consumed before fill confirmation",
        }
    )
    leaderboard = pd.DataFrame(rows)
    if "final_equity" in leaderboard:
        leaderboard = leaderboard.sort_values("final_equity", ascending=False, na_position="last")
    leaderboard.to_csv(output / "in_sample_leaderboard.csv", index=False)
    write_json(output / "in_sample_leaderboard.json", leaderboard.to_dict(orient="records"))
    return results, leaderboard


def run_benchmarks(
    data: Mapping[str, pd.DataFrame],
    config: BacktestConfig,
    output: Path,
) -> tuple[dict[str, BacktestResult], dict[str, pd.DataFrame], pd.DataFrame]:
    trade_symbols = sorted(set(data).difference(INDEX_SYMBOLS))
    benchmark_results: dict[str, BacktestResult] = {}
    rows: list[dict[str, Any]] = []
    benchmark_strategies: dict[str, Any] = {"cash": CashStrategy()}
    if trade_symbols:
        benchmark_strategies["static_equal_weight_sample"] = BuyAndHoldStrategy(
            {symbol: 1.0 / len(trade_symbols) for symbol in trade_symbols}
        )
    trio = [symbol for symbol in ("FFC", "HUBC", "OGDC") if symbol in data]
    if trio:
        benchmark_strategies["buy_hold_FFC_HUBC_OGDC"] = BuyAndHoldStrategy(
            {symbol: 1.0 / len(trio) for symbol in trio}
        )
    for name, strategy in benchmark_strategies.items():
        result = BacktestEngine(data, config=config).run(strategy)
        result.write_report(output / "benchmarks" / name, prefix=name)
        benchmark_results[name] = result
        rows.append({"benchmark": name, "kind": "tradable_static", **result.metrics})

    index_curves: dict[str, pd.DataFrame] = {}
    for symbol, label in (("KSE100", "kse100_total_return"), ("KSE100PR", "kse100_price_return")):
        if symbol not in data:
            continue
        curve, metrics = price_index_benchmark(data[symbol], name=label, initial_capital=config.initial_capital)
        curve.to_csv(output / "benchmarks" / f"{label}_curve.csv", index=False)
        write_json(output / "benchmarks" / f"{label}_metrics.json", metrics)
        index_curves[label] = curve
        rows.append({"benchmark": label, "kind": "nontradable_index", **metrics})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "benchmark_leaderboard.csv", index=False)
    write_json(output / "benchmark_leaderboard.json", frame.to_dict(orient="records"))
    return benchmark_results, index_curves, frame


def acceptance_table(
    audit_has_blockers: bool,
    stitched: pd.DataFrame,
    benchmark_curves: Mapping[str, pd.DataFrame],
    *,
    minimum_outer_folds: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if stitched.empty:
        return pd.DataFrame()
    for strategy in sorted(stitched["strategy"].unique()):
        group = stitched[stitched["strategy"] == strategy]
        base = group[group["scenario"] == "base"]
        if base.empty:
            continue
        item = base.iloc[0]
        stressed = group[group["scenario"] != "base"]
        benchmark_return = None
        benchmark_drawdown = None
        total_curve = benchmark_curves.get("kse100_total_return")
        if total_curve is not None:
            dates = pd.to_datetime(total_curve["date"])
            window = total_curve[
                (dates >= pd.Timestamp(item["start_date"]))
                & (dates <= pd.Timestamp(item["end_date"]))
            ]
            if len(window) > 1:
                benchmark_return = float(window["equity"].iloc[-1] / window["equity"].iloc[0] - 1.0)
                normalized = window["equity"] / float(window["equity"].iloc[0])
                benchmark_drawdown = float((normalized / normalized.cummax() - 1.0).min())
        reasons: list[str] = []
        if audit_has_blockers:
            reasons.append("blocking data-quality/provenance findings")
        if int(item.get("outer_folds", 0)) < minimum_outer_folds:
            reasons.append("too few outer folds")
        if float(item.get("total_return", 0.0) or 0.0) <= 0:
            reasons.append("non-positive stitched outer-OOS return")
        if benchmark_return is not None and float(item.get("total_return", 0.0) or 0.0) <= benchmark_return:
            reasons.append("did not beat KSE-100 total-return benchmark")
        if int(item.get("round_trip_count", 0) or 0) < 50:
            reasons.append("fewer than 50 completed sells")
        if float(item.get("best_symbol_profit_fraction", 1.0) or 1.0) > 0.50:
            reasons.append("more than half of gross profit came from one symbol")
        if not stressed.empty and (pd.to_numeric(stressed["total_return"], errors="coerce") <= 0).any():
            reasons.append("failed at least one execution stress")
        rows.append(
            {
                "strategy": strategy,
                "qualified": not reasons,
                "decision": "accept" if not reasons else "reject",
                "reasons": "; ".join(reasons),
                "benchmark_total_return": benchmark_return,
                "benchmark_max_drawdown": benchmark_drawdown,
                "benchmark_relative_return": (
                    float(item.get("total_return", 0.0) or 0.0) - benchmark_return
                    if benchmark_return is not None
                    else None
                ),
                **{key: item.get(key) for key in ("final_equity", "total_return", "cagr", "max_drawdown", "sharpe", "round_trip_count")},
            }
        )
    return pd.DataFrame(rows).sort_values(["qualified", "final_equity"], ascending=[False, False])


def write_reports(
    output: Path,
    *,
    audit: Any,
    manifest: Any,
    in_sample: pd.DataFrame,
    benchmarks: pd.DataFrame,
    walk_forward: Any,
    acceptance: pd.DataFrame,
    bootstrap_metrics: Mapping[str, Any],
    quarantined_symbols: set[str],
) -> None:
    history_start = audit.summary["start_date"].min() if not audit.summary.empty else None
    history_end = audit.summary["end_date"].max() if not audit.summary.empty else None
    best_oos = None
    if not walk_forward.stitched_leaderboard.empty:
        base = walk_forward.stitched_leaderboard[walk_forward.stitched_leaderboard["scenario"] == "base"]
        if not base.empty:
            best_oos = base.sort_values("final_equity", ascending=False).iloc[0]
    qualified = acceptance[acceptance["qualified"]] if not acceptance.empty else pd.DataFrame()
    final_decision = "No strategy qualifies for live use." if qualified.empty else f"Qualified strategy: {qualified.iloc[0]['strategy']}."

    research = f"""# PSX Technical Trading Research

## Decision

{final_decision} The only permissible next stage is paper trading. No live order was submitted, cancelled, or modified.

## Scope

- Portfolio: one-time PKR 50,000, cash equity, long-only, no leverage, no deposits.
- Snapshot: `{manifest.snapshot_id}`, collected `{manifest.collected_at_utc}`.
- Available history: {history_start} through {history_end}; this is about five years, not the preferred 15 years.
- Universe: {len(set(audit.summary['symbol']).difference(INDEX_SYMBOLS).difference(quarantined_symbols))} usable explicitly sampled current symbols, not historical point-in-time KSE-100 membership.
- Quarantined before testing: {', '.join(sorted(quarantined_symbols)) or 'none'}.
- Execution: close signal, next-session open, integer shares, brokerage minimum, sales tax, 25 bps default slippage, 0.5% prior-volume participation, T+2 switching to T+1 on 2026-02-09, and bounded target retries.
- Tax: conservative 15% per-profitable-sell CGT proxy. It is not NCCPL portfolio tax-lot netting.

## Evidence

Cross-sectional momentum is the strongest research hypothesis, but PSX-specific evidence is mixed. The exchange's JSMFI methodology combines adjusted 30-day return and traded value with monthly rebalancing, while broader momentum evidence supports medium-horizon continuation. These motivate tests; they do not prove a deployable PSX edge. See [PSX JSMFI]({SOURCE_LINKS['js_momentum']}), [original momentum evidence]({SOURCE_LINKS['momentum_original']}), [emerging-market evidence]({SOURCE_LINKS['emerging_momentum']}), and [mixed PSX evidence]({SOURCE_LINKS['psx_momentum_mixed']}).

KSE-100 is a total-return index and KSE100PR is a price-return index under PSX methodology, so both are reported. See [PSX index methodology]({SOURCE_LINKS['kse_methodology']}).

## Validation

- Parameters were selected on inner out-of-sample folds, then evaluated on untouched outer folds.
- Outer test periods were also stitched into one continuously carried account.
- Fixed stresses doubled costs, delayed entry, missed 10% of fills, and applied 100 bps adverse slippage.
- Parameter stability, block-bootstrap paths, market-regime splits, and symbol exclusions were generated.
- In-sample tables are explicitly exploratory and were not used as final evidence.

## Best Outer-OOS Observation

{_format_best(best_oos)}

## Bootstrap

`{json.dumps(dict(bootstrap_metrics), sort_keys=True)}`

## Limitations

The result is retrospective and not decision-grade because adjusted total-return stock histories, dividends, corporate actions, point-in-time membership, delisted names, historical board lots, sessions, circuit locks, and halts are unavailable. Public intraday endpoints expose only the latest session, so ORB, VWAP, relative-volume, spread, and depth tests were rejected. PSX terms restrict automated collection and database creation without permission; obtain licensed data before production research. See [PSX terms]({SOURCE_LINKS['psx_data_terms']}) and [data services]({SOURCE_LINKS['psx_data_services']}).
"""
    (output / "research_report.md").write_text(research, encoding="utf-8")

    issues = _markdown_table(audit.issues) if not audit.issues.empty else "No issues recorded."
    data_report = f"""# Data Quality And Bias Report

## Verdict

Blocked for strategy acceptance. The snapshot can support exploratory close-only mechanics, not a production claim.

## Findings

{issues}

## Provenance

```json
{json.dumps(asdict(audit.provenance), indent=2, sort_keys=True)}
```

The public EOD schema has open, close, and volume but no high/low. It returns roughly five rolling years. Some symbols mix adjusted historical closes with old-scale opens; those series were quarantined before the final run. Current tickers can backfill predecessor histories, so symbol identity is not point-in-time. Sector concentration cannot be reconstructed reliably without effective-dated classifications.
"""
    (output / "data_quality_and_bias_report.md").write_text(data_report, encoding="utf-8")

    rejected = _markdown_table(acceptance) if not acceptance.empty else "No strategy completed enough validation to score."
    red_team = f"""# Red-Team Report

## Decision

Reject all strategies for live use.

## Strategy Decisions

{rejected}

## Invalidated Or Blocked Tests

- Intraday ORB, VWAP, relative-volume, first-hour, and depth models: no trustworthy historical intraday archive.
- ATR, true Donchian, volatility-contraction, and high/low stop variants: historical high/low unavailable.
- Legacy pullback DCA: marks a threshold consumed when an order is emitted, even if it later rejects or misses.
- Static-current-universe results: survivorship-biased and unsuitable for acceptance.
- Dividend-heavy buy-and-hold comparisons: stock data omit cash dividends and corporate-action adjustments.
- Historical one-share execution: historical per-symbol board lots are missing.

The validator corrected a discarded-rotation-intent defect by retaining target buys through settlement, and corrected whole-portfolio freezes around suspended symbols. Those code fixes do not resolve the source-data blockers.
"""
    (output / "red_team_report.md").write_text(red_team, encoding="utf-8")

    paper = """# Paper-Trading Plan

## Candidate

Paper-test the pre-registered cross-sectional momentum rule only; this is not a live recommendation.

1. Use licensed, adjusted, point-in-time KSE-100 data and an effective-dated symbol/sector/lot/session master.
2. After the first regular-session close of each calendar month, require KSE100PR above SMA200, a current bar, price at least PKR 20, no more than 5% zero-volume sessions, and 60-session median traded value at least PKR 5 million.
3. Rank positive 126-session momentum ending 21 sessions earlier; retain only stocks above SMA200.
4. Hold the top three at 30% each, capped at 40% per stock; keep at least 10% cash. If no stock qualifies or the market filter fails, target cash.
5. Generate signals after close and simulate no earlier than the next regular-session open. Cap each fill at 0.5% of prior-session volume and reject circuit-locked fills without executable liquidity.
6. Sell first; retain replacement-buy intents for three tradable sessions while T+1 proceeds settle. Never infer same-day buying power from AHL's delayed portfolio endpoint.
7. Shadow the account for at least six months and 30 completed round trips. Reconcile every accepted order, fill, cancellation, fee, tax, settlement, and next-day holding.
8. Stop the paper system for a 20% drawdown, any ledger discrepancy, stale market data, ambiguous submission, or rule/data change.
"""
    (output / "paper_trading_plan.md").write_text(paper, encoding="utf-8")

    live = """# Disabled Live-Integration Proposal

- Keep `AHL(dry_run=True)` as the default and refuse startup if disabled without an immediate interactive confirmation.
- Build a local shadow ledger keyed by client intent and broker order ID; accepted, ambiguous, rejected, filled, cancelled, and settled are distinct states.
- Never retry an ambiguous HTTP submission automatically.
- Reconcile read-only order logs, fills, balances, and next-day holdings before issuing another batch.
- Size only from settled shadow cash, not the delayed portfolio endpoint.
- Apply symbol allowlists, per-order value/quantity caps, price-band checks, market-status checks, and a kill switch.
- Produce an order preview and require a separate explicit confirmation immediately before a batch.
- This integration remains disabled until licensed data, prospective paper evidence, and independent review clear every acceptance gate.
"""
    (output / "live_integration_disabled.md").write_text(live, encoding="utf-8")


def _format_best(row: Any) -> str:
    if row is None:
        return "No stitched outer-OOS result completed."
    return (
        f"`{row['strategy']}` under `{row['scenario']}` ended at PKR {float(row['final_equity']):,.2f}, "
        f"return {float(row['total_return']):.2%}, CAGR {float(row['cagr']):.2%}, "
        f"max drawdown {float(row['max_drawdown']):.2%}, over {int(row['outer_folds'])} outer folds. "
        "This observation is rejected if the data audit has blockers."
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = []
    for values in frame.astype(object).where(pd.notna(frame), "").itertuples(index=False, name=None):
        rows.append([str(value).replace("|", "\\|").replace("\n", " ") for value in values])
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(values) + " |" for values in rows]
    return "\n".join([header, separator, *body])


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir or Path("artifacts") / "research" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=True)
    data, manifest = load_eod_snapshot(args.snapshot_dir)
    missing_indices = INDEX_SYMBOLS.difference(data)
    if missing_indices:
        raise SystemExit(f"snapshot is missing required index series: {sorted(missing_indices)}")
    provenance = DataProvenance(**manifest_provenance(manifest))
    audit = audit_market_data(data, provenance=provenance)
    audit.write_report(output)
    print(f"data audit: {len(audit.issues)} findings, blockers={audit.has_blockers}", flush=True)
    quarantined_symbols = set(
        audit.issues.loc[
            audit.issues["code"] == "open_close_scale_mismatch",
            "symbol",
        ].astype(str)
    )
    if quarantined_symbols.intersection(INDEX_SYMBOLS):
        raise SystemExit(f"required benchmark series failed open/close scale audit: {sorted(quarantined_symbols.intersection(INDEX_SYMBOLS))}")
    data = {symbol: frame for symbol, frame in data.items() if symbol not in quarantined_symbols}
    write_json(output / "quarantined_symbols.json", sorted(quarantined_symbols))
    print(f"quarantined symbols: {', '.join(sorted(quarantined_symbols)) or 'none'}", flush=True)

    config = research_config(args)
    specs = strategy_specs(args.mode, args.min_traded_value)
    if args.finalize_existing:
        mode = "rolling" if (output / "walk_forward_rolling").exists() else "anchored"
        in_sample = pd.read_csv(output / "in_sample_leaderboard.csv")
        benchmark_table = pd.read_csv(output / "benchmark_leaderboard.csv")
        acceptance = pd.read_csv(output / "strategy_acceptance.csv")
        stitched = pd.read_csv(output / f"walk_forward_{mode}" / "walk_forward_stitched_leaderboard.csv")
        bootstrap_path = output / "bootstrap_summary.json"
        bootstrap_metrics = json.loads(bootstrap_path.read_text(encoding="utf-8")) if bootstrap_path.exists() else {}
        write_reports(
            output,
            audit=audit,
            manifest=manifest,
            in_sample=in_sample,
            benchmarks=benchmark_table,
            walk_forward=SimpleNamespace(stitched_leaderboard=stitched),
            acceptance=acceptance,
            bootstrap_metrics=bootstrap_metrics,
            quarantined_symbols=quarantined_symbols,
        )
        write_json(
            output / "run_manifest.json",
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "snapshot_id": manifest.snapshot_id,
                "arguments": vars(args),
                "backtest_config": asdict(config),
                "source_links": SOURCE_LINKS,
                "live_orders_submitted": False,
                "finalized_from_existing_outputs": True,
            },
        )
        print(f"finalized reports: {output.resolve()}", flush=True)
        return 0
    continue_on_interrupt = not args.fail_on_interrupt
    benchmark_results, index_curves, benchmark_table = run_benchmarks(data, config, output)
    exploration_results, in_sample = run_exploration(
        data,
        specs,
        config,
        output,
        continue_on_interrupt=continue_on_interrupt,
    )

    validation_specs = [
        spec
        for spec in specs
        if args.mode == "full" or spec.name in {"cross_sectional_momentum", "technical_ensemble"}
    ]
    modes = ["rolling", "anchored"] if args.walk_forward_mode == "both" else [args.walk_forward_mode]
    validation_configs: dict[str, WalkForwardConfig] = {}
    walk_forwards: dict[str, Any] = {}
    for mode in modes:
        validation = WalkForwardConfig(
            train_months=args.train_months,
            test_months=args.test_months,
            step_months=args.step_months,
            mode=mode,
            embargo_sessions=1,
            inner_train_months=args.inner_train_months,
            inner_test_months=args.inner_test_months,
            inner_step_months=args.inner_step_months,
            minimum_inner_folds=args.minimum_inner_folds,
            minimum_round_trips=5,
            continue_on_interrupt=continue_on_interrupt,
        )
        validation_configs[mode] = validation
        print(f"nested walk-forward ({mode}): starting", flush=True)
        walk_forward = run_walk_forward_validation(
            data,
            validation_specs,
            backtest_config=config,
            validation_config=validation,
            stress_scenarios=default_stress_scenarios(config),
        )
        walk_forward.write_report(output / f"walk_forward_{mode}")
        walk_forwards[mode] = walk_forward
        print(f"nested walk-forward ({mode}): complete", flush=True)
    primary_mode = "rolling" if "rolling" in walk_forwards else modes[0]
    walk_forward = walk_forwards[primary_mode]

    selection_frames = [
        result.selection_runs.assign(validation_mode=mode)
        for mode, result in walk_forwards.items()
        if not result.selection_runs.empty
    ]
    combined_selection = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
    stability = parameter_stability_table(combined_selection)
    stability.to_csv(output / "parameter_stability.csv", index=False)
    write_json(output / "parameter_stability.json", stability.to_dict(orient="records"))
    acceptance_frames = []
    for mode, result in walk_forwards.items():
        mode_acceptance = acceptance_table(audit.has_blockers, result.stitched_leaderboard, index_curves, minimum_outer_folds=4)
        if not mode_acceptance.empty:
            acceptance_frames.append(mode_acceptance.assign(validation_mode=mode))
    acceptance = pd.concat(acceptance_frames, ignore_index=True) if acceptance_frames else pd.DataFrame()
    acceptance.to_csv(output / "strategy_acceptance.csv", index=False)
    write_json(output / "strategy_acceptance.json", acceptance.to_dict(orient="records"))

    stitched_base: dict[str, BacktestResult] = {
        run_id.removeprefix("stitched_").removesuffix("_base"): result
        for run_id, result in walk_forward.results.items()
        if run_id.startswith("stitched_") and run_id.endswith("_base")
    }
    bootstrap_paths = pd.DataFrame()
    bootstrap_metrics: dict[str, Any] = {}
    best_name = None
    if stitched_base:
        best_name, best_result = max(stitched_base.items(), key=lambda item: item[1].metrics.get("final_equity", 0.0))
        simulations = args.bootstrap_simulations or {"smoke": 100, "quick": 2_000, "full": 10_000}[args.mode]
        bootstrap_paths = block_bootstrap_returns(best_result.equity_curve, simulations=simulations, block_size=5, seed=20260716)
        bootstrap_paths.to_csv(output / "bootstrap_paths.csv", index=False)
        bootstrap_metrics = {"strategy": best_name, **bootstrap_summary(bootstrap_paths)}
        write_json(output / "bootstrap_summary.json", bootstrap_metrics)
        regimes = performance_by_market_regime(best_result.equity_curve, data["KSE100PR"], sma_window=200)
        regimes.to_csv(output / "market_regime_performance.csv", index=False)
        write_json(output / "market_regime_performance.json", regimes.to_dict(orient="records"))

    if not args.skip_symbol_exclusion:
        momentum_spec = next(spec for spec in specs if spec.name == "cross_sectional_momentum")
        exclusions = run_symbol_exclusion_tests(
            data,
            momentum_spec.factory,
            first_parameters(momentum_spec),
            config=config,
            protected_symbols=INDEX_SYMBOLS,
        )
        exclusions.to_csv(output / "symbol_exclusion.csv", index=False)
        write_json(output / "symbol_exclusion.json", exclusions.to_dict(orient="records"))

    chart_results = dict(stitched_base)
    chart_results.update({f"benchmark_{name}": result for name, result in benchmark_results.items()})
    write_research_charts(
        output / "charts",
        results=chart_results,
        benchmarks=index_curves,
        folds=walk_forward.folds,
        stability=stability,
        bootstrap_paths=bootstrap_paths,
    )
    write_reports(
        output,
        audit=audit,
        manifest=manifest,
        in_sample=in_sample,
        benchmarks=benchmark_table,
        walk_forward=walk_forward,
        acceptance=acceptance,
        bootstrap_metrics=bootstrap_metrics,
        quarantined_symbols=quarantined_symbols,
    )
    write_json(
        output / "run_manifest.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "snapshot_id": manifest.snapshot_id,
            "arguments": vars(args),
            "backtest_config": asdict(config),
            "validation_configs": {mode: asdict(value) for mode, value in validation_configs.items()},
            "validated_strategies": [spec.name for spec in validation_specs],
            "quarantined_symbols": sorted(quarantined_symbols),
            "strategy_specs": [
                {"name": spec.name, "parameter_grid": {key: list(value) for key, value in spec.parameter_grid.items()}}
                for spec in specs
            ],
            "source_links": SOURCE_LINKS,
            "live_orders_submitted": False,
        },
    )
    print(f"reports: {output.resolve()}", flush=True)
    if acceptance.empty or not bool(acceptance["qualified"].any()):
        print("decision: no strategy qualifies; paper trading only", flush=True)
    else:
        print(f"decision: {acceptance[acceptance['qualified']].iloc[0]['strategy']} qualifies for paper review only", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
