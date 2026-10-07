from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ahl_api.backtest import BacktestConfig, BacktestEngine, Order
from ahl_api.client import AHL


DEFAULT_SYMBOLS = ["FFC", "LUCK", "OGDC", "HUBC", "MARI"]
KSE100_TOP10 = ["FFC", "UBL", "ENGROH", "MEBL", "HUBC", "OGDC", "LUCK", "HBL", "MCB", "PPL"]
DEFAULT_ALLOCATIONS = [1.0]
RANK_WEIGHTS = {
    "mom_6m": 0.35,
    "mom_3m": 0.25,
    "trend": 0.20,
    "low_vol": 0.10,
    "liquidity": 0.10,
}
RELATIVE_STRENGTH_WEIGHTS = {
    "mom_12m": 0.40,
    "mom_6m": 0.30,
    "mom_3m": 0.15,
    "trend": 0.10,
    "low_drawdown": 0.05,
}


@dataclass
class RankedDcaBacktestStrategy:
    monthly_contribution: float
    top_n: int = 3
    cash_reserve: float = 0.0
    allocations: tuple[float, ...] = tuple(DEFAULT_ALLOCATIONS)
    max_position_weight: float = 1.0
    strict_entry_filters: bool = False
    min_trend_score: float = 0.50
    max_ema20_extension: float = 0.08
    max_rsi: float = 75.0
    min_order_value: float = 5_000.0
    rank_snapshots: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.allocations = tuple(self.allocations[: self.top_n])
        if len(self.allocations) < self.top_n:
            self.allocations = self.allocations + tuple(0.0 for _ in range(self.top_n - len(self.allocations)))
        if self.max_position_weight <= 0:
            raise ValueError("max_position_weight must be positive")
        self._last_period: tuple[int, int] | None = None

    def orders(self, context) -> list[Order]:
        period = (context.date.year, context.date.month)
        if period == self._last_period:
            return []
        self._last_period = period

        ranking = rank_context(context.history)
        if ranking.empty:
            return []
        self._record_ranking(context.date, ranking)

        investable_cash = max(context.available_cash - context.equity * self.cash_reserve, 0.0)
        if investable_cash < self.min_order_value:
            return []

        candidates = ranking[ranking.apply(lambda row: is_entry_allowed(row, self.min_trend_score, self.max_ema20_extension, self.max_rsi), axis=1)] if self.strict_entry_filters else ranking
        candidates = candidates.head(self.top_n)
        if candidates.empty:
            return []

        allocation_sum = sum(self.allocations[: len(candidates)]) or 1.0
        orders: list[Order] = []
        for offset, (_, row) in enumerate(candidates.iterrows()):
            symbol = str(row["symbol"])
            rank = int(row["rank"])
            allocation = self.allocations[offset]
            current_value = context.positions.get(symbol).market_value if symbol in context.positions else 0.0
            cap_room = max(context.equity * self.max_position_weight - current_value, 0.0)
            target_value = investable_cash * allocation / allocation_sum
            order_value = min(target_value, cap_room)
            if order_value >= self.min_order_value:
                orders.append(Order(symbol=symbol, side="buy", value=order_value, tag=f"ranked_dca_rank_{rank}"))
        return orders

    def _record_ranking(self, signal_date: pd.Timestamp, ranking: pd.DataFrame) -> None:
        for _, row in ranking.iterrows():
            self.rank_snapshots.append(
                {
                    "signal_date": pd.Timestamp(signal_date).date().isoformat(),
                    "rank": int(row["rank"]),
                    "symbol": str(row["symbol"]),
                    "score": round(float(row["score"]), 6),
                    "close": round(float(row["close"]), 4),
                    "mom_3m": round(float(row["mom_3m"]), 6),
                    "mom_6m": round(float(row["mom_6m"]), 6),
                    "trend_score": round(float(row["trend_score"]), 6),
                    "dist_ema20": round(float(row["dist_ema20"]), 6) if pd.notna(row.get("dist_ema20")) else None,
                    "rsi14": round(float(row["rsi14"]), 4) if pd.notna(row.get("rsi14")) else None,
                    "vol_60d": round(float(row["vol_60d"]), 6),
                    "liquidity_60d": round(float(row["liquidity_60d"]), 2),
                    "entry_allowed": is_entry_allowed(row, self.min_trend_score, self.max_ema20_extension, self.max_rsi),
                }
            )


@dataclass
class MonthlyBuyHoldStrategy:
    symbols: tuple[str, ...]
    cash_reserve: float = 0.0
    min_order_value: float = 5_000.0
    _last_period: tuple[int, int] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        symbols = tuple(dict.fromkeys(symbol.upper() for symbol in self.symbols if symbol))
        if not symbols:
            raise ValueError("MonthlyBuyHoldStrategy requires at least one symbol")
        self.symbols = symbols

    def orders(self, context) -> list[Order]:
        period = (context.date.year, context.date.month)
        if period == self._last_period:
            return []
        self._last_period = period

        available_symbols = [symbol for symbol in self.symbols if symbol in context.history]
        if not available_symbols:
            return []

        investable_cash = max(context.available_cash - context.equity * self.cash_reserve, 0.0)
        order_value = investable_cash / len(available_symbols)
        if order_value < self.min_order_value:
            return []
        return [
            Order(symbol=symbol, side="buy", value=order_value, tag="benchmark_monthly_buy_hold")
            for symbol in available_symbols
        ]


@dataclass
class ConcentratedRelativeStrengthDcaStrategy:
    monthly_contribution: float
    top_n: int = 3
    keep_top_n: int = 5
    allocations: tuple[float, ...] = (0.50, 0.30, 0.20)
    cash_reserve: float = 0.0
    min_order_value: float = 5_000.0
    require_uptrend: bool = True
    trend_sma: int = 200
    rank_snapshots: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _last_period: tuple[int, int] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.allocations = tuple(self.allocations[: self.top_n])
        if len(self.allocations) < self.top_n:
            self.allocations = self.allocations + tuple(0.0 for _ in range(self.top_n - len(self.allocations)))
        if self.top_n <= 0 or self.keep_top_n < self.top_n:
            raise ValueError("keep_top_n must be greater than or equal to top_n")

    def orders(self, context) -> list[Order]:
        period = (context.date.year, context.date.month)
        if period == self._last_period:
            return []
        self._last_period = period

        ranking = relative_strength_ranking(context.history, trend_sma=self.trend_sma)
        if ranking.empty:
            return []
        self._record_ranking(context.date, ranking)

        eligible = ranking[ranking["is_uptrend"]] if self.require_uptrend else ranking
        if eligible.empty:
            return []

        keep_symbols = set(eligible.head(self.keep_top_n)["symbol"])
        buy_candidates = list(eligible.head(self.top_n)["symbol"])
        orders: list[Order] = []

        for symbol, position in context.positions.items():
            if symbol not in keep_symbols:
                orders.append(Order(symbol=symbol, side="sell", shares=position.shares, tag="rs_exit"))

        investable_cash = max(context.available_cash - context.equity * self.cash_reserve, 0.0)
        if investable_cash < self.min_order_value or not buy_candidates:
            return orders

        allocation_sum = sum(self.allocations[: len(buy_candidates)]) or 1.0
        for offset, symbol in enumerate(buy_candidates):
            order_value = investable_cash * self.allocations[offset] / allocation_sum
            if order_value >= self.min_order_value:
                orders.append(Order(symbol=symbol, side="buy", value=order_value, tag=f"rs_dca_rank_{offset + 1}"))
        return orders

    def _record_ranking(self, signal_date: pd.Timestamp, ranking: pd.DataFrame) -> None:
        for _, row in ranking.iterrows():
            self.rank_snapshots.append(
                {
                    "signal_date": pd.Timestamp(signal_date).date().isoformat(),
                    "rank": int(row["rank"]),
                    "symbol": str(row["symbol"]),
                    "score": round(float(row["score"]), 6),
                    "close": round(float(row["close"]), 4),
                    "mom_3m": round(float(row["mom_3m"]), 6),
                    "mom_6m": round(float(row["mom_6m"]), 6),
                    "mom_12m": round(float(row["mom_12m"]), 6),
                    "drawdown_252d": round(float(row["drawdown_252d"]), 6),
                    "sma200": round(float(row["sma200"]), 4) if pd.notna(row.get("sma200")) else None,
                    "is_uptrend": bool(row["is_uptrend"]),
                    "liquidity_60d": round(float(row["liquidity_60d"]), 2),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest PSX Ranked DCA monthly contribution strategy.")
    parser.add_argument("--monthly-deposit", type=float, default=200_000.0)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--warmup-years", type=int, default=1)
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    parser.add_argument("--include-kse100-top10", action="store_true")
    parser.add_argument("--benchmark-symbols", nargs="*", default=None)
    parser.add_argument("--no-benchmarks", action="store_true")
    parser.add_argument("--top-n", type=int, default=1)
    parser.add_argument("--rs-top-n", type=int, default=3)
    parser.add_argument("--rs-keep-top-n", type=int, default=5)
    parser.add_argument("--rs-allocations", default="50,30,20")
    parser.add_argument("--rs-no-trend-filter", action="store_true")
    parser.add_argument("--cash-reserve", type=float, default=0.0)
    parser.add_argument("--max-position-weight", type=float, default=1.0)
    parser.add_argument("--strict-entry-filters", action="store_true")
    parser.add_argument("--allocations", default="100")
    parser.add_argument("--run-exit-rs", action="store_true")
    parser.add_argument("--min-order-value", type=float, default=5_000.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = unique_symbols([symbol.upper() for symbol in args.symbols] + (KSE100_TOP10 if args.include_kse100_top10 else []))
    benchmark_symbols = unique_symbols([symbol.upper() for symbol in (args.benchmark_symbols or symbols)])
    output_dir = Path(args.output_dir) if args.output_dir else Path("backtest") / f"ranked_dca_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    load_years = args.years + max(args.warmup_years, 0)
    print(f"loading {load_years} years of data for: {', '.join(symbols)}")
    client = AHL(audit_enabled=True)
    data = {}
    for symbol in symbols:
        rows = client.fetch_historical_daily(symbol, years=load_years)
        frame = pd.DataFrame(rows)
        if frame.empty:
            print(f"skip {symbol}: no data")
            continue
        data[symbol] = frame
        print(f"loaded {symbol}: {len(frame)} rows ({frame['date'].iloc[0]} to {frame['date'].iloc[-1]})")
    if len(data) < 2:
        raise SystemExit("not enough data to run Ranked DCA backtest")

    latest_date = max(pd.to_datetime(frame["date"]).max() for frame in data.values())
    backtest_start = (pd.Timestamp(latest_date) - pd.DateOffset(years=args.years)).normalize()
    backtest_end = pd.Timestamp(latest_date).normalize()
    print(f"backtest window: {backtest_start.date()} to {backtest_end.date()} (warmup data excluded from deposits/results)")

    config = BacktestConfig(
        initial_capital=0.0,
        recurring_contribution=args.monthly_deposit,
        contribution_frequency="monthly",
        start_date=backtest_start,
        end_date=backtest_end,
        slippage_bps=args.slippage_bps,
    )
    strategy = RankedDcaBacktestStrategy(
        monthly_contribution=args.monthly_deposit,
        top_n=args.top_n,
        cash_reserve=args.cash_reserve,
        allocations=tuple(parse_allocations(args.allocations, args.top_n)),
        max_position_weight=args.max_position_weight,
        strict_entry_filters=args.strict_entry_filters,
        min_order_value=args.min_order_value,
    )
    result = BacktestEngine(data, config=config).run(strategy)
    strategy_name = "leader_ranked_dca" if args.top_n == 1 else "ranked_dca"
    results: list[tuple[str, Any]] = [(strategy_name, result)]
    paths = {f"{strategy_name}_{name}": path for name, path in result.write_report(output_dir, prefix=strategy_name).items()}

    if strategy.rank_snapshots:
        ranking_path = output_dir / "ranked_dca_monthly_rankings.csv"
        pd.DataFrame(strategy.rank_snapshots).to_csv(ranking_path, index=False)
        paths["ranked_dca_monthly_rankings"] = ranking_path

    if args.run_exit_rs:
        rs_strategy = ConcentratedRelativeStrengthDcaStrategy(
            monthly_contribution=args.monthly_deposit,
            top_n=args.rs_top_n,
            keep_top_n=args.rs_keep_top_n,
            cash_reserve=args.cash_reserve,
            allocations=tuple(parse_allocations(args.rs_allocations, args.rs_top_n)),
            require_uptrend=not args.rs_no_trend_filter,
            min_order_value=args.min_order_value,
        )
        rs_result = BacktestEngine(data, config=config).run(rs_strategy)
        results.append(("experimental_exit_relative_strength_dca", rs_result))
        paths.update(
            {
                f"experimental_exit_relative_strength_dca_{name}": path
                for name, path in rs_result.write_report(output_dir, prefix="experimental_exit_relative_strength_dca").items()
            }
        )
        if rs_strategy.rank_snapshots:
            ranking_path = output_dir / "experimental_exit_relative_strength_monthly_rankings.csv"
            pd.DataFrame(rs_strategy.rank_snapshots).to_csv(ranking_path, index=False)
            paths["experimental_exit_relative_strength_monthly_rankings"] = ranking_path

    if not args.no_benchmarks:
        available_benchmarks = [symbol for symbol in benchmark_symbols if symbol in data]
        if len(available_benchmarks) > 1:
            benchmark_result = BacktestEngine(data, config=config).run(
                MonthlyBuyHoldStrategy(tuple(available_benchmarks), cash_reserve=args.cash_reserve, min_order_value=args.min_order_value)
            )
            results.append(("benchmark_equal_weight_buy_hold", benchmark_result))
            paths.update(
                {
                    f"benchmark_equal_weight_buy_hold_{key}": path
                    for key, path in benchmark_result.write_report(output_dir, prefix="benchmark_equal_weight_buy_hold").items()
                }
            )
        for symbol in available_benchmarks:
            benchmark_result = BacktestEngine(data, config=config).run(
                MonthlyBuyHoldStrategy((symbol,), cash_reserve=args.cash_reserve, min_order_value=args.min_order_value)
            )
            name = f"benchmark_buy_hold_{symbol}"
            results.append((name, benchmark_result))
            paths.update({f"{name}_{key}": path for key, path in benchmark_result.write_report(output_dir, prefix=name).items()})

    summary = summarize(result)
    print()
    print(f"{strategy_name.replace('_', ' ')} summary")
    print(pd.DataFrame([summary]).to_string(index=False))

    final_positions = latest_positions(result.positions)
    print()
    print("final positions")
    if final_positions.empty:
        print("none")
    else:
        print(final_positions.to_string(index=False))

    yearly = yearly_summary(result.equity_curve)
    print()
    print("yearly summary")
    print(yearly.to_string(index=False))

    comparison = comparison_table(results)
    comparison_path = output_dir / "strategy_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    paths["comparison"] = comparison_path
    print()
    print("strategy comparison")
    print(comparison.to_string(index=False))

    print()
    print("reports")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


def summarize(result) -> dict[str, Any]:
    metrics = dict(result.metrics)
    contributions = metrics.get("total_contributions", 0.0)
    final_equity = metrics.get("final_equity", 0.0)
    trades = result.trades
    buys = trades[trades["side"] == "buy"] if not trades.empty else pd.DataFrame()
    return {
        "start": result.equity_curve["date"].iloc[0].date().isoformat(),
        "end": result.equity_curve["date"].iloc[-1].date().isoformat(),
        "total_contributions": round(contributions, 2),
        "final_equity": round(final_equity, 2),
        "net_profit": round(final_equity - contributions, 2),
        "return_on_contributions_pct": round(metrics.get("total_return", 0.0) * 100, 2),
        "contribution_adjusted_cagr_pct": round(metrics.get("cagr", 0.0) * 100, 2),
        "xirr_pct": round(metrics["xirr"] * 100, 2) if metrics.get("xirr") is not None else None,
        "max_drawdown_pct": round(metrics.get("max_drawdown", 0.0) * 100, 2),
        "trade_count": int(metrics.get("trade_count", 0)),
        "buy_count": int(len(buys)),
        "cash_utilization_pct": round(metrics.get("cash_utilization", 0.0) * 100, 2),
    }


def comparison_table(results: list[tuple[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, result in results:
        summary = summarize(result)
        rows.append(
            {
                "strategy": name,
                "final_equity": summary["final_equity"],
                "total_contributions": summary["total_contributions"],
                "net_profit": summary["net_profit"],
                "return_on_contributions_pct": summary["return_on_contributions_pct"],
                "xirr_pct": summary["xirr_pct"],
                "max_drawdown_pct": summary["max_drawdown_pct"],
                "trade_count": summary["trade_count"],
                "cash_utilization_pct": summary["cash_utilization_pct"],
            }
        )
    return pd.DataFrame(rows).sort_values(["xirr_pct", "final_equity"], ascending=[False, False], na_position="last").reset_index(drop=True)


def latest_positions(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return positions
    last_date = positions["date"].max()
    frame = positions[positions["date"] == last_date].copy()
    total_value = frame["market_value"].sum()
    frame["weight_pct"] = frame["market_value"] / total_value * 100 if total_value else 0.0
    columns = ["date", "symbol", "shares", "avg_price", "close", "market_value", "unrealized_pnl", "weight_pct"]
    for column in ("avg_price", "close", "market_value", "unrealized_pnl", "weight_pct"):
        frame[column] = frame[column].round(2)
    return frame[columns].sort_values("market_value", ascending=False)


def yearly_summary(equity_curve: pd.DataFrame) -> pd.DataFrame:
    frame = equity_curve.copy()
    frame["year"] = pd.to_datetime(frame["date"]).dt.year
    grouped = frame.groupby("year", as_index=False).agg(
        ending_equity=("equity", "last"),
        contributions=("contribution", "sum"),
        ending_contributions=("cumulative_contributions", "last"),
        max_drawdown=("drawdown", "min"),
        cash_utilization=("invested_value", lambda values: 0.0),
    )
    invested_ratio = frame.assign(ratio=frame["invested_value"] / frame["equity"].replace(0, pd.NA))
    utilization = invested_ratio.groupby("year")["ratio"].mean().fillna(0.0) * 100
    grouped["cash_utilization_pct"] = grouped["year"].map(utilization).round(2)
    grouped["profit_vs_contributions"] = grouped["ending_equity"] - grouped["ending_contributions"]
    grouped["return_on_contributions_pct"] = grouped["profit_vs_contributions"] / grouped["ending_contributions"].replace(0, pd.NA) * 100
    grouped["max_drawdown_pct"] = grouped["max_drawdown"] * 100
    for column in ("ending_equity", "contributions", "ending_contributions", "profit_vs_contributions", "return_on_contributions_pct", "max_drawdown_pct"):
        grouped[column] = grouped[column].round(2)
    return grouped[
        [
            "year",
            "contributions",
            "ending_contributions",
            "ending_equity",
            "profit_vs_contributions",
            "return_on_contributions_pct",
            "max_drawdown_pct",
            "cash_utilization_pct",
        ]
    ]


def rank_context(history: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for symbol, frame in history.items():
        if frame.empty:
            continue
        close = pd.to_numeric(frame["close"], errors="coerce")
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        latest_close = last_valid(close)
        ema20 = last_valid(ema(close, 20))
        sma100 = last_valid(sma(close, 100))
        sma200 = last_valid(sma(close, 200))
        mom3m = return_over_available(close, 63)
        mom6m = return_over_available(close, 126)
        vol60 = ann_vol(close, 60) or 0.0
        liquidity60 = avg_traded_value(close, volume, 60)
        if any(value is None for value in (latest_close, liquidity60)):
            continue
        rows.append(
            {
                "symbol": symbol,
                "close": latest_close,
                "mom_3m": mom3m,
                "mom_6m": mom6m,
                "trend_score": trend_score(latest_close, sma100, sma200),
                "dist_ema20": distance(latest_close, ema20),
                "rsi14": last_valid(rsi(close, 14)),
                "vol_60d": vol60,
                "liquidity_60d": liquidity60,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["mom_6m_rank_score"] = percentile_scores(frame, "mom_6m", higher_is_better=True)
    frame["mom_3m_rank_score"] = percentile_scores(frame, "mom_3m", higher_is_better=True)
    frame["low_vol_rank_score"] = percentile_scores(frame, "vol_60d", higher_is_better=False)
    frame["liquidity_rank_score"] = percentile_scores(frame, "liquidity_60d", higher_is_better=True)
    frame["score"] = (
        RANK_WEIGHTS["mom_6m"] * frame["mom_6m_rank_score"]
        + RANK_WEIGHTS["mom_3m"] * frame["mom_3m_rank_score"]
        + RANK_WEIGHTS["trend"] * frame["trend_score"]
        + RANK_WEIGHTS["low_vol"] * frame["low_vol_rank_score"]
        + RANK_WEIGHTS["liquidity"] * frame["liquidity_rank_score"]
    )
    frame = frame.sort_values("score", ascending=False).reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame


def relative_strength_ranking(history: dict[str, pd.DataFrame], *, trend_sma: int = 200) -> pd.DataFrame:
    rows = []
    for symbol, frame in history.items():
        if frame.empty:
            continue
        close = pd.to_numeric(frame["close"], errors="coerce")
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        latest_close = last_valid(close)
        sma_value = last_valid(sma(close, trend_sma))
        liquidity60 = avg_traded_value(close, volume, 60)
        if latest_close is None or liquidity60 is None:
            continue

        rows.append(
            {
                "symbol": symbol,
                "close": latest_close,
                "mom_3m": return_over_available(close, 63),
                "mom_6m": return_over_available(close, 126),
                "mom_12m": return_over_available(close, 252),
                "drawdown_252d": rolling_drawdown(close, 252),
                "sma200": sma_value,
                "is_uptrend": latest_close >= sma_value if sma_value is not None else True,
                "liquidity_60d": liquidity60,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["mom_12m_rank_score"] = percentile_scores(frame, "mom_12m", higher_is_better=True)
    frame["mom_6m_rank_score"] = percentile_scores(frame, "mom_6m", higher_is_better=True)
    frame["mom_3m_rank_score"] = percentile_scores(frame, "mom_3m", higher_is_better=True)
    frame["low_drawdown_rank_score"] = percentile_scores(frame, "drawdown_252d", higher_is_better=True)
    frame["trend_rank_score"] = frame["is_uptrend"].astype(float)
    frame["score"] = (
        RELATIVE_STRENGTH_WEIGHTS["mom_12m"] * frame["mom_12m_rank_score"]
        + RELATIVE_STRENGTH_WEIGHTS["mom_6m"] * frame["mom_6m_rank_score"]
        + RELATIVE_STRENGTH_WEIGHTS["mom_3m"] * frame["mom_3m_rank_score"]
        + RELATIVE_STRENGTH_WEIGHTS["trend"] * frame["trend_rank_score"]
        + RELATIVE_STRENGTH_WEIGHTS["low_drawdown"] * frame["low_drawdown_rank_score"]
    )
    frame = frame.sort_values(["score", "liquidity_60d"], ascending=[False, False]).reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame


def is_entry_allowed(row: pd.Series, min_trend_score: float, max_ema20_extension: float, max_rsi: float) -> bool:
    if float(row.get("trend_score") or 0.0) < min_trend_score:
        return False
    dist_ema20 = row.get("dist_ema20")
    if pd.notna(dist_ema20) and float(dist_ema20) > max_ema20_extension:
        return False
    rsi14 = row.get("rsi14")
    if pd.notna(rsi14) and float(rsi14) > max_rsi:
        return False
    return True


def parse_allocations(value: str, top_n: int) -> list[float]:
    allocations = [float(part.strip()) for part in value.split(",") if part.strip()]
    allocations = [allocation / 100 if allocation > 1 else allocation for allocation in allocations]
    if len(allocations) < top_n:
        allocations.extend([0.0] * (top_n - len(allocations)))
    return allocations[:top_n]


def unique_symbols(symbols: list[str]) -> list[str]:
    return list(dict.fromkeys(symbol.upper() for symbol in symbols if symbol))


def percentile_scores(frame: pd.DataFrame, column: str, *, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    ranks = values.rank(method="min", ascending=not higher_is_better, na_option="bottom")
    return ((len(frame) - ranks + 1) / len(frame)).clip(lower=0.0, upper=1.0)


def sma(series: pd.Series, window: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    delta = values.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window, min_periods=window).mean()
    rs = gain / loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def return_over(series: pd.Series, lookback: int) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= lookback:
        return None
    past = float(values.iloc[-lookback])
    if past == 0:
        return None
    return float(values.iloc[-1]) / past - 1.0


def return_over_available(series: pd.Series, lookback: int) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return 0.0
    effective_lookback = min(lookback, len(values) - 1)
    past = float(values.iloc[-effective_lookback - 1])
    if past == 0:
        return 0.0
    return float(values.iloc[-1]) / past - 1.0


def ann_vol(series: pd.Series, lookback: int) -> float | None:
    returns = pd.to_numeric(series, errors="coerce").pct_change().dropna().tail(lookback)
    if len(returns) < max(20, lookback // 2):
        return None
    return float(returns.std() * math.sqrt(252))


def avg_traded_value(close: pd.Series, volume: pd.Series, lookback: int) -> float | None:
    value = (pd.to_numeric(close, errors="coerce") * pd.to_numeric(volume, errors="coerce")).dropna().tail(lookback)
    if value.empty:
        return None
    return float(value.mean())


def rolling_drawdown(series: pd.Series, lookback: int) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().tail(lookback)
    if values.empty:
        return 0.0
    peak = float(values.max())
    if peak <= 0:
        return 0.0
    return float(values.iloc[-1]) / peak - 1.0


def trend_score(close: float | None, sma100: float | None, sma200: float | None) -> float:
    if close is None:
        return 0.0
    score = 0.0
    if sma100 and close > sma100:
        score += 0.45
        score += min(max((close / sma100 - 1.0) / 0.15, 0.0), 1.0) * 0.10
    if sma200 and close > sma200:
        score += 0.35
        score += min(max((close / sma200 - 1.0) / 0.25, 0.0), 1.0) * 0.10
    return min(score, 1.0)


def distance(value: float | None, reference: float | None) -> float | None:
    if value is None or reference in (None, 0):
        return None
    return value / reference - 1.0


def last_valid(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


if __name__ == "__main__":
    raise SystemExit(main())
