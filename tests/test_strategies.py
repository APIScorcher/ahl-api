from __future__ import annotations

import unittest

import pandas as pd

from ahl_api.backtest import BacktestConfig, BacktestEngine, Position, StrategyContext, run_parameter_sweep
from ahl_api.strategies import (
    LiquidityFilteredMomentumStrategy,
    MomentumTrendFilterStrategy,
    PullbackDcaStrategy,
    average_traded_value,
    ema,
    rank_momentum_universe,
    rate_of_change,
    sma,
)


def frame(symbol: str, closes: list[float], *, volumes: list[int] | None = None, start: str = "2026-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=len(closes))
    volumes = volumes or [10_000] * len(closes)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": closes,
            "high": [None] * len(closes),
            "low": [None] * len(closes),
            "close": closes,
            "volume": volumes,
        }
    )


def context(date: str, data: dict[str, pd.DataFrame], *, positions=None, cash: float = 1000.0, equity: float = 1000.0) -> StrategyContext:
    current = pd.Timestamp(date)
    history = {symbol: bars[bars["date"] <= current].copy() for symbol, bars in data.items()}
    prices = {
        symbol: float(symbol_history["close"].iloc[-1])
        for symbol, symbol_history in history.items()
        if not symbol_history.empty
    }
    return StrategyContext(
        date=current,
        history=history,
        prices=prices,
        positions=positions or {},
        available_cash=cash,
        unsettled_cash=0.0,
        equity=equity,
    )


def zero_cost_config() -> BacktestConfig:
    return BacktestConfig(
        initial_capital=1000,
        brokerage_bps=0,
        brokerage_per_share=0,
        sales_tax_rate=0,
        other_fees_bps=0,
    )


class StrategyTests(unittest.TestCase):
    def test_indicators(self) -> None:
        values = pd.Series([10, 12, 15, 18])

        self.assertEqual(sma(values, 2).iloc[-1], 16.5)
        self.assertAlmostEqual(rate_of_change(values, 2).iloc[-1], 0.5)
        self.assertFalse(pd.isna(ema(values, 2).iloc[-1]))
        traded = average_traded_value(pd.DataFrame({"close": [10, 20], "volume": [100, 200]}), 2)
        self.assertEqual(traded.iloc[-1], 2500)

    def test_momentum_ranking_respects_top_n(self) -> None:
        data = {
            "AAA": frame("AAA", [10, 11, 15]),
            "BBB": frame("BBB", [10, 12, 13]),
            "CCC": frame("CCC", [10, 9, 9]),
        }

        ranked = rank_momentum_universe(data, top_n=2, momentum_lookback=2, liquidity_lookback=2)

        self.assertEqual([item.symbol for item in ranked], ["AAA", "BBB"])

    def test_liquidity_filter_excludes_thin_symbols(self) -> None:
        data = {
            "LIQ": frame("LIQ", [10, 11, 15], volumes=[1000, 1000, 1000]),
            "THIN": frame("THIN", [10, 20, 40], volumes=[1, 1, 1]),
        }

        strategy = LiquidityFilteredMomentumStrategy(
            top_n=2,
            momentum_lookback=2,
            liquidity_lookback=2,
            min_avg_traded_value=10_000,
            rebalance="daily",
        )
        weights = strategy.target_weights(context("2026-01-05", data))

        self.assertEqual(weights, {"LIQ": 1.0})

    def test_monthly_rebalance_emits_once_per_month(self) -> None:
        data = {"AAA": frame("AAA", [10, 11, 12, 13, 14, 15], start="2026-01-28")}
        strategy = LiquidityFilteredMomentumStrategy(top_n=1, momentum_lookback=1, liquidity_lookback=1, rebalance="monthly")

        first = strategy.target_weights(context("2026-01-29", data))
        second = strategy.target_weights(context("2026-01-30", data))
        next_month = strategy.target_weights(context("2026-02-02", data))

        self.assertEqual(first, {"AAA": 1.0})
        self.assertEqual(second, {})
        self.assertEqual(next_month, {"AAA": 1.0})

    def test_trend_filter_excludes_symbols_below_sma(self) -> None:
        data = {
            "UP": frame("UP", [10, 11, 12, 13]),
            "DOWN": frame("DOWN", [13, 12, 11, 10]),
        }
        strategy = MomentumTrendFilterStrategy(
            top_n=2,
            momentum_lookback=1,
            liquidity_lookback=1,
            trend_sma=3,
            rebalance="daily",
        )

        weights = strategy.target_weights(context("2026-01-06", data))

        self.assertEqual(weights, {"UP": 1.0})

    def test_pullback_dca_emits_tranches_and_does_not_duplicate(self) -> None:
        data = {"AAA": frame("AAA", [50, 250, 151])}
        strategy = PullbackDcaStrategy(
            top_n=1,
            momentum_lookback=1,
            liquidity_lookback=1,
            trend_sma=3,
            dip_ema=2,
            dip_thresholds=(0.03, 0.06),
            max_position_weight=0.20,
            universe_rebalance="daily",
        )
        ctx = context("2026-01-05", data, equity=1000)

        first = strategy.orders(ctx)
        second = strategy.orders(ctx)

        self.assertEqual([order.tag for order in first], ["dca_buy_3pct", "dca_buy_6pct"])
        self.assertEqual([order.value for order in first], [100.0, 100.0])
        self.assertEqual(second, [])

    def test_pullback_dca_exits_on_trend_break(self) -> None:
        data = {"AAA": frame("AAA", [100, 100, 80])}
        position = Position("AAA", shares=5, avg_price=100, close=80, market_value=400, unrealized_pnl=-100)
        strategy = PullbackDcaStrategy(top_n=1, momentum_lookback=1, liquidity_lookback=1, trend_sma=2, dip_ema=2)

        orders = strategy.orders(context("2026-01-05", data, positions={"AAA": position}))

        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].side, "sell")
        self.assertEqual(orders[0].tag, "trend_exit")

    def test_pullback_dca_exits_on_stretched_weakening_close(self) -> None:
        data = {"AAA": frame("AAA", [100, 140, 130])}
        position = Position("AAA", shares=5, avg_price=100, close=130, market_value=650, unrealized_pnl=150)
        strategy = PullbackDcaStrategy(
            top_n=1,
            momentum_lookback=1,
            liquidity_lookback=1,
            trend_sma=3,
            dip_ema=2,
            take_profit_above_ema=0.0,
        )

        orders = strategy.orders(context("2026-01-05", data, positions={"AAA": position}))

        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].tag, "take_profit")

    def test_strategies_run_with_engine_and_sweeps(self) -> None:
        data = {
            "AAA": frame("AAA", [10, 11, 12, 13, 14]),
            "BBB": frame("BBB", [10, 10, 10, 10, 10]),
        }
        engine = BacktestEngine(data, config=zero_cost_config())
        result = engine.run(
            LiquidityFilteredMomentumStrategy(top_n=1, momentum_lookback=1, liquidity_lookback=1, rebalance="daily")
        )

        self.assertGreaterEqual(result.metrics["trade_count"], 1)

        sweep = run_parameter_sweep(
            data,
            MomentumTrendFilterStrategy,
            {"trend_sma": [2, 3], "momentum_lookback": [1], "liquidity_lookback": [1], "rebalance": ["daily"]},
            config=zero_cost_config(),
        )
        self.assertEqual(len(sweep.runs), 2)


if __name__ == "__main__":
    unittest.main()
