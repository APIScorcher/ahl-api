from __future__ import annotations

import unittest

import pandas as pd

from ahl_api.backtest import Position, StrategyContext
from ahl_api.research_strategies import (
    BollingerTrendReversionStrategy,
    CrossSectionalMomentumStrategy,
    DataUnavailableError,
    RsiTrendPullbackStrategy,
    TechnicalEnsembleStrategy,
    lagged_return,
    require_high_low,
    rsi,
)


def frame(closes: list[float], *, end: str = "2026-06-30", volume: int = 1_000_000) -> pd.DataFrame:
    dates = pd.bdate_range(end=end, periods=len(closes))
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": [None] * len(closes),
            "low": [None] * len(closes),
            "close": closes,
            "volume": [volume] * len(closes),
        }
    )


def context(data: dict[str, pd.DataFrame], *, positions=None, cash: float = 50_000) -> StrategyContext:
    date = max(pd.Timestamp(item["date"].max()) for item in data.values())
    prices = {
        symbol: float(item.loc[pd.to_datetime(item["date"]) <= date, "close"].iloc[-1])
        for symbol, item in data.items()
    }
    return StrategyContext(date, data, prices, positions or {}, cash, 0.0, 50_000.0)


class ResearchStrategyTests(unittest.TestCase):
    def test_indicator_helpers(self) -> None:
        values = pd.Series([100, 102, 104, 106, 108, 110])

        self.assertAlmostEqual(lagged_return(values, 3, 1), 108 / 102 - 1)
        self.assertEqual(rsi(values, 3).iloc[-1], 100.0)

    def test_momentum_selects_positive_current_liquid_symbol(self) -> None:
        market = [100 + index * 0.2 for index in range(260)]
        winner = [100 + index * 0.5 for index in range(260)]
        loser = [150 - index * 0.2 for index in range(260)]
        data = {"KSE100PR": frame(market), "WIN": frame(winner), "LOSE": frame(loser)}
        position = Position("LOSE", 10, 100, 98, 980, -20)
        strategy = CrossSectionalMomentumStrategy(
            top_n=1,
            momentum_lookback=20,
            momentum_skip=0,
            trend_sma=20,
            liquidity_lookback=5,
            min_median_traded_value=0,
            min_price=0,
            volatility_lookback=5,
            max_gross_exposure=0.9,
            max_position_weight=1.0,
            market_sma=20,
        )

        weights = strategy.target_weights(context(data, positions={"LOSE": position}))

        self.assertEqual(weights["LOSE"], 0.0)
        self.assertAlmostEqual(weights["WIN"], 0.9)

    def test_negative_market_regime_liquidates_positions(self) -> None:
        market = [200 - index * 0.3 for index in range(260)]
        stock = [100 + index * 0.4 for index in range(260)]
        position = Position("AAA", 10, 100, 150, 1500, 500)
        strategy = CrossSectionalMomentumStrategy(
            momentum_lookback=20,
            momentum_skip=0,
            trend_sma=20,
            liquidity_lookback=5,
            volatility_lookback=5,
            market_sma=20,
        )

        weights = strategy.target_weights(context({"KSE100PR": frame(market), "AAA": frame(stock)}, positions={"AAA": position}))

        self.assertEqual(weights, {"AAA": 0.0})

    def test_stale_symbol_is_not_selected(self) -> None:
        market = [100 + index for index in range(50)]
        stale = frame([100 + index * 2 for index in range(50)], end="2026-06-29")
        strategy = CrossSectionalMomentumStrategy(
            top_n=1,
            momentum_lookback=10,
            momentum_skip=0,
            trend_sma=10,
            liquidity_lookback=5,
            volatility_lookback=5,
            min_median_traded_value=0,
            min_price=0,
            market_sma=10,
        )

        weights = strategy.target_weights(context({"KSE100PR": frame(market), "STALE": stale}))

        self.assertEqual(weights, {})

    def test_rsi_pullback_generates_recovery_entry(self) -> None:
        market = [100 + index * 0.2 for index in range(80)]
        stock = [100 + index * 0.5 for index in range(70)] + [134, 132, 130, 131]
        strategy = RsiTrendPullbackStrategy(
            rsi_period=2,
            oversold_rsi=30,
            recovery_ceiling_rsi=60,
            oversold_window=3,
            trend_sma=20,
            trend_slope_lookback=5,
            exit_ema=5,
            liquidity_lookback=5,
            min_median_traded_value=0,
            min_price=0,
            market_sma=20,
        )

        orders = strategy.orders(context({"KSE100PR": frame(market), "AAA": frame(stock)}))

        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].side, "buy")
        self.assertEqual(orders[0].tag, "rsi_pullback_entry")

    def test_bollinger_and_ensemble_do_not_use_future_or_intraday_fields(self) -> None:
        market = [100 + index * 0.2 for index in range(240)]
        stock = [100 + index * 0.4 for index in range(235)] + [190, 185, 180, 185, 192]
        data = {"KSE100PR": frame(market), "AAA": frame(stock)}

        bollinger = BollingerTrendReversionStrategy(
            band_window=5,
            band_std=1.0,
            trend_sma=20,
            liquidity_lookback=5,
            min_median_traded_value=0,
            min_price=0,
            market_sma=20,
        )
        ensemble = TechnicalEnsembleStrategy(
            top_n=1,
            momentum_lookback=20,
            trend_sma=20,
            breakout_lookback=10,
            liquidity_lookback=5,
            min_median_traded_value=0,
            min_price=0,
            market_sma=20,
            min_votes=2,
        )

        self.assertIsInstance(bollinger.orders(context(data)), list)
        self.assertIn("AAA", ensemble.target_weights(context(data)))

    def test_high_low_strategy_guard_fails_clearly(self) -> None:
        with self.assertRaises(DataUnavailableError):
            require_high_low({"AAA": frame([100, 101, 102])}, "ATR breakout")


if __name__ == "__main__":
    unittest.main()
