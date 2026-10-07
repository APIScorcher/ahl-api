from __future__ import annotations

import tempfile
import unittest

import pandas as pd

from ahl_api.backtest import BacktestConfig, BacktestEngine, StrategyContext
from ahl_api.research_reporting import (
    block_bootstrap_returns,
    bootstrap_summary,
    parameter_stability_table,
    performance_by_market_regime,
    run_symbol_exclusion_tests,
    write_research_charts,
)


def frame(periods: int = 300, growth: float = 0.001) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=periods)
    close = pd.Series([(1.0 + growth) ** index * 100 for index in range(periods)])
    return pd.DataFrame({"date": dates, "open": close, "close": close, "volume": [100_000] * periods})


class OneShotWeights:
    def __init__(self, symbol: str = "AAA") -> None:
        self.symbol = symbol
        self.sent = False

    def target_weights(self, context: StrategyContext):
        if self.sent:
            return {}
        self.sent = True
        return {self.symbol: 1.0}


class ResearchReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = BacktestConfig(
            initial_capital=50_000,
            brokerage_bps=0,
            brokerage_per_share=0,
            sales_tax_rate=0,
        )
        self.result = BacktestEngine({"AAA": frame()}, config=self.config).run(OneShotWeights())

    def test_block_bootstrap_is_reproducible(self) -> None:
        first = block_bootstrap_returns(self.result.equity_curve, simulations=20, block_size=5, seed=9)
        second = block_bootstrap_returns(self.result.equity_curve, simulations=20, block_size=5, seed=9)

        pd.testing.assert_frame_equal(first, second)
        summary = bootstrap_summary(first)
        self.assertEqual(summary["simulations"], 20)
        self.assertIn("probability_positive", summary)

    def test_regime_and_parameter_stability_tables(self) -> None:
        regimes = performance_by_market_regime(self.result.equity_curve, frame(), sma_window=20)
        selections = pd.DataFrame(
            {
                "strategy": ["a", "a", "a"],
                "parameters": ["{\"x\": 1}", "{\"x\": 1}", "{\"x\": 2}"],
                "selection_score": [0.1, 0.2, -0.1],
                "total_return": [0.2, 0.1, -0.1],
            }
        )

        stability = parameter_stability_table(selections)

        self.assertFalse(regimes.empty)
        self.assertEqual(stability.iloc[0]["parameters"], "{\"x\": 1}")

    def test_symbol_exclusion_and_charts(self) -> None:
        data = {"AAA": frame(), "BBB": frame(growth=0.0005)}
        exclusions = run_symbol_exclusion_tests(
            data,
            OneShotWeights,
            {},
            config=self.config,
            protected_symbols=["AAA"],
        )
        self.assertEqual(list(exclusions["excluded_symbol"]), ["BBB"])

        bootstrap = block_bootstrap_returns(self.result.equity_curve, simulations=10, block_size=5)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_research_charts(directory, results={"strategy": self.result}, bootstrap_paths=bootstrap)
            self.assertTrue(paths["equity"].exists())
            self.assertTrue(paths["drawdown"].exists())
            self.assertTrue(paths["bootstrap"].exists())


if __name__ == "__main__":
    unittest.main()
