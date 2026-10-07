"""Integration checks for public exports and financial simulation invariants."""

import json
from pathlib import Path

import pandas as pd
import pytest

import ahl_api
from ahl_api.backtest import BacktestConfig, BacktestEngine
from ahl_api.research_strategies import CloseChannelBreakoutStrategy
from test_research_strategies import context, frame


def test_every_public_export_resolves():
    for name in ahl_api.__all__:
        assert getattr(ahl_api, name) is not None
    with pytest.raises(AttributeError):
        getattr(ahl_api, "not_an_export")


def test_close_breakout_enters_and_exits():
    strategy = CloseChannelBreakoutStrategy(
        entry_lookback=5,
        exit_lookback=3,
        trend_sma=5,
        liquidity_lookback=3,
        min_median_traded_value=0,
        min_price=0,
        market_symbol=None,
        max_position_weight=1.0,
    )
    history = {"AAA": frame([100 + i for i in range(20)])}
    weights = strategy.target_weights(context(history))
    assert weights.get("AAA", 0) > 0
    from ahl_api.backtest import Position

    falling = {"AAA": frame([100 + i for i in range(19)] + [90])}
    weights = strategy.target_weights(context(falling, positions={"AAA": Position("AAA", 1, 100)}))
    assert weights.get("AAA") == 0


@pytest.mark.parametrize("name", ["initial_capital", "slippage_bps", "brokerage_bps", "sales_tax_rate"])
def test_nonfinite_backtest_inputs_rejected(name):
    with pytest.raises(ValueError):
        BacktestEngine({"AAA": frame([100, 101, 102])}, BacktestConfig(**{name: float("nan")}))


def test_nonfinite_target_weights_rejected():
    class InvalidWeights:
        def target_weights(self, context):
            return {"AAA": float("nan")}

    with pytest.raises(ValueError):
        BacktestEngine({"AAA": frame([100, 101, 102])}).run(InvalidWeights())


def test_strategy_history_never_contains_future_bars():
    class InspectHistory:
        def target_weights(self, context):
            for data in context.history.values():
                assert pd.to_datetime(data["date"]).max() <= context.date
            return {}

    BacktestEngine({"AAA": frame([100, 101, 102, 103])}).run(InspectHistory())


def test_chart_optional_paths(tmp_path):
    from ahl_api.research_reporting import write_json, write_research_charts

    folds = pd.DataFrame(
        {"fold": [1, 2], "strategy": ["demo", "demo"], "scenario": ["base", "base"], "total_return": [0.1, -0.1]}
    )
    stability = pd.DataFrame({"strategy": ["demo"], "median_selection_score": [0.1]})
    paths = write_research_charts(tmp_path, folds=folds, stability=stability)
    assert all(Path(p).stat().st_size > 100 for p in paths.values())
    output = write_json(tmp_path / "demo.json", {"date": pd.Timestamp("2026-01-01")})
    assert json.loads(output.read_text())["date"].startswith("2026-01-01")
