from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ahl_api.backtest import (
    BacktestConfig,
    BacktestEngine,
    Order,
    StrategyContext,
    calculate_xirr,
    calculate_trade_cost,
    load_historical_data,
    run_parameter_sweep,
)


def zero_cost_config(**kwargs) -> BacktestConfig:
    values = {
        "initial_capital": 1_000.0,
        "brokerage_bps": 0.0,
        "brokerage_per_share": 0.0,
        "sales_tax_rate": 0.0,
        "slippage_bps": 0.0,
        "other_fees_bps": 0.0,
    }
    values.update(kwargs)
    return BacktestConfig(**values)


def bars(symbol: str, opens: list[float], closes: list[float] | None = None) -> pd.DataFrame:
    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"][: len(opens)])
    closes = closes or opens
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": opens,
            "high": [None] * len(opens),
            "low": [None] * len(opens),
            "close": closes,
            "volume": [100_000] * len(opens),
        }
    )


class OneShotWeights:
    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = weights
        self.sent = False

    def target_weights(self, context: StrategyContext) -> dict[str, float]:
        if self.sent:
            return {}
        self.sent = True
        return self.weights


class OneShotOrders:
    def __init__(self, orders: list[Order]) -> None:
        self._orders = orders
        self.sent = False

    def orders(self, context: StrategyContext) -> list[Order]:
        if self.sent:
            return []
        self.sent = True
        return self._orders


class DateOrders:
    def __init__(self, orders_by_date: dict[str, list[Order]]) -> None:
        self.orders_by_date = orders_by_date

    def orders(self, context: StrategyContext) -> list[Order]:
        key = context.date.strftime("%Y-%m-%d")
        return self.orders_by_date.get(key, [])


class SizedBuy:
    def __init__(self, shares: int = 1) -> None:
        self.shares = shares
        self.sent = False

    def orders(self, context: StrategyContext) -> list[Order]:
        if self.sent:
            return []
        self.sent = True
        return [Order("OGDC", "buy", shares=self.shares)]


class WeightsAndOrders:
    def __init__(self) -> None:
        self.sent = False

    def target_weights(self, context: StrategyContext) -> dict[str, float]:
        if self.sent:
            return {}
        return {"A": 0.50}

    def orders(self, context: StrategyContext) -> list[Order]:
        if self.sent:
            return []
        self.sent = True
        return [Order("B", "buy", shares=1)]


class SuspensionRebalance:
    def __init__(self) -> None:
        self.buy_sent = False
        self.rebalance_sent = False

    def orders(self, context: StrategyContext) -> list[Order]:
        if context.date == pd.Timestamp("2026-01-01") and not self.buy_sent:
            self.buy_sent = True
            return [Order("A", "buy", shares=5)]
        return []

    def target_weights(self, context: StrategyContext) -> dict[str, float]:
        if context.date == pd.Timestamp("2026-01-02") and not self.rebalance_sent:
            self.rebalance_sent = True
            return {"A": 0.0, "B": 0.5}
        return {}


class TargetRotationAfterBuy:
    def __init__(self) -> None:
        self.buy_sent = False
        self.target_sent = False

    def orders(self, context: StrategyContext) -> list[Order]:
        if context.date == pd.Timestamp("2026-01-01") and not self.buy_sent:
            self.buy_sent = True
            return [Order("A", "buy", shares=10)]
        return []

    def target_weights(self, context: StrategyContext) -> dict[str, float]:
        if context.date == pd.Timestamp("2026-01-02") and not self.target_sent:
            self.target_sent = True
            return {"B": 1.0}
        return {}


class FakeClient:
    def fetch_historical_daily(self, symbol, *, since=None, until=None, years=None):
        return [
            {"date": "2026-01-01", "open": 10, "high": None, "low": None, "close": 11, "volume": 1000},
            {"date": "2026-01-02", "open": 12, "high": None, "low": None, "close": 13, "volume": 2000},
        ]


class BacktestEngineTests(unittest.TestCase):
    def test_target_weights_execute_at_next_open(self) -> None:
        engine = BacktestEngine({"OGDC": bars("OGDC", [100, 110, 120])}, config=zero_cost_config())
        result = engine.run(OneShotWeights({"OGDC": 0.5}))

        self.assertEqual(len(result.trades), 1)
        trade = result.trades.iloc[0]
        self.assertEqual(trade["fill_date"], pd.Timestamp("2026-01-02"))
        self.assertEqual(trade["price"], 110)
        self.assertEqual(trade["shares"], 4)
        self.assertAlmostEqual(result.equity_curve.iloc[-1]["equity"], 1040.0)
        self.assertEqual(result.metrics["trade_count"], 1)

    def test_explicit_orders_partially_fill_when_cash_is_insufficient(self) -> None:
        engine = BacktestEngine({"HBL": bars("HBL", [100, 100, 100])}, config=zero_cost_config(initial_capital=250))
        result = engine.run(OneShotOrders([Order("HBL", "buy", shares=10)]))

        self.assertEqual(result.orders.iloc[0]["status"], "partially_filled")
        self.assertEqual(result.orders.iloc[0]["filled_shares"], 2)
        self.assertEqual(result.trades.iloc[0]["shares"], 2)
        self.assertGreaterEqual(result.equity_curve.iloc[-1]["available_cash"], 0)

    def test_sell_without_position_is_rejected_and_does_not_short(self) -> None:
        engine = BacktestEngine({"FFC": bars("FFC", [100, 100, 100])}, config=zero_cost_config())
        result = engine.run(OneShotOrders([Order("FFC", "sell", shares=5)]))

        self.assertEqual(result.orders.iloc[0]["status"], "rejected")
        self.assertEqual(result.orders.iloc[0]["reason"], "no_position")
        self.assertTrue(result.trades.empty)
        self.assertTrue(result.positions.empty)

    def test_sell_cash_settles_next_trading_day(self) -> None:
        data = {"A": bars("A", [100, 100, 100, 100, 100]), "B": bars("B", [100, 100, 100, 100, 100])}
        strategy = DateOrders(
            {
                "2026-01-01": [Order("A", "buy", shares=10)],
                "2026-01-02": [Order("A", "sell", shares=10), Order("B", "buy", shares=10)],
            }
        )
        engine = BacktestEngine(data, config=zero_cost_config())
        result = engine.run(strategy)

        same_day_buy = result.orders[(result.orders["symbol"] == "B")].iloc[0]
        self.assertEqual(same_day_buy["status"], "rejected")
        self.assertEqual(same_day_buy["reason"], "insufficient_cash")

        curve = result.equity_curve.set_index("date")
        self.assertEqual(curve.loc[pd.Timestamp("2026-01-05"), "available_cash"], 0)
        self.assertEqual(curve.loc[pd.Timestamp("2026-01-05"), "unsettled_cash"], 1000)
        self.assertEqual(curve.loc[pd.Timestamp("2026-01-06"), "available_cash"], 1000)

    def test_monthly_recurring_contributions_are_added_to_cash(self) -> None:
        data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-02-02", "2026-02-03"]),
                "open": [100, 100, 100, 100],
                "close": [100, 100, 100, 100],
                "volume": [1000, 1000, 1000, 1000],
            }
        )
        engine = BacktestEngine(
            {"A": data},
            config=zero_cost_config(initial_capital=0, recurring_contribution=200, contribution_frequency="monthly"),
        )
        result = engine.run(OneShotOrders([]))
        curve = result.equity_curve.set_index("date")

        self.assertEqual(curve.loc[pd.Timestamp("2026-01-01"), "contribution"], 200)
        self.assertEqual(curve.loc[pd.Timestamp("2026-02-02"), "contribution"], 200)
        self.assertEqual(curve.iloc[-1]["available_cash"], 400)
        self.assertEqual(result.metrics["total_contributions"], 400)
        self.assertAlmostEqual(result.metrics["xirr"], 0.0, places=6)

    def test_xirr_calculates_money_weighted_return(self) -> None:
        cashflows = [
            (pd.Timestamp("2025-01-01"), -1_000.0),
            (pd.Timestamp("2026-01-01"), 1_100.0),
        ]

        self.assertAlmostEqual(calculate_xirr(cashflows), 0.10, places=3)

    def test_start_date_limits_results_without_removing_warmup_history(self) -> None:
        data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-12-31", "2026-01-01", "2026-01-02"]),
                "open": [100, 100, 100],
                "close": [100, 100, 100],
                "volume": [1000, 1000, 1000],
            }
        )

        class WarmupAware:
            def orders(self, context: StrategyContext) -> list[Order]:
                if context.date == pd.Timestamp("2026-01-01"):
                    self.history_rows = len(context.history["A"])
                return []

        strategy = WarmupAware()
        result = BacktestEngine(
            {"A": data},
            config=zero_cost_config(initial_capital=0, recurring_contribution=100, contribution_frequency="monthly", start_date="2026-01-01"),
        ).run(strategy)

        self.assertEqual(result.equity_curve.iloc[0]["date"], pd.Timestamp("2026-01-01"))
        self.assertEqual(result.equity_curve.iloc[-1]["available_cash"], 100)
        self.assertEqual(strategy.history_rows, 2)

    def test_orders_wait_for_symbol_next_available_open(self) -> None:
        data = {
            "A": pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-01-01", "2026-01-05"]),
                    "open": [100, 120],
                    "close": [100, 125],
                    "volume": [1000, 1000],
                }
            ),
            "B": bars("B", [50, 50, 50]),
        }
        engine = BacktestEngine(data, config=zero_cost_config())
        result = engine.run(OneShotOrders([Order("A", "buy", shares=2)]))

        self.assertEqual(result.orders.iloc[0]["status"], "filled")
        self.assertEqual(result.orders.iloc[0]["fill_date"], pd.Timestamp("2026-01-05"))
        self.assertEqual(result.trades.iloc[0]["price"], 120)

    def test_strategy_can_emit_weights_and_orders(self) -> None:
        engine = BacktestEngine({"A": bars("A", [100, 100, 100]), "B": bars("B", [50, 50, 50])}, config=zero_cost_config())
        result = engine.run(WeightsAndOrders())

        self.assertEqual(len(result.trades), 2)
        self.assertEqual(set(result.orders["source"]), {"target_weights", "orders"})
        self.assertEqual(set(result.trades["symbol"]), {"A", "B"})

    def test_fee_calculation_uses_minimum_commission_rule(self) -> None:
        low_price = calculate_trade_cost(100, 10)
        high_price = calculate_trade_cost(100, 100)

        self.assertAlmostEqual(low_price["brokerage"], 3.0)
        self.assertAlmostEqual(low_price["sales_tax"], 0.45)
        self.assertAlmostEqual(low_price["total_fees"], 3.45)
        self.assertAlmostEqual(high_price["brokerage"], 15.0)
        self.assertAlmostEqual(high_price["sales_tax"], 2.25)
        self.assertAlmostEqual(high_price["total_fees"], 17.25)

    def test_normalize_and_client_loader(self) -> None:
        loaded = load_historical_data(FakeClient(), ["ogdc"])

        self.assertEqual(list(loaded), ["OGDC"])
        self.assertEqual(list(loaded["OGDC"].columns), ["date", "symbol", "open", "high", "low", "close", "volume"])
        self.assertEqual(loaded["OGDC"].iloc[0]["symbol"], "OGDC")

    def test_invalid_target_weights_reject_leverage(self) -> None:
        engine = BacktestEngine({"A": bars("A", [100, 100, 100]), "B": bars("B", [100, 100, 100])}, config=zero_cost_config())
        with self.assertRaises(ValueError):
            engine.run(OneShotWeights({"A": 0.75, "B": 0.50}))

    def test_metrics_and_result_shapes(self) -> None:
        engine = BacktestEngine({"OGDC": bars("OGDC", [100, 100, 130, 130])}, config=zero_cost_config())
        result = engine.run(OneShotOrders([Order("OGDC", "buy", shares=5)]))

        self.assertIn("daily_return", result.equity_curve.columns)
        self.assertIn("drawdown", result.equity_curve.columns)
        self.assertIn("final_equity", result.metrics)
        self.assertEqual(result.metrics["final_equity"], result.equity_curve.iloc[-1]["equity"])
        self.assertGreater(result.metrics["total_return"], 0)

    def test_parameter_sweep_runs_cartesian_grid_and_finds_best(self) -> None:
        data = {"OGDC": bars("OGDC", [100, 100, 130, 130])}
        result = run_parameter_sweep(
            data,
            SizedBuy,
            {"shares": [1, 2, 3]},
            config=zero_cost_config(),
            config_grid={"slippage_bps": [0.0, 10.0]},
        )

        self.assertEqual(len(result.runs), 6)
        self.assertEqual(len(result.summary), 6)
        self.assertIn("config.slippage_bps", result.summary.columns)
        self.assertEqual(result.best("final_equity").parameters["shares"], 3)

    def test_parameter_sweep_rejects_bad_config_field(self) -> None:
        with self.assertRaises(ValueError):
            run_parameter_sweep(
                {"OGDC": bars("OGDC", [100, 100, 130])},
                SizedBuy,
                {"shares": [1]},
                config=zero_cost_config(),
                config_grid={"not_a_field": [1]},
            )

    def test_backtest_report_writes_metrics_and_tables(self) -> None:
        engine = BacktestEngine({"OGDC": bars("OGDC", [100, 100, 130])}, config=zero_cost_config())
        result = engine.run(OneShotOrders([Order("OGDC", "buy", shares=2)]))

        with tempfile.TemporaryDirectory() as tmp:
            paths = result.write_report(tmp, prefix="demo")

            self.assertTrue(paths["metrics"].exists())
            self.assertTrue(paths["equity_curve"].exists())
            metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
            self.assertEqual(metrics["trade_count"], 1)
            self.assertIn("equity", pd.read_csv(paths["equity_curve"]).columns)

    def test_sweep_report_writes_summary_and_optional_details(self) -> None:
        sweep = run_parameter_sweep(
            {"OGDC": bars("OGDC", [100, 100, 130])},
            SizedBuy,
            {"shares": [1, 2]},
            config=zero_cost_config(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            paths = sweep.write_report(Path(tmp), include_run_details=True)

            self.assertTrue(paths["summary"].exists())
            self.assertIn("run_0000.metrics", paths)
            self.assertTrue(paths["run_0000.metrics"].exists())
            self.assertEqual(len(pd.read_csv(paths["summary"])), 2)

    def test_signal_delay_defers_fill_by_configured_sessions(self) -> None:
        engine = BacktestEngine(
            {"OGDC": bars("OGDC", [100, 110, 120, 130])},
            config=zero_cost_config(signal_delay_sessions=2),
        )

        result = engine.run(OneShotOrders([Order("OGDC", "buy", shares=1)]))

        self.assertEqual(result.trades.iloc[0]["fill_date"], pd.Timestamp("2026-01-05"))
        self.assertEqual(result.trades.iloc[0]["price"], 120)

    def test_prior_volume_participation_caps_fill(self) -> None:
        engine = BacktestEngine(
            {"OGDC": bars("OGDC", [10, 10, 10])},
            config=zero_cost_config(initial_capital=1_000_000, max_volume_participation=0.001),
        )

        result = engine.run(OneShotOrders([Order("OGDC", "buy", shares=500)]))

        self.assertEqual(result.orders.iloc[0]["status"], "partially_filled")
        self.assertEqual(result.trades.iloc[0]["shares"], 100)

    def test_board_lot_schedule_rounds_orders_down(self) -> None:
        engine = BacktestEngine(
            {"OGDC": bars("OGDC", [10, 10, 10])},
            config=zero_cost_config(board_lot_size=10),
        )

        result = engine.run(OneShotOrders([Order("OGDC", "buy", shares=15)]))

        self.assertEqual(result.orders.iloc[0]["requested_shares"], 10)
        self.assertEqual(result.trades.iloc[0]["shares"], 10)

    def test_missed_fill_stress_is_deterministic(self) -> None:
        engine = BacktestEngine(
            {"OGDC": bars("OGDC", [100, 100, 100])},
            config=zero_cost_config(missed_fill_probability=1.0, random_seed=7),
        )

        result = engine.run(OneShotOrders([Order("OGDC", "buy", shares=1)]))

        self.assertTrue(result.trades.empty)
        self.assertEqual(result.orders.iloc[0]["reason"], "simulated_missed_fill")

    def test_capital_gains_tax_is_charged_on_profitable_sell(self) -> None:
        strategy = DateOrders(
            {
                "2026-01-01": [Order("A", "buy", shares=5)],
                "2026-01-02": [Order("A", "sell", shares=5)],
            }
        )
        engine = BacktestEngine(
            {"A": bars("A", [100, 100, 130, 130])},
            config=zero_cost_config(capital_gains_tax_rate=0.15),
        )

        result = engine.run(strategy)
        sell = result.trades[result.trades["side"] == "sell"].iloc[0]

        self.assertAlmostEqual(sell["capital_gains_tax"], 22.5)
        self.assertAlmostEqual(sell["realized_pnl"], 127.5)
        self.assertAlmostEqual(result.metrics["total_capital_gains_tax"], 22.5)

    def test_effective_dated_settlement_schedule_supports_historical_t2(self) -> None:
        strategy = DateOrders(
            {
                "2026-01-01": [Order("A", "buy", shares=10)],
                "2026-01-02": [Order("A", "sell", shares=10)],
            }
        )
        engine = BacktestEngine(
            {"A": bars("A", [100, 100, 100, 100, 100])},
            config=zero_cost_config(
                settlement_lag_days=1,
                settlement_lag_schedule=(("2026-01-05", 2),),
            ),
        )

        curve = engine.run(strategy).equity_curve.set_index("date")

        self.assertEqual(curve.loc[pd.Timestamp("2026-01-06"), "unsettled_cash"], 1000)
        self.assertEqual(curve.loc[pd.Timestamp("2026-01-07"), "available_cash"], 1000)

    def test_suspended_holding_does_not_freeze_other_target_orders(self) -> None:
        a = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-06"]),
                "open": [100, 100, 100],
                "close": [100, 100, 100],
                "volume": [100_000, 100_000, 100_000],
            }
        )
        engine = BacktestEngine(
            {"A": a, "B": bars("B", [100, 100, 100, 100])},
            config=zero_cost_config(),
        )

        result = engine.run(SuspensionRebalance())

        b_trades = result.trades[result.trades["symbol"] == "B"]
        self.assertEqual(len(b_trades), 2)
        self.assertEqual(int(b_trades["shares"].sum()), 5)
        self.assertEqual(b_trades.iloc[0]["fill_date"], pd.Timestamp("2026-01-05"))
        final_positions = result.positions[result.positions["date"] == result.positions["date"].max()]
        self.assertEqual(set(final_positions["symbol"]), {"B"})

    def test_target_rotation_retries_buy_after_sale_settles(self) -> None:
        data = {
            "A": bars("A", [100, 100, 100, 100, 100]),
            "B": bars("B", [100, 100, 100, 100, 100]),
        }
        engine = BacktestEngine(data, config=zero_cost_config())

        result = engine.run(TargetRotationAfterBuy())
        b_orders = result.orders[result.orders["symbol"] == "B"]

        self.assertEqual(list(b_orders["status"]), ["rejected", "filled"])
        self.assertEqual(list(b_orders["reason"]), ["insufficient_cash", ""])
        self.assertEqual(b_orders.iloc[-1]["fill_date"], pd.Timestamp("2026-01-06"))
        final_positions = result.positions[result.positions["date"] == result.positions["date"].max()]
        self.assertEqual(set(final_positions["symbol"]), {"B"})


if __name__ == "__main__":
    unittest.main()
