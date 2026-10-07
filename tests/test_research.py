from __future__ import annotations

import tempfile
import unittest

import pandas as pd

from ahl_api.backtest import BacktestConfig, StrategyContext
from ahl_api.research import (
    DataProvenance,
    MembershipSchedule,
    StrategySpec,
    StressScenario,
    WalkForwardConfig,
    audit_market_data,
    default_stress_scenarios,
    generate_walk_forward_folds,
    price_index_benchmark,
    run_walk_forward_validation,
)


def daily_frame(periods: int = 900, *, growth: float = 0.001) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=periods)
    closes = pd.Series([(1.0 + growth) ** index * 100.0 for index in range(periods)])
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes.values,
            "close": closes.values,
            "volume": [100_000] * periods,
        }
    )


class WeightedStrategy:
    def __init__(self, weight: float) -> None:
        self.weight = weight
        self.sent = False

    def target_weights(self, context: StrategyContext) -> dict[str, float]:
        if self.sent:
            return {}
        self.sent = True
        return {"AAA": self.weight}


class ResearchTests(unittest.TestCase):
    def test_data_audit_blocks_unadjusted_survivorship_biased_data(self) -> None:
        result = audit_market_data({"AAA": daily_frame(300)})

        self.assertTrue(result.has_blockers)
        self.assertIn("corporate_actions_unverified", set(result.issues["code"]))
        self.assertIn("dividends_missing", set(result.issues["code"]))
        self.assertIn("survivorship_bias", set(result.issues["code"]))

    def test_data_audit_detects_duplicate_dates(self) -> None:
        frame = daily_frame(300)
        frame = pd.concat([frame, frame.iloc[[-1]]], ignore_index=True)
        provenance = DataProvenance(
            source="synthetic",
            adjusted_for_corporate_actions=True,
            includes_cash_dividends=True,
            point_in_time_universe=True,
            historical_membership_complete=True,
            historical_board_lots_complete=True,
            historical_sessions_complete=True,
            circuit_and_halt_data_complete=True,
            systematic_use_authorized=True,
        )

        result = audit_market_data({"AAA": frame}, provenance=provenance)

        self.assertTrue(result.has_blockers)
        self.assertIn("duplicate_dates", set(result.issues["code"]))

    def test_data_audit_blocks_mixed_open_and_adjusted_close_scales(self) -> None:
        frame = daily_frame(300)
        frame.loc[:250, "open"] = frame.loc[:250, "open"] * 5.0

        result = audit_market_data({"AAA": frame})

        issue = result.issues[result.issues["code"] == "open_close_scale_mismatch"]
        self.assertEqual(len(issue), 1)
        self.assertEqual(issue.iloc[0]["severity"], "blocker")

    def test_membership_schedule_is_effective_dated_and_inclusive(self) -> None:
        schedule = MembershipSchedule.from_frame(
            pd.DataFrame(
                {
                    "symbol": ["aaa", "BBB"],
                    "effective_from": ["2020-01-01", "2020-02-01"],
                    "effective_until": ["2020-01-31", None],
                }
            )
        )

        self.assertEqual(schedule.symbols_on("2020-01-31"), {"AAA"})
        self.assertEqual(schedule.symbols_on("2020-02-01"), {"BBB"})

    def test_walk_forward_folds_have_embargo_and_no_overlap(self) -> None:
        calendar = pd.bdate_range("2020-01-01", "2023-12-31")
        config = WalkForwardConfig(train_months=12, test_months=3, step_months=3, embargo_sessions=2)

        folds = generate_walk_forward_folds(calendar, config)

        self.assertGreater(len(folds), 5)
        for fold in folds:
            self.assertLess(fold["train_end"], fold["test_start"])
            later = calendar[calendar > fold["train_end"]]
            self.assertEqual(fold["test_start"], later[2])

    def test_nested_walk_forward_selects_only_on_train_windows(self) -> None:
        config = BacktestConfig(
            initial_capital=50_000,
            brokerage_bps=0,
            brokerage_per_share=0,
            sales_tax_rate=0,
            settlement_lag_days=1,
        )
        result = run_walk_forward_validation(
            {"AAA": daily_frame()},
            [StrategySpec("weighted", WeightedStrategy, {"weight": [0.5, 1.0]})],
            backtest_config=config,
            validation_config=WalkForwardConfig(
                train_months=12,
                test_months=3,
                step_months=6,
                inner_train_months=6,
                inner_test_months=2,
                inner_step_months=2,
                minimum_inner_folds=1,
                minimum_round_trips=0,
            ),
            stress_scenarios=[StressScenario("base")],
        )

        self.assertFalse(result.selection_runs.empty)
        self.assertFalse(result.folds.empty)
        self.assertTrue((result.folds["scenario"] == "base").all())
        self.assertTrue((result.folds["test_start"] > result.folds["train_end"]).all())
        self.assertIn("inner_fold", result.selection_runs["selection_level"].unique())
        self.assertFalse(result.stitched_leaderboard.empty)
        self.assertFalse(result.leaderboard.empty)
        with tempfile.TemporaryDirectory() as directory:
            paths = result.write_report(directory)
            self.assertTrue(paths["leaderboard"].exists())

    def test_price_index_benchmark_reports_drawdown_and_equity(self) -> None:
        curve, metrics = price_index_benchmark(daily_frame(300), name="INDEX", initial_capital=50_000)

        self.assertEqual(curve.iloc[0]["equity"], 50_000)
        self.assertGreater(metrics["final_equity"], 50_000)
        self.assertIn("max_drawdown", metrics)

    def test_default_stresses_include_cost_delay_and_fill_cases(self) -> None:
        config = BacktestConfig(slippage_bps=25)
        scenarios = default_stress_scenarios(config)

        self.assertEqual(scenarios[0].name, "base")
        self.assertEqual({scenario.name for scenario in scenarios}, {
            "base",
            "double_costs",
            "delayed_entry",
            "missed_fills",
            "adverse_slippage",
        })


if __name__ == "__main__":
    unittest.main()
