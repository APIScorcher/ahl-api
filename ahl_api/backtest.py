"""Pandas-first long-only backtesting for PSX cash equity strategies."""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """Execution and cost assumptions for a daily cash-equity backtest."""

    initial_capital: float = 1_000_000.0
    recurring_contribution: float = 0.0
    contribution_frequency: str = "none"
    start_date: date | datetime | str | None = None
    end_date: date | datetime | str | None = None
    brokerage_bps: float = 15.0
    brokerage_per_share: float = 0.03
    sales_tax_rate: float = 0.15
    slippage_bps: float = 0.0
    other_fees_bps: float = 0.0
    capital_gains_tax_rate: float = 0.0
    annual_risk_free_rate: float = 0.0
    settlement_lag_days: int = 1
    settlement_lag_schedule: tuple[tuple[str, int], ...] = ()
    signal_delay_sessions: int = 1
    target_intent_expiry_sessions: int = 3
    board_lot_size: int = 1
    board_lot_schedule: tuple[tuple[str, int], ...] = ()
    max_volume_participation: float | None = None
    minimum_trade_value: float = 0.0
    missed_fill_probability: float = 0.0
    random_seed: int = 0


@dataclass(frozen=True)
class Order:
    """A next-open buy/sell intent emitted by an order-based strategy."""

    symbol: str
    side: str
    shares: int | None = None
    value: float | None = None
    tag: str = ""


@dataclass(frozen=True)
class Fill:
    order_id: int
    signal_date: pd.Timestamp
    fill_date: pd.Timestamp
    symbol: str
    side: str
    shares: int
    reference_price: float
    price: float
    trade_value: float
    brokerage: float
    sales_tax: float
    other_fees: float
    capital_gains_tax: float
    total_fees: float
    net_cash: float
    realized_pnl: float = 0.0
    slippage_cost: float = 0.0
    holding_days: int | None = None


@dataclass(frozen=True)
class Position:
    symbol: str
    shares: int
    avg_price: float
    close: float | None = None
    market_value: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass(frozen=True)
class StrategyContext:
    date: pd.Timestamp
    history: dict[str, pd.DataFrame]
    prices: dict[str, float]
    positions: dict[str, Position]
    available_cash: float
    unsettled_cash: float
    equity: float


class TargetWeightStrategy(Protocol):
    def target_weights(self, context: StrategyContext) -> Mapping[str, float]:
        """Return desired long-only portfolio weights by symbol."""


class OrderStrategy(Protocol):
    def orders(self, context: StrategyContext) -> Sequence[Order]:
        """Return explicit buy/sell orders to execute at the next open."""


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    orders: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
    metrics: dict[str, Any]

    def write_report(self, output_dir: str | Path, *, prefix: str = "backtest") -> dict[str, Path]:
        """Write result tables and metrics to disk."""

        return write_backtest_report(self, output_dir, prefix=prefix)


@dataclass(frozen=True)
class SweepRun:
    run_id: str
    parameters: dict[str, Any]
    result: BacktestResult


@dataclass(frozen=True)
class SweepResult:
    runs: list[SweepRun]
    summary: pd.DataFrame

    def best(self, metric: str = "total_return", *, ascending: bool = False) -> SweepRun:
        if self.summary.empty:
            raise ValueError("sweep result has no runs")
        if metric not in self.summary.columns:
            raise ValueError(f"unknown metric: {metric}")
        ordered = self.summary.sort_values(metric, ascending=ascending, na_position="last")
        return self.runs[int(ordered.iloc[0]["run_index"])]

    def write_report(self, output_dir: str | Path, *, prefix: str = "sweep", include_run_details: bool = False) -> dict[str, Path]:
        """Write sweep summary and optional per-run reports to disk."""

        return write_sweep_report(self, output_dir, prefix=prefix, include_run_details=include_run_details)


@dataclass
class _Holding:
    shares: int = 0
    avg_price: float = 0.0
    entry_date: pd.Timestamp | None = None


@dataclass
class _Pending:
    order_id: int
    signal_date: pd.Timestamp
    source: str
    orders: list[Order] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)


def normalize_bars(symbol: str, rows: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Normalize PSX daily bars into the engine's pandas shape."""

    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    required = {"date", "open", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{symbol.upper()} bars missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["symbol"] = symbol.upper()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    for column in ("open", "high", "low", "close", "volume"):
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["date", "open", "close"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    return frame[["date", "symbol", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def load_historical_data(
    client: Any,
    symbols: Sequence[str],
    *,
    since: str | date | datetime | int | float | None = None,
    until: str | date | datetime | int | float | None = None,
    years: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Load and normalize daily bars through ``AHL.fetch_historical_daily``."""

    return {
        symbol.upper(): normalize_bars(
            symbol,
            client.fetch_historical_daily(symbol, since=since, until=until, years=years),
        )
        for symbol in symbols
    }


def calculate_trade_cost(shares: int, price: float, config: BacktestConfig | None = None) -> dict[str, float]:
    """Calculate default AHL/PSX-style per-trade costs."""

    cfg = config or BacktestConfig()
    if shares <= 0 or price <= 0:
        return {
            "trade_value": 0.0,
            "brokerage": 0.0,
            "sales_tax": 0.0,
            "other_fees": 0.0,
            "total_fees": 0.0,
        }

    trade_value = float(shares) * float(price)
    brokerage = max(cfg.brokerage_per_share * shares, trade_value * cfg.brokerage_bps / 10_000)
    sales_tax = brokerage * cfg.sales_tax_rate
    other_fees = trade_value * cfg.other_fees_bps / 10_000
    total_fees = brokerage + sales_tax + other_fees
    return {
        "trade_value": trade_value,
        "brokerage": brokerage,
        "sales_tax": sales_tax,
        "other_fees": other_fees,
        "total_fees": total_fees,
    }


def run_parameter_sweep(
    data: Mapping[str, pd.DataFrame | Sequence[Mapping[str, Any]]],
    strategy_factory: Any,
    parameter_grid: Mapping[str, Sequence[Any]],
    *,
    config: BacktestConfig | None = None,
    config_grid: Mapping[str, Sequence[Any]] | None = None,
) -> SweepResult:
    """Run a Cartesian parameter sweep over strategy and optional config params.

    ``strategy_factory`` is called once per run with the selected strategy
    parameters as keyword arguments. ``config_grid`` keys must match
    ``BacktestConfig`` fields and override ``config`` for that run.
    """

    strategy_params = _parameter_combinations(parameter_grid)
    config_params = _parameter_combinations(config_grid or {})
    base_config = config or BacktestConfig()
    runs: list[SweepRun] = []
    summary_rows: list[dict[str, Any]] = []

    for run_index, (strategy_kwargs, config_kwargs) in enumerate(product(strategy_params, config_params)):
        run_config = _replace_config(base_config, config_kwargs)
        engine = BacktestEngine(data, config=run_config)
        strategy = strategy_factory(**strategy_kwargs)
        result = engine.run(strategy)
        run_id = f"run_{run_index:04d}"
        parameters = {**strategy_kwargs, **{f"config.{key}": value for key, value in config_kwargs.items()}}
        runs.append(SweepRun(run_id=run_id, parameters=parameters, result=result))
        summary_rows.append(
            {
                "run_index": run_index,
                "run_id": run_id,
                **parameters,
                **result.metrics,
            }
        )

    return SweepResult(runs=runs, summary=pd.DataFrame(summary_rows))


def write_backtest_report(result: BacktestResult, output_dir: str | Path, *, prefix: str = "backtest") -> dict[str, Path]:
    """Write one backtest result to CSV/JSON report files."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": directory / f"{prefix}_metrics.json",
        "equity_curve": directory / f"{prefix}_equity_curve.csv",
        "orders": directory / f"{prefix}_orders.csv",
        "trades": directory / f"{prefix}_trades.csv",
        "positions": directory / f"{prefix}_positions.csv",
    }
    paths["metrics"].write_text(json.dumps(_json_ready(result.metrics), indent=2, sort_keys=True), encoding="utf-8")
    result.equity_curve.to_csv(paths["equity_curve"], index=False)
    result.orders.to_csv(paths["orders"], index=False)
    result.trades.to_csv(paths["trades"], index=False)
    result.positions.to_csv(paths["positions"], index=False)
    return paths


def write_sweep_report(
    result: SweepResult,
    output_dir: str | Path,
    *,
    prefix: str = "sweep",
    include_run_details: bool = False,
) -> dict[str, Path]:
    """Write sweep summary and optional per-run backtest files."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {"summary": directory / f"{prefix}_summary.csv"}
    result.summary.to_csv(paths["summary"], index=False)

    if include_run_details:
        for run in result.runs:
            run_paths = run.result.write_report(directory / run.run_id, prefix=run.run_id)
            for key, path in run_paths.items():
                paths[f"{run.run_id}.{key}"] = path
    return paths


def calculate_xirr(cashflows: Sequence[tuple[date | datetime | pd.Timestamp, float]]) -> float | None:
    """Calculate annualized money-weighted return for irregular cash flows."""

    flows = [(pd.Timestamp(flow_date).normalize(), float(amount)) for flow_date, amount in cashflows if float(amount) != 0.0]
    if not flows:
        return None
    if not any(amount > 0 for _, amount in flows) or not any(amount < 0 for _, amount in flows):
        return None

    start = flows[0][0]

    def npv(rate: float) -> float:
        total = 0.0
        base = 1.0 + rate
        for flow_date, amount in flows:
            years = (flow_date - start).days / 365.25
            total += amount / (base**years)
        return total

    if abs(npv(0.0)) < 1e-7:
        return 0.0

    low = -0.999999
    high = 10.0
    low_value = npv(low)
    high_value = npv(high)
    while low_value * high_value > 0 and high < 1_000_000:
        high *= 10.0
        high_value = npv(high)
    if low_value * high_value > 0:
        return None

    for _ in range(200):
        mid = (low + high) / 2.0
        mid_value = npv(mid)
        if abs(mid_value) < 1e-7:
            return float(mid)
        if low_value * mid_value <= 0:
            high = mid
            high_value = mid_value
        else:
            low = mid
            low_value = mid_value
    return float((low + high) / 2.0)


def _parameter_combinations(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid)
    values = [list(grid[key]) for key in keys]
    if any(not choices for choices in values):
        raise ValueError("parameter grid values cannot be empty")
    return [dict(zip(keys, combination)) for combination in product(*values)]


def _replace_config(config: BacktestConfig, overrides: Mapping[str, Any]) -> BacktestConfig:
    values = config.__dict__.copy()
    for key, value in overrides.items():
        if key not in values:
            raise ValueError(f"unknown BacktestConfig field: {key}")
        values[key] = value
    return BacktestConfig(**values)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _maximum_underwater_days(equity_curve: pd.DataFrame) -> int:
    if equity_curve.empty:
        return 0
    values = equity_curve["strategy_index"] if "strategy_index" in equity_curve else equity_curve["equity"]
    dates = pd.to_datetime(equity_curve["date"])
    peak_value = -math.inf
    peak_date = pd.Timestamp(dates.iloc[0])
    longest = 0
    for current_date, value in zip(dates, pd.to_numeric(values, errors="coerce")):
        if pd.isna(value):
            continue
        if float(value) >= peak_value:
            peak_value = float(value)
            peak_date = pd.Timestamp(current_date)
            continue
        longest = max(longest, (pd.Timestamp(current_date) - peak_date).days)
    return int(longest)


class BacktestEngine:
    """Daily long-only backtest engine with next-open PSX cash execution."""

    def __init__(self, data: Mapping[str, pd.DataFrame | Sequence[Mapping[str, Any]]], config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        for name in ("initial_capital", "recurring_contribution", "brokerage_bps", "brokerage_per_share",
                     "sales_tax_rate", "slippage_bps", "other_fees_bps", "capital_gains_tax_rate",
                     "minimum_trade_value"):
            value = getattr(self.config, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.config.initial_capital < 0:
            raise ValueError("initial_capital cannot be negative")
        if self.config.recurring_contribution < 0:
            raise ValueError("recurring_contribution cannot be negative")
        if self.config.initial_capital == 0 and self.config.recurring_contribution == 0:
            raise ValueError("initial_capital or recurring_contribution must be positive")
        if self.config.contribution_frequency not in {"none", "monthly"}:
            raise ValueError("contribution_frequency must be one of: none, monthly")
        if self.config.settlement_lag_days < 0:
            raise ValueError("settlement_lag_days cannot be negative")
        if self.config.signal_delay_sessions < 1:
            raise ValueError("signal_delay_sessions must be at least 1")
        if self.config.target_intent_expiry_sessions < 1:
            raise ValueError("target_intent_expiry_sessions must be at least 1")
        if int(self.config.board_lot_size) != self.config.board_lot_size or self.config.board_lot_size < 1:
            raise ValueError("board_lot_size must be a positive integer")
        if self.config.capital_gains_tax_rate < 0 or self.config.capital_gains_tax_rate > 1:
            raise ValueError("capital_gains_tax_rate must be in the range [0, 1]")
        if self.config.max_volume_participation is not None and not 0 < self.config.max_volume_participation <= 1:
            raise ValueError("max_volume_participation must be in the range (0, 1]")
        if self.config.minimum_trade_value < 0:
            raise ValueError("minimum_trade_value cannot be negative")
        if not 0 <= self.config.missed_fill_probability <= 1:
            raise ValueError("missed_fill_probability must be in the range [0, 1]")
        for effective_date, lag in self.config.settlement_lag_schedule:
            pd.Timestamp(effective_date)
            if int(lag) != lag or lag < 0:
                raise ValueError("settlement_lag_schedule lags must be non-negative integers")
        for effective_date, lot_size in self.config.board_lot_schedule:
            pd.Timestamp(effective_date)
            if int(lot_size) != lot_size or lot_size < 1:
                raise ValueError("board_lot_schedule sizes must be positive integers")

        self.data = {symbol.upper(): normalize_bars(symbol, rows) for symbol, rows in data.items()}
        if not self.data:
            raise ValueError("data must include at least one symbol")

        self._by_date = {
            symbol: frame.set_index("date", drop=False)
            for symbol, frame in self.data.items()
        }
        self.calendar = pd.Index(
            sorted({bar_date for frame in self.data.values() for bar_date in frame["date"]}),
            name="date",
        )
        if self.config.start_date is not None:
            start_date = pd.Timestamp(self.config.start_date).normalize()
            self.calendar = self.calendar[self.calendar >= start_date]
        if self.config.end_date is not None:
            end_date = pd.Timestamp(self.config.end_date).normalize()
            self.calendar = self.calendar[self.calendar <= end_date]
        if self.calendar.empty:
            raise ValueError("data does not contain any tradable bars")
        self._calendar_positions = {pd.Timestamp(value): index for index, value in enumerate(self.calendar)}

    @classmethod
    def from_client(
        cls,
        client: Any,
        symbols: Sequence[str],
        *,
        config: BacktestConfig | None = None,
        since: str | date | datetime | int | float | None = None,
        until: str | date | datetime | int | float | None = None,
        years: int | None = None,
    ) -> "BacktestEngine":
        return cls(load_historical_data(client, symbols, since=since, until=until, years=years), config=config)

    def run(self, strategy: TargetWeightStrategy | OrderStrategy) -> BacktestResult:
        prepare = getattr(strategy, "prepare", None)
        if callable(prepare):
            prepare(self.data)
        available_cash = float(self.config.initial_capital)
        cumulative_contributions = float(self.config.initial_capital)
        last_contribution_period: Any = None
        settlements: dict[pd.Timestamp, float] = {}
        holdings: dict[str, _Holding] = {}
        last_close: dict[str, float] = {}
        pending: list[_Pending] = []
        next_order_id = 1

        order_rows: list[dict[str, Any]] = []
        trade_rows: list[dict[str, Any]] = []
        equity_rows: list[dict[str, Any]] = []
        position_rows: list[dict[str, Any]] = []

        for index, current_date in enumerate(self.calendar):
            current_date = pd.Timestamp(current_date)
            contribution, last_contribution_period = self._contribution_for_date(current_date, last_contribution_period)
            if contribution:
                available_cash += contribution
                cumulative_contributions += contribution
            available_cash += settlements.pop(current_date, 0.0)

            available_cash, pending = self._execute_pending(
                current_date,
                pending,
                available_cash,
                settlements,
                holdings,
                order_rows,
                trade_rows,
            )

            for symbol, frame in self._by_date.items():
                if current_date in frame.index:
                    last_close[symbol] = float(frame.loc[current_date, "close"])

            equity, invested_value, unsettled_cash = self._mark_equity(available_cash, settlements, holdings, last_close)
            equity_rows.append(
                {
                    "date": current_date,
                    "equity": equity,
                    "available_cash": available_cash,
                    "unsettled_cash": unsettled_cash,
                    "invested_value": invested_value,
                    "contribution": contribution,
                    "cumulative_contributions": cumulative_contributions,
                }
            )
            self._record_positions(current_date, holdings, last_close, position_rows)

            if index == len(self.calendar) - 1:
                continue

            needs_history = getattr(strategy, "needs_history", None)
            include_history = bool(needs_history(current_date)) if callable(needs_history) else True
            context = StrategyContext(
                date=current_date,
                history=self._history_until(current_date) if include_history else {},
                prices=dict(last_close),
                positions=self._context_positions(holdings, last_close),
                available_cash=available_cash,
                unsettled_cash=unsettled_cash,
                equity=equity,
            )
            target_weights = self._strategy_target_weights(strategy, context)
            if target_weights:
                self._validate_weights(target_weights)
                pending = [item for item in pending if item.source != "target_weights"]
                pending.append(
                    _Pending(
                        order_id=next_order_id,
                        signal_date=current_date,
                        source="target_weights",
                        weights={symbol.upper(): float(weight) for symbol, weight in target_weights.items()},
                    )
                )
                next_order_id += 1

            strategy_orders = self._strategy_orders(strategy, context)
            if strategy_orders:
                pending.append(
                    _Pending(
                        order_id=next_order_id,
                        signal_date=current_date,
                        source="orders",
                        orders=[self._normalize_order(order) for order in strategy_orders],
                    )
                )
                next_order_id += 1

        equity_curve = pd.DataFrame(equity_rows)
        if not equity_curve.empty:
            prior_equity = equity_curve["equity"].shift(1)
            external_flow = equity_curve.get("contribution", pd.Series(0.0, index=equity_curve.index))
            equity_curve["daily_return"] = ((equity_curve["equity"] - external_flow) / prior_equity - 1.0).fillna(0.0)
            equity_curve["strategy_index"] = (1.0 + equity_curve["daily_return"]).cumprod()
            running_peak = equity_curve["strategy_index"].cummax()
            equity_curve["drawdown"] = (equity_curve["strategy_index"] / running_peak) - 1.0

        orders_frame = pd.DataFrame(order_rows)
        trades_frame = pd.DataFrame(trade_rows)
        positions_frame = pd.DataFrame(position_rows)

        return BacktestResult(
            equity_curve=equity_curve,
            orders=orders_frame,
            trades=trades_frame,
            positions=positions_frame,
            metrics=self._metrics(equity_curve, trades_frame, positions_frame),
        )

    def _contribution_for_date(self, current_date: pd.Timestamp, last_period: Any) -> tuple[float, Any]:
        if self.config.recurring_contribution <= 0 or self.config.contribution_frequency == "none":
            return 0.0, last_period
        if self.config.contribution_frequency == "monthly":
            period = (current_date.year, current_date.month)
            if period != last_period:
                return float(self.config.recurring_contribution), period
            return 0.0, last_period
        return 0.0, last_period

    def _execute_pending(
        self,
        current_date: pd.Timestamp,
        pending: list[_Pending],
        available_cash: float,
        settlements: dict[pd.Timestamp, float],
        holdings: dict[str, _Holding],
        order_rows: list[dict[str, Any]],
        trade_rows: list[dict[str, Any]],
    ) -> tuple[float, list[_Pending]]:
        remaining: list[_Pending] = []
        for item in pending:
            if self._sessions_since_signal(item.signal_date, current_date) < self.config.signal_delay_sessions:
                remaining.append(item)
                continue
            if item.source == "target_weights":
                required_symbols = set(item.weights) | {symbol for symbol, holding in holdings.items() if holding.shares > 0}
                tradable_symbols = {
                    symbol
                    for symbol in required_symbols
                    if symbol in self._by_date and current_date in self._by_date[symbol].index
                }
                if not tradable_symbols:
                    age = self._sessions_since_signal(item.signal_date, current_date)
                    has_exit_intent = any(
                        holdings.get(symbol, _Holding()).shares > 0 and item.weights.get(symbol, 0.0) == 0.0
                        for symbol in required_symbols
                    )
                    has_buy_intent = any(item.weights.get(symbol, 0.0) > 0.0 for symbol in required_symbols)
                    if has_exit_intent or (has_buy_intent and age < self.config.target_intent_expiry_sessions):
                        remaining.append(item)
                    continue
                orders = self._orders_from_target_weights(item, current_date, available_cash, settlements, holdings)
                first_order_row = len(order_rows)
                for order in orders:
                    available_cash = self._execute_order(
                        item.order_id,
                        item.signal_date,
                        current_date,
                        item.source,
                        order,
                        available_cash,
                        settlements,
                        holdings,
                        order_rows,
                        trade_rows,
                    )
                attempts = order_rows[first_order_row:]
                needs_sell_retry = any(
                    row["side"] == "sell" and row["status"] != "filled"
                    for row in attempts
                )
                needs_buy_retry = any(
                    row["side"] == "buy" and row["status"] != "filled"
                    for row in attempts
                )
                missing_symbols = required_symbols.difference(tradable_symbols)
                missing_sell = any(
                    holdings.get(symbol, _Holding()).shares > 0 and item.weights.get(symbol, 0.0) == 0.0
                    for symbol in missing_symbols
                )
                missing_buy = any(item.weights.get(symbol, 0.0) > 0.0 for symbol in missing_symbols)
                age = self._sessions_since_signal(item.signal_date, current_date)
                if needs_sell_retry or missing_sell or (
                    (needs_buy_retry or missing_buy) and age < self.config.target_intent_expiry_sessions
                ):
                    remaining.append(item)
            else:
                not_ready: list[Order] = []
                for order in item.orders:
                    symbol = order.symbol.upper()
                    if symbol in self._by_date and current_date not in self._by_date[symbol].index:
                        not_ready.append(order)
                        continue
                    available_cash = self._execute_order(
                        item.order_id,
                        item.signal_date,
                        current_date,
                        item.source,
                        order,
                        available_cash,
                        settlements,
                        holdings,
                        order_rows,
                        trade_rows,
                    )
                if not_ready:
                    remaining.append(_Pending(order_id=item.order_id, signal_date=item.signal_date, source=item.source, orders=not_ready))
        return available_cash, remaining

    def _orders_from_target_weights(
        self,
        item: _Pending,
        current_date: pd.Timestamp,
        available_cash: float,
        settlements: Mapping[pd.Timestamp, float],
        holdings: Mapping[str, _Holding],
    ) -> list[Order]:
        open_prices = self._open_prices(current_date)
        if not open_prices:
            return []

        equity_at_open = available_cash + sum(settlements.values())
        for symbol, holding in holdings.items():
            price = open_prices.get(symbol)
            if price is not None:
                equity_at_open += holding.shares * price

        target_symbols = set(item.weights) | {symbol for symbol, holding in holdings.items() if holding.shares > 0}
        sells: list[Order] = []
        buys: list[Order] = []
        for symbol in sorted(target_symbols):
            price = open_prices.get(symbol)
            if price is None or price <= 0:
                continue
            target_weight = item.weights.get(symbol, 0.0)
            current_shares = holdings.get(symbol, _Holding()).shares
            lot_size = self._board_lot_size(current_date)
            target_shares = int((equity_at_open * target_weight) // price)
            target_shares = target_shares // lot_size * lot_size
            delta = target_shares - current_shares
            if delta < 0:
                sells.append(Order(symbol=symbol, side="sell", shares=abs(delta), tag="rebalance"))
            elif delta > 0:
                buys.append(Order(symbol=symbol, side="buy", shares=delta, tag="rebalance"))
        return sells + buys

    def _execute_order(
        self,
        order_id: int,
        signal_date: pd.Timestamp,
        fill_date: pd.Timestamp,
        source: str,
        order: Order,
        available_cash: float,
        settlements: dict[pd.Timestamp, float],
        holdings: dict[str, _Holding],
        order_rows: list[dict[str, Any]],
        trade_rows: list[dict[str, Any]],
    ) -> float:
        symbol = order.symbol.upper()
        side = order.side.lower()
        lot_size = self._board_lot_size(fill_date)
        requested_shares = self._requested_shares(order, fill_date, available_cash, lot_size)
        status = "rejected"
        reason = ""
        filled_shares = 0

        if requested_shares <= 0:
            reason = "invalid_size"
        elif symbol not in self._by_date or fill_date not in self._by_date[symbol].index:
            reason = "no_market_data"
        elif side not in {"buy", "sell"}:
            reason = "invalid_side"
        elif self._is_simulated_missed_fill(order_id, symbol, side, fill_date):
            reason = "simulated_missed_fill"
        elif side == "buy":
            open_price = float(self._by_date[symbol].loc[fill_date, "open"])
            price = self._slipped_price(open_price, side)
            liquidity_cap = self._liquidity_share_cap(symbol, fill_date)
            filled_shares = min(requested_shares, self._affordable_shares(available_cash, price, lot_size), liquidity_cap)
            filled_shares = filled_shares // lot_size * lot_size
            if filled_shares <= 0:
                reason = "liquidity_limit" if liquidity_cap <= 0 else "insufficient_cash"
            elif filled_shares * price < self.config.minimum_trade_value:
                reason = "below_minimum_trade_value"
                filled_shares = 0
            else:
                fill = self._fill(order_id, signal_date, fill_date, symbol, side, filled_shares, open_price, price)
                available_cash -= fill.net_cash
                holding = holdings.setdefault(symbol, _Holding())
                if holding.shares == 0:
                    holding.entry_date = fill_date
                total_cost = holding.avg_price * holding.shares + fill.trade_value + fill.total_fees
                holding.shares += filled_shares
                holding.avg_price = total_cost / holding.shares if holding.shares else 0.0
                self._record_fill(fill, trade_rows)
                status = "filled" if filled_shares == requested_shares else "partially_filled"
        else:
            holding = holdings.get(symbol, _Holding())
            liquidity_cap = self._liquidity_share_cap(symbol, fill_date)
            filled_shares = min(requested_shares, holding.shares, liquidity_cap)
            filled_shares = filled_shares // lot_size * lot_size
            if filled_shares <= 0:
                reason = "no_position" if holding.shares <= 0 else "liquidity_limit"
            else:
                open_price = float(self._by_date[symbol].loc[fill_date, "open"])
                price = self._slipped_price(open_price, side)
                if filled_shares * price < self.config.minimum_trade_value:
                    reason = "below_minimum_trade_value"
                    filled_shares = 0
                else:
                    holding_days = (fill_date - holding.entry_date).days if holding.entry_date is not None else None
                    fill = self._fill(
                        order_id,
                        signal_date,
                        fill_date,
                        symbol,
                        side,
                        filled_shares,
                        open_price,
                        price,
                        holding.avg_price,
                        holding_days,
                    )
                    holding.shares -= filled_shares
                    if holding.shares == 0:
                        holding.avg_price = 0.0
                        holding.entry_date = None
                    settlement_date = self._settlement_date(fill_date)
                    if settlement_date == fill_date:
                        available_cash += fill.net_cash
                    elif settlement_date is not None:
                        settlements[settlement_date] = settlements.get(settlement_date, 0.0) + fill.net_cash
                    self._record_fill(fill, trade_rows)
                    status = "filled" if filled_shares == requested_shares else "partially_filled"

        order_rows.append(
            {
                "order_id": order_id,
                "signal_date": signal_date,
                "fill_date": fill_date,
                "source": source,
                "symbol": symbol,
                "side": side,
                "requested_shares": requested_shares,
                "filled_shares": filled_shares,
                "status": status,
                "reason": reason,
                "tag": order.tag,
            }
        )
        return available_cash

    def _requested_shares(self, order: Order, fill_date: pd.Timestamp, available_cash: float, lot_size: int) -> int:
        if order.shares is not None:
            shares = max(0, int(order.shares))
            return shares // lot_size * lot_size
        if order.value is None or order.value <= 0:
            return 0
        symbol = order.symbol.upper()
        if symbol not in self._by_date or fill_date not in self._by_date[symbol].index:
            return 0
        price = self._slipped_price(float(self._by_date[symbol].loc[fill_date, "open"]), order.side.lower())
        value = min(float(order.value), available_cash) if order.side.lower() == "buy" else float(order.value)
        shares = max(0, int(value // price))
        return shares // lot_size * lot_size

    def _affordable_shares(self, cash: float, price: float, lot_size: int = 1) -> int:
        if cash <= 0 or price <= 0:
            return 0
        estimate = int(cash // price) // lot_size * lot_size
        while estimate > 0:
            fees = calculate_trade_cost(estimate, price, self.config)
            if fees["trade_value"] + fees["total_fees"] <= cash + 1e-9:
                return estimate
            estimate -= lot_size
        return 0

    def _fill(
        self,
        order_id: int,
        signal_date: pd.Timestamp,
        fill_date: pd.Timestamp,
        symbol: str,
        side: str,
        shares: int,
        reference_price: float,
        price: float,
        avg_price: float = 0.0,
        holding_days: int | None = None,
    ) -> Fill:
        costs = calculate_trade_cost(shares, price, self.config)
        capital_gains_tax = 0.0
        total_fees = costs["total_fees"]
        net_cash = costs["trade_value"] + total_fees
        realized_pnl = 0.0
        if side == "sell":
            pre_tax_realized_pnl = (price - avg_price) * shares - total_fees
            capital_gains_tax = max(pre_tax_realized_pnl, 0.0) * self.config.capital_gains_tax_rate
            total_fees += capital_gains_tax
            net_cash = costs["trade_value"] - total_fees
            realized_pnl = (price - avg_price) * shares - total_fees
        return Fill(
            order_id=order_id,
            signal_date=signal_date,
            fill_date=fill_date,
            symbol=symbol,
            side=side,
            shares=shares,
            reference_price=reference_price,
            price=price,
            trade_value=costs["trade_value"],
            brokerage=costs["brokerage"],
            sales_tax=costs["sales_tax"],
            other_fees=costs["other_fees"],
            capital_gains_tax=capital_gains_tax,
            total_fees=total_fees,
            net_cash=net_cash,
            realized_pnl=realized_pnl,
            slippage_cost=abs(price - reference_price) * shares,
            holding_days=holding_days,
        )

    def _record_fill(self, fill: Fill, trade_rows: list[dict[str, Any]]) -> None:
        trade_rows.append(
            {
                "order_id": fill.order_id,
                "signal_date": fill.signal_date,
                "fill_date": fill.fill_date,
                "symbol": fill.symbol,
                "side": fill.side,
                "shares": fill.shares,
                "reference_price": fill.reference_price,
                "price": fill.price,
                "trade_value": fill.trade_value,
                "brokerage": fill.brokerage,
                "sales_tax": fill.sales_tax,
                "other_fees": fill.other_fees,
                "capital_gains_tax": fill.capital_gains_tax,
                "total_fees": fill.total_fees,
                "slippage_cost": fill.slippage_cost,
                "net_cash": fill.net_cash,
                "realized_pnl": fill.realized_pnl,
                "holding_days": fill.holding_days,
            }
        )

    def _mark_equity(
        self,
        available_cash: float,
        settlements: Mapping[pd.Timestamp, float],
        holdings: Mapping[str, _Holding],
        last_close: Mapping[str, float],
    ) -> tuple[float, float, float]:
        unsettled_cash = sum(settlements.values())
        invested_value = sum(holding.shares * last_close.get(symbol, 0.0) for symbol, holding in holdings.items())
        return available_cash + unsettled_cash + invested_value, invested_value, unsettled_cash

    def _record_positions(
        self,
        current_date: pd.Timestamp,
        holdings: Mapping[str, _Holding],
        last_close: Mapping[str, float],
        rows: list[dict[str, Any]],
    ) -> None:
        for symbol, holding in sorted(holdings.items()):
            if holding.shares <= 0:
                continue
            close = last_close.get(symbol)
            market_value = holding.shares * close if close is not None else 0.0
            rows.append(
                {
                    "date": current_date,
                    "symbol": symbol,
                    "shares": holding.shares,
                    "avg_price": holding.avg_price,
                    "close": close,
                    "market_value": market_value,
                    "unrealized_pnl": market_value - holding.avg_price * holding.shares,
                }
            )

    def _context_positions(self, holdings: Mapping[str, _Holding], last_close: Mapping[str, float]) -> dict[str, Position]:
        positions: dict[str, Position] = {}
        for symbol, holding in holdings.items():
            if holding.shares <= 0:
                continue
            close = last_close.get(symbol)
            market_value = holding.shares * close if close is not None else 0.0
            positions[symbol] = Position(
                symbol=symbol,
                shares=holding.shares,
                avg_price=holding.avg_price,
                close=close,
                market_value=market_value,
                unrealized_pnl=market_value - holding.avg_price * holding.shares,
            )
        return positions

    def _history_until(self, current_date: pd.Timestamp) -> dict[str, pd.DataFrame]:
        history: dict[str, pd.DataFrame] = {}
        for symbol, frame in self.data.items():
            stop = int(frame["date"].searchsorted(current_date, side="right"))
            history[symbol] = frame.iloc[:stop]
        return history

    def _open_prices(self, current_date: pd.Timestamp) -> dict[str, float]:
        return {
            symbol: float(frame.loc[current_date, "open"])
            for symbol, frame in self._by_date.items()
            if current_date in frame.index
        }

    def _slipped_price(self, price: float, side: str) -> float:
        multiplier = 1 + self.config.slippage_bps / 10_000
        if side == "sell":
            multiplier = 1 - self.config.slippage_bps / 10_000
        return price * multiplier

    def _liquidity_share_cap(self, symbol: str, fill_date: pd.Timestamp) -> int:
        participation = self.config.max_volume_participation
        if participation is None:
            return 2**63 - 1
        frame = self.data[symbol]
        prior_stop = int(frame["date"].searchsorted(fill_date, side="left"))
        if prior_stop <= 0:
            return 0
        prior_volume = pd.to_numeric(frame.iloc[:prior_stop]["volume"], errors="coerce").dropna()
        if prior_volume.empty or float(prior_volume.iloc[-1]) <= 0:
            return 0
        shares = max(0, int(float(prior_volume.iloc[-1]) * participation))
        lot_size = self._board_lot_size(fill_date)
        return shares // lot_size * lot_size

    def _board_lot_size(self, fill_date: pd.Timestamp) -> int:
        lot_size = int(self.config.board_lot_size)
        schedule = sorted(
            ((pd.Timestamp(effective).normalize(), int(value)) for effective, value in self.config.board_lot_schedule),
            key=lambda item: item[0],
        )
        for effective_date, scheduled_size in schedule:
            if fill_date >= effective_date:
                lot_size = scheduled_size
            else:
                break
        return lot_size

    def _is_simulated_missed_fill(self, order_id: int, symbol: str, side: str, fill_date: pd.Timestamp) -> bool:
        probability = self.config.missed_fill_probability
        if probability <= 0:
            return False
        key = f"{self.config.random_seed}|{order_id}|{symbol}|{side}|{fill_date.date().isoformat()}"
        digest = hashlib.sha256(key.encode("ascii")).digest()
        sample = int.from_bytes(digest[:8], "big") / float(2**64)
        return sample < probability

    def _sessions_since_signal(self, signal_date: pd.Timestamp, current_date: pd.Timestamp) -> int:
        signal_position = self._calendar_positions[pd.Timestamp(signal_date)]
        current_position = self._calendar_positions[pd.Timestamp(current_date)]
        return current_position - signal_position

    def _settlement_date(self, fill_date: pd.Timestamp) -> pd.Timestamp | None:
        start = self.calendar.get_loc(fill_date)
        lag = self.config.settlement_lag_days
        schedule = sorted(
            ((pd.Timestamp(effective).normalize(), int(value)) for effective, value in self.config.settlement_lag_schedule),
            key=lambda item: item[0],
        )
        for effective_date, scheduled_lag in schedule:
            if fill_date >= effective_date:
                lag = scheduled_lag
            else:
                break
        target = start + lag
        if target >= len(self.calendar):
            return None
        return pd.Timestamp(self.calendar[target])

    def _strategy_target_weights(self, strategy: Any, context: StrategyContext) -> Mapping[str, float]:
        method = getattr(strategy, "target_weights", None)
        if not callable(method):
            return {}
        weights = method(context)
        return dict(weights or {})

    def _strategy_orders(self, strategy: Any, context: StrategyContext) -> list[Order]:
        method = getattr(strategy, "orders", None)
        if not callable(method):
            return []
        return [self._normalize_order(order) for order in (method(context) or [])]

    def _normalize_order(self, order: Order) -> Order:
        return Order(
            symbol=order.symbol.upper(),
            side=order.side.lower(),
            shares=order.shares,
            value=order.value,
            tag=order.tag,
        )

    def _validate_weights(self, weights: Mapping[str, float]) -> None:
        total = 0.0
        for symbol, weight in weights.items():
            if not symbol:
                raise ValueError("target weight symbol cannot be empty")
            if symbol.upper() not in self.data:
                raise ValueError(f"target weight symbol is not in backtest data: {symbol}")
            if not math.isfinite(float(weight)) or weight < 0:
                raise ValueError("target weights must be long-only")
            total += float(weight)
        if total > 1.0 + 1e-9:
            raise ValueError("target weights cannot exceed 100% gross exposure")

    def _symbols_trade_on(self, symbols: set[str], current_date: pd.Timestamp) -> bool:
        for symbol in symbols:
            if symbol not in self._by_date or current_date not in self._by_date[symbol].index:
                return False
        return True

    def _metrics(self, equity_curve: pd.DataFrame, trades: pd.DataFrame, positions: pd.DataFrame) -> dict[str, Any]:
        if equity_curve.empty:
            return {}
        final_equity = float(equity_curve["equity"].iloc[-1])
        total_contributions = float(equity_curve["cumulative_contributions"].iloc[-1]) if "cumulative_contributions" in equity_curve else float(self.config.initial_capital)
        denominator = total_contributions if total_contributions > 0 else float(self.config.initial_capital)
        total_return = final_equity / denominator - 1.0 if denominator > 0 else 0.0
        start = pd.Timestamp(equity_curve["date"].iloc[0])
        end = pd.Timestamp(equity_curve["date"].iloc[-1])
        years = max((end - start).days / 365.25, 0.0)
        strategy_growth = float(equity_curve["strategy_index"].iloc[-1]) if "strategy_index" in equity_curve else final_equity / denominator
        cagr = strategy_growth ** (1 / years) - 1 if years > 0 and strategy_growth > 0 else 0.0
        max_drawdown = float(equity_curve["drawdown"].min()) if "drawdown" in equity_curve else 0.0
        invested_ratio = equity_curve["invested_value"] / equity_curve["equity"].replace(0, pd.NA)

        sell_trades = trades[trades["side"] == "sell"] if not trades.empty else pd.DataFrame()
        realized = pd.to_numeric(sell_trades.get("realized_pnl", pd.Series(dtype=float)), errors="coerce").dropna()
        winners = realized[realized > 0]
        losers = realized[realized < 0]
        win_rate = float((realized > 0).mean()) if not realized.empty else None
        loss_rate = float((realized < 0).mean()) if not realized.empty else None
        gross_profit = float(winners.sum()) if not winners.empty else 0.0
        gross_loss = float(losers.sum()) if not losers.empty else 0.0
        profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else None
        expectancy = float(realized.mean()) if not realized.empty else None
        average_winner = float(winners.mean()) if not winners.empty else None
        average_loser = float(losers.mean()) if not losers.empty else None
        best_trade_profit_fraction = float(winners.max() / gross_profit) if gross_profit > 0 else None

        symbol_profit_fraction = None
        if not sell_trades.empty and gross_profit > 0:
            by_symbol = sell_trades.groupby("symbol")["realized_pnl"].sum()
            positive_symbol_profit = by_symbol[by_symbol > 0]
            if not positive_symbol_profit.empty:
                symbol_profit_fraction = float(positive_symbol_profit.max() / gross_profit)

        average_holding_days = None
        if not sell_trades.empty and "holding_days" in sell_trades:
            holding_days = pd.to_numeric(sell_trades["holding_days"], errors="coerce").dropna()
            if not holding_days.empty:
                average_holding_days = float(holding_days.mean())

        turnover = 0.0
        one_way_turnover = 0.0
        annualized_gross_turnover = 0.0
        annualized_one_way_turnover = 0.0
        if not trades.empty:
            average_equity = float(equity_curve["equity"].mean())
            turnover = float(trades["trade_value"].abs().sum() / average_equity) if average_equity else 0.0
            one_way_turnover = turnover / 2.0
            if years > 0:
                annualized_gross_turnover = turnover / years
                annualized_one_way_turnover = one_way_turnover / years

        returns = pd.to_numeric(equity_curve.get("daily_return", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        observed_returns = returns.iloc[1:] if len(returns) > 1 else returns
        daily_risk_free = self.config.annual_risk_free_rate / 252.0
        excess_returns = observed_returns - daily_risk_free
        daily_volatility = float(observed_returns.std(ddof=1)) if len(observed_returns) > 1 else 0.0
        annualized_volatility = daily_volatility * math.sqrt(252.0)
        sharpe = None
        if daily_volatility > 0:
            sharpe = float(excess_returns.mean() / daily_volatility * math.sqrt(252.0))
        downside = excess_returns[excess_returns < 0]
        sortino = None
        if len(downside) > 1:
            downside_deviation = float(downside.std(ddof=1))
            if downside_deviation > 0:
                sortino = float(excess_returns.mean() / downside_deviation * math.sqrt(252.0))
        calmar = float(cagr / abs(max_drawdown)) if max_drawdown < 0 else None

        dated_returns = pd.DataFrame({"date": pd.to_datetime(equity_curve["date"]), "return": returns})
        monthly_returns = dated_returns.groupby(dated_returns["date"].dt.to_period("M"))["return"].apply(lambda values: (1.0 + values).prod() - 1.0)
        yearly_returns = dated_returns.groupby(dated_returns["date"].dt.to_period("Y"))["return"].apply(lambda values: (1.0 + values).prod() - 1.0)
        worst_day = float(observed_returns.min()) if not observed_returns.empty else None
        worst_month = float(monthly_returns.min()) if not monthly_returns.empty else None
        worst_year = float(yearly_returns.min()) if not yearly_returns.empty else None

        total_brokerage = float(trades["brokerage"].sum()) if not trades.empty and "brokerage" in trades else 0.0
        total_sales_tax = float(trades["sales_tax"].sum()) if not trades.empty and "sales_tax" in trades else 0.0
        total_other_fees = float(trades["other_fees"].sum()) if not trades.empty and "other_fees" in trades else 0.0
        total_capital_gains_tax = float(trades["capital_gains_tax"].sum()) if not trades.empty and "capital_gains_tax" in trades else 0.0
        total_fees = float(trades["total_fees"].sum()) if not trades.empty and "total_fees" in trades else 0.0
        total_slippage = float(trades["slippage_cost"].sum()) if not trades.empty and "slippage_cost" in trades else 0.0

        final_position_concentration = None
        if not positions.empty:
            final_date = pd.to_datetime(positions["date"]).max()
            final_positions = positions[pd.to_datetime(positions["date"]) == final_date]
            total_market_value = float(final_positions["market_value"].sum())
            if total_market_value > 0:
                final_position_concentration = float(final_positions["market_value"].max() / total_market_value)

        cashflows: list[tuple[pd.Timestamp, float]] = []
        first_date = pd.Timestamp(equity_curve["date"].iloc[0])
        if self.config.initial_capital:
            cashflows.append((first_date, -float(self.config.initial_capital)))
        if "contribution" in equity_curve:
            for row in equity_curve.loc[equity_curve["contribution"] != 0, ["date", "contribution"]].itertuples(index=False):
                cashflows.append((pd.Timestamp(row.date), -float(row.contribution)))
        cashflows.append((pd.Timestamp(equity_curve["date"].iloc[-1]), final_equity))
        xirr = calculate_xirr(cashflows)

        return {
            "initial_capital": float(self.config.initial_capital),
            "recurring_contribution": float(self.config.recurring_contribution),
            "contribution_frequency": self.config.contribution_frequency,
            "total_contributions": total_contributions,
            "final_equity": final_equity,
            "total_return": float(total_return),
            "cagr": float(cagr),
            "xirr": xirr,
            "max_drawdown": max_drawdown,
            "max_drawdown_recovery_days": _maximum_underwater_days(equity_curve),
            "annualized_volatility": annualized_volatility,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "trade_count": int(len(trades)),
            "round_trip_count": int(len(sell_trades)),
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "average_winner": average_winner,
            "average_loser": average_loser,
            "average_holding_days_approx": average_holding_days,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "best_trade_profit_fraction": best_trade_profit_fraction,
            "best_symbol_profit_fraction": symbol_profit_fraction,
            "turnover": turnover,
            "gross_turnover": turnover,
            "one_way_turnover": one_way_turnover,
            "annualized_gross_turnover": annualized_gross_turnover,
            "annualized_one_way_turnover": annualized_one_way_turnover,
            "cash_utilization": float(invested_ratio.fillna(0.0).mean()),
            "market_exposure": float((equity_curve["invested_value"] > 0).mean()),
            "final_position_concentration": final_position_concentration,
            "worst_day": worst_day,
            "worst_month": worst_month,
            "worst_year": worst_year,
            "total_brokerage": total_brokerage,
            "total_sales_tax": total_sales_tax,
            "total_other_fees": total_other_fees,
            "total_capital_gains_tax": total_capital_gains_tax,
            "total_fees": total_fees,
            "total_slippage": total_slippage,
            "capital_gains_tax_rate": float(self.config.capital_gains_tax_rate),
            "slippage_bps": float(self.config.slippage_bps),
            "board_lot_size": int(self.config.board_lot_size),
        }


__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "Fill",
    "Order",
    "OrderStrategy",
    "Position",
    "StrategyContext",
    "SweepResult",
    "SweepRun",
    "TargetWeightStrategy",
    "calculate_xirr",
    "calculate_trade_cost",
    "load_historical_data",
    "normalize_bars",
    "run_parameter_sweep",
    "write_backtest_report",
    "write_sweep_report",
]
