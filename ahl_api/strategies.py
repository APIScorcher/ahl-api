"""Reusable backtest strategies for PSX cash equities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from ahl_api.backtest import Order, Position, StrategyContext


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average with a full-window warmup."""

    _validate_positive_int("window", window)
    return pd.to_numeric(series, errors="coerce").rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with a full-span warmup."""

    _validate_positive_int("span", span)
    return pd.to_numeric(series, errors="coerce").ewm(span=span, adjust=False, min_periods=span).mean()


def rate_of_change(series: pd.Series, lookback: int) -> pd.Series:
    """Close-to-close rate of change over ``lookback`` rows."""

    _validate_positive_int("lookback", lookback)
    values = pd.to_numeric(series, errors="coerce")
    return values / values.shift(lookback) - 1.0


def average_traded_value(frame: pd.DataFrame, lookback: int) -> pd.Series:
    """Rolling average of close multiplied by volume."""

    _validate_positive_int("lookback", lookback)
    traded_value = pd.to_numeric(frame["close"], errors="coerce") * pd.to_numeric(frame["volume"], errors="coerce")
    return traded_value.rolling(window=lookback, min_periods=lookback).mean()


@dataclass(frozen=True)
class RankedSymbol:
    symbol: str
    momentum: float
    avg_traded_value: float
    close: float


def rank_momentum_universe(
    history: Mapping[str, pd.DataFrame],
    *,
    top_n: int = 5,
    momentum_lookback: int = 120,
    liquidity_lookback: int = 20,
    min_avg_traded_value: float = 0.0,
    trend_sma: int | None = None,
) -> list[RankedSymbol]:
    """Rank symbols by momentum after liquidity and optional trend filters."""

    _validate_positive_int("top_n", top_n)
    _validate_positive_int("momentum_lookback", momentum_lookback)
    _validate_positive_int("liquidity_lookback", liquidity_lookback)
    if trend_sma is not None:
        _validate_positive_int("trend_sma", trend_sma)

    ranked: list[RankedSymbol] = []
    for symbol, frame in history.items():
        if frame.empty:
            continue
        closes = pd.to_numeric(frame["close"], errors="coerce")
        if len(closes.dropna()) <= momentum_lookback:
            continue
        momentum = rate_of_change(closes, momentum_lookback).iloc[-1]
        avg_value = average_traded_value(frame, liquidity_lookback).iloc[-1]
        close = closes.iloc[-1]
        if pd.isna(momentum) or pd.isna(avg_value) or pd.isna(close):
            continue
        if float(avg_value) < min_avg_traded_value:
            continue
        if trend_sma is not None:
            trend_value = sma(closes, trend_sma).iloc[-1]
            if pd.isna(trend_value) or float(close) <= float(trend_value):
                continue
        ranked.append(
            RankedSymbol(
                symbol=symbol.upper(),
                momentum=float(momentum),
                avg_traded_value=float(avg_value),
                close=float(close),
            )
        )

    ranked.sort(key=lambda item: (-item.momentum, -item.avg_traded_value, item.symbol))
    return ranked[:top_n]


@dataclass
class LiquidityFilteredMomentumStrategy:
    top_n: int = 5
    momentum_lookback: int = 120
    liquidity_lookback: int = 20
    min_avg_traded_value: float = 0.0
    rebalance: str = "monthly"
    max_gross_exposure: float = 1.0
    _last_rebalance_period: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_common(self.top_n, self.momentum_lookback, self.liquidity_lookback, self.rebalance, self.max_gross_exposure)

    def needs_history(self, date: pd.Timestamp) -> bool:
        return _period_key(date, self.rebalance) != self._last_rebalance_period

    def target_weights(self, context: StrategyContext) -> dict[str, float]:
        period = _period_key(context.date, self.rebalance)
        if period == self._last_rebalance_period:
            return {}

        ranked = rank_momentum_universe(
            context.history,
            top_n=self.top_n,
            momentum_lookback=self.momentum_lookback,
            liquidity_lookback=self.liquidity_lookback,
            min_avg_traded_value=self.min_avg_traded_value,
        )
        self._last_rebalance_period = period
        if not ranked:
            return {symbol: 0.0 for symbol in context.positions}

        weight = self.max_gross_exposure / len(ranked)
        return {item.symbol: weight for item in ranked}


@dataclass
class MomentumTrendFilterStrategy:
    top_n: int = 5
    momentum_lookback: int = 120
    liquidity_lookback: int = 20
    min_avg_traded_value: float = 0.0
    trend_sma: int = 100
    rebalance: str = "monthly"
    max_gross_exposure: float = 1.0
    _last_rebalance_period: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_common(self.top_n, self.momentum_lookback, self.liquidity_lookback, self.rebalance, self.max_gross_exposure)
        _validate_positive_int("trend_sma", self.trend_sma)

    def needs_history(self, date: pd.Timestamp) -> bool:
        return _period_key(date, self.rebalance) != self._last_rebalance_period

    def target_weights(self, context: StrategyContext) -> dict[str, float]:
        period = _period_key(context.date, self.rebalance)
        if period == self._last_rebalance_period:
            return {}

        ranked = rank_momentum_universe(
            context.history,
            top_n=self.top_n,
            momentum_lookback=self.momentum_lookback,
            liquidity_lookback=self.liquidity_lookback,
            min_avg_traded_value=self.min_avg_traded_value,
            trend_sma=self.trend_sma,
        )
        self._last_rebalance_period = period
        if not ranked:
            return {symbol: 0.0 for symbol in context.positions}

        weight = self.max_gross_exposure / len(ranked)
        return {item.symbol: weight for item in ranked}


@dataclass
class PullbackDcaStrategy:
    top_n: int = 5
    momentum_lookback: int = 120
    liquidity_lookback: int = 20
    min_avg_traded_value: float = 0.0
    trend_sma: int = 100
    dip_ema: int = 20
    dip_thresholds: Sequence[float] = (0.03, 0.06, 0.09)
    take_profit_above_ema: float = 0.06
    max_position_weight: float = 0.20
    universe_rebalance: str = "monthly"
    _active_symbols: set[str] = field(default_factory=set, init=False, repr=False)
    _last_universe_period: Any = field(default=None, init=False, repr=False)
    _triggered_thresholds: dict[str, set[float]] = field(default_factory=dict, init=False, repr=False)
    _position_symbols: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_common(self.top_n, self.momentum_lookback, self.liquidity_lookback, self.universe_rebalance, 1.0)
        _validate_positive_int("trend_sma", self.trend_sma)
        _validate_positive_int("dip_ema", self.dip_ema)
        if self.max_position_weight <= 0 or self.max_position_weight > 1:
            raise ValueError("max_position_weight must be in the range (0, 1]")
        if self.take_profit_above_ema < 0:
            raise ValueError("take_profit_above_ema cannot be negative")
        thresholds = tuple(sorted(float(value) for value in self.dip_thresholds))
        if not thresholds or any(value <= 0 for value in thresholds):
            raise ValueError("dip_thresholds must contain positive values")
        self.dip_thresholds = thresholds

    def orders(self, context: StrategyContext) -> list[Order]:
        self._refresh_universe(context)
        self._reset_closed_positions(context.positions)

        orders: list[Order] = []
        exit_symbols = self._exit_orders(context, orders)
        self._entry_orders(context, exit_symbols, orders)
        return orders

    def _refresh_universe(self, context: StrategyContext) -> None:
        period = _period_key(context.date, self.universe_rebalance)
        if period == self._last_universe_period and self._active_symbols:
            return

        ranked = rank_momentum_universe(
            context.history,
            top_n=self.top_n,
            momentum_lookback=self.momentum_lookback,
            liquidity_lookback=self.liquidity_lookback,
            min_avg_traded_value=self.min_avg_traded_value,
            trend_sma=self.trend_sma,
        )
        self._active_symbols = {item.symbol for item in ranked}
        self._last_universe_period = period

    def _reset_closed_positions(self, positions: Mapping[str, Position]) -> None:
        current_symbols = set(positions)
        for symbol in self._position_symbols - current_symbols:
            if symbol not in positions:
                self._triggered_thresholds[symbol] = set()
        self._position_symbols = current_symbols

    def _exit_orders(self, context: StrategyContext, orders: list[Order]) -> set[str]:
        exit_symbols: set[str] = set()
        for symbol, position in context.positions.items():
            frame = context.history.get(symbol)
            if frame is None or frame.empty:
                continue
            if self._trend_broken(frame):
                orders.append(Order(symbol=symbol, side="sell", shares=position.shares, tag="trend_exit"))
                exit_symbols.add(symbol)
                continue
            if self._take_profit_signal(frame):
                orders.append(Order(symbol=symbol, side="sell", shares=position.shares, tag="take_profit"))
                exit_symbols.add(symbol)
        return exit_symbols

    def _entry_orders(self, context: StrategyContext, exit_symbols: set[str], orders: list[Order]) -> None:
        for symbol in sorted(self._active_symbols):
            if symbol in exit_symbols:
                continue
            frame = context.history.get(symbol)
            if frame is None or frame.empty or not self._in_uptrend(frame):
                continue
            close = _latest_numeric(frame["close"])
            ema_value = _latest_indicator(ema(frame["close"], self.dip_ema))
            if close is None or ema_value is None:
                continue

            position = context.positions.get(symbol)
            current_value = position.market_value if position else 0.0
            max_value = context.equity * self.max_position_weight
            remaining_value = max_value - current_value
            if remaining_value <= 0:
                continue

            triggered = self._triggered_thresholds.setdefault(symbol, set())
            tranche_value = max_value / len(self.dip_thresholds)
            for threshold in self.dip_thresholds:
                if threshold in triggered:
                    continue
                if close <= ema_value * (1.0 - threshold):
                    value = min(tranche_value, remaining_value)
                    if value <= 0:
                        break
                    orders.append(Order(symbol=symbol, side="buy", value=value, tag=f"dca_buy_{int(threshold * 100)}pct"))
                    triggered.add(threshold)
                    remaining_value -= value

    def _in_uptrend(self, frame: pd.DataFrame) -> bool:
        close = _latest_numeric(frame["close"])
        trend_value = _latest_indicator(sma(frame["close"], self.trend_sma))
        return close is not None and trend_value is not None and close > trend_value

    def _trend_broken(self, frame: pd.DataFrame) -> bool:
        close = _latest_numeric(frame["close"])
        trend_value = _latest_indicator(sma(frame["close"], self.trend_sma))
        return close is not None and trend_value is not None and close < trend_value

    def _take_profit_signal(self, frame: pd.DataFrame) -> bool:
        closes = pd.to_numeric(frame["close"], errors="coerce").dropna()
        if len(closes) < 2:
            return False
        close = float(closes.iloc[-1])
        previous_close = float(closes.iloc[-2])
        ema_value = _latest_indicator(ema(frame["close"], self.dip_ema))
        if ema_value is None:
            return False
        return close > ema_value * (1.0 + self.take_profit_above_ema) and close < previous_close


def _validate_positive_int(name: str, value: int) -> None:
    if int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_common(top_n: int, momentum_lookback: int, liquidity_lookback: int, rebalance: str, exposure: float) -> None:
    _validate_positive_int("top_n", top_n)
    _validate_positive_int("momentum_lookback", momentum_lookback)
    _validate_positive_int("liquidity_lookback", liquidity_lookback)
    _period_key(pd.Timestamp("2000-01-01"), rebalance)
    if exposure <= 0 or exposure > 1:
        raise ValueError("max exposure must be in the range (0, 1]")


def _period_key(value: pd.Timestamp, rebalance: str) -> Any:
    normalized = rebalance.lower()
    if normalized == "daily":
        return pd.Timestamp(value).normalize()
    if normalized == "weekly":
        return pd.Timestamp(value).to_period("W-FRI")
    if normalized == "monthly":
        return pd.Timestamp(value).to_period("M")
    raise ValueError("rebalance must be one of: daily, weekly, monthly")


def _latest_numeric(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def _latest_indicator(series: pd.Series) -> float | None:
    value = series.iloc[-1] if not series.empty else pd.NA
    if pd.isna(value):
        return None
    return float(value)


__all__ = [
    "LiquidityFilteredMomentumStrategy",
    "MomentumTrendFilterStrategy",
    "PullbackDcaStrategy",
    "RankedSymbol",
    "average_traded_value",
    "ema",
    "rank_momentum_universe",
    "rate_of_change",
    "sma",
]
