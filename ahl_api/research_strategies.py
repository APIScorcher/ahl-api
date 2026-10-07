"""Conservative close-only strategies for PSX research.

These strategies intentionally avoid ATR, true Donchian, VWAP, and intraday
rules because the public AHL/PSX history currently lacks trustworthy high/low
and historical intraday fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from ahl_api.backtest import Order, StrategyContext
from ahl_api.strategies import ema, sma


class DataUnavailableError(ValueError):
    """Raised when a strategy is asked to use fields the dataset cannot support."""


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Cutler RSI using simple rolling gains and losses."""

    _positive_int("period", period)
    values = pd.to_numeric(series, errors="coerce")
    delta = values.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.rolling(period, min_periods=period).mean()
    average_loss = losses.rolling(period, min_periods=period).mean()
    relative_strength = average_gain / average_loss.replace(0.0, pd.NA)
    result = 100.0 - 100.0 / (1.0 + relative_strength)
    result = result.mask((average_loss == 0.0) & (average_gain > 0.0), 100.0)
    return result.mask((average_loss == 0.0) & (average_gain == 0.0), 50.0)


def lagged_return(series: pd.Series, lookback: int, skip: int = 0) -> float | None:
    """Return over ``lookback`` rows ending ``skip`` rows before the latest close."""

    _positive_int("lookback", lookback)
    if skip < 0:
        raise ValueError("skip cannot be negative")
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= lookback + skip:
        return None
    end = float(values.iloc[-1 - skip])
    start = float(values.iloc[-1 - skip - lookback])
    if start <= 0:
        return None
    return end / start - 1.0


def median_traded_value(frame: pd.DataFrame, lookback: int = 60) -> float | None:
    """Median close times volume over the latest complete window."""

    _positive_int("lookback", lookback)
    if len(frame) < lookback:
        return None
    tail = frame.iloc[-lookback:]
    values = pd.to_numeric(tail["close"], errors="coerce") * pd.to_numeric(tail["volume"], errors="coerce")
    value = values.median(skipna=True)
    return None if pd.isna(value) else float(value)


def annualized_close_volatility(series: pd.Series, lookback: int = 63) -> float | None:
    _positive_int("lookback", lookback)
    values = pd.to_numeric(series, errors="coerce").dropna()
    returns = values.iloc[-lookback - 1 :].pct_change(fill_method=None).dropna()
    if len(returns) < lookback:
        return None
    value = returns.iloc[-lookback:].std(ddof=1) * (252.0**0.5)
    return None if pd.isna(value) else float(value)


def require_high_low(data: Mapping[str, pd.DataFrame], strategy_name: str) -> None:
    """Fail clearly instead of approximating high/low strategies from closes."""

    unavailable = [
        symbol
        for symbol, frame in data.items()
        if "high" not in frame
        or "low" not in frame
        or pd.to_numeric(frame["high"], errors="coerce").isna().all()
        or pd.to_numeric(frame["low"], errors="coerce").isna().all()
    ]
    if unavailable:
        preview = ", ".join(sorted(unavailable)[:5])
        raise DataUnavailableError(
            f"{strategy_name} requires verified high/low data; unavailable for {preview}"
        )


@dataclass(frozen=True)
class TechnicalCandidate:
    symbol: str
    score: float
    momentum: float
    volatility: float
    traded_value: float


@dataclass
class CrossSectionalMomentumStrategy:
    """Monthly 6/1-style momentum with liquidity, trend, and optional regime gates."""

    top_n: int = 3
    momentum_lookback: int = 126
    momentum_skip: int = 21
    trend_sma: int = 200
    liquidity_lookback: int = 60
    min_median_traded_value: float = 5_000_000.0
    min_price: float = 20.0
    max_zero_volume_fraction: float = 0.05
    volatility_lookback: int = 63
    sizing: str = "equal"
    max_gross_exposure: float = 0.90
    max_position_weight: float = 0.40
    rebalance: str = "monthly"
    market_symbol: str | None = "KSE100PR"
    market_sma: int = 200
    membership: Any = None
    _last_period: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_selection(self)

    def needs_history(self, date: pd.Timestamp) -> bool:
        return _period_key(date, self.rebalance) != self._last_period

    def target_weights(self, context: StrategyContext) -> dict[str, float]:
        period = _period_key(context.date, self.rebalance)
        if period == self._last_period:
            return {}
        self._last_period = period

        liquidations = {symbol: 0.0 for symbol in context.positions}
        if not _market_regime_is_positive(context, self.market_symbol, self.market_sma):
            return liquidations

        allowed = _membership_on(self.membership, context.date)
        candidates: list[TechnicalCandidate] = []
        for symbol, frame in context.history.items():
            symbol = symbol.upper()
            if symbol == (self.market_symbol or "").upper() or (allowed is not None and symbol not in allowed):
                continue
            candidate = _momentum_candidate(
                symbol,
                frame,
                context.date,
                momentum_lookback=self.momentum_lookback,
                momentum_skip=self.momentum_skip,
                trend_sma=self.trend_sma,
                liquidity_lookback=self.liquidity_lookback,
                min_median_traded_value=self.min_median_traded_value,
                min_price=self.min_price,
                max_zero_volume_fraction=self.max_zero_volume_fraction,
                volatility_lookback=self.volatility_lookback,
            )
            if candidate is not None:
                candidates.append(candidate)
        candidates.sort(key=lambda item: (-item.score, -item.traded_value, item.symbol))
        selected = candidates[: self.top_n]
        if not selected:
            return liquidations

        weights = _candidate_weights(
            selected,
            sizing=self.sizing,
            gross=self.max_gross_exposure,
            cap=self.max_position_weight,
        )
        return {**liquidations, **weights}


@dataclass
class CloseChannelBreakoutStrategy:
    """Breakout above prior closes with a lower close-channel/trend exit."""

    top_n: int = 3
    entry_lookback: int = 63
    exit_lookback: int = 20
    trend_sma: int = 150
    liquidity_lookback: int = 60
    min_median_traded_value: float = 5_000_000.0
    min_price: float = 20.0
    max_gross_exposure: float = 0.90
    max_position_weight: float = 0.40
    market_symbol: str | None = "KSE100PR"
    market_sma: int = 200
    membership: Any = None

    def __post_init__(self) -> None:
        for name in ("top_n", "entry_lookback", "exit_lookback", "trend_sma", "liquidity_lookback", "market_sma"):
            _positive_int(name, int(getattr(self, name)))
        _validate_exposure(self.max_gross_exposure, self.max_position_weight)

    def target_weights(self, context: StrategyContext) -> dict[str, float]:
        current = set(context.positions)
        if not _market_regime_is_positive(context, self.market_symbol, self.market_sma):
            return {symbol: 0.0 for symbol in current}

        allowed = _membership_on(self.membership, context.date)
        desired: set[str] = set()
        candidates: list[TechnicalCandidate] = []
        for symbol, frame in context.history.items():
            symbol = symbol.upper()
            if symbol == (self.market_symbol or "").upper() or (allowed is not None and symbol not in allowed):
                continue
            if not _has_current_bar(frame, context.date):
                if symbol in current:
                    desired.add(symbol)
                continue
            closes = pd.to_numeric(frame["close"], errors="coerce").dropna()
            if len(closes) <= max(self.entry_lookback, self.exit_lookback, self.trend_sma):
                continue
            close = float(closes.iloc[-1])
            trend = float(closes.iloc[-self.trend_sma :].mean())
            traded_value = median_traded_value(frame, self.liquidity_lookback)
            if traded_value is None or traded_value < self.min_median_traded_value or close < self.min_price:
                continue
            prior_exit = float(closes.iloc[-self.exit_lookback - 1 : -1].min())
            if symbol in current and close >= prior_exit and close >= trend:
                desired.add(symbol)
                continue
            prior_entry = float(closes.iloc[-self.entry_lookback - 1 : -1].max())
            if close <= prior_entry or close <= trend:
                continue
            volatility = annualized_close_volatility(closes, min(63, self.entry_lookback)) or 1.0
            strength = close / prior_entry - 1.0
            candidates.append(TechnicalCandidate(symbol, strength, strength, volatility, traded_value))

        slots = max(0, self.top_n - len(desired))
        candidates.sort(key=lambda item: (-item.score, -item.traded_value, item.symbol))
        desired.update(item.symbol for item in candidates[:slots])
        if desired == current:
            return {}
        selected = [
            TechnicalCandidate(symbol, 1.0, 0.0, 1.0, 0.0)
            for symbol in sorted(desired)
        ]
        weights = _candidate_weights(selected, sizing="equal", gross=self.max_gross_exposure, cap=self.max_position_weight)
        return {**{symbol: 0.0 for symbol in current}, **weights}


@dataclass
class RsiTrendPullbackStrategy:
    """Buy an RSI recovery after an oversold close while the long trend is rising."""

    rsi_period: int = 5
    oversold_rsi: float = 30.0
    recovery_ceiling_rsi: float = 50.0
    oversold_window: int = 3
    exit_rsi: float = 70.0
    trend_sma: int = 150
    trend_slope_lookback: int = 20
    exit_ema: int = 20
    max_positions: int = 2
    max_position_weight: float = 0.45
    cash_reserve: float = 0.10
    liquidity_lookback: int = 60
    min_median_traded_value: float = 5_000_000.0
    min_price: float = 20.0
    market_symbol: str | None = "KSE100PR"
    market_sma: int = 200
    membership: Any = None

    def __post_init__(self) -> None:
        for name in ("rsi_period", "oversold_window", "trend_sma", "trend_slope_lookback", "exit_ema", "max_positions"):
            _positive_int(name, int(getattr(self, name)))
        if not 0 < self.oversold_rsi < self.recovery_ceiling_rsi < self.exit_rsi <= 100:
            raise ValueError("RSI thresholds must be ordered within (0, 100]")
        if not 0 <= self.cash_reserve < 1:
            raise ValueError("cash_reserve must be in [0, 1)")
        if not 0 < self.max_position_weight <= 1:
            raise ValueError("max_position_weight must be in (0, 1]")

    def orders(self, context: StrategyContext) -> list[Order]:
        positive_regime = _market_regime_is_positive(context, self.market_symbol, self.market_sma)
        orders: list[Order] = []
        exits: set[str] = set()
        for symbol, position in context.positions.items():
            frame = context.history.get(symbol)
            if frame is None or not _has_current_bar(frame, context.date):
                continue
            closes = pd.to_numeric(frame["close"], errors="coerce").dropna()
            values = _recent_rsi(closes, self.rsi_period, 2)
            trend_value = float(closes.iloc[-self.trend_sma :].mean()) if len(closes) >= self.trend_sma else None
            exit_average = ema(closes, self.exit_ema)
            if len(closes) <= self.trend_sma or values.empty or pd.isna(values.iloc[-1]) or trend_value is None:
                continue
            trend_break = float(closes.iloc[-1]) < trend_value
            recovered = float(closes.iloc[-1]) >= float(exit_average.iloc[-1]) and float(values.iloc[-1]) >= 50.0
            if not positive_regime or trend_break or float(values.iloc[-1]) >= self.exit_rsi or recovered:
                orders.append(Order(symbol, "sell", shares=position.shares, tag="rsi_pullback_exit"))
                exits.add(symbol)

        if not positive_regime or context.unsettled_cash > 0:
            return orders
        allowed = _membership_on(self.membership, context.date)
        occupied = len(set(context.positions) - exits)
        slots = max(0, self.max_positions - occupied)
        if slots == 0 or context.available_cash <= 0:
            return orders

        candidates: list[tuple[float, str]] = []
        for symbol, frame in context.history.items():
            symbol = symbol.upper()
            if symbol in context.positions or symbol == (self.market_symbol or "").upper():
                continue
            if allowed is not None and symbol not in allowed:
                continue
            if not _basic_liquidity_gate(
                frame,
                context.date,
                self.liquidity_lookback,
                self.min_median_traded_value,
                self.min_price,
            ):
                continue
            closes = pd.to_numeric(frame["close"], errors="coerce").dropna()
            if len(closes) <= self.trend_sma + self.trend_slope_lookback:
                continue
            values = _recent_rsi(closes, self.rsi_period, self.oversold_window + 2)
            trend_value = float(closes.iloc[-self.trend_sma :].mean())
            prior_trend = float(
                closes.iloc[-self.trend_sma - self.trend_slope_lookback : -self.trend_slope_lookback].mean()
            )
            current_rsi = values.iloc[-1]
            if pd.isna(current_rsi):
                continue
            recent = values.iloc[-self.oversold_window - 1 : -1]
            rising_trend = trend_value > prior_trend
            if (
                float(closes.iloc[-1]) > trend_value
                and rising_trend
                and not recent.empty
                and float(recent.min()) <= self.oversold_rsi
                and float(values.iloc[-1]) > float(values.iloc[-2])
                and float(current_rsi) <= self.recovery_ceiling_rsi
            ):
                candidates.append((float(current_rsi), symbol))

        candidates.sort(key=lambda item: (item[0], item[1]))
        entrants = candidates[:slots]
        if not entrants:
            return orders
        deployable = min(context.available_cash, context.equity * (1.0 - self.cash_reserve))
        value = min(context.equity * self.max_position_weight, deployable / len(entrants))
        orders.extend(Order(symbol, "buy", value=value, tag="rsi_pullback_entry") for _, symbol in entrants)
        return orders


@dataclass
class BollingerTrendReversionStrategy:
    """Enter after a close crosses back above a lower Bollinger band in an uptrend."""

    band_window: int = 20
    band_std: float = 2.0
    trend_sma: int = 150
    max_positions: int = 2
    max_position_weight: float = 0.45
    cash_reserve: float = 0.10
    liquidity_lookback: int = 60
    min_median_traded_value: float = 5_000_000.0
    min_price: float = 20.0
    market_symbol: str | None = "KSE100PR"
    market_sma: int = 200

    def __post_init__(self) -> None:
        for name in ("band_window", "trend_sma", "max_positions", "liquidity_lookback", "market_sma"):
            _positive_int(name, int(getattr(self, name)))
        if self.band_std <= 0:
            raise ValueError("band_std must be positive")

    def orders(self, context: StrategyContext) -> list[Order]:
        regime = _market_regime_is_positive(context, self.market_symbol, self.market_sma)
        orders: list[Order] = []
        exits: set[str] = set()
        for symbol, position in context.positions.items():
            frame = context.history.get(symbol)
            if frame is None or not _has_current_bar(frame, context.date):
                continue
            closes = pd.to_numeric(frame["close"], errors="coerce").dropna()
            if len(closes) < max(self.band_window, self.trend_sma):
                continue
            middle_value = float(closes.iloc[-self.band_window :].mean())
            trend_value = float(closes.iloc[-self.trend_sma :].mean())
            if not regime or float(closes.iloc[-1]) >= middle_value or float(closes.iloc[-1]) < trend_value:
                orders.append(Order(symbol, "sell", shares=position.shares, tag="bollinger_exit"))
                exits.add(symbol)
        if not regime or context.unsettled_cash > 0:
            return orders

        slots = max(0, self.max_positions - len(set(context.positions) - exits))
        candidates: list[tuple[float, str]] = []
        for symbol, frame in context.history.items():
            symbol = symbol.upper()
            if symbol in context.positions or symbol == (self.market_symbol or "").upper():
                continue
            if not _basic_liquidity_gate(frame, context.date, self.liquidity_lookback, self.min_median_traded_value, self.min_price):
                continue
            closes = pd.to_numeric(frame["close"], errors="coerce").dropna()
            if len(closes) <= max(self.band_window, self.trend_sma):
                continue
            previous_window = closes.iloc[-self.band_window - 1 : -1]
            current_window = closes.iloc[-self.band_window :]
            previous_middle = float(previous_window.mean())
            current_middle = float(current_window.mean())
            previous_lower = previous_middle - self.band_std * float(previous_window.std(ddof=1))
            current_lower = current_middle - self.band_std * float(current_window.std(ddof=1))
            trend_value = float(closes.iloc[-self.trend_sma :].mean())
            crossed_back = float(closes.iloc[-2]) <= previous_lower and float(closes.iloc[-1]) > current_lower
            if crossed_back and float(closes.iloc[-1]) > trend_value:
                distance = float(closes.iloc[-1] / current_middle - 1.0)
                candidates.append((distance, symbol))
        candidates.sort(key=lambda item: (item[0], item[1]))
        entrants = candidates[:slots]
        if not entrants:
            return orders
        deployable = min(context.available_cash, context.equity * (1.0 - self.cash_reserve))
        value = min(context.equity * self.max_position_weight, deployable / len(entrants))
        orders.extend(Order(symbol, "buy", value=value, tag="bollinger_entry") for _, symbol in entrants)
        return orders


@dataclass
class TechnicalEnsembleStrategy:
    """Fixed-vote monthly ensemble; all votes are specified before evaluation."""

    top_n: int = 3
    momentum_lookback: int = 126
    trend_sma: int = 200
    breakout_lookback: int = 63
    rsi_period: int = 14
    min_votes: int = 3
    liquidity_lookback: int = 60
    min_median_traded_value: float = 5_000_000.0
    min_price: float = 20.0
    max_gross_exposure: float = 0.90
    max_position_weight: float = 0.40
    market_symbol: str | None = "KSE100PR"
    market_sma: int = 200
    rebalance: str = "monthly"
    _last_period: Any = field(default=None, init=False, repr=False)

    def needs_history(self, date: pd.Timestamp) -> bool:
        return _period_key(date, self.rebalance) != self._last_period

    def target_weights(self, context: StrategyContext) -> dict[str, float]:
        period = _period_key(context.date, self.rebalance)
        if period == self._last_period:
            return {}
        self._last_period = period
        liquidations = {symbol: 0.0 for symbol in context.positions}
        if not _market_regime_is_positive(context, self.market_symbol, self.market_sma):
            return liquidations

        candidates: list[TechnicalCandidate] = []
        required = max(self.momentum_lookback, self.trend_sma, self.breakout_lookback, self.liquidity_lookback)
        for symbol, frame in context.history.items():
            symbol = symbol.upper()
            if symbol == (self.market_symbol or "").upper() or not _has_current_bar(frame, context.date) or len(frame) <= required:
                continue
            if not _basic_liquidity_gate(frame, context.date, self.liquidity_lookback, self.min_median_traded_value, self.min_price):
                continue
            closes = pd.to_numeric(frame["close"], errors="coerce").dropna()
            momentum = lagged_return(closes, self.momentum_lookback, 21)
            trend = sma(closes, self.trend_sma).iloc[-1]
            short_average = ema(closes, 20).iloc[-1]
            rsi_value = rsi(closes, self.rsi_period).iloc[-1]
            prior_high = closes.iloc[-self.breakout_lookback - 1 : -1].max()
            if momentum is None or any(pd.isna(value) for value in (trend, short_average, rsi_value, prior_high)):
                continue
            votes = int(momentum > 0.0)
            votes += int(float(closes.iloc[-1]) > float(trend))
            votes += int(float(closes.iloc[-1]) > float(short_average))
            votes += int(float(closes.iloc[-1]) >= float(prior_high))
            votes += int(45.0 <= float(rsi_value) <= 75.0)
            if votes < self.min_votes:
                continue
            volatility = annualized_close_volatility(closes, 63) or 1.0
            traded_value = median_traded_value(frame, self.liquidity_lookback) or 0.0
            candidates.append(TechnicalCandidate(symbol, votes + momentum, momentum, volatility, traded_value))
        candidates.sort(key=lambda item: (-item.score, -item.traded_value, item.symbol))
        selected = candidates[: self.top_n]
        if not selected:
            return liquidations
        return {
            **liquidations,
            **_candidate_weights(selected, sizing="equal", gross=self.max_gross_exposure, cap=self.max_position_weight),
        }


def _momentum_candidate(
    symbol: str,
    frame: pd.DataFrame,
    date: pd.Timestamp,
    *,
    momentum_lookback: int,
    momentum_skip: int,
    trend_sma: int,
    liquidity_lookback: int,
    min_median_traded_value: float,
    min_price: float,
    max_zero_volume_fraction: float,
    volatility_lookback: int,
) -> TechnicalCandidate | None:
    required = max(momentum_lookback + momentum_skip, trend_sma, liquidity_lookback, volatility_lookback)
    if not _has_current_bar(frame, date) or len(frame) <= required:
        return None
    closes = pd.to_numeric(frame["close"], errors="coerce").dropna()
    volumes = pd.to_numeric(frame["volume"], errors="coerce")
    close = float(closes.iloc[-1])
    momentum = lagged_return(closes, momentum_lookback, momentum_skip)
    trend = sma(closes, trend_sma).iloc[-1]
    traded_value = median_traded_value(frame, liquidity_lookback)
    volatility = annualized_close_volatility(closes, volatility_lookback)
    zero_fraction = float((volumes.iloc[-liquidity_lookback:] <= 0).fillna(True).mean())
    if (
        momentum is None
        or momentum <= 0
        or pd.isna(trend)
        or close <= float(trend)
        or close < min_price
        or traded_value is None
        or traded_value < min_median_traded_value
        or volatility is None
        or volatility <= 0
        or zero_fraction > max_zero_volume_fraction
    ):
        return None
    return TechnicalCandidate(symbol, momentum, momentum, volatility, traded_value)


def _candidate_weights(
    candidates: list[TechnicalCandidate],
    *,
    sizing: str,
    gross: float,
    cap: float,
) -> dict[str, float]:
    if not candidates:
        return {}
    if sizing == "equal":
        raw = {item.symbol: 1.0 for item in candidates}
    elif sizing == "inverse_volatility":
        raw = {item.symbol: 1.0 / max(item.volatility, 1e-9) for item in candidates}
    else:
        raise ValueError("sizing must be equal or inverse_volatility")
    return _capped_normalize(raw, gross, cap)


def _capped_normalize(raw: Mapping[str, float], gross: float, cap: float) -> dict[str, float]:
    weights = {symbol: 0.0 for symbol in raw}
    remaining = min(gross, cap * len(raw))
    active = {symbol for symbol, value in raw.items() if value > 0}
    while active and remaining > 1e-12:
        scale = sum(raw[symbol] for symbol in active)
        if scale <= 0:
            break
        allocated = 0.0
        saturated: set[str] = set()
        for symbol in active:
            addition = remaining * raw[symbol] / scale
            room = cap - weights[symbol]
            actual = min(addition, room)
            weights[symbol] += actual
            allocated += actual
            if room - actual <= 1e-12:
                saturated.add(symbol)
        remaining -= allocated
        active -= saturated
        if allocated <= 1e-12:
            break
    return weights


def _market_regime_is_positive(context: StrategyContext, symbol: str | None, window: int) -> bool:
    if symbol is None:
        return True
    frame = context.history.get(symbol.upper())
    if frame is None or not _has_current_bar(frame, context.date):
        return False
    closes = pd.to_numeric(frame["close"], errors="coerce").dropna()
    if len(closes) < window:
        return False
    average = float(closes.iloc[-window:].mean())
    return float(closes.iloc[-1]) > average


def _recent_rsi(closes: pd.Series, period: int, values_needed: int) -> pd.Series:
    window = max(period + values_needed + 1, period + 2)
    return rsi(closes.iloc[-window:], period).dropna()


def _basic_liquidity_gate(
    frame: pd.DataFrame,
    date: pd.Timestamp,
    lookback: int,
    minimum: float,
    min_price: float,
) -> bool:
    if not _has_current_bar(frame, date) or len(frame) < lookback:
        return False
    close = pd.to_numeric(frame["close"], errors="coerce").iloc[-1]
    traded_value = median_traded_value(frame, lookback)
    return not pd.isna(close) and float(close) >= min_price and traded_value is not None and traded_value >= minimum


def _has_current_bar(frame: pd.DataFrame, date: pd.Timestamp) -> bool:
    if frame.empty or "date" not in frame:
        return False
    latest = pd.to_datetime(frame["date"], errors="coerce").dropna()
    return not latest.empty and pd.Timestamp(latest.iloc[-1]).normalize() == pd.Timestamp(date).normalize()


def _membership_on(schedule: Any, date: pd.Timestamp) -> set[str] | None:
    if schedule is None:
        return None
    method = getattr(schedule, "symbols_on", None)
    if not callable(method):
        raise TypeError("membership must provide symbols_on(date)")
    return {str(symbol).upper() for symbol in method(date)}


def _validate_selection(strategy: CrossSectionalMomentumStrategy) -> None:
    for name in (
        "top_n",
        "momentum_lookback",
        "trend_sma",
        "liquidity_lookback",
        "volatility_lookback",
        "market_sma",
    ):
        _positive_int(name, int(getattr(strategy, name)))
    if strategy.momentum_skip < 0:
        raise ValueError("momentum_skip cannot be negative")
    if not 0 <= strategy.max_zero_volume_fraction <= 1:
        raise ValueError("max_zero_volume_fraction must be in [0, 1]")
    _validate_exposure(strategy.max_gross_exposure, strategy.max_position_weight)
    _period_key(pd.Timestamp("2000-01-01"), strategy.rebalance)
    if strategy.sizing not in {"equal", "inverse_volatility"}:
        raise ValueError("sizing must be equal or inverse_volatility")


def _validate_exposure(gross: float, cap: float) -> None:
    if not 0 < gross <= 1:
        raise ValueError("max_gross_exposure must be in (0, 1]")
    if not 0 < cap <= 1:
        raise ValueError("max_position_weight must be in (0, 1]")


def _positive_int(name: str, value: int) -> None:
    if int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _period_key(value: pd.Timestamp, rebalance: str) -> Any:
    normalized = rebalance.lower()
    if normalized == "daily":
        return pd.Timestamp(value).normalize()
    if normalized == "weekly":
        return pd.Timestamp(value).to_period("W-FRI")
    if normalized == "monthly":
        return pd.Timestamp(value).to_period("M")
    raise ValueError("rebalance must be daily, weekly, or monthly")


__all__ = [
    "BollingerTrendReversionStrategy",
    "CloseChannelBreakoutStrategy",
    "CrossSectionalMomentumStrategy",
    "DataUnavailableError",
    "RsiTrendPullbackStrategy",
    "TechnicalCandidate",
    "TechnicalEnsembleStrategy",
    "annualized_close_volatility",
    "lagged_return",
    "median_traded_value",
    "require_high_low",
    "rsi",
]
