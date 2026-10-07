from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ahl_api.client import AHL, read_dotenv


DEFAULT_SYMBOLS = ["FFC", "LUCK", "OGDC", "HUBC", "MARI"]
DEFAULT_RANK_WEIGHTS = {
    "mom_6m": 0.35,
    "mom_3m": 0.25,
    "trend": 0.20,
    "low_vol": 0.10,
    "liquidity": 0.10,
}
DEFAULT_RANK_ALLOCATIONS = [0.40, 0.30, 0.20]
DEFAULT_SYMBOL_CAPS = {
    "OGDC": 0.30,
    "HUBC": 0.25,
    "LUCK": 0.25,
    "FFC": 0.25,
    "MARI": 0.20,
}


@dataclass
class Holding:
    symbol: str
    quantity: int
    avg_cost: float | None
    last_price: float | None
    market_value: float


@dataclass
class PlannedOrder:
    symbol: str
    rank: int
    score: float
    allocation: float
    target_spend: float
    cap_room: float
    planned_spend: float
    limit_price: float
    shares: int
    estimated_value: float
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ranked DCA allocator for PSX cash equities through AHL NxG.")
    parser.add_argument("--deposit", type=float, required=True, help="Fresh monthly deposit amount to allocate in PKR.")
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--cash-reserve", type=float, default=0.10, help="Fraction of deposit to keep as cash.")
    parser.add_argument("--allocations", default="40,30,20", help="Top-rank allocation percentages, e.g. 40,30,20.")
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--min-history-rows", type=int, default=220)
    parser.add_argument("--min-trend-score", type=float, default=0.50)
    parser.add_argument("--max-ema20-extension", type=float, default=0.08)
    parser.add_argument("--max-rsi", type=float, default=75.0)
    parser.add_argument("--price-buffer-bps", type=float, default=0.0)
    parser.add_argument("--price-step", type=float, default=0.01)
    parser.add_argument("--min-shares", type=int, default=1)
    parser.add_argument("--order-timeout", type=int, default=30)
    parser.add_argument("--poll-seconds", type=int, default=3)
    parser.add_argument("--continue-on-unknown", action="store_true")
    parser.add_argument(
        "--allow-insufficient-buying-power",
        action="store_true",
        help="Attempt live orders even if current API buying power is lower than the proposed order value.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Confirm flow but do not submit live broker orders.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--env", default=".env")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = [symbol.upper() for symbol in args.symbols]
    allocations = parse_allocations(args.allocations, args.top_n)
    env = read_dotenv(Path(args.env))
    pin = env.get("pin") or env.get("PIN") or env.get("trade_pin") or env.get("TRADING_PIN") or ""

    client = AHL(
        {"user": env.get("user"), "pass": env.get("pass"), "pin": pin},
        dry_run=args.dry_run,
        allow_market_orders=False,
        check_price_bands=True,
        check_buying_power=not args.allow_insufficient_buying_power,
        audit_enabled=True,
        timeout=30,
    )
    apply_settings_base_url(client)
    session = client.login()

    print(f"account: {session.account}")
    print(f"mode: {'DRY RUN' if args.dry_run else 'LIVE AFTER CONFIRMATION'}")
    print(f"symbols: {', '.join(symbols)}")

    balance = client.fetch_balance()
    buying_power = client.fetch_buying_power()
    cash = as_float(balance.get("cash")) or 0.0
    regular_buying_power = as_float(buying_power.get("regular"))

    portfolio = client.fetch_portfolio()
    holdings = parse_portfolio(portfolio.get("raw", ""))
    holdings_value = sum(holding.market_value for holding in holdings.values())
    post_deposit_equity = holdings_value + cash + args.deposit
    investable_deposit = max(args.deposit * (1.0 - args.cash_reserve), 0.0)
    investable_cash = investable_deposit

    print()
    print("account snapshot")
    print(pd.DataFrame(
        [
            {"metric": "cash", "value": cash},
            {"metric": "regular_buying_power", "value": regular_buying_power},
            {"metric": "current_holdings_value", "value": holdings_value},
            {"metric": "monthly_deposit", "value": args.deposit},
            {"metric": "cash_reserve", "value": args.deposit * args.cash_reserve},
            {"metric": "investable_for_planning", "value": investable_cash},
            {"metric": "post_deposit_equity_for_caps", "value": post_deposit_equity},
        ]
    ).to_string(index=False))

    frames, tickers = load_market_data(client, symbols, years=args.years, min_rows=args.min_history_rows)
    rankings = build_rankings(frames, tickers)
    holdings_frame = build_holdings_frame(symbols, holdings, post_deposit_equity)
    orders = build_order_plan(
        rankings,
        holdings,
        tickers,
        allocations=allocations,
        investable_cash=investable_cash,
        post_deposit_equity=post_deposit_equity,
        min_trend_score=args.min_trend_score,
        max_ema20_extension=args.max_ema20_extension,
        max_rsi=args.max_rsi,
        price_buffer_bps=args.price_buffer_bps,
        price_step=args.price_step,
        min_shares=args.min_shares,
    )
    orders_frame = orders_to_frame(orders)

    print()
    print("ranked dca technical ranking")
    print(format_rankings(rankings))

    print()
    print("current holdings and cap room")
    print(format_money_frame(holdings_frame))

    print()
    print("proposed immediate buy orders")
    if orders_frame.empty:
        print("No immediate buy orders. Reasons are shown in the ranking table.")
    else:
        print(format_money_frame(orders_frame))

    if args.output_dir:
        write_outputs(Path(args.output_dir), rankings, holdings_frame, orders_frame)

    executable_orders = [order for order in orders if order.shares >= args.min_shares and order.reason == "buy"]
    if not executable_orders:
        print()
        print("nothing to submit")
        return 0

    estimated_order_value = sum(order.estimated_value for order in executable_orders)
    if (
        not args.dry_run
        and not args.allow_insufficient_buying_power
        and regular_buying_power is not None
        and estimated_order_value > regular_buying_power
    ):
        print()
        print(
            "buying power guard: proposed orders total "
            f"{estimated_order_value:,.2f}, but current API regular buying power is {regular_buying_power:,.2f}."
        )
        print("No orders submitted. Run again after the deposit is reflected, or pass --allow-insufficient-buying-power.")
        return 0

    print()
    print("This will submit LIVE limit buy orders unless --dry-run was used.")
    print("Type BUY to submit, anything else exits:")
    confirmation = input("> ").strip()
    if confirmation != "BUY":
        print("aborted: no orders submitted")
        return 0

    before_raw = current_log_raw_set(client, symbols)
    for order in executable_orders:
        print()
        print(f"submitting {order.symbol}: buy {order.shares} @ {order.limit_price:.2f} limit")
        result = client.create_order(
            order.symbol,
            "buy",
            order.shares,
            price=order.limit_price,
            order_type="limit",
            pin=pin,
        )
        print(json.dumps(result, indent=2, default=str))
        if args.dry_run:
            continue
        confirmed_rows = wait_for_new_logs(
            client,
            order.symbol,
            before_raw,
            timeout=args.order_timeout,
            poll_seconds=args.poll_seconds,
        )
        if confirmed_rows:
            before_raw.update(row.get("raw") for row in confirmed_rows if row.get("raw"))
            print("confirmation rows")
            print(pd.DataFrame(compact_order_row(row) for row in confirmed_rows).to_string(index=False))
        else:
            print(f"warning: no broker log confirmation for {order.symbol} within {args.order_timeout}s")
            if not args.continue_on_unknown:
                print("stopping remaining orders")
                break

    return 0


def parse_allocations(value: str, top_n: int) -> list[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        allocations = DEFAULT_RANK_ALLOCATIONS
    else:
        allocations = [float(part) / 100.0 if float(part) > 1 else float(part) for part in parts]
    if len(allocations) < top_n:
        allocations.extend([0.0] * (top_n - len(allocations)))
    return allocations[:top_n]


def apply_settings_base_url(client: AHL) -> None:
    try:
        settings = client.settings()
    except Exception:
        return
    base_url = settings.get("base_url") or settings.get("new_server_url")
    if isinstance(base_url, str) and base_url.strip():
        client.base_url = base_url.rstrip("/") + "/"


def load_market_data(
    client: AHL,
    symbols: list[str],
    *,
    years: int,
    min_rows: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    frames: dict[str, pd.DataFrame] = {}
    tickers: dict[str, dict[str, Any]] = {}
    today = pd.Timestamp(date.today())
    for symbol in symbols:
        rows = client.fetch_historical_daily(symbol, years=years)
        frame = pd.DataFrame(rows)
        if frame.empty or len(frame) < min_rows:
            print(f"skip {symbol}: only {len(frame)} historical rows")
            continue
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values("date")
        for column in ("open", "high", "low", "close", "volume"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        ticker: dict[str, Any]
        try:
            ticker = client.fetch_ticker(symbol)
            last = as_float(ticker.get("last"))
            volume = as_float(ticker.get("volume"))
            if last is not None:
                live_row = pd.DataFrame(
                    [
                        {
                            "date": today,
                            "open": as_float(ticker.get("close")) or last,
                            "high": as_float(ticker.get("high")),
                            "low": as_float(ticker.get("low")),
                            "close": last,
                            "volume": volume or 0.0,
                        }
                    ]
                )
                frame = frame[frame["date"] < today]
                frame = pd.concat([frame, live_row], ignore_index=True).sort_values("date")
        except Exception as exc:
            ticker = {"error": f"{type(exc).__name__}: {exc}"}
        frames[symbol] = frame
        tickers[symbol] = ticker
    return frames, tickers


def build_rankings(frames: dict[str, pd.DataFrame], tickers: dict[str, dict[str, Any]]) -> pd.DataFrame:
    metrics = []
    for symbol, frame in frames.items():
        close = pd.to_numeric(frame["close"], errors="coerce")
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        latest_close = last_valid(close)
        ema20 = last_valid(ema(close, 20))
        ema50 = last_valid(ema(close, 50))
        sma50 = last_valid(sma(close, 50))
        sma100 = last_valid(sma(close, 100))
        sma200 = last_valid(sma(close, 200))
        rsi14 = last_valid(rsi(close, 14))
        vol60 = ann_vol(close, 60)
        avg_value20 = avg_traded_value(close, volume, 20)
        avg_value60 = avg_traded_value(close, volume, 60)
        mom1m = return_over(close, 21)
        mom3m = return_over(close, 63)
        mom6m = return_over(close, 126)
        mom12m = return_over(close, 252)
        drawdown63 = max_drawdown(close.tail(63))
        trend = trend_score(latest_close, sma100, sma200)
        dist_ema20 = distance(latest_close, ema20)
        dist_sma100 = distance(latest_close, sma100)
        dist_sma200 = distance(latest_close, sma200)
        metrics.append(
            {
                "symbol": symbol,
                "latest_date": frame["date"].iloc[-1].date().isoformat(),
                "close": latest_close,
                "bid": as_float(tickers.get(symbol, {}).get("buy")),
                "ask": as_float(tickers.get(symbol, {}).get("sell")),
                "mom_1m_pct": pct(mom1m),
                "mom_3m_pct": pct(mom3m),
                "mom_6m_pct": pct(mom6m),
                "mom_12m_pct": pct(mom12m),
                "trend_score": trend,
                "ema20": ema20,
                "ema50": ema50,
                "sma50": sma50,
                "sma100": sma100,
                "sma200": sma200,
                "dist_ema20_pct": pct(dist_ema20),
                "dist_sma100_pct": pct(dist_sma100),
                "dist_sma200_pct": pct(dist_sma200),
                "rsi14": rsi14,
                "vol_60d_pct_ann": pct(vol60),
                "max_drawdown_63d_pct": pct(drawdown63),
                "avg_value_20d_pkr": avg_value20,
                "avg_value_60d_pkr": avg_value60,
                "_mom_3m": mom3m,
                "_mom_6m": mom6m,
                "_vol_60d": vol60,
                "_liquidity_60d": avg_value60,
            }
        )
    frame = pd.DataFrame(metrics)
    if frame.empty:
        return frame
    frame["mom_6m_rank_score"] = percentile_scores(frame, "_mom_6m", higher_is_better=True)
    frame["mom_3m_rank_score"] = percentile_scores(frame, "_mom_3m", higher_is_better=True)
    frame["low_vol_rank_score"] = percentile_scores(frame, "_vol_60d", higher_is_better=False)
    frame["liquidity_rank_score"] = percentile_scores(frame, "_liquidity_60d", higher_is_better=True)
    frame["score"] = (
        DEFAULT_RANK_WEIGHTS["mom_6m"] * frame["mom_6m_rank_score"]
        + DEFAULT_RANK_WEIGHTS["mom_3m"] * frame["mom_3m_rank_score"]
        + DEFAULT_RANK_WEIGHTS["trend"] * frame["trend_score"]
        + DEFAULT_RANK_WEIGHTS["low_vol"] * frame["low_vol_rank_score"]
        + DEFAULT_RANK_WEIGHTS["liquidity"] * frame["liquidity_rank_score"]
    )
    frame["entry_status"] = frame.apply(entry_status, axis=1)
    frame = frame.sort_values("score", ascending=False).reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame


def build_order_plan(
    rankings: pd.DataFrame,
    holdings: dict[str, Holding],
    tickers: dict[str, dict[str, Any]],
    *,
    allocations: list[float],
    investable_cash: float,
    post_deposit_equity: float,
    min_trend_score: float,
    max_ema20_extension: float,
    max_rsi: float,
    price_buffer_bps: float,
    price_step: float,
    min_shares: int,
) -> list[PlannedOrder]:
    orders: list[PlannedOrder] = []
    for _, row in rankings.head(len(allocations)).iterrows():
        symbol = str(row["symbol"])
        rank = int(row["rank"])
        allocation = allocations[rank - 1]
        target_spend = investable_cash * allocation
        current_value = holdings.get(symbol, Holding(symbol, 0, None, None, 0.0)).market_value
        cap = DEFAULT_SYMBOL_CAPS.get(symbol, 0.25)
        cap_room = max(post_deposit_equity * cap - current_value, 0.0)
        planned_spend = min(target_spend, cap_room)
        reason = buy_reason(row, planned_spend, min_trend_score, max_ema20_extension, max_rsi)
        limit_price = buy_limit_price(tickers.get(symbol, {}), row, price_buffer_bps=price_buffer_bps, price_step=price_step)
        shares = math.floor(planned_spend / limit_price) if reason == "buy" and limit_price > 0 else 0
        if shares < min_shares and reason == "buy":
            reason = "too_small_after_rounding"
            shares = 0
        estimated_value = shares * limit_price
        orders.append(
            PlannedOrder(
                symbol=symbol,
                rank=rank,
                score=float(row["score"]),
                allocation=allocation,
                target_spend=target_spend,
                cap_room=cap_room,
                planned_spend=planned_spend,
                limit_price=limit_price,
                shares=shares,
                estimated_value=estimated_value,
                reason=reason,
            )
        )
    return orders


def buy_reason(
    row: pd.Series,
    planned_spend: float,
    min_trend_score: float,
    max_ema20_extension: float,
    max_rsi: float,
) -> str:
    if planned_spend <= 0:
        return "no_cap_room_or_cash"
    if float(row.get("trend_score") or 0.0) < min_trend_score:
        return "weak_trend"
    dist_ema20 = float(row.get("dist_ema20_pct") or 0.0) / 100.0
    if dist_ema20 > max_ema20_extension:
        return "wait_for_pullback"
    rsi14 = row.get("rsi14")
    if pd.notna(rsi14) and float(rsi14) > max_rsi:
        return "overbought_rsi"
    return "buy"


def buy_limit_price(ticker: dict[str, Any], row: pd.Series, *, price_buffer_bps: float, price_step: float) -> float:
    price = as_float(ticker.get("sell")) or as_float(ticker.get("last")) or as_float(row.get("close")) or 0.0
    price *= 1.0 + price_buffer_bps / 10_000.0
    cap = ticker.get("cap")
    if isinstance(cap, dict):
        upper = as_float(cap.get("upperCapped"))
        if upper is not None:
            price = min(price, upper)
    return round_to_step(price, price_step)


def parse_portfolio(raw: str) -> dict[str, Holding]:
    holdings: dict[str, Holding] = {}
    for chunk in raw.strip().split("|"):
        if not chunk or chunk.startswith("$"):
            continue
        fields = chunk.split(";")
        if len(fields) < 6:
            continue
        symbol = fields[0].strip().upper()
        quantity = as_int(fields[2]) or 0
        avg_cost = as_float(fields[3])
        last_price = as_float(fields[4])
        market_value = as_float(fields[5]) or (quantity * (last_price or 0.0))
        holdings[symbol] = Holding(symbol, quantity, avg_cost, last_price, market_value)
    return holdings


def build_holdings_frame(symbols: list[str], holdings: dict[str, Holding], post_deposit_equity: float) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        holding = holdings.get(symbol, Holding(symbol, 0, None, None, 0.0))
        cap = DEFAULT_SYMBOL_CAPS.get(symbol, 0.25)
        cap_value = post_deposit_equity * cap
        rows.append(
            {
                "symbol": symbol,
                "shares": holding.quantity,
                "avg_cost": holding.avg_cost,
                "last_price": holding.last_price,
                "market_value": holding.market_value,
                "current_weight_pct": pct(holding.market_value / post_deposit_equity if post_deposit_equity else 0.0),
                "cap_pct": pct(cap),
                "cap_value": cap_value,
                "cap_room": max(cap_value - holding.market_value, 0.0),
            }
        )
    return pd.DataFrame(rows)


def orders_to_frame(orders: list[PlannedOrder]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": order.symbol,
                "rank": order.rank,
                "score": order.score,
                "allocation_pct": pct(order.allocation),
                "target_spend": order.target_spend,
                "cap_room": order.cap_room,
                "planned_spend": order.planned_spend,
                "limit_price": order.limit_price,
                "shares": order.shares,
                "estimated_value": order.estimated_value,
                "reason": order.reason,
            }
            for order in orders
        ]
    )


def current_log_raw_set(client: AHL, symbols: list[str]) -> set[str]:
    rows = []
    for symbol in symbols:
        rows.extend(client.fetch_open_orders(symbol))
        rows.extend(client.fetch_closed_orders(symbol))
        rows.extend(client.fetch_activity_logs(symbol))
    return {str(row.get("raw")) for row in rows if row.get("raw")}


def wait_for_new_logs(
    client: AHL,
    symbol: str,
    before_raw: set[str],
    *,
    timeout: int,
    poll_seconds: int,
) -> list[dict[str, Any]]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        rows = client.fetch_open_orders(symbol) + client.fetch_closed_orders(symbol) + client.fetch_activity_logs(symbol)
        new_rows = [row for row in rows if row.get("raw") and row.get("raw") not in before_raw]
        if new_rows:
            return new_rows
        time.sleep(poll_seconds)
    return []


def compact_order_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "log": row.get("logname"),
        "symbol": row.get("symbol"),
        "id": row.get("id"),
        "house_order_no": row.get("house_order_no"),
        "datetime": row.get("datetime"),
        "side": row.get("side"),
        "action": row.get("action") or row.get("action_code"),
        "status": row.get("status"),
        "price": row.get("price"),
        "amount": row.get("amount"),
        "filled": row.get("filled"),
        "remaining": row.get("remaining"),
    }


def write_outputs(output_dir: Path, rankings: pd.DataFrame, holdings: pd.DataFrame, orders: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rankings.to_csv(output_dir / f"ranked_dca_rankings_{stamp}.csv", index=False)
    holdings.to_csv(output_dir / f"ranked_dca_holdings_{stamp}.csv", index=False)
    orders.to_csv(output_dir / f"ranked_dca_orders_{stamp}.csv", index=False)


def entry_status(row: pd.Series) -> str:
    if float(row.get("trend_score") or 0.0) < 0.50:
        return "weak_trend"
    if float(row.get("dist_ema20_pct") or 0.0) > 8.0:
        return "extended"
    rsi14 = row.get("rsi14")
    if pd.notna(rsi14) and float(rsi14) > 75.0:
        return "overbought"
    return "ok"


def format_rankings(frame: pd.DataFrame) -> str:
    display = frame.drop(columns=[column for column in frame.columns if column.startswith("_")], errors="ignore").copy()
    return format_money_frame(display)


def format_money_frame(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:,.4f}")
    return display.to_string(index=False)


def percentile_scores(frame: pd.DataFrame, column: str, *, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().all():
        return pd.Series([0.0] * len(frame), index=frame.index)
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
    now = float(values.iloc[-1])
    if past == 0:
        return None
    return now / past - 1.0


def ann_vol(series: pd.Series, lookback: int) -> float | None:
    returns = pd.to_numeric(series, errors="coerce").pct_change().dropna().tail(lookback)
    if len(returns) < max(20, lookback // 2):
        return None
    return float(returns.std() * math.sqrt(252))


def avg_traded_value(close: pd.Series, volume: pd.Series, lookback: int) -> float | None:
    value = (pd.to_numeric(close, errors="coerce") * pd.to_numeric(volume, errors="coerce")).dropna().tail(lookback)
    if len(value) < max(10, lookback // 2):
        return None
    return float(value.mean())


def max_drawdown(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    running_max = values.cummax()
    drawdowns = values / running_max - 1.0
    return float(drawdowns.min())


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


def round_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return round(round(value / step) * step, 8)


def pct(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) * 100.0


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).replace(",", "").replace("%", "").strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    number = as_float(value)
    if number is None:
        return None
    return int(number)


if __name__ == "__main__":
    raise SystemExit(main())
