# Research and backtesting guide

This workspace is for building a Python interoperability layer around your own AHL trading account.

Current boundary:
- Use only accounts/devices you own or are authorized to test.
- Do not bypass authentication, tamper with broker controls, or extract unrelated user data.
- Keep strategy development in dry-run mode until the broker responses are fully validated.

## Python API

Create `.env`:

```text
user=YOUR_USERNAME
pass=YOUR_PASSWORD
```

Use the client in default dry-run mode:

```python
from ahl_api.client import AHL, read_dotenv

env = read_dotenv()
ahl = AHL({"user": env["user"], "pass": env["pass"]})

session = ahl.login()
print(session.account)

print(ahl.fetch_accounts())
print(ahl.fetch_balance())
print(ahl.fetch_buying_power())
print(ahl.fetch_portfolio())
print(ahl.fetch_exposure())
print(ahl.fetch_exposure_by_market())

print(ahl.fetch_market_status())
print(ahl.fetch_ticker("HBL", market="REG"))
print(ahl.fetch_tickers(["HBL", "OGDC"], market="REG"))
print(ahl.fetch_ohlcv("HBL", market="REG", interval="5"))
print(ahl.fetch_historical_ohlcv("OGDC", years=5))
print(ahl.fetch_historical_daily("FFC", years=5))
print(ahl.fetch_market_cap("HBL"))
print(len(ahl.fetch_symbols()))

order = ahl.create_order("HBL", "buy", 1, price=298.0, order_type="limit")
cancel = ahl.cancel_order("ORIGINAL_ORDER_NUMBER")

print(order["dry_run"], order["status"], order["request"]["url"])
print(cancel["dry_run"], cancel["status"], cancel["request"]["url"])
```

Live mode is selected once when creating the client:

```python
ahl = AHL(
    {"user": env["user"], "pass": env["pass"]},
    dry_run=False,
    allow_market_orders=False,
    allow_short_sell=False,
    max_order_value=50_000,
    max_order_quantity=500,
)

order = ahl.create_order("HBL", "buy", 1, price=298.0, order_type="limit")
```

## Trading Behavior

- `dry_run=True` is the default.
- `dry_run=False` makes `create_order()` and `cancel_order()` submit live HTTP requests for that client.
- Market orders are blocked unless `allow_market_orders=True`.
- Short selling is blocked unless `allow_short_sell=True`.
- Live buy orders can be checked against broker buying power.
- Limit prices can be checked against upper/lower cap data.
- Public request previews redact `SESSION_ID`, password, and PIN.
- Audit logging is opt-in and credentials are redacted; logs can still contain private account data.
- `max_order` is persisted under `artifacts/private/session_state.json`.

## Historical Data

- `fetch_ohlcv()` uses AHL/NxG Tick intraday chart data.
- `fetch_historical_ohlcv()` uses PSX Data Portal end-of-day data for multi-year history.
- PSX EOD rows include timestamp, open, close, and volume. High/low are not present, so the CCXT-style rows use `None` for high and low:

```python
rows = ahl.fetch_historical_ohlcv("OGDC", years=5)
# [timestamp_ms, open, high, low, close, volume]

daily = ahl.fetch_historical_daily("FFC", years=5)
# {"date": "2021-06-21", "open": ..., "high": None, "low": None, "close": ..., "volume": ...}
```

## Backtesting

The backtest engine is pandas-first and models PSX cash equity behavior: long-only positions, integer shares, next-open fills from close-based signals, configurable costs, and T+1 sell-cash settlement.

```python
from ahl_api.backtest import BacktestConfig, BacktestEngine, Order


class BuyOnce:
    def __init__(self):
        self.sent = False

    def target_weights(self, context):
        if self.sent:
            return {}
        self.sent = True
        return {"OGDC": 0.50}


engine = BacktestEngine.from_client(
    ahl,
    ["OGDC", "FFC", "HBL"],
    years=5,
    config=BacktestConfig(initial_capital=1_000_000),
)
result = engine.run(BuyOnce())

print(result.metrics)
print(result.equity_curve.tail())
print(result.trades.tail())
```

Strategies can either return target weights:

```python
def target_weights(self, context):
    return {"OGDC": 0.40, "FFC": 0.30}
```

Or explicit next-open orders:

```python
def orders(self, context):
    return [Order("OGDC", "buy", shares=100)]
```

Default trade costs use the published AHL/PSX minimum commission rule: higher of `PKR 0.03 * shares` or `0.15%` of trade value, plus sales tax on brokerage. Slippage and other account-specific charges default to zero and can be set on `BacktestConfig`.

### Parameter Sweeps

Use `run_parameter_sweep()` to test a strategy factory across a Cartesian parameter grid. Strategy parameters are passed to your factory as keyword arguments. Optional `config_grid` values override `BacktestConfig` fields per run.

```python
from ahl_api.backtest import Order, run_parameter_sweep


class BuySized:
    def __init__(self, shares):
        self.shares = shares
        self.sent = False

    def orders(self, context):
        if self.sent:
            return []
        self.sent = True
        return [Order("OGDC", "buy", shares=self.shares)]


sweep = run_parameter_sweep(
    engine.data,
    BuySized,
    {"shares": [100, 200, 300]},
    config=BacktestConfig(initial_capital=1_000_000),
    config_grid={"slippage_bps": [0, 10, 25]},
)

print(sweep.summary.sort_values("final_equity", ascending=False).head())
print(sweep.best("final_equity").parameters)
```

### Reports

Backtest and sweep results can be exported to CSV/JSON files:

```python
result.write_report("artifacts/reports/backtests/ogdc_demo", prefix="ogdc_demo")
sweep.write_report("artifacts/reports/backtests/ogdc_sweep", include_run_details=True)
```

### Strategy Runner

Run the built-in strategies, buy-and-hold benchmarks, and optional parameter sweeps from the command line:

```powershell
python tools\backtest_strategies.py --years 5 --initial-capital 100000 --top-n 5 --symbols OGDC FFC HBL MCB UBL MEBL LUCK HUBC MARI --benchmark-symbols FFC HUBC MARI --run-sweeps
```

Useful options:

```text
--symbols                  Strategy universe to rank and trade
--benchmark-symbols        Buy-and-hold symbols to compare against; fetched even if outside --symbols
--years                    Historical years to fetch
--initial-capital          Starting cash in PKR
--top-n                    Number of ranked symbols selected by momentum strategies
--min-avg-traded-value     Optional liquidity threshold in PKR daily traded value
--min-rows                 Minimum loaded rows required per symbol
--min-coverage             Drop symbols with too many missing rows versus the fullest symbol
--allow-stale-symbols      Keep symbols with stale last dates
--run-sweeps               Run parameter sweeps for the built-in strategies
--sweep-mode               quick or full sweep grid; quick is the default
--sweep-strategies         Select momentum, trend, and/or dca sweep families
--continue-on-interrupt    On Ctrl+C, mark the current strategy/sweep partial and continue
--walk-forward             Optimize on rolling train windows and test on the next window
--walk-forward-strategy    momentum, trend, or dca
--train-months             Walk-forward training window length
--test-months              Walk-forward test window length
--step-months              Months to advance each fold
--output-dir               Report folder
```

Walk-forward example for DCA:

```powershell
python tools\backtest_strategies.py --years 5 --initial-capital 100000 --top-n 5 --symbols OGDC FFC HBL MCB UBL MEBL LUCK HUBC MARI --benchmark-symbols FFC HUBC MARI --walk-forward --walk-forward-strategy dca --sweep-mode quick --train-months 36 --test-months 12 --step-months 12 --continue-on-interrupt --output-dir .\backtest\dca_walk_forward
```

### Built-In Strategies

The strategy module includes conservative PSX defaults: monthly rebalance, 120-day momentum, 20-day average traded value, top 5 symbols, and no hard liquidity threshold unless you set one.

```python
from ahl_api.backtest import BacktestEngine, run_parameter_sweep
from ahl_api.strategies import (
    LiquidityFilteredMomentumStrategy,
    MomentumTrendFilterStrategy,
    PullbackDcaStrategy,
)

symbols = ["OGDC", "FFC", "HBL", "LUCK", "MARI"]
engine = BacktestEngine.from_client(ahl, symbols, years=5)

momentum = engine.run(
    LiquidityFilteredMomentumStrategy(
        top_n=5,
        momentum_lookback=120,
        liquidity_lookback=20,
        min_avg_traded_value=0,
    )
)

trend_filtered = engine.run(
    MomentumTrendFilterStrategy(
        top_n=5,
        momentum_lookback=120,
        trend_sma=100,
    )
)

dca = engine.run(
    PullbackDcaStrategy(
        top_n=5,
        trend_sma=100,
        dip_ema=20,
        dip_thresholds=(0.03, 0.06, 0.09),
        max_position_weight=0.20,
    )
)
```

Strategy parameters can be swept like any other strategy factory:

```python
sweep = run_parameter_sweep(
    engine.data,
    PullbackDcaStrategy,
    {
        "trend_sma": [100, 200],
        "dip_ema": [20, 50],
        "dip_thresholds": [(0.03, 0.06, 0.09), (0.05, 0.10)],
    },
)
```

## Mapped Endpoints

- New order: `order?FromActivity=OrderActivityOrderCall&abc=...&SESSION_ID=...`
- Cancel order: `cancelOrder?FromActivity=CancelActivityCancelOrder&abc=...`
- Cancel/order detail: `cancelorderinfo`
- Market data: `GetSingleFeedWithMarketCap`, `singlefeed`, `ScripServlet`, `GetMarketStatus`, `GetOHLCData`, `GetMarketCap`
- Historical market data: `https://dps.psx.com.pk/timeseries/eod/{symbol}`
- Account data: `GetAccount`, `getAccountBalance`, `GetAccountBuyingPowers`, `GetAccountsAndPortfolioDetails`, `GetUserAccounts`, `GetExposureByMarket`

CLI helper:

```powershell
python .\tools\ahl_client.py snapshot
```
