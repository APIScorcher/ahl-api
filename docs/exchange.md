# Exchange interface and trading PIN

The facade follows [CCXT's public Python method signatures](https://github.com/ccxt/ccxt/blob/master/python/ccxt/base/exchange.py), but does not inherit from CCXT or claim its full exchange feature set. It is synchronous and supports REG/ODL cash equities. Use the lower-level `AHL` client for broker-specific research calls and other market segments.

## Construction and credentials

```python
import ahl_api

exchange = ahl_api.ahl({
    "username": "YOUR_USERNAME",
    "password": "YOUR_PASSWORD",
    "pin": "YOUR_TRADING_PIN",  # string; preserves leading zeros
    "timeout": 30_000,           # milliseconds, following CCXT
    "enableRateLimit": True,
    "rateLimit": 1000,           # minimum request spacing in milliseconds
    "options": {
        "defaultMarket": "REG",
        "dryRun": True,
        "maxOrderValue": 50_000,
        "maxOrderQuantity": 500,
    },
})
```

AHL uses account login credentials, not `apiKey`/`secret`. `requiredCredentials` and `check_required_credentials()` describe account authentication. PIN is optional for public/account reads and required for live orders/cancellations. `username`, `password`, and `pin` can be assigned after construction; changing login credentials clears the session.

`options` also supports `allowedSymbols`, `blockedSymbols`, `allowMarketOrders`, `allowShortSell`, `checkPriceBands`, `checkBuyingPower`, `auditEnabled`, `auditDir`, and `sessionStatePath`. Configure options at construction; changing this dictionary later does not reconfigure the underlying client. `baseUrl`, `virtualUrl`, and `psxDataUrl` override hosts. Core `AHL(timeout=...)` remains in **seconds**; facade `timeout` is **milliseconds**. Rate limiting is configured at construction and covers all underlying requests, including login and retries. The default delay is a client pacing choice, not a published broker quota.

## PIN handling

- Default: `exchange.pin`, or `AHL({"pin": ...})`.
- Facade override: `create_order(..., params={"pin": ...})` or `cancel_order(..., params={"pin": ...})`.
- Core override: `client.create_order(..., pin=...)`, `client.cancel_order(..., pin=...)`, or `client.fetch_account_statement(..., pin=...)`.
- Statement calls fall back to the configured PIN. Statement availability and valid date/PIN requirements still need a market-hours check.
- `AHL_PIN` is supported by the CLI; `pin=` is supported in a local `.env` file.

The PIN is passed to the broker using its existing protocol. It is excluded from `describe()`, redacted in request previews and opt-in audit logs, and not persisted in order-counter state. Audit output can still contain private account data. Keep PINs as strings; the SDK does not enforce an unverified PIN length or fabricate one. Dry-run previews work without a PIN. A supplied PIN has not been tested with real execution in this release.

## Methods and broker params

| Method | Supported params |
|---|---|
| `load_markets(reload=False, params=None)` | None; caches the configured default market |
| `fetch_markets(params=None)` | `market` (REG/ODL) |
| `fetch_balance(params=None)` | None; logged-in account |
| `fetch_ticker(symbol, params=None)` | `market` |
| `fetch_tickers(symbols=None, params=None)` | `market`; explicit symbols required |
| `fetch_ohlcv(symbol, timeframe="1m", since=None, limit=None, params=None)` | Intraday: `date`, `market`; daily: `until` |
| `create_order(symbol, type, side, amount, price=None, params=None)` | `market`, `account`, `pin`, `limitPrice`, `orderNo`, `exchange` |
| `cancel_order(id, symbol=None, params=None)` | `pin`, `orderNo` |
| `fetch_order(id, symbol=None, params=None)` | None |
| `fetch_open_orders(symbol=None, since=None, limit=None, params=None)` | `market`, `account`, `pageNo`, `recordSize` |
| `fetch_closed_orders(symbol=None, since=None, limit=None, params=None)` | `market`, `account`, `pageNo`, `recordSize` |
| `cancel_all_orders(symbol=None, params=None)` | `pin`; each request uses a fresh broker counter |

Order side is `buy`/`sell`; broker-specific short-side names belong in the raw client. `orderNo` is an integer broker sequence, not a general `clientOrderId` or idempotency key. Unknown params such as `postOnly`, `reduceOnly`, and `timeInForce` raise `NotSupported`; they are never silently ignored. Order-log methods filter to the selected cash market and do not automatically paginate. `since` is Unix milliseconds and `limit` is a positive row count.

Convenience methods include `create_limit_buy_order`, `create_limit_sell_order`, `create_market_buy_order`, and `create_market_sell_order`. CamelCase aliases such as `loadMarkets`, `fetchTicker`, `fetchOHLCV`, and `createOrder` are also available.

## Unified data and limits

- `OGDC/PKR` is the unified symbol; bare `OGDC` resolves to it. Market metadata identifies `type=spot` cash trading and `stock=True`, preserving broker metadata under `info`. Derivative metadata is outside this facade.
- `fetch_balance()` includes `free`, `used`, `total`, and per-asset dictionaries. PKR total is ledger cash, not net worth or margin buying power; base-asset totals are share counts. Free/blocked/settled amounts remain `None` because the current responses do not establish them.
- Quotes expose `bid`, `ask`, `last`, volumes, high/low, change, and `info`. Timestamp is `None` when the broker supplies only a time of day.
- Orders expose `id`, `clientOrderId`, timestamps, standard fields, and `info`. Unverified fills, averages, fees, and original outstanding-order quantities remain `None`. A forwarding acknowledgement or dry-run has `status=None`; the original broker status remains in `info`. Cancellations normalize confirmed `cancelled` to `canceled`.
- OHLCV uses Unix milliseconds. Daily PSX rows retain missing high/low. Intraday calls **require** `params={"date": "YYYY-MM-DD"}` because the broker omits the date; callers must supply the correct session date. Times are interpreted in Pakistan time (UTC+5). Supported facade timeframes: `1m`, `5m`, `1d`.
- `amount_to_precision()` checks whole shares. Price tick sizes are unverified; `price_to_precision()` raises rather than guessing.
- `has` uses camelCase capability names with booleans or `emulated`. Unsupported order books, individual public/private trade history, derivative positions, and currency metadata are declared unavailable. `fetch_status()` preserves broker status in `info`; it cannot guarantee service health.

`options.dryRun` builds previews; it is **not** a broker sandbox. `set_sandbox_mode()` raises `NotSupported`. Neither the facade nor this SDK is a CCXT registry integration. Authentication retries and live trading safeguards are shared with the core client. After-close account-data limitations apply to both interfaces.

Error names include `ExchangeError` (alias of `AhlError`), `NetworkError` (alias of `TransportError`), `InvalidOrder` (alias of `RiskCheckError`), `AuthenticationError`, `BadSymbol`, and `NotSupported`. These are this SDK's exceptions, not subclasses of CCXT exceptions.
