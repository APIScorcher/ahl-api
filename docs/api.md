# API reference

## Client lifecycle

`AHL(config=None, *, base_url=..., timeout=30, dry_run=True, audit_enabled=False, ...)` accepts `user`/`pass` or `username`/`password`. Optional trading PIN keys: `pin`, `PIN`, `trade_pin`, or `TRADING_PIN`. Use a context manager or call `close()` to release HTTP resources. A client is synchronous and intended for one account and one execution thread. Separate processes must not share an order-counter file.

`login()` authenticates, checks required fields, and persists an account-scoped order counter locally. `ensure_session()` logs in on first authenticated use. A failed login clears the previous session. `read_dotenv(Path(...))` reads simple key/value files; it is not a complete dotenv interpolation parser.

`settings()` returns the broker's advertised hosts. Explicit `base_url` overrides support broker routing changes. Settings are not silently followed to arbitrary hosts. The virtual intraday-data host uses HTTP; credentials are not sent to it.

## Account methods

| Method | Return |
|---|---|
| `fetch_accounts()` | List of account IDs and display names |
| `fetch_balance()` | `ok`, `cash`, `raw` |
| `fetch_buying_power(account=None)` | `regular`, `future`, `raw` |
| `fetch_portfolio()` | `positions`, `summary`, legacy `values`, `raw` |
| `fetch_exposure()` | Parsed nested broker JSON and raw response |
| `fetch_exposure_by_market(account=None)` | Fields grouped by market and raw response |
| `fetch_account_statement(date_start, date_end, account=None, pin="")` | Parsed response when JSON, raw response otherwise; broker error bodies raise |

Account methods can return empty/zero data after market close. Empty data has no guaranteed semantic meaning. Statements are experimental: PIN/date requirements are broker-specific and have not been successfully verified live.

## Market methods

- `fetch_market_status()` preserves the broker status string; this is not a trading calendar.
- `fetch_ticker(symbol, market="REG", with_market_cap=True)` returns quote fields and optional upper/lower price caps.
- `fetch_tickers(symbols, market="REG")` returns quotes keyed by symbol.
- `fetch_symbols(reload=True, is_shariah=False, since="")` returns the broker symbol catalogue.
- `fetch_market_cap(symbol)` returns shares and market capitalization.
- `fetch_ohlcv(symbol, market="REG", interval="1", check="symbol")` returns broker intraday rows `[time_string, open, high, low, close, None]`. Time strings lack a date; they are not Unix timestamps.
- `fetch_historical_ohlcv(symbol, since=None, until=None, years=None)` returns PSX EOD rows `[timestamp_ms, open, None, None, close, volume]` with inclusive UTC date bounds. `fetch_historical_daily()` returns dictionaries. Availability, corporate-action adjustments, and historical coverage depend on PSX.
- `fetch_movers(identifier="KSE100", count=20, mover_type=None, with_feed=False)` preserves mover groups and raw output.

## Orders and risk checks

`fetch_open_orders(symbol=None, market=None)`, `fetch_closed_orders(symbol=None, since=None, until=None)`, `fetch_activity_logs(symbol=None, market=None)`, and `fetch_order_logs(logname, page_no=1, record_size=15000)` read broker logs. Closed-order date filters are applied locally to recognized timestamps. No automatic pagination is performed. Empty logs after close cannot prove that no orders exist.

`fetch_order(id)` searches outstanding, trade, and activity logs, then uses `fetch_order_cancel_info(id)` as a best-effort fallback. This is not an exchange execution-status guarantee.

`build_order_request(...)` and `build_cancel_order_request(...)` return credential-redacted protocol previews. `create_order(symbol, side, amount, price=None, order_type="limit", ...)` applies risk checks. `cancel_order(id, ...)` and `cancel_all_orders(symbol=None)` use the client's dry-run setting. Preview builders do not perform account risk checks or submit requests.

Constructor guards: `max_order_value`, `max_order_quantity`, `allowed_symbols`, `blocked_symbols`, `allow_market_orders=False`, `allow_short_sell=False`, `check_price_bands=True`, `check_buying_power=True`. Quantities must be positive integers and prices finite. Market-order value checks use an available quote and cannot guarantee fill cost. Buying-power checks support REG/FUT, exclude fees, and are a preflight estimate. Stop-loss orders require positive price and limit price. Concurrent trading clients and special broker order types are outside verified live coverage.

Orders report `dry_run`, `submitted`, `closed`, `rejected`, or `unknown`; cancellations report `dry_run`, `pending_cancel`, `cancelled`, `rejected`, or `unknown`. Unknown responses have `ok=False`. Persisted counters do not guarantee exactly-once execution.

## Errors

Catch `AhlError` for SDK errors, or its subclasses: `AuthenticationError`, `SessionExpiredError`, `TransportError`, `BrokerRejectedError`, and `RiskCheckError`. Network/HTTP messages omit credential-bearing URLs. Invalid argument names/values can raise `ValueError`. Read-only expired sessions are refreshed once with the new session ID. Submit/cancel requests are never replayed automatically.
