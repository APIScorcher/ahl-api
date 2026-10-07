# Release validation — 0.4.0

Validated on 7 October 2026 with Python 3.14 on Windows. Evidence distinguishes deterministic offline tests from calls to external services. It does not claim that every broker feature works at every time of day.

## Offline checks

- **147 tests passed**, plus four subtests. Tests block real HTTP requests.
- **87.3% total statement coverage**. Coverage measures executed code, not the correctness of all possible broker responses.
- Lint passes for SDK, tests, and examples.
- Wheel and source distribution build; Twine metadata checks pass.
- Wheel installs in an isolated environment with only requests and its dependencies; core import and CLI work without pandas.
- Dependency consistency check passes. The installed development/research environment audit reported no known vulnerabilities at the time of validation; this is not a guarantee about future dependency releases.
- Credential fixtures are synthetic. Repository and archive contents were inspected for local account IDs, credentials, logs, datasets, and extracted resources before publication.

| Module | Statement coverage |
|---|---:|
| `__init__.py` | 81.0% |
| `backtest.py` | 89.4% |
| `cli.py` | 100.0% |
| `client.py` | 88.0% |
| `datasets.py` | 91.4% |
| `exchange.py` | 88.8% |
| `research.py` | 84.8% |
| `research_reporting.py` | 85.1% |
| `research_strategies.py` | 81.0% |
| `strategies.py` | 91.2% |

Facade/PIN tests cover configuration units, market caching, symbols, precision, quote and balance normalization, dated candles, CCXT order argument order, log filtering, per-call PIN routing/redaction, leading zeros, missing-PIN rejection before HTTP, rate limiting, and explicit unsupported parameters.

Regression coverage includes authentication failure and stale-session clearing; refreshed tokens; bounded read retries; no mutation replay; credential-redacted logging and transport errors; integer shares; finite prices; symbol/value/quantity restrictions; missing risk data; price bands; market/stop order value guards; unknown order replies; cancellation replies; date filters; portfolio reconciliation; UTC historical dates; CLI failures; lazy exports; next-open execution; settlement; fees; recurring contributions; walk-forward selection; dataset checksums; indicator strategies; and report generation.

## Live read-only checks

No real orders were submitted or cancelled during this validation. Mutations, unusual order types, and broker rejection/confirmation paths use mocked responses.

| Capability | Observation | Release status |
|---|---|---|
| Login and settings | Authentication succeeds; settings returns advertised hosts | Live transport/authentication verified |
| Portfolio and balance | Earlier market-hours query returned holdings and cash; later after-close query returned zero/empty data | Known market-hours limitation |
| Accounts, buying power, exposure | Endpoints responded; after-close values may be unusable | Protocol tested; live data constrained by market availability |
| Tickers, price caps, symbols, market cap | Responses parsed into quote/cap/catalogue fields | Live read smoke checks passed |
| Intraday candles | Endpoint returned parseable candles | Live read smoke check passed; time strings lack dates |
| Movers | Standard endpoint returned groups; feed variant returned empty body | Feed variant remains experimental |
| Order logs and lookup | After-close logs were empty; specific-order lookup was not exercised against a real order in this run | Synthetic protocol tests only for nonempty rows and order lookup |
| Account statements | Broker returned invalid/incomplete-parameters error | Experimental; SDK now raises instead of presenting an error body as success |
| PSX historical data | Public timeseries URL returned HTTP 404 | Live unavailable in this run; parser/date filtering tested offline |
| Trading | Request construction, guards, confirmations, rejections, and cancellations tested with mock HTTP | Real execution unverified |
| Backtesting and research | Synthetic datasets and offline test suite | Simulation correctness tested; not a profitability claim |

The account owner reports that the API stops returning useful account data after market close. Empty portfolios, zero balances, and empty logs must not be interpreted as real account state without a market-hours recheck. The SDK preserves the broker response and does not infer liquidation, cache stale holdings as current, or guess a trading calendar.

Remaining live work: during market hours, recheck statements with the account's valid PIN/date format, PSX history availability, the feed mover variant, and nonempty order logs. Live trading requires separate, deliberate validation by the account owner.

## CI and reproducing

GitHub Actions runs the offline suite on Windows/Linux with Python 3.10, 3.12, and 3.14. A separate job builds/checks artifacts and installs the core wheel. See the repository's Actions page for actual job results.

```sh
pip install -e ".[research,dev]"
python -m pytest --cov=ahl_api
python -m ruff check ahl_api tests examples
python -m build
python -m twine check dist/*
python -m pip check
python -m pip_audit --skip-editable
```

Packaging follows the [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/). Historical data is obtained from the [PSX Data Portal](https://dps.psx.com.pk/); the SDK license does not license its data.
