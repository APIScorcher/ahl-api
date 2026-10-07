# AHL API

[![CI](https://github.com/APIScorcher/ahl-api/actions/workflows/ci.yml/badge.svg)](https://github.com/APIScorcher/ahl-api/actions/workflows/ci.yml)

An **unofficial Python SDK for Arif Habib Limited (AHL) NxG Tick**, with account access, PSX market data, guarded order requests, and optional strategy research.

Independent community software; not affiliated with or endorsed by AHL, PSX, or Catalyst. Uses endpoints observed in the broker client, which can change without notice. Alpha release.

## Install

Python 3.10 or newer:

```sh
pip install "git+https://github.com/APIScorcher/ahl-api.git"
```

For backtesting and research:

```sh
pip install "ahl-api[research] @ git+https://github.com/APIScorcher/ahl-api.git"
```

The package builds as a wheel and source distribution. It has not been published to PyPI; `pip install ahl-api` is not the installation command for this release.

## Retrieve your portfolio

Set `AHL_USERNAME` and `AHL_PASSWORD` in your environment, then:

```python
import os
from ahl_api import AHL

with AHL({"user": os.environ["AHL_USERNAME"], "pass": os.environ["AHL_PASSWORD"]}) as client:
    portfolio = client.fetch_portfolio()
    print(portfolio["positions"])
    print(portfolio["summary"])
    print(client.fetch_balance())
```

Each position includes symbol, quantity, average cost, last price, market value, unrealized P/L, and market. The summary includes cash, holdings value, cost basis, P/L, and net worth. Monetary account fields are in PKR. Raw broker responses remain available for troubleshooting.

The authenticated account API is known to stop returning useful data after market close. It may return zero balances and an empty portfolio even when holdings exist. **Do not interpret these responses as liquidation or use them to generate trades.** Run account validation during market hours. This SDK does not invent a trading calendar or substitute a stale snapshot for live data.

## Read-only command line

```sh
ahl portfolio
ahl balance
ahl accounts
ahl ticker OGDC
ahl market-status
```

Alternatively, keep credentials in a local `.env` file using the placeholders in `.env.example`, and run `ahl portfolio --env .env`. Keep account output private.

## Trading defaults

`dry_run=True` is the default. `create_order()` and `cancel_order()` return redacted request previews without submitting orders. Market orders and short selling require explicit enablement. Price bands and buying power are checked for live orders; unavailable risk data blocks submission.

```python
with AHL({"user": "YOUR_USERNAME", "pass": "YOUR_PASSWORD"}, max_order_value=50_000, max_order_quantity=500) as client:
    preview = client.create_order("OGDC", "buy", 1, price=300)
    assert preview["dry_run"]
```

`dry_run=False` enables actual submissions and cancellations. Mutation requests are never automatically replayed after a session error. A transport failure can leave order state uncertain: reconcile the broker logs before retrying. `submitted` means the broker acknowledged forwarding a request, not that it filled. Live trading has been validated with mocked responses, not real submissions in this release.

## Capabilities

| Area | Methods / modules |
|---|---|
| Accounts | Login, accounts, balance, buying power, portfolio, exposure, statements |
| Market data | Tickers, symbols, market caps, market status, movers, intraday candles |
| Historical data | PSX daily open, close, volume; high/low unavailable |
| Orders | Open/closed/activity logs, order lookup, request previews, guarded submit/cancel |
| Research extra | Next-open backtesting, contributions, settlement, costs, parameter sweeps, walk-forward validation, stress scenarios, data snapshots, charts |

Historical-data and some broker calls have limited live validation. See [validation evidence and limitations](docs/validation.md), [API reference](docs/api.md), and [research guide](docs/research.md).

## Development

```sh
python -m venv .venv
# Activate the virtual environment for your platform.
pip install -e ".[research,dev]"
python -m pytest --cov=ahl_api
python -m ruff check ahl_api tests examples
python -m build
python -m twine check dist/*
```

Tests use synthetic account fixtures and mocked HTTP. CI runs on Windows and Linux across Python 3.10, 3.12, and 3.14 and builds installable artifacts. The core client depends only on requests; research dependencies load lazily.

## Privacy and license

Credentials, broker logs, account snapshots, downloaded datasets, APKs, and extracted resources are excluded from Git and distributions. Audit logging is disabled by default. If enabled, credentials are redacted, but account information remains private data. See [SECURITY.md](SECURITY.md).

MIT applies to this SDK's source code. It grants no rights to broker software, market data, logos, or third-party services. Use your own authorized account and review the providers' applicable terms before use.
