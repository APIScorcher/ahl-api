"""Synchronous exchange interface for regular PSX cash equities.

No dependency on CCXT, and no claim of integration into its exchange registry.
The protocol-oriented AHL client remains available through ``client``.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from ahl_api.client import (
    AHL,
    AhlError,
    AuthenticationError,
    BrokerRejectedError,
    NotSupported,
    _redact_sensitive,
    _safe_float,
)

KARACHI = timezone(timedelta(hours=5))


class BadSymbol(AhlError, ValueError):
    pass


class ahl:
    id = "ahl"
    name = "Arif Habib Limited"
    countries = ["PK"]
    version = "0.4.0"
    timeframes = {"1m": "1", "5m": "5", "1d": "1d"}
    requiredCredentials = {"apiKey": False, "secret": False, "username": True, "password": True, "pin": False}
    has = {
        "fetchMarkets": True,
        "fetchCurrencies": False,
        "fetchBalance": True,
        "fetchTicker": True,
        "fetchTickers": True,
        "fetchOHLCV": True,
        "fetchOrder": "emulated",
        "fetchOpenOrders": True,
        "fetchClosedOrders": True,
        "fetchOrders": False,
        "fetchMyTrades": False,
        "fetchTrades": False,
        "fetchOrderBook": False,
        "fetchPositions": False,
        "fetchStatus": True,
        "createOrder": True,
        "cancelOrder": True,
        "cancelAllOrders": "emulated",
        "editOrder": False,
        "withdraw": False,
        "fetchDepositAddress": False,
        "spot": True,
        "margin": False,
        "swap": False,
        "future": False,
    }

    def __init__(self, config: dict[str, Any] | None = None):
        config = dict(config or {})
        self.options = {"defaultMarket": "REG", "dryRun": True, **config.get("options", {})}
        if self.options["defaultMarket"] not in {"REG", "ODL"}:
            raise NotSupported("The unified facade supports REG/ODL cash equities; use AHL for other broker markets.")
        self.enableRateLimit = config.get("enableRateLimit", True)
        self.rateLimit = config.get("rateLimit", 1000)
        self.timeout = config.get("timeout", 30_000)
        self.has = dict(type(self).has)
        self.requiredCredentials = dict(type(self).requiredCredentials)
        self.markets: dict[str, dict[str, Any]] | None = None
        self.markets_by_id: dict[str, list[dict[str, Any]]] = {}
        self.symbols: list[str] = []
        self.currencies: dict[str, Any] = {}
        kwargs = {
            "timeout": self.timeout / 1000,
            "dry_run": self.options["dryRun"],
            "rate_limit_ms": self.rateLimit if self.enableRateLimit else 0,
        }
        config_keys = {"baseUrl": "base_url", "virtualUrl": "virtual_url", "psxDataUrl": "psx_data_url"}
        option_keys = {
            "maxOrderValue": "max_order_value",
            "maxOrderQuantity": "max_order_quantity",
            "allowedSymbols": "allowed_symbols",
            "blockedSymbols": "blocked_symbols",
            "allowMarketOrders": "allow_market_orders",
            "allowShortSell": "allow_short_sell",
            "checkPriceBands": "check_price_bands",
            "checkBuyingPower": "check_buying_power",
            "auditEnabled": "audit_enabled",
            "auditDir": "audit_dir",
            "sessionStatePath": "session_state_path",
        }
        kwargs.update({target: config[key] for key, target in config_keys.items() if key in config})
        kwargs.update({target: self.options[key] for key, target in option_keys.items() if key in self.options})
        for key in ("allowed_symbols", "blocked_symbols"):
            if key in kwargs:
                kwargs[key] = [str(symbol).split("/")[0] for symbol in kwargs[key]]
        self.client = AHL(config, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.client.close()

    @property
    def username(self):
        return self.client.username

    @username.setter
    def username(self, value):
        self.client.username = value
        self.client.session = None

    @property
    def password(self):
        return self.client.password

    @password.setter
    def password(self, value):
        self.client.password = value
        self.client.session = None

    @property
    def pin(self):
        return self.client.pin

    @pin.setter
    def pin(self, value):
        self.client.pin = "" if value is None else str(value)

    def describe(self):
        return {
            "id": self.id,
            "name": self.name,
            "countries": list(self.countries),
            "version": self.version,
            "has": dict(self.has),
            "timeframes": dict(self.timeframes),
            "requiredCredentials": dict(self.requiredCredentials),
            "rateLimit": self.rateLimit,
            "enableRateLimit": self.enableRateLimit,
            "timeout": self.timeout,
            "options": {"defaultMarket": self.options["defaultMarket"], "dryRun": self.client.dry_run},
        }

    def check_required_credentials(self, error=True):
        valid = bool(self.username and self.password)
        if not valid and error:
            raise AuthenticationError("AHL requires username and password")
        return valid

    def fetch_markets(self, params=None):
        params = self._params(params, {"market"})
        broker_market = self._broker_market(params)
        markets = []
        for raw in self.client.fetch_symbols():
            if raw.get("market") != broker_market or not raw.get("symbol"):
                continue
            base = str(raw["symbol"]).upper()
            markets.append(
                {
                    "id": base,
                    "symbol": base + "/PKR",
                    "base": base,
                    "quote": "PKR",
                    "baseId": base,
                    "quoteId": "PKR",
                    "type": "spot",
                    "spot": True,
                    "stock": True,
                    "margin": False,
                    "swap": False,
                    "future": False,
                    "option": False,
                    "contract": False,
                    "active": None,
                    "precision": {"amount": 1, "price": None},
                    "limits": {
                        "amount": {"min": 1, "max": None},
                        "price": {"min": None, "max": None},
                        "cost": {"min": None, "max": None},
                    },
                    "info": {**raw, "assetType": "equity"},
                }
            )
        return markets

    def load_markets(self, reload=False, params=None):
        self._params(params, set())
        if self.markets is None or reload:
            # A cache contains one broker market. Per-call routing belongs in quote/order params.
            markets = self.fetch_markets()
            self.markets = {market["symbol"]: market for market in markets}
            self.markets_by_id = {market["id"]: [market] for market in markets}
            self.symbols = sorted(self.markets)
        return self.markets

    def market(self, symbol):
        symbol = self._symbol(symbol)
        markets = self.load_markets()
        if symbol not in markets:
            raise BadSymbol(f"Unknown AHL symbol: {symbol}")
        return markets[symbol]

    def amount_to_precision(self, symbol, amount):
        self.market(symbol)
        try:
            value = Decimal(str(amount))
            if isinstance(amount, bool) or not value.is_finite() or value <= 0 or value != value.to_integral_value():
                raise InvalidOperation
        except (InvalidOperation, ValueError):
            raise ValueError("Cash-equity amounts must be positive whole shares") from None
        return str(int(value))

    def fetch_balance(self, params=None):
        self._params(params, set())
        balance = self.client.fetch_balance()
        if balance.get("ok") is not True or balance.get("cash") is None:
            raise BrokerRejectedError("Broker did not confirm the cash balance")
        portfolio = self.client.fetch_portfolio()
        totals = {"PKR": balance["cash"]}
        for position in portfolio["positions"]:
            if position["market"] in {"REG", "ODL"}:
                code = position["symbol"]
                if position["quantity"] is None:
                    raise BrokerRejectedError("Broker returned an unrecognized holding quantity")
                totals[code] = totals.get(code, 0) + position["quantity"]
        # Ledger balance and share holdings do not tell us settled/free/blocked amounts.
        result = {
            "free": {code: None for code in totals},
            "used": {code: None for code in totals},
            "total": totals,
            "timestamp": None,
            "datetime": None,
            "info": _redact_sensitive({"balance": balance, "portfolio": portfolio}),
        }
        result.update({code: {"free": None, "used": None, "total": total} for code, total in totals.items()})
        return result

    def fetch_portfolio(self):
        return self.client.fetch_portfolio()

    def fetch_accounts(self, params=None):
        self._params(params, set())
        return [{"id": row.get("id"), "type": None, "code": "PKR", "info": row} for row in self.client.fetch_accounts()]

    def fetch_ticker(self, symbol, params=None):
        params = self._params(params, {"market"})
        market = self.market(symbol)
        raw = self.client.fetch_ticker(market["id"], market=self._broker_market(params))
        return self._ticker(raw, market["symbol"])

    def fetch_tickers(self, symbols=None, params=None):
        params = self._params(params, {"market"})
        if symbols is None:
            raise NotSupported("Specify symbols explicitly; fetching the entire catalogue is not supported.")
        if not symbols:
            return {}
        markets = [self.market(symbol) for symbol in symbols]
        raw = self.client.fetch_tickers([market["id"] for market in markets], market=self._broker_market(params))
        return {
            market["symbol"]: self._ticker(raw[market["id"]], market["symbol"])
            for market in markets
            if market["id"] in raw
        }

    def fetch_ohlcv(self, symbol, timeframe="1m", since=None, limit=None, params=None):
        params = self._params(params, {"market", "date", "until"})
        self._bounds(since, limit)
        if timeframe not in self.timeframes:
            raise NotSupported(f"Unsupported timeframe: {timeframe}")
        market = self.market(symbol)
        if timeframe == "1d":
            if "date" in params or "market" in params:
                raise ValueError("Daily PSX history accepts until, not broker date/market params")
            rows = self.client.fetch_historical_ohlcv(market["id"], since=since, until=params.get("until"))
        else:
            if "date" not in params:
                raise ValueError("Intraday candles require params={'date': 'YYYY-MM-DD'}; the broker supplies no date")
            if "until" in params:
                raise ValueError("until is supported only for daily candles")
            day = datetime.strptime(params["date"], "%Y-%m-%d").date()
            raw = self.client.fetch_ohlcv(
                market["id"], market=self._broker_market(params), interval=self.timeframes[timeframe]
            )
            rows = []
            for row in raw:
                instant = datetime.fromisoformat(f"{day.isoformat()}T{row[0]}").replace(tzinfo=KARACHI)
                rows.append([int(instant.timestamp() * 1000), *row[1:]])
            rows.sort(key=lambda row: row[0])
        rows = [row for row in rows if since is None or row[0] >= since]
        return rows[:limit] if limit is not None else rows

    def create_order(self, symbol, type, side, amount, price=None, params=None):
        if side not in {"buy", "sell"}:
            raise ValueError("Unified order side must be buy or sell")
        params = self._params(params, {"market", "account", "pin", "limitPrice", "orderNo", "exchange"})
        self._require_pin(params)
        market = self.market(symbol)
        kwargs = self._order_params(params)
        raw = self.client.create_order(market["id"], side, amount, price, order_type=type, **kwargs)
        result = self._order(raw, market["symbol"])
        result["dry_run"] = raw["dry_run"]
        return result

    def create_limit_buy_order(self, symbol, amount, price, params=None):
        return self.create_order(symbol, "limit", "buy", amount, price, params)

    def create_limit_sell_order(self, symbol, amount, price, params=None):
        return self.create_order(symbol, "limit", "sell", amount, price, params)

    def create_market_buy_order(self, symbol, amount, params=None):
        return self.create_order(symbol, "market", "buy", amount, None, params)

    def create_market_sell_order(self, symbol, amount, params=None):
        return self.create_order(symbol, "market", "sell", amount, None, params)

    def cancel_order(self, id, symbol=None, params=None):
        params = self._params(params, {"pin", "orderNo"})
        self._require_pin(params)
        unified = self.market(symbol)["symbol"] if symbol is not None else None
        raw = self.client.cancel_order(str(id), pin=params.get("pin", ""), order_no=params.get("orderNo"))
        result = self._order(raw, unified)
        result["dry_run"] = raw["dry_run"]
        return result

    def fetch_order(self, id, symbol=None, params=None):
        self._params(params, set())
        raw = self.client.fetch_order(str(id))
        if raw.get("market") not in (None, "REG", "ODL"):
            raise NotSupported("This broker order is outside the facade's cash-equity markets")
        result = self._order(raw)
        if symbol is not None and result["symbol"] != self._symbol(symbol):
            raise BadSymbol("Broker order does not match the requested symbol")
        return result

    def fetch_open_orders(self, symbol=None, since=None, limit=None, params=None):
        return self._orders("outstanding", symbol, since, limit, params)

    def fetch_closed_orders(self, symbol=None, since=None, limit=None, params=None):
        return self._orders("trade", symbol, since, limit, params)

    def cancel_all_orders(self, symbol=None, params=None):
        params = self._params(params, {"pin", "orderNo"})
        if "orderNo" in params:
            raise ValueError("cancel_all_orders uses a fresh broker counter for each cancellation")
        self._require_pin(params)
        return [self.cancel_order(order["id"], None, params) for order in self.fetch_open_orders(symbol) if order["id"]]

    def fetch_status(self, params=None):
        self._params(params, set())
        raw = self.client.fetch_market_status()
        return {"status": None, "updated": None, "eta": None, "url": None, "info": raw}

    def fetch_order_book(self, symbol, limit=None, params=None):
        raise NotSupported("AHL order-book depth has not been implemented")

    def fetch_trades(self, symbol, since=None, limit=None, params=None):
        raise NotSupported("Public trade history has not been implemented")

    def fetch_my_trades(self, symbol=None, since=None, limit=None, params=None):
        raise NotSupported("Broker order logs do not provide verified individual execution IDs")

    def set_sandbox_mode(self, enabled):
        raise NotSupported("AHL has no verified broker sandbox; options.dryRun creates request previews only")

    def fetch_positions(self, symbols=None, params=None):
        raise NotSupported(
            "Derivative positions are not implemented; use fetch_balance or fetch_portfolio for cash holdings"
        )

    def fetch_orders(self, symbol=None, since=None, limit=None, params=None):
        raise NotSupported("Use fetch_open_orders and fetch_closed_orders; a complete combined history is unavailable")

    def fetch_currencies(self, params=None):
        raise NotSupported("Currency metadata is unavailable")

    def price_to_precision(self, symbol, price):
        raise NotSupported("The broker catalogue does not provide verified price tick sizes")

    def _orders(self, logname, symbol, since, limit, params):
        params = self._params(params, {"market", "account", "pageNo", "recordSize"})
        self._bounds(since, limit)
        raw = self.client.fetch_order_logs(
            logname,
            page_no=params.get("pageNo", 1),
            record_size=params.get("recordSize", 15000),
            account=params.get("account"),
        )
        orders = [self._order(order) for order in raw]
        if symbol is not None:
            orders = [order for order in orders if order["symbol"] == self._symbol(symbol)]
        broker_market = self._broker_market(params)
        orders = [order for order in orders if order["info"].get("market") == broker_market]
        if since is not None:
            if any(order["timestamp"] is None for order in orders):
                raise ValueError("Cannot apply since to undated broker logs")
            orders = [order for order in orders if order["timestamp"] >= since]
        return orders[:limit] if limit is not None else orders

    def _require_pin(self, params):
        if not self.client.dry_run and not self.client._resolve_pin(params.get("pin")):
            raise AuthenticationError("A trading PIN is required for live orders/cancellations")

    def _order_params(self, params):
        mapping = {
            "account": "account",
            "pin": "pin",
            "limitPrice": "limit_price",
            "orderNo": "order_no",
            "exchange": "exchange",
        }
        return {
            "market": self._broker_market(params),
            **{target: params[key] for key, target in mapping.items() if key in params},
        }

    def _broker_market(self, params):
        market = str(params.get("market", self.options["defaultMarket"])).upper()
        if market not in {"REG", "ODL"}:
            raise NotSupported("The unified facade supports REG/ODL cash equities only")
        return market

    @staticmethod
    def _params(params, allowed):
        params = dict(params or {})
        unsupported = set(params).difference(allowed)
        if unsupported:
            raise NotSupported("Unsupported params: " + ", ".join(sorted(unsupported)))
        return params

    @staticmethod
    def _symbol(symbol):
        parts = str(symbol).upper().strip().split("/")
        if len(parts) > 2 or (len(parts) == 2 and parts[1] != "PKR") or not parts[0]:
            raise BadSymbol("Use SYMBOL/PKR or a bare broker symbol")
        return parts[0] + "/PKR"

    @staticmethod
    def _bounds(since, limit):
        if since is not None and (isinstance(since, bool) or not isinstance(since, int) or since < 0):
            raise ValueError("since must be a nonnegative Unix timestamp in milliseconds")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
            raise ValueError("limit must be a positive integer")

    @staticmethod
    def _ticker(raw, symbol):
        last, change = _safe_float(raw.get("last")), _safe_float(raw.get("change"))
        previous = last - change if last is not None and change is not None else None
        return {
            "symbol": symbol,
            "timestamp": None,
            "datetime": None,
            "high": _safe_float(raw.get("high")),
            "low": _safe_float(raw.get("low")),
            "bid": _safe_float(raw.get("buy")),
            "bidVolume": _safe_float(raw.get("buy_volume")),
            "ask": _safe_float(raw.get("sell")),
            "askVolume": _safe_float(raw.get("sell_volume")),
            "vwap": None,
            "open": None,
            "close": last,
            "last": last,
            "previousClose": _safe_float(raw.get("close")),
            "change": change,
            "percentage": change / previous * 100 if previous else None,
            "average": _safe_float(raw.get("average")),
            "baseVolume": _safe_float(raw.get("volume")),
            "quoteVolume": None,
            "info": _redact_sensitive(raw),
        }

    @staticmethod
    def _order(raw, symbol=None):
        timestamp = None
        if raw.get("datetime"):
            try:
                instant = datetime.strptime(raw["datetime"], "%b %d, %Y %H:%M:%S").replace(tzinfo=KARACHI)
                timestamp = int(instant.timestamp() * 1000)
            except ValueError:
                pass
        statuses = {
            "open": "open",
            "partially_filled": "open",
            "closed": "closed",
            "cancelled": "canceled",
            "rejected": "rejected",
        }
        info = _redact_sensitive(raw)
        if symbol is None and raw.get("symbol"):
            symbol = ahl._symbol(raw["symbol"])
        return {
            "id": str(raw["id"]) if raw.get("id") is not None else None,
            "clientOrderId": str(raw["client_order_id"]) if raw.get("client_order_id") is not None else None,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp / 1000, timezone.utc).isoformat().replace("+00:00", "Z")
            if timestamp is not None
            else None,
            "lastTradeTimestamp": None,
            "status": statuses.get(raw.get("status")),
            "symbol": symbol,
            "type": raw.get("type"),
            "timeInForce": None,
            "side": raw.get("side"),
            "price": raw.get("price"),
            "average": None,
            "amount": None if raw.get("logname") == "outstanding" else raw.get("amount"),
            "filled": raw.get("filled"),
            "remaining": raw.get("remaining"),
            "cost": raw.get("value"),
            "fee": None,
            "fees": None,
            "trades": None,
            "info": info,
        }


for _snake, _camel in {
    "load_markets": "loadMarkets",
    "fetch_markets": "fetchMarkets",
    "fetch_balance": "fetchBalance",
    "fetch_ticker": "fetchTicker",
    "fetch_tickers": "fetchTickers",
    "fetch_ohlcv": "fetchOHLCV",
    "create_order": "createOrder",
    "cancel_order": "cancelOrder",
    "cancel_all_orders": "cancelAllOrders",
    "fetch_order": "fetchOrder",
    "fetch_open_orders": "fetchOpenOrders",
    "fetch_closed_orders": "fetchClosedOrders",
    "fetch_accounts": "fetchAccounts",
    "fetch_status": "fetchStatus",
    "amount_to_precision": "amountToPrecision",
    "create_limit_buy_order": "createLimitBuyOrder",
    "create_limit_sell_order": "createLimitSellOrder",
    "create_market_buy_order": "createMarketBuyOrder",
    "create_market_sell_order": "createMarketSellOrder",
    "check_required_credentials": "checkRequiredCredentials",
    "set_sandbox_mode": "setSandboxMode",
}.items():
    setattr(ahl, _camel, getattr(ahl, _snake))

Exchange = ahl
