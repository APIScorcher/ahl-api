from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import requests


DEFAULT_BASE_URL = "https://nxgtick2.ahletrade.com/pero/"
DEFAULT_VIRTUAL_URL = "http://virtualtrading.ahletrade.com/pero/"
DEFAULT_PSX_DATA_URL = "https://dps.psx.com.pk/"
APP_VERSION = "11"
CLIENT_VERSION = "1.0.3"
VERSION_NO = "v1.1"
DEVICE_USER_AGENT = "1.0.3"
DEFAULT_AUDIT_DIR = Path("artifacts/private/audit")
DEFAULT_SESSION_STATE_PATH = Path("artifacts/private/session_state.json")
SENSITIVE_RESPONSE_KEYS = {
    "password",
    "session_id",
    "feedsessionid",
    "feed_session_id",
    "email",
    "token",
    "pin",
    "pincode",
}
SENSITIVE_QUERY_KEYS = {"SESSION_ID", "password", "pin", "pincode", "token"}


class AhlError(RuntimeError):
    pass


class AuthenticationError(AhlError):
    pass


class SessionExpiredError(AuthenticationError):
    pass


class BrokerRejectedError(AhlError):
    pass


class RiskCheckError(AhlError):
    pass


class NotSupported(AhlError):
    pass


class TransportError(AhlError):
    """HTTP or network failure with credentials removed from the message."""


@dataclass
class AhlSession:
    user_id: str
    user_code: str
    account: str
    feed_session_id: str = field(repr=False)
    max_order: int
    raw: dict[str, Any] = field(repr=False)


def read_dotenv(path: Path = Path(".env")) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _clean_base_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/"


class AHL:
    """Small CCXT-style interface for AHL NxG Tick.

    The client defaults to dry-run trading. Construct it with ``dry_run=False``
    to submit live order/cancel requests with the mapped Android HTTP endpoints.
    """

    id = "ahl"
    name = "Arif Habib Limited NxG Tick"
    version = "0.3.0"
    has = {
        "fetch_accounts": True,
        "fetch_balance": True,
        "fetch_portfolio": True,
        "fetch_exposure": True,
        "fetch_market_status": True,
        "fetch_ticker": True,
        "fetch_tickers": True,
        "fetch_symbols": True,
        "fetch_ohlcv": True,
        "fetch_historical_ohlcv": True,
        "fetch_open_orders": "best_effort",
        "fetch_closed_orders": "best_effort",
        "fetch_order": "best_effort",
        "create_order": True,
        "cancel_order": True,
    }

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        virtual_url: str = DEFAULT_VIRTUAL_URL,
        psx_data_url: str = DEFAULT_PSX_DATA_URL,
        timeout: int = 30,
        dry_run: bool = True,
        allow_market_orders: bool = False,
        allow_short_sell: bool = False,
        max_order_value: float | None = None,
        max_order_quantity: int | None = None,
        allowed_symbols: set[str] | list[str] | tuple[str, ...] | None = None,
        blocked_symbols: set[str] | list[str] | tuple[str, ...] | None = None,
        check_price_bands: bool = True,
        check_buying_power: bool = True,
        audit_dir: str | Path = DEFAULT_AUDIT_DIR,
        session_state_path: str | Path = DEFAULT_SESSION_STATE_PATH,
        audit_enabled: bool = False,
    ) -> None:
        config = config or {}
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if max_order_value is not None and (not math.isfinite(max_order_value) or max_order_value <= 0):
            raise ValueError("max_order_value must be positive and finite")
        if max_order_quantity is not None:
            _order_quantity(max_order_quantity)
        self.base_url = _clean_base_url(base_url)
        self.virtual_url = _clean_base_url(virtual_url)
        self.psx_data_url = _clean_base_url(psx_data_url)
        self.timeout = timeout
        self.dry_run = dry_run
        self.allow_market_orders = allow_market_orders
        self.allow_short_sell = allow_short_sell
        self.max_order_value = max_order_value
        self.max_order_quantity = max_order_quantity
        self.allowed_symbols = _normalize_symbol_set(allowed_symbols)
        self.blocked_symbols = _normalize_symbol_set(blocked_symbols)
        self.check_price_bands = check_price_bands
        self.check_buying_power = check_buying_power
        self.audit_enabled = audit_enabled
        self.audit_dir = Path(audit_dir)
        self.session_state_path = Path(session_state_path)
        self.username = config.get("username") or config.get("user")
        self.password = config.get("password") or config.get("pass")
        self.pin = config.get("pin") or config.get("PIN") or config.get("trade_pin") or config.get("TRADING_PIN") or ""
        self.session: AhlSession | None = None
        self.http = requests.Session()
        self.http.headers.update(
            {
                "User-Agent": "okhttp/4.x",
                "Accept": "*/*",
                "Accept-Encoding": "identity",
            }
        )

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> AHL:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "urls": {"api": self.base_url, "virtual": self.virtual_url, "psx_data": self.psx_data_url},
            "dry_run": self.dry_run,
            "has": dict(self.has),
        }

    def _get_text(self, endpoint: str, params: dict[str, Any] | None = None) -> str:
        return self._request_text(endpoint=endpoint, params=params)

    def _request_text(
        self,
        *,
        endpoint: str | None = None,
        params: dict[str, Any] | None = None,
        raw_url: str | None = None,
        base_url: str | None = None,
        auth: bool = True,
        audit_action: str | None = None,
        retry_session: bool = True,
    ) -> str:
        url = raw_url or f"{base_url or self.base_url}{(endpoint or '').lstrip('/')}"
        request_record = {"url": url, "params": params or {}, "endpoint": endpoint}
        try:
            response = self.http.get(
                url,
                params=None if raw_url else params or {},
                timeout=self.timeout,
            )
            response.raise_for_status()
            text = response.text
            if auth and self._looks_session_expired(text):
                if raw_url or not retry_session:
                    raise SessionExpiredError("Session expired; request was not replayed.")
                self.login()
                params = dict(params or {})
                if "SESSION_ID" in params:
                    params["SESSION_ID"] = self.ensure_session().feed_session_id
                return self._request_text(
                    endpoint=endpoint,
                    params=params,
                    raw_url=raw_url,
                    base_url=base_url,
                    auth=auth,
                    audit_action=audit_action,
                    retry_session=False,
                )
            if text.strip().lower().startswith(("error", "invalid/incomplete", "access denied")):
                raise BrokerRejectedError("Broker rejected the request; verify parameters and market availability.")
            if audit_action:
                self._audit(audit_action, request=request_record, response=text)
            return text
        except requests.RequestException as exc:
            error = TransportError(f"{type(exc).__name__} while requesting {endpoint or 'broker operation'}")
            if audit_action:
                self._audit(audit_action, request=request_record, exception=error)
            raise error from None
        except Exception as exc:
            if audit_action:
                self._audit(audit_action, request=request_record, exception=exc)
            raise

    def settings(self) -> dict[str, Any]:
        body = self._request_text(
            endpoint="CheckSettingsServlet",
            params={"AppVersion": APP_VERSION, "OSType": "android"},
            auth=False,
        )
        fields = body.split("|")
        return {
            "raw": body,
            "field_count": len(fields),
            "timestamp": fields[0] if len(fields) > 0 else "",
            "base_url": fields[1] if len(fields) > 1 else "",
            "virtual_trading_url": fields[16] if len(fields) > 16 else "",
            "new_server_url": fields[54] if len(fields) > 54 else "",
            "trade_socket_flag": fields[58] if len(fields) > 58 else "",
            "socket_url": fields[59] if len(fields) > 59 else "",
            "trade_server_ip": fields[60] if len(fields) > 60 else "",
            "trade_server_port": fields[61] if len(fields) > 61 else "",
        }

    def login(self, username: str | None = None, password: str | None = None) -> AhlSession:
        username = username or self.username
        password = password or self.password
        if not username or not password:
            raise AuthenticationError("Missing username/password. Pass config or call login(username, password).")

        self.session = None
        self.username, self.password = username, password

        body = self._request_text(
            endpoint="TickLoginServlet",
            params={
                "FromActivity": "LoginActivityLoginCall",
                "userid": username,
                "password": password,
                "versionNo": VERSION_NO,
                "deviceuseragent": DEVICE_USER_AGENT,
                "OSType": "NXGandroid",
                "ClientVersion": CLIENT_VERSION,
                "isReLogin": "false",
            },
            auth=False,
            audit_action="login",
        )
        data = _loads_json(body)
        if not isinstance(data, dict):
            raise AuthenticationError("Login returned non-JSON response")
        if data.get("identifier") not in (None, 0, "0"):
            raise AuthenticationError("Broker rejected login")

        required = ["userID", "userCode", "account", "FeedSessionID"]
        missing = [field for field in required if not data.get(field)]
        if missing:
            raise AuthenticationError(f"Login response missing fields: {missing}")

        account = str(data["account"])
        broker_max_order = int(data.get("maxOrder") or 0)
        max_order = max(broker_max_order, self._load_max_order(account))
        self.session = AhlSession(
            user_id=str(data["userID"]),
            user_code=str(data["userCode"]),
            account=account,
            feed_session_id=str(data["FeedSessionID"]),
            max_order=max_order,
            raw=_redact_sensitive(data),
        )
        self._persist_max_order(max_order)
        return self.session

    def login_from_env(self, env_path: Path = Path(".env")) -> AhlSession:
        env = read_dotenv(env_path)
        if "user" not in env or "pass" not in env:
            raise AuthenticationError("Expected .env keys: user and pass")
        return self.login(env["user"], env["pass"])

    def ensure_session(self) -> AhlSession:
        if self.session is None:
            return self.login()
        return self.session

    def fetch_accounts(self) -> list[dict[str, str]]:
        session = self.ensure_session()
        text = self._request_text(
            endpoint="GetAccount",
            params={
                "FromActivity": "LoginActivityLoginCall",
                "userid": session.user_id,
                "SESSION_ID": session.feed_session_id,
                "account": "android",
            },
            audit_action="fetch_accounts",
        )
        try:
            outer = requests.models.complexjson.loads(text)
            accounts = []
            for item in outer:
                account_no, title = requests.models.complexjson.loads(item)
                accounts.append({"id": account_no, "account": account_no, "name": title})
            return accounts
        except (ValueError, TypeError):
            return [{"raw": text}]

    def fetch_balance(self) -> dict[str, Any]:
        session = self.ensure_session()
        text = self._request_text(
            endpoint="getAccountBalance",
            params={
                "account": session.account,
                "SESSION_ID": session.feed_session_id,
                "userid": session.user_id,
            },
            audit_action="fetch_balance",
        )
        parts = text.strip().split("|")
        return {
            "ok": parts[0].lower() == "true" if parts and parts[0] else None,
            "cash": _safe_float(parts[1]) if len(parts) > 1 else None,
            "raw": text,
        }

    def fetch_buying_power(self, account: str | None = None) -> dict[str, Any]:
        session = self.ensure_session()
        text = self._request_text(
            endpoint="GetAccountBuyingPowers",
            params={"account": account or session.account},
            audit_action="fetch_buying_power",
        )
        parts = text.strip().split("|")
        return {
            "regular": _safe_float(parts[0]) if len(parts) > 0 else None,
            "future": _safe_float(parts[1]) if len(parts) > 1 else None,
            "raw": text,
        }

    def fetch_portfolio(self) -> dict[str, Any]:
        session = self.ensure_session()
        text = self._request_text(
            endpoint="GetAccountsAndPortfolioDetails",
            params={
                "FromActivity": "PortfolioFragment",
                "SESSION_ID": session.feed_session_id,
                "userid": session.user_id,
                "acc": session.account,
            },
            audit_action="fetch_portfolio",
        )
        values = [part for part in text.strip().split(";") if part != ""]
        return {"values": values, **_parse_portfolio(text), "raw": text}

    def fetch_exposure(self) -> dict[str, Any]:
        session = self.ensure_session()
        text = self._request_text(
            endpoint="GetUserAccounts",
            params={
                "FromActivity": "ExposureFragment",
                "SESSION_ID": session.feed_session_id,
                "userid": session.user_id,
                "account": session.account,
                "date": "",
            },
            audit_action="fetch_exposure",
        )
        return {"parsed": _loads_json(text), "raw": text}

    def fetch_exposure_by_market(self, account: str | None = None) -> dict[str, Any]:
        session = self.ensure_session()
        text = self._request_text(
            endpoint="GetExposureByMarket",
            params={"userid": session.user_id, "account": account or session.account},
            audit_action="fetch_exposure_by_market",
        )
        return {"markets": _parse_exposure_by_market(text), "raw": text}

    def fetch_account_statement(
        self,
        date_start: str,
        date_end: str,
        *,
        account: str | None = None,
        pin: str = "",
    ) -> dict[str, Any]:
        session = self.ensure_session()
        text = self._request_text(
            endpoint="GetAccountStatement",
            params={
                "userid": session.user_id,
                "account": account or session.account,
                "SESSION_ID": session.feed_session_id,
                "date_start": date_start,
                "date_end": date_end,
                "pincode": self._resolve_pin(pin),
            },
            audit_action="fetch_account_statement",
        )
        parsed = _loads_json(text)
        return {"parsed": parsed, "raw": text}

    def fetch_market_status(self) -> dict[str, Any]:
        text = self._request_text(endpoint="GetMarketStatus", audit_action="fetch_market_status")
        return {"status": text.strip(), "raw": text}

    def fetch_ticker(self, symbol: str, market: str = "REG", *, with_market_cap: bool = True) -> dict[str, Any]:
        session = self.ensure_session()
        endpoint = "GetSingleFeedWithMarketCap" if with_market_cap else "singlefeed"
        text = self._request_text(
            endpoint=endpoint,
            params={
                "FromActivity": "OrderActivitySingleFeed",
                "SESSION_ID": session.feed_session_id,
                "symbollist": f"{symbol.upper()}:{market.upper()}|",
                "abc": _javaish_date(),
            },
            audit_action="fetch_ticker",
        )
        data = _loads_json(text)
        if not isinstance(data, dict):
            return {"symbol": symbol.upper(), "market": market.upper(), "raw": text}
        entries = _parse_feed_entries(str(data.get("feedString", "")))
        ticker = entries[0] if entries else {"symbol": symbol.upper(), "market": market.upper()}
        ticker["cap"] = data.get("capObject")
        ticker["raw"] = text
        return ticker

    def fetch_tickers(self, symbols: list[str] | tuple[str, ...], market: str = "REG") -> dict[str, dict[str, Any]]:
        session = self.ensure_session()
        symbol_list = "".join(f"{symbol.upper()}:{market.upper()}|" for symbol in symbols)
        text = self._request_text(
            endpoint="singlefeed",
            params={
                "FromActivity": "OrderActivitySingleFeed",
                "SESSION_ID": session.feed_session_id,
                "symbollist": symbol_list,
                "abc": _javaish_date(),
            },
            audit_action="fetch_tickers",
        )
        data = _loads_json(text)
        if not isinstance(data, dict):
            return {symbol.upper(): {"symbol": symbol.upper(), "market": market.upper(), "raw": text} for symbol in symbols}
        entries = _parse_feed_entries(str(data.get("feedString", "")))
        return {str(entry.get("symbol", "")).upper(): {**entry, "raw": text} for entry in entries}

    def fetch_symbols(self, *, reload: bool = True, is_shariah: bool = False, since: str = "") -> list[dict[str, Any]]:
        text = self._request_text(
            endpoint="ScripServlet",
            params={
                "ScripLoadDateTime": since,
                "IsReLoad": str(reload).lower(),
                "isShariah": str(is_shariah).lower(),
            },
            auth=False,
            audit_action="fetch_symbols",
        )
        data = _loads_json(text)
        if isinstance(data, list):
            return data
        return [{"raw": text}]

    def fetch_market_cap(self, symbol: str) -> dict[str, Any]:
        text = self._request_text(
            endpoint="GetMarketCap",
            params={"scrip": symbol.upper()},
            auth=False,
            audit_action="fetch_market_cap",
        )
        parts = text.strip().split("|")
        return {
            "symbol": symbol.upper(),
            "shares": _safe_int(parts[0]) if len(parts) > 0 else None,
            "market_cap": _safe_float(parts[1]) if len(parts) > 1 else None,
            "raw": text,
        }

    def fetch_ohlcv(
        self,
        symbol: str,
        market: str = "REG",
        interval: str = "1",
        *,
        check: str = "symbol",
    ) -> list[list[Any]]:
        text = self._request_text(
            endpoint="GetOHLCData",
            base_url=self.virtual_url,
            params={
                "scrip": symbol.upper(),
                "market": market.upper(),
                "check": check,
                "interval": interval,
            },
            auth=False,
            audit_action="fetch_ohlcv",
        )
        return _parse_ohlcv(text)

    def fetch_historical_ohlcv(
        self,
        symbol: str,
        *,
        since: str | date | datetime | int | float | None = None,
        until: str | date | datetime | int | float | None = None,
        years: int | None = None,
    ) -> list[list[Any]]:
        """Fetch PSX end-of-day history as CCXT-like OHLCV rows.

        PSX Data Portal's EOD time-series currently returns timestamp, close,
        volume, and open. High/low are not present, so those slots are None.
        """

        until_date = _coerce_date(until) if until is not None else None
        since_date = _coerce_date(since) if since is not None else None
        if years is not None:
            if isinstance(years, bool) or not isinstance(years, int) or years <= 0:
                raise ValueError("years must be a positive integer")
            cutoff = (until_date or date.today()) - timedelta(days=years * 365)
            since_date = max(since_date, cutoff) if since_date else cutoff
        if since_date and until_date and since_date > until_date:
            raise ValueError("since must be on or before until")

        text = self._request_text(
            endpoint=f"timeseries/eod/{symbol.upper()}",
            base_url=self.psx_data_url,
            auth=False,
            audit_action="fetch_historical_ohlcv",
        )
        data = _loads_json(text)
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise AhlError("Unrecognized PSX historical-data response")
        if data.get("status") not in (None, 1, "1"):
            raise BrokerRejectedError("PSX rejected the historical-data request")
        rows = data["data"]
        return _parse_psx_eod_ohlcv(rows, since=since_date, until=until_date)

    def fetch_historical_daily(
        self,
        symbol: str,
        *,
        since: str | date | datetime | int | float | None = None,
        until: str | date | datetime | int | float | None = None,
        years: int | None = None,
    ) -> list[dict[str, Any]]:
        rows = self.fetch_historical_ohlcv(symbol, since=since, until=until, years=years)
        return [
            {
                "date": datetime.fromtimestamp(row[0] / 1000, timezone.utc).date().isoformat(),
                "timestamp": row[0],
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
            }
            for row in rows
        ]

    def fetch_movers(
        self,
        identifier: str = "KSE100",
        count: int = 20,
        *,
        mover_type: str | None = None,
        with_feed: bool = False,
    ) -> dict[str, Any]:
        session = self.ensure_session()
        if with_feed:
            text = self._request_text(
                endpoint="TopMoversServletWithFeed",
                params={
                    "FromActivity": "TopMoversServletWithFeed",
                    "SESSION_ID": session.feed_session_id,
                    "identifier": identifier,
                    "count": str(count),
                    "type": mover_type or "",
                    "date": "0",
                },
                audit_action="fetch_movers",
            )
        else:
            text = self._request_text(
                endpoint="AllMoversFetcher",
                params={"identifier": identifier, "count": str(count)},
                auth=False,
                audit_action="fetch_movers",
            )
        return {"groups": _parse_mover_groups(text), "raw": text}

    def fetch_order_cancel_info(self, order_id: str) -> dict[str, Any]:
        session = self.ensure_session()
        text = self._request_text(
            endpoint="cancelorderinfo",
            params={
                "FromActivity": "CancelActivityCancelOrderInfo",
                "date": _javaish_date(),
                "orderNo": order_id,
                "userid": session.user_id,
                "usercode": session.user_code,
                "SESSION_ID": session.feed_session_id,
            },
            audit_action="fetch_order_cancel_info",
        )
        parsed = _parse_cancel_order_info(text)
        return {"id": order_id, **parsed, "raw": text}

    def fetch_order(self, order_id: str) -> dict[str, Any]:
        order_id_text = str(order_id)
        matches = []
        for logname in ("outstanding", "trade", "activity"):
            matches.extend(order for order in self.fetch_order_logs(logname) if str(order.get("id", "")) == order_id_text)
        if matches:
            primary = dict(matches[0])
            primary["events"] = [dict(match) for match in matches]
            return primary
        return self.fetch_order_cancel_info(order_id)

    def fetch_order_logs(
        self,
        logname: str,
        *,
        page_no: int = 1,
        record_size: int = 15000,
        account: str | None = None,
    ) -> list[dict[str, Any]]:
        session = self.ensure_session()
        mapped = _order_log_servlet_mapping(logname)
        if mapped is not None:
            from_activity, servlet_logname = mapped
            text = self._request_text(
                endpoint="LogsServletAndroid",
                params={
                    "FromActivity": from_activity,
                    "SESSION_ID": session.feed_session_id,
                    "userid": session.user_id,
                    "logname": servlet_logname,
                    "pageNo": page_no,
                    "recordSize": record_size,
                    "acc": account or session.account,
                },
                audit_action=f"fetch_order_logs_{servlet_logname}",
            )
            return _parse_log_servlet(text, servlet_logname, account=account or session.account)

        try:
            text = self._request_text(
                endpoint="GetOrderLogs",
                params={"logname": logname, "userid": session.user_id},
                audit_action="fetch_order_logs",
            )
        except TransportError:
            raise
        return _parse_order_logs(text)

    def fetch_open_orders(self, symbol: str | None = None, market: str | None = None) -> list[dict[str, Any]]:
        orders = self.fetch_order_logs("outstanding")
        return _filter_orders(orders, symbol=symbol, market=market)

    def fetch_closed_orders(
        self,
        symbol: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict[str, Any]]:
        orders = self.fetch_order_logs("trade")
        filtered = _filter_orders(orders, symbol=symbol, market=None)
        if since or until:
            lower = _coerce_date(since) if since else None
            upper = _coerce_date(until) if until else None
            if lower and upper and lower > upper:
                raise ValueError("since must be on or before until")
            dated = []
            for order in filtered:
                try:
                    value = datetime.strptime(str(order.get("datetime", "")), "%b %d, %Y %H:%M:%S").date()
                except ValueError:
                    raise AhlError("Cannot date-filter a trade with an unrecognized timestamp") from None
                if (lower is None or value >= lower) and (upper is None or value <= upper):
                    dated.append(order)
            filtered = dated
        return filtered

    def fetch_activity_logs(self, symbol: str | None = None, market: str | None = None) -> list[dict[str, Any]]:
        orders = self.fetch_order_logs("activity")
        return _filter_orders(orders, symbol=symbol, market=market)

    def cancel_all_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        results = []
        for order in self.fetch_open_orders(symbol=symbol):
            order_id = order.get("id") or order.get("order_id")
            if order_id:
                results.append(self.cancel_order(str(order_id)))
        return results

    def build_order_request(
        self,
        symbol: str,
        side: str,
        amount: int | str,
        price: float | str | None = None,
        *,
        order_type: str = "limit",
        market: str = "REG",
        account: str | None = None,
        pin: str = "",
        limit_price: float | str | None = None,
        exchange: str = "KSE",
        order_no: int | None = None,
    ) -> dict[str, Any]:
        request = self._build_order_request(
            symbol=symbol,
            side=side,
            amount=amount,
            price=price,
            order_type=order_type,
            market=market,
            account=account,
            pin=pin,
            limit_price=limit_price,
            exchange=exchange,
            order_no=order_no,
        )
        return request["public"]

    def create_order(
        self,
        symbol: str,
        side: str,
        amount: int | str,
        price: float | str | None = None,
        *,
        order_type: str = "limit",
        market: str = "REG",
        account: str | None = None,
        pin: str = "",
        limit_price: float | str | None = None,
        exchange: str = "KSE",
        order_no: int | None = None,
    ) -> dict[str, Any]:
        request = self._build_order_request(
            symbol=symbol,
            side=side,
            amount=amount,
            price=price,
            order_type=order_type,
            market=market,
            account=account,
            pin=pin,
            limit_price=limit_price,
            exchange=exchange,
            order_no=order_no,
        )
        self._run_order_risk_checks(request["order"], live=not self.dry_run)
        if self.dry_run:
            result = _normalize_order_result(request["order"], "", dry_run=True, status="dry_run")
            result["request"] = request["public"]
            self._audit("create_order_dry_run", request=request["private"], parsed=result)
            return result

        text = self._request_text(raw_url=request["private"]["url"], audit_action="create_order")
        self._advance_max_order(request["private"]["next_order_no"])
        result = _normalize_order_result(request["order"], text, dry_run=False)
        result["request"] = request["public"]
        self._audit("create_order_result", request=request["private"], response=text, parsed=result)
        return result

    def build_cancel_order_request(self, order_id: str, *, pin: str = "", order_no: int | None = None) -> dict[str, Any]:
        request = self._build_cancel_order_request(order_id, pin=pin, order_no=order_no)
        return request["public"]

    def cancel_order(self, order_id: str, *, pin: str = "", order_no: int | None = None) -> dict[str, Any]:
        request = self._build_cancel_order_request(order_id, pin=pin, order_no=order_no)
        if self.dry_run:
            result = _normalize_cancel_result(order_id, "", dry_run=True, status="dry_run")
            result["request"] = request["public"]
            self._audit("cancel_order_dry_run", request=request["private"], parsed=result)
            return result

        text = self._request_text(raw_url=request["private"]["url"], audit_action="cancel_order")
        self._advance_max_order(request["private"]["next_order_no"])
        result = _normalize_cancel_result(order_id, text, dry_run=False)
        result["request"] = request["public"]
        self._audit("cancel_order_result", request=request["private"], response=text, parsed=result)
        return result

    def _build_order_request(
        self,
        *,
        symbol: str,
        side: str,
        amount: int | str,
        price: float | str | None,
        order_type: str,
        market: str,
        account: str | None,
        pin: str,
        limit_price: float | str | None,
        exchange: str,
        order_no: int | None,
    ) -> dict[str, Any]:
        session = self.ensure_session()
        pin = self._resolve_pin(pin)
        quantity = _order_quantity(amount)
        normalized_symbol = symbol.strip().upper()
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]*", normalized_symbol):
            raise RiskCheckError("Symbol must be a nonempty broker symbol.")
        if price is not None and _safe_float(price) is None:
            raise RiskCheckError("Price must be finite and numeric.")
        if limit_price is not None and _safe_float(limit_price) is None:
            raise RiskCheckError("limit_price must be finite and numeric.")
        broker_side = _broker_side(side)
        broker_order_type = _broker_order_type(order_type)
        broker_market = "LB" if broker_side in {"LB Buy", "LB Sell"} else market.upper()
        effective_price = "0.0" if price is None else _clean_numeric_text(price)
        effective_limit_price = "" if limit_price is None else _clean_numeric_text(limit_price)
        order_no = session.max_order if order_no is None else int(order_no)
        account = account or session.account
        order = {
            "id": None,
            "client_order_id": order_no,
            "symbol": normalized_symbol,
            "side": _standard_side(broker_side),
            "type": _standard_order_type(broker_order_type),
            "broker_side": broker_side,
            "broker_order_type": broker_order_type,
            "market": broker_market,
            "price": _safe_float(effective_price),
            "amount": quantity,
            "account": account,
            "exchange": exchange.upper(),
            "limit_price": _safe_float(effective_limit_price) if effective_limit_price else None,
        }
        abc = (
            quote_plus(_javaish_date())
            + f"&userid={_q(session.user_id)}"
            + f"&usercode={_q(session.user_code)}"
            + f"&orderno={order_no}"
            + f"&account={_q(account)}"
            + f"&buysell={_q(broker_side)}"
            + f"&market={_q(broker_market)}"
            + f"&order={_q(broker_order_type)}"
            + f"&volume={quantity}"
            + f"&scrip={_q(normalized_symbol)}"
            + f"&price={_q(effective_price)}"
            + f"&pin={_q(pin)}"
            + f"&limitprice={_q(effective_limit_price)}"
            + f"&exchange={_q(exchange.upper())}"
        )
        raw_url = f"{self.base_url}order?FromActivity=OrderActivityOrderCall&abc={abc}&SESSION_ID={_q(session.feed_session_id)}"
        private = {"endpoint": "order", "url": raw_url, "abc": abc, "order_no": order_no, "next_order_no": order_no + 1}
        public = {
            **private,
            "url": _redact_request_preview(raw_url),
            "abc": _redact_request_preview(abc),
            "order": _redact_sensitive(order),
        }
        return {"private": private, "public": public, "order": order}

    def _build_cancel_order_request(self, order_id: str, *, pin: str, order_no: int | None) -> dict[str, Any]:
        if not str(order_id).strip():
            raise ValueError("order_id cannot be empty")
        session = self.ensure_session()
        pin = self._resolve_pin(pin)
        order_no = session.max_order if order_no is None else int(order_no)
        raw_url = (
            f"{self.base_url}cancelOrder?FromActivity=CancelActivityCancelOrder"
            f"&abc={quote_plus(_javaish_date())}"
            f"&userid={_q(session.user_id)}"
            f"&usercode={_q(session.user_code)}"
            f"&origorderno={_q(order_id)}"
            f"&orderno={order_no}"
            f"&pin={_q(pin)}"
            f"&SESSION_ID={_q(session.feed_session_id)}"
        )
        private = {"endpoint": "cancelOrder", "url": raw_url, "order_no": order_no, "next_order_no": order_no + 1}
        public = {**private, "url": _redact_request_preview(raw_url)}
        return {"private": private, "public": public}

    def _run_order_risk_checks(self, order: dict[str, Any], *, live: bool) -> None:
        symbol = str(order["symbol"]).upper()
        amount = order.get("amount")
        price = order.get("price")
        if amount is None or amount <= 0:
            raise RiskCheckError("Order amount must be a positive integer.")
        if order["type"] != "market" and (price is None or not math.isfinite(price) or price <= 0):
            raise RiskCheckError("Limit and stop-loss orders require a positive price.")
        if order["type"] == "market" and not self.allow_market_orders:
            raise RiskCheckError("Market orders are disabled for this client.")
        if order["side"] == "sell" and str(order.get("broker_side", "")).upper() == "SHORT SELL" and not self.allow_short_sell:
            raise RiskCheckError("Short selling is disabled for this client.")
        if self.allowed_symbols is not None and symbol not in self.allowed_symbols:
            raise RiskCheckError(f"{symbol} is not in allowed_symbols.")
        if self.blocked_symbols is not None and symbol in self.blocked_symbols:
            raise RiskCheckError(f"{symbol} is blocked for this client.")
        if self.max_order_quantity is not None and amount > self.max_order_quantity:
            raise RiskCheckError(f"Order quantity {amount} exceeds max_order_quantity {self.max_order_quantity}.")
        if price is not None and (not math.isfinite(price) or price < 0):
            raise RiskCheckError("Price must be finite and nonnegative.")
        if order["type"] == "stop_loss" and (order.get("limit_price") is None or order["limit_price"] <= 0):
            raise RiskCheckError("Stop-loss orders require a positive limit_price.")
        if order["type"] == "stop_loss":
            price = max(price, order["limit_price"])
        if order["type"] == "market" and (self.max_order_value is not None or (live and self.check_buying_power)):
            if not live:
                raise RiskCheckError("Market-order value cannot be checked in dry-run mode.")
            price = self._market_order_buying_power_price(symbol, str(order["market"]))
            if price is None:
                raise RiskCheckError("Cannot verify market-order value without a price.")
        if price is not None and self.max_order_value is not None and amount * price > self.max_order_value:
            raise RiskCheckError(f"Order value {amount * price} exceeds max_order_value {self.max_order_value}.")
        if live and self.check_price_bands and order["type"] != "market" and price is not None:
            self._check_price_band(symbol, str(order["market"]), price)
        if live and self.check_buying_power and order["side"] == "buy":
            buying_power_price = price
            if buying_power_price is not None:
                self._check_buying_power(amount, buying_power_price, account=order["account"], market=order["market"])

    def _check_price_band(self, symbol: str, market: str, price: float) -> None:
        ticker = self.fetch_ticker(symbol, market=market, with_market_cap=True)
        cap = ticker.get("cap")
        if not isinstance(cap, dict):
            raise RiskCheckError("Cannot verify broker price bands.")
        lower = _safe_float(str(cap.get("lowerLocked", "")))
        upper = _safe_float(str(cap.get("upperCapped", "")))
        if lower is None or upper is None or lower <= 0 or upper < lower:
            raise RiskCheckError("Cannot verify broker price bands.")
        if lower is not None and price < lower:
            raise RiskCheckError(f"Price {price} is below lower lock {lower}.")
        if upper is not None and price > upper:
            raise RiskCheckError(f"Price {price} is above upper cap {upper}.")

    def _check_buying_power(self, amount: int, price: float, *, account: str | None = None, market: str = "REG") -> None:
        if market not in {"REG", "FUT"}:
            raise RiskCheckError("Buying-power checks are supported for REG and FUT only.")
        buying_power = self.fetch_buying_power(account=account).get("future" if market == "FUT" else "regular")
        if buying_power is None:
            raise RiskCheckError("Cannot verify broker buying power.")
        if buying_power is not None and amount * price > buying_power:
            raise RiskCheckError(f"Order value {amount * price} exceeds {market} buying power {buying_power}.")

    def _market_order_buying_power_price(self, symbol: str, market: str) -> float | None:
        ticker = self.fetch_ticker(symbol, market=market, with_market_cap=True)
        cap = ticker.get("cap")
        if isinstance(cap, dict):
            upper = _safe_float(str(cap.get("upperCapped", "")))
            if upper is not None and upper > 0:
                return upper
        for key in ("sell", "last", "buy"):
            value = _safe_float(ticker.get(key))
            if value is not None and value > 0:
                return value
        return None

    def _resolve_pin(self, pin: str | None) -> str:
        return str(pin if pin not in (None, "") else self.pin)

    def _looks_session_expired(self, text: str) -> bool:
        lowered = text.lower()
        return "session" in lowered and any(token in lowered for token in ("expired", "invalid", "not valid", "logout"))

    def _load_session_state(self) -> dict[str, Any]:
        if not self.session_state_path.exists():
            return {}
        try:
            data = json.loads(self.session_state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _load_max_order(self, account: str) -> int:
        state = self._load_session_state()
        account_state = state.get("accounts", {}).get(account, {})
        try:
            return int(account_state.get("max_order", 0))
        except (TypeError, ValueError):
            return 0

    def _persist_max_order(self, value: int) -> None:
        if self.session is None:
            return
        state = self._load_session_state()
        accounts = state.setdefault("accounts", {})
        accounts[self.session.account] = {
            "max_order": int(value),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.session_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    def _advance_max_order(self, value: int) -> None:
        if self.session is None:
            return
        self.session.max_order = max(self.session.max_order, int(value))
        self._persist_max_order(self.session.max_order)

    def _audit(
        self,
        action: str,
        *,
        request: dict[str, Any] | None = None,
        response: str | None = None,
        parsed: Any | None = None,
        exception: Exception | None = None,
    ) -> None:
        if not self.audit_enabled:
            return
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "action": action,
            "request": request,
            "response": response,
            "parsed": parsed,
            "exception": repr(exception) if exception else None,
        }
        record = _redact_sensitive(record)
        serialized = json.dumps(record, default=str, ensure_ascii=True)
        for secret in (self.password, self.pin, self.session.feed_session_id if self.session else None):
            if secret:
                serialized = serialized.replace(json.dumps(str(secret))[1:-1], "[REDACTED]")
        path = self.audit_dir / f"{datetime.now().strftime('%Y%m%d')}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(serialized + "\n")


AhlClient = AHL
AhlClientError = AhlError


def _loads_json(value: str) -> Any:
    try:
        return requests.models.complexjson.loads(value)
    except ValueError:
        return None


def _order_quantity(value: Any) -> int:
    if isinstance(value, bool):
        raise RiskCheckError("Order amount must be a positive integer.")
    try:
        quantity = Decimal(str(value).replace(",", "").strip())
        if not quantity.is_finite() or quantity <= 0 or quantity != quantity.to_integral_value():
            raise InvalidOperation
        return int(quantity)
    except (InvalidOperation, ValueError, TypeError):
        raise RiskCheckError("Order amount must be a positive integer.") from None


def _parse_portfolio(text: str) -> dict[str, Any]:
    positions = []
    summary: dict[str, Any] = {}
    for chunk in text.strip().split("|"):
        if not chunk.strip():
            continue
        if chunk.startswith("$"):
            fields = chunk[1:].split(";")
            names = ("cash", "market_value", "day_pnl", "unrealized_pnl", "day_pnl_percent", "unrealized_pnl_percent", "cost_basis")
            summary = {name: _safe_float(fields[i]) for i, name in enumerate(names) if i < len(fields)}
            if summary.get("cash") is not None and summary.get("market_value") is not None:
                summary["net_worth"] = summary["cash"] + summary["market_value"]
            continue
        fields = chunk.split(";")
        if len(fields) < 16:
            raise AhlError("Unrecognized portfolio response; expected holding rows and a summary.")
        quantity = _safe_int(fields[2])
        last = _safe_float(fields[4])
        positions.append({
            "symbol": fields[0], "approval_status": fields[1], "quantity": quantity,
            "average_cost": _safe_float(fields[3]), "last": last,
            "cost_basis": _safe_float(fields[5]), "day_pnl": _safe_float(fields[7]),
            "unrealized_pnl": _safe_float(fields[8]), "unrealized_pnl_percent": _safe_float(fields[10]),
            "market": fields[15],
            "market_value": quantity * last if quantity is not None and last is not None else None,
        })
    return {"positions": positions, "summary": summary}


def _javaish_date() -> str:
    return datetime.now().strftime("%a %b %d %H:%M:%S GMT+05:00 %Y")


def _q(value: Any) -> str:
    return quote_plus(str(value))


def _clean_numeric_text(value: Any) -> str:
    return str(value).replace("PKR ", "").replace(",", "").strip()


def _normalize_symbol_set(values: set[str] | list[str] | tuple[str, ...] | None) -> set[str] | None:
    if values is None:
        return None
    return {str(value).upper() for value in values}


def _broker_side(side: str) -> str:
    normalized = side.strip().lower().replace("_", " ")
    mapping = {
        "buy": "BUY",
        "sell": "SEL",
        "short sell": "SHORT SELL",
        "lb buy": "LB Buy",
        "lb sell": "LB Sell",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported side: {side!r}")
    return mapping[normalized]


def _broker_order_type(order_type: str) -> str:
    normalized = order_type.strip().lower().replace("_", " ")
    mapping = {
        "limit": "limit",
        "market": "mktorder",
        "mkt": "mktorder",
        "stop loss": "StopLoss",
        "stoploss": "StopLoss",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported order_type: {order_type!r}")
    return mapping[normalized]


def _standard_side(broker_side: str) -> str:
    upper = broker_side.upper()
    if "BUY" in upper:
        return "buy"
    return "sell"


def _standard_order_type(broker_order_type: str) -> str:
    if broker_order_type == "mktorder":
        return "market"
    if broker_order_type == "StopLoss":
        return "stop_loss"
    return "limit"


def _parse_feed_entries(feed: str) -> list[dict[str, Any]]:
    body, _, status = feed.partition("|MKTSTATUS")
    market_status = status.replace("MarketCap", "").strip() or None
    entries = []
    for chunk in body.split("|"):
        fields = chunk.split(";")
        if len(fields) < 2 or not fields[0]:
            continue
        entry = _parse_feed_fields(fields)
        entry["market_status"] = market_status
        entries.append(entry)
    return entries


def _parse_feed_fields(fields: list[str]) -> dict[str, Any]:
    names = [
        "symbol",
        "buy_volume",
        "buy",
        "sell_volume",
        "sell",
        "last",
        "time",
        "close",
        "average",
        "high",
        "low",
        "change",
        "volume",
        "last_trade_volume",
    ]
    parsed: dict[str, Any] = {"fields": fields}
    for index, name in enumerate(names):
        if index < len(fields):
            parsed[name] = _number_or_text(fields[index])
    return parsed


def _parse_ohlcv(text: str) -> list[list[Any]]:
    candles = []
    for chunk in text.strip().split("|"):
        if not chunk:
            continue
        fields = chunk.split(";")
        if len(fields) < 7:
            continue
        candles.append(
            [
                fields[6],
                _safe_float(fields[2]),
                _safe_float(fields[3]),
                _safe_float(fields[4]),
                _safe_float(fields[5]),
                None,
            ]
        )
    return candles


def _parse_psx_eod_ohlcv(
    rows: list[Any],
    *,
    since: date | None = None,
    until: date | None = None,
) -> list[list[Any]]:
    candles = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 4:
            continue
        timestamp_seconds = _safe_int(row[0])
        if timestamp_seconds is None:
            continue
        row_date = datetime.fromtimestamp(timestamp_seconds, timezone.utc).date()
        if since and row_date < since:
            continue
        if until and row_date > until:
            continue
        close = _safe_float(row[1])
        volume = _safe_int(row[2])
        open_price = _safe_float(row[3])
        candles.append([timestamp_seconds * 1000, open_price, None, None, close, volume])
    return sorted(candles, key=lambda item: item[0])


def _parse_mover_groups(text: str) -> list[list[dict[str, str]]]:
    groups = []
    for group in text.strip().split("|"):
        if not group:
            continue
        symbols = []
        for item in group.split(";"):
            if not item:
                continue
            symbol, _, market = item.partition(":")
            symbols.append({"symbol": symbol, "market": market or ""})
        groups.append(symbols)
    return groups


def _parse_cancel_order_info(text: str) -> dict[str, Any]:
    fields = text.strip().split("|")
    names = ["side", "market", "type", "symbol", "amount", "price", "account", "id"]
    parsed: dict[str, Any] = {"fields": fields}
    for index, name in enumerate(names):
        if index < len(fields):
            parsed[name] = _number_or_text(fields[index])
    if "side" in parsed:
        parsed["side"] = str(parsed["side"]).lower()
    parsed.setdefault("status", "open")
    return parsed


def _parse_order_logs(text: str) -> list[dict[str, Any]]:
    data = _loads_json(text)
    if isinstance(data, list):
        return [_normalize_log_item(item) for item in data]
    rows = re.split(r"[\r\n$]+", text.strip())
    orders = []
    for row in rows:
        row = row.strip()
        if not row or row.startswith("<!DOCTYPE"):
            continue
        delimiter = "|" if "|" in row else ";"
        fields = [part.strip() for part in row.split(delimiter)]
        order = {"raw": row, "fields": fields}
        maybe_id = next((field for field in fields if field.isdigit()), None)
        if maybe_id:
            order["id"] = maybe_id
        orders.append(order)
    return orders


def _order_log_servlet_mapping(logname: str) -> tuple[str, str] | None:
    normalized = logname.strip().lower().replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    mapping = {
        "outstanding": ("outstandinglogRunnable", "outstanding"),
        "outstanding log": ("outstandinglogRunnable", "outstanding"),
        "open": ("outstandinglogRunnable", "outstanding"),
        "open orders": ("outstandinglogRunnable", "outstanding"),
        "trade": ("tradelog", "trade"),
        "trade log": ("tradelog", "trade"),
        "closed": ("tradelog", "trade"),
        "closed orders": ("tradelog", "trade"),
        "activity": ("activityLog", "activity"),
        "activity log": ("activityLog", "activity"),
    }
    return mapping.get(normalized)


def _parse_log_servlet(text: str, logname: str, *, account: str | None = None) -> list[dict[str, Any]]:
    rows = _split_log_servlet_rows(text)
    if logname == "trade":
        return [_parse_trade_log_row(row) for row in rows]
    if logname == "activity":
        return [_parse_activity_log_row(row) for row in rows]
    if logname == "outstanding":
        return [_parse_outstanding_log_row(row, account=account) for row in rows]
    return _parse_order_logs(text)


def _split_log_servlet_rows(text: str) -> list[str]:
    body = text.strip()
    if not body:
        return []
    body, _, _count = body.partition("^")
    return [row.strip() for row in body.split("|") if row.strip() and ";" in row]


def _parse_trade_log_row(row: str) -> dict[str, Any]:
    fields = [part.strip() for part in row.split(";")]
    order = _base_log_order(row, fields, "trade")
    names = [
        "symbol",
        "market",
        "id",
        "datetime",
        "broker_side",
        "price",
        "value",
        "filled",
        "amount",
        "remaining",
        "trader",
        "order_type",
    ]
    _apply_fields(order, fields, names)
    _normalize_common_order_fields(order)
    remaining = order.get("remaining")
    order["status"] = "closed" if remaining in (0, 0.0) else "partially_filled"
    return order


def _parse_activity_log_row(row: str) -> dict[str, Any]:
    fields = [part.strip() for part in row.split(";")]
    order = _base_log_order(row, fields, "activity")
    names = [
        "symbol",
        "market",
        "datetime",
        "broker_side",
        "id",
        "house_order_no",
        "action_code",
        "price",
        "amount",
        "filled",
        "remaining",
        "trader",
        "order_type",
    ]
    _apply_fields(order, fields, names)
    _normalize_common_order_fields(order)
    order["action"] = _order_action_name(str(order.get("action_code", "")))
    order["status"] = _status_from_action(str(order.get("action_code", "")), order.get("remaining"))
    return order


def _parse_outstanding_log_row(row: str, *, account: str | None = None) -> dict[str, Any]:
    fields = [part.strip() for part in row.split(";")]
    order = _base_log_order(row, fields, "outstanding")
    names = [
        "symbol",
        "market",
        "id",
        "datetime",
        "broker_side",
        "price",
        "remaining",
        "house_order_no",
        "trader",
        "order_type",
        "action_code",
    ]
    _apply_fields(order, fields, names)
    _normalize_common_order_fields(order)
    order["amount"] = order.get("remaining")
    order["status"] = "open"
    if account:
        order["account"] = account
    if "price" in order and "remaining" in order and order["price"] is not None and order["remaining"] is not None:
        order["value"] = order["price"] * order["remaining"]
    order["action"] = _order_action_name(str(order.get("action_code", "")))
    return order


def _base_log_order(row: str, fields: list[str], logname: str) -> dict[str, Any]:
    return {"raw": row, "fields": fields, "logname": logname}


def _apply_fields(order: dict[str, Any], fields: list[str], names: list[str]) -> None:
    numeric_fields = {"price", "value", "amount", "filled", "remaining"}
    for index, name in enumerate(names):
        if index >= len(fields):
            continue
        value = fields[index]
        if name in numeric_fields:
            number = _safe_float(value)
            if name in {"amount", "filled", "remaining"}:
                order[name] = _safe_int(value) if value != "" else None
            else:
                order[name] = number
        else:
            order[name] = value


def _normalize_common_order_fields(order: dict[str, Any]) -> None:
    if "id" in order:
        order["id"] = str(order["id"])
        order.setdefault("order_no", order["id"])
    broker_side = str(order.get("broker_side", ""))
    if broker_side:
        order["side"] = _standard_side(broker_side)
    order_type = str(order.get("order_type", "")).strip()
    if order_type:
        upper_order_type = order_type.upper()
        if upper_order_type in {"MKT", "MKTORDER", "MARKET"}:
            order["type"] = "market"
        elif upper_order_type in {"LIMIT", "LMT"}:
            order["type"] = "limit"
        elif upper_order_type in {"STOPLOSS", "STOP LOSS"}:
            order["type"] = "stop_loss"
        else:
            order["order_kind"] = order_type
    for key in ("symbol", "market"):
        if key in order and isinstance(order[key], str):
            order[key] = order[key].upper()


def _order_action_name(action_code: str) -> str:
    mapping = {
        "TRD": "traded",
        "QUE": "queued",
        "CXL": "cancelled",
        "REJ": "rejected",
        "ACP": "accepted",
        "ACT": "active",
        "NOR": "new_order",
        "CFO": "change_order",
    }
    return mapping.get(action_code.strip().upper(), action_code)


def _status_from_action(action_code: str, remaining: Any) -> str:
    action = action_code.strip().upper()
    if action == "TRD":
        return "closed" if remaining in (0, 0.0, "0") else "partially_filled"
    if action == "CXL":
        return "cancelled"
    if action == "REJ":
        return "rejected"
    if action in {"QUE", "ACP", "ACT", "NOR", "CFO"}:
        return "open"
    return "unknown"


def _normalize_log_item(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        order = dict(item)
    else:
        order = {"raw": item}
    for key in ("orderNo", "orderno", "order_no", "OrderNumber"):
        if key in order and "id" not in order:
            order["id"] = str(order[key])
    return order


def _filter_orders(orders: list[dict[str, Any]], *, symbol: str | None, market: str | None) -> list[dict[str, Any]]:
    symbol = symbol.upper() if symbol else None
    market = market.upper() if market else None
    filtered = []
    for order in orders:
        order_symbol = str(order.get("symbol", order.get("scrip", ""))).upper()
        order_market = str(order.get("market", "")).upper()
        if symbol and order_symbol and order_symbol != symbol:
            continue
        if market and order_market and order_market != market:
            continue
        filtered.append(order)
    return filtered


def _parse_exposure_by_market(text: str) -> dict[str, dict[str, Any]]:
    markets: dict[str, dict[str, Any]] = {}
    for section in text.strip().split("$"):
        if ">" not in section:
            continue
        market, payload = section.split(">", 1)
        values = {}
        for item in payload.split("|"):
            if ";" in item:
                key, value = item.split(";", 1)
                values[key.strip()] = _number_or_text(value)
        markets[market.strip()] = values
    return markets


def _normalize_order_result(order: dict[str, Any], text: str, *, dry_run: bool, status: str | None = None) -> dict[str, Any]:
    status = status or _infer_order_status(text)
    return {
        "id": _extract_order_id(text),
        "client_order_id": order.get("client_order_id"),
        "symbol": order.get("symbol"),
        "side": order.get("side"),
        "type": order.get("type"),
        "price": order.get("price"),
        "amount": order.get("amount"),
        "status": status,
        "ok": status in {"dry_run", "submitted", "open", "closed"},
        "dry_run": dry_run,
        "raw": text,
    }


def _normalize_cancel_result(order_id: str, text: str, *, dry_run: bool, status: str | None = None) -> dict[str, Any]:
    status = status or _infer_cancel_status(text)
    return {
        "id": order_id,
        "status": status,
        "ok": status in {"dry_run", "cancelled", "pending_cancel"},
        "dry_run": dry_run,
        "raw": text,
    }


def _infer_order_status(text: str) -> str:
    lowered = text.lower()
    if not text:
        return "unknown"
    if any(token in lowered for token in ("rejected", "invalid", "insufficient", "cannot", "error")):
        return "rejected"
    if any(token in lowered for token in ("bought", "sold", "filled", "executed")):
        return "closed"
    if any(token in lowered for token in ("sent", "trade server", "accepted", "new")):
        return "submitted"
    return "unknown"


def _infer_cancel_status(text: str) -> str:
    lowered = text.lower()
    if not text:
        return "unknown"
    if "cancelled" in lowered or "canceled" in lowered:
        return "cancelled"
    if any(token in lowered for token in ("rejected", "invalid", "cannot", "error")):
        return "rejected"
    if "sent" in lowered or "accepted" in lowered:
        return "pending_cancel"
    return "unknown"


def _extract_order_id(text: str) -> str | None:
    match = re.search(r"\b(?:order(?:no| number)?|orderno)\D+(\d+)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _number_or_text(value: str) -> Any:
    value = value.strip()
    if value == "":
        return value
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _safe_float(value: Any) -> float | None:
    try:
        cleaned = str(value).replace(",", "").replace("%", "").strip()
        if cleaned == "":
            return None
        number = float(cleaned)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        cleaned = str(value).replace(",", "").strip()
        if cleaned == "":
            return None
        return int(float(cleaned))
    except (TypeError, ValueError, OverflowError):
        return None


def _coerce_date(value: str | date | datetime | int | float) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)):
        timestamp = value / 1000 if value > 10_000_000_000 else value
        return datetime.fromtimestamp(timestamp, timezone.utc).date()
    text = str(value).strip()
    if not text:
        raise ValueError("Date value cannot be empty.")
    return datetime.fromisoformat(text[:10]).date()


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key.lower() in SENSITIVE_RESPONSE_KEYS else _redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    if isinstance(value, str):
        parsed = _loads_json(value)
        if isinstance(parsed, (dict, list)):
            return json.dumps(_redact_sensitive(parsed))
        return _redact_request_preview(value)
    return value


def _redact_request_preview(value: str) -> str:
    redacted = value
    for key in SENSITIVE_QUERY_KEYS:
        redacted = re.sub(
            rf"([?&]{re.escape(key)}=)[^&]*",
            r"\1[REDACTED]",
            redacted,
            flags=re.IGNORECASE,
        )
    return redacted


__all__ = [
    "AHL",
    "AhlClient",
    "AhlClientError",
    "AhlError",
    "AuthenticationError",
    "BrokerRejectedError",
    "NotSupported",
    "RiskCheckError",
    "SessionExpiredError",
    "TransportError",
    "read_dotenv",
]
