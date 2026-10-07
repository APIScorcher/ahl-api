"""Exchange contracts exercised through the actual broker parser and guards."""

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

import pytest

import ahl_api
from ahl_api import AuthenticationError, BadSymbol, NotSupported
from ahl_api.client import AHL
from test_client import FakeHTTP, seeded_client
from test_client_validation import HOLDING, TRADE


CATALOGUE = [
    {"symbol": "DEMO", "market": "REG", "symbolName": "Demo Equity"},
    {"symbol": "DEMO", "market": "ODL"},
    {"symbol": "OTHER", "market": "REG"},
    {"symbol": "DEMO-OCT", "market": "FUT"},
]


def exchange(**options):
    result = ahl_api.ahl(
        {
            "username": "DEMO_USER",
            "password": "DEMO_PASSWORD",
            "pin": "0007",
            "enableRateLimit": False,
            "options": {"checkPriceBands": False, "checkBuyingPower": False, **options},
        }
    )
    result.client.session = seeded_client().session
    result.client.fetch_symbols = Mock(return_value=CATALOGUE)
    return result


def test_config_metadata_units_and_core_compatibility():
    result = ahl_api.ahl(
        {
            "username": "U",
            "password": "P",
            "pin": "0007",
            "timeout": 2500,
            "rateLimit": 250,
            "options": {"maxOrderQuantity": 10},
        }
    )
    assert result.client.timeout == 2.5
    assert result.client.rate_limit_ms == 250
    assert result.client.max_order_quantity == 10
    assert result.has["fetchOrderBook"] is False
    assert result.requiredCredentials["pin"] is False
    assert result.pin == "0007"
    assert "0007" not in json.dumps(result.describe())
    assert ahl_api.Exchange is ahl_api.ahl
    assert AHL().describe()["id"] == "ahl"


def test_auth_check():
    result = ahl_api.ahl()
    assert result.check_required_credentials(False) is False
    with pytest.raises(AuthenticationError):
        result.check_required_credentials()
    assert exchange().check_required_credentials()


def test_markets_are_cached_and_partitioned():
    result = exchange()
    markets = result.load_markets()
    assert result.symbols == ["DEMO/PKR", "OTHER/PKR"]
    assert result.market("DEMO")["symbol"] == "DEMO/PKR"
    assert result.market("DEMO/PKR")["base"] == "DEMO"
    assert result.markets_by_id["DEMO"][0]["quote"] == "PKR"
    assert markets["DEMO/PKR"]["active"] is None
    result.load_markets()
    result.client.fetch_symbols.assert_called_once()
    result.load_markets(True)
    assert result.client.fetch_symbols.call_count == 2
    assert len(result.fetch_markets({"market": "ODL"})) == 1
    with pytest.raises(NotSupported):
        result.load_markets(params={"unexpected": True})


@pytest.mark.parametrize("symbol", ["DEMO/USD", "DEMO/PKR/REG", "UNKNOWN", ""])
def test_invalid_symbols(symbol):
    with pytest.raises(BadSymbol):
        exchange().market(symbol)


def test_whole_share_precision():
    result = exchange()
    assert result.amount_to_precision("DEMO", 2.0) == "2"
    for amount in (2.5, float("inf"), "nan", True):
        with pytest.raises(ValueError):
            result.amount_to_precision("DEMO", amount)


def test_unified_ticker_fields_and_params():
    result = exchange()
    result.client.http = FakeHTTP(
        [json.dumps({"feedString": "DEMO;10;100;20;102;101;10:00;99;100;103;98;2;500;10;|MKTSTATUSOpen"})]
    )
    quote = result.fetch_ticker("DEMO/PKR", {"market": "ODL"})
    assert quote["symbol"] == "DEMO/PKR"
    assert quote["bid"] == 100 and quote["ask"] == 102 and quote["last"] == 101
    assert quote["baseVolume"] == 500
    assert quote["timestamp"] is None  # Broker only supplies time of day.
    assert result.client.http.calls[0]["params"]["symbollist"] == "DEMO:ODL|"


def test_multiple_tickers():
    result = exchange()
    result.client.http = FakeHTTP(
        [
            json.dumps(
                {
                    "feedString": "DEMO;0;100;0;102;101;10:00;99;100;103;98;2;500;10;|OTHER;0;200;0;202;201;10:00;199;200;203;198;2;500;10;|"
                }
            )
        ]
    )
    assert set(result.fetch_tickers(["DEMO", "OTHER/PKR"])) == {"DEMO/PKR", "OTHER/PKR"}
    assert result.fetch_tickers([]) == {}
    with pytest.raises(NotSupported):
        result.fetch_tickers()


def test_balance_uses_cash_and_share_units_without_guessing_free():
    result = exchange()
    result.client.http = FakeHTTP(["true|25|", HOLDING])
    balance = result.fetch_balance()
    assert balance["PKR"] == {"free": None, "used": None, "total": 25}
    assert balance["total"]["DEMO"] == 10
    assert balance["free"]["DEMO"] is None
    assert balance["total"]["PKR"] != 1125  # Net worth is not cash balance.


def test_daily_candles_since_and_limit():
    result = exchange()
    result.client.http = FakeHTTP(['{"status":1,"data":[[1767225600,110,10,100],[1767312000,112,11,110]]}'])
    rows = result.fetch_ohlcv("DEMO", "1d", 1767312000000, 1)
    assert rows == [[1767312000000, 110.0, None, None, 112.0, 11]]


def test_intraday_date_is_explicit_and_karachi_time():
    result = exchange()
    with pytest.raises(ValueError, match="require"):
        result.fetch_ohlcv("DEMO", "5m")
    result.client.http = FakeHTTP(["DEMO;REG;100;102;99;101;09:30|"])
    rows = result.fetch_ohlcv("DEMO", "5m", params={"date": "2026-10-07"})
    expected = int(datetime(2026, 10, 7, 9, 30, tzinfo=timezone(timedelta(hours=5))).timestamp() * 1000)
    assert rows[0] == [expected, 100, 102, 99, 101, None]
    with pytest.raises(NotSupported):
        result.fetch_ohlcv("DEMO", "7m")


def test_ccxt_order_signature_keeps_dryrun_and_redacts_pin():
    result = exchange()
    params = {"pin": "0011", "market": "ODL"}
    order = result.create_order("DEMO/PKR", "limit", "buy", 2, 100, params)
    assert order["symbol"] == "DEMO/PKR" and order["side"] == "buy"
    assert order["amount"] == 2 and order["price"] == 100
    assert order["clientOrderId"] == "10"
    assert order["dry_run"] is True
    assert order["status"] is None and order["filled"] is None
    assert params == {"pin": "0011", "market": "ODL"}
    assert "0011" not in json.dumps(order)
    assert "0007" not in json.dumps(order)
    assert "SESSION1" not in json.dumps(order)


def test_live_pin_override_reaches_wire_only(tmp_path):
    result = exchange(dryRun=False, sessionStatePath=str(tmp_path / "state.json"))
    result.client.http = FakeHTTP(["Order has been sent to Trade Server."])
    order = result.create_order("DEMO", "limit", "buy", 1, 100, {"pin": "0011"})
    assert "pin=0011" in result.client.http.calls[0]["url"]
    assert "0011" not in json.dumps(order)
    assert "0011" not in (tmp_path / "state.json").read_text()
    assert order["status"] is None  # Forwarding acknowledgement does not prove open/fill state.
    assert order["info"]["status"] == "submitted"


def test_missing_pin_stops_live_order_and_cancel_before_http():
    result = exchange(dryRun=False)
    result.client.pin = ""
    result.client.http = FakeHTTP([])
    with pytest.raises(AuthenticationError):
        result.create_order("DEMO", "limit", "buy", 1, 100)
    with pytest.raises(AuthenticationError):
        result.cancel_order("DEMO_ORDER")
    with pytest.raises(AuthenticationError):
        result.client.create_order("DEMO", "buy", 1, 100)
    with pytest.raises(AuthenticationError):
        result.client.cancel_order("DEMO_ORDER")
    assert not result.client.http.calls


def test_default_pin_used_for_statement_and_cancel(tmp_path):
    result = exchange(dryRun=False, sessionStatePath=str(tmp_path / "state.json"))
    result.client.http = FakeHTTP(["[]", "Cancelled"])
    result.client.fetch_account_statement("2026-10-01", "2026-10-07")
    assert result.client.http.calls[0]["params"]["pincode"] == "0007"
    cancellation = result.cancel_order("DEMO_ORDER", "DEMO/PKR")
    assert "pin=0007" in result.client.http.calls[1]["url"]
    assert cancellation["status"] == "canceled"
    assert "0007" not in json.dumps(cancellation)


def test_orders_use_ms_date_filter_and_unified_fields():
    result = exchange()
    result.client.http = FakeHTTP([TRADE, TRADE])
    order = result.fetch_closed_orders("DEMO", 0, 1)[0]
    assert order["status"] == "closed" and order["filled"] == 10
    assert order["timestamp"] == int(datetime(2026, 10, 6, 5, tzinfo=timezone.utc).timestamp() * 1000)
    assert order["datetime"] == "2026-10-06T05:00:00Z"
    assert result.fetch_closed_orders("DEMO", order["timestamp"] + 1) == []


def test_order_lookup_and_helpers():
    result = exchange()
    result.client.http = FakeHTTP(["^0", TRADE, "^0"])
    assert result.fetch_order("DEMO_ORDER", "DEMO")["id"] == "DEMO_ORDER"
    assert result.create_limit_buy_order("DEMO", 1, 100)["side"] == "buy"
    assert result.create_limit_sell_order("DEMO", 1, 100)["side"] == "sell"
    with pytest.raises(ahl_api.InvalidOrder):
        result.create_market_buy_order("DEMO", 1)


def test_cancel_all_preserves_pin_override():
    result = exchange()
    result.client.http = FakeHTTP(["DEMO;REG;DEMO_ORDER;Oct 06, 2026 10:00:00;BUY;100;10;DEMO_HOUSE;ACC1;NOR;QUE|^1"])
    orders = result.cancel_all_orders("DEMO", {"pin": "0011"})
    assert len(orders) == 1 and orders[0]["dry_run"]
    assert "0011" not in json.dumps(orders)


def test_rate_limiter_throttles_actual_requests():
    client = seeded_client(rate_limit_ms=1000)
    client.http = FakeHTTP(["true|25|", "true|25|"])
    with (
        patch("ahl_api.client.time.monotonic", side_effect=[10, 10.2, 11]),
        patch("ahl_api.client.time.sleep") as sleep,
    ):
        client.fetch_balance()
        client.fetch_balance()
        assert sleep.call_args.args[0] == pytest.approx(0.8)


def test_unsupported_capabilities_and_invalid_params_fail_explicitly():
    result = exchange()
    for method in (result.fetch_order_book, result.fetch_trades, result.fetch_my_trades):
        with pytest.raises(NotSupported):
            method("DEMO")
    with pytest.raises(NotSupported):
        result.create_order("DEMO", "limit", "buy", 1, 100, {"postOnly": True})
    with pytest.raises(NotSupported):
        ahl_api.ahl({"options": {"defaultMarket": "FUT"}})
    with pytest.raises(ValueError):
        result.fetch_open_orders(since="2026-10-07")


def test_camelcase_aliases_and_credentials_setters():
    result = exchange()
    assert result.loadMarkets() == result.load_markets()
    result.pin = "0009"
    assert result.client.pin == "0009"
    assert result.createLimitBuyOrder("DEMO", 1, 100)["symbol"] == "DEMO/PKR"
