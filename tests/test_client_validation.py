"""Synthetic protocol and failure cases; no real account data or network calls."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import requests

from ahl_api import AHL, AuthenticationError, BrokerRejectedError, RiskCheckError, SessionExpiredError, TransportError
from ahl_api.client import AhlError, _safe_int, read_dotenv
from test_client import FakeHTTP, seeded_client


LOGIN = json.dumps(
    {
        "userID": "USER1",
        "userCode": "CODE1",
        "account": "ACC1",
        "FeedSessionID": "NEW_SESSION",
        "maxOrder": "3",
        "identifier": 0,
    }
)
HOLDING = "DEMO;Unapproved;10;100;110;1000;2;20;100;2;10;0;0;0;0;REG|$25;1100;20;100;2;10;1000;1100;0;"
TRADE = "DEMO;REG;DEMO_ORDER;Oct 06, 2026 10:00:00;BUY;100;1000;10;10;0;ACC1;NOR|^1"


class AuthenticationTests(unittest.TestCase):
    def test_success_and_persisted_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = AHL({"user": "USER1", "pass": "secret"}, session_state_path=Path(tmp) / "state.json")
            client.http = FakeHTTP([LOGIN])
            session = client.login()
            self.assertEqual(session.account, "ACC1")
            self.assertNotIn("NEW_SESSION", repr(session))
            self.assertEqual(json.loads((Path(tmp) / "state.json").read_text())["accounts"]["ACC1"]["max_order"], 3)

    def test_login_failures_clear_old_session(self):
        for body in [
            "bad json",
            "{}",
            '{"identifier":6}',
            '{"userID":"U","userCode":"C","account":"A","FeedSessionID":""}',
        ]:
            with self.subTest(body=body):
                client = seeded_client()
                client.http = FakeHTTP([body])
                with self.assertRaises(AuthenticationError):
                    client.login()
                self.assertIsNone(client.session)

    def test_missing_credentials(self):
        with self.assertRaises(AuthenticationError):
            AHL().login()

    def test_read_retry_uses_new_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = seeded_client(session_state_path=Path(tmp) / "state.json")
            client.http = FakeHTTP(["Session expired", LOGIN, "true|25|"])
            self.assertEqual(client.fetch_balance()["cash"], 25)
            self.assertEqual(client.http.calls[-1]["params"]["SESSION_ID"], "NEW_SESSION")

    def test_second_expiry_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = seeded_client(session_state_path=Path(tmp) / "state.json")
            client.http = FakeHTTP(["Session expired", LOGIN, "Invalid session"])
            with self.assertRaises(SessionExpiredError):
                client.fetch_balance()
            self.assertEqual(len(client.http.calls), 3)

    def test_mutations_are_never_replayed(self):
        for method in ("create_order", "cancel_order"):
            client = seeded_client(dry_run=False)
            client.http = FakeHTTP(["Session expired"])
            with self.assertRaises(SessionExpiredError):
                if method == "create_order":
                    client.create_order("DEMO", "buy", 1, price=100)
                else:
                    client.cancel_order("DEMO_ORDER")
            self.assertEqual(len(client.http.calls), 1)

    def test_network_error_does_not_expose_credentials(self):
        client = seeded_client()
        client.http = Mock()
        client.http.get.side_effect = requests.ConnectionError(
            "https://example.invalid/?password=secret&SESSION_ID=token"
        )
        with self.assertRaises(TransportError) as caught:
            client.fetch_balance()
        self.assertNotIn("secret", str(caught.exception))
        self.assertNotIn("token", str(caught.exception))

    def test_http_error_is_wrapped(self):
        client = seeded_client()
        client.http = Mock()
        client.http.get.return_value.raise_for_status.side_effect = requests.HTTPError("password=secret")
        with self.assertRaises(TransportError):
            client.fetch_balance()

    def test_error_body_is_not_success(self):
        client = seeded_client()
        client.http = FakeHTTP(["Error -- invalid/incomplete parameters"])
        with self.assertRaises(BrokerRejectedError):
            client.fetch_account_statement("2026-10-01", "2026-10-07")


class AccountReadTests(unittest.TestCase):
    def test_portfolio_reconciles(self):
        client = seeded_client()
        client.http = FakeHTTP([HOLDING])
        result = client.fetch_portfolio()
        self.assertEqual(result["positions"][0]["quantity"], 10)
        self.assertEqual(result["positions"][0]["market_value"], 1100)
        self.assertEqual(result["summary"]["net_worth"], 1125)
        self.assertEqual(
            result["positions"][0]["market_value"] - result["positions"][0]["cost_basis"],
            result["summary"]["unrealized_pnl"],
        )

    def test_empty_portfolio_preserves_broker_zero(self):
        client = seeded_client()
        client.http = FakeHTTP(["$0;0;0;0;0;0;0;0;0;"])
        self.assertEqual(client.fetch_portfolio()["positions"], [])

    def test_unrecognized_portfolio_raises(self):
        client = seeded_client()
        client.http = FakeHTTP(["not a portfolio"])
        with self.assertRaises(AhlError):
            client.fetch_portfolio()

    def test_accounts_nested_json(self):
        client = seeded_client()
        client.http = FakeHTTP([json.dumps([json.dumps(["ACC1", "Demo Account"])])])
        self.assertEqual(client.fetch_accounts()[0]["id"], "ACC1")

    def test_buying_power_fields(self):
        client = seeded_client()
        client.http = FakeHTTP(["1000|2000|"])
        self.assertEqual(client.fetch_buying_power(), {"regular": 1000.0, "future": 2000.0, "raw": "1000|2000|"})

    def test_exposure_shapes(self):
        client = seeded_client()
        client.http = FakeHTTP(['["demo"]', "REG;Cash;25|"])
        self.assertEqual(client.fetch_exposure()["parsed"], ["demo"])
        self.assertIn("markets", client.fetch_exposure_by_market())

    def test_statement_parameters(self):
        client = seeded_client()
        client.http = FakeHTTP(["[]"])
        self.assertEqual(client.fetch_account_statement("2026-10-01", "2026-10-07", pin="1234")["parsed"], [])
        self.assertEqual(client.http.calls[0]["params"]["pincode"], "1234")

    def test_closed_orders_actually_filter_dates(self):
        for since, until, count in [("2026-10-01", "2026-10-07", 1), ("2026-10-07", None, 0), (None, "2026-10-05", 0)]:
            client = seeded_client()
            client.http = FakeHTTP([TRADE])
            self.assertEqual(len(client.fetch_closed_orders(since=since, until=until)), count)

    def test_cancel_all_dry_run(self):
        client = seeded_client()
        client.fetch_open_orders = Mock(return_value=[{"id": "DEMO_ORDER"}])
        result = client.cancel_all_orders()
        self.assertTrue(result[0]["dry_run"])

    def test_fetch_order_fallback(self):
        client = seeded_client()
        client.http = FakeHTTP(["^0", "^0", "^0", "BUY|REG|limit|DEMO|10|100|ACC1|123|"])
        self.assertEqual(client.fetch_order("123")["symbol"], "DEMO")


class MarketReadTests(unittest.TestCase):
    def test_settings_market_status_symbols_and_cap(self):
        client = seeded_client()
        client.http = FakeHTTP(["|".join(["demo"] * 62), "CLOSE", '[{"symbol":"DEMO"}]', "1000|100000|"])
        self.assertEqual(client.settings()["field_count"], 62)
        self.assertEqual(client.fetch_market_status()["status"], "CLOSE")
        self.assertEqual(client.fetch_symbols()[0]["symbol"], "DEMO")
        self.assertEqual(client.fetch_market_cap("DEMO")["shares"], 1000)

    def test_ticker_cap_and_no_cap(self):
        feed = json.dumps(
            {
                "feedString": "DEMO;1;100;2;110;105;10:00;100;104;111;99;5;500;10;|MKTSTATUSOpen",
                "capObject": {"lowerLocked": 90, "upperCapped": 120},
            }
        )
        for with_cap in (True, False):
            client = seeded_client()
            client.http = FakeHTTP([feed])
            self.assertEqual(client.fetch_ticker("DEMO", with_market_cap=with_cap)["last"], 105)

    def test_movers_both_endpoints(self):
        client = seeded_client()
        client.http = FakeHTTP(["DEMO;REG|", "DEMO;REG|"])
        self.assertIn("groups", client.fetch_movers())
        self.assertIn("groups", client.fetch_movers(with_feed=True))

    def test_historical_bounds_and_invalid_data(self):
        client = seeded_client()
        for kwargs in ({"years": 0}, {"years": 1.5}, {"since": "2026-10-07", "until": "2026-10-01"}):
            with self.assertRaises(ValueError):
                client.fetch_historical_ohlcv("DEMO", **kwargs)
        for body in ("<html>not data</html>", '{"status":0,"data":[]}'):
            client.http = FakeHTTP([body])
            with self.assertRaises(AhlError):
                client.fetch_historical_ohlcv("DEMO")

    def test_historical_date_is_utc(self):
        client = seeded_client()
        client.http = FakeHTTP(['{"status":1,"data":[[1767225600,110,10,100]]}'])
        self.assertEqual(client.fetch_historical_daily("DEMO")[0]["date"], "2026-01-01")


class RiskTests(unittest.TestCase):
    def test_market_invalid_price_and_empty_symbol_rejected(self):
        client = seeded_client(allow_market_orders=True)
        with self.assertRaises(RiskCheckError):
            client.create_order("DEMO", "buy", 1, price="nan", order_type="market")
        with self.assertRaises(RiskCheckError):
            client.create_order(" ", "buy", 1, price=100)

    def test_stop_order_value_guard_uses_limit_price(self):
        client = seeded_client(max_order_value=150)
        with self.assertRaises(RiskCheckError):
            client.create_order("DEMO", "buy", 1, price=100, order_type="stop_loss", limit_price=200)

    def test_invalid_quantities_do_not_submit(self):
        for quantity in (-1, 0, 1.5, "2.1", "nan", "inf", True):
            client = seeded_client(dry_run=False)
            client.http = FakeHTTP([])
            with self.assertRaises(RiskCheckError):
                client.create_order("DEMO", "buy", quantity, price=100)
            self.assertFalse(client.http.calls)

    def test_invalid_prices_do_not_submit(self):
        for price in (0, -1, float("nan"), float("inf"), "abc"):
            client = seeded_client(dry_run=False)
            client.http = FakeHTTP([])
            with self.assertRaises(RiskCheckError):
                client.create_order("DEMO", "buy", 1, price=price)
            self.assertFalse(client.http.calls)

    def test_symbol_quantity_value_and_short_guards(self):
        for config, args in [
            ({"allowed_symbols": ["OTHER"]}, {}),
            ({"blocked_symbols": ["DEMO"]}, {}),
            ({"max_order_quantity": 1}, {"amount": 2}),
            ({"max_order_value": 50}, {}),
            ({}, {"side": "short_sell"}),
        ]:
            client = seeded_client(**config)
            call = {"symbol": "DEMO", "side": "buy", "amount": 1, "price": 100, **args}
            with self.assertRaises(RiskCheckError):
                client.create_order(**call)

    def test_missing_buying_power_fails_closed(self):
        client = seeded_client(dry_run=False, check_buying_power=True)
        client.http = FakeHTTP(["bad data"])
        with self.assertRaises(RiskCheckError):
            client.create_order("DEMO", "buy", 1, price=100)
        self.assertEqual(len(client.http.calls), 1)

    def test_insufficient_buying_power_fails_closed(self):
        client = seeded_client(dry_run=False, check_buying_power=True)
        client.http = FakeHTTP(["10|0|"])
        with self.assertRaises(RiskCheckError):
            client.create_order("DEMO", "buy", 1, price=100)

    def test_price_bands_guard_and_missing_data(self):
        for cap, price in [
            (None, 100),
            ({"lowerLocked": 90, "upperCapped": 110}, 80),
            ({"lowerLocked": 90, "upperCapped": 110}, 120),
        ]:
            client = seeded_client(dry_run=False, check_price_bands=True)
            client.fetch_ticker = Mock(return_value={"cap": cap})
            client.http = FakeHTTP([])
            with self.assertRaises(RiskCheckError):
                client.create_order("DEMO", "buy", 1, price=price)
            self.assertFalse(client.http.calls)

    def test_market_value_guard_uses_quote(self):
        client = seeded_client(dry_run=False, allow_market_orders=True, max_order_value=50)
        client.fetch_ticker = Mock(return_value={"cap": {"upperCapped": 100}})
        client.http = FakeHTTP([])
        with self.assertRaises(RiskCheckError):
            client.create_order("DEMO", "buy", 1, order_type="market")
        self.assertFalse(client.http.calls)

    def test_unknown_order_response_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = seeded_client(dry_run=False, session_state_path=Path(tmp) / "state.json")
            client.http = FakeHTTP(["unrecognized reply"])
            result = client.create_order("DEMO", "buy", 1, price=100)
            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "unknown")

    def test_cancel_live_and_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for body, ok, status in [("Cancelled", True, "cancelled"), ("cannot cancel", False, "rejected")]:
                client = seeded_client(dry_run=False, session_state_path=Path(tmp) / "state.json")
                client.http = FakeHTTP([body])
                result = client.cancel_order("DEMO_ORDER")
                self.assertEqual(result["ok"], ok)
                self.assertEqual(result["status"], status)

    def test_bad_limits_rejected(self):
        for config in ({"max_order_value": float("nan")}, {"max_order_quantity": 1.5}, {"timeout": 0}):
            with self.assertRaises((ValueError, RiskCheckError)):
                AHL(**config)


class PrivacyTests(unittest.TestCase):
    def test_audit_redacts_json_url_and_configured_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = seeded_client(audit_enabled=True, audit_dir=tmp)
            client.password = "secret-login"
            client.pin = "secret-pin"
            client._audit(
                "test",
                request={"url": "https://example.invalid/?password=secret-login&pin=secret-pin&SESSION_ID=SESSION1"},
                response='{"Password":"secret-login","FeedSessionID":"SESSION1"}',
            )
            data = next(Path(tmp).glob("*.jsonl")).read_text()
            for secret in ("secret-login", "secret-pin", "SESSION1"):
                self.assertNotIn(secret, data)
            self.assertIn("[REDACTED]", data)
            json.loads(data)

    def test_dotenv_and_session_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / ".env"
            p.write_text("# comment\nuser=\"DEMO\"\npass='secret'\n", encoding="utf-8")
            self.assertEqual(read_dotenv(p), {"user": "DEMO", "pass": "secret"})
            self.assertEqual(read_dotenv(Path(tmp) / "missing"), {})
        client = AHL()
        client.http = Mock()
        with client:
            pass
        client.http.close.assert_called_once()

    def test_nonfinite_integer_is_unavailable(self):
        self.assertIsNone(_safe_int("inf"))
