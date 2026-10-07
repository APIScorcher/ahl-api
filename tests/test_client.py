from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ahl_api.client import AHL, AhlSession, RiskCheckError


class FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeHTTP:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, params=None, timeout: int = 30):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if not self.responses:
            raise AssertionError("No fake response queued")
        return FakeResponse(self.responses.pop(0))


def seeded_client(**kwargs) -> AHL:
    kwargs.setdefault("audit_enabled", False)
    kwargs.setdefault("check_buying_power", False)
    kwargs.setdefault("check_price_bands", False)
    client = AHL(
        {"user": "U", "pass": "P"},
        **kwargs,
    )
    client.session = AhlSession(
        user_id="USER1",
        user_code="CODE1",
        account="ACC1",
        feed_session_id="SESSION1",
        max_order=10,
        raw={},
    )
    return client


class ClientModeTests(unittest.TestCase):
    def test_order_uses_client_dry_run_default(self) -> None:
        client = seeded_client()
        result = client.create_order("HBL", "buy", 2, price=298.0)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["status"], "dry_run")
        self.assertIn("[REDACTED]", result["request"]["url"])

    def test_live_order_uses_raw_url_and_advances_max_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = seeded_client(
                dry_run=False,
                audit_enabled=True,
                audit_dir=Path(tmp) / "audit",
                session_state_path=Path(tmp) / "session_state.json",
            )
            fake_http = FakeHTTP(["Order has been sent to Trade Server."])
            client.http = fake_http
            result = client.create_order("HBL", "buy", 2, price=298.0)
            self.assertFalse(result["dry_run"])
            self.assertEqual(result["status"], "submitted")
            self.assertEqual(client.session.max_order, 11)
            self.assertIn("SESSION_ID=SESSION1", str(fake_http.calls[0]["url"]))
            self.assertNotIn("[REDACTED]", str(fake_http.calls[0]["url"]))
            self.assertTrue((Path(tmp) / "session_state.json").exists())

    def test_live_order_uses_default_pin_from_config(self) -> None:
        client = seeded_client(dry_run=False)
        client.pin = "2468"
        fake_http = FakeHTTP(["Order has been sent to Trade Server."])
        client.http = fake_http
        client.create_order("HBL", "buy", 2, price=298.0)
        self.assertIn("pin=2468", str(fake_http.calls[0]["url"]))

    def test_market_order_uses_upper_band_for_buying_power_check(self) -> None:
        client = seeded_client(dry_run=False, allow_market_orders=True, check_buying_power=True, check_price_bands=True)
        fake_http = FakeHTTP(
            [
                '{"feedString":"PAEL;0;0.0;0;0.0;43.0;13:34:21;42.75;43.08;43.63;42.8;0.25;7249497;100;|MKTSTATUSOpenMarketCap","capObject":{"symbol":"PAEL","market":"REG","upperCapped":"47.03","lowerLocked":"38.48"}}',
                "1000.0|0.0|\r\n",
                "Order has been sent to Trade Server.",
            ]
        )
        client.http = fake_http
        result = client.create_order("PAEL", "buy", 10, order_type="market")
        self.assertEqual(result["status"], "submitted")
        self.assertIn("GetSingleFeedWithMarketCap", str(fake_http.calls[0]["url"]))
        self.assertIn("GetAccountBuyingPowers", str(fake_http.calls[1]["url"]))
        self.assertIn("order?", str(fake_http.calls[2]["url"]))

    def test_market_orders_blocked_by_default(self) -> None:
        client = seeded_client()
        with self.assertRaises(RiskCheckError):
            client.create_order("HBL", "buy", 1, order_type="market")

    def test_cancel_uses_client_dry_run_default(self) -> None:
        client = seeded_client()
        result = client.cancel_order("123")
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["status"], "dry_run")
        self.assertIn("[REDACTED]", result["request"]["url"])


class ParserTests(unittest.TestCase):
    def test_fetch_tickers_parses_multiple_feed_entries(self) -> None:
        client = seeded_client()
        client.http = FakeHTTP(
            [
                '{"feedString":"HBL;0;0.0;0;0.0;298.02;16:47:11;298.4;299.79;305.0;294.0;-6.02;2326748;87;|OGDC;0;0.0;0;0.0;332.74;16:49:15;331.28;333.8;339.49;329.0;-5.92;9526189;100;|MKTSTATUSOHO"}'
            ]
        )
        tickers = client.fetch_tickers(["HBL", "OGDC"])
        self.assertEqual(tickers["HBL"]["last"], 298.02)
        self.assertEqual(tickers["OGDC"]["high"], 339.49)

    def test_fetch_ohlcv_uses_virtual_url(self) -> None:
        client = seeded_client()
        fake_http = FakeHTTP(["HBL;REG;304.42;304.42;304.0;304.0;09:21|"])
        client.http = fake_http
        candles = client.fetch_ohlcv("HBL")
        self.assertEqual(candles[0], ["09:21", 304.42, 304.42, 304.0, 304.0, None])
        self.assertIn("virtualtrading", str(fake_http.calls[0]["url"]))

    def test_fetch_historical_ohlcv_parses_psx_eod_rows(self) -> None:
        client = seeded_client()
        fake_http = FakeHTTP(
            [
                '{"status":1,"message":"","data":[[1781866800,331.28,9526189,339],[1624446490,99.2,10269396,99.11]]}'
            ]
        )
        client.http = fake_http
        candles = client.fetch_historical_ohlcv("OGDC", since="2021-01-01", until="2026-12-31")
        self.assertEqual(candles[0], [1624446490000, 99.11, None, None, 99.2, 10269396])
        self.assertEqual(candles[-1], [1781866800000, 339.0, None, None, 331.28, 9526189])
        self.assertIn("dps.psx.com.pk", str(fake_http.calls[0]["url"]))

    def test_fetch_historical_daily_returns_dicts(self) -> None:
        client = seeded_client()
        client.http = FakeHTTP(['{"status":1,"message":"","data":[[1624446490,99.2,10269396,99.11]]}'])
        rows = client.fetch_historical_daily("OGDC")
        self.assertEqual(rows[0]["open"], 99.11)
        self.assertEqual(rows[0]["close"], 99.2)
        self.assertEqual(rows[0]["volume"], 10269396)

    def test_cancel_info_parser_shape(self) -> None:
        client = seeded_client()
        client.http = FakeHTTP(["BUY|REG|limit|HBL|10|298.0|ACC1|123|"])
        info = client.fetch_order_cancel_info("123")
        self.assertEqual(info["side"], "buy")
        self.assertEqual(info["symbol"], "HBL")
        self.assertEqual(info["amount"], 10)
        self.assertEqual(info["id"], 123)

    def test_fetch_closed_orders_uses_android_trade_log_servlet(self) -> None:
        client = seeded_client()
        client.http = FakeHTTP(
            ["OGDC;REG;DEMO_ORDER_001;Jun 22, 2026 09:57:54;BUY;333.98;50097.0000;150;150;0;ACC1;NOR|^1\r\n"]
        )
        orders = client.fetch_closed_orders("OGDC")
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["id"], "DEMO_ORDER_001")
        self.assertEqual(orders[0]["symbol"], "OGDC")
        self.assertEqual(orders[0]["side"], "buy")
        self.assertEqual(orders[0]["price"], 333.98)
        self.assertEqual(orders[0]["filled"], 150)
        self.assertEqual(orders[0]["amount"], 150)
        self.assertEqual(orders[0]["remaining"], 0)
        self.assertEqual(orders[0]["status"], "closed")
        self.assertEqual(client.http.calls[0]["params"]["logname"], "trade")
        self.assertEqual(client.http.calls[0]["params"]["recordSize"], 15000)

    def test_fetch_activity_logs_parses_queue_and_trade_events(self) -> None:
        client = seeded_client()
        client.http = FakeHTTP(
            [
                "OGDC;REG;Jun 22, 2026 09:57:54;BUY;DEMO_ORDER_001;DEMO_HOUSE_001;TRD;333.98;150;150;0;ACC1;NOR|"
                "OGDC;REG;Jun 22, 2026 09:57:32;BUY;DEMO_ORDER_001;DEMO_HOUSE_001;QUE;333.98;150;;150;ACC1;NOR|^2\r\n"
            ]
        )
        events = client.fetch_activity_logs("OGDC")
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["action"], "traded")
        self.assertEqual(events[0]["status"], "closed")
        self.assertEqual(events[0]["house_order_no"], "DEMO_HOUSE_001")
        self.assertEqual(events[1]["action"], "queued")
        self.assertEqual(events[1]["status"], "open")
        self.assertIsNone(events[1]["filled"])

    def test_fetch_open_orders_parses_android_outstanding_log_servlet(self) -> None:
        client = seeded_client()
        client.http = FakeHTTP(
            ["PAEL;REG;DEMO_ORDER_002;Jun 22, 2026 10:01:00;BUY;43.00;10;DEMO_HOUSE_002;ACC1;NOR;QUE|^1\r\n"]
        )
        orders = client.fetch_open_orders("PAEL")
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["id"], "DEMO_ORDER_002")
        self.assertEqual(orders[0]["remaining"], 10)
        self.assertEqual(orders[0]["amount"], 10)
        self.assertEqual(orders[0]["value"], 430.0)
        self.assertEqual(orders[0]["status"], "open")

    def test_fetch_order_searches_trade_and_activity_logs_before_cancel_info(self) -> None:
        client = seeded_client()
        client.http = FakeHTTP(
            [
                "^0\r\n",
                "OGDC;REG;DEMO_ORDER_001;Jun 22, 2026 09:57:54;BUY;333.98;50097.0000;150;150;0;ACC1;NOR|^1\r\n",
                "OGDC;REG;Jun 22, 2026 09:57:54;BUY;DEMO_ORDER_001;DEMO_HOUSE_001;TRD;333.98;150;150;0;ACC1;NOR|^1\r\n",
            ]
        )
        order = client.fetch_order("DEMO_ORDER_001")
        self.assertEqual(order["id"], "DEMO_ORDER_001")
        self.assertEqual(order["status"], "closed")
        self.assertEqual(len(order["events"]), 2)


if __name__ == "__main__":
    unittest.main()
