import io
import json
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import pytest

from ahl_api import AuthenticationError
from ahl_api.cli import main


@pytest.mark.parametrize(
    "command,method",
    [
        ("portfolio", "fetch_portfolio"),
        ("balance", "fetch_balance"),
        ("accounts", "fetch_accounts"),
        ("market-status", "fetch_market_status"),
    ],
)
def test_read_commands(command, method):
    with patch("ahl_api.cli.AHL") as factory:
        client = factory.return_value.__enter__.return_value
        getattr(client, method).return_value = {"demo": True}
        output = io.StringIO()
        with redirect_stdout(output):
            assert main([command]) == 0
        assert json.loads(output.getvalue()) == {"demo": True}
        getattr(client, method).assert_called_once_with()
        assert "dry_run" not in factory.call_args.kwargs


def test_ticker_requires_symbol():
    with pytest.raises(SystemExit) as error:
        main(["ticker"])
    assert error.value.code == 2


def test_ticker_uses_environment_credentials(monkeypatch):
    monkeypatch.setenv("AHL_USERNAME", "DEMO_USER")
    monkeypatch.setenv("AHL_PASSWORD", "DEMO_PASSWORD")
    with patch("ahl_api.cli.AHL") as factory, redirect_stdout(io.StringIO()):
        factory.return_value.__enter__.return_value.fetch_ticker.return_value = {"last": 100}
        assert main(["ticker", "DEMO"]) == 0
        assert factory.call_args.args[0] == {"user": "DEMO_USER", "pass": "DEMO_PASSWORD"}
        factory.return_value.__enter__.return_value.fetch_ticker.assert_called_once_with("DEMO")


def test_auth_failure_has_nonzero_exit():
    with patch("ahl_api.cli.AHL") as factory:
        factory.return_value.__enter__.return_value.fetch_balance.side_effect = AuthenticationError(
            "Missing credentials"
        )
        errors = io.StringIO()
        with redirect_stderr(errors):
            assert main(["balance"]) == 1
        assert "Missing credentials" in errors.getvalue()
