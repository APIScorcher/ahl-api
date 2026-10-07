"""The release test suite must never reach a broker or public market-data host."""

import pytest
import requests


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Tests must mock HTTP instead of making network requests")

    monkeypatch.setattr(requests.sessions.Session, "request", denied)
