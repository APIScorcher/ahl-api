"""AHL credentials are username/password/PIN rather than crypto API keys."""

import os

import ahl_api

with ahl_api.ahl(
    {
        "username": os.environ["AHL_USERNAME"],
        "password": os.environ["AHL_PASSWORD"],
        "pin": os.environ.get("AHL_PIN"),
        "enableRateLimit": True,
        "options": {"dryRun": True, "maxOrderValue": 50_000},
    }
) as exchange:
    exchange.load_markets()
    print(exchange.fetch_ticker("OGDC/PKR"))
    print(exchange.fetch_balance())
    preview = exchange.create_order("OGDC/PKR", "limit", "buy", 1, 300)
    assert preview["dry_run"]
