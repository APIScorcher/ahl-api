"""Run after setting AHL_USERNAME and AHL_PASSWORD in the environment."""
import json
import os

from ahl_api import AHL

with AHL({"user": os.environ["AHL_USERNAME"], "pass": os.environ["AHL_PASSWORD"]}) as client:
    portfolio = client.fetch_portfolio()
    print(json.dumps({"positions": portfolio["positions"], "summary": portfolio["summary"]}, indent=2))
