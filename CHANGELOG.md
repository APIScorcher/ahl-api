# Changelog

## 0.4.0 — 2026-10-07

- Add a standalone `ahl`/`Exchange` facade with markets, unified symbols, standard signatures/params, normalized responses, aliases, and paced requests.
- Preserve the `AHL` core interface and make live trading PIN checks fail before HTTP requests.
- Support `AHL_PIN`, per-call PIN overrides, leading-zero PINs, and statement PIN fallback.
- Keep unknown free balances, fill states, fees, and timestamps unset; declare unsupported capabilities explicitly.
- Interpret broker request dates and intraday/session timestamps in Pakistan time.

## 0.3.0 — 2026-10-07

- Add installable wheel/sdist metadata, optional research dependencies, read-only CLI, documentation, and CI.
- Add structured portfolio positions and summaries while retaining raw/legacy fields.
- Fix read-only session refresh to use the new token; avoid replaying order mutations.
- Reject fractional quantities, nonfinite prices, invalid risk limits, and unknown broker success responses.
- Fail closed when enabled broker price-band or buying-power checks cannot be completed.
- Apply closed-order date filters and use UTC for historical dates.
- Disable audit logging by default, redact credentials in opt-in logs, and sanitize transport errors.
- Anonymize protocol test fixtures and document market-hours and live-validation limits.
