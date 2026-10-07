# Security and privacy

Report vulnerabilities privately through GitHub private vulnerability reporting when enabled, or contact the repository owner before sharing details publicly. Do not include account credentials or broker responses in issues.

The SDK authenticates only to the configured AHL host. Review a custom host before supplying credentials. Read-only calls can expose personal account information in their return values or terminal output. Do not post those outputs publicly.

Audit logging is opt-in. Passwords, PINs, session tokens, and credential-bearing query strings are redacted. Other account information is retained; store audit files privately. Older local research logs may be unredacted and are never shipped in the repository or package.

The HTTP-based virtual intraday endpoint receives no account credentials. Authenticated calls use HTTPS by default. Do not disable TLS verification. Local session state stores account IDs and order counters, not passwords or session tokens.

Git and package builds exclude `.env`, runtime logs, account snapshots, historical datasets, broker APKs, and extracted broker resources. Tests contain synthetic fixtures. CI does not authenticate to AHL and requires no broker secrets.
