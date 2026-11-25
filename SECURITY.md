# Security policy

## Supported versions

Only the `main` branch is supported. The project is a pre-release (`0.0.1`);
there are no maintained release branches and no backported fixes.

## Reporting a vulnerability

Use GitHub private vulnerability reporting for this repository: the
**Security** tab, then **Report a vulnerability**
(<https://github.com/marcus-vreis/custody-watch/security/advisories/new>).
Please do not open a public issue for an unfixed vulnerability.

Include the affected file or route, the commit you tested, and steps to
reproduce. This is a single-maintainer research project: there is no response
time commitment, no bug bounty and no formal disclosure process.

## Threat model of the HTTP servers

This describes what the code does today, not a hardening guarantee.

**Live operator screen** (`scripts/serve.py`, `src/custody_watch/servidor.py`):

- Binds to `127.0.0.1` by default (`HOST_PADRAO`). Listening elsewhere needs an
  explicit `--host`.
- Refuses to start on a non-loopback address without TLS: `--cert` and `--key`
  are required. Only `localhost` and loopback IP literals count as local; any
  other name is treated as network, without resolving it. TLS 1.2 is the
  minimum version, and the TLS handshake has a 10 s deadline.
- Validates the `Host` header (DNS-rebinding defence) and answers `421` for
  names not in the accepted set: loopback names, the listen address and any
  `--allowed-host NAME` given (repeatable). A wildcard listen address also
  accepts IP literals.
- **There is no authentication or authorization.** TLS encrypts traffic; it does
  not identify clients (no client certificates are requested). Anyone who can
  reach the port can watch the annotated video, read `/estado`, and send the
  two POST routes (`/ancora`, `/dono`) that add operator pointing to the live
  session. No Origin or CORS handling is implemented. Do not expose this
  server to a network without an authenticating reverse proxy or equivalent.
- POST bodies are limited to 4096 bytes and must be a JSON object; click
  coordinates must be fractions in [0, 1]. Video is capped at 15 frames per
  second per tab. Beyond these, no request-rate limit exists.

**Multicamera replay laboratory** (`scripts/serve_multicamera.py`,
`src/custody_watch/multicamera_server.py`):

- Always binds to `127.0.0.1`; only `--port` is configurable. No TLS and no
  authentication, because it is meant for the local machine only.
- Validates the `Host` header like the live screen (`421`, connection closed,
  no content leaked): only `localhost`, `127.0.0.1`, `[::1]` and any
  `--allowed-host NAME` on its own port are accepted. This blocks DNS
  rebinding; it is not authentication.
- Serves fixed assets and read-only GET/HEAD routes; query parameters must match
  an exact expected set. Responses carry `nosniff`, `no-referrer` and a
  `default-src 'self'` Content-Security-Policy.

Out of scope: the security of the machine running the software, of the video
sources, and of any proxy you place in front of it. Dependencies are locked in
`uv.lock` but have not been audited for this document.
