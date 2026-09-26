# Security policy

## Reporting a vulnerability

Report privately through
[GitHub security advisories](https://github.com/fernandogarzaaa/chimeralang-mcp/security/advisories/new).
Please do not open a public issue for a vulnerability.

Include what you would need to reproduce it yourself: version or commit,
the tool or operation involved, and a minimal case. You should get an
initial response within a week.

## Supported versions

Only the latest published minor release receives security fixes.
chimeralang-mcp is pre-1.0, so older minors are not patched — upgrade
first, then report if the issue persists.

## Scope notes

A few things about chimeralang-mcp's design are worth knowing before
reporting:

- **Certificates carry trust claims.** ChimeraLang issues tamper-evident
  certificates with HMAC/Ed25519 signatures. The offline verifier is the
  trust boundary: keep signing keys out of repositories and logs, and
  treat any certificate that fails verification as untrusted.
- **The MCP server executes locally.** It exposes ChimeraLang's
  reasoning tools to whatever agent it is wired into. Configure it the
  way you would any local tool server, and do not expose it to a network.
- **Confidence scores are advisory.** The uncertainty annotations are
  calibrated estimates, not guarantees. Downstream decisions that treat
  a high-confidence output as infallible are a usage error, not a
  vulnerability — but a systematic miscalibration is worth reporting.
