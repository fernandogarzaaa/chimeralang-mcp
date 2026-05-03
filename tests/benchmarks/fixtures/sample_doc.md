# Authentication Service Architecture

The authentication service handles user login, token issuance, and session
management for the platform. It is built on Postgres 16 with Redis 7 as the
session cache. The primary endpoint is /auth/v2/login which accepts an email
and password and returns a signed JWT.

It is worth noting that the JWT is signed using RS256 with a 4096-bit key
stored in HashiCorp Vault. Please note that the public key is rotated every
30 days; clients should fetch the current key from /auth/v2/jwks.

The login flow basically validates the email format, looks up the user in
Postgres by email, verifies the bcrypt password hash, and issues a JWT with
a 1-hour expiry. The JWT contains the user_id, email, and role claims.

Of course, failed logins are rate-limited per IP using a sliding window in
Redis. The limit is 5 attempts per minute. Beyond that, the IP is blocked
for 15 minutes and an alert is sent to PagerDuty.

The session cache stores active JWTs keyed by jti (JWT ID) so that we can
revoke sessions on logout. Logout simply deletes the jti from Redis. Token
verification checks both the signature and the Redis cache.

Needless to say, all endpoints require HTTPS. The TLS cert is provisioned
by Let's Encrypt and rotated every 60 days via cert-manager.

The service is deployed on Kubernetes in the auth-prod namespace. There are
3 replicas behind an internal ALB. Health checks hit /healthz which returns
200 if Postgres and Redis are both reachable.

Metrics are exported to Prometheus via /metrics. Key metrics are:
- auth_login_total{result="success|failure"}
- auth_login_duration_seconds
- auth_jwt_issued_total
- auth_jwt_revoked_total
- auth_rate_limit_blocks_total

Alerts in PagerDuty fire when the failure rate exceeds 10% over 5 minutes,
or when p99 login latency exceeds 500ms over 10 minutes.
