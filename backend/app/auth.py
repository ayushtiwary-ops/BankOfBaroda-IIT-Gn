"""API authentication + per-caller scopes.

SECURITY: every endpoint requires a signed API key mapped to
explicit scopes (``events:write``, ``audit:read``, ``identity:read``,
``stepup:write``). Unauthenticated calls are 401; authenticated-but-unscoped
calls are 403. Keys are compared in constant time (``hmac.compare_digest``) to
avoid a timing side-channel on key existence.

Models OAuth2 client-credentials at the prototype scale (each channel gateway /
SOC tool holds its own scoped key, issued from KMS); the same dependency swaps
to JWT/mTLS verification in production without touching the routes.
"""
from __future__ import annotations

import hmac
from dataclasses import dataclass

from fastapi import Header, HTTPException


@dataclass(frozen=True)
class AuthContext:
    client_id: str
    scopes: frozenset[str]


def make_auth(settings):
    """Build the authenticate + require-scope dependencies bound to a config."""
    api_keys: dict[str, frozenset[str]] = settings.api_keys

    def authenticate(x_api_key: str | None = Header(default=None, alias="X-API-Key")
                    ) -> AuthContext:
        if not x_api_key:
            raise HTTPException(status_code=401, detail="missing API key")
        matched: AuthContext | None = None
        for key, scopes in api_keys.items():
            # compare ALL keys (no early return) → constant-ish time, no oracle
            if hmac.compare_digest(key, x_api_key):
                matched = AuthContext(client_id=key, scopes=scopes)
        if matched is None:
            raise HTTPException(status_code=401, detail="invalid API key")
        return matched

    def require(scope: str):
        def _dep(x_api_key: str | None = Header(default=None, alias="X-API-Key")
                ) -> AuthContext:
            ctx = authenticate(x_api_key)
            if scope not in ctx.scopes:
                raise HTTPException(status_code=403, detail=f"missing scope: {scope}")
            return ctx

        return _dep

    return authenticate, require
