"""Trusted step-up verifier service - holds the PRIVATE key.

This is the trust boundary made into a container: the verifier (OTP/WebAuthn/
video-KYC provider) holds the Ed25519 PRIVATE key and is the ONLY component that
can mint a step-up assertion. The scoring pods hold only the matching PUBLIC
key, so they can verify an assertion but can never forge one.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from .verifier import Ed25519Signer, TrustedVerifier

_signer = Ed25519Signer.from_private_b64(os.environ["PRAMAAN_STEPUP_PRIVKEY"])
_verifier = TrustedVerifier(_signer)

app = FastAPI(title="PRAMAAN - Step-up Verifier", version="3.0.0")


class IssueRequest(BaseModel):
    challenge_id: str
    identity_id: str
    method: str = "otp_sms"
    result: str = "pass"          # the verifier's verdict (modelled OTP/KYC outcome)


@app.get("/health")
def health():
    # expose ONLY the public key - never the private one
    return {"status": "ok", "service": "verifier", "public_key": _signer.public_key_b64}


@app.post("/issue")
def issue(req: IssueRequest):
    return {"assertion": _verifier.issue(
        challenge_id=req.challenge_id, identity_id=req.identity_id,
        method=req.method, result=req.result)}
