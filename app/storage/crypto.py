"""AES-256-GCM encryption for everything that is written to Redis (job
payloads, results and session mappings all contain PII)."""

from __future__ import annotations

import base64
import hashlib
import logging
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

log = logging.getLogger(__name__)

JOB_AAD = b"pii-job"  # associated data binding ciphertexts to their purpose


class Cipher:
    def __init__(self, key: bytes):
        if len(key) != 32:
            raise ValueError("vault key must be 32 bytes")
        self._aes = AESGCM(key)

    @classmethod
    def from_env(cls, var: str = "PII_VAULT_KEY") -> "Cipher":
        raw = os.environ.get(var)
        if raw:
            return cls(base64.b64decode(raw))
        if os.environ.get("PII_INSECURE_DEV") == "1":
            log.warning("%s not set - using an INSECURE development key", var)
            return cls(hashlib.sha256(b"pii-service-insecure-dev-key").digest())
        raise RuntimeError(f"{var} is not set (generate one with: openssl rand -base64 32)")

    def encrypt(self, data: bytes, aad: bytes = b"") -> bytes:
        nonce = os.urandom(12)
        return nonce + self._aes.encrypt(nonce, data, aad)

    def decrypt(self, blob: bytes, aad: bytes = b"") -> bytes:
        return self._aes.decrypt(blob[:12], blob[12:], aad)
