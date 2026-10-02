"""Encrypted session store in Redis."""

from __future__ import annotations

from contextlib import asynccontextmanager

from redis.asyncio import Redis

from ..anonymize.session import SessionState
from .crypto import Cipher

PREFIX = "pii:session:"


class Vault:
    def __init__(self, redis: Redis, cipher: Cipher, ttl_seconds: int):
        self.redis = redis
        self.cipher = cipher
        self.ttl = ttl_seconds

    async def load(self, session_id: str) -> SessionState | None:
        blob = await self.redis.get(PREFIX + session_id)
        if blob is None:
            return None
        data = self.cipher.decrypt(blob, aad=session_id.encode())
        return SessionState.from_json(data)

    async def save(self, state: SessionState) -> None:
        blob = self.cipher.encrypt(state.to_json().encode(), aad=state.session_id.encode())
        await self.redis.set(PREFIX + state.session_id, blob, ex=self.ttl)

    async def delete(self, session_id: str) -> bool:
        return bool(await self.redis.delete(PREFIX + session_id))

    @asynccontextmanager
    async def lock(self, session_id: str, timeout: float = 4 * 3600):
        """Serialise jobs of one session so mappings are not lost by concurrent writes."""
        async with self.redis.lock(f"pii:lock:{session_id}", timeout=timeout, blocking_timeout=timeout):
            yield
