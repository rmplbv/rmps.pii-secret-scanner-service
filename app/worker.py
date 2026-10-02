"""arq worker. Loads the models once and processes jobs one at a time (the
work is CPU bound). Start with:  arq app.worker.WorkerSettings

Env:
  REDIS_URL           redis://redis:6379/0
  PII_QUEUE           queue to consume (pii:fast / pii:full, see config.queues)
  PII_WORKER_LAYERS   comma list of layers to load (default: all layers of the
                      profiles routed to PII_QUEUE)
  PII_LLM_URL         override layers.llm.base_url
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from arq.connections import RedisSettings
from redis.asyncio import Redis

from .api.schemas import AnalyzeRequest, AnonymizeRequest
from .config import get_settings
from .service import PiiService
from .storage.crypto import JOB_AAD, Cipher
from .storage.vault import Vault

log = logging.getLogger("pii.worker")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")




def worker_layers(settings, queue: str) -> list[str]:
    if os.environ.get("PII_WORKER_LAYERS"):
        return [l.strip() for l in os.environ["PII_WORKER_LAYERS"].split(",") if l.strip()]
    profiles = [p for p, q in settings.queues.items() if q == queue] or list(settings.profiles)
    return sorted({l for p in profiles for l in settings.profiles[p]})


async def startup(ctx):
    settings = get_settings()
    if os.environ.get("PII_LLM_URL"):
        settings.layers.llm.base_url = os.environ["PII_LLM_URL"]
    queue = os.environ.get("PII_QUEUE", "pii:full")
    layers = worker_layers(settings, queue)
    log.info("loading layers %s for queue %s", layers, queue)
    ctx["service"] = await asyncio.to_thread(PiiService, settings, layers)
    ctx["cipher"] = Cipher.from_env(settings.vault.key_env)
    ctx["vault"] = Vault(Redis.from_url(_redis_url()), ctx["cipher"], settings.vault.session_ttl_seconds)
    log.info("worker ready")


async def shutdown(ctx):
    if "vault" in ctx:
        await ctx["vault"].redis.aclose()


async def process(ctx, blob: bytes) -> bytes:
    cipher: Cipher = ctx["cipher"]
    service: PiiService = ctx["service"]
    vault: Vault = ctx["vault"]
    payload = json.loads(cipher.decrypt(blob, JOB_AAD))
    kind, raw = payload["kind"], payload["request"]
    if kind == "analyze":
        result = await asyncio.to_thread(service.analyze, AnalyzeRequest.model_validate(raw))
    else:
        req = AnonymizeRequest.model_validate(raw)
        result = await _anonymize(service, vault, req)
    return cipher.encrypt(json.dumps(result, ensure_ascii=False).encode(), JOB_AAD)


async def _anonymize(service: PiiService, vault: Vault, req: AnonymizeRequest) -> dict:
    persist = req.reversible or req.session_id is not None
    if not persist:
        result, _ = await asyncio.to_thread(service.anonymize, req)
        return result
    if req.session_id is None:
        result, state = await asyncio.to_thread(service.anonymize, req)
        await vault.save(state)
        return result
    async with vault.lock(req.session_id, timeout=WorkerSettings.job_timeout):
        state = await vault.load(req.session_id)
        if state is None:
            from .anonymize.session import SessionState
            state = SessionState(session_id=req.session_id)
        result, state = await asyncio.to_thread(service.anonymize, req, state)
        await vault.save(state)
    return result


def _redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://localhost:6379/0")


class WorkerSettings:
    functions = [process]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(_redis_url())
    queue_name = os.environ.get("PII_QUEUE", "pii:full")
    max_jobs = 1
    job_timeout = int(os.environ.get("PII_JOB_TIMEOUT", 6 * 3600))
    keep_result = get_settings().vault.job_ttl_seconds
    max_tries = 1
    health_check_interval = 30
