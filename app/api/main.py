"""HTTP API. Holds no models: every detection request becomes an encrypted arq
job, consumed by a worker. Synchronous endpoints simply wait for the job."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import uuid
from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import RedisSettings
from arq.jobs import Job, JobStatus as ArqStatus
from fastapi import Depends, FastAPI, HTTPException, Query, Response, Security
from fastapi.security import APIKeyHeader

from ..anonymize.engine import deanonymize
from ..config import get_settings
from ..models import SPECIAL_CATEGORY, TIMESHEET
from ..storage.crypto import JOB_AAD, Cipher
from ..storage.vault import Vault

from .schemas import AnalyzeRequest, AnonymizeRequest, DeanonymizeRequest, JobRequest, JobStatus

log = logging.getLogger("pii.api")
settings = get_settings()
API_KEYS = {k.strip() for k in os.environ.get("PII_API_KEYS", "").split(",") if k.strip()}
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_key(key: str | None = Security(api_key_header)) -> None:
    if not API_KEYS:
        return  # auth disabled (development); a warning is logged at startup
    if not key or not any(secrets.compare_digest(key, k) for k in API_KEYS):
        raise HTTPException(401, "invalid or missing X-API-Key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_KEYS:
        log.warning("PII_API_KEYS is empty - API authentication is DISABLED")
    app.state.cipher = Cipher.from_env(settings.vault.key_env)
    app.state.arq = await create_pool(RedisSettings.from_dsn(os.environ.get("REDIS_URL", "redis://localhost:6379/0")))
    app.state.vault = Vault(app.state.arq, app.state.cipher, settings.vault.session_ttl_seconds)
    yield
    await app.state.arq.aclose()


app = FastAPI(title="PII Service", version="0.1.0", lifespan=lifespan,
              description="Local PII detection, redaction and pseudonymization (German/English).")
auth = [Depends(require_key)]


# -- jobs -----------------------------------------------------------------------------
async def _enqueue(kind: str, req: AnalyzeRequest | AnonymizeRequest) -> Job:
    profile = req.profile or settings.default_profile
    payload = json.dumps({"kind": kind, "request": req.model_dump(mode="json")}, ensure_ascii=False)
    blob = app.state.cipher.encrypt(payload.encode(), JOB_AAD)
    job = await app.state.arq.enqueue_job("process", blob, _job_id=uuid.uuid4().hex,
                                          _queue_name=settings.queue_for(profile))
    if job is None:
        raise HTTPException(500, "could not enqueue job")
    return job


async def _status(job_id: str) -> JobStatus:
    for queue in set(settings.queues.values()) or {"pii:full"}:
        job = Job(job_id, app.state.arq, _queue_name=queue)
        status = await job.status()
        if status == ArqStatus.not_found:
            continue
        if status != ArqStatus.complete:
            return JobStatus(job_id=job_id, status=status.value)
        info = await job.result_info()
        if info is None:
            break
        if not info.success:
            return JobStatus(job_id=job_id, status="failed", error=str(info.result))
        result = json.loads(app.state.cipher.decrypt(info.result, JOB_AAD))
        return JobStatus(job_id=job_id, status="complete", result=result)
    return JobStatus(job_id=job_id, status="not_found")


async def _run(kind: str, req, wait: float, response: Response):
    job = await _enqueue(kind, req)
    deadline = asyncio.get_running_loop().time() + wait
    while True:
        st = await _status(job.job_id)
        if st.status in ("complete", "failed"):
            if st.status == "failed":
                raise HTTPException(500, f"job failed: {st.error}")
            return st.result
        if asyncio.get_running_loop().time() >= deadline:
            response.status_code = 202
            return {"job_id": job.job_id, "status": st.status, "status_url": f"/v1/jobs/{job.job_id}"}
        await asyncio.sleep(0.25)


@app.post("/v1/jobs", status_code=202, dependencies=auth, tags=["jobs"])
async def create_job(body: JobRequest):
    """Submit a long-running request. Poll GET /v1/jobs/{job_id}."""
    model = AnalyzeRequest if body.kind == "analyze" else AnonymizeRequest
    try:
        req = model.model_validate(body.request)
    except ValueError as err:
        raise HTTPException(422, str(err)) from err
    job = await _enqueue(body.kind, req)
    return {"job_id": job.job_id, "status": "queued", "status_url": f"/v1/jobs/{job.job_id}"}


@app.get("/v1/jobs/{job_id}", response_model=JobStatus, dependencies=auth, tags=["jobs"])
async def get_job(job_id: str):
    st = await _status(job_id)
    if st.status == "not_found":
        raise HTTPException(404, "job not found (unknown or expired)")
    return st


@app.delete("/v1/jobs/{job_id}", status_code=204, dependencies=auth, tags=["jobs"])
async def delete_job(job_id: str):
    """Delete a stored result (or abort a queued job)."""
    for queue in set(settings.queues.values()) or {"pii:full"}:
        job = Job(job_id, app.state.arq, _queue_name=queue)
        if await job.status() in (ArqStatus.queued, ArqStatus.deferred):
            await job.abort(timeout=0)
    await app.state.arq.delete(f"arq:result:{job_id}")


# -- synchronous convenience endpoints ------------------------------------------------
WAIT = Query(60.0, ge=0, le=600, description="Seconds to wait before answering 202 with a job id")


@app.post("/v1/anonymize", dependencies=auth, tags=["pii"])
async def anonymize(req: AnonymizeRequest, response: Response, wait: float = WAIT):
    return await _run("anonymize", req, wait, response)


@app.post("/v1/analyze", dependencies=auth, tags=["pii"])
async def analyze(req: AnalyzeRequest, response: Response, wait: float = WAIT):
    return await _run("analyze", req, wait, response)


@app.post("/v1/deanonymize", dependencies=auth, tags=["pii"])
async def deanonymize_text(req: DeanonymizeRequest):
    """Restore originals in any text containing this session's replacements."""
    state = await app.state.vault.load(req.session_id)
    if state is None:
        raise HTTPException(404, "session not found (unknown or expired)")
    text, count = deanonymize(req.text, state)
    return {"text": text, "replaced": count}


@app.delete("/v1/sessions/{session_id}", status_code=204, dependencies=auth, tags=["pii"])
async def delete_session(session_id: str):
    """Erase a session's mapping (afterwards its pseudonyms are anonymous)."""
    if not await app.state.vault.delete(session_id):
        raise HTTPException(404, "session not found")


# -- meta -----------------------------------------------------------------------------
@app.get("/v1/entities", dependencies=auth, tags=["meta"])
async def entities():
    return {
        "special_category": sorted(SPECIAL_CATEGORY),
        "timesheet": sorted(TIMESHEET),
        "modes": ["redact", "placeholder", "mask", "fake", "hash"],
        "profiles": settings.profiles,
        "timesheet_defaults": settings.timesheet.model_dump(),
    }


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}


@app.get("/ready", tags=["meta"])
async def ready(response: Response):
    """Ready when Redis answers and every queue has a live worker."""
    try:
        await app.state.arq.ping()
    except Exception as err:  # noqa: BLE001
        response.status_code = 503
        return {"redis": False, "error": str(err)}
    workers = {}
    for queue in sorted(set(settings.queues.values())):
        workers[queue] = bool(await app.state.arq.exists(f"{queue}:health-check"))
    ok = all(workers.values())
    response.status_code = 200 if ok else 503
    return {"redis": True, "workers": workers, "ready": ok}
