"""API + worker integration through a real Redis (skipped if none is running).
The worker runs in-process in burst mode with the fast layers only."""

import os

import httpx
import pytest
import redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/15")
os.environ["REDIS_URL"] = REDIS_URL


def _redis_up() -> bool:
    try:
        return redis.Redis.from_url(REDIS_URL).ping()
    except redis.RedisError:
        return False


pytestmark = pytest.mark.skipif(not _redis_up(), reason="no Redis at REDIS_URL")


@pytest.fixture
async def env():
    from arq.worker import Worker

    from app import worker as w
    from app.api.main import app

    redis.Redis.from_url(REDIS_URL).flushdb()
    os.environ["PII_WORKER_LAYERS"] = "patterns,spacy"
    os.environ["PII_QUEUE"] = "pii:fast"
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            ctx = {}
            await w.startup(ctx)

            async def drain():
                worker = Worker(functions=[w.process], redis_pool=app.state.arq, queue_name="pii:fast",
                                burst=True, ctx=ctx, max_jobs=1, poll_delay=0.05)
                await worker.main()

            yield client, drain
            await w.shutdown(ctx)


async def submit_and_run(client, drain, path, body):
    r = await client.post(path + "?wait=0", json=body)
    assert r.status_code == 202, r.text
    await drain()
    job = await client.get(r.json()["status_url"])
    assert job.status_code == 200 and job.json()["status"] == "complete", job.text
    return job.json()["result"]


TEXT = "Max Müller (max@firma.de) war am 03.10.2026 krank. IBAN DE89 3704 0044 0532 0130 00."


async def test_anonymize_reversible_roundtrip(env):
    client, drain = env
    res = await submit_and_run(client, drain, "/v1/anonymize",
                               {"text": TEXT, "profile": "fast", "mode": "fake", "reversible": True})
    assert "Müller" not in res["text"] and "max@firma.de" not in res["text"]
    sid = res["session_id"]
    back = await client.post("/v1/deanonymize", json={"text": res["text"], "session_id": sid})
    assert back.json()["text"] == TEXT
    # erase the mapping -> no longer reversible
    assert (await client.delete(f"/v1/sessions/{sid}")).status_code == 204
    assert (await client.post("/v1/deanonymize", json={"text": "x", "session_id": sid})).status_code == 404


async def test_analyze_does_not_leak_text_by_default(env):
    client, drain = env
    res = await submit_and_run(client, drain, "/v1/analyze", {"text": TEXT, "profile": "fast"})
    assert res["stats"]["IBAN_CODE"] == 1
    assert all("text" not in e for e in res["entities"])


async def test_redis_holds_no_plaintext(env):
    client, drain = env
    await submit_and_run(client, drain, "/v1/anonymize",
                         {"text": TEXT, "profile": "fast", "mode": "placeholder", "reversible": True})
    r = redis.Redis.from_url(REDIS_URL)
    for key in r.scan_iter("*"):
        if r.type(key) == b"string":
            assert b"M\xc3\xbcller" not in r.get(key) and b"max@firma.de" not in r.get(key)


async def test_validation_errors(env):
    client, _ = env
    r = await client.post("/v1/anonymize", json={"text": "x", "mode": "redact", "reversible": True})
    assert r.status_code == 422
    r = await client.get("/v1/jobs/doesnotexist")
    assert r.status_code == 404
