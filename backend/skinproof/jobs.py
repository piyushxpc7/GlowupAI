from __future__ import annotations

import json
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def dump(value) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def load(value, default=None):
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default if default is not None else {}


class JobRunner:
    """Minimal background job queue backed by a thread pool.

    Requests that would otherwise block on model latency (reprocessing,
    shelf-scan extraction) submit here instead: the row is written
    synchronously so the caller gets a job id immediately, and the work runs
    off the request thread. This is a stand-in for a real worker process
    (Celery/RQ against Postgres/Redis) that keeps the same job-row contract,
    so swapping the executor later does not change the service or API layer.
    """

    def __init__(self, db, max_workers: int = 4) -> None:
        self.db = db
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="skinproof-job")

    def submit(self, job_type: str, fn, *args, user_id: str | None = None, payload: dict | None = None) -> str:
        job_id = str(uuid.uuid4())
        self.db.execute(
            "INSERT INTO jobs (id, user_id, job_type, payload_json) VALUES (?, ?, ?, ?)",
            (job_id, user_id, job_type, dump(payload or {})),
        )
        self._executor.submit(self._run, job_id, fn, args)
        return job_id

    def _run(self, job_id: str, fn, args) -> None:
        self.db.execute("UPDATE jobs SET status='running', started_at=? WHERE id=?", (now_iso(), job_id))
        try:
            result = fn(*args)
            self.db.execute(
                "UPDATE jobs SET status='completed', result_json=?, completed_at=? WHERE id=?",
                (dump(result), now_iso(), job_id),
            )
        except Exception as exc:
            self.db.execute(
                "UPDATE jobs SET status='failed', error=?, completed_at=? WHERE id=?",
                (f"{exc}\n{traceback.format_exc(limit=3)}", now_iso(), job_id),
            )

    def get(self, job_id: str, user_id: str | None = None) -> dict | None:
        row = self.db.fetchone("SELECT * FROM jobs WHERE id=?", (job_id,))
        if not row:
            return None
        row = dict(row)
        if user_id is not None and row.get("user_id") not in (None, user_id):
            return None
        row["payload"] = load(row.pop("payload_json"), {})
        result_json = row.pop("result_json")
        row["result"] = load(result_json, None) if result_json is not None else None
        return row

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
