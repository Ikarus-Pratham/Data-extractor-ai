import os
import json
import time
import uuid
import errno
from typing import Optional, Tuple, Any, Dict


class FileQueue:
    """
    Minimal file-backed FIFO queue using JSONL for jobs and a lock file.

    Directory layout:
      queue_dir/
        jobs.jsonl
        processing.jsonl
        completed/
        failed/
        lock
    """

    def __init__(self, queue_dir: str = "queue_data") -> None:
        self.queue_dir = queue_dir
        self.jobs_path = os.path.join(queue_dir, "jobs.jsonl")
        self.processing_path = os.path.join(queue_dir, "processing.jsonl")
        self.completed_dir = os.path.join(queue_dir, "completed")
        self.failed_dir = os.path.join(queue_dir, "failed")
        self.lock_path = os.path.join(queue_dir, "lock")
        os.makedirs(self.queue_dir, exist_ok=True)
        os.makedirs(self.completed_dir, exist_ok=True)
        os.makedirs(self.failed_dir, exist_ok=True)
        for p in (self.jobs_path, self.processing_path):
            if not os.path.exists(p):
                with open(p, "w", encoding="utf-8"):
                    pass

    def _acquire_lock(self, timeout_seconds: float = 5.0, retry_delay: float = 0.01) -> None:
        start = time.time()
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                if (time.time() - start) > timeout_seconds:
                    raise TimeoutError("Could not acquire queue lock in time")
                time.sleep(retry_delay)

    def _release_lock(self) -> None:
        try:
            os.remove(self.lock_path)
        except FileNotFoundError:
            pass

    def enqueue(self, payload: Dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        job = {"id": job_id, "created_at": time.time(), "payload": payload}
        self._acquire_lock()
        try:
            with open(self.jobs_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(job) + "\n")
        finally:
            self._release_lock()
        return job_id

    def _read_jsonl(self, path: str) -> list:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        items = []
        for line in lines:
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return items

    def _write_jsonl(self, path: str, items: list) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    def dequeue(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        self._acquire_lock()
        try:
            # If any job is currently in processing, do not dequeue a new one
            processing_now = self._read_jsonl(self.processing_path)
            if processing_now:
                return None
            jobs = self._read_jsonl(self.jobs_path)
            if not jobs:
                return None
            job = jobs.pop(0)
            self._write_jsonl(self.jobs_path, jobs)
            with open(self.processing_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(job) + "\n")
            return job["id"], job["payload"]
        finally:
            self._release_lock()

    def mark_completed(self, job_id: str, result: Optional[Dict[str, Any]] = None) -> None:
        self._acquire_lock()
        try:
            processing = self._read_jsonl(self.processing_path)
            remaining = [j for j in processing if j.get("id") != job_id]
            self._write_jsonl(self.processing_path, remaining)
            out_path = os.path.join(self.completed_dir, f"{job_id}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"id": job_id, "completed_at": time.time(), "result": result or {}}, f)
        finally:
            self._release_lock()

    def mark_failed(self, job_id: str, error: str) -> None:
        self._acquire_lock()
        try:
            processing = self._read_jsonl(self.processing_path)
            remaining = [j for j in processing if j.get("id") != job_id]
            self._write_jsonl(self.processing_path, remaining)
            out_path = os.path.join(self.failed_dir, f"{job_id}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"id": job_id, "failed_at": time.time(), "error": error}, f)
        finally:
            self._release_lock()

    def get_status(self, job_id: str) -> str:
        if os.path.exists(os.path.join(self.completed_dir, f"{job_id}.json")):
            return "completed"
        if os.path.exists(os.path.join(self.failed_dir, f"{job_id}.json")):
            return "failed"
        processing = self._read_jsonl(self.processing_path)
        if any(j.get("id") == job_id for j in processing):
            return "processing"
        jobs = self._read_jsonl(self.jobs_path)
        if any(j.get("id") == job_id for j in jobs):
            return "queued"
        return "unknown"


