from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
TASK_TABLE = ROOT / "data/00_environment_versions_and_data_inventory/download_task_master.csv"
STATUS_TABLE = ROOT / "data/00_environment_versions_and_data_inventory/download_status_table.csv"
EVENT_LOG = ROOT / "data/00_environment_versions_and_data_inventory/run_log/download_events.jsonl"
TASK_LOG_DIR = ROOT / "data/00_environment_versions_and_data_inventory/run_log/task_logs"
MAX_WORKERS = int(os.environ.get("OSM_DOWNLOAD_MAX_WORKERS", "3"))
CHUNK_SIZE = 1024 * 1024


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def read_tasks() -> list[dict[str, str]]:
    with TASK_TABLE.open("r", encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r["status"] == "ready_background"]


def write_event(event: dict) -> None:
    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EVENT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def log_line(task_id: str, text: str) -> None:
    TASK_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (TASK_LOG_DIR / f"{task_id}.log").open("a", encoding="utf-8") as f:
        f.write(f"[{utc_now()}] {text}\n")


def download_one(task: dict) -> dict:
    task_id = task["task_id"]
    url = task["url"]
    output = ROOT / task["output_path"]
    part = output.with_suffix(output.suffix + ".part")
    output.parent.mkdir(parents=True, exist_ok=True)
    expected = int(task["expected_bytes"]) if task.get("expected_bytes") else None
    result = {
        "task_id": task_id,
        "url": url,
        "output_path": task["output_path"],
        "started_at": utc_now(),
        "finished_at": "",
        "status": "running",
        "bytes_downloaded": "",
        "error": "",
    }
    write_event({**result, "event": "start"})
    log_line(task_id, f"START {url}")
    if output.exists() and (expected is None or output.stat().st_size == expected):
        result.update(status="already_complete", finished_at=utc_now(), bytes_downloaded=str(output.stat().st_size))
        write_event({**result, "event": "finish"})
        log_line(task_id, "already complete")
        return result
    retries = int(task.get("max_retries") or 5)
    for attempt in range(1, retries + 1):
        try:
            existing = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "Codex-OSM-study-downloader/1.0"}
            mode = "ab"
            if existing:
                headers["Range"] = f"bytes={existing}-"
            with requests.get(url, headers=headers, stream=True, timeout=(30, 180)) as r:
                if r.status_code == 416 and output.exists():
                    result.update(status="complete", finished_at=utc_now(), bytes_downloaded=str(output.stat().st_size))
                    return result
                if existing and r.status_code == 200:
                    # Server ignored Range; restart cleanly to avoid corrupt append.
                    part.unlink(missing_ok=True)
                    existing = 0
                    mode = "wb"
                elif not existing:
                    mode = "wb"
                r.raise_for_status()
                with part.open(mode + "") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
            final_size = part.stat().st_size
            if expected is not None and final_size != expected:
                raise RuntimeError(f"size mismatch expected={expected} got={final_size}")
            part.replace(output)
            result.update(status="complete", finished_at=utc_now(), bytes_downloaded=str(output.stat().st_size))
            write_event({**result, "event": "finish"})
            log_line(task_id, f"COMPLETE bytes={result['bytes_downloaded']}")
            return result
        except Exception as exc:
            err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            log_line(task_id, f"attempt {attempt}/{retries} failed: {err}")
            write_event({**result, "event": "retry", "attempt": attempt, "error": err})
            time.sleep(min(60, 5 * attempt))
    result.update(status="failed", finished_at=utc_now(), bytes_downloaded=str(part.stat().st_size if part.exists() else 0), error="exhausted retries")
    write_event({**result, "event": "finish"})
    return result


def write_status(rows: list[dict]) -> None:
    STATUS_TABLE.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task_id", "status", "bytes_downloaded", "started_at", "finished_at", "output_path", "url", "error"]
    with STATUS_TABLE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in fields} for r in rows])


def main() -> int:
    tasks = read_tasks()
    write_event({"event": "manager_start", "time": utc_now(), "task_count": len(tasks), "max_workers": MAX_WORKERS})
    print(f"Starting {len(tasks)} ready download tasks with max_workers={MAX_WORKERS}", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(download_one, task) for task in tasks]
        for fut in as_completed(futures):
            results.append(fut.result())
            write_status(results)
    write_status(results)
    write_event({"event": "manager_finish", "time": utc_now(), "task_count": len(tasks)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
