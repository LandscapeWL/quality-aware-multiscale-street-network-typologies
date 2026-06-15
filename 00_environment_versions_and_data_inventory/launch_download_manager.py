from __future__ import annotations

import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "data/00_environment_versions_and_data_inventory/run_log"
PID_TABLE = ROOT / "data/00_environment_versions_and_data_inventory/background_task_pid_table.csv"
MANAGER = ROOT / "code_upload/00_environment_versions_and_data_inventory/download_manager.py"

LOG_DIR.mkdir(parents=True, exist_ok=True)
stdout_path = LOG_DIR / "download_manager.stdout.log"
stderr_path = LOG_DIR / "download_manager.stderr.log"

with stdout_path.open("ab") as out, stderr_path.open("ab") as err:
    proc = subprocess.Popen(
        [sys.executable, str(MANAGER)],
        cwd=str(ROOT),
        stdout=out,
        stderr=err,
        start_new_session=True,
        env={**os.environ, "OSM_DOWNLOAD_MAX_WORKERS": os.environ.get("OSM_DOWNLOAD_MAX_WORKERS", "3")},
    )

PID_TABLE.parent.mkdir(parents=True, exist_ok=True)
exists = PID_TABLE.exists()
with PID_TABLE.open("a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["started_at", "pid", "command", "stdout_log", "stderr_log"])
    if not exists:
        writer.writeheader()
    writer.writerow({
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "pid": proc.pid,
        "command": f"{sys.executable} {MANAGER}",
        "stdout_log": str(stdout_path.relative_to(ROOT)),
        "stderr_log": str(stderr_path.relative_to(ROOT)),
    })

print(proc.pid)
