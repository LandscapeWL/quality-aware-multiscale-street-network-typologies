from __future__ import annotations

import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "data/02_download_road_network_data/run_log"
PID_TABLE = ROOT / "data/02_download_road_network_data/road_network_build_pid_table.csv"
MANAGER = ROOT / "code_upload/02_download_road_network_data/road_network_build_manager.py"
TMP_DIR = ROOT / "tmp" / "osmium"

LOG_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)
stdout_path = LOG_DIR / "road_build_manager.stdout.log"
stderr_path = LOG_DIR / "road_build_manager.stderr.log"

env = {
    **os.environ,
    "OSM_ROAD_BUILD_MAX_WORKERS": os.environ.get("OSM_ROAD_BUILD_MAX_WORKERS", "1"),
    "OSM_ROAD_BUILD_BUFFER_METERS": os.environ.get("OSM_ROAD_BUILD_BUFFER_METERS", "500"),
    "OSM_ROAD_BUILD_TMPDIR": os.environ.get("OSM_ROAD_BUILD_TMPDIR", str(TMP_DIR)),
    "TMPDIR": os.environ.get("OSM_ROAD_BUILD_TMPDIR", str(TMP_DIR)),
    "TEMP": os.environ.get("OSM_ROAD_BUILD_TMPDIR", str(TMP_DIR)),
    "TMP": os.environ.get("OSM_ROAD_BUILD_TMPDIR", str(TMP_DIR)),
}

args = [sys.executable, str(MANAGER)]
if os.environ.get("OSM_ROAD_BUILD_FAILED_ONLY", "1") != "0":
    args.append("--failed-only")

with stdout_path.open("ab") as out, stderr_path.open("ab") as err:
    proc = subprocess.Popen(
        args,
        cwd=str(ROOT),
        stdout=out,
        stderr=err,
        start_new_session=True,
        env=env,
    )

PID_TABLE.parent.mkdir(parents=True, exist_ok=True)
exists = PID_TABLE.exists()
with PID_TABLE.open("a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "started_at",
            "pid",
            "command",
            "stdout_log",
            "stderr_log",
            "max_workers",
            "buffer_meters",
            "tmp_dir",
            "failed_only",
        ],
    )
    if not exists:
        writer.writeheader()
    writer.writerow(
        {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "pid": proc.pid,
            "command": " ".join(args),
            "stdout_log": str(stdout_path.relative_to(ROOT)),
            "stderr_log": str(stderr_path.relative_to(ROOT)),
            "max_workers": env["OSM_ROAD_BUILD_MAX_WORKERS"],
            "buffer_meters": env["OSM_ROAD_BUILD_BUFFER_METERS"],
            "tmp_dir": env["OSM_ROAD_BUILD_TMPDIR"],
            "failed_only": "1" if "--failed-only" in args else "0",
        }
    )

print(proc.pid)
