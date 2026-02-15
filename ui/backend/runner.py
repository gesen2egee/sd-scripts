"""Sequential batch runner and process management for ANIMA UI."""

from __future__ import annotations

import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config_builder import ROOT_DIR, RUNTIME_ROOT, build_job

CREATE_NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)


@dataclass
class JobRecord:
    id: str
    name: str
    profile_id: str
    status: str = "queued"
    message: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    return_code: Optional[int] = None
    runtime_dir: Optional[str] = None
    log_file: Optional[str] = None
    command: Optional[List[str]] = None


class BatchRunner:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: List[JobRecord] = []
        self._worker: Optional[threading.Thread] = None
        self._pending_job_ids: List[str] = []
        self._stop_requested = False
        self._current_process: Optional[subprocess.Popen] = None
        self._current_job_id: Optional[str] = None

        self._tb_process: Optional[subprocess.Popen] = None
        self._tb_log_file: Optional[str] = None
        self._tb_logdir: Optional[str] = None
        self._tb_port: int = 6006

    def start_batch(self, profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        with self._lock:
            if self._worker and self._worker.is_alive():
                return {"ok": False, "error": "batch_already_running"}

            self._jobs = []
            self._pending_job_ids = []
            self._stop_requested = False
            self._current_job_id = None

            for profile in profiles:
                if profile.get("run_enabled", True) is False:
                    continue
                profile_id = str(profile.get("id") or uuid.uuid4().hex)
                job_id = uuid.uuid4().hex
                name = str(profile.get("name") or f"job-{len(self._jobs)+1}")
                job = JobRecord(id=job_id, name=name, profile_id=profile_id)
                self._jobs.append(job)
                self._pending_job_ids.append(job_id)

            if not self._jobs:
                return {"ok": False, "error": "empty_batch"}

            profile_map = {str(p.get("id") or ""): p for p in profiles}
            self._worker = threading.Thread(target=self._run_loop, args=(profile_map,), daemon=True)
            self._worker.start()
            return {"ok": True, "jobs": [asdict(j) for j in self._jobs]}

    def stop_batch(self) -> Dict[str, Any]:
        with self._lock:
            self._stop_requested = True
            for job in self._jobs:
                if job.status == "queued":
                    job.status = "cancelled"
                    job.message = "Cancelled before start"
                    job.ended_at = time.time()

            if self._current_process and self._current_process.poll() is None:
                self._kill_process_tree(self._current_process.pid)

            return {"ok": True}

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            jobs = [asdict(j) for j in self._jobs]
            for job in jobs:
                log_file = job.get("log_file")
                job["log_tail"] = self._tail_file(log_file) if log_file else ""

            batch_running = bool(self._worker and self._worker.is_alive())
            tb_running = bool(self._tb_process and self._tb_process.poll() is None)
            return {
                "batch_running": batch_running,
                "stop_requested": self._stop_requested,
                "current_job_id": self._current_job_id,
                "jobs": jobs,
                "tensorboard": {
                    "running": tb_running,
                    "log_file": self._tb_log_file,
                    "logdir": self._tb_logdir,
                    "port": self._tb_port,
                    "url": f"http://127.0.0.1:{self._tb_port}" if tb_running else None,
                    "log_tail": self._tail_file(self._tb_log_file) if self._tb_log_file else "",
                },
            }

    def start_tensorboard(self, logdir: str, port: int = 6006) -> Dict[str, Any]:
        with self._lock:
            if self._tb_process and self._tb_process.poll() is None:
                return {"ok": False, "error": "tensorboard_already_running", "url": f"http://127.0.0.1:{self._tb_port}"}

            tb_exe = ROOT_DIR / "venv" / "Scripts" / "tensorboard.exe"
            root_python = ROOT_DIR / "venv" / "Scripts" / "python.exe"
            if tb_exe.exists():
                cmd = [str(tb_exe), "--logdir", logdir, "--port", str(port)]
            else:
                cmd = [str(root_python), "-m", "tensorboard.main", "--logdir", logdir, "--port", str(port)]

            RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
            tb_log = RUNTIME_ROOT / "tensorboard.log"
            cmdline = subprocess.list2cmdline(cmd)
            shell_cmd = f'{cmdline} > "{tb_log}" 2>&1'

            proc = subprocess.Popen(
                ["cmd", "/c", shell_cmd],
                cwd=ROOT_DIR,
                creationflags=CREATE_NEW_CONSOLE,
            )

            self._tb_process = proc
            self._tb_log_file = str(tb_log)
            self._tb_logdir = logdir
            self._tb_port = int(port)
            return {"ok": True, "url": f"http://127.0.0.1:{self._tb_port}"}

    def stop_tensorboard(self) -> Dict[str, Any]:
        with self._lock:
            if not self._tb_process or self._tb_process.poll() is not None:
                return {"ok": True, "message": "not_running"}

            self._kill_process_tree(self._tb_process.pid)
            return {"ok": True}

    def _run_loop(self, profile_map: Dict[str, Dict[str, Any]]) -> None:
        while True:
            with self._lock:
                if self._stop_requested or not self._pending_job_ids:
                    break
                job_id = self._pending_job_ids.pop(0)
                job = self._find_job(job_id)
                if job is None:
                    continue
                profile = profile_map.get(job.profile_id)
                if not profile:
                    job.status = "failed"
                    job.message = "Missing profile payload"
                    job.ended_at = time.time()
                    continue
                self._current_job_id = job_id
                job.status = "running"
                job.started_at = time.time()

            self._run_single_job(job, profile)

            with self._lock:
                self._current_job_id = None
                if self._stop_requested:
                    break

        with self._lock:
            if self._stop_requested:
                for job in self._jobs:
                    if job.status == "queued":
                        job.status = "cancelled"
                        job.message = "Cancelled"
                        job.ended_at = time.time()
            self._current_process = None

    def _run_single_job(self, job: JobRecord, profile: Dict[str, Any]) -> None:
        runtime_dir = RUNTIME_ROOT / job.id
        runtime_dir.mkdir(parents=True, exist_ok=True)

        try:
            built = build_job(runtime_dir, profile)
            log_file = runtime_dir / "train.log"
            cmd = built["command"]
            cmdline = subprocess.list2cmdline(cmd)
            shell_cmd = f'{cmdline} > "{log_file}" 2>&1'

            with self._lock:
                job.runtime_dir = str(runtime_dir)
                job.log_file = str(log_file)
                job.command = cmd
                job.message = "Running"

            proc = subprocess.Popen(
                ["cmd", "/c", shell_cmd],
                cwd=ROOT_DIR,
                creationflags=CREATE_NEW_CONSOLE,
            )

            with self._lock:
                self._current_process = proc

            while proc.poll() is None:
                with self._lock:
                    if self._stop_requested:
                        self._kill_process_tree(proc.pid)
                        break
                time.sleep(1.0)

            with self._lock:
                code = proc.poll()
                job.return_code = code
                job.ended_at = time.time()
                if self._stop_requested:
                    job.status = "cancelled"
                    job.message = "Stopped by user"
                elif code == 0:
                    job.status = "succeeded"
                    job.message = "Completed"
                else:
                    job.status = "failed"
                    job.message = f"Exited with code {code}"
        except Exception as exc:
            with self._lock:
                job.status = "failed"
                job.message = f"Build/launch error: {exc}"
                job.ended_at = time.time()

    def _find_job(self, job_id: str) -> Optional[JobRecord]:
        for job in self._jobs:
            if job.id == job_id:
                return job
        return None

    @staticmethod
    def _kill_process_tree(pid: int) -> None:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True)

    @staticmethod
    def _tail_file(path: Optional[str], max_lines: int = 40) -> str:
        if not path:
            return ""
        file_path = Path(path)
        if not file_path.exists():
            return ""
        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-max_lines:])
        except Exception:
            return ""

