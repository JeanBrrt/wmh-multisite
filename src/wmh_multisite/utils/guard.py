"""Memory watchdog for heavy jobs (J-005).

Two entry points:
- ``run_guarded(cmd)``: run an external command (ANTs CLI, nnU-Net, ...).
- ``call_guarded(func, *args)``: run a Python function in a child process.

Both poll the resident memory of the whole process tree (and optionally the GPU memory
reported by nvidia-smi) and kill the tree if a limit is exceeded, so that a runaway job
fails cleanly instead of freezing the machine.
"""

from __future__ import annotations

import multiprocessing as mp
import shutil
import subprocess
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil


@dataclass
class GuardResult:
    returncode: int | None
    seconds: float
    peak_ram_gb: float
    peak_vram_gb: float
    killed: str | None = None  # reason if the watchdog killed the job
    value: Any = None  # return value of the function (call_guarded only)
    error: str | None = None  # traceback raised inside the function (call_guarded only)

    @property
    def ok(self) -> bool:
        return self.killed is None and self.error is None and self.returncode == 0


def _tree_rss_gb(proc: psutil.Process) -> float:
    """RSS of a process and its children; 0 for a process that has already exited.

    The process can end between any two psutil calls (a fast job): ``children()`` then raises
    ``NoSuchProcess``. This race made the watchdog tests fail intermittently (ROADMAP, points to study).
    """
    total = 0
    try:
        tree = [proc, *proc.children(recursive=True)]
    except psutil.Error:
        return 0.0
    for p in tree:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total / 1e9


def gpu_used_gb() -> float:
    """Total GPU memory in use (all GPUs), 0.0 if nvidia-smi is unavailable."""
    if shutil.which("nvidia-smi") is None:
        return 0.0
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout
    try:
        return sum(float(v) for v in out.split()) / 1024
    except ValueError:
        return 0.0


def _kill_tree(proc: psutil.Process) -> None:
    for p in [*proc.children(recursive=True), proc]:
        try:
            p.kill()
        except psutil.Error:
            pass


def _watch(
    pid: int,
    is_running: Callable[[], bool],
    ram_limit_gb: float,
    vram_limit_gb: float | None,
    timeout_s: float | None,
    poll_s: float,
) -> tuple[float, float, str | None]:
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:  # already finished: nothing to watch
        return 0.0, 0.0, None
    t0, gpu0 = time.time(), gpu_used_gb() if vram_limit_gb is not None else 0.0
    peak_ram = peak_vram = 0.0
    while is_running():
        ram = _tree_rss_gb(proc)
        vram = gpu_used_gb() - gpu0 if vram_limit_gb is not None else 0.0
        peak_ram, peak_vram = max(peak_ram, ram), max(peak_vram, vram)
        reason = None
        if ram > ram_limit_gb:
            reason = f"RAM {ram:.1f} GB > {ram_limit_gb} GB"
        elif vram_limit_gb is not None and vram > vram_limit_gb:
            reason = f"VRAM {vram:.1f} GB > {vram_limit_gb} GB"
        elif timeout_s is not None and time.time() - t0 > timeout_s:
            reason = f"timeout {timeout_s:.0f} s"
        if reason:
            _kill_tree(proc)
            return peak_ram, peak_vram, reason
        time.sleep(poll_s)
    return peak_ram, peak_vram, None


def wait_for_free_ram(
    min_free_gb: float, poll_s: float = 5.0, log: Callable[[str], None] | None = None
) -> None:
    """Block until the machine has at least ``min_free_gb`` of available RAM.

    Machine-wide safeguard, complementary to the per-job limit: before a new job starts, check the
    RAM left for everything (other jobs and the user's applications), so that parallel jobs slow
    down instead of saturating the machine.
    """
    warned = False
    while psutil.virtual_memory().available / 1e9 < min_free_gb:
        if log and not warned:
            log(f"waiting: {psutil.virtual_memory().available / 1e9:.1f} GB available < {min_free_gb} GB")
            warned = True
        time.sleep(poll_s)


def run_guarded(
    cmd: list[str] | str,
    ram_limit_gb: float = 10.0,
    vram_limit_gb: float | None = None,
    timeout_s: float | None = None,
    poll_s: float = 1.0,
    log_path: str | Path | None = None,
    **popen_kwargs: Any,
) -> GuardResult:
    """Run an external command under the watchdog. stdout/stderr go to ``log_path`` if given."""
    t0 = time.time()
    log = open(log_path, "w") if log_path else subprocess.DEVNULL
    try:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, shell=isinstance(cmd, str), **popen_kwargs
        )
        peak_ram, peak_vram, killed = _watch(
            proc.pid, lambda: proc.poll() is None, ram_limit_gb, vram_limit_gb, timeout_s, poll_s
        )
        rc = proc.wait()
    finally:
        if log_path:
            log.close()
    return GuardResult(rc, round(time.time() - t0, 1), round(peak_ram, 2), round(peak_vram, 2), killed)


def _child(conn: Any, func: Callable, args: tuple, kwargs: dict) -> None:
    try:
        conn.send(("ok", func(*args, **kwargs)))
    except Exception:  # noqa: BLE001 - the traceback is sent back to the parent
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


def call_guarded(
    func: Callable,
    *args: Any,
    ram_limit_gb: float = 10.0,
    vram_limit_gb: float | None = None,
    timeout_s: float | None = None,
    poll_s: float = 1.0,
    **kwargs: Any,
) -> GuardResult:
    """Run ``func(*args, **kwargs)`` in a child process under the watchdog.

    ``func`` must be importable (defined at module level) and its return value picklable,
    because Windows starts child processes with "spawn".
    """
    t0 = time.time()
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(send_conn, func, args, kwargs))
    proc.start()
    send_conn.close()
    # Stop watching as soon as a result is ready: a large result blocks the child until it is read.
    peak_ram, peak_vram, killed = _watch(
        proc.pid,
        lambda: proc.is_alive() and not recv_conn.poll(),
        ram_limit_gb,
        vram_limit_gb,
        timeout_s,
        poll_s,
    )
    value = error = None
    if killed is None:
        try:
            status, payload = recv_conn.recv()
            value, error = (payload, None) if status == "ok" else (None, payload)
        except EOFError:
            error = "child process exited without returning a result"
    proc.join()
    recv_conn.close()
    return GuardResult(
        proc.exitcode,
        round(time.time() - t0, 1),
        round(peak_ram, 2),
        round(peak_vram, 2),
        killed,
        value,
        error,
    )
