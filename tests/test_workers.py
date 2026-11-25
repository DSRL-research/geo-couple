import json
import os
import sys
import time
from pathlib import Path

import pytest
import yaml

from geocouple import PROJECT_ROOT
from geocouple.data.build import PipelineSpec
from geocouple.data.simulate import cache_key
from geocouple.geometry import compose
from geocouple.graph.coupling import CouplingSpec
from geocouple.workers import config as wconfig
from geocouple.workers import daemon
from geocouple.workers.job import assembly_from_dict, assembly_to_dict, code_hash, read_job, write_job
from geocouple.workers.scheduler import route


def test_code_hash_is_stable():
    assert code_hash() == code_hash() and len(code_hash()) == 16


def test_assembly_roundtrip_keeps_cache_key(stack):
    asm = compose.parallel_bars(length=30.0, gap=3.0)
    back = assembly_from_dict(json.loads(json.dumps(assembly_to_dict(asm))))
    spec, cpl = PipelineSpec(), CouplingSpec()
    assert cache_key(back, spec, cpl, stack) == cache_key(asm, spec, cpl, stack)


def test_config_defaults_merge(tmp_path):
    f = tmp_path / "gpu.yaml"
    f.write_text("ssh_host: h\nworker: {disk_quota_gb: 3}\n")
    c = wconfig.load(f)
    assert c["worker"]["disk_quota_gb"] == 3 and c["worker"]["max_cores_frac"] == 0.5 and c["ssh_host"] == "h"


def test_route_by_memory(tmp_path):
    jobs = [write_job(tmp_path / k, k, "selftest", "x:y", {"cores": 1, "mem_gb": m}, 10) for k, m in
            (("small", 1.0), ("big", 9.0))]
    local, remote = route(jobs, "auto", {"local_max_mem_gb": 2.5})
    assert [p.name for p in local] == ["small"] and [p.name for p in remote] == ["big"]
    assert route(jobs, "remote", {"local_max_mem_gb": 2.5})[0] == []


@pytest.fixture
def quiet_machine(monkeypatch):
    """The daemon tests must not depend on what else runs on this machine: no load from others, plenty of RAM."""
    monkeypatch.setattr(daemon.os, "getloadavg", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(daemon, "meminfo_gb", lambda: {"total": 64.0, "available": 48.0})
    monkeypatch.setattr(daemon, "gpus", lambda: [])


def _worker_root(tmp_path) -> Path:
    root = tmp_path / "w"
    (root / "env" / "venv" / "bin").mkdir(parents=True)
    os.symlink(sys.executable, root / "env" / "venv" / "bin" / "python")
    (root / "code").mkdir()
    os.symlink(PROJECT_ROOT, root / "code" / code_hash())
    (root / "worker.yaml").write_text(yaml.safe_dump({"max_cores_frac": 1.0, "reserve_cores": 0, "mem_margin_gb": 0,
                                                       "min_avail_gb": 0, "busy_load_frac": 100, "poll_s": 0,
                                                       "min_disk_free_gb": 0, "max_retries": 1}))
    return root


def _submit(root: Path, jid: str, **meta) -> None:
    d = write_job(root / "queue" / "staging" / jid, jid, "selftest", "geocouple.workers.job:selftest",
                  {"cores": 1, "mem_gb": meta.pop("mem_gb", 0.25), "gpu": 0}, meta.pop("timeout_s", 60), meta=meta,
                  max_retries=1)
    os.rename(d, root / "queue" / "pending" / jid)


def _drive(w: daemon.Worker, until, timeout=60):
    t0 = time.time()
    while not until():
        w.watch()
        w.schedule()
        assert time.time() - t0 < timeout
        time.sleep(0.2)


def test_daemon_runs_retries_and_fails(tmp_path, quiet_machine):
    root = _worker_root(tmp_path)
    w = daemon.Worker(root)
    w.cleanup()
    _submit(root, "ok1", mem_mib=8)
    _submit(root, "hog", mem_mib=600, sleep_s=5, mem_gb=0.1)   # above its 0.15 GB limit -> retried, then failed
    _submit(root, "slow", sleep_s=30, timeout_s=1)
    q = root / "queue"
    _drive(w, lambda: not w.running and not any((q / "pending").iterdir()))
    assert (q / "done" / "ok1" / "result" / "selftest.txt").exists()
    assert not (q / "done" / "ok1" / "inputs").exists()          # inputs of finished jobs are cleaned up
    hog = read_job(q / "failed" / "hog")
    assert hog["retries"] == 2 and hog["resources"]["mem_gb"] == 0.2   # one retry with doubled memory
    run = json.loads((q / "failed" / "slow" / "result" / "run.json").read_text())
    assert run["worker"]["killed"] == "timeout"
    st = daemon.status(root)
    assert [j["id"] for j in st["queue"]["done"]] == ["ok1"] and len(st["queue"]["failed"]) == 2


def test_daemon_recovers_lost_jobs(tmp_path, quiet_machine):
    root = _worker_root(tmp_path)
    w = daemon.Worker(root)
    _submit(root, "lost")
    os.rename(root / "queue" / "pending" / "lost", root / "queue" / "running" / "lost")
    w.recover()
    assert (root / "queue" / "pending" / "lost").exists()
    assert read_job(root / "queue" / "pending" / "lost")["retries"] == 1


@pytest.mark.parametrize("busy", [True, False])
def test_daemon_backs_off_when_server_busy(tmp_path, busy, quiet_machine):
    root = _worker_root(tmp_path)
    w = daemon.Worker(root)
    w.cfg["min_avail_gb"] = 1e6 if busy else 0
    _submit(root, "j")
    w.schedule()
    assert (w.state == "backoff") == busy and bool(w.running) == (not busy)
    _drive(w, lambda: not w.running)


def test_gpu_query_survives_odd_nvidia_smi_output(monkeypatch):
    """2026-11-05: one error line from nvidia-smi instead of the table stopped the daemon."""
    class R:
        stdout = "Unable to determine the device handle for GPU0000:01:00.0: Unknown Error\n" \
                 "0, NVIDIA GeForce RTX 4060 Ti, 100, 7287, 8188\n"
    monkeypatch.setattr(daemon.subprocess, "run", lambda *a, **k: R())
    g = daemon.gpus()
    assert len(g) == 1 and g[0]["util"] == 100.0 and g[0]["mem_total_mib"] == 8188.0


def test_merge_summary_only(tmp_path, monkeypatch):
    from geocouple.workers import remote
    monkeypatch.setattr(remote, "PROJECT_ROOT", tmp_path)
    d = write_job(tmp_path / "jobs" / "abc", "abc", "simulate", "x:y", {"cores": 1, "mem_gb": 1}, 10,
                  meta={"family": "G0", "name": "bar_0000", "data_root": "data/set", "keep_remote": True})
    (d / "result" / "sample").mkdir(parents=True)
    (d / "result" / "sample" / "labels.json").write_text(json.dumps({"cache_key": "abc", "files": {"fine.vtkhdf": {}}}))
    target = remote.merge_result(d, summary_only=True)
    assert sorted(p.name for p in target.iterdir()) == ["labels.json", "remote.json"]
    assert json.loads((target / "remote.json").read_text())["files"] == ["fine.vtkhdf"]


@pytest.mark.parametrize("by_hand", [False, True])
def test_requeued_job_is_not_deleted_before_a_new_sync(tmp_path, quiet_machine, by_hand):
    """E5 bus@1e6 (2026-11-24): fetched after a timeout, requeued, finished, then deleted by cleanup because the
    old `fetched` marker came along. The new result must stay until it is fetched again."""
    root = _worker_root(tmp_path)
    w = daemon.Worker(root)
    q = root / "queue"
    _submit(root, "e5", sleep_s=2, timeout_s=0.5)
    _drive(w, lambda: (q / "failed" / "e5").exists())
    old = time.time() - 3 * 86400  # the laptop fetched the failed run three days ago
    (q / "failed" / "e5" / "fetched").touch()
    os.utime(q / "failed" / "e5" / "fetched", (old, old))
    if by_hand:  # what was done on the server: mv failed/<id> pending/ (+ a longer timeout)
        job = read_job(q / "failed" / "e5")
        job["timeout_s"] = 60
        (q / "failed" / "e5" / "job.yaml").write_text(yaml.safe_dump(job, sort_keys=False))
        os.rename(q / "failed" / "e5", q / "pending" / "e5")
    else:
        daemon.requeue(root, "e5", timeout_s=60)
        assert not (q / "pending" / "e5" / "fetched").exists()
        assert read_job(q / "pending" / "e5")["retries"] == 0
    _drive(w, lambda: (q / "done" / "e5").exists())
    assert not (q / "done" / "e5" / "fetched").exists()
    w.cleanup()
    assert (q / "done" / "e5" / "result" / "run.json").exists()       # kept: not fetched since it finished
    (q / "done" / "e5" / "fetched").touch()                           # a new sync
    w.cleanup()
    assert (q / "done" / "e5").exists()                               # kept for keep_done_h after that sync
    assert daemon.removable(q / "done" / "e5", w.cfg["keep_done_h"], time.time() + (w.cfg["keep_done_h"] + 1) * 3600)


def test_stale_fetched_marker_never_deletes_a_newer_result(tmp_path):
    d = tmp_path / "done" / "j"
    (d / "result").mkdir(parents=True)
    (d / "result" / "run.json").write_text(json.dumps({"worker": {"finished": daemon.now()}}))
    (d / "fetched").touch()
    old = time.time() - 3 * 86400
    os.utime(d / "fetched", (old, old))
    assert not daemon.removable(d, 24, time.time() + 10 * 86400)
