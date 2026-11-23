"""Configurația worker-ului: configs/remote/gpu.yaml peste valorile implicite de mai jos."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from geocouple import PROJECT_ROOT

CONFIG = PROJECT_ROOT / "configs" / "remote" / "gpu.yaml"

DEFAULTS = {
    "ssh_host": None,
    "ssh_opts": ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30"],
    "remote_root": "geo-couple",          # relative to the remote home
    "tmux_session": "geocouple-worker",
    "worker": {                            # shared-server etiquette, read by the daemon on the server
        "max_cores_frac": 0.5,             # never more than this fraction of the cores ...
        "reserve_cores": 2,                # ... nor more than (cores - others' load - reserve)
        "max_mem_frac": 0.5,               # never more than this fraction of the RAM ...
        "mem_margin_gb": 16.0,             # ... nor more than MemAvailable - margin
        "busy_load_frac": 0.6,             # others' load above this fraction of the cores -> back off
        "min_avail_gb": 24.0,              # MemAvailable below this -> back off
        "max_parallel_jobs": 8,
        "poll_s": 10,
        "backoff_s": 120,
        "nice": 15,
        "ionice_class": 3,                 # idle
        "mem_limit_factor": 1.5,           # hard per-job limit = reservation x factor
        "oom_retry_factor": 2.0,           # killed for memory -> retry with reservation x factor
        "max_retries": 2,
        "disk_quota_gb": 15.0,             # whole ~/geo-couple folder
        "min_disk_free_gb": 20.0,          # the disk is shared and almost full
        "keep_done_h": 24.0,               # fetched jobs are deleted after this
        "keep_code_versions": 3,
        "gpus_allowed": [0],
        "gpu_busy_util": 30,               # % utilisation by others -> GPU considered busy
        "gpu_busy_mem_frac": 0.5,
    },
    "routing": {                           # laptop scheduler
        "default_where": "auto",           # auto | local | remote
        "local_max_mem_gb": 2.5,           # bigger estimates go remote
        "local_mem_margin_gb": 1.5,        # local start needs MemAvailable >= estimate + margin
        "local_mem_limit_factor": 1.5,     # local MemoryMax = estimate x factor
        "mem_base_mib": 320.0,             # peak RSS ~ base + per_ktet x fine ktets (order 2, P2 pilot)
        "mem_mib_per_ktet": 21.5,
        "fine_over_coarse": 1.15,          # fine tets ~ max(factor x coarse tets, ZZ target)
        "safety": 1.2,
        "cores": 1,                        # GetDP runs single-threaded
        "timeout_s": 6 * 3600,
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(path: Path | None = None) -> dict:
    path = Path(path or CONFIG)
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    return _merge(DEFAULTS, raw)
