# Remote worker

FEM labels cost minutes per sample and several GB of RAM. geocouple can send them to a remote Linux machine over
SSH; the laptop stays the coordinator (planning, content-hash cache, dashboard, git) and the server only executes
jobs. Every connection starts from the laptop (SSH + rsync with a key); the server never connects back.

## Set-up

```bash
cp configs/remote/gpu.example.yaml configs/remote/gpu.yaml   # not versioned: host, remote folder, limits
scripts/remote_setup.sh        # on the server, inside ~/<remote_root>: venv with the laptop's versions, ONELAB,
                               # the X11 libraries gmsh needs (extracted from .deb files, no root)
geocouple workers deploy       # copies the current code to code/<hash>/ on the server
geocouple workers start        # starts the daemon in tmux if it is not running
geocouple workers status       # heartbeat, queue, server load
```

Requirements on the server: Python 3.11+, rsync, tmux; `systemd-run --user` with the memory controller is used for
hard memory limits when available. No root, nothing written outside `~/<remote_root>`.

## Jobs

```bash
geocouple workers submit-p2 --names bar_0001 ground_plane_0000 --where remote
geocouple workers submit-full --families G0 G1 G2 --dry-run      # the whole dataset of configs/datasets/train.yaml
geocouple workers sync                                           # fetch results, merge into data/<set>/
```

A job is a folder `<id>/` with `job.yaml` (type, entry point, `code_hash`, git commit, resources, timeout, retries),
`inputs/` (for a simulation: the assembly, the pipeline settings, the stack) and, once run, `result/` (`sample/`,
`run.json` with status, duration, peak memory of the process tree, versions; `log.txt`).

The **job id is a content hash** (for simulations, the sample's cache key), so resubmitting, retrying and merging
are idempotent: a result is accepted only if its `labels.json` carries the expected key. Jobs run with exactly the
code version they were submitted with (`PYTHONPATH=code/<hash>`).

Queue on the server: `queue/staging → pending → running → done | failed`, each move an atomic rename. The daemon
puts jobs left in `running/` without a process back in `pending/` at start-up (up to `max_retries`).

The scheduler on the laptop estimates each simulation's memory from a coarse mesh built locally (a few seconds) and
routes jobs above `routing.local_max_mem_gb` to the server; smaller ones run locally, one at a time, under a
`MemoryMax` limit so a runaway job is killed, not the session.

## Shared-server etiquette

The defaults assume a machine shared with other people. All limits are keys under `worker:` in the config.

- **Caps:** cores ≤ `max_cores_frac` of the total and ≤ what others leave free minus `reserve_cores`; RAM ≤
  `max_mem_frac` of the total and ≤ `MemAvailable − mem_margin_gb`. A job starts only if it fits in what is left.
- **Back-off:** before every start the daemon reads the load, free memory and GPU use. If others are busy
  (load above `busy_load_frac` × cores, or free memory below `min_avail_gb`) it starts nothing and retries after
  `backoff_s`. Running jobs are never stopped.
- **Low priority:** every job runs under `nice -n 15` and `ionice -c 3` (idle).
- **Hard memory limit per job:** `systemd-run --user --scope -p MemoryMax=…` (reservation × `mem_limit_factor`),
  plus a watchdog that sums the process tree's RSS; a job over its limit is killed and retried with a larger
  reservation (`oom_retry_factor`).
- **Disk:** a quota for the whole remote folder (`disk_quota_gb`) and a minimum of free space on the disk
  (`min_disk_free_gb`); over either, no new job is claimed. Inputs of finished jobs are deleted at once, fetched
  jobs after `keep_done_h`, scratch folders at the end of each job.
- **GPU:** jobs that ask for a GPU get the least used one among `gpus_allowed`, and wait if others use it above
  `gpu_busy_util` / `gpu_busy_mem_frac`.
- **No daemons that outlive you unexpectedly:** the worker runs in a named tmux session (`tmux_session`);
  `geocouple workers stop` ends it.

Before using a shared machine, agree on these limits with its administrator.
