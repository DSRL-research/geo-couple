"""CLI pentru worker (vezi docs/workers.md).

    python -m geocouple.workers deploy | start | stop | status | sync
    python -m geocouple.workers submit-p2 [--names a b ...] [--families G0 G1] [--where auto|local|remote] [--wait]
    python -m geocouple.workers submit-full [--families G0 G1 G2] [--dry-run]   # setul complet, tot pe server
"""

from __future__ import annotations

import argparse
import json

from geocouple.workers import config as wconfig
from geocouple.workers.job import read_job
from geocouple.workers.remote import Remote
from geocouple.workers.scheduler import full_jobs, p2_jobs, route, run_local, wait_remote


def _summary(st: dict) -> str:
    hb = st.get("heartbeat") or {}
    cap = hb.get("capacity", {})
    lines = [f"daemon: {'up' if st['daemon_alive'] else 'DOWN'}, state {hb.get('state')} {hb.get('note', '')}".rstrip(),
             f"server: load {cap.get('load1', 0):.1f} (others {cap.get('others_load', 0):.1f}) / {cap.get('ncpu')}, "
             f"MemAvailable {cap.get('mem_available_gb', 0):.0f} GB, free for jobs {cap.get('cores_free')} cores / "
             f"{cap.get('mem_free_gb', 0):.0f} GB, folder {hb.get('disk_gb', 0):.2f} GB, disk free "
             f"{hb.get('disk_free_gb', 0):.0f} GB",
             "queue: " + ", ".join(f"{k} {len(v)}" for k, v in st["queue"].items())]
    for r in hb.get("running", []):
        lines.append(f"  running {r['id']} {r['elapsed_s']:.0f} s, {r['rss_mib']:.0f} MiB")
    for j in st["queue"]["failed"]:
        lines.append(f"  failed {j['id']} {j['meta'].get('name')}: {(j.get('error') or '')[:120]}")
    return "\n".join(lines)


def main():
    from geocouple import _logging
    _logging.setup()
    ap = argparse.ArgumentParser(prog="python -m geocouple.workers")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("deploy", "start", "stop", "status", "sync"):
        sub.add_parser(c)
    s = sub.add_parser("submit-p2")
    s.add_argument("--names", nargs="*")
    s.add_argument("--families", nargs="*")
    s.add_argument("--where", choices=["auto", "local", "remote"])
    s.add_argument("--wait", action="store_true")
    f = sub.add_parser("submit-full")
    f.add_argument("--families", nargs="*")
    f.add_argument("--dry-run", action="store_true")
    f.add_argument("--shell-from", help="P3 splits.json: also export cells_shell for fields + test samples")
    a = ap.parse_args()
    cfg = wconfig.load()

    if a.cmd == "submit-full":  # every job on the server; the data stays there (docs/dev/design.md §12)
        shell = None
        if a.shell_from:
            sp = json.loads(open(a.shell_from).read())
            shell = set(sp["E1"]["test"]) | set(sp["E2"]["test"])
        jobs, errors, asms = full_jobs(families=a.families, shell_names=shell)
        for name, err in errors.items():
            print(f"{name}: no job ({err})")
        by_fam: dict = {}
        for d in jobs:
            j = read_job(d)
            fam = by_fam.setdefault(j["meta"]["family"], {"n": 0, "fields": 0, "mem_gb": 0.0, "max_gb": 0.0})
            fam["n"] += 1
            fam["fields"] += j["meta"]["fields"]
            fam["mem_gb"] += j["resources"]["mem_gb"]
            fam["max_gb"] = max(fam["max_gb"], j["resources"]["mem_gb"])
        for fam, v in sorted(by_fam.items()):
            print(f"{fam}: {v['n']} jobs ({v['fields']} with fields), est. mem mean {v['mem_gb'] / v['n']:.2f} GB, "
                  f"max {v['max_gb']:.2f} GB")
        print(f"total: {len(jobs)} jobs of {len(asms)} samples, {len(errors)} without job")
        if not a.dry_run and jobs:
            r = Remote(cfg)
            r.deploy()
            sent = r.submit(jobs)
            print("submitted:", len(sent))
            print(r.start())
            import time

            from geocouple.workers.remote import LOCAL
            f = LOCAL / "full_submitted.json"
            prev = json.loads(f.read_text()) if f.exists() else {}
            f.write_text(json.dumps({"time": prev.get("time", time.time()), "jobs": len(jobs),
                                     "submitted": prev.get("submitted", 0) + len(sent)}))
        return

    if a.cmd == "submit-p2":
        jobs, errors = p2_jobs(a.names, a.families)
        for name, err in errors.items():
            print(f"{name}: no job ({err})")
        local, remote = route(jobs, a.where or cfg["routing"]["default_where"], cfg["routing"])
        for d in jobs:
            j = read_job(d)
            print(f"{j['meta']['family']} {j['meta']['name']:<22} {j['id']} est {j['meta']['fine_tets_est']:>8} tets "
                  f"{j['resources']['mem_gb']:>6.2f} GB -> {'remote' if d in remote else 'local'}")
        if remote:
            r = Remote(cfg)
            r.deploy()
            print("submitted:", r.submit(remote))
            print(r.start())
        for d in local:
            info = run_local(d, cfg["routing"])
            print(f"local {d.name}: {info['status']} {info['worker']['wall_s']:.0f} s")
        if remote and a.wait:
            print(_summary(wait_remote(Remote(cfg), {read_job(d)["id"] for d in remote})["status"]))
        return

    r = Remote(cfg)
    if a.cmd == "deploy":
        print(r.deploy())
    elif a.cmd == "start":
        r.deploy()
        print(r.start())
    elif a.cmd == "stop":
        print(r.stop())
    elif a.cmd == "status":
        print(_summary(r.status()))
    elif a.cmd == "sync":
        res = r.sync()
        print(json.dumps(res["fetched"]))
        print(_summary(res["status"]))


if __name__ == "__main__":
    main()
