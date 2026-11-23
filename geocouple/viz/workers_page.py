"""Pagina „Workers” a dashboard-ului: coada, joburile, debitul și sarcina serverului, din runs/workers/."""

from __future__ import annotations

import datetime as dt
import html
import json

import plotly.graph_objects as go
import yaml

from geocouple import PROJECT_ROOT
from geocouple.viz import style

ROOT = PROJECT_ROOT / "runs" / "workers"

GUIDE = """
<p><b>Ce arată pagina.</b> Simulările grele rulează pe serverul partajat (worker), cele mici pe laptop. Laptopul
trimite joburile și aduce rezultatele când e pornit; serverul lucrează și când laptopul doarme. Datele vin din ultima
sincronizare (<code>python -m geocouple.workers sync</code>).</p>
<ul>
<li><b>Starea worker-ului.</b> „running” = rulează joburi; „idle” = coada e goală; „backoff” = alți utilizatori
folosesc mult serverul și worker-ul așteaptă; „disk_full” = cota de disc e atinsă.</li>
<li><b>Coada.</b> pending → running → done / failed. Un job e un eșantion (id = cheia de cache a eșantionului).</li>
<li><b>Memorie estimată vs. măsurată.</b> Fiecare punct e un job terminat. Punctele sub diagonală înseamnă că estimarea
a fost prudentă (bine); peste diagonală, estimarea a fost prea mică. Estimarea decide unde rulează jobul.</li>
<li><b>Sarcina serverului.</b> Sarcina totală, partea altor utilizatori și câte joburi ale noastre rulau, la fiecare
sincronizare.</li>
</ul>"""


def _jobs() -> list[dict]:
    rows = []
    for d in sorted((ROOT / "jobs").glob("*")) if (ROOT / "jobs").exists() else []:
        if not (d / "job.yaml").exists():
            continue
        job = yaml.safe_load((d / "job.yaml").read_text())
        run_f = d / "result" / "run.json"
        run = json.loads(run_f.read_text()) if run_f.exists() else {}
        state = (d / "state").read_text().strip() if (d / "state").exists() else "?"
        rows.append({"job": job, "run": run, "state": state})
    return rows


def _peak_gb(run: dict) -> float:
    """The larger of the worker's sampled tree RSS and the job's own getrusage peak (which catches short spikes)."""
    w, o = run.get("worker", {}), run.get("output") or {}
    return max(w.get("peak_tree_rss_mib", 0), o.get("peak_child_rss_mib", 0),
               (run.get("peak_rss_mib") or {}).get("children", 0)) / 1024


def _when(t: float) -> str:
    return dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")


def workers_body(table, fig_html) -> str | None:
    st_f = ROOT / "status.json"
    if not st_f.exists() and not (ROOT / "jobs").exists():
        return None
    st = json.loads(st_f.read_text()) if st_f.exists() else {"queue": {}, "heartbeat": None}
    hb = st.get("heartbeat") or {}
    cap = hb.get("capacity", {})
    q = st.get("queue", {})
    jobs = _jobs()
    out = [f"<p class='sub'>Ultima sincronizare {_when(st['synced'])}" if st.get("synced") else "<p class='sub'>",
           f" · heartbeat de pe <code>{html.escape(hb.get('host', '?'))}</code> la {_when(hb['unix'])}</p>"
           if hb else "</p>",
           "<details class='card guide' open><summary>Cum citești această pagină</summary>" + GUIDE + "</details>"]

    ok = [j for j in jobs if j["state"] == "done"]
    t_done = [dt.datetime.fromisoformat(j["run"]["finished"]).timestamp() for j in ok if j["run"].get("finished")]
    span_h = (max(t_done) - min(t_done)) / 3600 if len(t_done) > 1 else 0
    wall = [j["run"]["wall_s"] for j in ok if "wall_s" in j["run"]]
    kpi = [["Worker", f"{'activ' if st.get('daemon_alive') else 'oprit'} · {html.escape(str(hb.get('state', '—')))}"
            + (f" <span class='sub'>({html.escape(hb['note'])})</span>" if hb.get("note") else "")],
           ["Coada pe server", ", ".join(f"{k} {len(v)}" for k, v in q.items()) or "—"],
           ["Joburi aduse pe laptop", f"{len(ok)} done, {sum(j['state'] == 'failed' for j in jobs)} failed"],
           ["Debit", (f"{len(t_done) / span_h:.1f} joburi/oră" if span_h > 0.05 else "—")
            + (f" · mediană {sorted(wall)[len(wall) // 2]:.0f} s/job" if wall else "")],
           ["Server", f"sarcină {cap.get('load1', 0):.1f} (alții {cap.get('others_load', 0):.1f}) din "
                      f"{cap.get('ncpu', '?')} fire · RAM disponibil {cap.get('mem_available_gb', 0):.0f} / "
                      f"{cap.get('mem_total_gb', 0):.0f} GB · liber pentru noi {cap.get('cores_free', 0)} fire, "
                      f"{cap.get('mem_free_gb', 0):.0f} GB" if cap else "—"],
           ["Disc", f"folder {hb.get('disk_gb', 0):.2f} GB din cota de {hb.get('config', {}).get('disk_quota_gb', '?')} GB"
                    f" · liber pe disc {hb.get('disk_free_gb', 0):.0f} GB" if hb else "—"],
           ["GPU", ", ".join(f"{g['name']}: {g['util']:.0f}%, {g['mem_used_mib']:.0f}/{g['mem_total_mib']:.0f} MiB"
                             for g in hb.get("gpus", [])) or "—"]]
    out.append("<h2>Stare</h2><div class='card'>" + table(["", ""], kpi) + "</div>")

    live = [[f"<code>{r['id']}</code>", html.escape(next((j["meta"].get("name", "") for j in q.get("running", [])
                                                          if j["id"] == r["id"]), "")),
             f"{r['elapsed_s']:.0f} s", f"{r['rss_mib']:.0f} MiB", f"{r['mem_gb']:.1f} GB"]
            for r in hb.get("running", [])]
    live += [[f"<code>{j['id']}</code>", html.escape(j["meta"].get("name", "")), "în așteptare", "—",
              f"{j['resources']['mem_gb']:.1f} GB"] for j in q.get("pending", [])]
    if live:
        out.append("<h2>Pe server acum</h2><div class='card'>" + table(
            ["Job", "Eșantion", "Durată", "RAM vârf", "Rezervat"], live) + "</div>")

    rows = []
    for j in sorted(jobs, key=lambda j: j["run"].get("finished", ""), reverse=True):
        m, w, out_ = j["job"].get("meta", {}), j["run"].get("worker", {}), j["run"].get("output") or {}
        rows.append([f"<code>{j['job']['id']}</code>", html.escape(f"{m.get('family', '')} {m.get('name', '')}"),
                     html.escape(w.get("host", "?")),
                     ("<span class='badge ok'>done</span>" if j["state"] == "done" else
                      f"<span class='badge no'>{html.escape(j['state'])}</span>"),
                     f"{j['run'].get('wall_s', 0):.0f}", f"{out_.get('fine_tets', 0):,}",
                     f"{m.get('mem_gb', j['job']['resources']['mem_gb']):.1f}",
                     f"{_peak_gb(j['run']):.2f}",
                     str(w.get("retries", 0)), html.escape((j["run"].get("error") or "")[:80])])
    if rows:
        out.append("<h2>Joburi terminate</h2><div class='card'>" + table(
            ["Job", "Eșantion", "Unde", "Stare", "Timp [s]", "Tetraedre fine", "RAM estimat [GB]", "RAM vârf [GB]",
             "Reîncercări", "Eroare"], rows) + "</div>")

    figs = []
    pts = [(j["job"]["meta"].get("mem_gb", j["job"]["resources"]["mem_gb"]),
            _peak_gb(j["run"]), j["job"]["meta"].get("name", ""),
            j["run"].get("worker", {}).get("host", "")) for j in ok]
    if pts:
        f1 = go.Figure()
        hosts = sorted({p[3] for p in pts})
        for k, h in enumerate(hosts):
            sel = [p for p in pts if p[3] == h]
            f1.add_trace(go.Scatter(x=[p[0] for p in sel], y=[p[1] for p in sel], mode="markers", name=h,
                                    text=[p[2] for p in sel], marker=dict(size=10, color=style.CATEGORICAL[k]),
                                    hovertemplate="%{text}<br>estimat %{x:.2f} GB<br>măsurat %{y:.2f} GB<extra></extra>"))
        top = max(max(p[0] for p in pts), max(p[1] for p in pts)) * 1.1
        f1.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=style.MUTED, dash="dot", width=1),
                                showlegend=False, hoverinfo="skip"))
        style.layout(f1, "RAM estimat [GB]", "RAM vârf măsurat [GB]", height=360)
        figs.append(f"<div class='card'><h3>Memorie estimată vs. măsurată</h3>{fig_html(f1)}</div>")
    hist_f = ROOT / "history.jsonl"
    hist = [json.loads(l) for l in hist_f.read_text().splitlines() if l.strip()] if hist_f.exists() else []
    if len(hist) > 1:
        t = [dt.datetime.fromtimestamp(h["t"]) for h in hist]
        f2 = go.Figure()
        for k, (key, label) in enumerate((("load1", "sarcină totală"), ("others_load", "sarcina altora"),
                                          ("running", "joburile noastre"))):
            f2.add_trace(go.Scatter(x=t, y=[h[key] for h in hist], mode="lines+markers", name=label,
                                    line=dict(color=style.CATEGORICAL[k], width=2), marker=dict(size=5)))
        style.layout(f2, None, "fire", height=360)
        figs.append(f"<div class='card'><h3>Sarcina serverului</h3>{fig_html(f2)}</div>")
    if figs:
        out.append("<h2>Grafice</h2><div class='grid'>" + "".join(figs) + "</div>")
    return "\n".join(out)
