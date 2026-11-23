# Worker la distanță (server partajat)

Laptopul rămâne **coordonatorul** (planificare, cache pe hash de conținut, dashboard, git). Serverul
`<user>@<server>` (adresa în `configs/remote/gpu.yaml`, neversionat) este un **worker**: execută joburi și nu știe nimic despre laptop. Toate conexiunile
pornesc de pe laptop (SSH + rsync, cu cheie); laptopul nu primește nicio conexiune.

Prima utilizare: simulările FEM (Gmsh + GetDP sunt limitate de CPU și RAM — pilotul P2 s-a oprit de două ori din OOM
pe laptop). Din P3–P4: antrenarea pe GPU, prin același mecanism.

## Serverul (inspectat pe 2026-11-03)

| | |
| --- | --- |
| CPU | AMD Ryzen 9 9900X, 12 nuclee / 24 fire |
| RAM | 123 GiB (≈ 110 GiB disponibili), swap 7 GiB |
| GPU | 1 × RTX 4060 Ti, 8 GiB, CUDA 12.9 (driver 575) |
| Disc | `/` 1,8 TB, **98 % ocupat, ≈ 44 GB liberi** — partajat cu toți utilizatorii |
| OS | Ubuntu 24.04.2, Python 3.12.3, rsync, tmux, `systemd-run --user` (cu controller `memory`), `prlimit`, `ionice` |
| Alți utilizatori | ~10 sesiuni, sarcină tipică 0,5–5 (din 24) |
| Restricții | fără sudo, nimic în afara `~/geo-couple`; `Linger=no` (serviciile `systemd --user` mor la logout), `KillUserProcesses=no` (tmux supraviețuiește) |

## Arhitectura

```
laptop (coordonator)                                   server: ~/geo-couple
──────────────────────                                 ───────────────────────────────────────────────
geocouple.workers.scheduler                            env/        venv, ONELAB (getdp, gmsh), biblioteci X11
  estimare RAM (tetraedre) → local | remote            code/<hash>/ instantanee ale pachetului geocouple
  local: un job pe rând, verificare RAM liber,         queue/staging/  → pending/ → running/ → done/ | failed/
         plafon MemoryMax pe job                       scratch/<job>/  foldere de lucru GetDP (șterse la final)
geocouple.workers.remote  ── rsync/ssh (ieșire) ──►    heartbeat.json  starea worker-ului (la fiecare ciclu)
  deploy · submit · sync · status · start · stop       worker.yaml     configurația (copiată de pe laptop)
runs/workers/  rezultate aduse, status.json            logs/           jurnalul daemonului
data/<set>/    eșantioane (îmbinare idempotentă)       daemon: geocouple.workers.daemon în tmux
dashboard: pagina „Workers”
```

### Jobul

Un job este un folder `<id>/`:

- `job.yaml`: `id`, `type` (`simulate` | `train`), `entry` (funcția Python care îl execută), `input_hash` (= id-ul),
  `code_hash` (hash-ul surselor `geocouple/`) și `git_commit` (dacă există), `resources` (`cores`, `mem_gb`, `gpu`),
  `timeout_s`, `max_retries`, metadate (familie, nume eșantion).
- `inputs/`: tot ce trebuie jobului (pentru `simulate`: asamblarea JSON, configurația pipeline-ului, stack-ul).
- `result/` (scris la execuție): ieșirile jobului (pentru `simulate`: `sample/` cu `graph.json`, `coarse.vtkhdf`,
  `fine.vtkhdf`, `labels.json`), `run.json` (stare, cod de ieșire, durată, vârf RAM al arborelui de procese, gazdă,
  versiuni), `log.txt`.

**Id-ul jobului este hash-ul de conținut** (pentru simulări, chiar cheia de cache a eșantionului din
`data/simulate.py`). Aceeași intrare dă același id, deci retrimiterea, reîncercarea și îmbinarea rezultatelor sunt
idempotente: un rezultat se acceptă doar dacă `labels.json` are cheia de cache așteptată.

**Versiunea codului:** repo-ul nu are încă commituri, deci versiunea este `code_hash` = SHA-256 peste sursele
pachetului (`geocouple/**/*.py`, șabloanele `.pro`, `pyproject.toml`); `git_commit` se înregistrează când există.
Fiecare versiune se copiază o singură dată în `code/<hash>/`, iar jobul rulează exact cu versiunea din `job.yaml`
(`PYTHONPATH=code/<hash>`). Venv-ul conține doar dependențele, cu aceleași versiuni ca pe laptop (gmsh 4.15.2,
numpy 2.5.4, scipy 1.18.1, h5py 3.16.0); versiunile se scriu în `env/versions.json` și în `run.json`.

### Coada

- Trimitere: `rsync` în `queue/staging/<id>/`, apoi `mv` în `queue/pending/<id>` (atomic). Un job deja existent în
  orice stare nu se retrimite.
- Revendicare: daemonul face `rename(pending/<id>, running/<id>)` — atomic, deci un job nu poate fi luat de două ori.
- Final: `running/<id>` → `done/<id>` (cod 0 și `result/` complet) sau `failed/<id>`.
- La pornire, daemonul mută înapoi în `pending/` joburile rămase în `running/` fără proces (după o oprire bruscă),
  până la `max_retries`; apoi în `failed/`.
- `heartbeat.json`: ora, pid, starea (`idle` / `running` / `backoff` / `disk_full` / `stopping`), joburile curente,
  resursele rezervate, plafoanele calculate, sarcina serverului, RAM liber, GPU, discul folosit.

### Daemonul pe server

Rulează în `tmux` (sesiunea `geocouple-worker`), nu ca serviciu `systemd --user`: fără linger, managerul de
utilizator se oprește la logout, iar tmux supraviețuiește (`KillUserProcesses=no`). Daemonul nu depinde de laptop:
laptopul poate dormi, rezultatele așteaptă în `done/` până la următorul `sync`.

### Bun-simț pe serverul partajat (configurabil în `configs/remote/gpu.yaml`)

- **Plafoane:** nuclee ≤ `max_cores_frac` (implicit 50 %) din total și ≤ nucleele libere (total − sarcina altora −
  `reserve_cores`); RAM ≤ `max_mem_frac` (50 %) din total și ≤ `MemAvailable` − `mem_margin_gb`. Un job pornește
  doar dacă resursele lui încap în ce rămâne după joburile proprii care rulează.
- **Retragere:** înaintea fiecărei porniri se citesc `/proc/loadavg`, `/proc/meminfo` și `nvidia-smi`. Dacă alții
  folosesc mult (sarcina altora > `busy_load_frac` × nuclee sau RAM disponibil sub prag), daemonul nu pornește nimic
  nou și reîncearcă după `backoff_s`. Joburile deja pornite nu se opresc.
- **Prioritate scăzută:** `nice -n 15` și `ionice -c 3` (idle) pentru fiecare job.
- **Plafon de memorie per job:** `systemd-run --user --scope -p MemoryMax=… -p MemorySwapMax=0` (verificat că
  funcționează pe server); în plus, un watchdog al daemonului însumează RSS-ul arborelui de procese și omoară jobul
  peste plafon (rezervă dacă `systemd-run` lipsește). Vârful măsurat se scrie în `run.json`.
- **GPU:** joburile cu `gpu: 1` primesc GPU-ul cu cea mai mică utilizare și memorie ocupată (`CUDA_VISIBLE_DEVICES`),
  numai dintre `gpus_allowed`; dacă GPU-ul e ocupat de alții peste prag, jobul așteaptă.
- **Disc:** cotă pentru `~/geo-couple` (`disk_quota_gb`, implicit 15 GB) și prag minim de spațiu liber pe disc
  (`min_disk_free_gb`, implicit 20 GB — discul e aproape plin). Peste cotă sau sub prag nu se revendică joburi noi.
  Intrările joburilor terminate se șterg imediat; joburile aduse pe laptop (marcate `fetched`) se șterg după
  `keep_done_h`; `scratch/<job>` se șterge la finalul fiecărui job.

### Planificatorul de pe laptop

- **Estimarea memoriei** dintr-un mesh coarse construit local (doar Gmsh, câteva secunde): tetraedre fine
  ≈ max(1,15 × coarse, ținta ZZ); RAM ≈ `mem_base_mib` + `mem_mib_per_ktet` × ktet (ordinul 2, calibrat pe pilot:
  ≈ 320 MiB + 21 MiB per mie de tetraedre), × 1,6 pentru referința de convergență. Estimările se păstrează în
  `data/work/estimates.json`.
- **Rutare:** joburile peste `local_max_mem_gb` merg pe server; restul rămân local, dacă nu se cere altfel
  (`--where remote|local|auto`).
- **Local:** câte un job pe rând; înainte de pornire se verifică `MemAvailable` ≥ estimare + marjă, iar jobul rulează
  sub `systemd-run --user --scope -p MemoryMax=…`, deci o depășire omoară doar jobul, nu sesiunea.
- **Sync:** aduce `done/` și `failed/` în `runs/workers/jobs/<id>/`, îmbină eșantioanele în `data/<set>/` (doar dacă
  cheia de cache se potrivește), marchează joburile `fetched` pe server și scrie `runs/workers/status.json` pentru
  dashboard.

## Comenzi

```bash
cd ~/work/geo-couple
cp configs/remote/gpu.example.yaml configs/remote/gpu.yaml     # o singură dată; se completează
scripts/remote_setup.sh                                        # mediul pe server (venv, ONELAB, biblioteci)
.venv/bin/python -m geocouple.workers deploy                   # copiază codul curent în code/<hash>/
.venv/bin/python -m geocouple.workers start                    # pornește daemonul (tmux) dacă nu rulează
.venv/bin/python -m geocouple.workers status                   # heartbeat + coada
.venv/bin/python -m geocouple.workers submit-p2 --names bar_0001 ground_plane_0000 --where remote
.venv/bin/python -m geocouple.workers sync                     # aduce rezultatele, îmbină, scrie status.json
.venv/bin/python -m geocouple.workers stop                     # oprește daemonul după joburile curente
```

Pe server, daemonul se poate vedea cu `tmux attach -t geocouple-worker` (ieșire: `Ctrl-b d`).
