# TODO

## Worker la distanță

Construit pe 2026-11-03 pentru simulări FEM (`geocouple/workers`, `docs/workers.md`). Rămâne de făcut pentru
antrenare (P3–P4): tipul de job `train` (intrare, checkpoint, metrici), PyTorch CUDA + PyG în venv-ul de pe server.

Planul inițial (păstrat ca referință):

Context: stația locală nu are GPU; un GPU există pe un alt server. Pașii intensivi (antrenarea B2–B4 și A1 pe mesh-uri
fine, eventual inferența pe G2 mare în E5) se vor trimite unui worker la distanță. Restul (mesh, GetDP, date, metrici,
figuri, dashboard) rămâne pe localhost.

De făcut, când va fi nevoie (P3–P4):

- [ ] Interfața de job: un folder `jobs/<id>/` cu `job.yaml` (tip: train | infer, model, configurație de experiment,
      seed), referințe la date (fișiere VTKHDF + GeoGraph JSON + etichete), iar la ieșire `result/` (checkpoint,
      metrici JSON, predicții, log). Fără stare în afara folderului.
- [ ] Runner local care decide: job pe CPU local sau trimis worker-ului (după `device:` din configurație).
- [ ] Transport: sincronizare folder (rsync/ssh) + pornire job la distanță; polling pentru stare (queued/running/
      done/failed); aducerea rezultatelor înapoi în `runs/`.
- [ ] Worker pe serverul cu GPU: același pachet `geocouple` (versiune fixată), mediu cu PyTorch CUDA + PyG.
- [ ] Reproductibilitate: hash de configurație + versiune de cod în `result/`, seed fix.
- [ ] Configurarea: `configs/remote/gpu.yaml` (din `gpu.example.yaml`) + alias SSH în `~/.ssh/config`, cu cheie,
      fără parolă; utilizatorul completează adresa, folderul de lucru de pe server și interpretorul.

## Altele

- [x] Compilarea FastHenry și FastCap în `third_party/` (`scripts/build_fast_solvers.sh`).
