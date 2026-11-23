# Dashboard de rezultate

Site static generat din `runs/` și `reports/`. Nicio valoare nu se scrie de mână: fiecare pagină citește
`results.json`, câmpurile eșantionate (`fields/*.npz`) și mesh-urile (`*.msh`) din ultima rulare a experimentului.

```bash
cd ~/work/geo-couple
.venv/bin/python -m geocouple.viz.dashboard serve          # build + http://localhost:8765/
.venv/bin/python -m geocouple.viz.dashboard serve --port 9000
.venv/bin/python -m geocouple.viz.dashboard build          # doar scrie dashboard/site/ (se poate deschide direct
                                                           # dashboard/site/index.html în browser)
```

Conținut: o pagină de prezentare (stare de acceptanță per pas) și câte o pagină per pas (P0, P1, …) cu tabelele de
acceptanță, graficele interactive (plotly, inclus local — funcționează offline), vederi 3D ale geometriilor și mesh-urilor
(rotire / zoom cu mouse-ul) și raportul pasului. Runner-ele pașilor reconstruiesc dashboard-ul la final.

`dashboard/site/` este generat și nu se versionează.
