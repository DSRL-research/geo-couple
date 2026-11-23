# Results dashboard

A static site generated from `runs/` and `docs/reports/`. No value is written by hand: every page reads the
`results.json`, the sampled fields (`fields/*.npz`) and the meshes (`*.msh`) of the latest run of each step.

```bash
geocouple dashboard serve               # build + http://localhost:8765/
geocouple dashboard serve --port 9000
geocouple dashboard build               # only writes dashboard/site/ (open dashboard/site/index.html directly)
```

Content: an overview page (acceptance per step) and one page per step (P0, P1, …) with the acceptance tables,
interactive plots (plotly, bundled: works offline), 3D views of the geometries and meshes, and the step's report.
The step runners rebuild the dashboard when they finish.

`dashboard/site/` is generated and not versioned.
