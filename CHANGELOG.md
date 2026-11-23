# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [0.1.0] — unreleased

First public version: the proof of concept (steps P0–P5) packaged for other users.

### Added
- `geocouple` command line: `doctor`, `quickstart`, `gds-import`, `graph`, `mesh`, `solve`, `b0`, `train`,
  `predict`, `dataset`, and pass-through to `bench`, `workers`, `dashboard`.
- Public Python API in `geocouple` (lazy imports) and `geocouple.pipeline` (the steps with the dataset settings).
- Saved models: `geocouple.learn.model` trains S0 / G1 / A1, saves the network with its local GP models, and predicts
  R, L, C for a new layout.
- `geocouple.learn.surrogate`: the GP / MLP surrogates, no longer imported from the downstream project; identical
  predictions (tested).
- Clear errors and lookup order for GetDP, FastHenry and FastCap; `scripts/get_onelab.sh`.
- Packaging extras (`learn`, `viz`, `g3`, `dev`, `all`), test markers (`fem`, `solvers`, `server`), ruff,
  pre-commit and a CI workflow.
- English documentation: README, installation, tutorial, data format, extending, workers, FAQ.

### Changed
- Reports moved to `docs/reports/`, the internal design notes to `docs/dev/`.
- Logging instead of `print` in the library and the job runners (`GEOCOUPLE_LOG` sets the level).
- Machine-specific paths replaced by environment variables and `configs/remote/gpu.yaml` (`torch_path`).

### Fixed
- `geocouple/data/`, `geocouple/export/` and `configs/export/` were excluded from version control by unanchored
  `.gitignore` patterns.

### Results
- Unchanged: the PoC numbers in `docs/reports/` (P0–P5) are those of `sim_version` p2.3.
