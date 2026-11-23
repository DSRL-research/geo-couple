# FAQ

**`geocouple doctor` says GetDP is missing.**
Run `scripts/get_onelab.sh`, or point `GETDP_BIN` / `ONELAB_ROOT` to an existing install. Everything except meshing
+ solving (and the `fem` tests) works without it.

**`import gmsh` fails with a missing `libGLU.so.1` (or `libXcursor`, `libXft`, `libXinerama`).**
The gmsh wheel links against X11 / OpenGL libraries even without a display. Install `libglu1-mesa libxcursor1
libxinerama1 libxft2` (Debian / Ubuntu), or extract them without root as `scripts/remote_setup.sh` does and add the
folder to `LD_LIBRARY_PATH`.

**How long does one FEM sample take?**
The PoC samples took ≈ 345 s (median, one core) and up to several GB of RAM; time grows like N^1.28 and memory like
N^0.88 with the number N of tetrahedra (`docs/reports/P5.md`, E5). The quickstart layout takes about 3 minutes.
Large layouts belong on a worker ([workers.md](workers.md)).

**Why does `geocouple train` need `data/train_v1`?**
The models learn corrections to the analytic kernel from FEM labels. The PoC dataset (430 samples) is generated with
`geocouple workers submit-full` or sample by sample with `geocouple dataset simulate`; it is not part of the
repository because of its size.

**Which model should I use?**
On compositions like the training ones (G2), S0 is the most accurate (C mutual ≈ 5% median); on real layouts (G3)
the context models transfer better (A1: C self 6.0%, C mutual 8.4%; S0 14.0% / 13.6%, no better than B0). L and R
are within ≈ 0.1% (self L) for all of them. See `docs/reports/PoC.md`.

**Are the stack and the numbers those of a real process?**
No. `configs/stack/default.yaml` is an illustrative two-metal stack in air. The method does not depend on it; the
error figures do.

**Curved shapes in my GDS are ignored by the models.**
Non-rectilinear polygons are imported as `polygon` primitives: they are meshed and solved by FEM, but have no centre
line, so B0 and the learned models (built on the analytic kernel) skip them.

**The reports are in Romanian.**
`docs/reports/` holds the generated reports of the PoC steps as they were produced; the README and `docs/` are in
English. `docs/dev/` keeps the internal design notes.

**A saved model fails to load after upgrading scikit-learn or torch.**
Model files contain pickled scikit-learn GPs. Retrain (`geocouple train`) with the installed versions, or install
the versions listed in [installation.md](installation.md).

**Results differ from the reports in the 7th digit.**
GP hyper-parameter fits and BLAS differ slightly between library versions and machines (≈ 1e-7 relative). The
code itself is deterministic for a given seed and environment.
