#!/usr/bin/env bash
# Mediul worker-ului pe server, numai în ~/<remote_root> (fără sudo, fără pachete de sistem). Idempotent.
#   venv cu aceleași versiuni ca .venv local; ONELAB (getdp, gmsh) copiat de pe laptop; bibliotecile X11 cerute de
#   libgmsh extrase din .deb-uri Ubuntu (apt-get download + dpkg-deb -x, ca utilizator) în env/syslibs.
# Versiunile se scriu în env/versions.json. Utilizare: scripts/remote_setup.sh  (citește configs/remote/gpu.yaml)
set -euo pipefail
cd "$(dirname "$0")/.."
read -r HOST ROOT < <(.venv/bin/python -c "from geocouple.workers import config as c; k=c.load(); print(k['ssh_host'], k['remote_root'])")
ONELAB="${ONELAB_ROOT:-$HOME/work/onelab-Linux64real}"
pin() { .venv/bin/python -c "import importlib.metadata as m; print('$1=='+m.version('$1'))"; }
PINS="$(pin gmsh) $(pin numpy) $(pin scipy) $(pin h5py) $(pin PyYAML) $(pin jsonschema) $(pin matplotlib) $(pin shapely) $(pin gdstk) pytest"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST")

"${SSH[@]}" "mkdir -p $ROOT/env $ROOT/code $ROOT/logs $ROOT/scratch $ROOT/queue/{staging,pending,running,done,failed}"
rsync -a --delete "$ONELAB/" "$HOST:$ROOT/env/onelab/"
"${SSH[@]}" bash -s -- "$ROOT" "'$PINS'" <<'EOF'  # ssh re-splits arguments: keep PINS one word
set -euo pipefail
ROOT=$1; PINS=$2
cd "$ROOT/env"
[ -x venv/bin/python ] || python3 -m venv venv
venv/bin/pip -q install --upgrade pip
venv/bin/pip -q install $PINS
if [ ! -e syslibs/libGLU.so.1 ]; then
  rm -rf debs sysroot && mkdir -p debs syslibs
  (cd debs && apt-get download libglu1-mesa libxrender1 libxcursor1 libxfixes3 libxft2 libxinerama1)
  for d in debs/*.deb; do dpkg-deb -x "$d" sysroot; done
  find sysroot -name '*.so*' -exec cp -P {} syslibs/ \;
  rm -rf debs sysroot
fi
export LD_LIBRARY_PATH=$PWD/syslibs
venv/bin/python - <<'PY'
import json, platform, subprocess, importlib.metadata as m
v = {p: m.version(p) for p in ("gmsh", "numpy", "scipy", "h5py", "PyYAML", "jsonschema", "matplotlib", "shapely", "gdstk", "pytest")}
import gmsh; gmsh.initialize(); v["gmsh_api"] = gmsh.option.getString("General.Version"); gmsh.finalize()
r = subprocess.run(["onelab/getdp", "--version"], capture_output=True, text=True)
v["getdp"] = (r.stdout.strip() or r.stderr.strip()).splitlines()[-1]
v["python"] = platform.python_version(); v["os"] = platform.platform()
open("versions.json", "w").write(json.dumps(v, indent=1)); print(json.dumps(v))
PY
EOF
