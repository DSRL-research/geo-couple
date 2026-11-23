#!/usr/bin/env bash
# Download the ONELAB bundle (GetDP + Gmsh binaries, Linux x86-64) into third_party/onelab, without a system install.
# geocouple finds it there automatically (see geocouple/tools.py); ONELAB_ROOT or GETDP_BIN override it.
# GetDP and Gmsh are GPL-licensed and are not redistributed with geocouple.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/onelab"
URL="${ONELAB_URL:-https://onelab.info/files/onelab-Linux64.zip}"
if [ -x "$DEST/getdp" ]; then echo "already there: $DEST/getdp"; exit 0; fi
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
echo "downloading $URL"
curl -fL --retry 3 -o "$tmp/onelab.zip" "$URL"
unzip -q "$tmp/onelab.zip" -d "$tmp/x"
src="$(dirname "$(find "$tmp/x" -name getdp -type f | head -1)")"
mkdir -p "$(dirname "$DEST")"; rm -rf "$DEST"; mv "$src" "$DEST"
"$DEST/getdp" --version 2>&1 | tail -1
echo "getdp: $DEST/getdp"
echo "gmsh:  $DEST/gmsh"
