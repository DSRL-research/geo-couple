#!/usr/bin/env bash
# Compilează FastHenry și FastCap (FastFieldSolvers, ramura master) în third_party/, fără instalare în sistem.
# GCC >= 10: codul K&R are nevoie de -fcommon; la FastCap, funcțiile care întorc pointeri sunt declarate implicit
# (int pe 64 de biți → pointer trunchiat → segfault), deci se forțează un header cu prototipurile lor. Sursele nu
# se modifică.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"
mkdir -p "$TP"
[ -d "$TP/FastHenry2" ] || git clone -q --depth 1 https://github.com/ediloren/FastHenry2 "$TP/FastHenry2"
[ -d "$TP/FastCap2" ] || git clone -q --depth 1 https://github.com/ediloren/FastCap2 "$TP/FastCap2"

cd "$TP/FastHenry2"
make fasthenry CFLAGS="-O -DFOUR -m64 -fcommon" > build.log 2>&1

cd "$TP/FastCap2"
grep -hE "^(static )?(char|double) *\*+ *[A-Za-z_0-9]+ *\(" src/*.c | sed -E 's/^static //; s/\(.*/();/' \
  | grep -v "calloc\|malloc" | sort -u > compat_protos.txt
{ echo "/* geocouple: prototypes for 64-bit builds */"
  printf '#include <%s>\n' stdio.h stdlib.h string.h unistd.h
  cat compat_protos.txt; } > compat.h
rm -f src/*.o
(cd src && gcc -O -DOTHER -fcommon -include ../compat.h -Dstatic= -w -c mulSetup.c -o mulSetup.o \
        && gcc -O -DOTHER -fcommon -w -c uglyalloc.c -o uglyalloc.o)
make fastcap CFLAGS="-O -DOTHER -fcommon -include ../compat.h" > build.log 2>&1

echo "fasthenry: $TP/FastHenry2/bin/fasthenry"
echo "fastcap:   $TP/FastCap2/bin/fastcap"
