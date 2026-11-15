#!/bin/bash
# sass_diff.sh A_BUILD_DIR B_BUILD_DIR [OUT_DIR]
# Compare the sm_60 SASS of every CUDA object (*.cu.o) present in both build folders.
# Prints one line per object: "same" or "DIFF <n> lines", then a summary. Exit 0 when every object is identical.
set -u
A=$1; B=$2; OUT=${3:-/tmp/sass_diff}
CUOBJDUMP=${CUOBJDUMP:-/usr/local/cuda-12.8/bin/cuobjdump}
mkdir -p "$OUT"
same=0; diff=0; missing=0
while IFS= read -r rel; do
  a="$A/$rel"; b="$B/$rel"
  if [ ! -f "$b" ]; then missing=$((missing + 1)); continue; fi
  key=$(echo "$rel" | tr '/' '_')
  # drop the lines that name the input file; keep function names and instructions
  "$CUOBJDUMP" -sass "$a" 2>/dev/null | grep -v -E '^(Fatbin|\s*code for|\s*arch|\s*host|\s*compile_size|\s*identifier)' > "$OUT/$key.a"
  "$CUOBJDUMP" -sass "$b" 2>/dev/null | grep -v -E '^(Fatbin|\s*code for|\s*arch|\s*host|\s*compile_size|\s*identifier)' > "$OUT/$key.b"
  if cmp -s "$OUT/$key.a" "$OUT/$key.b"; then
    same=$((same + 1)); echo "same   $rel ($(wc -l < "$OUT/$key.a") lines)"
    rm -f "$OUT/$key.a" "$OUT/$key.b"
  else
    n=$(diff "$OUT/$key.a" "$OUT/$key.b" | grep -c '^[<>]')
    diff=$((diff + 1)); echo "DIFF   $rel ($n differing lines; $OUT/$key.a vs .b)"
  fi
done < <(cd "$A" && find . -name '*.cu.o' | sed 's#^\./##' | sort)
echo "summary: $same identical, $diff different, $missing only in $A"
[ "$diff" -eq 0 ]
