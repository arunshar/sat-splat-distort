#!/usr/bin/env bash
# Download IEEE GRSS DFC2019 Track-3 multi-view satellite data.
#
# DFC2019 requires registering an IEEE DataPort account and downloading the
# zip manually. Once you have it, drop it under data/_raw/dfc2019.zip and run
# this script to extract + index.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data/dfc2019"
RAW="$ROOT/data/_raw/dfc2019.zip"

mkdir -p "$DATA"

if [[ ! -f "$RAW" ]]; then
  echo "Place the DFC2019 Track-3 zip at $RAW first."
  echo "Get it from https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019"
  exit 1
fi

echo "Extracting DFC2019..."
unzip -q "$RAW" -d "$DATA"

echo "Building train/test splits..."
python -m sat_splat.scripts.build_dfc2019_splits --root "$DATA"

echo "Done. Sites available:"
ls -d "$DATA"/*/ | xargs -n1 basename
