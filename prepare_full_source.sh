#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PACKAGE_DIR"

mkdir -p data/full_database
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile

package = Path.cwd()
archive = package / "source_archives" / "Borneo_DZ_database_FINAL_v1_2026-09-14.zip"
destination = package / "data" / "full_database"
with ZipFile(archive) as source:
    source.extractall(destination)

if not (destination / "MASTER").exists():
    matches = list(destination.glob("*/MASTER"))
    if len(matches) != 1:
        raise RuntimeError(f"Cannot locate one MASTER directory after extraction: {matches}")
    inner = matches[0].parent
    for item in inner.iterdir():
        item.rename(destination / item.name)
print(destination)
PY

python src/build_training_inputs.py

