"""Write SHA-256 manifest for immutable inputs and reference outputs."""

from __future__ import annotations

import hashlib
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
INCLUDED = ("data", "source_archives", "reference_outputs", "src")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    files = []
    for directory in INCLUDED:
        root = PACKAGE / directory
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file() and "node_modules" not in path.parts)
    lines = [f"{digest(path)}  {path.relative_to(PACKAGE).as_posix()}" for path in sorted(files)]
    (PACKAGE / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} checksums")


if __name__ == "__main__":
    main()

