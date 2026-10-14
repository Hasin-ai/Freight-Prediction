"""Check that the four input files exist, have the expected schema and match the published SHA-256s."""

from __future__ import annotations

import sys

from freight import data
from freight.config import data_config


def main() -> int:
    expected = data_config()["expected_sha256"]
    failures = 0
    for kind in data_config()["files"]:
        path = data.file_path(kind)
        if not path.exists():
            print(f"MISSING  {path}")
            failures += 1
            continue
        digest = data.sha256(path)
        status = "ok" if expected.get(kind, digest) == digest else "DIFFERENT DATA (results will not match)"
        print(f"{kind:10s} {path.name:40s} sha256 {digest[:12]}  {status}")
        if kind in ("train", "validation"):
            problems = data.schema_problems(data.load(kind), kind)
            failures += len(problems)
            for problem in problems:
                print(f"  SCHEMA  {problem}")
    print("data check passed" if failures == 0 else f"data check failed ({failures} problems)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
