"""Standalone Milestone 1 pipeline runner (no Airflow scheduler required).

Runs the same extract → validate → load chain the Airflow DAG orchestrates, in
dependency order, in one process. This is what ``uv run pipeline`` invokes so the
grader can run the pipeline end-to-end without standing up Airflow.

    uv run python scripts/run_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.revenue import pipeline_m1  # noqa: E402  (path set up above)


def main() -> None:
    manifest = pipeline_m1.run()
    print("\n✓ pipeline complete")
    for k, v in manifest.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
