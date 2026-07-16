"""Airflow DAG: monthly revenue data pipeline (Milestone 1).

Three tasks, handing off through persisted artifacts:

    extract  →  validate  →  load

  • extract  — read the raw payments source (Stripe snapshot if configured,
               else the synthetic historical CSV) and persist a raw snapshot.
  • validate — enforce the raw-charges data-quality contract with Pandera;
               the task (and therefore the DAG run) FAILS on invalid data.
  • load     — clean, aggregate to a monthly series, engineer features, and
               write a versioned clean dataset (run-id in the filename).

The task *logic* lives in ``src/revenue/pipeline_m1.py`` so it is unit-testable
and runnable without Airflow (see ``scripts/run_pipeline.py``); this module is
just the orchestration wiring.

Task hand-off is through the storage backend (``src/storage.py``), not XCom, so
the tasks share the persisted raw snapshot. The bundled docker-compose runs a
single-node SequentialExecutor, where the local ``outputs/`` dir is shared. On a
*distributed* executor (Celery/Kubernetes), set ``OUTPUTS_BUCKET`` to a shared
S3 backend so a task that lands on a different worker can read the snapshot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pendulum
from airflow.decorators import dag, task

# Make the `src` package importable when Airflow loads this file from dags/.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dag(
    dag_id="revenue_pipeline",
    schedule="@monthly",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["revenue", "milestone1"],
    doc_md=__doc__,
)
def revenue_pipeline():
    @task
    def extract() -> dict:
        from src.revenue import pipeline_m1

        return pipeline_m1.extract()

    @task
    def validate(extract_meta: dict) -> int:
        from src.revenue import pipeline_m1

        # extract_meta establishes the task dependency; validate re-reads the
        # persisted snapshot so it is reproducible in isolation.
        _ = extract_meta
        return pipeline_m1.validate()

    @task
    def load(n_valid_rows: int) -> dict:
        from src.revenue import pipeline_m1

        _ = n_valid_rows
        return pipeline_m1.load()

    load(validate(extract()))


revenue_pipeline()
