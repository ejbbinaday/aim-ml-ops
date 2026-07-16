"""storage.py — single-switch artifact I/O.

If the `OUTPUTS_BUCKET` environment variable is set, reads and writes go to
`s3://$OUTPUTS_BUCKET/<name>` via boto3 (which honours `AWS_ENDPOINT_URL` for
LocalStack). Otherwise, they go to a local `outputs/` directory next to the
repo root.

Reads can also target an explicit bucket other than `OUTPUTS_BUCKET` via the
`bucket=` keyword (used by the data-mart contract reader, which reads a bucket
this repo does not own). The `bucket=`-capable read family (`load_parquet`,
`load_json`, `load_bytes`) is **fail-loud**: when a bucket is in play (either
`bucket=` or `OUTPUTS_BUCKET`), a missing or unreadable object raises rather
than silently falling back to local data. The local-`outputs/` fallback applies
only when no bucket is configured at all (offline development).

Usage:
    from src.storage import save_csv, save_json, save_text, save_figure
    from src.storage import load_csv, load_json, load_parquet
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from functools import lru_cache
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pandas as pd

_LOCAL_DIR = Path(__file__).resolve().parent.parent / "outputs"


def _bucket() -> str | None:
    name = os.getenv("OUTPUTS_BUCKET")
    return name or None


@lru_cache(maxsize=1)
def _s3_client():
    import boto3

    return boto3.client("s3")


def _put_bytes(bucket: str, name: str, data: bytes, content_type: str | None = None) -> None:
    kwargs: dict[str, Any] = {"Bucket": bucket, "Key": name, "Body": data}
    if content_type:
        kwargs["ContentType"] = content_type
    _s3_client().put_object(**kwargs)


def _get_bytes(bucket: str, name: str) -> bytes:
    obj = _s3_client().get_object(Bucket=bucket, Key=name)
    return obj["Body"].read()


def _local_path(name: str) -> Path:
    path = _LOCAL_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _local_atomic_replace(tmp_path: Path, dst_path: Path) -> None:
    """Rename `tmp_path` to `dst_path` atomically (POSIX `os.replace`).

    Caller is responsible for ensuring `tmp_path` and `dst_path` are on the
    same filesystem so the rename is atomic.
    """
    os.replace(tmp_path, dst_path)


# ── Writes ───────────────────────────────────────────────────────────────────


def save_csv(name: str, df: pd.DataFrame, **to_csv_kwargs: Any) -> None:
    bucket = _bucket()
    if bucket:
        buf = StringIO()
        df.to_csv(buf, **to_csv_kwargs)
        _put_bytes(bucket, name, buf.getvalue().encode("utf-8"), content_type="text/csv")
    else:
        df.to_csv(_local_path(name), **to_csv_kwargs)


def save_json(name: str, obj: Any, **dump_kwargs: Any) -> None:
    body = json.dumps(obj, **dump_kwargs)
    bucket = _bucket()
    if bucket:
        _put_bytes(bucket, name, body.encode("utf-8"), content_type="application/json")
    else:
        _local_path(name).write_text(body)


def save_text(name: str, content: str, content_type: str = "text/plain") -> None:
    bucket = _bucket()
    if bucket:
        _put_bytes(bucket, name, content.encode("utf-8"), content_type=content_type)
    else:
        _local_path(name).write_text(content)


def save_figure(name: str, fig: Any = None, **savefig_kwargs: Any) -> None:
    """Save a matplotlib figure as `name`.

    When `fig` is supplied, calls `fig.savefig(...)`. Otherwise falls back to
    `plt.savefig(...)` (which writes whatever `plt.gcf()` resolves to). Callers
    that build multiple figures in sequence should pass `fig=` explicitly.
    """
    import matplotlib.pyplot as plt

    target = fig if fig is not None else plt
    bucket = _bucket()
    if bucket:
        fmt = name.rsplit(".", 1)[-1].lower()
        buf = BytesIO()
        target.savefig(buf, format=fmt, **savefig_kwargs)
        _put_bytes(bucket, name, buf.getvalue(), content_type=f"image/{fmt}")
    else:
        target.savefig(_local_path(name), **savefig_kwargs)


def save_text_atomic(name: str, content: str, content_type: str = "text/plain") -> None:
    """Write `content` to `name` atomically.

    Local path: write to a sibling `<name>.tmp.<uuid>` then `os.replace` to
    the final path so a partial write cannot leave a corrupted file visible.
    S3 path: `PutObject` is itself atomic per-key, so no rename dance needed.
    """
    bucket = _bucket()
    if bucket:
        _put_bytes(bucket, name, content.encode("utf-8"), content_type=content_type)
        return
    final_path = _local_path(name)
    tmp_path = final_path.with_name(f"{final_path.name}.tmp.{uuid.uuid4().hex}")
    tmp_path.write_text(content)
    _local_atomic_replace(tmp_path, final_path)


def save_csv_atomic(name: str, df: pd.DataFrame, **to_csv_kwargs: Any) -> None:
    """Write `df` to `name` atomically.

    Local path: write to a sibling `<name>.tmp.<uuid>` then `os.replace` to
    the final path. S3 path: `PutObject` is itself atomic per-key.
    """
    bucket = _bucket()
    if bucket:
        buf = StringIO()
        df.to_csv(buf, **to_csv_kwargs)
        _put_bytes(bucket, name, buf.getvalue().encode("utf-8"), content_type="text/csv")
        return
    final_path = _local_path(name)
    tmp_path = final_path.with_name(f"{final_path.name}.tmp.{uuid.uuid4().hex}")
    df.to_csv(tmp_path, **to_csv_kwargs)
    _local_atomic_replace(tmp_path, final_path)


def save_bytes(name: str, data: bytes, content_type: str | None = None) -> None:
    """Write raw bytes to `name`. Used for binary artifacts (e.g., joblib bundles)."""
    bucket = _bucket()
    if bucket:
        _put_bytes(bucket, name, data, content_type=content_type)
    else:
        _local_path(name).write_bytes(data)


# ── Reads ────────────────────────────────────────────────────────────────────


def load_csv(name: str, **read_csv_kwargs: Any) -> pd.DataFrame:
    bucket = _bucket()
    if bucket:
        return pd.read_csv(BytesIO(_get_bytes(bucket, name)), **read_csv_kwargs)
    return pd.read_csv(_LOCAL_DIR / name, **read_csv_kwargs)


def load_parquet(name: str, *, bucket: str | None = None) -> pd.DataFrame:
    """Read a Parquet object as a DataFrame.

    When `bucket` is given, reads from that bucket (the data-mart contract bucket,
    which differs from `OUTPUTS_BUCKET`). Otherwise uses the default backend:
    `OUTPUTS_BUCKET` when set, else the local `outputs/` directory. Fail-loud —
    with a bucket in play a missing/unreadable object raises (via `get_object`);
    the local fallback applies only when no bucket is configured at all.

    Real nulls and declared dtypes round-trip through Parquet, unlike CSV.
    """
    target = bucket or _bucket()
    if target:
        return pd.read_parquet(BytesIO(_get_bytes(target, name)))
    return pd.read_parquet(_LOCAL_DIR / name)


def load_json(name: str, *, bucket: str | None = None) -> Any:
    target = bucket or _bucket()
    if target:
        return json.loads(_get_bytes(target, name).decode("utf-8"))
    return json.loads((_LOCAL_DIR / name).read_text())


def load_bytes(name: str, *, bucket: str | None = None) -> bytes:
    """Read raw bytes from `name`.

    When `bucket` is given, reads from that bucket; otherwise uses the default
    backend (`OUTPUTS_BUCKET` when set, else local `outputs/`). Fail-loud when a
    bucket is in play; the local fallback applies only with no bucket configured.
    """
    target = bucket or _bucket()
    if target:
        return _get_bytes(target, name)
    return (_LOCAL_DIR / name).read_bytes()


def exists(name: str) -> bool:
    """Return True iff `name` exists in the configured backend.

    Used by phase 1 to choose between the Stripe snapshot and the historical
    CSV without raising on the missing case.
    """
    bucket = _bucket()
    if bucket:
        try:
            _s3_client().head_object(Bucket=bucket, Key=name)
            return True
        except Exception:  # noqa: BLE001 — botocore raises ClientError on 404; treat all as absent
            return False
    return (_LOCAL_DIR / name).exists()


def compute_sha256(name: str) -> str:
    """Return the SHA-256 hex digest of the bytes at `name`.

    Used by the idempotent-no-op gate (comparing snapshot hash against the
    prior manifest's `input_sha256`) and by Phase 3's content-addressed bundle.
    """
    return hashlib.sha256(load_bytes(name)).hexdigest()
