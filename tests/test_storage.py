"""Unit tests for the storage switch (`src/storage.py`), local backend.

Every pipeline artifact flows through this module, so its round-trip
behaviour underpins all of the integration tests. The autouse
`_isolate_storage` fixture already points `_LOCAL_DIR` at a temp dir.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import storage


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def test_csv_roundtrip(frame):
    storage.save_csv("tables/t.csv", frame, index=False)
    assert storage.exists("tables/t.csv")
    out = storage.load_csv("tables/t.csv")
    pd.testing.assert_frame_equal(out, frame)


def test_csv_atomic_roundtrip(frame):
    storage.save_csv_atomic("tables/t_atomic.csv", frame, index=False)
    out = storage.load_csv("tables/t_atomic.csv")
    pd.testing.assert_frame_equal(out, frame)


def test_json_roundtrip():
    obj = {"run_id": "abc", "rows": 5, "nested": {"ok": True}}
    storage.save_json("tables/m.json", obj, indent=2)
    assert storage.load_json("tables/m.json") == obj


def test_text_and_bytes_roundtrip():
    storage.save_text("notes/readme.txt", "hello")
    storage.save_text_atomic("notes/readme_atomic.txt", "world")
    storage.save_bytes("blobs/raw.bin", b"\x00\x01\x02")
    assert storage.load_bytes("notes/readme.txt") == b"hello"
    assert storage.load_bytes("notes/readme_atomic.txt") == b"world"
    assert storage.load_bytes("blobs/raw.bin") == b"\x00\x01\x02"


def test_exists_is_false_for_missing_key():
    assert not storage.exists("tables/nope.csv")


def test_compute_sha256_is_stable():
    storage.save_bytes("blobs/x.bin", b"determinism")
    first = storage.compute_sha256("blobs/x.bin")
    second = storage.compute_sha256("blobs/x.bin")
    assert first == second
    assert len(first) == 64  # hex-encoded SHA-256


def test_save_figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    storage.save_figure("figures/f.png", fig=fig)
    plt.close(fig)
    assert storage.exists("figures/f.png")
    assert storage.load_bytes("figures/f.png")[:8] == b"\x89PNG\r\n\x1a\n"
