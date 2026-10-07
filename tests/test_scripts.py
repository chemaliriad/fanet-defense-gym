import csv
import gzip
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def experiments() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_experiments", ROOT / "scripts" / "run_experiments.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rows() -> list[dict[str, Any]]:
    return [
        {"policy": 'watchdog, "tuned"', "seed": -1, "ret": 1.25, "suite": "val"},
        {"policy": "défense\nwatchdog", "seed": 0, "ret": -0.5, "suite": "test"},
    ]


def test_write_episodes_csv_round_trip(
    experiments: ModuleType, rows: list[dict[str, Any]], tmp_path: Path
) -> None:
    path = tmp_path / "episodes.csv.gz"
    experiments.write_episodes_csv(rows, path)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(rows[0])
        actual = list(reader)
    assert len(actual) == len(rows)
    assert actual == [{key: str(value) for key, value in row.items()} for row in rows]


def test_write_episodes_csv_deterministic(
    experiments: ModuleType, rows: list[dict[str, Any]], tmp_path: Path
) -> None:
    first, second = tmp_path / "first.csv.gz", tmp_path / "second.csv.gz"
    experiments.write_episodes_csv(rows, first)
    experiments.write_episodes_csv(rows, second)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes()[4:8] == b"\x00\x00\x00\x00"


def test_bench_smoke() -> None:
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT / "src"),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    result = subprocess.run(
        [sys.executable, "scripts/bench.py", "--episodes", "1", "--specs", "50"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ms per swarm step" in result.stdout
