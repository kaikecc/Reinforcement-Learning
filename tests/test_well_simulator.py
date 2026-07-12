from pathlib import Path

import numpy as np
import pandas as pd

from well_simulator import DEFAULT_VARIABLES, DigitalTwinWellSimulator, Scenario


def write_3w_fixture(root: Path, class_code: int, class_value: int) -> None:
    class_dir = root / str(class_code)
    class_dir.mkdir(parents=True, exist_ok=True)
    rows = 80
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="s"),
            "P-PDG": np.linspace(1_000_000, 1_010_000, rows),
            "P-TPT": np.linspace(800_000, 805_000, rows),
            "T-TPT": np.linspace(70, 72, rows),
            "P-MON-CKP": np.linspace(400_000, 402_000, rows),
            "T-JUS-CKP": np.linspace(60, 61, rows),
            "class": class_value,
        }
    )
    frame.to_parquet(class_dir / f"WELL-00001_20260101000000.parquet", index=False)


def test_simulate_many_returns_pipeline_contract(tmp_path: Path) -> None:
    write_3w_fixture(tmp_path, class_code=0, class_value=0)
    write_3w_fixture(tmp_path, class_code=1, class_value=1)

    simulator = DigitalTwinWellSimulator(tmp_path, max_profile_files=1)
    frame = simulator.simulate_many(
        event_code=1,
        scenarios=2,
        normal_rows=10,
        event_rows=12,
        source="real",
        normal_source="real",
    )

    assert list(frame.columns) == [
        "timestamp",
        *DEFAULT_VARIABLES,
        "class",
        "well",
        "code",
    ]
    assert len(frame) == 44
    assert set(frame["class"]) == {0, 1}
    assert set(frame["code"]) == {1}


def test_to_pipeline_numpy_formats_timestamp(tmp_path: Path) -> None:
    write_3w_fixture(tmp_path, class_code=0, class_value=0)
    simulator = DigitalTwinWellSimulator(tmp_path, max_profile_files=1)
    frame = simulator.simulate(Scenario(event_code=0, normal_rows=2, event_rows=1))

    exported = simulator.to_pipeline_numpy(frame)

    assert exported.shape == (3, 9)
    assert isinstance(exported[0, 0], str)
