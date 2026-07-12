from pathlib import Path

import numpy as np
import pandas as pd

from classes._LoadInstances import LoadInstances


def write_instance(root: Path, class_code: int, class_value: int) -> None:
    class_dir = root / str(class_code)
    class_dir.mkdir(parents=True, exist_ok=True)
    rows = 8
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="s"),
            "P-PDG": np.arange(rows, dtype=float) + 100,
            "P-TPT": np.arange(rows, dtype=float) + 80,
            "T-TPT": np.arange(rows, dtype=float) + 60,
            "P-MON-CKP": np.arange(rows, dtype=float) + 40,
            "T-JUS-CKP": np.arange(rows, dtype=float) + 30,
            "class": class_value,
        }
    )
    frame.to_parquet(class_dir / f"WELL-00001_2026010100000{class_code}.parquet", index=False)


def test_load_instance_and_prepare_data(tmp_path: Path) -> None:
    write_instance(tmp_path, class_code=0, class_value=0)
    write_instance(tmp_path, class_code=1, class_value=1)

    instances = LoadInstances(str(tmp_path))
    dataset, arrays = instances.load_instance_with_numpy(
        {0: "Normal", 1: "Abrupt Increase of BSW"},
        type_instance="real",
    )

    assert len(arrays) == 2
    assert dataset.shape[1] == 9

    train, test, validation = instances.data_preparation(dataset[:, :-1], 0.5)

    assert train.shape[1] == 6
    assert test.shape[1] == 6
    assert validation.shape[1] == 8
    assert instances.scaler is not None
