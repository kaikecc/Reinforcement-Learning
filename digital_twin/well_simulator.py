"""Operational well simulator backed by the 3W dataset.

The simulator replays real/simulated 3W instances as a digital twin stream:
normal operation first, then an injected failure event. Its default output
matches the `Reinforcement-Learning/src/train_pipeline.py` data contract:

    timestamp, P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP, class, well, code
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_VARIABLES = ["P-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP"]
OUTPUT_COLUMNS = ["timestamp", *DEFAULT_VARIABLES, "class", "well", "code"]
SOURCE_PREFIXES = {
    "real": "WELL",
    "simulated": "SIMULATED",
    "drawn": "DRAWN",
}


@dataclass(frozen=True)
class Scenario:
    """Configuration for one simulated well scenario."""

    event_code: int
    source: str = "real"
    normal_source: str = "real"
    normal_rows: int = 3600
    event_rows: int = 3600
    well_name: str = "TWIN-WELL-00001"
    noise_std: float = 0.0
    seed: int = 42
    start: str = "2026-01-01 00:00:00"


class DigitalTwinWellSimulator:
    """Replay-based simulator for 3W oil-well behavior."""

    def __init__(
        self,
        dataset_path: str | Path,
        variables: Iterable[str] = DEFAULT_VARIABLES,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.variables = list(variables)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

    def simulate(self, scenario: Scenario) -> pd.DataFrame:
        """Return one synthetic operational timeline for a well.

        The timeline starts with class 0 normal behavior and then switches to
        the requested event class. Sensor values come from 3W files and can be
        perturbed with small Gaussian noise.
        """
        rng = np.random.default_rng(scenario.seed)
        normal = self._load_segment(
            class_code=0,
            source=scenario.normal_source,
            rows=scenario.normal_rows,
            rng=rng,
        )
        if scenario.event_code == 0:
            event = self._load_segment(
                class_code=0,
                source=scenario.normal_source,
                rows=scenario.event_rows,
                rng=rng,
            )
        else:
            event = self._load_segment(
                class_code=scenario.event_code,
                source=scenario.source,
                rows=scenario.event_rows,
                rng=rng,
            )

        frame = pd.concat([normal, event], ignore_index=True)
        frame["timestamp"] = pd.date_range(
            start=pd.Timestamp(scenario.start),
            periods=len(frame),
            freq="s",
        )
        frame["well"] = scenario.well_name
        frame["code"] = scenario.event_code
        frame = frame[OUTPUT_COLUMNS]

        if scenario.noise_std > 0:
            frame = self._add_noise(frame, scenario.noise_std, rng)
        return frame

    def simulate_many(
        self,
        event_code: int,
        scenarios: int,
        source: str = "real",
        normal_source: str = "real",
        normal_rows: int = 3600,
        event_rows: int = 3600,
        noise_std: float = 0.0,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate multiple simulated wells for the same event code."""
        frames = []
        base_start = pd.Timestamp("2026-01-01 00:00:00")
        for index in range(scenarios):
            scenario = Scenario(
                event_code=event_code,
                source=source,
                normal_source=normal_source,
                normal_rows=normal_rows,
                event_rows=event_rows,
                well_name=f"TWIN-WELL-{index + 1:05d}",
                noise_std=noise_std,
                seed=seed + index,
                start=str(base_start + pd.Timedelta(days=index)),
            )
            frames.append(self.simulate(scenario))
        return pd.concat(frames, ignore_index=True)

    def to_pipeline_numpy(self, frame: pd.DataFrame) -> np.ndarray:
        """Convert a simulated frame to the numpy array used by train_pipeline."""
        export = frame[OUTPUT_COLUMNS].copy()
        export["timestamp"] = pd.to_datetime(export["timestamp"]).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        return export.to_numpy()

    def save(self, frame: pd.DataFrame, output_path: str | Path) -> Path:
        """Save a simulation as parquet or CSV."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".parquet":
            frame.to_parquet(path, engine="pyarrow", compression="brotli", index=False)
        elif path.suffix.lower() == ".csv":
            frame.to_csv(path, index=False)
        else:
            raise ValueError("Output path must end with .parquet or .csv")
        return path

    def _load_segment(
        self,
        class_code: int,
        source: str,
        rows: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        files = self._candidate_files(class_code, source)
        if not files:
            raise ValueError(f"No files found for class={class_code}, source={source}")

        columns = ["timestamp", *self.variables, "class"]
        frame = None
        for file_index in rng.permutation(len(files)):
            path = files[int(file_index)]
            candidate = pd.read_parquet(path)
            if "timestamp" not in candidate.columns:
                candidate = candidate.reset_index()

            missing = [column for column in columns if column not in candidate.columns]
            if missing:
                continue

            candidate = candidate[columns].replace([np.inf, -np.inf], np.nan).dropna()
            if not candidate.empty:
                frame = candidate
                break

        if frame is None:
            raise ValueError(
                f"No complete rows found for class={class_code}, source={source}, "
                f"variables={self.variables}"
            )

        if len(frame) >= rows:
            start = int(rng.integers(0, len(frame) - rows + 1))
            segment = frame.iloc[start : start + rows].copy()
        else:
            repeats = int(np.ceil(rows / len(frame)))
            segment = pd.concat([frame] * repeats, ignore_index=True).iloc[:rows].copy()

        if class_code == 0:
            segment["class"] = 0
        else:
            segment["class"] = segment["class"].fillna(class_code)
            segment.loc[segment["class"] == 0, "class"] = class_code
        return segment[["timestamp", *self.variables, "class"]]

    def _candidate_files(self, class_code: int, source: str) -> list[Path]:
        if source not in SOURCE_PREFIXES and source != "any":
            raise ValueError("source must be one of: real, simulated, drawn, any")

        class_dir = self.dataset_path / str(class_code)
        if not class_dir.exists():
            return []

        files = sorted(class_dir.glob("*.parquet"))
        if source == "any":
            return files

        prefix = SOURCE_PREFIXES[source]
        if source == "real":
            return [path for path in files if path.name.upper().startswith(prefix)]
        return [path for path in files if path.name.upper().startswith(prefix)]

    def _add_noise(
        self,
        frame: pd.DataFrame,
        noise_std: float,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        noisy = frame.copy()
        for variable in self.variables:
            series = pd.to_numeric(noisy[variable], errors="coerce")
            scale = float(series.std(skipna=True))
            if not np.isfinite(scale) or scale == 0:
                scale = max(abs(float(series.mean(skipna=True) or 1.0)), 1.0)
            noisy[variable] = series + rng.normal(0.0, noise_std * scale, len(series))
        return noisy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a 3W digital twin well.")
    parser.add_argument(
        "--dataset-path",
        default=str(Path(__file__).resolve().parents[1].parent / "3W" / "dataset"),
        help="Path to the 3W dataset directory.",
    )
    parser.add_argument("--event-code", type=int, default=1)
    parser.add_argument("--source", choices=["real", "simulated", "drawn", "any"], default="real")
    parser.add_argument("--normal-source", choices=["real", "simulated", "drawn", "any"], default="real")
    parser.add_argument("--scenarios", type=int, default=1)
    parser.add_argument("--normal-rows", type=int, default=3600)
    parser.add_argument("--event-rows", type=int, default=3600)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "outputs" / "simulated_well.parquet"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulator = DigitalTwinWellSimulator(args.dataset_path)
    frame = simulator.simulate_many(
        event_code=args.event_code,
        scenarios=args.scenarios,
        source=args.source,
        normal_source=args.normal_source,
        normal_rows=args.normal_rows,
        event_rows=args.event_rows,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    path = simulator.save(frame, args.output)
    print(f"Saved simulation with {len(frame)} rows to: {path}")


if __name__ == "__main__":
    main()
