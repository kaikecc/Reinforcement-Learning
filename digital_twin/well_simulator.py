"""Operational well simulator backed by physical equations and 3W statistics.

The simulator creates new sensor signals from simplified well-flow equations
calibrated with real, simulated and hand-drawn 3W instances. Its default output
matches the `Reinforcement-Learning/src/train_pipeline.py` data contract:

    timestamp, P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP, class, well, code
"""

from __future__ import annotations

import argparse
import os
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
EVENT_EFFECTS = {
    1: {"flow": 0.84, "choke": 0.95, "pressure": 0.10, "thermal": -0.08, "trend": 0.20, "osc": 0.10},
    2: {"flow": 0.18, "choke": 0.12, "pressure": 0.35, "thermal": -0.05, "trend": -0.80, "osc": 0.04},
    3: {"flow": 0.72, "choke": 0.85, "pressure": 0.20, "thermal": -0.02, "trend": 0.00, "osc": 0.90},
    4: {"flow": 0.82, "choke": 0.80, "pressure": 0.12, "thermal": -0.02, "trend": 0.00, "osc": 0.55},
    5: {"flow": 0.55, "choke": 0.90, "pressure": -0.15, "thermal": -0.04, "trend": -0.65, "osc": 0.08},
    6: {"flow": 0.48, "choke": 0.38, "pressure": 0.25, "thermal": -0.03, "trend": -0.45, "osc": 0.10},
    7: {"flow": 0.60, "choke": 0.55, "pressure": 0.22, "thermal": -0.02, "trend": -0.35, "osc": 0.06},
    8: {"flow": 0.42, "choke": 0.42, "pressure": 0.28, "thermal": -0.22, "trend": -0.55, "osc": 0.18},
    9: {"flow": 0.50, "choke": 0.48, "pressure": 0.24, "thermal": -0.18, "trend": -0.50, "osc": 0.16},
}


@dataclass(frozen=True)
class Scenario:
    """Configuration for one simulated well scenario."""

    event_code: int
    source: str = "any"
    normal_source: str = "any"
    normal_rows: int = 3600
    event_rows: int = 3600
    well_name: str = "TWIN-WELL-00001"
    noise_std: float = 0.0
    seed: int = 42
    start: str = "2026-01-01 00:00:00"


class DigitalTwinWellSimulator:
    """Physics-based simulator for 3W oil-well behavior."""

    def __init__(
        self,
        dataset_path: str | Path,
        variables: Iterable[str] = DEFAULT_VARIABLES,
        max_profile_files: int = 24,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.variables = list(variables)
        self.max_profile_files = max_profile_files
        self._profile_cache: dict[tuple[int, str], dict[str, dict[str, float]]] = {}
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

    def simulate(self, scenario: Scenario) -> pd.DataFrame:
        """Return one synthetic operational timeline for a well.

        The timeline starts with class 0 normal behavior and then switches to
        the requested event class. Sensor values are created from physical
        well-flow equations and calibrated by 3W statistics, not replayed from
        original 3W rows.
        """
        rng = np.random.default_rng(scenario.seed)
        normal, state = self._synthesize_segment(
            class_code=0,
            source=scenario.normal_source,
            rows=scenario.normal_rows,
            rng=rng,
        )
        if scenario.event_code == 0:
            event, _ = self._synthesize_segment(
                class_code=0,
                source=scenario.normal_source,
                rows=scenario.event_rows,
                rng=rng,
                initial_state=state,
            )
        else:
            event, _ = self._synthesize_segment(
                class_code=scenario.event_code,
                source=scenario.source,
                rows=scenario.event_rows,
                rng=rng,
                initial_state=state,
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

    def simulate_replay(self, scenario: Scenario) -> pd.DataFrame:
        """Replay 3W rows as the old simulator did.

        This method is preserved for comparisons and debugging, but the default
        simulator path no longer calls it.
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

    def synthesize_fault_chunk(
        self,
        class_code: int,
        source: str,
        rows: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Create a fault-only physical segment without replaying dataset rows."""
        frame, _ = self._synthesize_segment(
            class_code=class_code,
            source=source,
            rows=rows,
            rng=rng,
        )
        return frame

    def simulate_many(
        self,
        event_code: int,
        scenarios: int,
        source: str = "any",
        normal_source: str = "any",
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

    def _synthesize_segment(
        self,
        class_code: int,
        source: str,
        rows: int,
        rng: np.random.Generator,
        initial_state: dict[str, float] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        profile = self._profile(class_code, source, rng)
        if rows <= 0:
            empty = pd.DataFrame(columns=["timestamp", *self.variables, "class"])
            return empty, initial_state or {"q": 0.0, "p_res": 1.0, "temperature": 1.0}

        if class_code == 0:
            return self._synthesize_normal_segment(profile, rows, rng, initial_state)

        effect = EVENT_EFFECTS.get(class_code, {"flow": 1.0, "choke": 1.0, "pressure": 0.0, "thermal": 0.0, "trend": 0.0, "osc": 0.04})
        t = np.linspace(0.0, 1.0, rows)
        dt = 1.0 / max(rows, 1)

        pressure_scale = self._profile_scale(profile, ["P-PDG", "P-TPT", "P-MON-CKP"])
        temp_scale = self._profile_scale(profile, ["T-TPT", "T-JUS-CKP"])
        p_res = (
            initial_state["p_res"]
            if initial_state
            else self._mean(profile, "P-PDG") + 1.8 * pressure_scale
        )
        q = initial_state["q"] if initial_state else rng.uniform(0.55, 0.85)
        temp = initial_state["temperature"] if initial_state else self._mean(profile, "T-TPT")
        base_choke = rng.uniform(0.55, 0.95)
        tau_q = rng.uniform(0.025, 0.080)
        tau_p = rng.uniform(0.015, 0.055)
        tau_t = rng.uniform(0.020, 0.070)
        phase = rng.uniform(0.0, 2.0 * np.pi)

        q_series = np.empty(rows)
        p_res_series = np.empty(rows)
        temp_series = np.empty(rows)
        choke_series = np.empty(rows)
        for index, frac in enumerate(t):
            event_ramp = self._smoothstep(frac)
            osc = effect["osc"] * np.sin(2.0 * np.pi * (3.0 + 4.0 * event_ramp) * frac + phase)
            choke_target = np.clip(
                base_choke * effect["choke"]
                + 0.035 * np.sin(2.0 * np.pi * frac + phase)
                + 0.015 * rng.normal(),
                0.05,
                1.0,
            )
            flow_target = np.clip(
                effect["flow"]
                + effect["trend"] * (frac - 0.5) * 0.45
                + osc,
                0.02,
                1.80,
            )
            p_target = (
                self._mean(profile, "P-PDG")
                + effect["pressure"] * pressure_scale * event_ramp
                + 0.10 * pressure_scale * np.sin(2.0 * np.pi * 0.25 * frac + phase)
            )
            temp_target = (
                self._mean(profile, "T-TPT")
                + effect["thermal"] * temp_scale * event_ramp
                - 0.08 * temp_scale * (q - 0.7)
            )

            inflow = flow_target * choke_target * np.sqrt(max(p_res - self._mean(profile, "P-MON-CKP"), 1.0) / max(pressure_scale, 1.0))
            q += (inflow - q) * min(dt / tau_q, 1.0)
            p_res += (p_target - p_res) * min(dt / tau_p, 1.0)
            temp += (temp_target - temp) * min(dt / tau_t, 1.0)

            q_series[index] = q
            p_res_series[index] = p_res
            temp_series[index] = temp
            choke_series[index] = choke_target

        liquid_head = 0.22 * pressure_scale * q_series
        tubing_loss = 0.14 * pressure_scale * np.square(q_series)
        choke_drop = 0.18 * pressure_scale * np.square(q_series / np.maximum(choke_series, 0.08))
        thermal_drop = 0.06 * temp_scale * q_series

        raw = pd.DataFrame(
            {
                "P-PDG": p_res_series - liquid_head + 0.05 * pressure_scale * q_series,
                "P-TPT": p_res_series - liquid_head - tubing_loss,
                "P-MON-CKP": p_res_series - liquid_head - tubing_loss - choke_drop,
                "T-TPT": temp_series - thermal_drop,
                "T-JUS-CKP": temp_series - thermal_drop - 0.04 * temp_scale * np.sqrt(np.maximum(choke_drop, 0.0) / max(pressure_scale, 1.0)),
            }
        )
        raw = self._apply_event_shape(raw, class_code, profile, t, phase, rng)

        frame = pd.DataFrame()
        for variable in self.variables:
            signal = raw[variable].to_numpy(dtype=float)
            frame[variable] = self._calibrate_signal(signal, profile[variable], rng)
        frame = self._enforce_event_trends(frame, class_code, profile)
        frame["timestamp"] = pd.NaT
        frame["class"] = class_code
        return frame[["timestamp", *self.variables, "class"]], {
            "q": float(q_series[-1]),
            "p_res": float(p_res_series[-1]),
            "temperature": float(temp_series[-1]),
        }

    def _synthesize_normal_segment(
        self,
        profile: dict[str, dict[str, float]],
        rows: int,
        rng: np.random.Generator,
        initial_state: dict[str, float] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        frame = pd.DataFrame()
        for variable in self.variables:
            stats = profile[variable]
            median = float(stats["median"])
            iqr = max(float(stats.get("iqr", 0.0)), 1.0 if variable.startswith("P-") else 0.05)
            if bool(stats.get("frozen", False)):
                frame[variable] = np.full(rows, median, dtype=float)
                continue

            operating_point = float(
                np.clip(
                    median + rng.normal(0.0, 0.12 * iqr),
                    float(stats["q1"]),
                    float(stats["q3"]),
                )
            )
            cap = 0.004 * max(abs(operating_point), 1.0) if variable.startswith("P-") else 0.18
            amplitude = min(0.12 * iqr, cap)
            amplitude = max(amplitude, 1.0 if variable.startswith("P-") else 0.02)

            innovations = rng.normal(0.0, amplitude * 0.08, rows)
            ar = np.zeros(rows, dtype=float)
            for index in range(1, rows):
                ar[index] = 0.985 * ar[index - 1] + innovations[index]
            seasonal = 0.25 * amplitude * np.sin(
                np.linspace(0.0, 2.0 * np.pi, rows) + rng.uniform(0.0, 2.0 * np.pi)
            )
            signal = operating_point + ar + seasonal

            lower = max(float(stats["whisker_low"]), operating_point - 2.5 * amplitude)
            upper = min(float(stats["whisker_high"]), operating_point + 2.5 * amplitude)
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                lower = operating_point - 2.5 * amplitude
                upper = operating_point + 2.5 * amplitude
            frame[variable] = np.clip(signal, lower, upper)

        frame["timestamp"] = pd.NaT
        frame["class"] = 0
        return frame[["timestamp", *self.variables, "class"]], {
            "q": float(rng.uniform(0.55, 0.85) if initial_state is None else initial_state.get("q", 0.7)),
            "p_res": float(frame["P-PDG"].median() if "P-PDG" in frame else 1.0),
            "temperature": float(frame["T-TPT"].median() if "T-TPT" in frame else 1.0),
        }

    def _apply_event_shape(
        self,
        raw: pd.DataFrame,
        class_code: int,
        profile: dict[str, dict[str, float]],
        t: np.ndarray,
        phase: float,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        shaped = raw.copy()
        if class_code == 0:
            return shaped

        ramp = np.array([self._smoothstep(value) for value in t])
        slow = ramp - 0.5
        periodic = np.sin(2.0 * np.pi * 4.5 * t + phase)
        irregular = (
            np.sin(2.0 * np.pi * 2.7 * t + phase)
            + 0.55 * np.sin(2.0 * np.pi * 6.3 * t + 0.4 * phase)
            + 0.25 * rng.normal(0.0, 1.0, len(t))
        )

        def scale(variable: str, minimum: float) -> float:
            stats = profile.get(variable, {})
            return max(float(stats.get("iqr", 0.0)), minimum)

        def add(variable: str, pattern: np.ndarray, amount: float) -> None:
            if variable in shaped.columns:
                shaped[variable] = shaped[variable] + amount * pattern

        pressure_floor = 5.0e4
        if class_code == 1:
            # Abrupt BSW: this 3W signature is enforced again after calibration:
            # P-TPT/P-MON-CKP fall while T-TPT/T-JUS-CKP rise.
            add("P-PDG", ramp, 1.4 * scale("P-PDG", pressure_floor))
            add("P-TPT", ramp, -1.2 * scale("P-TPT", pressure_floor))
            add("P-MON-CKP", ramp, -1.4 * scale("P-MON-CKP", pressure_floor))
            add("T-TPT", ramp, 1.4 * scale("T-TPT", 0.2))
            add("T-JUS-CKP", ramp, 1.2 * scale("T-JUS-CKP", 0.2))
        elif class_code == 2:
            # Spurious DHSV closure: flow collapses toward shutdown.
            add("P-PDG", ramp, 0.7 * scale("P-PDG", pressure_floor))
            add("P-TPT", ramp, -1.4 * scale("P-TPT", pressure_floor))
            add("P-MON-CKP", ramp, -1.6 * scale("P-MON-CKP", pressure_floor))
            add("T-TPT", ramp, -1.2 * scale("T-TPT", 0.2))
            add("T-JUS-CKP", ramp, -1.4 * scale("T-JUS-CKP", 0.2))
        elif class_code == 3:
            for variable in self.variables:
                add(variable, periodic, 1.4 * scale(variable, pressure_floor if variable.startswith("P-") else 0.2))
        elif class_code == 4:
            for variable in self.variables:
                add(variable, irregular, 0.65 * scale(variable, pressure_floor if variable.startswith("P-") else 0.2))
        elif class_code == 5:
            add("P-PDG", slow, -0.9 * scale("P-PDG", pressure_floor))
            add("P-TPT", ramp, -1.0 * scale("P-TPT", pressure_floor))
            add("P-MON-CKP", ramp, -1.2 * scale("P-MON-CKP", pressure_floor))
            add("T-TPT", ramp, -0.6 * scale("T-TPT", 0.2))
            add("T-JUS-CKP", ramp, -0.7 * scale("T-JUS-CKP", 0.2))
        elif class_code in {6, 7}:
            add("P-PDG", ramp, 0.4 * scale("P-PDG", pressure_floor))
            add("P-TPT", ramp, 0.8 * scale("P-TPT", pressure_floor))
            add("P-MON-CKP", ramp, 1.2 * scale("P-MON-CKP", pressure_floor))
            add("T-TPT", ramp, -0.3 * scale("T-TPT", 0.2))
            add("T-JUS-CKP", ramp, -0.4 * scale("T-JUS-CKP", 0.2))
        elif class_code in {8, 9}:
            add("P-PDG", ramp, 0.5 * scale("P-PDG", pressure_floor))
            add("P-TPT", ramp, -0.8 * scale("P-TPT", pressure_floor))
            add("P-MON-CKP", ramp, -0.9 * scale("P-MON-CKP", pressure_floor))
            add("T-TPT", ramp, -1.8 * scale("T-TPT", 0.2))
            add("T-JUS-CKP", ramp, -1.6 * scale("T-JUS-CKP", 0.2))
        return shaped

    def _enforce_event_trends(
        self,
        frame: pd.DataFrame,
        class_code: int,
        profile: dict[str, dict[str, float]],
    ) -> pd.DataFrame:
        rules = self._event_trend_rules(class_code)
        if not rules or frame.empty:
            return frame

        adjusted = frame.copy()
        ramp = np.linspace(0.0, 1.0, len(adjusted))
        for variable, direction in rules.items():
            if variable not in adjusted.columns or direction == 0:
                continue
            stats = profile.get(variable, {})
            if bool(stats.get("frozen", False)):
                continue

            values = pd.to_numeric(adjusted[variable], errors="coerce").to_numpy(dtype=float)
            if len(values) < 2 or not np.isfinite(values).any():
                continue

            current_delta = float(np.nanmedian(values[-max(3, len(values) // 10) :]) - np.nanmedian(values[: max(3, len(values) // 10)]))
            scale = self._trend_scale(variable, stats)
            desired_delta = float(direction) * max(abs(current_delta), scale)
            correction = ramp * (desired_delta - current_delta)
            trended = values + correction
            lower, upper = self._trend_bounds(stats, variable)
            trended = np.clip(trended, lower, upper)

            final_delta = float(np.nanmedian(trended[-max(3, len(trended) // 10) :]) - np.nanmedian(trended[: max(3, len(trended) // 10)]))
            if np.sign(final_delta) != np.sign(direction) or abs(final_delta) < 0.2 * scale:
                center = float(stats.get("median", np.nanmedian(values)))
                amplitude = min(0.5 * scale, max(abs(upper - lower) * 0.35, 1e-6))
                start = center - float(direction) * amplitude
                end = center + float(direction) * amplitude
                trended = np.linspace(start, end, len(values))
                trended = np.clip(trended, lower, upper)

            adjusted[variable] = trended
        return adjusted

    @staticmethod
    def _event_trend_rules(class_code: int) -> dict[str, int]:
        return {
            1: {"P-PDG": 1, "P-TPT": -1, "T-TPT": 1, "P-MON-CKP": -1, "T-JUS-CKP": 1},
            2: {"P-PDG": 1, "P-TPT": -1, "T-TPT": -1, "P-MON-CKP": -1, "T-JUS-CKP": -1},
            5: {"P-PDG": -1, "P-TPT": -1, "T-TPT": -1, "P-MON-CKP": -1, "T-JUS-CKP": -1},
            6: {"P-PDG": 1, "P-TPT": 1, "T-TPT": -1, "P-MON-CKP": 1, "T-JUS-CKP": -1},
            7: {"P-PDG": 1, "P-TPT": 1, "T-TPT": -1, "P-MON-CKP": 1, "T-JUS-CKP": -1},
            8: {"P-PDG": 1, "P-TPT": -1, "T-TPT": -1, "P-MON-CKP": -1, "T-JUS-CKP": -1},
            9: {"P-PDG": 1, "P-TPT": -1, "T-TPT": -1, "P-MON-CKP": -1, "T-JUS-CKP": -1},
        }.get(class_code, {})

    @staticmethod
    def _trend_scale(variable: str, stats: dict[str, float]) -> float:
        iqr = float(stats.get("iqr", 0.0))
        if np.isfinite(iqr) and iqr > 0:
            return max(0.8 * iqr, 1.0 if variable.startswith("P-") else 0.05)
        median = abs(float(stats.get("median", 0.0)))
        return max(0.02 * median, 1.0 if variable.startswith("P-") else 0.05)

    @staticmethod
    def _trend_bounds(stats: dict[str, float], variable: str) -> tuple[float, float]:
        iqr = max(float(stats.get("iqr", 0.0)), 1.0 if variable.startswith("P-") else 0.05)
        lower = float(stats.get("whisker_low", stats.get("q1", 0.0) - 1.5 * iqr))
        upper = float(stats.get("whisker_high", stats.get("q3", 0.0) + 1.5 * iqr))
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            center = float(stats.get("median", 0.0))
            lower = center - 2.0 * iqr
            upper = center + 2.0 * iqr
        return lower, upper

    def _profile(
        self,
        class_code: int,
        source: str,
        rng: np.random.Generator,
    ) -> dict[str, dict[str, float]]:
        key = (class_code, source)
        if key in self._profile_cache:
            return self._profile_cache[key]

        files = self._candidate_files(class_code, source)
        if not files and source != "any":
            files = self._candidate_files(class_code, "any")
        if not files:
            raise ValueError(f"No files found for class={class_code}, source={source}")

        selected = [files[int(i)] for i in rng.permutation(len(files))[: self.max_profile_files]]
        chunks = []
        for path in selected:
            try:
                candidate = pd.read_parquet(path, columns=[*self.variables, "class"])
            except Exception:
                candidate = pd.read_parquet(path)
            available_variables = [
                column for column in self.variables if column in candidate.columns
            ]
            if not available_variables:
                continue
            candidate = candidate[
                [*available_variables, "class"]
                if "class" in candidate.columns
                else available_variables
            ]
            candidate = self._normalize_source_units(candidate, path)
            candidate = candidate.replace([np.inf, -np.inf], np.nan)
            if "class" in candidate.columns:
                if class_code == 0:
                    filtered = candidate[candidate["class"] == 0]
                else:
                    filtered = candidate[candidate["class"] != 0]
                if not filtered.empty:
                    candidate = filtered
            if not candidate.empty:
                chunk_profile = self._boxplot_profile(candidate[available_variables])
                if chunk_profile:
                    chunks.append(chunk_profile)

        if not chunks:
            raise ValueError(
                f"No complete rows found for class={class_code}, source={source}, "
                f"variables={self.variables}"
            )

        profile = {}
        fallback_profile = None
        for variable in self.variables:
            stats = [chunk[variable] for chunk in chunks if variable in chunk]
            if not stats:
                if class_code != 0:
                    if fallback_profile is None:
                        fallback_profile = self._profile(0, source, rng)
                    profile[variable] = dict(fallback_profile[variable])
                    profile[variable]["frozen"] = False
                    continue
                center = 0.0
                iqr = 1.0
                q05 = -1.0
                q95 = 1.0
                q1 = -0.5
                q3 = 0.5
                whisker_low = -1.0
                whisker_high = 1.0
                frozen = False
            else:
                center = float(np.median([stat["median"] for stat in stats]))
                q1 = float(np.median([stat["q1"] for stat in stats]))
                q3 = float(np.median([stat["q3"] for stat in stats]))
                iqr = max(float(np.median([stat["iqr"] for stat in stats])), 0.0)
                q05 = float(np.median([stat["q05"] for stat in stats]))
                q95 = float(np.median([stat["q95"] for stat in stats]))
                whisker_low = float(np.median([stat["whisker_low"] for stat in stats]))
                whisker_high = float(np.median([stat["whisker_high"] for stat in stats]))
                fallback = max(abs(center) * 0.005, 1.0 if variable.startswith("P-") else 0.05)
                freeze_threshold = max(
                    abs(center) * (1e-6 if variable.startswith("P-") else 1e-5),
                    1.0 if variable.startswith("P-") else 0.001,
                )
                frozen = iqr <= freeze_threshold
                if not np.isfinite(iqr) or iqr <= 0:
                    iqr = fallback
            spread = max(iqr / 1.349, 1e-6)
            profile[variable] = {
                "mean": center,
                "median": center,
                "std": spread,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "q05": q05,
                "q95": q95,
                "whisker_low": whisker_low,
                "whisker_high": whisker_high,
                "frozen": frozen,
            }

        self._profile_cache[key] = profile
        return profile

    def _boxplot_profile(self, frame: pd.DataFrame) -> dict[str, dict[str, float]]:
        profile = {}
        for variable in self.variables:
            if variable not in frame.columns:
                continue
            series = pd.to_numeric(frame[variable], errors="coerce").dropna()
            series = self._sanitize_profile_series(variable, series)
            if series.empty:
                continue
            q05, q1, q2, q3, q95 = series.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
            iqr = float(q3 - q1)
            whisker_low = float(max(series.min(), q1 - 1.5 * iqr))
            whisker_high = float(min(series.max(), q3 + 1.5 * iqr))
            profile[variable] = {
                "q05": float(q05),
                "q1": float(q1),
                "median": float(q2),
                "q3": float(q3),
                "q95": float(q95),
                "iqr": iqr,
                "whisker_low": whisker_low,
                "whisker_high": whisker_high,
            }
        return profile

    def _normalize_source_units(self, frame: pd.DataFrame, path: Path) -> pd.DataFrame:
        normalized = frame.copy()
        if path.name.upper().startswith(SOURCE_PREFIXES["drawn"]):
            for variable in self.variables:
                if variable.startswith("P-") and variable in normalized.columns:
                    normalized[variable] = pd.to_numeric(
                        normalized[variable], errors="coerce"
                    ) * 98066.5
        return normalized

    @staticmethod
    def _sanitize_profile_series(variable: str, series: pd.Series) -> pd.Series:
        clean = series[np.isfinite(series)]
        if variable.startswith("P-"):
            clean = clean[clean > 0]
            if not clean.empty:
                low, high = clean.quantile([0.01, 0.99])
                clean = clean[(clean >= low) & (clean <= high)]
        elif variable.startswith("T-"):
            clean = clean[(clean > -50) & (clean < 250)]
            if not clean.empty:
                low, high = clean.quantile([0.01, 0.99])
                clean = clean[(clean >= low) & (clean <= high)]
        return clean

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def _mean(profile: dict[str, dict[str, float]], variable: str) -> float:
        return float(profile.get(variable, {"mean": 0.0})["mean"])

    @staticmethod
    def _profile_scale(profile: dict[str, dict[str, float]], variables: list[str]) -> float:
        spreads = [
            abs(profile[variable]["q95"] - profile[variable]["q05"])
            for variable in variables
            if variable in profile
        ]
        finite = [value for value in spreads if np.isfinite(value) and value > 0]
        return float(np.median(finite)) if finite else 1.0

    @staticmethod
    def _calibrate_signal(
        signal: np.ndarray,
        stats: dict[str, float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        if bool(stats.get("frozen", False)):
            return np.full(len(signal), float(stats["median"]), dtype=float)

        raw_std = float(np.std(signal))
        if not np.isfinite(raw_std) or raw_std <= 0:
            normalized = np.zeros_like(signal, dtype=float)
        else:
            normalized = (signal - float(np.mean(signal))) / raw_std

        target_iqr = max(float(stats["iqr"]), 1e-6)
        target_std = max(target_iqr / 1.349, 1e-6)
        target_median = float(stats["median"]) + rng.normal(0.0, 0.03 * target_iqr)
        calibrated = target_median + normalized * target_std
        jitter = rng.normal(0.0, 0.015 * target_iqr, len(calibrated))
        lower = float(stats["whisker_low"])
        upper = float(stats["whisker_high"])
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            lower = float(stats["q1"]) - 1.5 * target_iqr
            upper = float(stats["q3"]) + 1.5 * target_iqr
        return np.clip(calibrated + jitter, lower, upper)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a 3W digital twin well.")
    parser.add_argument(
        "--dataset-path",
        default=str(
            Path(os.environ.get("THREE_W_ROOT", Path(__file__).resolve().parents[1].parent / "3W"))
            / "dataset"
        ),
        help="Path to the 3W dataset directory.",
    )
    parser.add_argument("--event-code", type=int, default=1)
    parser.add_argument("--source", choices=["real", "simulated", "drawn", "any"], default="any")
    parser.add_argument("--normal-source", choices=["real", "simulated", "drawn", "any"], default="any")
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
