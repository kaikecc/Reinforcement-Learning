"""Real-time 3W digital twin simulator with a browser dashboard."""

from __future__ import annotations

import argparse
import os
import socket
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

try:
    from well_simulator import DEFAULT_VARIABLES, OUTPUT_COLUMNS, DigitalTwinWellSimulator, Scenario
except ModuleNotFoundError:
    from .well_simulator import DEFAULT_VARIABLES, OUTPUT_COLUMNS, DigitalTwinWellSimulator, Scenario


ROOT = Path(__file__).resolve().parents[1]
APP_VERSION = "realtime-dashboard-layout-v10"
DEFAULT_DATASET = ROOT.parent / "3W" / "dataset"
DEFAULT_MODEL_PATH = (
    ROOT
    / "models"
    / "Abrupt Increase of BSW"
    / "realtime_twin"
    / "DQN"
    / "150000"
    / "_DQN.zip"
)


@dataclass
class RuntimeConfig:
    dataset_path: str = str(DEFAULT_DATASET)
    event_code: int = 1
    source: str = "real"
    normal_source: str = "real"
    normal_rows: int = 600
    event_rows: int = 600
    rows_per_tick: int = 5
    tick_seconds: float = 0.5
    noise_std: float = 0.0
    seed: int = 42
    well_name: str = "TWIN-WELL-REALTIME"
    max_history_rows: int = 50000
    model_type: str = "DQN"
    model_path: str = str(DEFAULT_MODEL_PATH)
    scaler_path: str = ""
    use_online_scaler: bool = True


class RealtimeTwinState:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.lock = threading.Lock()
        self.running = False
        self.pointer = 0
        self.cycle = 0
        self.total_emitted = 0
        self.next_start = pd.Timestamp("2026-01-01 00:00:00")
        self.generated_rows: list[dict[str, Any]] = []
        self.model_error: str | None = None
        self.scaler_error: str | None = None
        self.last_prediction: dict[str, Any] | None = None
        self.manual_fault_active = False
        self.manual_fault_class = 1
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        self.timeline = self._build_timeline()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @staticmethod
    def _cast_config_value(current: Any, value: Any) -> Any:
        if isinstance(current, bool):
            if isinstance(value, str):
                return value.lower() in {"1", "true", "yes", "sim", "on"}
            return bool(value)
        return type(current)(value)

    def _load_model(self) -> Any | None:
        path = self._resolve_model_path(self.config.model_path)
        if path is None:
            self.model_error = (
                f"Modelo nao encontrado: {self.config.model_path}"
                if self.config.model_path
                else "Caminho do modelo nao informado"
            )
            return None

        try:
            if self.config.model_type == "DQN":
                from stable_baselines3 import DQN

                return DQN.load(path)
            if self.config.model_type == "PPO":
                from stable_baselines3 import PPO

                return PPO.load(path)
            if self.config.model_type == "A2C":
                from stable_baselines3 import A2C

                return A2C.load(path)
            if self.config.model_type == "RNA":
                from tensorflow.keras.models import load_model

                import sys

                src_path = ROOT / "src"
                if str(src_path) not in sys.path:
                    sys.path.append(str(src_path))
                from classes._F1Score import F1Score

                return load_model(path, custom_objects={"F1Score": F1Score})
        except Exception as exc:
            self.model_error = str(exc)
            return None

        self.model_error = f"Unsupported model_type: {self.config.model_type}"
        return None

    @staticmethod
    def _resolve_model_path(model_path: str) -> Path | None:
        if not model_path:
            return None

        path = Path(model_path)
        zip_path = path.with_suffix(".zip")
        if path.suffix != ".zip" and zip_path.exists():
            return zip_path

        if path.exists() and (path.is_file() or path.is_dir()):
            return path

        return None

    def _load_scaler(self) -> Any | None:
        path = Path(self.config.scaler_path)
        if not self.config.scaler_path:
            return None
        if not path.exists():
            self.scaler_error = f"Scaler not found: {path}"
            return None

        try:
            import joblib

            return joblib.load(path)
        except Exception as exc:
            self.scaler_error = str(exc)
            return None

    def _fit_online_scaler(self, frame: pd.DataFrame) -> None:
        if self.scaler is not None or not self.config.use_online_scaler:
            return

        try:
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(frame[DEFAULT_VARIABLES].to_numpy(dtype=np.float32))
            self.scaler = scaler
            self.scaler_error = None
        except Exception as exc:
            self.scaler_error = str(exc)

    def _build_observation(self, row: dict[str, Any]) -> np.ndarray:
        obs = np.array([[row[variable] for variable in DEFAULT_VARIABLES]], dtype=np.float32)
        if self.scaler is not None:
            obs = self.scaler.transform(obs).astype(np.float32)
        return obs[0]

    def _predict_action(self, obs: np.ndarray) -> int:
        if self.config.model_type == "RNA":
            probabilities = self.model.predict(np.atleast_2d(obs), verbose=0)
            return int(np.argmax(probabilities, axis=1)[0])

        action = self.model.predict(obs, deterministic=True)[0]
        return int(np.asarray(action).item())

    def _predict_row(self, row: dict[str, Any]) -> dict[str, Any]:
        if self.model is None:
            row["model_action"] = None
            row["expected_action"] = None
            row["model_status"] = "sem modelo"
            row["model_validation"] = "sem modelo"
            row["model_correct"] = None
            self.last_prediction = {
                "executed": False,
                "model_loaded": False,
                "model_type": self.config.model_type,
                "model_path": self.config.model_path,
                "error": self.model_error,
                "message": "Predicao nao executada porque o modelo nao foi carregado.",
            }
            return row

        try:
            started = time.perf_counter()
            obs = self._build_observation(row)
            action_value = self._predict_action(obs)
            elapsed_ms = (time.perf_counter() - started) * 1000
            expected_action = 0 if int(row["class"]) == 0 else 1
            correct = action_value == expected_action
            obs_values = [float(value) for value in np.asarray(obs, dtype=np.float32).tolist()]
            row["model_action"] = action_value
            row["expected_action"] = expected_action
            row["model_status"] = "normal" if action_value == 0 else "falha"
            row["model_validation"] = "correto" if correct else "incorreto"
            row["model_correct"] = correct
            row["prediction_time_ms"] = round(elapsed_ms, 3)
            self.model_error = None
            self.last_prediction = {
                "executed": True,
                "model_loaded": True,
                "model_type": self.config.model_type,
                "model_path": self.config.model_path,
                "timestamp": row.get("timestamp"),
                "class": int(row["class"]),
                "variables": DEFAULT_VARIABLES,
                "observation": obs_values,
                "action": action_value,
                "expected_action": expected_action,
                "status": row["model_status"],
                "validation": row["model_validation"],
                "correct": correct,
                "elapsed_ms": round(elapsed_ms, 3),
                "error": None,
            }
        except Exception as exc:
            self.model_error = str(exc)
            row["model_action"] = None
            row["expected_action"] = None
            row["model_status"] = "erro"
            row["model_validation"] = "erro"
            row["model_correct"] = None
            self.last_prediction = {
                "executed": False,
                "model_loaded": True,
                "model_type": self.config.model_type,
                "model_path": self.config.model_path,
                "timestamp": row.get("timestamp"),
                "error": str(exc),
                "message": "Erro ao executar a predicao do modelo.",
            }
        return row

    def _build_timeline(self) -> pd.DataFrame:
        simulator = DigitalTwinWellSimulator(self.config.dataset_path)
        scenario = Scenario(
            event_code=self.config.event_code,
            source=self.config.source,
            normal_source=self.config.normal_source,
            normal_rows=self.config.normal_rows,
            event_rows=self.config.event_rows,
            well_name=self.config.well_name,
            noise_std=self.config.noise_std,
            seed=self.config.seed + self.cycle,
            start=str(self.next_start),
        )
        frame = simulator.simulate(scenario)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self._fit_online_scaler(frame)
        return frame[OUTPUT_COLUMNS]

    def _start_next_cycle(self) -> None:
        if self.generated_rows:
            latest = pd.Timestamp(self.generated_rows[-1]["timestamp"])
            self.next_start = latest + pd.Timedelta(seconds=1)
        self.cycle += 1
        self.pointer = 0
        self.timeline = self._build_timeline()

    def _next_timestamp_start(self) -> pd.Timestamp:
        if self.generated_rows:
            return pd.Timestamp(self.generated_rows[-1]["timestamp"]) + pd.Timedelta(seconds=1)
        return pd.Timestamp(self.next_start)

    def _build_manual_fault_chunk(self, rows: int) -> pd.DataFrame:
        rng = np.random.default_rng(
            self.config.seed + self.total_emitted + self.manual_fault_class + self.cycle
        )
        simulator = DigitalTwinWellSimulator(self.config.dataset_path)
        frame = simulator._load_segment(
            class_code=int(self.manual_fault_class),
            source=self.config.source,
            rows=rows,
            rng=rng,
        )
        start = self._next_timestamp_start()
        frame["timestamp"] = pd.date_range(start=start, periods=len(frame), freq="s").strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        frame["class"] = int(self.manual_fault_class)
        frame["well"] = f"{self.config.well_name}-MANUAL"
        frame["code"] = int(self.manual_fault_class)
        return frame[OUTPUT_COLUMNS]

    def _run(self) -> None:
        while True:
            time.sleep(self.config.tick_seconds)
            with self.lock:
                if not self.running:
                    continue
                if self.manual_fault_active:
                    chunk = self._build_manual_fault_chunk(self.config.rows_per_tick)
                    end = self.pointer
                else:
                    if self.pointer >= len(self.timeline):
                        self._start_next_cycle()
                    end = min(self.pointer + self.config.rows_per_tick, len(self.timeline))
                    chunk = self.timeline.iloc[self.pointer:end]
                rows = [self._predict_row(row) for row in chunk.to_dict(orient="records")]
                self.generated_rows.extend(rows)
                if len(self.generated_rows) > self.config.max_history_rows:
                    self.generated_rows = self.generated_rows[-self.config.max_history_rows :]
                self.total_emitted += len(chunk)
                if not self.manual_fault_active:
                    self.pointer = end

    def start(self) -> None:
        with self.lock:
            self.running = True

    def pause(self) -> None:
        with self.lock:
            self.running = False

    def toggle_manual_fault(self, class_code: int | None = None) -> dict[str, Any]:
        with self.lock:
            if self.manual_fault_active:
                self.manual_fault_active = False
            else:
                if class_code is None or int(class_code) < 1:
                    raise ValueError("class_code must be >= 1 for a manual fault")
                self.manual_fault_class = int(class_code)
                self.manual_fault_active = True
                self.running = True
            return self.status_unlocked()

    def reset(self, updates: dict[str, Any] | None = None) -> None:
        with self.lock:
            if updates:
                for key, value in updates.items():
                    if hasattr(self.config, key) and value not in (None, ""):
                        current = getattr(self.config, key)
                        setattr(self.config, key, self._cast_config_value(current, value))
            self.running = False
            self.pointer = 0
            self.cycle = 0
            self.total_emitted = 0
            self.next_start = pd.Timestamp("2026-01-01 00:00:00")
            self.generated_rows = []
            self.model_error = None
            self.scaler_error = None
            self.last_prediction = None
            self.manual_fault_active = False
            self.manual_fault_class = 1
            self.model = self._load_model()
            self.scaler = self._load_scaler()
            self.timeline = self._build_timeline()

    def status_unlocked(self) -> dict[str, Any]:
            latest = self.generated_rows[-1] if self.generated_rows else None
            class_counts: dict[str, int] = {}
            for row in self.generated_rows:
                key = str(row["class"])
                class_counts[key] = class_counts.get(key, 0) + 1
            current_class = int(latest["class"]) if latest else 0
            evaluated = [row for row in self.generated_rows if row.get("model_correct") is not None]
            correct = sum(1 for row in evaluated if row["model_correct"])
            tp = tn = fp = fn = 0
            for row in evaluated:
                actual_fault = int(row["class"]) != 0
                predicted_fault = int(row["model_action"]) != 0
                if actual_fault and predicted_fault:
                    tp += 1
                elif not actual_fault and not predicted_fault:
                    tn += 1
                elif not actual_fault and predicted_fault:
                    fp += 1
                elif actual_fault and not predicted_fault:
                    fn += 1
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1_score = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            evaluated_count = len(evaluated)
            confusion_percent = {
                "TP": tp / evaluated_count if evaluated_count else 0.0,
                "TN": tn / evaluated_count if evaluated_count else 0.0,
                "FP": fp / evaluated_count if evaluated_count else 0.0,
                "FN": fn / evaluated_count if evaluated_count else 0.0,
            }
            latest_action = latest.get("model_action") if latest else None
            latest_expected = latest.get("expected_action") if latest else None
            return {
                "running": self.running,
                "pointer": self.pointer,
                "total_rows": len(self.timeline),
                "generated_rows": self.total_emitted,
                "buffer_rows": len(self.generated_rows),
                "cycle": self.cycle + 1,
                "cycle_progress": self.pointer / len(self.timeline) if len(self.timeline) else 0,
                "event_started": current_class != 0,
                "current_class": current_class,
                "manual_fault_active": self.manual_fault_active,
                "manual_fault_class": self.manual_fault_class,
                "model_loaded": self.model is not None,
                "model_error": self.model_error,
                "model_action": latest_action,
                "expected_action": latest_expected,
                "model_status": latest.get("model_status") if latest else None,
                "model_validation": latest.get("model_validation") if latest else None,
                "model_accuracy": correct / len(evaluated) if evaluated else None,
                "model_correct": correct,
                "model_evaluated": len(evaluated),
                "model_precision": precision,
                "model_recall": recall,
                "model_f1_score": f1_score,
                "confusion_matrix": {
                    "TP": tp,
                    "TN": tn,
                    "FP": fp,
                    "FN": fn,
                },
                "confusion_matrix_percent": confusion_percent,
                "prediction": self.last_prediction,
                "scaler_loaded": self.scaler is not None,
                "scaler_error": self.scaler_error,
                "scaler_mode": "arquivo" if self.config.scaler_path and self.scaler is not None else (
                    "online" if self.scaler is not None else "sem scaler"
                ),
                "latest": latest,
                "class_counts": class_counts,
                "variables": DEFAULT_VARIABLES,
                "columns": OUTPUT_COLUMNS,
                "config": asdict(self.config),
                "app_version": APP_VERSION,
                "served_from": str(Path(__file__).resolve()),
                "served_file": Path(__file__).name,
            }

    def status(self) -> dict[str, Any]:
        with self.lock:
            return self.status_unlocked()

    def window(self, limit: int) -> list[dict[str, Any]]:
        with self.lock:
            return self.generated_rows[-limit:]

    def samples(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.generated_rows if limit is None else self.generated_rows[-limit:]
            return list(rows)


def create_app(state: RealtimeTwinState) -> Flask:
    app = Flask(__name__)

    @app.after_request
    def add_no_cache_headers(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.headers["X-Realtime-Simulator-Version"] = APP_VERSION
        return response

    @app.get("/")
    def index() -> str:
        return DASHBOARD_HTML

    @app.get("/api/version")
    def api_version():
        return jsonify(
            {
                "app_version": APP_VERSION,
                "served_from": str(Path(__file__).resolve()),
                "served_file": Path(__file__).name,
                "pid": os.getpid(),
            }
        )

    @app.get("/api/status")
    def api_status():
        return jsonify(state.status())

    @app.get("/api/window")
    def api_window():
        limit = int(request.args.get("limit", "300"))
        return jsonify({"rows": state.window(limit), "status": state.status()})

    @app.get("/api/samples")
    def api_samples():
        limit_arg = request.args.get("limit")
        limit = int(limit_arg) if limit_arg else None
        return jsonify({"rows": state.samples(limit), "status": state.status()})

    @app.post("/api/control")
    def api_control():
        payload = request.get_json(silent=True) or {}
        action = payload.get("action")
        if action == "start":
            state.start()
        elif action == "pause":
            state.pause()
        elif action == "reset":
            state.reset(payload.get("config"))
        else:
            return jsonify({"error": "action must be start, pause, or reset"}), 400
        return jsonify(state.status())

    @app.post("/api/manual-fault")
    def api_manual_fault():
        payload = request.get_json(silent=True) or {}
        class_code = payload.get("class_code")
        try:
            status = state.toggle_manual_fault(int(class_code) if class_code is not None else None)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"status": status})

    return app


DASHBOARD_HTML = """<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>3W Real-Time Digital Twin</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --line: #d7dee6;
      --text: #18242f;
      --muted: #52616f;
      --accent: #0f766e;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    html { overflow-x: hidden; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      overflow-x: hidden;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 0; font-size: 24px; letter-spacing: 0; }
    .header-status { display: grid; gap: 2px; min-width: 0; text-align: right; }
    .header-status small {
      color: var(--muted);
      font-size: 12px;
      max-width: min(520px, 46vw);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    main { padding: 18px; max-width: 1560px; margin: 0 auto; }
    .toolbar, .dashboard-layout, .metric-grid, .grid, .charts-grid {
      display: grid;
      gap: 12px;
    }
    .toolbar {
      grid-template-columns: repeat(12, minmax(0, 1fr));
      align-items: end;
      margin-bottom: 12px;
      overflow: hidden;
    }
    button {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      border-radius: 6px;
      padding: 10px 14px;
      cursor: pointer;
      font-size: 14px;
      min-width: 0;
    }
    button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    button.danger { color: var(--danger); }
    label { display: grid; gap: 4px; min-width: 0; color: var(--muted); font-size: 12px; }
    input, select {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      min-width: 0;
      width: 100%;
      background: #fff;
      color: var(--text);
    }
    .control-button { grid-column: span 1; }
    .control-small { grid-column: span 1; }
    .control-medium { grid-column: span 2; }
    .control-path { grid-column: span 3; }
    .modelPath { width: 100%; min-width: 0; }
    .dashboard-layout {
      grid-template-columns: minmax(0, 1fr);
      margin-bottom: 12px;
    }
    .metric-section, .metric, section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    .metric-section h2 {
      margin: 0 0 10px;
      color: var(--text);
      font-size: 15px;
      text-align: left;
    }
    .metric-section.ok {
      background: #f0fbf6;
      border-color: #78d7ad;
    }
    .metric-section.bad {
      background: #fff1f1;
      border-color: #f2a8a8;
    }
    .metric-grid {
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    }
    .metric {
      text-align: center;
      min-height: 82px;
      display: grid;
      align-content: center;
    }
    .metric.ok {
      background: #dff8ed;
      border-color: #78d7ad;
    }
    .metric.bad {
      background: #fde7e7;
      border-color: #f2a8a8;
    }
    .metric.warn {
      background: #fff4d6;
      border-color: #f3cf72;
    }
    .metric strong {
      display: block;
      font-size: clamp(20px, 2vw, 26px);
      line-height: 1.1;
      margin-bottom: 3px;
      overflow-wrap: anywhere;
    }
    .metric span { color: var(--muted); }
    .grid { grid-template-columns: minmax(0, 1fr) minmax(340px, 420px); align-items: start; }
    .side-panel { display: grid; gap: 12px; }
    .charts-grid { grid-template-columns: repeat(auto-fit, minmax(min(420px, 100%), 1fr)); }
    section h2 { margin: 0 0 8px; font-size: 16px; }
    canvas {
      display: block;
      width: 100%;
      background: #fff;
    }
    .sensorCanvas { height: 260px; }
    #classesCanvas { height: 250px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    td { border-top: 1px solid var(--line); padding: 7px 5px; }
    td:first-child { color: var(--muted); width: 42%; }
    .prediction-status {
      display: inline-block;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      font-weight: 700;
      background: #eef2f6;
      color: var(--muted);
    }
    .prediction-status.ok { background: #dff8ed; color: #067647; }
    .prediction-status.bad { background: #fde7e7; color: var(--danger); }
    .prediction-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
      text-align: left;
    }
    .prediction-item {
      border-top: 1px solid var(--line);
      min-width: 0;
      padding-top: 8px;
    }
    .prediction-item span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 3px;
    }
    .prediction-item strong {
      display: block;
      overflow-wrap: anywhere;
      font-size: 14px;
    }
    .prediction-obs {
      grid-column: 1 / -1;
      font-family: Consolas, "Courier New", monospace;
      font-size: 12px;
      line-height: 1.45;
    }
    @media (max-width: 980px) {
      main { padding: 12px; }
      header { align-items: flex-start; flex-direction: column; }
      .header-status { width: 100%; text-align: left; }
      .header-status small { max-width: 100%; }
      .toolbar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .control-button, .control-small, .control-medium, .control-path { grid-column: span 1; }
      .control-path { grid-column: 1 / -1; }
      .modelPath { min-width: 0; width: 100%; }
      .grid, .charts-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 560px) {
      .toolbar { grid-template-columns: 1fr; }
      .control-button, .control-small, .control-medium, .control-path { grid-column: 1 / -1; }
      .metric-grid, .prediction-grid { grid-template-columns: 1fr; }
    }
    @media (min-width: 1180px) {
      .dashboard-layout { grid-template-columns: 1.1fr 1.2fr 0.9fr; align-items: stretch; }
    }
  </style>
</head>
<body>
  <header>
    <h1>3W Real-Time Digital Twin</h1>
    <div class="header-status">
      <div id="stateLabel">parado</div>
      <small id="versionLabel">carregando versao</small>
    </div>
  </header>
  <main>
    <div class="toolbar">
      <button class="primary control-button" onclick="control('start')">Iniciar</button>
      <button class="control-button" onclick="control('pause')">Pausar</button>
      <button class="danger control-button" onclick="resetSimulation()">Resetar</button>
      <label class="control-medium">Evento
        <select id="eventCode">
          <option value="0">0 - Normal</option>
          <option value="1">1 - Abrupt Increase of BSW</option>
          <option value="2">2 - Spurious Closure of DHSV</option>
          <option value="3">3 - Severe Slugging</option>
          <option value="4">4 - Flow Instability</option>
          <option value="5">5 - Rapid Productivity Loss</option>
          <option value="6">6 - Quick Restriction in PCK</option>
          <option value="7">7 - Scaling in PCK</option>
          <option value="8">8 - Hydrate in Production Line</option>
          <option value="9">9 - Hydrate in Service Line</option>
        </select>
      </label>
      <label class="control-small">Fonte
        <select id="source">
          <option value="real">real</option>
          <option value="simulated">simulated</option>
          <option value="drawn">drawn</option>
          <option value="any">any</option>
        </select>
      </label>
      <label class="control-small">Ruido
        <input id="noiseStd" type="number" min="0" max="1" step="0.01" value="0">
      </label>
      <label class="control-small">Modelo
        <select id="modelType">
          <option value="DQN">DQN</option>
          <option value="PPO">PPO</option>
          <option value="A2C">A2C</option>
          <option value="RNA">RNA</option>
        </select>
      </label>
      <label class="control-path">Caminho do modelo
        <input id="modelPath" class="modelPath" type="text" value="">
      </label>
      <label class="control-path">Caminho do scaler
        <input id="scalerPath" class="modelPath" type="text" value="">
      </label>
      <label class="control-small">Scaler online
        <select id="useOnlineScaler">
          <option value="true">ativo</option>
          <option value="false">inativo</option>
        </select>
      </label>
      <label class="control-small">Falha manual
        <input id="manualFaultCode" type="number" min="1" max="9" step="1" value="1">
      </label>
      <button id="manualFaultButton" class="danger control-button" onclick="toggleManualFault()">Testar falha</button>
    </div>
    <div class="dashboard-layout">
      <section class="metric-section" id="operationSection">
        <h2>Operacao em Tempo Real</h2>
        <div class="metric-grid">
      <div class="metric operation-card" id="rowsCard"><strong id="rows">0</strong><span>amostras emitidas</span></div>
      <div class="metric operation-card" id="progressCard"><strong id="progress">0%</strong><span>progresso do ciclo</span></div>
      <div class="metric operation-card" id="phaseCard"><strong id="phase">normal</strong><span>fase operacional</span></div>
      <div class="metric operation-card" id="faultCard"><strong id="fault">0</strong><span>classe atual</span></div>
      <div class="metric operation-card" id="cycleCard"><strong id="cycle">1</strong><span>ciclo</span></div>
        </div>
      </section>
      <section class="metric-section">
        <h2>Resultado da Predicao</h2>
        <div class="metric-grid">
      <div class="metric prediction-card" id="expectedActionCard"><strong id="expectedAction">-</strong><span>acao esperada</span></div>
      <div class="metric prediction-card" id="modelValidationCard"><strong id="modelValidation">-</strong><span>validacao modelo</span></div>
      <div class="metric"><strong id="modelAction">-</strong><span>ação do modelo</span></div>
      <div class="metric"><strong id="modelAccuracy">-</strong><span>acurácia online</span></div>
      <div class="metric prediction-card" id="modelPrecisionCard"><strong id="modelPrecision">-</strong><span>precisao</span></div>
      <div class="metric prediction-card" id="modelRecallCard"><strong id="modelRecall">-</strong><span>recall</span></div>
      <div class="metric prediction-card" id="modelF1ScoreCard"><strong id="modelF1Score">-</strong><span>F1 score</span></div>
      <div class="metric prediction-card" id="modelTPCard"><strong id="modelTP">0%</strong><span>TP</span></div>
      <div class="metric prediction-card" id="modelTNCard"><strong id="modelTN">0%</strong><span>TN</span></div>
      <div class="metric prediction-card" id="modelFPCard"><strong id="modelFP">0%</strong><span>FP</span></div>
      <div class="metric prediction-card" id="modelFNCard"><strong id="modelFN">0%</strong><span>FN</span></div>
        </div>
      </section>
      <section class="metric-section">
        <h2>Modelo e Normalizacao</h2>
        <div class="metric-grid">
      <div class="metric"><strong id="modelLoad">-</strong><span>modelo carregado</span></div>
      <div class="metric"><strong id="scalerMode">-</strong><span>normalizacao</span></div>
        </div>
      </section>
    </div>
    <div class="grid">
      <div class="charts-grid" id="sensorCharts"></div>
      <div class="side-panel">
        <section>
          <h2>Distribuição de Classes</h2>
          <canvas id="classesCanvas"></canvas>
        </section>
        <section>
          <h2>Predicao do Modelo</h2>
          <div id="predictionDetails"></div>
        </section>
        <section>
          <h2>Última Amostra</h2>
          <table id="latest"></table>
        </section>
      </div>
    </div>
  </main>
  <script>
    const variables = ["P-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP"];
    const colors = ["#0f766e", "#1d4ed8", "#b45309", "#7c3aed", "#be123c"];
    let controlsInitialized = false;

    function chartId(variable) {
      return `chart-${variable.replaceAll("-", "_")}`;
    }

    function setupSensorCharts() {
      const container = document.getElementById("sensorCharts");
      container.innerHTML = variables.map((variable) => `
        <section>
          <h2>${variable}</h2>
          <canvas class="sensorCanvas" id="${chartId(variable)}"></canvas>
        </section>
      `).join("");
    }

    function fitCanvas(canvas) {
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return {ctx, width: rect.width, height: rect.height};
    }

    function drawAxes(ctx, width, height, left, right, top, bottom) {
      ctx.lineWidth = 1;
      ctx.strokeStyle = "#e7edf3";
      for (let idx = 1; idx <= 4; idx += 1) {
        const y = top + (idx / 5) * (height - top - bottom);
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(width - right, y);
        ctx.stroke();
      }
      for (let idx = 1; idx <= 5; idx += 1) {
        const x = left + (idx / 6) * (width - left - right);
        ctx.beginPath();
        ctx.moveTo(x, top);
        ctx.lineTo(x, height - bottom);
        ctx.stroke();
      }
      ctx.strokeStyle = "#d7dee6";
      ctx.beginPath();
      ctx.moveTo(left, top);
      ctx.lineTo(left, height - bottom);
      ctx.lineTo(width - right, height - bottom);
      ctx.stroke();
      ctx.fillStyle = "#52616f";
      ctx.font = "12px Arial";
    }

    function drawVariable(rows, variable, color) {
      const canvas = document.getElementById(chartId(variable));
      if (!canvas) return;
      const {ctx, width, height} = fitCanvas(canvas);
      ctx.clearRect(0, 0, width, height);
      const left = 66, right = 18, top = 16, bottom = 42;
      drawAxes(ctx, width, height, left, right, top, bottom);
      if (!rows.length) return;

      const values = rows.map((row) => Number(row[variable])).filter((value) => Number.isFinite(value));
      if (!values.length) return;
      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = max - min || 1;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      rows.forEach((row, rowIdx) => {
        const value = Number(row[variable]);
        if (!Number.isFinite(value)) return;
        const x = left + (rowIdx / Math.max(rows.length - 1, 1)) * (width - left - right);
        const y = top + (1 - ((value - min) / span)) * (height - top - bottom);
        if (rowIdx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.fillStyle = "#52616f";
      ctx.fillText("tempo", width - right - 44, height - 12);
      ctx.save();
      ctx.translate(14, top + 88);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("valor", 0, 0);
      ctx.restore();
      ctx.fillText(max.toExponential(2), 6, top + 10);
      ctx.fillText(min.toExponential(2), 6, height - bottom);
    }

    function drawClasses(classCounts) {
      const canvas = document.getElementById("classesCanvas");
      const {ctx, width, height} = fitCanvas(canvas);
      ctx.clearRect(0, 0, width, height);
      const left = 44, right = 14, top = 16, bottom = 34;
      drawAxes(ctx, width, height, left, right, top, bottom);
      const labels = Object.keys(classCounts);
      if (!labels.length) return;
      const max = Math.max(...labels.map((label) => classCounts[label]), 1);
      const slot = (width - left - right) / labels.length;
      labels.forEach((label, idx) => {
        const barHeight = (classCounts[label] / max) * (height - top - bottom);
        const x = left + idx * slot + slot * 0.18;
        const y = height - bottom - barHeight;
        ctx.fillStyle = label === "0" ? "#0f766e" : "#b42318";
        ctx.fillRect(x, y, slot * 0.64, barHeight);
        ctx.fillStyle = "#52616f";
        ctx.fillText(label, x + slot * 0.22, height - 12);
        ctx.fillText(classCounts[label], x + slot * 0.12, Math.max(12, y - 6));
      });
    }

    async function control(action) {
      await fetch("/api/control", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({action})
      });
      await refresh();
    }
    async function resetSimulation() {
      await fetch("/api/control", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          action: "reset",
          config: {
            event_code: document.getElementById("eventCode").value,
            source: document.getElementById("source").value,
            noise_std: document.getElementById("noiseStd").value,
            model_type: document.getElementById("modelType").value,
            model_path: document.getElementById("modelPath").value,
            scaler_path: document.getElementById("scalerPath").value,
            use_online_scaler: document.getElementById("useOnlineScaler").value
          }
        })
      });
      await refresh();
    }
    async function toggleManualFault() {
      const classCode = Number(document.getElementById("manualFaultCode").value || 1);
      const response = await fetch("/api/manual-fault", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({class_code: classCode})
      });
      if (!response.ok) {
        const payload = await response.json();
        alert(payload.error || "Falha manual nao foi injetada.");
      }
      await refresh();
    }
    function renderLatest(row) {
      const table = document.getElementById("latest");
      if (!row) {
        table.innerHTML = "<tr><td>status</td><td>sem amostras</td></tr>";
        return;
      }
      table.innerHTML = Object.entries(row).map(([key, value]) =>
        `<tr><td>${key}</td><td>${value}</td></tr>`
      ).join("");
    }
    function formatPredictionValue(value) {
      if (value === null || value === undefined || value === "") return "-";
      if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
      if (typeof value === "boolean") return value ? "sim" : "nao";
      return String(value);
    }
    function formatPercent(value) {
      return value === null || value === undefined ? "-" : `${Math.round(value * 100)}%`;
    }
    function setMetricState(elementId, stateClass) {
      const card = document.getElementById(elementId)?.closest(".metric");
      if (!card) return;
      card.classList.remove("ok", "bad", "warn");
      if (stateClass) card.classList.add(stateClass);
    }
    function stateFromRatio(value, warnBelow = 0.7) {
      if (value === null || value === undefined) return "";
      return value >= warnBelow ? "ok" : "bad";
    }
    function updatePredictionCards(status) {
      const predictedFault = Number(status.model_action) !== 0;
      const expectedFault = Number(status.expected_action) !== 0;
      const hasPrediction = status.model_action !== null && status.model_action !== undefined;
      setMetricState("expectedAction", expectedFault ? "bad" : "ok");
      setMetricState("modelAction", hasPrediction ? (predictedFault ? "bad" : "ok") : "");
      setMetricState("modelValidation", status.model_validation === "correto" ? "ok" : (status.model_validation ? "bad" : ""));
      setMetricState("modelAccuracy", stateFromRatio(status.model_accuracy));
      setMetricState("modelPrecision", stateFromRatio(status.model_precision));
      setMetricState("modelRecall", stateFromRatio(status.model_recall));
      setMetricState("modelF1Score", stateFromRatio(status.model_f1_score));
      setMetricState("modelTP", status.confusion_matrix?.TP > 0 ? "bad" : "");
      setMetricState("modelTN", status.confusion_matrix?.TN > 0 ? "ok" : "");
      setMetricState("modelFP", status.confusion_matrix?.FP > 0 ? "bad" : "");
      setMetricState("modelFN", status.confusion_matrix?.FN > 0 ? "bad" : "");
    }
    function updateOperationCards(status) {
      const faultActive = status.manual_fault_active || status.event_started;
      const stateClass = faultActive ? "bad" : "ok";
      const section = document.getElementById("operationSection");
      section.classList.remove("ok", "bad");
      section.classList.add(stateClass);
      setMetricState("rows", status.generated_rows > 0 ? stateClass : "");
      setMetricState("progress", stateClass);
      setMetricState("phase", stateClass);
      setMetricState("fault", stateClass);
      setMetricState("cycle", stateClass);
    }
    function syncControls(status) {
      if (controlsInitialized) return;
      document.getElementById("eventCode").value = String(status.config.event_code);
      document.getElementById("source").value = status.config.source;
      document.getElementById("noiseStd").value = status.config.noise_std;
      document.getElementById("modelType").value = status.config.model_type;
      document.getElementById("modelPath").value = status.config.model_path;
      document.getElementById("scalerPath").value = status.config.scaler_path;
      document.getElementById("useOnlineScaler").value = String(status.config.use_online_scaler);
      document.getElementById("manualFaultCode").value = status.manual_fault_class || 1;
      controlsInitialized = true;
    }
    function renderPrediction(status) {
      const container = document.getElementById("predictionDetails");
      const prediction = status.prediction;
      const statusClass = prediction?.executed
        ? (prediction.correct ? "ok" : "bad")
        : (status.model_loaded ? "bad" : "");
      const label = prediction?.executed
        ? `executada - ${prediction.validation}`
        : (status.model_loaded ? "aguardando amostra" : "modelo nao carregado");
      const obsText = prediction?.observation
        ? prediction.variables.map((variable, idx) => `${variable}=${Number(prediction.observation[idx]).toFixed(4)}`).join(" | ")
        : "-";
      container.innerHTML = `
        <span class="prediction-status ${statusClass}">${label}</span>
        <div class="prediction-grid">
          <div class="prediction-item"><span>Modelo</span><strong>${formatPredictionValue(status.config.model_type)}</strong></div>
          <div class="prediction-item"><span>Carregado</span><strong>${formatPredictionValue(status.model_loaded)}</strong></div>
          <div class="prediction-item"><span>Timestamp</span><strong>${formatPredictionValue(prediction?.timestamp)}</strong></div>
          <div class="prediction-item"><span>Classe real</span><strong>${formatPredictionValue(prediction?.class ?? status.current_class)}</strong></div>
          <div class="prediction-item"><span>Acao prevista</span><strong>${formatPredictionValue(prediction?.action ?? status.model_action)}</strong></div>
          <div class="prediction-item"><span>Acao esperada</span><strong>${formatPredictionValue(prediction?.expected_action ?? status.expected_action)}</strong></div>
          <div class="prediction-item"><span>Tempo predict</span><strong>${formatPredictionValue(prediction?.elapsed_ms)} ms</strong></div>
          <div class="prediction-item"><span>Amostras avaliadas</span><strong>${formatPredictionValue(status.model_evaluated)}</strong></div>
          <div class="prediction-item"><span>Precisao</span><strong>${formatPercent(status.model_precision)}</strong></div>
          <div class="prediction-item"><span>Recall</span><strong>${formatPercent(status.model_recall)}</strong></div>
          <div class="prediction-item"><span>F1 score</span><strong>${formatPercent(status.model_f1_score)}</strong></div>
          <div class="prediction-item"><span>TP / TN</span><strong>${formatPercent(status.confusion_matrix_percent?.TP)} / ${formatPercent(status.confusion_matrix_percent?.TN)}</strong></div>
          <div class="prediction-item"><span>FP / FN</span><strong>${formatPercent(status.confusion_matrix_percent?.FP)} / ${formatPercent(status.confusion_matrix_percent?.FN)}</strong></div>
          <div class="prediction-item prediction-obs"><span>obs usada no model.predict</span><strong>${obsText}</strong></div>
          <div class="prediction-item prediction-obs"><span>Erro</span><strong>${formatPredictionValue(prediction?.error || status.model_error)}</strong></div>
        </div>
      `;
    }
    async function refresh() {
      const response = await fetch("/api/samples");
      const payload = await response.json();
      const rows = payload.rows;
      const status = payload.status;
      document.getElementById("stateLabel").textContent = status.running ? "executando" : "parado";
      document.getElementById("versionLabel").textContent = `${status.app_version} | ${status.served_file}`;
      document.getElementById("versionLabel").title = status.served_from;
      document.getElementById("rows").textContent = status.generated_rows.toLocaleString("pt-BR");
      document.getElementById("progress").textContent = `${Math.round(status.cycle_progress * 100)}%`;
      document.getElementById("phase").textContent = status.manual_fault_active
        ? "falha manual"
        : (status.event_started ? "falha" : "normal");
      document.getElementById("fault").textContent = status.manual_fault_active
        ? status.manual_fault_class
        : status.current_class;
      document.getElementById("modelAction").textContent = status.model_loaded
        ? `${status.model_action ?? "-"} (${status.model_status ?? "-"})`
        : "sem modelo";
      document.getElementById("expectedAction").textContent = status.expected_action ?? "-";
      document.getElementById("modelValidation").textContent = status.model_validation ?? "-";
      document.getElementById("modelAccuracy").textContent = status.model_accuracy === null
        ? "-"
        : `${Math.round(status.model_accuracy * 100)}%`;
      document.getElementById("modelPrecision").textContent = formatPercent(status.model_precision);
      document.getElementById("modelRecall").textContent = formatPercent(status.model_recall);
      document.getElementById("modelF1Score").textContent = formatPercent(status.model_f1_score);
      document.getElementById("modelTP").textContent = formatPercent(status.confusion_matrix_percent?.TP);
      document.getElementById("modelTN").textContent = formatPercent(status.confusion_matrix_percent?.TN);
      document.getElementById("modelFP").textContent = formatPercent(status.confusion_matrix_percent?.FP);
      document.getElementById("modelFN").textContent = formatPercent(status.confusion_matrix_percent?.FN);
      document.getElementById("modelLoad").textContent = status.model_loaded ? "sim" : "nao";
      document.getElementById("modelLoad").title = status.model_error || "";
      document.getElementById("scalerMode").textContent = status.scaler_mode;
      document.getElementById("scalerMode").title = status.scaler_error || "";
      document.getElementById("cycle").textContent = status.cycle;
      syncControls(status);
      document.getElementById("manualFaultButton").textContent = status.manual_fault_active
        ? "Parar falha"
        : "Testar falha";
      document.getElementById("manualFaultCode").disabled = status.manual_fault_active;
      updateOperationCards(status);
      updatePredictionCards(status);
      renderPrediction(status);
      renderLatest(status.latest);

      variables.forEach((variable, idx) => drawVariable(rows, variable, colors[idx % colors.length]));
      drawClasses(status.class_counts);
    }
    setupSensorCharts();
    refresh();
    setInterval(refresh, 1000);
    window.addEventListener("resize", refresh);
  </script>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET))
    parser.add_argument("--event-code", type=int, default=1)
    parser.add_argument("--source", choices=["real", "simulated", "drawn", "any"], default="real")
    parser.add_argument("--normal-source", choices=["real", "simulated", "drawn", "any"], default="real")
    parser.add_argument("--normal-rows", type=int, default=600)
    parser.add_argument("--event-rows", type=int, default=600)
    parser.add_argument("--rows-per-tick", type=int, default=5)
    parser.add_argument("--tick-seconds", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-history-rows", type=int, default=50000)
    parser.add_argument("--model-type", choices=["DQN", "PPO", "A2C", "RNA"], default="DQN")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--scaler-path", default="")
    parser.add_argument("--no-online-scaler", action="store_true")
    return parser.parse_args()


def is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) != 0


def select_available_port(host: str, requested_port: int) -> int:
    if is_port_available(host, requested_port):
        return requested_port

    for port in range(requested_port + 1, requested_port + 101):
        if is_port_available(host, port):
            print(
                f"Porta {requested_port} ja esta em uso. "
                f"Usando porta livre {port} para evitar dashboard antigo."
            )
            return port

    raise OSError(
        f"Nenhuma porta livre encontrada entre {requested_port} e {requested_port + 100}"
    )


def main() -> None:
    args = parse_args()
    selected_port = select_available_port(args.host, args.port)
    config = RuntimeConfig(
        dataset_path=args.dataset_path,
        event_code=args.event_code,
        source=args.source,
        normal_source=args.normal_source,
        normal_rows=args.normal_rows,
        event_rows=args.event_rows,
        rows_per_tick=args.rows_per_tick,
        tick_seconds=args.tick_seconds,
        noise_std=args.noise_std,
        seed=args.seed,
        max_history_rows=args.max_history_rows,
        model_type=args.model_type,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        use_online_scaler=not args.no_online_scaler,
    )
    state = RealtimeTwinState(config)
    app = create_app(state)
    print(f"Real-time digital twin: http://{args.host}:{selected_port}")
    print(f"Versao da dashboard: {APP_VERSION}")
    print(f"Arquivo servido: {Path(__file__).resolve()}")
    print(f"PID: {os.getpid()}")
    app.run(host=args.host, port=selected_port, threaded=True)


if __name__ == "__main__":
    main()
