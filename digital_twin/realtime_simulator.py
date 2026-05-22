"""Real-time 3W digital twin simulator with a browser dashboard."""

from __future__ import annotations

import argparse
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
        self.model = self._load_model()
        self.timeline = self._build_timeline()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _load_model(self) -> Any | None:
        path = Path(self.config.model_path)
        if not self.config.model_path or not path.exists():
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
        except Exception as exc:
            self.model_error = str(exc)
            return None

        self.model_error = f"Unsupported model_type: {self.config.model_type}"
        return None

    def _predict_row(self, row: dict[str, Any]) -> dict[str, Any]:
        if self.model is None:
            row["model_action"] = None
            row["model_status"] = "sem modelo"
            row["model_correct"] = None
            return row

        obs = np.array([row[variable] for variable in DEFAULT_VARIABLES], dtype=np.float32)
        action = self.model.predict(obs, deterministic=True)[0]
        action_value = int(np.asarray(action).item())
        expected_action = 0 if int(row["class"]) == 0 else 1
        row["model_action"] = action_value
        row["model_status"] = "normal" if action_value == 0 else "falha"
        row["model_correct"] = action_value == expected_action
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
        return frame[OUTPUT_COLUMNS]

    def _start_next_cycle(self) -> None:
        if self.generated_rows:
            latest = pd.Timestamp(self.generated_rows[-1]["timestamp"])
            self.next_start = latest + pd.Timedelta(seconds=1)
        self.cycle += 1
        self.pointer = 0
        self.timeline = self._build_timeline()

    def _run(self) -> None:
        while True:
            time.sleep(self.config.tick_seconds)
            with self.lock:
                if not self.running:
                    continue
                if self.pointer >= len(self.timeline):
                    self._start_next_cycle()
                end = min(self.pointer + self.config.rows_per_tick, len(self.timeline))
                chunk = self.timeline.iloc[self.pointer:end]
                rows = [self._predict_row(row) for row in chunk.to_dict(orient="records")]
                self.generated_rows.extend(rows)
                if len(self.generated_rows) > self.config.max_history_rows:
                    self.generated_rows = self.generated_rows[-self.config.max_history_rows :]
                self.total_emitted += len(chunk)
                self.pointer = end

    def start(self) -> None:
        with self.lock:
            self.running = True

    def pause(self) -> None:
        with self.lock:
            self.running = False

    def reset(self, updates: dict[str, Any] | None = None) -> None:
        with self.lock:
            if updates:
                for key, value in updates.items():
                    if hasattr(self.config, key) and value not in (None, ""):
                        current = getattr(self.config, key)
                        setattr(self.config, key, type(current)(value))
            self.running = False
            self.pointer = 0
            self.cycle = 0
            self.total_emitted = 0
            self.next_start = pd.Timestamp("2026-01-01 00:00:00")
            self.generated_rows = []
            self.model_error = None
            self.model = self._load_model()
            self.timeline = self._build_timeline()

    def status(self) -> dict[str, Any]:
        with self.lock:
            latest = self.generated_rows[-1] if self.generated_rows else None
            class_counts: dict[str, int] = {}
            for row in self.generated_rows:
                key = str(row["class"])
                class_counts[key] = class_counts.get(key, 0) + 1
            current_class = int(latest["class"]) if latest else 0
            evaluated = [row for row in self.generated_rows if row.get("model_correct") is not None]
            correct = sum(1 for row in evaluated if row["model_correct"])
            latest_action = latest.get("model_action") if latest else None
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
                "model_loaded": self.model is not None,
                "model_error": self.model_error,
                "model_action": latest_action,
                "model_status": latest.get("model_status") if latest else None,
                "model_accuracy": correct / len(evaluated) if evaluated else None,
                "model_correct": correct,
                "model_evaluated": len(evaluated),
                "latest": latest,
                "class_counts": class_counts,
                "variables": DEFAULT_VARIABLES,
                "columns": OUTPUT_COLUMNS,
                "config": asdict(self.config),
            }

    def window(self, limit: int) -> list[dict[str, Any]]:
        with self.lock:
            return self.generated_rows[-limit:]

    def samples(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.generated_rows if limit is None else self.generated_rows[-limit:]
            return list(rows)


def create_app(state: RealtimeTwinState) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return DASHBOARD_HTML

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
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
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
    main { padding: 18px; max-width: 1500px; margin: 0 auto; }
    .toolbar, .metrics, .grid, .charts-grid {
      display: grid;
      gap: 12px;
    }
    .toolbar {
      grid-template-columns: repeat(auto-fit, minmax(150px, max-content));
      align-items: end;
      margin-bottom: 12px;
    }
    button {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      border-radius: 6px;
      padding: 10px 14px;
      cursor: pointer;
      font-size: 14px;
    }
    button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    button.danger { color: var(--danger); }
    label { display: grid; gap: 4px; color: var(--muted); font-size: 12px; }
    input, select {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      min-width: 120px;
      background: #fff;
      color: var(--text);
    }
    .modelPath { min-width: min(520px, 90vw); }
    .metrics {
      grid-template-columns: repeat(auto-fit, minmax(180px, 240px));
      justify-content: center;
      margin-bottom: 12px;
    }
    .metric, section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      text-align: center;
    }
    .metric strong { display: block; font-size: 26px; margin-bottom: 3px; }
    .metric span { color: var(--muted); }
    .grid { grid-template-columns: minmax(0, 1fr) minmax(320px, 360px); }
    .charts-grid { grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); }
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
    @media (max-width: 980px) {
      .grid, .charts-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>3W Real-Time Digital Twin</h1>
    <div id="stateLabel">parado</div>
  </header>
  <main>
    <div class="toolbar">
      <button class="primary" onclick="control('start')">Iniciar</button>
      <button onclick="control('pause')">Pausar</button>
      <button class="danger" onclick="resetSimulation()">Resetar</button>
      <label>Evento
        <select id="eventCode">
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
      <label>Fonte
        <select id="source">
          <option value="real">real</option>
          <option value="simulated">simulated</option>
          <option value="drawn">drawn</option>
          <option value="any">any</option>
        </select>
      </label>
      <label>Ruído
        <input id="noiseStd" type="number" min="0" max="1" step="0.01" value="0">
      </label>
      <label>Modelo
        <select id="modelType">
          <option value="DQN">DQN</option>
          <option value="PPO">PPO</option>
          <option value="A2C">A2C</option>
        </select>
      </label>
      <label>Caminho do modelo
        <input id="modelPath" class="modelPath" type="text" value="">
      </label>
    </div>
    <div class="metrics">
      <div class="metric"><strong id="rows">0</strong><span>amostras emitidas</span></div>
      <div class="metric"><strong id="progress">0%</strong><span>progresso do ciclo</span></div>
      <div class="metric"><strong id="phase">normal</strong><span>fase operacional</span></div>
      <div class="metric"><strong id="fault">0</strong><span>classe atual</span></div>
      <div class="metric"><strong id="modelAction">-</strong><span>ação do modelo</span></div>
      <div class="metric"><strong id="modelAccuracy">-</strong><span>acurácia online</span></div>
      <div class="metric"><strong id="cycle">1</strong><span>ciclo</span></div>
    </div>
    <div class="grid">
      <div class="charts-grid" id="sensorCharts"></div>
      <div>
        <section>
          <h2>Distribuição de Classes</h2>
          <canvas id="classesCanvas"></canvas>
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
      ctx.strokeStyle = "#d7dee6";
      ctx.lineWidth = 1;
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
            model_path: document.getElementById("modelPath").value
          }
        })
      });
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
    async function refresh() {
      const response = await fetch("/api/window?limit=300");
      const payload = await response.json();
      const rows = payload.rows;
      const status = payload.status;
      document.getElementById("stateLabel").textContent = status.running ? "executando" : "parado";
      document.getElementById("rows").textContent = status.generated_rows.toLocaleString("pt-BR");
      document.getElementById("progress").textContent = `${Math.round(status.cycle_progress * 100)}%`;
      document.getElementById("phase").textContent = status.event_started ? "falha" : "normal";
      document.getElementById("fault").textContent = status.current_class;
      document.getElementById("modelAction").textContent = status.model_loaded
        ? `${status.model_action ?? "-"} (${status.model_status ?? "-"})`
        : "sem modelo";
      document.getElementById("modelAccuracy").textContent = status.model_accuracy === null
        ? "-"
        : `${Math.round(status.model_accuracy * 100)}%`;
      document.getElementById("cycle").textContent = status.cycle;
      document.getElementById("modelType").value = status.config.model_type;
      document.getElementById("modelPath").value = status.config.model_path;
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
    parser.add_argument("--model-type", choices=["DQN", "PPO", "A2C"], default="DQN")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    )
    state = RealtimeTwinState(config)
    app = create_app(state)
    print(f"Real-time digital twin: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
