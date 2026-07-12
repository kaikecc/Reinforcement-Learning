"""Build a local digital twin layer for the 3W dataset."""

from __future__ import annotations

import argparse
import configparser
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(os.environ.get("THREE_W_ROOT", ROOT.parent / "3W"))
DATASET_DIR = DATASET_ROOT / "dataset"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CONFIG_PATH = DATASET_DIR / "dataset.ini"
SENSOR_COLUMNS = [
    "ABER-CKGL",
    "ABER-CKP",
    "ESTADO-DHSV",
    "ESTADO-M1",
    "ESTADO-M2",
    "ESTADO-PXO",
    "ESTADO-SDV-GL",
    "ESTADO-SDV-P",
    "ESTADO-W1",
    "ESTADO-W2",
    "ESTADO-XO",
    "P-ANULAR",
    "P-JUS-BS",
    "P-JUS-CKGL",
    "P-JUS-CKP",
    "P-MON-CKGL",
    "P-MON-CKP",
    "P-MON-SDV-P",
    "P-PDG",
    "PT-P",
    "P-TPT",
    "QBS",
    "QGL",
    "T-JUS-CKP",
    "T-MON-CKP",
    "T-PDG",
    "T-TPT",
]
PREFERRED_SERIES = ["P-PDG", "P-TPT", "P-MON-CKP", "QGL", "T-TPT"]


@dataclass
class RunningStats:
    count: int = 0
    missing: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    min_value: float | None = None
    max_value: float | None = None

    def update(self, values: pd.Series, row_count: int) -> None:
        numeric = pd.to_numeric(values, errors="coerce")
        valid = numeric.dropna()
        self.missing += int(row_count - valid.size)
        if valid.empty:
            return

        self.count += int(valid.size)
        self.total += float(valid.sum())
        self.total_sq += float((valid * valid).sum())
        current_min = float(valid.min())
        current_max = float(valid.max())
        self.min_value = current_min if self.min_value is None else min(self.min_value, current_min)
        self.max_value = current_max if self.max_value is None else max(self.max_value, current_max)

    def as_dict(self) -> dict[str, float | int | None]:
        if self.count == 0:
            mean = None
            std = None
        else:
            mean = self.total / self.count
            variance = max((self.total_sq / self.count) - (mean * mean), 0.0)
            std = math.sqrt(variance)
        total_rows = self.count + self.missing
        coverage = self.count / total_rows if total_rows else 0.0
        return {
            "valid_count": self.count,
            "missing_count": self.missing,
            "coverage": coverage,
            "mean": mean,
            "std": std,
            "min": self.min_value,
            "max": self.max_value,
        }


@dataclass
class TwinState:
    catalog: list[dict[str, Any]] = field(default_factory=list)
    class_source_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    class_label_counts: Counter[str] = field(default_factory=Counter)
    class_state_counts: Counter[str] = field(default_factory=Counter)
    stats: dict[tuple[str, str], RunningStats] = field(default_factory=lambda: defaultdict(RunningStats))
    samples: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_events() -> dict[str, dict[str, str]]:
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    events: dict[str, dict[str, str]] = {}
    for section in config.sections():
        if "LABEL" not in config[section]:
            continue
        label = config[section]["LABEL"]
        events[label] = {
            "name": section,
            "description": config[section].get("DESCRIPTION", section),
        }
    return events


def source_from_filename(path: Path) -> str:
    name = path.name.upper()
    if name.startswith("SIMULATED"):
        return "SIMULATED"
    if name.startswith("DRAWN"):
        return "DRAWN"
    if name.startswith("WELL"):
        return "REAL"
    return "UNKNOWN"


def well_from_filename(path: Path) -> str | None:
    if not path.name.upper().startswith("WELL"):
        return None
    return path.name.split("_", maxsplit=1)[0]


def downsample_frame(df: pd.DataFrame, columns: list[str], max_points: int) -> pd.DataFrame:
    cols = [column for column in columns if column in df.columns]
    sample = df[cols].copy()
    if len(sample) > max_points:
        step = math.ceil(len(sample) / max_points)
        sample = sample.iloc[::step]
    sample = sample.reset_index()
    sample["timestamp"] = sample["timestamp"].astype(str)
    return sample


def scan_file(path: Path, events: dict[str, dict[str, str]], state: TwinState, max_sample_points: int) -> None:
    df = pd.read_parquet(path)
    class_label = path.parent.name
    event = events.get(class_label, {"name": f"CLASS_{class_label}", "description": f"Class {class_label}"})
    source = source_from_filename(path)
    row_count = int(len(df))
    start = df.index.min()
    end = df.index.max()
    duration_seconds = float((end - start).total_seconds()) if row_count else 0.0
    present_sensors = [column for column in SENSOR_COLUMNS if column in df.columns]
    non_null = df[present_sensors].notna().sum() if present_sensors else pd.Series(dtype="int64")
    coverage = float(non_null.sum() / (row_count * len(present_sensors))) if row_count and present_sensors else 0.0

    label_counts = {}
    if "class" in df.columns:
        label_counts = {str(key): int(value) for key, value in df["class"].value_counts(dropna=False).items()}
        state.class_label_counts.update(label_counts)
    state_counts = {}
    if "state" in df.columns:
        state_counts = {str(key): int(value) for key, value in df["state"].value_counts(dropna=False).items()}
        state.class_state_counts.update(state_counts)

    for column in present_sensors:
        state.stats[(class_label, column)].update(df[column], row_count)

    state.class_source_counts[(class_label, source)] += 1
    state.catalog.append(
        {
            "file": str(path.relative_to(DATASET_ROOT)).replace("\\", "/"),
            "class_label": class_label,
            "event_name": event["name"],
            "event_description": event["description"],
            "source": source,
            "well": well_from_filename(path),
            "rows": row_count,
            "start": str(start) if pd.notna(start) else None,
            "end": str(end) if pd.notna(end) else None,
            "duration_hours": duration_seconds / 3600,
            "sensor_count": len(present_sensors),
            "sensor_coverage": coverage,
            "label_counts": label_counts,
            "state_counts": state_counts,
        }
    )

    if class_label not in state.samples:
        sample_columns = PREFERRED_SERIES + ["class", "state"]
        state.samples[class_label] = {
            "file": str(path.relative_to(DATASET_ROOT)).replace("\\", "/"),
            "event_description": event["description"],
            "source": source,
            "data": downsample_frame(df, sample_columns, max_sample_points).to_dict(orient="records"),
        }


def iter_parquet_files(max_files_per_class: int | None) -> list[Path]:
    files: list[Path] = []
    class_dirs = sorted(
        [path for path in DATASET_DIR.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )
    for class_dir in class_dirs:
        class_files = sorted(class_dir.glob("*.parquet"))
        if max_files_per_class is not None:
            class_files = class_files[:max_files_per_class]
        files.extend(class_files)
    return files


def build_summary(catalog: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(catalog)
    summary = (
        frame.groupby(["class_label", "event_description"], as_index=False)
        .agg(
            files=("file", "count"),
            rows=("rows", "sum"),
            duration_hours=("duration_hours", "sum"),
            avg_sensor_coverage=("sensor_coverage", "mean"),
            real_instances=("source", lambda values: int((values == "REAL").sum())),
            simulated_instances=("source", lambda values: int((values == "SIMULATED").sum())),
            drawn_instances=("source", lambda values: int((values == "DRAWN").sum())),
        )
        .sort_values("class_label", key=lambda col: col.astype(int))
    )
    return summary


def build_variable_profiles(state: TwinState, events: dict[str, dict[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (class_label, variable), stats in sorted(state.stats.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        row = {
            "class_label": class_label,
            "event_description": events.get(class_label, {}).get("description", f"Class {class_label}"),
            "variable": variable,
        }
        row.update(stats.as_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def make_dashboard(
    summary: pd.DataFrame,
    catalog: pd.DataFrame,
    profiles: pd.DataFrame,
    samples: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    class_order = sorted(summary["class_label"].tolist(), key=int)
    summary = summary.copy()
    summary["class_display"] = summary["class_label"] + " - " + summary["event_description"]

    files_fig = px.bar(
        summary,
        x="class_display",
        y="files",
        color="class_display",
        title="Instâncias por classe de evento",
        labels={"class_display": "Classe", "files": "Arquivos"},
    )
    files_fig.update_layout(showlegend=False, xaxis_tickangle=-35)

    source_frame = (
        catalog.groupby(["class_label", "event_description", "source"], as_index=False)
        .size()
        .rename(columns={"size": "files"})
    )
    source_frame["class_display"] = source_frame["class_label"] + " - " + source_frame["event_description"]
    source_fig = px.bar(
        source_frame,
        x="class_display",
        y="files",
        color="source",
        title="Origem das instâncias",
        labels={"class_display": "Classe", "files": "Arquivos", "source": "Fonte"},
    )
    source_fig.update_layout(xaxis_tickangle=-35)

    timeline_fig = px.scatter(
        catalog[catalog["source"] == "REAL"],
        x="start",
        y="well",
        color="event_description",
        size="duration_hours",
        hover_data=["file", "rows", "sensor_coverage"],
        title="Linha do tempo das instâncias reais",
        labels={"start": "Início", "well": "Poço", "event_description": "Evento"},
    )

    coverage_pivot = profiles.pivot(index="variable", columns="event_description", values="coverage").fillna(0)
    coverage_fig = px.imshow(
        coverage_pivot,
        aspect="auto",
        color_continuous_scale="Viridis",
        zmin=0,
        zmax=1,
        title="Cobertura média das variáveis por classe",
        labels={"color": "Cobertura"},
    )

    rows_fig = px.bar(
        summary,
        x="class_display",
        y="rows",
        color="avg_sensor_coverage",
        title="Volume de observações e cobertura média",
        labels={"class_display": "Classe", "rows": "Observações", "avg_sensor_coverage": "Cobertura"},
    )
    rows_fig.update_layout(xaxis_tickangle=-35)

    sample_fig = go.Figure()
    first_trace = True
    buttons = []
    trace_visibility: list[list[bool]] = []
    trace_index = 0
    for class_label in class_order:
        sample = samples.get(class_label)
        if not sample:
            continue
        sample_df = pd.DataFrame(sample["data"])
        visible_for_button = [False] * trace_index
        added = 0
        for variable in PREFERRED_SERIES:
            if variable not in sample_df.columns:
                continue
            sample_fig.add_trace(
                go.Scatter(
                    x=sample_df["timestamp"],
                    y=sample_df[variable],
                    mode="lines",
                    name=variable,
                    visible=first_trace,
                    hovertemplate=f"{variable}: %{{y}}<extra></extra>",
                )
            )
            visible_for_button.append(True)
            trace_index += 1
            added += 1
        for previous in trace_visibility:
            previous.extend([False] * added)
        trace_visibility.append(visible_for_button)
        buttons.append(
            {
                "label": f"{class_label} - {sample['event_description']}",
                "method": "update",
                "args": [
                    {"visible": visible_for_button},
                    {"title": f"Série representativa: {class_label} - {sample['event_description']} ({sample['source']})"},
                ],
            }
        )
        first_trace = False

    if buttons:
        sample_fig.update_layout(
            title=buttons[0]["label"],
            updatemenus=[{"buttons": buttons, "direction": "down", "x": 0, "y": 1.18}],
            xaxis_title="Tempo",
            yaxis_title="Valor",
        )

    charts = [
        to_html(files_fig, include_plotlyjs="cdn", full_html=False),
        to_html(source_fig, include_plotlyjs=False, full_html=False),
        to_html(rows_fig, include_plotlyjs=False, full_html=False),
        to_html(timeline_fig, include_plotlyjs=False, full_html=False),
        to_html(coverage_fig, include_plotlyjs=False, full_html=False),
        to_html(sample_fig, include_plotlyjs=False, full_html=False),
    ]

    total_files = int(summary["files"].sum())
    total_rows = int(summary["rows"].sum())
    total_hours = float(summary["duration_hours"].sum())
    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>3W Digital Twin</title>
  <style>
    body {{
      margin: 0;
      color: #182026;
      background: #f3f5f7;
      font-family: Arial, Helvetica, sans-serif;
    }}
    header {{
      background: #ffffff;
      border-bottom: 1px solid #d8dee4;
      padding: 24px 36px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      letter-spacing: 0;
    }}
    main {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .metric {{
      background: #ffffff;
      border: 1px solid #d8dee4;
      border-radius: 8px;
      padding: 16px;
    }}
    .metric strong {{
      display: block;
      font-size: 28px;
      margin-bottom: 4px;
    }}
    .metric span {{
      display: block;
      color: #4d5a63;
    }}
    section {{
      background: #ffffff;
      border: 1px solid #d8dee4;
      border-radius: 8px;
      margin: 16px 0;
      padding: 12px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>3W Digital Twin</h1>
    <div>Catálogo operacional e perfil estatístico da base local 3W.</div>
  </header>
  <main>
    <div class="metrics">
      <div class="metric"><strong>{total_files:,}</strong><span>arquivos Parquet</span></div>
      <div class="metric"><strong>{total_rows:,}</strong><span>observações</span></div>
      <div class="metric"><strong>{total_hours:,.1f}</strong><span>horas de séries</span></div>
      <div class="metric"><strong>{len(class_order)}</strong><span>classes de evento</span></div>
    </div>
    {''.join(f'<section>{chart}</section>' for chart in charts)}
  </main>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events = load_events()
    files = iter_parquet_files(args.max_files_per_class)
    state = TwinState()

    for index, path in enumerate(files, start=1):
        if args.progress and (index == 1 or index % args.progress == 0 or index == len(files)):
            print(f"[{index}/{len(files)}] {path.relative_to(DATASET_ROOT)}")
        scan_file(path, events, state, args.max_sample_points)

    summary = build_summary(state.catalog)
    profiles = build_variable_profiles(state, events)
    catalog_frame = pd.DataFrame(state.catalog)

    (OUTPUT_DIR / "twin_catalog.json").write_text(
        json.dumps(
            {
                "dataset_dir": str(DATASET_DIR),
                "generated_files": len(state.catalog),
                "catalog": state.catalog,
                "representative_samples": state.samples,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summary.to_csv(OUTPUT_DIR / "twin_summary.csv", index=False)
    profiles.to_csv(OUTPUT_DIR / "variable_profiles.csv", index=False)
    make_dashboard(summary, catalog_frame, profiles, state.samples, OUTPUT_DIR / "dashboard.html")

    print(f"Digital twin generated in: {OUTPUT_DIR}")
    print(f"Dashboard: {OUTPUT_DIR / 'dashboard.html'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-files-per-class",
        type=int,
        default=None,
        help="Limit the scan to the first N files of each class for quick tests.",
    )
    parser.add_argument(
        "--max-sample-points",
        type=int,
        default=600,
        help="Maximum points stored for each representative time series.",
    )
    parser.add_argument(
        "--progress",
        type=int,
        default=100,
        help="Print progress every N files. Use 0 to disable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
