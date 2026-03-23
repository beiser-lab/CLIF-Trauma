from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd


PHASE_COLORS = {
    "ED": "#c46210",
    "SICU_24h": "#0f766e",
}

PHASE_LABELS = {
    "ED": "ED",
    "SICU_24h": "SICU",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build direct ED-to-SICU subgroup summaries and SVG figures."
    )
    parser.add_argument("--output-dir", required=True, help="Pipeline output directory containing cohort.csv and related files.")
    parser.add_argument(
        "--max-hour",
        type=int,
        default=24,
        help="Maximum elapsed hour to include in hourly intervention plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    artifacts = run_direct_sicu_analysis(output_dir, max_hour=args.max_hour)
    write_direct_sicu_outputs(output_dir, artifacts)


def run_direct_sicu_analysis(output_dir: Path, max_hour: int = 24) -> dict[str, pd.DataFrame]:
    cohort, phase_windows, interventions = load_output_tables(output_dir)
    direct_cohort = cohort.loc[~cohort["has_procedural_bridge"].fillna(False)].copy()
    direct_ids = direct_cohort["hospitalization_id"].drop_duplicates()
    direct_phase_windows = phase_windows.loc[phase_windows["hospitalization_id"].isin(direct_ids)].copy()
    direct_interventions = interventions.loc[interventions["hospitalization_id"].isin(direct_ids)].copy()

    summary = build_direct_sicu_summary(direct_cohort)
    phase_rates = build_phase_rate_comparison(direct_phase_windows)
    elapsed_hour_rates = build_elapsed_hour_rates(direct_phase_windows, direct_interventions, max_hour=max_hour)
    boarding_mortality = build_boarding_mortality_bins(direct_cohort)

    return {
        "summary": summary,
        "phase_rates": phase_rates,
        "elapsed_hour_rates": elapsed_hour_rates,
        "boarding_mortality": boarding_mortality,
    }


def load_output_tables(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort = pd.read_csv(
        output_dir / "cohort.csv",
        parse_dates=[
            "admission_dttm",
            "discharge_dttm",
            "ed_in_dttm",
            "ed_out_dttm",
            "sicu_in_dttm",
            "initial_sicu_exit_dttm",
            "next_after_sicu_in_dttm",
            "first_imv_dttm",
            "transfer_outcome_dttm",
        ],
    )
    phase_windows = pd.read_csv(
        output_dir / "phase_windows.csv",
        parse_dates=["phase_start_dttm", "phase_end_dttm"],
    )
    interventions = pd.read_csv(
        output_dir / "interventions.csv",
        parse_dates=["event_dttm"],
    )
    return cohort, phase_windows, interventions


def build_direct_sicu_summary(direct_cohort: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("direct_ed_to_sicu_hospitalizations", len(direct_cohort)),
        ("in_hospital_mortality_rate", direct_cohort["in_hospital_mortality"].mean()),
        ("median_ed_los_hours", direct_cohort["ed_los_hours"].median()),
        ("median_hospital_los_hours", direct_cohort["hospital_los_hours"].median()),
        ("median_sicu_preward_los_hours", direct_cohort["sicu_preward_los_hours"].dropna().median()),
    ]
    return pd.DataFrame([{"metric": metric, "value": value} for metric, value in metrics])


def build_phase_rate_comparison(phase_windows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for phase, phase_data in phase_windows.groupby("phase", sort=False):
        rates = pd.to_numeric(phase_data["interventions_per_vent_hour"], errors="coerce")
        weights = pd.to_numeric(phase_data["phase_duration_hours"], errors="coerce").fillna(0)
        total_interventions = pd.to_numeric(phase_data["intervention_count"], errors="coerce").fillna(0).sum()
        total_phase_hours = weights.sum()
        rows.append(
            {
                "phase": phase,
                "hospitalizations": int(phase_data["hospitalization_id"].nunique()),
                "total_phase_hours": float(total_phase_hours),
                "total_interventions": int(total_interventions),
                "mean_patient_rate": float(total_interventions / total_phase_hours) if total_phase_hours else pd.NA,
                "winsorized_mean_patient_rate": winsorized_weighted_mean(rates, weights),
                "median_patient_rate": float(rates.median()) if not rates.dropna().empty else pd.NA,
                "pct_zero": float((phase_data["intervention_count"].fillna(0) == 0).mean()),
            }
        )
    summary = pd.DataFrame.from_records(rows)
    summary["phase"] = pd.Categorical(summary["phase"], categories=["ED", "SICU_24h"], ordered=True)
    return summary.sort_values("phase").reset_index(drop=True)


def build_elapsed_hour_rates(
    phase_windows: pd.DataFrame,
    interventions: pd.DataFrame,
    max_hour: int = 24,
) -> pd.DataFrame:
    patient_hour_rows: list[dict[str, object]] = []
    for row in phase_windows.itertuples(index=False):
        duration = float(row.phase_duration_hours)
        for hour in range(max_hour):
            exposure = max(0.0, min(duration, hour + 1) - hour)
            if exposure <= 0:
                continue
            patient_hour_rows.append(
                {
                    "hospitalization_id": row.hospitalization_id,
                    "phase": row.phase,
                    "elapsed_hour": hour,
                    "exposure_hours": exposure,
                    "phase_start_dttm": row.phase_start_dttm,
                }
            )
    patient_hours = pd.DataFrame.from_records(patient_hour_rows)
    if patient_hours.empty:
        return pd.DataFrame(
            columns=[
                "phase",
                "elapsed_hour",
                "n_active",
                "exposure_hours",
                "event_count",
                "mean_rate",
                "winsorized_mean_rate",
                "median_rate",
                "pct_zero",
            ]
        )

    event_rates = interventions.merge(
        patient_hours[["hospitalization_id", "phase", "phase_start_dttm"]].drop_duplicates(),
        on=["hospitalization_id", "phase"],
        how="left",
    )
    event_rates["elapsed_hour"] = (
        (event_rates["event_dttm"] - event_rates["phase_start_dttm"]).dt.total_seconds() / 3600.0
    ).floordiv(1).astype("Int64")
    event_rates = event_rates.loc[
        event_rates["elapsed_hour"].notna()
        & event_rates["elapsed_hour"].ge(0)
        & event_rates["elapsed_hour"].lt(max_hour)
    ].copy()
    event_counts = (
        event_rates.groupby(["hospitalization_id", "phase", "elapsed_hour"], as_index=False)
        .size()
        .rename(columns={"size": "event_count"})
    )
    patient_hours = patient_hours.merge(
        event_counts,
        on=["hospitalization_id", "phase", "elapsed_hour"],
        how="left",
    )
    patient_hours["event_count"] = patient_hours["event_count"].fillna(0).astype(int)
    patient_hours["patient_hour_rate"] = patient_hours["event_count"] / patient_hours["exposure_hours"].replace(0, pd.NA)

    rows: list[dict[str, object]] = []
    for (phase, elapsed_hour), hour_data in patient_hours.groupby(["phase", "elapsed_hour"], sort=False):
        rates = pd.to_numeric(hour_data["patient_hour_rate"], errors="coerce")
        weights = pd.to_numeric(hour_data["exposure_hours"], errors="coerce").fillna(0)
        total_events = hour_data["event_count"].sum()
        total_exposure = weights.sum()
        rows.append(
            {
                "phase": phase,
                "elapsed_hour": int(elapsed_hour),
                "n_active": int(len(hour_data)),
                "exposure_hours": float(total_exposure),
                "event_count": int(total_events),
                "mean_rate": float(total_events / total_exposure) if total_exposure else pd.NA,
                "winsorized_mean_rate": winsorized_weighted_mean(rates, weights),
                "median_rate": float(rates.median()) if not rates.dropna().empty else pd.NA,
                "pct_zero": float((hour_data["event_count"] == 0).mean()),
            }
        )
    summary = pd.DataFrame.from_records(rows)
    summary["phase"] = pd.Categorical(summary["phase"], categories=["ED", "SICU_24h"], ordered=True)
    return summary.sort_values(["phase", "elapsed_hour"]).reset_index(drop=True)


def build_boarding_mortality_bins(direct_cohort: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 2, 4, 6, 12, 24, float("inf")]
    labels = ["0-<2", "2-<4", "4-<6", "6-<12", "12-<24", "24+"]
    data = direct_cohort.copy()
    data["boarding_bin"] = pd.cut(
        data["ed_los_hours"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    summary = (
        data.groupby("boarding_bin", observed=False, as_index=False)
        .agg(
            hospitalizations=("hospitalization_id", "size"),
            deaths=("in_hospital_mortality", "sum"),
            median_boarding_hours=("ed_los_hours", "median"),
        )
    )
    summary["mortality_rate"] = summary["deaths"] / summary["hospitalizations"].replace(0, pd.NA)
    return summary


def write_direct_sicu_outputs(output_dir: Path, artifacts: dict[str, pd.DataFrame]) -> None:
    summary_dir = output_dir / "summary"
    figures_dir = output_dir / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    artifacts["summary"].to_csv(summary_dir / "direct_ed_to_sicu_summary.csv", index=False)
    artifacts["phase_rates"].to_csv(summary_dir / "direct_ed_to_sicu_phase_rates.csv", index=False)
    artifacts["elapsed_hour_rates"].to_csv(summary_dir / "direct_ed_to_sicu_elapsed_hour_rates.csv", index=False)
    artifacts["boarding_mortality"].to_csv(summary_dir / "direct_ed_to_sicu_boarding_mortality.csv", index=False)

    write_text(
        figures_dir / "direct_ed_to_sicu_intervention_rates.svg",
        render_intervention_rates_svg(artifacts["phase_rates"], artifacts["elapsed_hour_rates"]),
    )
    write_text(
        figures_dir / "direct_ed_to_sicu_boarding_mortality.svg",
        render_boarding_mortality_svg(artifacts["boarding_mortality"]),
    )


def render_intervention_rates_svg(phase_rates: pd.DataFrame, elapsed_hour_rates: pd.DataFrame) -> str:
    width = 1200
    height = 780
    lines = [
        svg_prelude(width, height),
        rect(0, 0, width, height, "#f8fafc"),
        title("Direct ED to SICU: intervention rates with robustness companion", 36),
        subtitle("Solid lines show the upper-95% winsorized mean rate; lighter dashed lines show the raw mean rate.", 62),
        subtitle("Bottom panel shows the active-patient denominator by elapsed hour.", 84),
    ]

    lines.extend(draw_overall_rate_panel(phase_rates, 70, 130, 320, 250))
    lines.extend(draw_hourly_rate_panel(elapsed_hour_rates, 440, 130, 700, 250))
    lines.extend(draw_denominator_panel(elapsed_hour_rates, 440, 440, 700, 220))
    lines.append("</svg>")
    return "\n".join(lines)


def render_boarding_mortality_svg(boarding_mortality: pd.DataFrame) -> str:
    width = 980
    height = 520
    lines = [svg_prelude(width, height), rect(0, 0, width, height, "#f8fafc"), title("Direct ED to SICU: boarding time and mortality", 36)]
    lines.append(subtitle("Mortality rate by ED length-of-stay bin for patients without an ED to OR bridge", 62))
    lines.extend(draw_simple_bar_panel(boarding_mortality, 80, 120, 820, 320, "mortality_rate", "Mortality rate", x_col="boarding_bin"))
    for idx, row in boarding_mortality.reset_index(drop=True).iterrows():
        lines.append(
            svg_text(
                140 + idx * 130,
                470,
                f"n={int(row['hospitalizations'])}",
                size=12,
                fill="#475569",
                anchor="middle",
            )
        )
    lines.append("</svg>")
    return "\n".join(lines)


def draw_overall_rate_panel(
    phase_rates: pd.DataFrame,
    x: int,
    y: int,
    width: int,
    height: int,
) -> list[str]:
    if phase_rates.empty:
        return [svg_text(x + width / 2, y + height / 2, "No data", size=18, fill="#64748b", anchor="middle")]

    max_value = float(
        phase_rates[["mean_patient_rate", "winsorized_mean_patient_rate"]]
        .astype(float)
        .fillna(0)
        .to_numpy()
        .max()
    )
    max_value = max(max_value, 0.01)
    group_width = width / max(len(phase_rates), 1)
    bar_width = 44

    lines = [rect(x, y, width, height, "white", stroke="#cbd5e1"), svg_text(x, y - 12, "Overall phase rates", size=15, fill="#0f172a")]
    for tick in range(5):
        value = max_value * tick / 4
        y_pos = y + height - (height * tick / 4)
        lines.append(line(x, y_pos, x + width, y_pos, "#e2e8f0"))
        lines.append(svg_text(x - 10, y_pos + 4, f"{value:.2f}", size=12, fill="#475569", anchor="end"))

    for idx, row in phase_rates.reset_index(drop=True).iterrows():
        center = x + group_width * idx + group_width / 2
        label = PHASE_LABELS.get(str(row["phase"]), str(row["phase"]))
        for metric_idx, metric in enumerate(["mean_patient_rate", "winsorized_mean_patient_rate"]):
            value = float(row[metric]) if pd.notna(row[metric]) else 0.0
            bar_height = (value / max_value) * (height - 10)
            left = center - 54 + metric_idx * 56
            top = y + height - bar_height
            fill = PHASE_COLORS.get(str(row["phase"]), "#2563eb")
            opacity = "0.35" if metric == "mean_patient_rate" else "1.0"
            lines.append(rect(left, top, bar_width, bar_height, fill, opacity=opacity))
            lines.append(svg_text(left + bar_width / 2, top - 8, f"{value:.3f}", size=11, fill="#0f172a", anchor="middle"))
        lines.append(svg_text(center, y + height + 22, label, size=12, fill="#0f172a", anchor="middle"))

    legend_y = y + 18
    lines.append(rect(x + width - 150, legend_y - 12, 14, 14, "#334155", opacity="0.35"))
    lines.append(svg_text(x + width - 128, legend_y, "Mean", size=12, fill="#0f172a"))
    lines.append(rect(x + width - 80, legend_y - 12, 14, 14, "#334155"))
    lines.append(svg_text(x + width - 58, legend_y, "Winsorized", size=12, fill="#0f172a"))
    return lines


def draw_hourly_rate_panel(
    data: pd.DataFrame,
    x: int,
    y: int,
    width: int,
    height: int,
) -> list[str]:
    if data.empty:
        return [svg_text(x + width / 2, y + height / 2, "No data", size=18, fill="#64748b", anchor="middle")]

    max_x = max(int(data["elapsed_hour"].max()), 1)
    max_y = float(
        data[["mean_rate", "winsorized_mean_rate"]]
        .astype(float)
        .fillna(0)
        .to_numpy()
        .max()
    )
    max_y = max(max_y, 0.01)

    lines = [rect(x, y, width, height, "white", stroke="#cbd5e1"), svg_text(x, y - 12, "Hourly intervention rates", size=15, fill="#0f172a")]
    for tick in range(5):
        value = max_y * tick / 4
        y_pos = y + height - (height * tick / 4)
        lines.append(line(x, y_pos, x + width, y_pos, "#e2e8f0"))
        lines.append(svg_text(x - 10, y_pos + 4, f"{value:.2f}", size=12, fill="#475569", anchor="end"))

    for tick in range(max_x + 1):
        x_pos = x + (width * tick / max_x if max_x else 0)
        lines.append(line(x_pos, y, x_pos, y + height, "#f1f5f9"))
        lines.append(svg_text(x_pos, y + height + 18, str(tick), size=11, fill="#475569", anchor="middle"))
    lines.append(svg_text(x + width / 2, y + height + 42, "Elapsed hour from phase start", size=13, fill="#0f172a", anchor="middle"))

    for phase, phase_data in data.groupby("phase", sort=False):
        color = PHASE_COLORS.get(str(phase), "#2563eb")
        mean_points = []
        winsor_points = []
        for row in phase_data.itertuples(index=False):
            px = x + (width * float(row.elapsed_hour) / max_x if max_x else 0)
            mean_y = y + height - (height * float(row.mean_rate) / max_y)
            winsor_y = y + height - (height * float(row.winsorized_mean_rate) / max_y)
            mean_points.append((px, mean_y))
            winsor_points.append((px, winsor_y))
        lines.append(polyline(mean_points, color, opacity="0.35", dasharray="10 6"))
        lines.append(polyline(winsor_points, color, stroke_width=3))
        for px, py in winsor_points:
            lines.append(circle(px, py, 2.8, color))

    legend_y = y + 18
    legend_x = x + 18
    for idx, phase in enumerate(["ED", "SICU_24h"]):
        offset = idx * 180
        color = PHASE_COLORS[phase]
        lines.append(line(legend_x + offset, legend_y, legend_x + 28 + offset, legend_y, color, width=3))
        lines.append(svg_text(legend_x + 36 + offset, legend_y + 4, f"{PHASE_LABELS[phase]} winsorized", size=12, fill="#0f172a"))
        lines.append(line(legend_x + 96 + offset, legend_y, legend_x + 124 + offset, legend_y, color, width=3, opacity="0.35", dasharray="10 6"))
        lines.append(svg_text(legend_x + 132 + offset, legend_y + 4, f"{PHASE_LABELS[phase]} mean", size=12, fill="#0f172a"))
    return lines


def draw_denominator_panel(
    data: pd.DataFrame,
    x: int,
    y: int,
    width: int,
    height: int,
) -> list[str]:
    if data.empty:
        return [svg_text(x + width / 2, y + height / 2, "No data", size=18, fill="#64748b", anchor="middle")]

    max_x = max(int(data["elapsed_hour"].max()), 1)
    max_y = float(data["n_active"].max())
    max_y = max(max_y, 1.0)
    lines = [rect(x, y, width, height, "white", stroke="#cbd5e1"), svg_text(x, y - 12, "Denominator: active patients by hour", size=15, fill="#0f172a")]
    for tick in range(5):
        value = max_y * tick / 4
        y_pos = y + height - (height * tick / 4)
        lines.append(line(x, y_pos, x + width, y_pos, "#e2e8f0"))
        lines.append(svg_text(x - 10, y_pos + 4, f"{int(round(value))}", size=12, fill="#475569", anchor="end"))

    for tick in range(max_x + 1):
        x_pos = x + (width * tick / max_x if max_x else 0)
        lines.append(line(x_pos, y, x_pos, y + height, "#f8fafc"))
        lines.append(svg_text(x_pos, y + height + 18, str(tick), size=11, fill="#475569", anchor="middle"))

    for phase, phase_data in data.groupby("phase", sort=False):
        points = []
        color = PHASE_COLORS.get(str(phase), "#2563eb")
        for row in phase_data.itertuples(index=False):
            px = x + (width * float(row.elapsed_hour) / max_x if max_x else 0)
            py = y + height - (height * float(row.n_active) / max_y)
            points.append((px, py))
        lines.append(polyline(points, color, stroke_width=3))
        for px, py in points:
            lines.append(circle(px, py, 2.8, color))
    lines.append(svg_text(x + width / 2, y + height + 42, "Elapsed hour from phase start", size=13, fill="#0f172a", anchor="middle"))
    return lines


def draw_simple_bar_panel(
    data: pd.DataFrame,
    x: int,
    y: int,
    width: int,
    height: int,
    value_col: str,
    y_label: str,
    x_col: str = "phase",
) -> list[str]:
    if data.empty:
        return [svg_text(x + width / 2, y + height / 2, "No data", size=18, fill="#64748b", anchor="middle")]

    max_value = float(data[value_col].fillna(0).max())
    max_value = max(max_value, 0.01)
    bar_count = len(data)
    gap = 24
    bar_width = (width - gap * (bar_count + 1)) / max(bar_count, 1)

    lines = [rect(x, y, width, height, "white", stroke="#cbd5e1"), svg_text(x, y - 12, y_label, size=15, fill="#0f172a")]
    for tick in range(5):
        value = max_value * tick / 4
        y_pos = y + height - (height * tick / 4)
        lines.append(line(x, y_pos, x + width, y_pos, "#e2e8f0"))
        lines.append(svg_text(x - 10, y_pos + 4, f"{value:.2f}", size=12, fill="#475569", anchor="end"))

    for idx, row in data.reset_index(drop=True).iterrows():
        left = x + gap + idx * (bar_width + gap)
        bar_height = 0 if pd.isna(row[value_col]) else (float(row[value_col]) / max_value) * (height - 10)
        top = y + height - bar_height
        fill = "#2563eb"
        lines.append(rect(left, top, bar_width, bar_height, fill))
        lines.append(svg_text(left + bar_width / 2, y + height + 20, str(row[x_col]), size=12, fill="#0f172a", anchor="middle"))
        lines.append(svg_text(left + bar_width / 2, top - 8, f"{float(row[value_col]):.3f}", size=12, fill="#0f172a", anchor="middle"))
    return lines


def winsorized_weighted_mean(
    values: pd.Series,
    weights: pd.Series,
    upper_quantile: float = 0.95,
) -> float | pd.NA:
    frame = pd.DataFrame({"value": values, "weight": weights}).dropna()
    frame = frame.loc[frame["weight"] > 0].copy()
    if frame.empty:
        return pd.NA
    cap = weighted_quantile(frame["value"], frame["weight"], upper_quantile)
    clipped = frame["value"].clip(upper=cap)
    return float((clipped * frame["weight"]).sum() / frame["weight"].sum())


def weighted_quantile(values: pd.Series, weights: pd.Series, quantile: float) -> float:
    frame = pd.DataFrame({"value": values, "weight": weights}).dropna().sort_values("value")
    frame = frame.loc[frame["weight"] > 0].copy()
    if frame.empty:
        return float("nan")
    cumulative = frame["weight"].cumsum()
    cutoff = quantile * frame["weight"].sum()
    return float(frame.loc[cumulative >= cutoff, "value"].iloc[0])


def svg_prelude(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def title(text: str, y: int) -> str:
    return svg_text(40, y, text, size=24, fill="#0f172a", weight="700")


def subtitle(text: str, y: int) -> str:
    return svg_text(40, y, text, size=14, fill="#475569")


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    stroke: str | None = None,
    opacity: str | None = None,
) -> str:
    stroke_attr = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
    opacity_attr = f' opacity="{opacity}"' if opacity else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{fill}"{stroke_attr}{opacity_attr} rx="6" />'


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
    width: int = 1,
    opacity: str | None = None,
    dasharray: str | None = None,
) -> str:
    opacity_attr = f' opacity="{opacity}"' if opacity else ""
    dash_attr = f' stroke-dasharray="{dasharray}"' if dasharray else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{opacity_attr}{dash_attr} />'


def polyline(
    points: list[tuple[float, float]],
    color: str,
    stroke_width: int = 3,
    opacity: str | None = None,
    dasharray: str | None = None,
) -> str:
    point_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    opacity_attr = f' opacity="{opacity}"' if opacity else ""
    dash_attr = f' stroke-dasharray="{dasharray}"' if dasharray else ""
    return f'<polyline points="{point_str}" fill="none" stroke="{color}" stroke-width="{stroke_width}"{opacity_attr}{dash_attr} />'


def circle(x: float, y: float, radius: float, fill: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" />'


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    fill: str = "#0f172a",
    anchor: str = "start",
    weight: str = "400",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
        f'font-family="Helvetica, Arial, sans-serif" font-weight="{weight}" text-anchor="{anchor}">'
        f"{html.escape(str(text))}</text>"
    )


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
