from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd

from .direct_sicu_analysis import (
    render_boarding_mortality_svg,
    render_intervention_rates_svg,
    run_direct_sicu_analysis,
    write_direct_sicu_outputs,
)
from .pipeline import (
    classify_adt_locations,
    collapse_respiratory_timestamps,
    empty_location_map,
    is_imv_device,
    is_truthy,
    load_location_map,
    normalize_token,
    read_named_table,
)


BOARDING_BINS = [0, 2, 4, 6, 12, 24, float("inf")]
BOARDING_LABELS = ["0-<2", "2-<4", "4-<6", "6-<12", "12-<24", "24+"]
SANKEY_COLOR_MAP = {
    "ED": "#c46210",
    "OR": "#8b5cf6",
    "ICU": "#0f766e",
    "Ward": "#2563eb",
    "Discharge": "#16a34a",
    "Death": "#dc2626",
}
SANKEY_ORDER = ["ED", "OR", "ICU", "Ward", "Discharge", "Death"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build collaborator-facing report artifacts for the CLIF trauma ventilation project."
    )
    parser.add_argument("--output-dir", required=True, help="Pipeline output directory containing cohort.csv and related files.")
    parser.add_argument(
        "--input-dir",
        help="Optional CLIF source directory used for diagnosis summaries and observed IMV duration estimates.",
    )
    parser.add_argument(
        "--location-map",
        help="Optional location crosswalk used when classifying raw ADT rows.",
    )
    parser.add_argument(
        "--diagnosis-dictionary",
        help="Optional CSV with diagnosis_code_format, diagnosis_code, and diagnosis_name columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    input_dir = Path(args.input_dir) if args.input_dir else None
    location_map_path = Path(args.location_map) if args.location_map else None
    diagnosis_dictionary_path = Path(args.diagnosis_dictionary) if args.diagnosis_dictionary else None
    build_full_report(
        output_dir,
        input_dir=input_dir,
        location_map_path=location_map_path,
        diagnosis_dictionary_path=diagnosis_dictionary_path,
    )


def build_full_report(
    output_dir: Path,
    input_dir: Path | None = None,
    location_map_path: Path | None = None,
    diagnosis_dictionary_path: Path | None = None,
) -> dict[str, object]:
    direct_artifacts = run_direct_sicu_analysis(output_dir)
    write_direct_sicu_outputs(output_dir, direct_artifacts)

    cohort, phase_windows, interventions, transfer_outcomes, cohort_flow = load_output_tables(output_dir)
    location_map = load_location_map(location_map_path) if location_map_path else empty_location_map()
    raw_artifacts = load_raw_subset(input_dir, cohort["hospitalization_id"].tolist(), location_map)
    diagnosis_dictionary = load_diagnosis_dictionary(diagnosis_dictionary_path)
    report_date_range = load_report_date_range(input_dir, cohort)

    imv_summary = build_observed_imv_summary(
        raw_artifacts.get("respiratory_support"),
        raw_artifacts.get("adt"),
        cohort,
    )
    table1 = build_table1(cohort, phase_windows, imv_summary)
    top_diagnoses = build_top_diagnoses_summary(
        raw_artifacts.get("hospital_diagnosis"),
        cohort,
        diagnosis_dictionary,
    )
    imv_vs_boarding = build_imv_vs_boarding_summary(cohort, imv_summary)
    intervention_svg = render_intervention_rates_svg(direct_artifacts["phase_rates"], direct_artifacts["elapsed_hour_rates"])
    boarding_svg = render_boarding_mortality_svg(direct_artifacts["boarding_mortality"])
    imv_svg = render_imv_vs_boarding_svg(imv_vs_boarding)
    consort_svg = render_consort_svg(cohort_flow, cohort, report_date_range)
    sankey_svg = render_sankey_svg(build_sankey_sequences(cohort, raw_artifacts.get("adt")))

    html_report = render_html_report(
        cohort=cohort,
        cohort_flow=cohort_flow,
        direct_artifacts=direct_artifacts,
        table1=table1,
        top_diagnoses=top_diagnoses,
        imv_vs_boarding=imv_vs_boarding,
        figures={
            "consort": consort_svg,
            "intervention": intervention_svg,
            "boarding": boarding_svg,
            "imv": imv_svg,
            "sankey": sankey_svg,
        },
        imv_summary_available=not imv_summary.empty,
        diagnoses_available=not top_diagnoses.empty,
    )

    write_report_outputs(
        output_dir=output_dir,
        table1=table1,
        top_diagnoses=top_diagnoses,
        imv_vs_boarding=imv_vs_boarding,
        figures={
            "cohort_consort.svg": consort_svg,
            "imv_trajectory_sankey.svg": sankey_svg,
            "direct_ed_to_sicu_imv_vs_boarding.svg": imv_svg,
        },
        html_report=html_report,
    )

    return {
        "table1": table1,
        "top_diagnoses": top_diagnoses,
        "imv_vs_boarding": imv_vs_boarding,
        "html_report": html_report,
    }


def load_output_tables(
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    transfer_outcomes = pd.read_csv(
        output_dir / "transfer_outcomes.csv",
        parse_dates=["transfer_outcome_dttm"],
    )
    cohort_flow = pd.read_csv(output_dir / "summary" / "cohort_flow.csv")
    for frame in [cohort, phase_windows, interventions, transfer_outcomes]:
        if "hospitalization_id" in frame.columns:
            frame["hospitalization_id"] = frame["hospitalization_id"].astype(str)
    return cohort, phase_windows, interventions, transfer_outcomes, cohort_flow


def load_raw_subset(
    input_dir: Path | None,
    cohort_ids: list[object],
    location_map: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if input_dir is None or not cohort_ids:
        return {}

    cohort_ids = [str(value) for value in cohort_ids]
    filters = [("hospitalization_id", "in", cohort_ids)]
    hospital_diagnosis = read_named_table(input_dir, "hospital_diagnosis", filters=filters)
    hospital_diagnosis["hospitalization_id"] = hospital_diagnosis["hospitalization_id"].astype(str)
    adt = read_named_table(input_dir, "adt", filters=filters)
    adt["hospitalization_id"] = adt["hospitalization_id"].astype(str)
    adt["in_dttm"] = pd.to_datetime(adt["in_dttm"], errors="coerce")
    adt["out_dttm"] = pd.to_datetime(adt["out_dttm"], errors="coerce")
    adt = classify_adt_locations(adt, location_map)
    adt = adt.sort_values(["hospitalization_id", "in_dttm", "out_dttm"], kind="stable").reset_index(drop=True)
    adt["next_in_dttm"] = adt.groupby("hospitalization_id")["in_dttm"].shift(-1)
    adt["segment_end_dttm"] = adt["out_dttm"].fillna(adt["next_in_dttm"])

    respiratory_support = read_named_table(input_dir, "respiratory_support", filters=filters)
    respiratory_support["hospitalization_id"] = respiratory_support["hospitalization_id"].astype(str)
    respiratory_support = collapse_respiratory_timestamps(respiratory_support)
    respiratory_support["recorded_dttm"] = pd.to_datetime(respiratory_support["recorded_dttm"], errors="coerce")
    respiratory_support["device_category_norm"] = respiratory_support["device_category"].map(normalize_token)
    respiratory_support["is_imv"] = respiratory_support["device_category_norm"].map(is_imv_device)
    return {
        "hospital_diagnosis": hospital_diagnosis,
        "adt": adt,
        "respiratory_support": respiratory_support,
    }


def load_diagnosis_dictionary(path: Path | None = None) -> pd.DataFrame:
    dictionary_path = path or Path(__file__).resolve().parents[2] / "config" / "icd10cm_dictionary.csv"
    if not dictionary_path.exists():
        return pd.DataFrame(columns=["diagnosis_code_format", "diagnosis_code", "diagnosis_name", "source"])
    dictionary = pd.read_csv(dictionary_path)
    required = {"diagnosis_code_format", "diagnosis_code", "diagnosis_name"}
    missing = required.difference(dictionary.columns)
    if missing:
        raise ValueError(f"Diagnosis dictionary is missing columns: {sorted(missing)}")
    dictionary = dictionary.copy()
    dictionary["diagnosis_code_format"] = dictionary["diagnosis_code_format"].astype(str)
    dictionary["diagnosis_code"] = dictionary["diagnosis_code"].astype(str)
    dictionary["diagnosis_name"] = dictionary["diagnosis_name"].astype(str)
    return dictionary.drop_duplicates(["diagnosis_code_format", "diagnosis_code"]).reset_index(drop=True)


def load_report_date_range(
    input_dir: Path | None,
    cohort: pd.DataFrame,
) -> dict[str, pd.Timestamp | pd.NaT]:
    if input_dir is not None:
        hospitalization = read_named_table(input_dir, "hospitalization")
        hospitalization["admission_dttm"] = pd.to_datetime(hospitalization["admission_dttm"], errors="coerce")
        hospitalization["discharge_dttm"] = pd.to_datetime(hospitalization["discharge_dttm"], errors="coerce")
        return {
            "admission_min": hospitalization["admission_dttm"].min(),
            "admission_max": hospitalization["admission_dttm"].max(),
            "discharge_max": hospitalization["discharge_dttm"].max(),
        }
    return {
        "admission_min": cohort["admission_dttm"].min(),
        "admission_max": cohort["admission_dttm"].max(),
        "discharge_max": cohort["discharge_dttm"].max(),
    }


def build_observed_imv_summary(
    respiratory_support: pd.DataFrame | None,
    adt: pd.DataFrame | None,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    if respiratory_support is None or adt is None or respiratory_support.empty:
        return pd.DataFrame(
            columns=[
                "hospitalization_id",
                "imv_episode_start_dttm",
                "imv_episode_end_dttm",
                "observed_total_imv_hours",
                "observed_ed_icu_imv_hours",
            ]
        )

    cohort_lookup = cohort.set_index("hospitalization_id")[
        ["first_imv_dttm", "discharge_dttm"]
    ]
    rows: list[dict[str, object]] = []
    for hospitalization_id, group in respiratory_support.groupby("hospitalization_id", sort=False):
        if hospitalization_id not in cohort_lookup.index:
            continue
        window = cohort_lookup.loc[hospitalization_id]
        start = window["first_imv_dttm"]
        if pd.isna(start):
            continue
        group = group.sort_values("recorded_dttm", kind="stable").copy()
        group = group.loc[group["recorded_dttm"] >= start]
        if group.empty:
            continue
        non_imv = group.loc[(~group["is_imv"]) & (group["recorded_dttm"] > start), "recorded_dttm"]
        if not non_imv.empty:
            end = non_imv.iloc[0]
        else:
            imv_rows = group.loc[group["is_imv"], "recorded_dttm"]
            if imv_rows.empty:
                continue
            end = imv_rows.iloc[-1]
        if pd.isna(end) or end < start:
            end = start
        observed_total_imv_hours = hours_between(start, end)
        observed_ed_icu_imv_hours = overlap_adt_hours(
            adt.loc[adt["hospitalization_id"] == hospitalization_id].copy(),
            start,
            end,
            include_mask=lambda row: bool(getattr(row, "is_ed", False)) or bool(getattr(row, "is_icu", False)),
        )
        rows.append(
            {
                "hospitalization_id": hospitalization_id,
                "imv_episode_start_dttm": start,
                "imv_episode_end_dttm": end,
                "observed_total_imv_hours": observed_total_imv_hours,
                "observed_ed_icu_imv_hours": observed_ed_icu_imv_hours,
            }
        )
    return pd.DataFrame.from_records(rows)


def overlap_adt_hours(
    adt_rows: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    include_mask,
) -> float | pd.NA:
    if adt_rows.empty or pd.isna(start) or pd.isna(end) or end < start:
        return pd.NA
    adt_rows = adt_rows.sort_values("in_dttm", kind="stable").copy()
    total = 0.0
    for row in adt_rows.itertuples(index=False):
        if not bool(include_mask(row)):
            continue
        segment_start = row.in_dttm
        segment_end = row.segment_end_dttm if pd.notna(row.segment_end_dttm) else end
        if pd.isna(segment_start) or pd.isna(segment_end):
            continue
        overlap_start = max(segment_start, start)
        overlap_end = min(segment_end, end)
        if overlap_end > overlap_start:
            total += (overlap_end - overlap_start).total_seconds() / 3600.0
    return total


def build_table1(
    cohort: pd.DataFrame,
    phase_windows: pd.DataFrame,
    imv_summary: pd.DataFrame,
) -> pd.DataFrame:
    data = cohort.copy()
    data["initial_sicu_los_hours"] = (
        data["initial_sicu_exit_dttm"] - data["sicu_in_dttm"]
    ).dt.total_seconds() / 3600.0
    if not imv_summary.empty:
        data = data.merge(
            imv_summary[["hospitalization_id", "observed_ed_icu_imv_hours"]],
            on="hospitalization_id",
            how="left",
        )
    else:
        data["observed_ed_icu_imv_hours"] = pd.NA

    phase_flags = (
        phase_windows.pivot_table(
            index="hospitalization_id",
            columns="phase",
            values="any_intervention",
            aggfunc="max",
        )
        .rename(columns={"ED": "ed_any_intervention", "SICU_24h": "sicu_any_intervention"})
        .reset_index()
    )
    data = data.merge(phase_flags, on="hospitalization_id", how="left")
    data["ed_any_intervention"] = data["ed_any_intervention"].fillna(False)
    data["sicu_any_intervention"] = data["sicu_any_intervention"].fillna(False)

    groups = {
        "Overall final cohort": data,
        "Direct ED -> SICU": data.loc[~data["has_procedural_bridge"].fillna(False)].copy(),
        "ED -> OR -> SICU": data.loc[data["has_procedural_bridge"].fillna(False)].copy(),
    }
    rows: list[dict[str, object]] = []

    rows.append(table1_row("Hospitalizations, n", groups, lambda frame: str(len(frame))))
    rows.append(table1_row("Age, years", groups, lambda frame: format_median_iqr(frame["age_at_admission"], digits=0)))

    for category in top_categories(data["sex_category"], max_categories=3):
        rows.append(
            table1_row(
                f"Sex: {category}, n (%)",
                groups,
                lambda frame, category=category: format_n_pct(frame["sex_category"].fillna("Unknown").eq(category).sum(), len(frame)),
            )
        )

    for category in top_categories(data["race_category"], max_categories=3):
        rows.append(
            table1_row(
                f"Race: {category}, n (%)",
                groups,
                lambda frame, category=category: format_n_pct(frame["race_category"].fillna("Unknown").eq(category).sum(), len(frame)),
            )
        )

    for category in top_categories(data["ethnicity_category"], max_categories=3):
        rows.append(
            table1_row(
                f"Ethnicity: {category}, n (%)",
                groups,
                lambda frame, category=category: format_n_pct(frame["ethnicity_category"].fillna("Unknown").eq(category).sum(), len(frame)),
            )
        )

    rows.extend(
        [
            table1_row("ED LOS, h", groups, lambda frame: format_median_iqr(frame["ed_los_hours"], digits=1)),
            table1_row("Initial SICU LOS, h", groups, lambda frame: format_median_iqr(frame["initial_sicu_los_hours"], digits=1)),
            table1_row("Hospital LOS, h", groups, lambda frame: format_median_iqr(frame["hospital_los_hours"], digits=1)),
            table1_row(
                "SICU-to-ward LOS, h",
                groups,
                lambda frame: format_median_iqr(frame["sicu_preward_los_hours"], digits=1),
            ),
            table1_row(
                "In-hospital mortality, n (%)",
                groups,
                lambda frame: format_n_pct(frame["in_hospital_mortality"].fillna(False).sum(), len(frame)),
            ),
            table1_row(
                "Any ED ventilator intervention, n (%)",
                groups,
                lambda frame: format_n_pct(frame["ed_any_intervention"].fillna(False).sum(), len(frame)),
            ),
            table1_row(
                "Any SICU 24h ventilator intervention, n (%)",
                groups,
                lambda frame: format_n_pct(frame["sicu_any_intervention"].fillna(False).sum(), len(frame)),
            ),
            table1_row(
                "First observed ED+ICU IMV time, h",
                groups,
                lambda frame: format_median_iqr(frame["observed_ed_icu_imv_hours"], digits=1),
            ),
            table1_row(
                "First post-SICU transfer to ward, n (%)",
                groups,
                lambda frame: format_n_pct(frame["transfer_outcome"].fillna("").eq("ward").sum(), len(frame)),
            ),
            table1_row(
                "First post-SICU transfer to OR, n (%)",
                groups,
                lambda frame: format_n_pct(frame["transfer_outcome"].fillna("").eq("procedural").sum(), len(frame)),
            ),
        ]
    )
    return pd.DataFrame(rows)


def table1_row(
    metric: str,
    groups: dict[str, pd.DataFrame],
    formatter,
) -> dict[str, object]:
    row = {"Metric": metric}
    for label, frame in groups.items():
        row[label] = formatter(frame) if not frame.empty else "0"
    return row


def build_top_diagnoses_summary(
    hospital_diagnosis: pd.DataFrame | None,
    cohort: pd.DataFrame,
    diagnosis_dictionary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if hospital_diagnosis is None or hospital_diagnosis.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "diagnosis_code_format",
                "diagnosis_code",
                "diagnosis_name",
                "overall_n",
                "overall_pct",
                "direct_ed_to_sicu_n",
                "ed_to_or_to_sicu_n",
            ]
        )

    diagnosis = hospital_diagnosis.copy()
    diagnosis["primary_flag"] = diagnosis["diagnosis_primary"].map(is_truthy)
    diagnosis = diagnosis.loc[diagnosis["primary_flag"]].copy()
    if diagnosis.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "diagnosis_code_format",
                "diagnosis_code",
                "diagnosis_name",
                "overall_n",
                "overall_pct",
                "direct_ed_to_sicu_n",
                "ed_to_or_to_sicu_n",
            ]
        )

    direct_ids = set(cohort.loc[~cohort["has_procedural_bridge"].fillna(False), "hospitalization_id"])
    bridge_ids = set(cohort.loc[cohort["has_procedural_bridge"].fillna(False), "hospitalization_id"])
    overall_total = cohort["hospitalization_id"].nunique()

    overall = diagnosis.groupby(["diagnosis_code_format", "diagnosis_code"], as_index=False)["hospitalization_id"].nunique()
    overall = overall.rename(columns={"hospitalization_id": "overall_n"})
    overall["overall_pct"] = overall["overall_n"] / overall_total

    direct = (
        diagnosis.loc[diagnosis["hospitalization_id"].isin(direct_ids)]
        .groupby(["diagnosis_code_format", "diagnosis_code"], as_index=False)["hospitalization_id"]
        .nunique()
        .rename(columns={"hospitalization_id": "direct_ed_to_sicu_n"})
    )
    bridge = (
        diagnosis.loc[diagnosis["hospitalization_id"].isin(bridge_ids)]
        .groupby(["diagnosis_code_format", "diagnosis_code"], as_index=False)["hospitalization_id"]
        .nunique()
        .rename(columns={"hospitalization_id": "ed_to_or_to_sicu_n"})
    )
    summary = (
        overall.merge(direct, on=["diagnosis_code_format", "diagnosis_code"], how="left")
        .merge(bridge, on=["diagnosis_code_format", "diagnosis_code"], how="left")
        .fillna(0)
        .sort_values(["overall_n", "diagnosis_code"], ascending=[False, True], kind="stable")
        .head(10)
        .reset_index(drop=True)
    )
    if diagnosis_dictionary is not None and not diagnosis_dictionary.empty:
        summary = summary.merge(
            diagnosis_dictionary[["diagnosis_code_format", "diagnosis_code", "diagnosis_name"]],
            on=["diagnosis_code_format", "diagnosis_code"],
            how="left",
        )
    else:
        summary["diagnosis_name"] = pd.NA
    summary["rank"] = summary.index + 1
    for column in ["direct_ed_to_sicu_n", "ed_to_or_to_sicu_n"]:
        summary[column] = summary[column].astype(int)
    summary["overall_n"] = summary["overall_n"].astype(int)
    summary["diagnosis_name"] = summary["diagnosis_name"].fillna("Description unavailable")
    return summary[
        [
            "rank",
            "diagnosis_code_format",
            "diagnosis_code",
            "diagnosis_name",
            "overall_n",
            "overall_pct",
            "direct_ed_to_sicu_n",
            "ed_to_or_to_sicu_n",
        ]
    ]


def build_imv_vs_boarding_summary(
    cohort: pd.DataFrame,
    imv_summary: pd.DataFrame,
) -> pd.DataFrame:
    direct = cohort.loc[~cohort["has_procedural_bridge"].fillna(False)].copy()
    if imv_summary.empty:
        return pd.DataFrame(
            columns=[
                "boarding_bin",
                "hospitalizations",
                "median_observed_ed_icu_imv_hours",
                "q1_observed_ed_icu_imv_hours",
                "q3_observed_ed_icu_imv_hours",
            ]
        )

    direct = direct.merge(
        imv_summary[["hospitalization_id", "observed_ed_icu_imv_hours"]],
        on="hospitalization_id",
        how="left",
    )
    direct["boarding_bin"] = pd.cut(
        direct["ed_los_hours"],
        bins=BOARDING_BINS,
        labels=BOARDING_LABELS,
        right=False,
        include_lowest=True,
    )
    summary = (
        direct.groupby("boarding_bin", observed=False, as_index=False)
        .agg(
            hospitalizations=("hospitalization_id", "size"),
            median_observed_ed_icu_imv_hours=("observed_ed_icu_imv_hours", "median"),
            q1_observed_ed_icu_imv_hours=("observed_ed_icu_imv_hours", lambda series: series.quantile(0.25)),
            q3_observed_ed_icu_imv_hours=("observed_ed_icu_imv_hours", lambda series: series.quantile(0.75)),
        )
    )
    return summary


def build_sankey_sequences(
    cohort: pd.DataFrame,
    adt: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    max_len = 0
    sequences: list[list[str]] = []
    adt_lookup = (
        {
            hospitalization_id: group.sort_values(["in_dttm", "out_dttm"], kind="stable")
            for hospitalization_id, group in adt.groupby("hospitalization_id", sort=False)
        }
        if adt is not None and not adt.empty
        else {}
    )

    for row in cohort.itertuples(index=False):
        if row.hospitalization_id in adt_lookup:
            sequence = build_location_sequence_from_adt(adt_lookup[row.hospitalization_id])
        else:
            sequence = ["ED"]
            if bool(row.has_procedural_bridge):
                sequence.extend(["OR", "ICU"])
            else:
                sequence.append("ICU")
            post_sicu_node = {
                "ward": "Ward",
                "procedural": "OR",
                "another_icu": "ICU",
                "death": "Death",
                "discharge": "Discharge",
            }.get(str(getattr(row, "transfer_outcome", "")), None)
            if post_sicu_node and post_sicu_node != sequence[-1]:
                sequence.append(post_sicu_node)
        final_outcome = "Death" if bool(row.in_hospital_mortality) else "Discharge"
        if not sequence or sequence[-1] != final_outcome:
            sequence.append(final_outcome)
        sequences.append(sequence)
        max_len = max(max_len, len(sequence))

    for hospitalization_id, sequence in zip(cohort["hospitalization_id"], sequences):
        record = {"hospitalization_id": hospitalization_id}
        for idx in range(max_len):
            record[f"stage_{idx}"] = sequence[idx] if idx < len(sequence) else pd.NA
        rows.append(record)
    return pd.DataFrame(rows)


def build_location_sequence_from_adt(adt_rows: pd.DataFrame) -> list[str]:
    sequence: list[str] = []
    for row in adt_rows.itertuples(index=False):
        label = adt_row_label(row)
        if label is None:
            continue
        if not sequence or sequence[-1] != label:
            sequence.append(label)
    return sequence


def adt_row_label(row) -> str | None:
    if bool(getattr(row, "is_ed", False)):
        return "ED"
    if bool(getattr(row, "is_procedural", False)):
        return "OR"
    if bool(getattr(row, "is_icu", False)):
        return "ICU"
    if bool(getattr(row, "is_ward", False)):
        return "Ward"
    return None


def render_consort_svg(
    cohort_flow: pd.DataFrame,
    cohort: pd.DataFrame,
    date_range: dict[str, pd.Timestamp | pd.NaT],
) -> str:
    counts = dict(zip(cohort_flow["stage"], cohort_flow["count"]))
    direct_n = int((~cohort["has_procedural_bridge"].fillna(False)).sum())
    bridge_n = int(cohort["has_procedural_bridge"].fillna(False).sum())
    date_caption = (
        f"Admissions from {format_calendar_date(date_range['admission_min'])} "
        f"through {format_calendar_date(date_range['admission_max'])}, inclusive; "
        f"follow-up through {format_calendar_date(date_range['discharge_max'])}."
    )

    width = 980
    height = 860
    box_width = 420
    box_height = 82
    center_x = 280
    lines = [
        svg_prelude(width, height),
        rect(0, 0, width, height, "#f8fafc"),
        title("Cohort flow", 38),
        subtitle("Mechanically ventilated trauma patients meeting the ED to SICU pathway definition", 66),
        subtitle(date_caption, 88),
    ]
    stages = [
        ("Adult hospitalizations", counts.get("adult_hospitalizations", 0), 128),
        ("Trauma-coded hospitalizations", counts.get("adult_trauma_hospitalizations", 0), 253),
        ("Trauma hospitalizations with ED exposure", counts.get("adult_trauma_with_ed", 0), 378),
        ("Valid ED -> SICU or ED -> OR -> SICU pathway", counts.get("adult_trauma_with_valid_pathway", 0), 503),
        ("Final cohort with first IMV documented in ED", counts.get("final_cohort", 0), 628),
    ]

    for idx, (label, count, top) in enumerate(stages):
        lines.append(rect(center_x, top, box_width, box_height, "white", stroke="#cbd5e1"))
        lines.append(svg_text(center_x + 24, top + 34, label, size=16, fill="#0f172a", weight="700"))
        lines.append(svg_text(center_x + 24, top + 60, f"n = {format_int(count)}", size=15, fill="#334155"))
        if idx < len(stages) - 1:
            start_y = top + box_height
            next_top = stages[idx + 1][2]
            lines.append(line(width / 2, start_y, width / 2, next_top, "#94a3b8", width=3))
            lines.append(arrowhead(width / 2, next_top, "#94a3b8"))

    branch_y = 710
    left_x = 70
    right_x = 520
    branch_box_width = 320
    branch_box_height = 100
    lines.append(line(width / 2, 692, width / 2, 730, "#94a3b8", width=3))
    lines.append(line(width / 2, 730, left_x + branch_box_width / 2, 730, "#94a3b8", width=3))
    lines.append(line(width / 2, 730, right_x + branch_box_width / 2, 730, "#94a3b8", width=3))
    lines.append(line(left_x + branch_box_width / 2, 730, left_x + branch_box_width / 2, 760, "#94a3b8", width=3))
    lines.append(line(right_x + branch_box_width / 2, 730, right_x + branch_box_width / 2, 760, "#94a3b8", width=3))
    lines.append(arrowhead(left_x + branch_box_width / 2, 760, "#94a3b8"))
    lines.append(arrowhead(right_x + branch_box_width / 2, 760, "#94a3b8"))

    lines.append(rect(left_x, 760, branch_box_width, branch_box_height, "#fff7ed", stroke="#fdba74"))
    lines.append(svg_text(left_x + 20, 798, "Direct ED -> SICU subgroup", size=16, fill="#9a3412", weight="700"))
    lines.append(svg_text(left_x + 20, 826, f"n = {format_int(direct_n)}", size=15, fill="#7c2d12"))
    lines.append(svg_text(left_x + 20, 846, "Primary hourly intervention and boarding analyses", size=13, fill="#9a3412"))

    lines.append(rect(right_x, 760, branch_box_width, branch_box_height, "#f5f3ff", stroke="#c4b5fd"))
    lines.append(svg_text(right_x + 20, 798, "ED -> OR -> SICU bridge subgroup", size=16, fill="#5b21b6", weight="700"))
    lines.append(svg_text(right_x + 20, 826, f"n = {format_int(bridge_n)}", size=15, fill="#4c1d95"))
    lines.append(svg_text(right_x + 20, 846, "Retained for broader trajectory and outcome summaries", size=13, fill="#5b21b6"))
    lines.append("</svg>")
    return "\n".join(lines)


def render_sankey_svg(sequences: pd.DataFrame) -> str:
    if sequences.empty:
        return empty_figure("Trajectory sankey unavailable")

    stage_columns = [column for column in sequences.columns if column.startswith("stage_")]
    stage_titles = ["Arrival"] + [f"Transition {idx}" for idx in range(1, len(stage_columns) - 1)] + ["Outcome"]
    width = max(1280, 240 * len(stage_columns) + 120)
    height = 820
    node_width = 26
    top_margin = 110
    bottom_margin = 48
    node_padding = 18
    x_positions = [70 + idx * ((width - 140) / max(len(stage_columns) - 1, 1)) for idx in range(len(stage_columns))]
    scale = (height - top_margin - bottom_margin) / max(len(sequences), 1)

    node_layout: dict[tuple[int, str], dict[str, float]] = {}
    stage_counts: dict[int, pd.Series] = {}
    for idx, column in enumerate(stage_columns):
        counts = sequences[column].dropna().value_counts()
        counts = counts.reindex([label for label in SANKEY_ORDER if label in counts.index])
        used_height = counts.sum() * scale + max(len(counts) - 1, 0) * node_padding
        y_cursor = top_margin + max((height - top_margin - bottom_margin - used_height) / 2, 0)
        for label, count in counts.items():
            node_height = count * scale
            node_layout[(idx, label)] = {
                "x": x_positions[idx],
                "y": y_cursor,
                "height": node_height,
                "count": int(count),
            }
            y_cursor += node_height + node_padding

    flow_rows = []
    for idx in range(len(stage_columns) - 1):
        left = stage_columns[idx]
        right = stage_columns[idx + 1]
        pair_counts = (
            sequences[[left, right]]
            .dropna()
            .groupby([left, right], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        pair_counts["stage"] = idx
        flow_rows.append(pair_counts)
    flow_data = pd.concat(flow_rows, ignore_index=True) if flow_rows else pd.DataFrame(columns=["stage", "count"])

    source_offsets = {(stage, label): 0.0 for stage, label in node_layout}
    target_offsets = {(stage, label): 0.0 for stage, label in node_layout}
    paths: list[str] = []
    for stage in range(len(stage_columns) - 1):
        stage_flows = flow_data.loc[flow_data["stage"] == stage].copy()
        stage_flows[left_column_name(stage_columns, stage)] = stage_flows[left_column_name(stage_columns, stage)].astype(str)
        stage_flows[right_column_name(stage_columns, stage)] = stage_flows[right_column_name(stage_columns, stage)].astype(str)
        stage_flows["source_rank"] = stage_flows[left_column_name(stage_columns, stage)].map(SANKEY_ORDER.index)
        stage_flows["target_rank"] = stage_flows[right_column_name(stage_columns, stage)].map(SANKEY_ORDER.index)
        stage_flows = stage_flows.sort_values(["source_rank", "target_rank"], kind="stable")
        for row in stage_flows.itertuples(index=False):
            source_label = getattr(row, left_column_name(stage_columns, stage))
            target_label = getattr(row, right_column_name(stage_columns, stage))
            source = node_layout[(stage, source_label)]
            target = node_layout[(stage + 1, target_label)]
            band_height = row.count * scale
            sy0 = source["y"] + source_offsets[(stage, source_label)]
            sy1 = sy0 + band_height
            ty0 = target["y"] + target_offsets[(stage + 1, target_label)]
            ty1 = ty0 + band_height
            source_offsets[(stage, source_label)] += band_height
            target_offsets[(stage + 1, target_label)] += band_height
            x0 = source["x"] + node_width
            x1 = target["x"]
            paths.append(
                sankey_band(
                    x0=x0,
                    x1=x1,
                    sy0=sy0,
                    sy1=sy1,
                    ty0=ty0,
                    ty1=ty1,
                    fill=SANKEY_COLOR_MAP.get(source_label, "#94a3b8"),
                )
            )

    lines = [
        svg_prelude(width, height),
        rect(0, 0, width, height, "#f8fafc"),
        title("Unit trajectory sankey", 38),
        subtitle("Raw ADT-derived location sequences for the final mechanically ventilated trauma cohort", 66),
        subtitle("Includes ICU -> OR -> ICU returns and ED -> OR bridge cases when they occur in the source trajectory.", 88),
    ]
    lines.extend(paths)

    for idx, column in enumerate(stage_columns):
        lines.append(svg_text(x_positions[idx] + node_width / 2, 96, stage_titles[idx], size=14, fill="#475569", anchor="middle"))
        for label, meta in node_layout.items():
            if label[0] != idx:
                continue
            node_label = label[1]
            color = SANKEY_COLOR_MAP.get(node_label, "#94a3b8")
            lines.append(rect(meta["x"], meta["y"], node_width, meta["height"], color, opacity="0.95"))
            label_y = meta["y"] + min(meta["height"] / 2 + 5, meta["height"] - 4)
            lines.append(svg_text(meta["x"] + node_width + 10, label_y - 6, node_label, size=13, fill="#0f172a"))
            lines.append(svg_text(meta["x"] + node_width + 10, label_y + 12, f"n={format_int(meta['count'])}", size=12, fill="#475569"))

    lines.append("</svg>")
    return "\n".join(lines)


def left_column_name(stage_columns: list[str], stage_idx: int) -> str:
    return stage_columns[stage_idx]


def right_column_name(stage_columns: list[str], stage_idx: int) -> str:
    return stage_columns[stage_idx + 1]


def render_imv_vs_boarding_svg(summary: pd.DataFrame) -> str:
    if summary.empty:
        return empty_figure("Observed IMV duration summary unavailable")

    width = 980
    height = 520
    x = 90
    y = 120
    plot_width = 820
    plot_height = 300
    max_value = float(summary["q3_observed_ed_icu_imv_hours"].fillna(0).max())
    max_value = max(max_value, 1.0)
    gap = 18
    bar_width = (plot_width - gap * (len(summary) + 1)) / max(len(summary), 1)

    lines = [
        svg_prelude(width, height),
        rect(0, 0, width, height, "#f8fafc"),
        title("Observed ED+ICU IMV duration by ED LOS bucket", 36),
        subtitle("Direct ED -> SICU subgroup; bars show median first IMV episode time and whiskers show IQR", 62),
        rect(x, y, plot_width, plot_height, "white", stroke="#cbd5e1"),
    ]
    for tick in range(5):
        value = max_value * tick / 4
        y_pos = y + plot_height - (plot_height * tick / 4)
        lines.append(horizontal_rule(x, y_pos, x + plot_width, "#e2e8f0"))
        lines.append(svg_text(x - 10, y_pos + 4, f"{value:.0f}", size=12, fill="#475569", anchor="end"))
    lines.append(svg_text(x + plot_width / 2, y + plot_height + 42, "ED LOS bucket, hours", size=13, fill="#0f172a", anchor="middle"))
    lines.append(svg_text(26, y + plot_height / 2, "Observed ED+ICU IMV hours", size=13, fill="#0f172a"))

    for idx, row in summary.reset_index(drop=True).iterrows():
        left = x + gap + idx * (bar_width + gap)
        median = 0.0 if pd.isna(row["median_observed_ed_icu_imv_hours"]) else float(row["median_observed_ed_icu_imv_hours"])
        q1 = 0.0 if pd.isna(row["q1_observed_ed_icu_imv_hours"]) else float(row["q1_observed_ed_icu_imv_hours"])
        q3 = 0.0 if pd.isna(row["q3_observed_ed_icu_imv_hours"]) else float(row["q3_observed_ed_icu_imv_hours"])
        bar_height = (median / max_value) * (plot_height - 10)
        top = y + plot_height - bar_height
        lines.append(rect(left, top, bar_width, bar_height, "#0f766e"))
        q1_y = y + plot_height - (q1 / max_value) * plot_height
        q3_y = y + plot_height - (q3 / max_value) * plot_height
        center_x = left + bar_width / 2
        lines.append(vertical_rule(center_x, q3_y, q1_y, "#0f172a", width=2))
        lines.append(horizontal_rule(center_x - 10, q1_y, center_x + 10, "#0f172a", width=2))
        lines.append(horizontal_rule(center_x - 10, q3_y, center_x + 10, "#0f172a", width=2))
        lines.append(svg_text(center_x, y + plot_height + 20, str(row["boarding_bin"]), size=12, fill="#0f172a", anchor="middle"))
        lines.append(svg_text(center_x, top - 8, f"{median:.1f}", size=11, fill="#0f172a", anchor="middle"))
        lines.append(svg_text(center_x, y + plot_height + 40, f"n={int(row['hospitalizations'])}", size=11, fill="#475569", anchor="middle"))
    lines.append("</svg>")
    return "\n".join(lines)


def render_html_report(
    cohort: pd.DataFrame,
    cohort_flow: pd.DataFrame,
    direct_artifacts: dict[str, pd.DataFrame],
    table1: pd.DataFrame,
    top_diagnoses: pd.DataFrame,
    imv_vs_boarding: pd.DataFrame,
    figures: dict[str, str],
    imv_summary_available: bool,
    diagnoses_available: bool,
) -> str:
    direct_n = int((~cohort["has_procedural_bridge"].fillna(False)).sum())
    bridge_n = int(cohort["has_procedural_bridge"].fillna(False).sum())
    mortality_rate = cohort["in_hospital_mortality"].mean()
    median_ed_los = cohort["ed_los_hours"].median()
    median_hospital_los = cohort["hospital_los_hours"].median()
    phase_rates = direct_artifacts["phase_rates"].copy()

    key_cards = [
        ("Final cohort", format_int(len(cohort)), "Adults with trauma, valid ED-to-SICU pathway, and first IMV documented in the ED"),
        ("Direct ED -> SICU", format_int(direct_n), "Primary subgroup for hourly intervention and boarding analyses"),
        ("ED -> OR -> SICU", format_int(bridge_n), "Retained in the broader trajectory and outcome cohort"),
        ("In-hospital mortality", format_pct(mortality_rate), "From CLIF discharge category death mapping"),
        ("Median ED LOS", f"{median_ed_los:.1f} h", "Measured from first ED ADT in/out interval"),
        ("Median hospital LOS", f"{median_hospital_los:.1f} h", "Measured from admission to discharge"),
    ]

    phase_rate_table = phase_rates.copy()
    for column in ["mean_patient_rate", "winsorized_mean_patient_rate", "median_patient_rate"]:
        if column in phase_rate_table.columns:
            phase_rate_table[column] = phase_rate_table[column].map(format_rate_value)
    if "pct_zero" in phase_rate_table.columns:
        phase_rate_table["pct_zero"] = phase_rate_table["pct_zero"].map(lambda value: "NA" if pd.isna(value) else format_pct(float(value)))
    phase_rate_table["hospitalizations"] = phase_rate_table["hospitalizations"].map(format_int)
    phase_rate_table["total_phase_hours"] = phase_rate_table["total_phase_hours"].map(lambda value: f"{float(value):.1f}")
    phase_rate_table["total_interventions"] = phase_rate_table["total_interventions"].map(format_int)
    diagnosis_table = top_diagnoses.copy()
    if not diagnosis_table.empty and "overall_pct" in diagnosis_table.columns:
        diagnosis_table["overall_pct"] = diagnosis_table["overall_pct"].map(lambda value: format_pct(float(value)))

    notes = []
    notes.append("Hourly intervention plots use the direct ED -> SICU subgroup to reduce confounding from immediate ED-to-OR triage.")
    notes.append("Solid lines represent an upper-95% winsorized mean rate, with dashed lines showing the raw mean and a denominator panel showing active patients by hour.")
    if imv_summary_available:
        notes.append("Observed IMV duration is based on the first IMV episode after ED intubation and sums only ED and ICU time, excluding OR and ward intervals.")
    if diagnoses_available:
        notes.append("Top diagnoses are labeled with CDC FY2024 ICD-10-CM names when available; codes without a local match fall back to a code-only label.")
    notes.append("This repository and report were developed with assistance from OpenAI Codex, with human review and responsibility for the analysis.")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>CLIF Trauma Ventilation Report</title>
  <style>
    body {{
      margin: 0;
      padding: 32px;
      font-family: Helvetica, Arial, sans-serif;
      background: #e2e8f0;
      color: #0f172a;
    }}
    .page {{
      max-width: 1280px;
      margin: 0 auto;
    }}
    .hero {{
      background: linear-gradient(135deg, #fff7ed 0%, #eff6ff 55%, #ecfeff 100%);
      border: 1px solid #cbd5e1;
      border-radius: 20px;
      padding: 28px 32px;
      margin-bottom: 24px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 34px;
    }}
    .hero p {{
      margin: 0;
      max-width: 980px;
      line-height: 1.5;
      color: #334155;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 14px;
      margin-top: 22px;
    }}
    .card {{
      background: rgba(255,255,255,0.88);
      border: 1px solid #dbeafe;
      border-radius: 16px;
      padding: 16px;
      min-height: 108px;
    }}
    .card .label {{
      font-size: 13px;
      color: #475569;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    .card .value {{
      font-size: 28px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .card .detail {{
      font-size: 13px;
      color: #475569;
      line-height: 1.45;
    }}
    .section {{
      background: white;
      border: 1px solid #cbd5e1;
      border-radius: 18px;
      padding: 24px;
      margin-bottom: 24px;
    }}
    .section h2 {{
      margin: 0 0 10px;
      font-size: 24px;
    }}
    .section p {{
      margin: 0 0 16px;
      color: #334155;
      line-height: 1.55;
    }}
    .two-up {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
      gap: 18px;
      align-items: start;
    }}
    .figure {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 16px;
      padding: 12px;
      overflow: auto;
    }}
    .figure svg {{
      max-width: 100%;
      height: auto;
      display: block;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      margin-top: 10px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid #e2e8f0;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f8fafc;
      color: #334155;
      font-weight: 700;
    }}
    .small {{
      font-size: 13px;
      color: #475569;
    }}
    ul {{
      margin: 10px 0 0 18px;
      color: #334155;
      line-height: 1.55;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Trauma ventilation management across the ED and SICU</h1>
      <p>This report summarizes the CLIF trauma ventilation project using adult trauma hospitalizations with invasive mechanical ventilation initiated in the ED and a valid ED-to-SICU pathway. The primary hourly intervention analyses focus on the direct ED -> SICU subgroup, while trajectory and outcome summaries retain bridge patients who went from the ED to the OR before SICU arrival.</p>
      <div class="cards">
        {''.join(render_card(label, value, detail) for label, value, detail in key_cards)}
      </div>
    </section>

    <section class="section">
      <h2>Cohort flow and patient trajectory</h2>
      <p>The funnel below shows how the final mechanically ventilated trauma cohort was assembled. The accompanying sankey plot retains the broader final cohort, including ED -> OR -> SICU bridge cases, to show early location transitions and terminal outcomes.</p>
      <div class="two-up">
        <div class="figure">{figures["consort"]}</div>
        <div class="figure">{figures["sankey"]}</div>
      </div>
    </section>

    <section class="section">
      <h2>Ventilator intervention frequency</h2>
      <p>The intervention figure adds a robustness companion metric for the sparse hourly counts. Dashed curves show the raw mean events per patient-hour, solid curves show the upper-95% winsorized mean, and the bottom panel shows how many patients remain at risk in each elapsed hour.</p>
      <div class="figure">{figures["intervention"]}</div>
      {dataframe_to_html(phase_rate_table, class_name="phase-rate-table")}
    </section>

    <section class="section">
      <h2>Boarding, mortality, and observed IMV duration</h2>
      <p>These panels keep the direct ED -> SICU subgroup and use the same ED LOS buckets to support a cleaner boarding analysis, while avoiding the most acute ED-to-OR triage pattern.</p>
      <div class="two-up">
        <div class="figure">{figures["boarding"]}</div>
        <div class="figure">{figures["imv"]}</div>
      </div>
      {dataframe_to_html(imv_vs_boarding, class_name="imv-table")}
    </section>

    <section class="section">
      <h2>Table 1</h2>
      <p>Table 1 summarizes the final cohort overall and then splits it into the direct ED -> SICU and ED -> OR -> SICU bridge subgroups.</p>
      {dataframe_to_html(table1, class_name="table-one")}
      <p class="small">SICU-to-ward LOS is only defined for patients whose first post-SICU transfer was to a ward. The observed IMV measure reflects the first IMV episode after ED intubation and sums only ED and ICU time.</p>
    </section>

    <section class="section">
      <h2>Top primary diagnoses</h2>
      <p>This table shows the most common primary diagnosis codes among the final cohort with CDC FY2024 ICD-10-CM names added for readability. Counts are displayed for the overall cohort and both pathway subgroups.</p>
      {dataframe_to_html(diagnosis_table, class_name="diagnosis-table") if not top_diagnoses.empty else '<p class="small">Top diagnosis summaries were not generated because the source hospital diagnosis table was not available during this run.</p>'}
    </section>

    <section class="section">
      <h2>Interpretation notes</h2>
      <ul>
        {''.join(f'<li>{html.escape(note)}</li>' for note in notes)}
      </ul>
    </section>
  </div>
</body>
</html>"""


def render_card(label: str, value: str, detail: str) -> str:
    return (
        '<div class="card">'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div>'
        f'<div class="detail">{html.escape(detail)}</div>'
        "</div>"
    )


def dataframe_to_html(frame: pd.DataFrame, class_name: str) -> str:
    if frame.empty:
        return '<p class="small">No rows available.</p>'
    return frame.to_html(index=False, classes=class_name, escape=False, border=0)


def write_report_outputs(
    output_dir: Path,
    table1: pd.DataFrame,
    top_diagnoses: pd.DataFrame,
    imv_vs_boarding: pd.DataFrame,
    figures: dict[str, str],
    html_report: str,
) -> None:
    summary_dir = output_dir / "summary"
    figures_dir = output_dir / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    table1.to_csv(summary_dir / "table1.csv", index=False)
    top_diagnoses.to_csv(summary_dir / "top_primary_diagnoses.csv", index=False)
    imv_vs_boarding.to_csv(summary_dir / "direct_ed_to_sicu_imv_vs_boarding.csv", index=False)
    for name, content in figures.items():
        write_text(figures_dir / name, content)
    write_text(output_dir / "trauma_ventilation_report.html", html_report)


def top_categories(series: pd.Series, max_categories: int) -> list[str]:
    counts = series.fillna("Unknown").astype(str).value_counts()
    return counts.head(max_categories).index.tolist()


def format_median_iqr(series: pd.Series, digits: int = 1) -> str:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return "NA"
    median = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return f"{median:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"


def format_n_pct(count: float, denominator: int) -> str:
    if denominator == 0:
        return "0"
    return f"{int(count)} ({count / denominator:.1%})"


def format_pct(value: float) -> str:
    return f"{value:.1%}"


def format_rate_value(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def format_int(value: object) -> str:
    return f"{int(value):,}"


def format_calendar_date(value: pd.Timestamp | pd.NaT) -> str:
    if pd.isna(value):
        return "unknown date"
    return pd.Timestamp(value).strftime("%B %d, %Y").replace(" 0", " ")


def hours_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return (end - start).total_seconds() / 3600.0


def empty_figure(message: str) -> str:
    return "\n".join(
        [
            svg_prelude(800, 300),
            rect(0, 0, 800, 300, "#f8fafc"),
            rect(40, 60, 720, 180, "white", stroke="#cbd5e1"),
            svg_text(400, 158, message, size=24, fill="#475569", anchor="middle"),
            "</svg>",
        ]
    )


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
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{fill}"{stroke_attr}{opacity_attr} rx="8" />'


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
    width: int = 1,
) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}" />'


def horizontal_rule(x1: float, y: float, x2: float, color: str, width: int = 1) -> str:
    return f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="{width}" />'


def vertical_rule(x: float, y1: float, y2: float, color: str, width: int = 1) -> str:
    return f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}" />'


def arrowhead(x: float, y: float, color: str) -> str:
    return f'<polygon points="{x - 8:.1f},{y - 10:.1f} {x + 8:.1f},{y - 10:.1f} {x:.1f},{y:.1f}" fill="{color}" />'


def sankey_band(
    x0: float,
    x1: float,
    sy0: float,
    sy1: float,
    ty0: float,
    ty1: float,
    fill: str,
) -> str:
    curve = (x1 - x0) * 0.42
    return (
        f'<path d="M{x0:.1f},{sy0:.1f} '
        f'C{x0 + curve:.1f},{sy0:.1f} {x1 - curve:.1f},{ty0:.1f} {x1:.1f},{ty0:.1f} '
        f'L{x1:.1f},{ty1:.1f} '
        f'C{x1 - curve:.1f},{ty1:.1f} {x0 + curve:.1f},{sy1:.1f} {x0:.1f},{sy1:.1f} Z" '
        f'fill="{fill}" opacity="0.35" stroke="none" />'
    )


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
