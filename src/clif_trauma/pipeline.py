from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


REQUIRED_COLUMNS = {
    "patient": [
        "patient_id",
        "sex_category",
        "race_category",
        "ethnicity_category",
    ],
    "hospitalization": [
        "patient_id",
        "hospitalization_id",
        "admission_dttm",
        "discharge_dttm",
        "age_at_admission",
        "admission_type_name",
        "admission_type_category",
        "discharge_category",
    ],
    "hospital_diagnosis": [
        "hospitalization_id",
        "diagnosis_code",
        "diagnosis_code_format",
        "diagnosis_primary",
        "poa_present",
    ],
    "adt": [
        "hospitalization_id",
        "in_dttm",
        "out_dttm",
        "location_name",
        "location_category",
        "location_type",
    ],
    "respiratory_support": [
        "hospitalization_id",
        "recorded_dttm",
        "device_category",
        "mode_category",
        "tracheostomy",
        "fio2_set",
        "tidal_volume_set",
        "resp_rate_set",
        "pressure_control_set",
        "pressure_support_set",
        "peep_set",
        "tidal_volume_obs",
        "resp_rate_obs",
        "plateau_pressure_obs",
        "peak_inspiratory_pressure_obs",
        "peep_obs",
        "minute_vent_obs",
        "mean_airway_pressure_obs",
    ],
    "patient_assessments": [
        "hospitalization_id",
        "recorded_dttm",
        "assessment_category",
        "assessment_group",
        "numerical_value",
        "categorical_value",
    ],
}

INTERVENTION_COLUMNS = [
    "mode_category",
    "fio2_set",
    "tidal_volume_set",
    "resp_rate_set",
    "pressure_control_set",
    "pressure_support_set",
    "peep_set",
]

PHASE_ORDER = ["ED", "SICU_24h"]


@dataclass(frozen=True)
class AnalysisArtifacts:
    cohort: pd.DataFrame
    phase_windows: pd.DataFrame
    interventions: pd.DataFrame
    transfer_outcomes: pd.DataFrame
    assessment_context: pd.DataFrame
    cohort_flow: pd.DataFrame
    phase_intervention_summary: pd.DataFrame
    handoff_summary: pd.DataFrame
    outcome_summary: pd.DataFrame
    transfer_summary: pd.DataFrame


def run_pipeline(
    input_dir: Path,
    trauma_code_set_path: Path,
    location_map_path: Path | None = None,
) -> AnalysisArtifacts:
    trauma_code_set = load_trauma_code_set(trauma_code_set_path)
    location_map = load_location_map(location_map_path) if location_map_path else empty_location_map()
    staged_tables = {
        table_name: read_named_table(input_dir, table_name)
        for table_name in ["patient", "hospitalization", "hospital_diagnosis", "adt", "respiratory_support"]
    }
    base, respiratory_support = build_base_dataframe(
        patient=staged_tables["patient"],
        hospitalization=staged_tables["hospitalization"],
        diagnoses=staged_tables["hospital_diagnosis"],
        adt=staged_tables["adt"],
        respiratory_support=staged_tables["respiratory_support"],
        trauma_code_set=trauma_code_set,
        location_map=location_map,
    )
    cohort_ids = (
        base.loc[base["cohort_inclusion_flag"], "hospitalization_id"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    filtered_tables = {
        "patient": staged_tables["patient"],
        "hospitalization": filter_table_by_ids(staged_tables["hospitalization"], cohort_ids, "hospitalization_id"),
        "hospital_diagnosis": filter_table_by_ids(staged_tables["hospital_diagnosis"], cohort_ids, "hospitalization_id"),
        "adt": filter_table_by_ids(staged_tables["adt"], cohort_ids, "hospitalization_id"),
        "respiratory_support": filter_table_by_ids(respiratory_support, cohort_ids, "hospitalization_id"),
        "patient_assessments": read_named_table(
            input_dir,
            "patient_assessments",
            filters=[("hospitalization_id", "in", cohort_ids)],
        )
        if cohort_ids
        else empty_required_table("patient_assessments"),
    }
    patient_ids = filtered_tables["hospitalization"]["patient_id"].dropna().drop_duplicates().tolist()
    filtered_tables["patient"] = filter_table_by_ids(staged_tables["patient"], patient_ids, "patient_id")
    artifacts = build_analysis_artifacts(filtered_tables, trauma_code_set, location_map)
    return replace(artifacts, cohort_flow=build_cohort_flow(base))


def write_outputs(output_dir: Path, artifacts: AnalysisArtifacts) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    artifacts.cohort.to_csv(output_dir / "cohort.csv", index=False)
    artifacts.phase_windows.to_csv(output_dir / "phase_windows.csv", index=False)
    artifacts.interventions.to_csv(output_dir / "interventions.csv", index=False)
    artifacts.transfer_outcomes.to_csv(output_dir / "transfer_outcomes.csv", index=False)
    artifacts.assessment_context.to_csv(output_dir / "assessment_context.csv", index=False)
    artifacts.cohort_flow.to_csv(summary_dir / "cohort_flow.csv", index=False)
    artifacts.phase_intervention_summary.to_csv(summary_dir / "phase_intervention_summary.csv", index=False)
    artifacts.handoff_summary.to_csv(summary_dir / "handoff_summary.csv", index=False)
    artifacts.outcome_summary.to_csv(summary_dir / "outcome_summary.csv", index=False)
    artifacts.transfer_summary.to_csv(summary_dir / "transfer_summary.csv", index=False)


def build_analysis_artifacts(
    tables: Mapping[str, pd.DataFrame],
    trauma_code_set: pd.DataFrame,
    location_map: pd.DataFrame | None = None,
) -> AnalysisArtifacts:
    validate_tables(tables)
    trauma_code_set = normalize_trauma_code_set(trauma_code_set)
    location_map = empty_location_map() if location_map is None else normalize_location_map(location_map)

    patient = tables["patient"].copy()
    hospitalization = tables["hospitalization"].copy()
    diagnoses = tables["hospital_diagnosis"].copy()
    adt = tables["adt"].copy()
    respiratory_support = tables["respiratory_support"].copy()
    assessments = tables["patient_assessments"].copy()
    base, respiratory_support = build_base_dataframe(
        patient=patient,
        hospitalization=hospitalization,
        diagnoses=diagnoses,
        adt=adt,
        respiratory_support=respiratory_support,
        trauma_code_set=trauma_code_set,
        location_map=location_map,
    )
    assessments["recorded_dttm"] = pd.to_datetime(assessments["recorded_dttm"], errors="coerce")

    cohort = base.loc[base["cohort_inclusion_flag"]].copy()
    transfer_outcomes = build_transfer_outcomes(cohort)
    cohort = cohort.merge(
        transfer_outcomes[
            [
                "hospitalization_id",
                "transfer_outcome",
                "transfer_outcome_dttm",
                "sicu_preward_los_hours",
            ]
        ],
        on="hospitalization_id",
        how="left",
    )

    phase_windows = build_phase_windows(cohort)
    interventions = build_intervention_log(respiratory_support, cohort)
    phase_windows = attach_phase_intervention_metrics(phase_windows, interventions)
    assessment_context = build_assessment_context(assessments, phase_windows)
    phase_windows = phase_windows.merge(
        assessment_context,
        on=["hospitalization_id", "phase"],
        how="left",
    )
    handoff_summary = build_handoff_summary(respiratory_support, cohort)

    cohort_flow = build_cohort_flow(base)
    phase_intervention_summary = build_phase_intervention_summary(phase_windows, interventions)
    outcome_summary = build_outcome_summary(cohort)
    transfer_summary = build_transfer_summary(transfer_outcomes)

    cohort = cohort.sort_values(["hospitalization_id"]).reset_index(drop=True)
    phase_windows = phase_windows.sort_values(["hospitalization_id", "phase_start_dttm"]).reset_index(drop=True)
    interventions = interventions.sort_values(["hospitalization_id", "event_dttm", "variable"]).reset_index(drop=True)
    transfer_outcomes = transfer_outcomes.sort_values(["hospitalization_id"]).reset_index(drop=True)
    assessment_context = assessment_context.sort_values(["hospitalization_id", "phase"]).reset_index(drop=True)
    cohort_flow = cohort_flow.reset_index(drop=True)
    phase_intervention_summary = phase_intervention_summary.reset_index(drop=True)
    handoff_summary = handoff_summary.reset_index(drop=True)
    outcome_summary = outcome_summary.reset_index(drop=True)
    transfer_summary = transfer_summary.reset_index(drop=True)

    return AnalysisArtifacts(
        cohort=cohort,
        phase_windows=phase_windows,
        interventions=interventions,
        transfer_outcomes=transfer_outcomes,
        assessment_context=assessment_context,
        cohort_flow=cohort_flow,
        phase_intervention_summary=phase_intervention_summary,
        handoff_summary=handoff_summary,
        outcome_summary=outcome_summary,
        transfer_summary=transfer_summary,
    )


def load_required_tables(input_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for table_name in REQUIRED_COLUMNS:
        tables[table_name] = read_named_table(input_dir, table_name)
    validate_tables(tables)
    return tables


def read_named_table(
    input_dir: Path,
    table_name: str,
    filters: list[tuple[str, str, object]] | None = None,
) -> pd.DataFrame:
    required_columns = REQUIRED_COLUMNS[table_name]
    filename_roots = [table_name, f"clif_{table_name}"]
    for root in filename_roots:
        for extension in (".parquet", ".csv", ".csv.gz", ".tsv"):
            candidate = input_dir / f"{root}{extension}"
            if not candidate.exists():
                continue
            if extension == ".parquet":
                kwargs = {"columns": required_columns}
                if filters:
                    kwargs["filters"] = filters
                return pd.read_parquet(candidate, **kwargs)
            if extension == ".tsv":
                table = pd.read_csv(candidate, sep="\t", usecols=required_columns)
                return apply_filters(table, filters)
            table = pd.read_csv(candidate, usecols=required_columns)
            return apply_filters(table, filters)
    raise FileNotFoundError(f"Could not find a file for table '{table_name}' in {input_dir}.")


def load_trauma_code_set(path: Path) -> pd.DataFrame:
    trauma_code_set = pd.read_csv(path)
    return normalize_trauma_code_set(trauma_code_set)


def normalize_trauma_code_set(trauma_code_set: pd.DataFrame) -> pd.DataFrame:
    required = {"diagnosis_code_format", "prefix"}
    missing = required.difference(trauma_code_set.columns)
    if missing:
        raise ValueError(f"Trauma code set is missing columns: {sorted(missing)}")
    trauma_code_set = trauma_code_set.copy()
    trauma_code_set["diagnosis_code_format"] = trauma_code_set["diagnosis_code_format"].map(normalize_token)
    trauma_code_set["prefix"] = trauma_code_set["prefix"].map(normalize_code)
    trauma_code_set = trauma_code_set.dropna(subset=["diagnosis_code_format", "prefix"]).drop_duplicates()
    return trauma_code_set


def load_location_map(path: Path) -> pd.DataFrame:
    location_map = pd.read_csv(path)
    return normalize_location_map(location_map)


def normalize_location_map(location_map: pd.DataFrame) -> pd.DataFrame:
    required = {"match_column", "match_value"}
    missing = required.difference(location_map.columns)
    if missing:
        raise ValueError(f"Location map is missing columns: {sorted(missing)}")
    location_map = location_map.copy()
    for column in ["is_sicu", "is_ward", "is_procedural", "is_ed"]:
        if column not in location_map.columns:
            location_map[column] = pd.NA
        else:
            location_map[column] = location_map[column].map(parse_optional_bool)
    if "normalized_unit" not in location_map.columns:
        location_map["normalized_unit"] = pd.NA
    location_map["match_column"] = location_map["match_column"].map(normalize_token)
    location_map["match_value"] = location_map["match_value"].map(normalize_token)
    return location_map


def empty_location_map() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "match_column",
            "match_value",
            "normalized_unit",
            "is_sicu",
            "is_ward",
            "is_procedural",
            "is_ed",
        ]
    )


def validate_tables(tables: Mapping[str, pd.DataFrame]) -> None:
    missing_tables = sorted(set(REQUIRED_COLUMNS).difference(tables))
    if missing_tables:
        raise ValueError(f"Missing required tables: {missing_tables}")
    for table_name, required_columns in REQUIRED_COLUMNS.items():
        missing_columns = sorted(set(required_columns).difference(tables[table_name].columns))
        if missing_columns:
            raise ValueError(f"Table '{table_name}' is missing columns: {missing_columns}")


def empty_required_table(table_name: str) -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_COLUMNS[table_name])


def apply_filters(
    table: pd.DataFrame,
    filters: list[tuple[str, str, object]] | None,
) -> pd.DataFrame:
    if not filters:
        return table
    filtered = table
    for column, operator, value in filters:
        if operator == "in":
            filtered = filtered.loc[filtered[column].isin(value)]
        elif operator == "==":
            filtered = filtered.loc[filtered[column] == value]
        else:
            raise ValueError(f"Unsupported filter operator: {operator}")
    return filtered


def filter_table_by_ids(table: pd.DataFrame, ids: Iterable[object], id_column: str) -> pd.DataFrame:
    ids = list(ids)
    if not ids:
        return empty_required_like(table)
    return table.loc[table[id_column].isin(ids)].copy()


def empty_required_like(table: pd.DataFrame) -> pd.DataFrame:
    return table.iloc[0:0].copy()


def build_base_dataframe(
    patient: pd.DataFrame,
    hospitalization: pd.DataFrame,
    diagnoses: pd.DataFrame,
    adt: pd.DataFrame,
    respiratory_support: pd.DataFrame,
    trauma_code_set: pd.DataFrame,
    location_map: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trauma_code_set = normalize_trauma_code_set(trauma_code_set)
    location_map = empty_location_map() if location_map is None else normalize_location_map(location_map)

    patient = patient.copy()
    hospitalization = hospitalization.copy()
    diagnoses = diagnoses.copy()
    adt = adt.copy()
    respiratory_support = respiratory_support.copy()

    hospitalization["admission_dttm"] = pd.to_datetime(hospitalization["admission_dttm"], errors="coerce")
    hospitalization["discharge_dttm"] = pd.to_datetime(hospitalization["discharge_dttm"], errors="coerce")
    hospitalization["age_at_admission"] = pd.to_numeric(hospitalization["age_at_admission"], errors="coerce")

    adt["in_dttm"] = pd.to_datetime(adt["in_dttm"], errors="coerce")
    adt["out_dttm"] = pd.to_datetime(adt["out_dttm"], errors="coerce")
    adt = classify_adt_locations(adt, location_map)

    respiratory_support["recorded_dttm"] = pd.to_datetime(respiratory_support["recorded_dttm"], errors="coerce")

    trauma_flags = build_trauma_flags(diagnoses, trauma_code_set)
    pathway = build_pathway_table(adt)
    first_imv = build_first_imv_table(respiratory_support)

    base = hospitalization.merge(
        patient[["patient_id", "sex_category", "race_category", "ethnicity_category"]],
        on="patient_id",
        how="left",
    )
    base = base.merge(trauma_flags, on="hospitalization_id", how="left")
    base = base.merge(pathway, on="hospitalization_id", how="left")
    base = base.merge(first_imv, on="hospitalization_id", how="left")

    base["trauma_flag"] = base["trauma_flag"].fillna(False)
    base["adult_flag"] = base["age_at_admission"].ge(18)
    base["imv_in_ed_flag"] = (
        base["first_imv_dttm"].notna()
        & base["ed_in_dttm"].notna()
        & base["ed_out_dttm"].notna()
        & (base["first_imv_dttm"] >= base["ed_in_dttm"])
        & (base["first_imv_dttm"] <= base["ed_out_dttm"])
    )
    base["in_hospital_mortality"] = base["discharge_category"].map(is_death_discharge)
    base["discharge_group"] = base["discharge_category"].map(harmonize_discharge_category)
    base["hospital_los_hours"] = hours_between(base["admission_dttm"], base["discharge_dttm"])
    base["ed_los_hours"] = hours_between(base["ed_in_dttm"], base["ed_out_dttm"])
    base["cohort_inclusion_flag"] = (
        base["adult_flag"]
        & base["trauma_flag"]
        & base["pathway_valid"].fillna(False)
        & base["imv_in_ed_flag"]
    )
    return base, respiratory_support


def build_trauma_flags(diagnoses: pd.DataFrame, trauma_code_set: pd.DataFrame) -> pd.DataFrame:
    diagnoses = diagnoses.copy()
    diagnoses["diagnosis_code_format_norm"] = diagnoses["diagnosis_code_format"].map(normalize_token)
    diagnoses["diagnosis_code_norm"] = diagnoses["diagnosis_code"].map(normalize_code)
    diagnoses["poa_flag"] = diagnoses["poa_present"].map(is_truthy)
    diagnoses = diagnoses.loc[diagnoses["poa_flag"]].copy()

    if diagnoses.empty or trauma_code_set.empty:
        return pd.DataFrame({"hospitalization_id": [], "trauma_flag": []})

    diagnoses["trauma_flag"] = False
    for diagnosis_code_format, prefixes in trauma_code_set.groupby("diagnosis_code_format")["prefix"]:
        escaped_prefixes = [re.escape(prefix) for prefix in prefixes if prefix]
        if not escaped_prefixes:
            continue
        pattern = r"^(?:" + "|".join(sorted(escaped_prefixes, key=len, reverse=True)) + r")"
        mask = diagnoses["diagnosis_code_format_norm"].eq(diagnosis_code_format)
        diagnoses.loc[mask, "trauma_flag"] = diagnoses.loc[mask, "diagnosis_code_norm"].str.match(pattern, na=False)

    trauma_flags = (
        diagnoses.groupby("hospitalization_id", as_index=False)["trauma_flag"]
        .max()
    )
    return trauma_flags


def classify_adt_locations(adt: pd.DataFrame, location_map: pd.DataFrame) -> pd.DataFrame:
    adt = adt.copy()
    for column in ["location_name", "location_category", "location_type"]:
        adt[f"{column}_norm"] = adt[column].map(normalize_token)

    adt["is_ed"] = (
        adt["location_category_norm"].isin({"ed", "emergency department", "emergency"})
        | adt["location_type_norm"].str.contains(r"\bed\b|emergency", na=False)
        | adt["location_name_norm"].str.contains(r"\bed\b|emergency", na=False)
    )
    adt["is_icu"] = (
        adt["location_category_norm"].eq("icu")
        | adt["location_type_norm"].str.contains("icu", na=False)
        | adt["location_name_norm"].str.contains(r"\bicu\b", na=False)
    )
    adt["is_sicu"] = adt["is_icu"] & (
        adt["location_type_norm"].str.contains("sicu|surgical", na=False)
        | adt["location_name_norm"].str.contains("sicu|surgical", na=False)
    )
    adt["is_procedural"] = (
        adt["location_category_norm"].isin({"or", "procedure", "procedural", "operating room", "pacu"})
        | adt["location_type_norm"].str.contains(r"\bor\b|procedure|operating|periop|pacu", na=False)
        | adt["location_name_norm"].str.contains(r"\bor\b|procedure|operating|periop|pacu", na=False)
    )
    adt["is_ward"] = (
        adt["location_category_norm"].isin({"ward", "floor", "inpatient ward", "inpatient floor"})
        | adt["location_type_norm"].str.contains("ward|floor", na=False)
        | adt["location_name_norm"].str.contains("ward|floor", na=False)
    ) & ~adt["is_icu"]

    adt["normalized_unit"] = pd.Series(pd.NA, index=adt.index, dtype="object")
    adt.loc[adt["is_ed"], "normalized_unit"] = "ED"
    adt.loc[adt["is_sicu"], "normalized_unit"] = "SICU"
    adt.loc[adt["is_procedural"], "normalized_unit"] = "PROCEDURAL"
    adt.loc[adt["is_ward"], "normalized_unit"] = "WARD"

    for row in location_map.itertuples(index=False):
        match_column = getattr(row, "match_column")
        match_value = getattr(row, "match_value")
        norm_column = f"{match_column}_norm"
        if norm_column not in adt.columns:
            raise ValueError(f"Location map match_column '{match_column}' is not supported.")
        mask = adt[norm_column] == match_value
        for column in ["is_sicu", "is_ward", "is_procedural", "is_ed"]:
            value = getattr(row, column)
            if pd.notna(value):
                adt.loc[mask, column] = bool(value)
        if pd.notna(getattr(row, "normalized_unit")):
            adt.loc[mask, "normalized_unit"] = getattr(row, "normalized_unit")

    adt["is_icu"] = adt["is_icu"] | adt["is_sicu"]
    adt.loc[adt["is_sicu"], "normalized_unit"] = "SICU"
    adt.loc[adt["is_procedural"], "normalized_unit"] = "PROCEDURAL"
    adt.loc[adt["is_ward"], "normalized_unit"] = "WARD"
    adt.loc[adt["is_ed"], "normalized_unit"] = "ED"
    return adt


def build_pathway_table(adt: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    ordered = adt.sort_values(["hospitalization_id", "in_dttm", "out_dttm", "location_name"], kind="stable")
    for hospitalization_id, group in ordered.groupby("hospitalization_id", sort=False):
        rows = group.reset_index(drop=True)
        record: dict[str, object] = {
            "hospitalization_id": hospitalization_id,
            "pathway_valid": False,
            "pathway_reason": "missing_adt",
            "ed_in_dttm": pd.NaT,
            "ed_out_dttm": pd.NaT,
            "sicu_in_dttm": pd.NaT,
            "initial_sicu_exit_dttm": pd.NaT,
            "has_procedural_bridge": False,
            "next_after_sicu_in_dttm": pd.NaT,
            "next_after_sicu_location_name": pd.NA,
            "next_after_sicu_location_type": pd.NA,
            "next_after_sicu_location_category": pd.NA,
            "next_after_sicu_is_ward": False,
            "next_after_sicu_is_icu": False,
            "next_after_sicu_is_procedural": False,
        }

        ed_positions = rows.index[rows["is_ed"]].tolist()
        if not ed_positions:
            record["pathway_reason"] = "no_ed_segment"
            records.append(record)
            continue
        ed_start = ed_positions[0]
        ed_end = ed_start
        while ed_end + 1 < len(rows) and bool(rows.loc[ed_end + 1, "is_ed"]):
            ed_end += 1

        record["ed_in_dttm"] = rows.loc[ed_start, "in_dttm"]
        record["ed_out_dttm"] = coalesce_segment_end(rows, ed_end)

        sicu_positions = [
            position
            for position in range(ed_end + 1, len(rows))
            if bool(rows.loc[position, "is_sicu"])
        ]
        if not sicu_positions:
            record["pathway_reason"] = "no_sicu_segment"
            records.append(record)
            continue
        sicu_start = sicu_positions[0]
        intermediate = rows.iloc[ed_end + 1 : sicu_start]
        if not intermediate.empty and not intermediate["is_procedural"].fillna(False).all():
            record["pathway_reason"] = "non_procedural_intermediate_location"
            records.append(record)
            continue

        sicu_end = sicu_start
        while sicu_end + 1 < len(rows) and bool(rows.loc[sicu_end + 1, "is_sicu"]):
            sicu_end += 1

        record["pathway_valid"] = True
        record["pathway_reason"] = "included"
        record["has_procedural_bridge"] = bool(intermediate["is_procedural"].fillna(False).any())
        record["sicu_in_dttm"] = rows.loc[sicu_start, "in_dttm"]
        record["initial_sicu_exit_dttm"] = coalesce_segment_end(rows, sicu_end)

        if sicu_end + 1 < len(rows):
            next_row = rows.loc[sicu_end + 1]
            record["next_after_sicu_in_dttm"] = next_row["in_dttm"]
            record["next_after_sicu_location_name"] = next_row["location_name"]
            record["next_after_sicu_location_type"] = next_row["location_type"]
            record["next_after_sicu_location_category"] = next_row["location_category"]
            record["next_after_sicu_is_ward"] = bool(next_row["is_ward"])
            record["next_after_sicu_is_icu"] = bool(next_row["is_icu"]) and not bool(next_row["is_sicu"])
            record["next_after_sicu_is_procedural"] = bool(next_row["is_procedural"])

        records.append(record)

    return pd.DataFrame.from_records(
        records,
        columns=[
            "hospitalization_id",
            "pathway_valid",
            "pathway_reason",
            "ed_in_dttm",
            "ed_out_dttm",
            "sicu_in_dttm",
            "initial_sicu_exit_dttm",
            "has_procedural_bridge",
            "next_after_sicu_in_dttm",
            "next_after_sicu_location_name",
            "next_after_sicu_location_type",
            "next_after_sicu_location_category",
            "next_after_sicu_is_ward",
            "next_after_sicu_is_icu",
            "next_after_sicu_is_procedural",
        ],
    )


def build_first_imv_table(respiratory_support: pd.DataFrame) -> pd.DataFrame:
    respiratory_support = respiratory_support.copy()
    respiratory_support["device_category_norm"] = respiratory_support["device_category"].map(normalize_token)
    respiratory_support["is_imv"] = respiratory_support["device_category_norm"].map(is_imv_device)
    first_imv = (
        respiratory_support.loc[respiratory_support["is_imv"]]
        .sort_values(["hospitalization_id", "recorded_dttm"], kind="stable")
        .groupby("hospitalization_id", as_index=False)["recorded_dttm"]
        .first()
        .rename(columns={"recorded_dttm": "first_imv_dttm"})
    )
    return first_imv


def build_transfer_outcomes(cohort: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in cohort.itertuples(index=False):
        outcome = "other"
        outcome_dttm = row.next_after_sicu_in_dttm
        if pd.notna(row.next_after_sicu_in_dttm):
            if row.next_after_sicu_is_ward:
                outcome = "ward"
            elif row.next_after_sicu_is_icu:
                outcome = "another_icu"
            elif row.next_after_sicu_is_procedural:
                outcome = "procedural"
            else:
                outcome = "other"
        else:
            if row.in_hospital_mortality:
                outcome = "death"
                outcome_dttm = row.discharge_dttm
            elif pd.notna(row.discharge_dttm):
                outcome = "discharge"
                outcome_dttm = row.discharge_dttm

        sicu_preward_los_hours = pd.NA
        if outcome == "ward" and pd.notna(row.sicu_in_dttm) and pd.notna(outcome_dttm):
            sicu_preward_los_hours = (outcome_dttm - row.sicu_in_dttm).total_seconds() / 3600.0

        records.append(
            {
                "hospitalization_id": row.hospitalization_id,
                "transfer_outcome": outcome,
                "transfer_outcome_dttm": outcome_dttm,
                "sicu_preward_los_hours": sicu_preward_los_hours,
                "next_location_name": row.next_after_sicu_location_name,
                "next_location_type": row.next_after_sicu_location_type,
                "next_location_category": row.next_after_sicu_location_category,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "hospitalization_id",
            "transfer_outcome",
            "transfer_outcome_dttm",
            "sicu_preward_los_hours",
            "next_location_name",
            "next_location_type",
            "next_location_category",
        ],
    )


def build_phase_windows(cohort: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in cohort.itertuples(index=False):
        ed_start = row.first_imv_dttm
        ed_end = row.ed_out_dttm
        if pd.notna(ed_start) and pd.notna(ed_end) and ed_end >= ed_start:
            records.append(
                phase_record(
                    hospitalization_id=row.hospitalization_id,
                    phase="ED",
                    phase_start_dttm=ed_start,
                    phase_end_dttm=ed_end,
                )
            )

        sicu_start = row.sicu_in_dttm
        sicu_end_candidates = [value for value in [row.discharge_dttm, row.next_after_sicu_in_dttm] if pd.notna(value)]
        if pd.notna(sicu_start):
            sicu_limit = sicu_start + pd.Timedelta(hours=24)
            if sicu_end_candidates:
                sicu_end = min(sicu_end_candidates + [sicu_limit])
            else:
                sicu_end = sicu_limit
            if pd.notna(sicu_end) and sicu_end > sicu_start:
                records.append(
                    phase_record(
                        hospitalization_id=row.hospitalization_id,
                        phase="SICU_24h",
                        phase_start_dttm=sicu_start,
                        phase_end_dttm=sicu_end,
                    )
                )

    return pd.DataFrame.from_records(
        records,
        columns=[
            "hospitalization_id",
            "phase",
            "phase_start_dttm",
            "phase_end_dttm",
            "phase_duration_hours",
        ],
    )


def build_intervention_log(respiratory_support: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    if cohort.empty:
        return pd.DataFrame(
            columns=[
                "hospitalization_id",
                "event_dttm",
                "phase",
                "variable",
                "old_value",
                "new_value",
            ]
        )

    cohort_windows = cohort.set_index("hospitalization_id")[
        ["first_imv_dttm", "ed_out_dttm", "sicu_in_dttm", "next_after_sicu_in_dttm", "discharge_dttm"]
    ]
    respiratory_support = collapse_respiratory_timestamps(respiratory_support)
    respiratory_support["device_category_norm"] = respiratory_support["device_category"].map(normalize_token)
    respiratory_support["is_imv"] = respiratory_support["device_category_norm"].map(is_imv_device)

    events: list[dict[str, object]] = []
    for hospitalization_id, group in respiratory_support.groupby("hospitalization_id", sort=False):
        if hospitalization_id not in cohort_windows.index:
            continue
        window = cohort_windows.loc[hospitalization_id]
        group = group.sort_values("recorded_dttm", kind="stable").reset_index(drop=True)
        if group.empty:
            continue

        imv_positions = group.index[group["is_imv"]].tolist()
        if not imv_positions:
            continue
        first_imv_position = imv_positions[0]
        analysis_stop = calculate_analysis_stop(window)
        group = group.loc[first_imv_position:].copy()
        extubation_positions = group.index[(~group["is_imv"]) & (group["recorded_dttm"] > window["first_imv_dttm"])].tolist()
        if extubation_positions:
            first_extubation_time = group.loc[extubation_positions[0], "recorded_dttm"]
            if pd.notna(first_extubation_time):
                analysis_stop = min(
                    [value for value in [analysis_stop, first_extubation_time - pd.Timedelta(microseconds=1)] if pd.notna(value)]
                )
        if pd.notna(analysis_stop):
            group = group.loc[group["recorded_dttm"] <= analysis_stop].copy()

        last_seen: dict[str, object] = {}
        for row in group.itertuples(index=False):
            event_phase = determine_phase(row.recorded_dttm, window)
            for variable in INTERVENTION_COLUMNS:
                current_value = getattr(row, variable)
                if pd.isna(current_value):
                    continue
                previous_value = last_seen.get(variable)
                if previous_value is not None and not values_equal(previous_value, current_value):
                    if event_phase is not None:
                        events.append(
                            {
                                "hospitalization_id": hospitalization_id,
                                "event_dttm": row.recorded_dttm,
                                "phase": event_phase,
                                "variable": variable,
                                "old_value": previous_value,
                                "new_value": current_value,
                            }
                        )
                last_seen[variable] = current_value

    return pd.DataFrame.from_records(
        events,
        columns=[
            "hospitalization_id",
            "event_dttm",
            "phase",
            "variable",
            "old_value",
            "new_value",
        ],
    )


def attach_phase_intervention_metrics(
    phase_windows: pd.DataFrame,
    interventions: pd.DataFrame,
) -> pd.DataFrame:
    phase_windows = phase_windows.copy()
    if phase_windows.empty:
        return phase_windows

    counts = (
        interventions.groupby(["hospitalization_id", "phase"], as_index=False)
        .size()
        .rename(columns={"size": "intervention_count"})
    )
    phase_windows = phase_windows.merge(counts, on=["hospitalization_id", "phase"], how="left")
    phase_windows["intervention_count"] = phase_windows["intervention_count"].fillna(0).astype(int)
    phase_windows["any_intervention"] = phase_windows["intervention_count"].gt(0)
    phase_windows["interventions_per_vent_hour"] = phase_windows["intervention_count"] / phase_windows["phase_duration_hours"].replace(0, pd.NA)
    return phase_windows


def build_assessment_context(assessments: pd.DataFrame, phase_windows: pd.DataFrame) -> pd.DataFrame:
    if phase_windows.empty:
        return pd.DataFrame(
            columns=[
                "hospitalization_id",
                "phase",
                "assessment_count",
                "sedation_assessment_count",
                "neurologic_assessment_count",
                "readiness_assessment_count",
            ]
        )

    assessments = assessments.copy()
    assessments["assessment_label"] = (
        assessments["assessment_category"].fillna("").astype(str)
        + " "
        + assessments["assessment_group"].fillna("").astype(str)
    ).map(normalize_token)
    assessments["sedation_flag"] = assessments["assessment_label"].str.contains("rass|sedat|agitat|sas", na=False)
    assessments["neurologic_flag"] = assessments["assessment_label"].str.contains("gcs|glasgow|neuro|cam|delirium|pupil", na=False)
    assessments["readiness_flag"] = assessments["assessment_label"].str.contains("sat|sbt|spontaneous awakening|spontaneous breathing", na=False)

    records: list[dict[str, object]] = []
    for phase in phase_windows.itertuples(index=False):
        phase_assessments = assessments.loc[
            (assessments["hospitalization_id"] == phase.hospitalization_id)
            & (assessments["recorded_dttm"] >= phase.phase_start_dttm)
            & (assessments["recorded_dttm"] <= phase.phase_end_dttm)
        ]
        records.append(
            {
                "hospitalization_id": phase.hospitalization_id,
                "phase": phase.phase,
                "assessment_count": int(len(phase_assessments)),
                "sedation_assessment_count": int(phase_assessments["sedation_flag"].sum()),
                "neurologic_assessment_count": int(phase_assessments["neurologic_flag"].sum()),
                "readiness_assessment_count": int(phase_assessments["readiness_flag"].sum()),
            }
        )

    return pd.DataFrame.from_records(records)


def build_handoff_summary(respiratory_support: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    if cohort.empty:
        return pd.DataFrame(
            columns=["variable", "paired_hospitalizations", "changed_hospitalizations", "pct_changed", "median_abs_delta"]
        )

    respiratory_support = collapse_respiratory_timestamps(respiratory_support)
    respiratory_support["device_category_norm"] = respiratory_support["device_category"].map(normalize_token)
    respiratory_support["is_imv"] = respiratory_support["device_category_norm"].map(is_imv_device)
    paired_rows: list[dict[str, object]] = []

    for row in cohort.itertuples(index=False):
        group = respiratory_support.loc[respiratory_support["hospitalization_id"] == row.hospitalization_id].copy()
        if group.empty:
            continue
        group = group.sort_values("recorded_dttm", kind="stable")
        ed_row = group.loc[
            group["is_imv"]
            & (group["recorded_dttm"] >= row.first_imv_dttm)
            & (group["recorded_dttm"] <= row.ed_out_dttm)
        ].tail(1)
        sicu_row = group.loc[
            group["is_imv"]
            & (group["recorded_dttm"] >= row.sicu_in_dttm)
            & (
                pd.isna(row.next_after_sicu_in_dttm)
                | (group["recorded_dttm"] < row.next_after_sicu_in_dttm)
            )
        ].head(1)
        if ed_row.empty or sicu_row.empty:
            continue
        ed_values = ed_row.iloc[0]
        sicu_values = sicu_row.iloc[0]
        for variable in INTERVENTION_COLUMNS:
            ed_value = ed_values[variable]
            sicu_value = sicu_values[variable]
            if pd.isna(ed_value) or pd.isna(sicu_value):
                continue
            abs_delta = pd.NA
            if is_number(ed_value) and is_number(sicu_value):
                abs_delta = abs(float(sicu_value) - float(ed_value))
            paired_rows.append(
                {
                    "variable": variable,
                    "changed": not values_equal(ed_value, sicu_value),
                    "abs_delta": abs_delta,
                }
            )

    paired = pd.DataFrame.from_records(paired_rows)
    if paired.empty:
        return pd.DataFrame(
            columns=["variable", "paired_hospitalizations", "changed_hospitalizations", "pct_changed", "median_abs_delta"]
        )

    summary = (
        paired.groupby("variable", as_index=False)
        .agg(
            paired_hospitalizations=("changed", "size"),
            changed_hospitalizations=("changed", "sum"),
            pct_changed=("changed", "mean"),
            median_abs_delta=("abs_delta", "median"),
        )
    )
    return summary


def build_cohort_flow(base: pd.DataFrame) -> pd.DataFrame:
    sequential_masks = [
        ("hospitalizations_total", pd.Series(True, index=base.index)),
        ("adult_hospitalizations", base["adult_flag"]),
        ("adult_trauma_hospitalizations", base["adult_flag"] & base["trauma_flag"]),
        (
            "adult_trauma_with_ed",
            base["adult_flag"] & base["trauma_flag"] & base["ed_in_dttm"].notna(),
        ),
        (
            "adult_trauma_with_valid_pathway",
            base["adult_flag"] & base["trauma_flag"] & base["pathway_valid"].fillna(False),
        ),
        (
            "adult_trauma_with_imv_in_ed",
            base["adult_flag"] & base["trauma_flag"] & base["pathway_valid"].fillna(False) & base["imv_in_ed_flag"],
        ),
        ("final_cohort", base["cohort_inclusion_flag"]),
    ]
    return pd.DataFrame(
        [{"stage": stage, "count": int(mask.fillna(False).sum())} for stage, mask in sequential_masks]
    )


def build_phase_intervention_summary(
    phase_windows: pd.DataFrame,
    interventions: pd.DataFrame,
) -> pd.DataFrame:
    if phase_windows.empty:
        return pd.DataFrame(
            columns=[
                "phase",
                "hospitalizations",
                "hospitalizations_with_any_intervention",
                "total_interventions",
                "median_interventions_per_hospitalization",
                "interventions_per_vent_hour",
            ]
        )

    total_events = (
        interventions.groupby("phase", as_index=False)
        .size()
        .rename(columns={"size": "total_interventions"})
    )
    summary = (
        phase_windows.groupby("phase", as_index=False)
        .agg(
            hospitalizations=("hospitalization_id", "size"),
            hospitalizations_with_any_intervention=("any_intervention", "sum"),
            median_interventions_per_hospitalization=("intervention_count", "median"),
            total_vent_hours=("phase_duration_hours", "sum"),
        )
        .merge(total_events, on="phase", how="left")
    )
    summary["total_interventions"] = summary["total_interventions"].fillna(0).astype(int)
    summary["interventions_per_vent_hour"] = summary["total_interventions"] / summary["total_vent_hours"].replace(0, pd.NA)
    summary["phase"] = pd.Categorical(summary["phase"], categories=PHASE_ORDER, ordered=True)
    return summary.drop(columns=["total_vent_hours"]).sort_values("phase").reset_index(drop=True)


def build_outcome_summary(cohort: pd.DataFrame) -> pd.DataFrame:
    if cohort.empty:
        return pd.DataFrame(columns=["metric", "value"])

    metrics = [
        ("cohort_hospitalizations", len(cohort)),
        ("in_hospital_mortality_rate", cohort["in_hospital_mortality"].mean()),
        ("median_hospital_los_hours", cohort["hospital_los_hours"].median()),
        ("median_ed_los_hours", cohort["ed_los_hours"].median()),
        ("median_sicu_preward_los_hours", cohort["sicu_preward_los_hours"].dropna().median()),
    ]
    return pd.DataFrame([{"metric": metric, "value": value} for metric, value in metrics])


def build_transfer_summary(transfer_outcomes: pd.DataFrame) -> pd.DataFrame:
    if transfer_outcomes.empty:
        return pd.DataFrame(columns=["transfer_outcome", "count", "share"])
    summary = (
        transfer_outcomes.groupby("transfer_outcome", as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    summary["share"] = summary["count"] / summary["count"].sum()
    return summary


def collapse_respiratory_timestamps(respiratory_support: pd.DataFrame) -> pd.DataFrame:
    respiratory_support = respiratory_support.copy()
    grouped = (
        respiratory_support.sort_values(["hospitalization_id", "recorded_dttm"], kind="stable")
        .groupby(["hospitalization_id", "recorded_dttm"], as_index=False)
        .agg({column: first_non_null for column in respiratory_support.columns if column not in {"hospitalization_id", "recorded_dttm"}})
    )
    return grouped


def determine_phase(event_dttm: pd.Timestamp, window: pd.Series) -> str | None:
    if pd.isna(event_dttm):
        return None
    if pd.notna(window["first_imv_dttm"]) and pd.notna(window["ed_out_dttm"]):
        if window["first_imv_dttm"] <= event_dttm <= window["ed_out_dttm"]:
            return "ED"
    if pd.notna(window["sicu_in_dttm"]):
        sicu_phase_end = calculate_analysis_stop(window)
        if pd.notna(sicu_phase_end) and window["sicu_in_dttm"] <= event_dttm <= sicu_phase_end:
            return "SICU_24h"
    return None


def calculate_analysis_stop(window: pd.Series) -> pd.Timestamp:
    if pd.isna(window["sicu_in_dttm"]):
        return window["ed_out_dttm"]
    candidates = [window["sicu_in_dttm"] + pd.Timedelta(hours=24)]
    for column in ["next_after_sicu_in_dttm", "discharge_dttm"]:
        if pd.notna(window[column]):
            candidates.append(window[column])
    return min(candidates)


def phase_record(
    hospitalization_id: object,
    phase: str,
    phase_start_dttm: pd.Timestamp,
    phase_end_dttm: pd.Timestamp,
) -> dict[str, object]:
    return {
        "hospitalization_id": hospitalization_id,
        "phase": phase,
        "phase_start_dttm": phase_start_dttm,
        "phase_end_dttm": phase_end_dttm,
        "phase_duration_hours": (phase_end_dttm - phase_start_dttm).total_seconds() / 3600.0,
    }


def coalesce_segment_end(rows: pd.DataFrame, position: int) -> pd.Timestamp:
    current_out = rows.loc[position, "out_dttm"]
    if pd.notna(current_out):
        return current_out
    if position + 1 < len(rows):
        return rows.loc[position + 1, "in_dttm"]
    return pd.NaT


def hours_between(start: pd.Series, end: pd.Series) -> pd.Series:
    delta = end - start
    return delta.dt.total_seconds() / 3600.0


def is_death_discharge(value: object) -> bool:
    token = normalize_token(value)
    return bool(re.search(r"death|expire|expired|deceased|morgue", token))


def harmonize_discharge_category(value: object) -> str:
    token = normalize_token(value)
    if re.search(r"death|expire|expired|deceased|morgue", token):
        return "death"
    if "hospice" in token:
        return "hospice"
    if "home" in token:
        return "home"
    if "rehab" in token:
        return "rehab"
    if re.search(r"ltac|snf|nursing|skilled", token):
        return "post_acute"
    if re.search(r"acute|transfer|hospital", token):
        return "acute_care"
    if token:
        return "other"
    return "unknown"


def normalize_code(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", str(value).upper())


def normalize_token(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def is_truthy(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    token = normalize_token(value)
    return token in {"1", "true", "t", "y", "yes", "present"}


def parse_optional_bool(value: object) -> bool | pd.NA:
    if pd.isna(value):
        return pd.NA
    token = normalize_token(value)
    if token in {"0", "false", "f", "n", "no", "absent"}:
        return False
    return is_truthy(value)


def is_imv_device(value: object) -> bool:
    token = normalize_token(value)
    return bool(re.search(r"\bimv\b|invasive mechanical ventilation|ventilator", token))


def first_non_null(series: pd.Series) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[0]


def values_equal(left: object, right: object) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if is_number(left) and is_number(right):
        return float(left) == float(right)
    return str(left) == str(right)


def is_number(value: object) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
