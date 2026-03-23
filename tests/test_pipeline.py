from __future__ import annotations

import unittest

import pandas as pd

from clif_trauma.pipeline import build_analysis_artifacts


def make_tables() -> dict[str, pd.DataFrame]:
    patient = pd.DataFrame(
        [
            {
                "patient_id": 1,
                "sex_category": "male",
                "race_category": "white",
                "ethnicity_category": "not hispanic",
            },
            {
                "patient_id": 2,
                "sex_category": "female",
                "race_category": "black",
                "ethnicity_category": "not hispanic",
            },
            {
                "patient_id": 3,
                "sex_category": "male",
                "race_category": "white",
                "ethnicity_category": "not hispanic",
            },
            {
                "patient_id": 4,
                "sex_category": "female",
                "race_category": "white",
                "ethnicity_category": "not hispanic",
            },
        ]
    )

    hospitalization = pd.DataFrame(
        [
            {
                "patient_id": 1,
                "hospitalization_id": 100,
                "admission_dttm": "2026-01-01 08:00:00",
                "discharge_dttm": "2026-01-05 11:00:00",
                "age_at_admission": 35,
                "admission_type_name": "ED",
                "admission_type_category": "emergency",
                "discharge_category": "home",
            },
            {
                "patient_id": 2,
                "hospitalization_id": 200,
                "admission_dttm": "2026-01-02 08:00:00",
                "discharge_dttm": "2026-01-06 10:00:00",
                "age_at_admission": 46,
                "admission_type_name": "ED",
                "admission_type_category": "emergency",
                "discharge_category": "expired",
            },
            {
                "patient_id": 3,
                "hospitalization_id": 300,
                "admission_dttm": "2026-01-03 08:00:00",
                "discharge_dttm": "2026-01-04 10:00:00",
                "age_at_admission": 52,
                "admission_type_name": "ED",
                "admission_type_category": "emergency",
                "discharge_category": "acute care hospital",
            },
            {
                "patient_id": 4,
                "hospitalization_id": 400,
                "admission_dttm": "2026-01-04 08:00:00",
                "discharge_dttm": "2026-01-08 10:00:00",
                "age_at_admission": 41,
                "admission_type_name": "ED",
                "admission_type_category": "emergency",
                "discharge_category": "rehab",
            },
        ]
    )

    hospital_diagnosis = pd.DataFrame(
        [
            {
                "hospitalization_id": 100,
                "diagnosis_code": "S06.5",
                "diagnosis_code_format": "ICD10CM",
                "diagnosis_primary": True,
                "poa_present": 1,
            },
            {
                "hospitalization_id": 200,
                "diagnosis_code": "S22.0",
                "diagnosis_code_format": "ICD10CM",
                "diagnosis_primary": True,
                "poa_present": 1,
            },
            {
                "hospitalization_id": 300,
                "diagnosis_code": "S27.0",
                "diagnosis_code_format": "ICD10CM",
                "diagnosis_primary": True,
                "poa_present": 1,
            },
            {
                "hospitalization_id": 400,
                "diagnosis_code": "S36.1",
                "diagnosis_code_format": "ICD10CM",
                "diagnosis_primary": True,
                "poa_present": 1,
            },
        ]
    )

    adt = pd.DataFrame(
        [
            {
                "hospitalization_id": 100,
                "in_dttm": "2026-01-01 08:00:00",
                "out_dttm": "2026-01-01 11:00:00",
                "location_name": "Emergency Department",
                "location_category": "ed",
                "location_type": "ed",
            },
            {
                "hospitalization_id": 100,
                "in_dttm": "2026-01-01 11:00:00",
                "out_dttm": "2026-01-02 16:00:00",
                "location_name": "Surgical ICU",
                "location_category": "icu",
                "location_type": "sicu",
            },
            {
                "hospitalization_id": 100,
                "in_dttm": "2026-01-02 16:00:00",
                "out_dttm": "2026-01-05 11:00:00",
                "location_name": "Trauma Ward",
                "location_category": "ward",
                "location_type": "ward",
            },
            {
                "hospitalization_id": 200,
                "in_dttm": "2026-01-02 08:00:00",
                "out_dttm": "2026-01-02 10:00:00",
                "location_name": "Emergency Department",
                "location_category": "ed",
                "location_type": "ed",
            },
            {
                "hospitalization_id": 200,
                "in_dttm": "2026-01-02 10:00:00",
                "out_dttm": "2026-01-02 12:00:00",
                "location_name": "Operating Room",
                "location_category": "or",
                "location_type": "or",
            },
            {
                "hospitalization_id": 200,
                "in_dttm": "2026-01-02 12:00:00",
                "out_dttm": "2026-01-04 09:00:00",
                "location_name": "Surgical ICU",
                "location_category": "icu",
                "location_type": "sicu",
            },
            {
                "hospitalization_id": 300,
                "in_dttm": "2026-01-03 08:00:00",
                "out_dttm": "2026-01-03 10:00:00",
                "location_name": "Emergency Department",
                "location_category": "ed",
                "location_type": "ed",
            },
            {
                "hospitalization_id": 300,
                "in_dttm": "2026-01-03 10:00:00",
                "out_dttm": "2026-01-03 12:00:00",
                "location_name": "Medical Ward",
                "location_category": "ward",
                "location_type": "ward",
            },
            {
                "hospitalization_id": 300,
                "in_dttm": "2026-01-03 12:00:00",
                "out_dttm": "2026-01-04 10:00:00",
                "location_name": "Surgical ICU",
                "location_category": "icu",
                "location_type": "sicu",
            },
            {
                "hospitalization_id": 400,
                "in_dttm": "2026-01-04 08:00:00",
                "out_dttm": "2026-01-04 11:00:00",
                "location_name": "Emergency Department",
                "location_category": "ed",
                "location_type": "ed",
            },
            {
                "hospitalization_id": 400,
                "in_dttm": "2026-01-04 11:00:00",
                "out_dttm": "2026-01-05 08:00:00",
                "location_name": "Surgical ICU",
                "location_category": "icu",
                "location_type": "sicu",
            },
            {
                "hospitalization_id": 400,
                "in_dttm": "2026-01-05 08:00:00",
                "out_dttm": "2026-01-08 10:00:00",
                "location_name": "Cardiac ICU",
                "location_category": "icu",
                "location_type": "icu",
            },
        ]
    )

    respiratory_support = pd.DataFrame(
        [
            {
                "hospitalization_id": 100,
                "recorded_dttm": "2026-01-01 09:00:00",
                "device_category": "IMV",
                "mode_category": "VC",
                "tracheostomy": False,
                "fio2_set": 40,
                "tidal_volume_set": 450,
                "resp_rate_set": 16,
                "pressure_control_set": pd.NA,
                "pressure_support_set": pd.NA,
                "peep_set": 5,
                "tidal_volume_obs": 440,
                "resp_rate_obs": 16,
                "plateau_pressure_obs": 18,
                "peak_inspiratory_pressure_obs": 22,
                "peep_obs": 5,
                "minute_vent_obs": 8.0,
                "mean_airway_pressure_obs": 10,
            },
            {
                "hospitalization_id": 100,
                "recorded_dttm": "2026-01-01 10:00:00",
                "device_category": "IMV",
                "mode_category": "VC",
                "tracheostomy": False,
                "fio2_set": 50,
                "tidal_volume_set": 450,
                "resp_rate_set": 16,
                "pressure_control_set": pd.NA,
                "pressure_support_set": pd.NA,
                "peep_set": 5,
                "tidal_volume_obs": 440,
                "resp_rate_obs": 16,
                "plateau_pressure_obs": 18,
                "peak_inspiratory_pressure_obs": 22,
                "peep_obs": 5,
                "minute_vent_obs": 8.0,
                "mean_airway_pressure_obs": 10,
            },
            {
                "hospitalization_id": 100,
                "recorded_dttm": "2026-01-01 10:00:00",
                "device_category": "IMV",
                "mode_category": "VC",
                "tracheostomy": False,
                "fio2_set": 50,
                "tidal_volume_set": 450,
                "resp_rate_set": 16,
                "pressure_control_set": pd.NA,
                "pressure_support_set": pd.NA,
                "peep_set": 5,
                "tidal_volume_obs": 440,
                "resp_rate_obs": 16,
                "plateau_pressure_obs": 18,
                "peak_inspiratory_pressure_obs": 22,
                "peep_obs": 5,
                "minute_vent_obs": 8.0,
                "mean_airway_pressure_obs": 10,
            },
            {
                "hospitalization_id": 100,
                "recorded_dttm": "2026-01-01 12:00:00",
                "device_category": "IMV",
                "mode_category": "VC",
                "tracheostomy": False,
                "fio2_set": 45,
                "tidal_volume_set": 450,
                "resp_rate_set": 14,
                "pressure_control_set": pd.NA,
                "pressure_support_set": pd.NA,
                "peep_set": 8,
                "tidal_volume_obs": 440,
                "resp_rate_obs": 14,
                "plateau_pressure_obs": 18,
                "peak_inspiratory_pressure_obs": 22,
                "peep_obs": 8,
                "minute_vent_obs": 8.0,
                "mean_airway_pressure_obs": 10,
            },
            {
                "hospitalization_id": 200,
                "recorded_dttm": "2026-01-02 09:00:00",
                "device_category": "IMV",
                "mode_category": "VC",
                "tracheostomy": False,
                "fio2_set": 60,
                "tidal_volume_set": 500,
                "resp_rate_set": 18,
                "pressure_control_set": pd.NA,
                "pressure_support_set": pd.NA,
                "peep_set": 8,
                "tidal_volume_obs": 490,
                "resp_rate_obs": 18,
                "plateau_pressure_obs": 20,
                "peak_inspiratory_pressure_obs": 24,
                "peep_obs": 8,
                "minute_vent_obs": 9.0,
                "mean_airway_pressure_obs": 11,
            },
            {
                "hospitalization_id": 200,
                "recorded_dttm": "2026-01-02 11:00:00",
                "device_category": "IMV",
                "mode_category": "PC",
                "tracheostomy": False,
                "fio2_set": 50,
                "tidal_volume_set": 500,
                "resp_rate_set": 18,
                "pressure_control_set": 18,
                "pressure_support_set": pd.NA,
                "peep_set": 10,
                "tidal_volume_obs": 490,
                "resp_rate_obs": 18,
                "plateau_pressure_obs": 20,
                "peak_inspiratory_pressure_obs": 24,
                "peep_obs": 10,
                "minute_vent_obs": 9.0,
                "mean_airway_pressure_obs": 11,
            },
            {
                "hospitalization_id": 200,
                "recorded_dttm": "2026-01-02 13:00:00",
                "device_category": "IMV",
                "mode_category": "PC",
                "tracheostomy": False,
                "fio2_set": 45,
                "tidal_volume_set": 500,
                "resp_rate_set": 16,
                "pressure_control_set": 16,
                "pressure_support_set": pd.NA,
                "peep_set": 10,
                "tidal_volume_obs": 490,
                "resp_rate_obs": 16,
                "plateau_pressure_obs": 20,
                "peak_inspiratory_pressure_obs": 24,
                "peep_obs": 10,
                "minute_vent_obs": 9.0,
                "mean_airway_pressure_obs": 11,
            },
            {
                "hospitalization_id": 300,
                "recorded_dttm": "2026-01-03 09:00:00",
                "device_category": "IMV",
                "mode_category": "VC",
                "tracheostomy": False,
                "fio2_set": 50,
                "tidal_volume_set": 450,
                "resp_rate_set": 16,
                "pressure_control_set": pd.NA,
                "pressure_support_set": pd.NA,
                "peep_set": 5,
                "tidal_volume_obs": 430,
                "resp_rate_obs": 16,
                "plateau_pressure_obs": 18,
                "peak_inspiratory_pressure_obs": 22,
                "peep_obs": 5,
                "minute_vent_obs": 8.0,
                "mean_airway_pressure_obs": 10,
            },
            {
                "hospitalization_id": 400,
                "recorded_dttm": "2026-01-04 12:00:00",
                "device_category": "IMV",
                "mode_category": "VC",
                "tracheostomy": False,
                "fio2_set": 50,
                "tidal_volume_set": 450,
                "resp_rate_set": 18,
                "pressure_control_set": pd.NA,
                "pressure_support_set": pd.NA,
                "peep_set": 8,
                "tidal_volume_obs": 450,
                "resp_rate_obs": 18,
                "plateau_pressure_obs": 19,
                "peak_inspiratory_pressure_obs": 24,
                "peep_obs": 8,
                "minute_vent_obs": 9.0,
                "mean_airway_pressure_obs": 11,
            },
        ]
    )

    patient_assessments = pd.DataFrame(
        [
            {
                "hospitalization_id": 100,
                "recorded_dttm": "2026-01-01 09:30:00",
                "assessment_category": "GCS",
                "assessment_group": "Neuro",
                "numerical_value": 7,
                "categorical_value": pd.NA,
            },
            {
                "hospitalization_id": 100,
                "recorded_dttm": "2026-01-01 12:30:00",
                "assessment_category": "RASS",
                "assessment_group": "Sedation",
                "numerical_value": -3,
                "categorical_value": pd.NA,
            },
            {
                "hospitalization_id": 200,
                "recorded_dttm": "2026-01-02 13:30:00",
                "assessment_category": "RASS",
                "assessment_group": "Sedation",
                "numerical_value": -2,
                "categorical_value": pd.NA,
            },
        ]
    )

    return {
        "patient": patient,
        "hospitalization": hospitalization,
        "hospital_diagnosis": hospital_diagnosis,
        "adt": adt,
        "respiratory_support": respiratory_support,
        "patient_assessments": patient_assessments,
    }


def make_trauma_codes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "diagnosis_code_format": "icd10cm",
                "prefix": "S",
            }
        ]
    )


def make_location_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_column": "location_type",
                "match_value": "sicu",
                "normalized_unit": "SICU",
                "is_sicu": True,
                "is_ward": False,
                "is_procedural": False,
                "is_ed": False,
            }
        ]
    )


class PipelineTests(unittest.TestCase):
    def test_pipeline_builds_expected_outputs(self) -> None:
        artifacts = build_analysis_artifacts(
            tables=make_tables(),
            trauma_code_set=make_trauma_codes(),
            location_map=make_location_map(),
        )

        cohort_ids = set(artifacts.cohort["hospitalization_id"])
        self.assertEqual(cohort_ids, {100, 200})

        patient_100 = artifacts.cohort.loc[artifacts.cohort["hospitalization_id"] == 100].iloc[0]
        self.assertAlmostEqual(patient_100["ed_los_hours"], 3.0)
        self.assertAlmostEqual(patient_100["sicu_preward_los_hours"], 29.0)
        self.assertEqual(patient_100["transfer_outcome"], "ward")
        self.assertFalse(patient_100["in_hospital_mortality"])

        patient_200 = artifacts.cohort.loc[artifacts.cohort["hospitalization_id"] == 200].iloc[0]
        self.assertTrue(patient_200["has_procedural_bridge"])
        self.assertTrue(patient_200["in_hospital_mortality"])
        self.assertEqual(patient_200["transfer_outcome"], "death")
        self.assertTrue(pd.isna(patient_200["sicu_preward_los_hours"]))

        phase_counts = artifacts.phase_windows.set_index(["hospitalization_id", "phase"])["intervention_count"].to_dict()
        self.assertEqual(phase_counts[(100, "ED")], 1)
        self.assertEqual(phase_counts[(100, "SICU_24h")], 3)
        self.assertEqual(phase_counts[(200, "ED")], 0)
        self.assertEqual(phase_counts[(200, "SICU_24h")], 3)

        interventions_100 = artifacts.interventions.loc[artifacts.interventions["hospitalization_id"] == 100]
        self.assertEqual(len(interventions_100), 4)
        self.assertEqual(interventions_100["phase"].tolist(), ["ED", "SICU_24h", "SICU_24h", "SICU_24h"])

        assessment_context = artifacts.assessment_context.set_index(["hospitalization_id", "phase"])
        self.assertEqual(int(assessment_context.loc[(100, "ED"), "neurologic_assessment_count"]), 1)
        self.assertEqual(int(assessment_context.loc[(100, "SICU_24h"), "sedation_assessment_count"]), 1)

        transfer_summary = artifacts.transfer_summary.set_index("transfer_outcome")["count"].to_dict()
        self.assertEqual(transfer_summary["ward"], 1)
        self.assertEqual(transfer_summary["death"], 1)

    def test_cohort_flow_tracks_exclusions(self) -> None:
        artifacts = build_analysis_artifacts(
            tables=make_tables(),
            trauma_code_set=make_trauma_codes(),
            location_map=make_location_map(),
        )
        flow = artifacts.cohort_flow.set_index("stage")["count"].to_dict()
        self.assertEqual(flow["hospitalizations_total"], 4)
        self.assertEqual(flow["adult_trauma_with_valid_pathway"], 3)
        self.assertEqual(flow["final_cohort"], 2)


if __name__ == "__main__":
    unittest.main()
