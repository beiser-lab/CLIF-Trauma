from __future__ import annotations

import unittest

import pandas as pd

from clif_trauma.report import (
    build_imv_vs_boarding_summary,
    build_observed_imv_summary,
    build_sankey_sequences,
    build_top_diagnoses_summary,
)


class ReportTests(unittest.TestCase):
    def test_build_observed_imv_summary_sums_ed_and_icu_only(self) -> None:
        cohort = pd.DataFrame(
            [
                {
                    "hospitalization_id": 1,
                    "first_imv_dttm": pd.Timestamp("2026-01-01 10:00:00"),
                    "discharge_dttm": pd.Timestamp("2026-01-02 12:00:00"),
                }
            ]
        )
        respiratory_support = pd.DataFrame(
            [
                {"hospitalization_id": 1, "recorded_dttm": pd.Timestamp("2026-01-01 10:00:00"), "is_imv": True},
                {"hospitalization_id": 1, "recorded_dttm": pd.Timestamp("2026-01-01 12:00:00"), "is_imv": True},
                {"hospitalization_id": 1, "recorded_dttm": pd.Timestamp("2026-01-01 16:00:00"), "is_imv": False},
            ]
        )
        adt = pd.DataFrame(
            [
                {
                    "hospitalization_id": 1,
                    "in_dttm": pd.Timestamp("2026-01-01 09:00:00"),
                    "segment_end_dttm": pd.Timestamp("2026-01-01 11:00:00"),
                    "is_ed": True,
                    "is_icu": False,
                },
                {
                    "hospitalization_id": 1,
                    "in_dttm": pd.Timestamp("2026-01-01 11:00:00"),
                    "segment_end_dttm": pd.Timestamp("2026-01-01 13:00:00"),
                    "is_ed": False,
                    "is_icu": True,
                },
                {
                    "hospitalization_id": 1,
                    "in_dttm": pd.Timestamp("2026-01-01 13:00:00"),
                    "segment_end_dttm": pd.Timestamp("2026-01-01 14:00:00"),
                    "is_ed": False,
                    "is_icu": False,
                },
                {
                    "hospitalization_id": 1,
                    "in_dttm": pd.Timestamp("2026-01-01 14:00:00"),
                    "segment_end_dttm": pd.Timestamp("2026-01-01 18:00:00"),
                    "is_ed": False,
                    "is_icu": True,
                },
            ]
        )

        summary = build_observed_imv_summary(respiratory_support, adt, cohort)
        row = summary.iloc[0]
        self.assertAlmostEqual(float(row["observed_total_imv_hours"]), 6.0)
        self.assertAlmostEqual(float(row["observed_ed_icu_imv_hours"]), 5.0)

    def test_build_top_diagnoses_summary_splits_direct_and_bridge(self) -> None:
        cohort = pd.DataFrame(
            [
                {"hospitalization_id": 1, "has_procedural_bridge": False},
                {"hospitalization_id": 2, "has_procedural_bridge": False},
                {"hospitalization_id": 3, "has_procedural_bridge": True},
            ]
        )
        hospital_diagnosis = pd.DataFrame(
            [
                {"hospitalization_id": 1, "diagnosis_code_format": "ICD10CM", "diagnosis_code": "S06.5X0A", "diagnosis_primary": True},
                {"hospitalization_id": 2, "diagnosis_code_format": "ICD10CM", "diagnosis_code": "S06.5X0A", "diagnosis_primary": True},
                {"hospitalization_id": 3, "diagnosis_code_format": "ICD10CM", "diagnosis_code": "S36.039A", "diagnosis_primary": True},
                {"hospitalization_id": 3, "diagnosis_code_format": "ICD10CM", "diagnosis_code": "T14.90XA", "diagnosis_primary": False},
            ]
        )
        dictionary = pd.DataFrame(
            [
                {
                    "diagnosis_code_format": "ICD10CM",
                    "diagnosis_code": "S06.5X0A",
                    "diagnosis_name": "Traumatic subdural hemorrhage with no loss of consciousness, initial encounter",
                }
            ]
        )

        summary = build_top_diagnoses_summary(hospital_diagnosis, cohort, dictionary)
        top = summary.iloc[0]
        self.assertEqual(top["diagnosis_code"], "S06.5X0A")
        self.assertEqual(top["diagnosis_name"], "Traumatic subdural hemorrhage with no loss of consciousness, initial encounter")
        self.assertEqual(int(top["overall_n"]), 2)
        self.assertEqual(int(top["direct_ed_to_sicu_n"]), 2)
        self.assertEqual(int(top["ed_to_or_to_sicu_n"]), 0)

    def test_build_sankey_sequences_tracks_bridge_and_terminal_outcomes(self) -> None:
        cohort = pd.DataFrame(
            [
                {
                    "hospitalization_id": 1,
                    "has_procedural_bridge": False,
                    "transfer_outcome": "ward",
                    "in_hospital_mortality": False,
                },
                {
                    "hospitalization_id": 2,
                    "has_procedural_bridge": True,
                    "transfer_outcome": "procedural",
                    "in_hospital_mortality": True,
                },
            ]
        )
        adt = pd.DataFrame(
            [
                {"hospitalization_id": 1, "in_dttm": pd.Timestamp("2026-01-01 09:00:00"), "out_dttm": pd.Timestamp("2026-01-01 10:00:00"), "is_ed": True, "is_procedural": False, "is_icu": False, "is_ward": False},
                {"hospitalization_id": 1, "in_dttm": pd.Timestamp("2026-01-01 10:00:00"), "out_dttm": pd.Timestamp("2026-01-01 14:00:00"), "is_ed": False, "is_procedural": False, "is_icu": True, "is_ward": False},
                {"hospitalization_id": 1, "in_dttm": pd.Timestamp("2026-01-01 14:00:00"), "out_dttm": pd.Timestamp("2026-01-01 18:00:00"), "is_ed": False, "is_procedural": False, "is_icu": False, "is_ward": True},
                {"hospitalization_id": 2, "in_dttm": pd.Timestamp("2026-01-01 09:00:00"), "out_dttm": pd.Timestamp("2026-01-01 10:00:00"), "is_ed": True, "is_procedural": False, "is_icu": False, "is_ward": False},
                {"hospitalization_id": 2, "in_dttm": pd.Timestamp("2026-01-01 10:00:00"), "out_dttm": pd.Timestamp("2026-01-01 12:00:00"), "is_ed": False, "is_procedural": True, "is_icu": False, "is_ward": False},
                {"hospitalization_id": 2, "in_dttm": pd.Timestamp("2026-01-01 12:00:00"), "out_dttm": pd.Timestamp("2026-01-01 18:00:00"), "is_ed": False, "is_procedural": False, "is_icu": True, "is_ward": False},
                {"hospitalization_id": 2, "in_dttm": pd.Timestamp("2026-01-02 08:00:00"), "out_dttm": pd.Timestamp("2026-01-02 10:00:00"), "is_ed": False, "is_procedural": True, "is_icu": False, "is_ward": False},
                {"hospitalization_id": 2, "in_dttm": pd.Timestamp("2026-01-02 10:00:00"), "out_dttm": pd.Timestamp("2026-01-02 16:00:00"), "is_ed": False, "is_procedural": False, "is_icu": True, "is_ward": False},
            ]
        )

        sequences = build_sankey_sequences(cohort, adt)
        direct = sequences.loc[sequences["hospitalization_id"] == 1].iloc[0]
        bridge = sequences.loc[sequences["hospitalization_id"] == 2].iloc[0]
        self.assertEqual(direct["stage_0"], "ED")
        self.assertEqual(direct["stage_1"], "ICU")
        self.assertEqual(direct["stage_2"], "Ward")
        self.assertEqual(direct["stage_3"], "Discharge")
        self.assertEqual(bridge["stage_1"], "OR")
        self.assertEqual(bridge["stage_2"], "ICU")
        self.assertEqual(bridge["stage_3"], "OR")
        self.assertEqual(bridge["stage_4"], "ICU")
        self.assertEqual(bridge["stage_5"], "Death")

    def test_build_imv_vs_boarding_summary_uses_direct_subgroup(self) -> None:
        cohort = pd.DataFrame(
            [
                {"hospitalization_id": 1, "has_procedural_bridge": False, "ed_los_hours": 1.5},
                {"hospitalization_id": 2, "has_procedural_bridge": False, "ed_los_hours": 7.0},
                {"hospitalization_id": 3, "has_procedural_bridge": True, "ed_los_hours": 7.0},
            ]
        )
        imv_summary = pd.DataFrame(
            [
                {"hospitalization_id": 1, "observed_ed_icu_imv_hours": 10.0},
                {"hospitalization_id": 2, "observed_ed_icu_imv_hours": 20.0},
                {"hospitalization_id": 3, "observed_ed_icu_imv_hours": 40.0},
            ]
        )

        summary = build_imv_vs_boarding_summary(cohort, imv_summary)
        early = summary.loc[summary["boarding_bin"] == "0-<2"].iloc[0]
        late = summary.loc[summary["boarding_bin"] == "6-<12"].iloc[0]
        self.assertEqual(int(early["hospitalizations"]), 1)
        self.assertAlmostEqual(float(late["median_observed_ed_icu_imv_hours"]), 20.0)


if __name__ == "__main__":
    unittest.main()
