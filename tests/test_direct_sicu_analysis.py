from __future__ import annotations

import unittest

import pandas as pd

from clif_trauma.direct_sicu_analysis import (
    build_boarding_mortality_bins,
    build_elapsed_hour_rates,
    build_phase_rate_comparison,
)


class DirectSicuAnalysisTests(unittest.TestCase):
    def test_phase_rate_comparison(self) -> None:
        phase_windows = pd.DataFrame(
            [
                {"hospitalization_id": 1, "phase": "ED", "phase_duration_hours": 2.0, "intervention_count": 2, "interventions_per_vent_hour": 1.0},
                {"hospitalization_id": 2, "phase": "ED", "phase_duration_hours": 1.0, "intervention_count": 1, "interventions_per_vent_hour": 1.0},
                {"hospitalization_id": 1, "phase": "SICU_24h", "phase_duration_hours": 4.0, "intervention_count": 2, "interventions_per_vent_hour": 0.5},
            ]
        )
        summary = build_phase_rate_comparison(phase_windows)
        ed = summary.loc[summary["phase"] == "ED"].iloc[0]
        self.assertEqual(int(ed["hospitalizations"]), 2)
        self.assertAlmostEqual(float(ed["mean_patient_rate"]), 1.0)
        self.assertAlmostEqual(float(ed["winsorized_mean_patient_rate"]), 1.0)

    def test_elapsed_hour_rates(self) -> None:
        phase_windows = pd.DataFrame(
            [
                {
                    "hospitalization_id": 1,
                    "phase": "ED",
                    "phase_start_dttm": pd.Timestamp("2026-01-01 10:00:00"),
                    "phase_duration_hours": 2.5,
                },
                {
                    "hospitalization_id": 1,
                    "phase": "SICU_24h",
                    "phase_start_dttm": pd.Timestamp("2026-01-01 12:00:00"),
                    "phase_duration_hours": 3.0,
                },
            ]
        )
        interventions = pd.DataFrame(
            [
                {"hospitalization_id": 1, "phase": "ED", "event_dttm": pd.Timestamp("2026-01-01 10:30:00")},
                {"hospitalization_id": 1, "phase": "ED", "event_dttm": pd.Timestamp("2026-01-01 11:10:00")},
                {"hospitalization_id": 1, "phase": "SICU_24h", "event_dttm": pd.Timestamp("2026-01-01 13:15:00")},
            ]
        )
        rates = build_elapsed_hour_rates(phase_windows, interventions, max_hour=4)
        ed_hour_0 = rates.loc[(rates["phase"] == "ED") & (rates["elapsed_hour"] == 0)].iloc[0]
        ed_hour_2 = rates.loc[(rates["phase"] == "ED") & (rates["elapsed_hour"] == 2)].iloc[0]
        self.assertEqual(int(ed_hour_0["event_count"]), 1)
        self.assertAlmostEqual(float(ed_hour_2["exposure_hours"]), 0.5)
        self.assertEqual(int(ed_hour_0["n_active"]), 1)
        self.assertGreaterEqual(float(ed_hour_0["winsorized_mean_rate"]), 0.0)

    def test_boarding_mortality_bins(self) -> None:
        cohort = pd.DataFrame(
            [
                {"hospitalization_id": 1, "ed_los_hours": 1.5, "in_hospital_mortality": False},
                {"hospitalization_id": 2, "ed_los_hours": 3.0, "in_hospital_mortality": True},
                {"hospitalization_id": 3, "ed_los_hours": 7.0, "in_hospital_mortality": False},
                {"hospitalization_id": 4, "ed_los_hours": 25.0, "in_hospital_mortality": True},
            ]
        )
        summary = build_boarding_mortality_bins(cohort)
        first_bin = summary.loc[summary["boarding_bin"] == "0-<2"].iloc[0]
        last_bin = summary.loc[summary["boarding_bin"] == "24+"].iloc[0]
        self.assertEqual(int(first_bin["hospitalizations"]), 1)
        self.assertEqual(int(last_bin["deaths"]), 1)


if __name__ == "__main__":
    unittest.main()
