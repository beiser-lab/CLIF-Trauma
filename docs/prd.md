# PRD: Trauma Ventilation Management and Outcomes Across ED and SICU

## Summary

Use CLIF 2.1.0 to identify adult trauma hospitalizations with invasive mechanical ventilation initiated after ED presentation and admitted to the surgical ICU, then quantify ventilator-management changes across the ED and early SICU course. Expand the analysis to include discharge outcomes and duration metrics: hospital mortality, discharge status, hospital length of stay, ED length of stay, and time in SICU before first transfer to a ward.

## Core Deliverables

- Cohort table with one row per hospitalization
- Phase window table for `ED` and `SICU_24h`
- Ventilator intervention log
- First post-SICU transfer outcome table
- Summary tables for cohort flow, phase-level intervention rates, handoff differences, outcomes, and transfer destinations
- Direct `ED -> SICU` robustness summaries with winsorized hourly intervention rates, denominator tracking, and ED boarding mortality
- Table 1 with cohort characteristics, LOS metrics, mortality, intervention exposure, and observed ED+ICU IMV duration
- Top 10 primary diagnosis table using raw CLIF diagnosis codes plus a report-level ICD-10-CM name lookup for interpretability
- Figure set with a cohort flow diagram, unit-location sankey, direct-subgroup intervention plots, and ED LOS versus observed ED+ICU IMV duration
- Self-contained HTML report for collaborator review

## Cohort Rules

- Adults only: `age_at_admission >= 18`
- Trauma cohort is diagnosis-based using a versioned trauma ICD value set
- Require an ED segment in `adt`
- Require SICU admission identified by `adt.location_category = 'icu'` plus site-mapped surgical ICU in `location_type` or `location_name`
- Allow `ED -> SICU` or `ED -> procedural -> SICU`
- Use the first `respiratory_support.recorded_dttm` with an IMV device as the intubation proxy
- Require the first IMV timestamp to occur during the ED interval so ED ventilation management is observable

## Outcomes

- In-hospital mortality from `hospitalization.discharge_category`
- Harmonized discharge grouping plus raw CLIF discharge category
- Hospital LOS from hospitalization timestamps
- ED LOS from the first ED ADT interval
- SICU pre-ward LOS from first SICU entry to first transfer to a non-ICU ward
- Initial SICU LOS from first SICU entry to first SICU exit
- First observed ED+ICU IMV duration from the first IMV episode after ED intubation, summing only ED and ICU time

## Notes

- `patient_assessments` is used for contextual neurologic, sedation, and readiness information only
- Procedural bridge intervals can update carried-forward ventilator settings but are excluded from phase-level intervention counts
- The primary hourly intervention analysis excludes ED-to-OR bridge patients and focuses on the direct `ED -> SICU` subgroup
- Sparse hourly intervention rates are presented with both raw mean rates and an upper-95% winsorized mean companion metric
- The sankey plot retains the broader final cohort, including bridge patients, and is derived from raw ADT location sequences so ICU-to-OR-to-ICU returns are visible
- The CONSORT figure should display the inclusive study date range used for the source hospitalization set
