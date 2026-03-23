from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline, write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the CLIF trauma ventilation cohort and analysis outputs."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing CLIF input tables.")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV outputs.")
    parser.add_argument(
        "--trauma-code-set",
        required=True,
        help="CSV file containing diagnosis_code_format and prefix columns.",
    )
    parser.add_argument(
        "--location-map",
        help="Optional CSV crosswalk for ADT unit classification.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    artifacts = run_pipeline(
        input_dir=Path(args.input_dir),
        trauma_code_set_path=Path(args.trauma_code_set),
        location_map_path=Path(args.location_map) if args.location_map else None,
    )
    write_outputs(Path(args.output_dir), artifacts)


if __name__ == "__main__":
    main()
