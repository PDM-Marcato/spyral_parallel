# estimation_compile.py

from pathlib import Path
import json
import sys

import numpy as np
import polars as pl

from config import (
    WORKSPACE_ESTIMATION_ASSETS_PATH,
    WORKSPACE_ESTIMATION_PATH,
)


def run_workspace_dir(
    workspace_path: Path,
    run_number: int,
) -> Path:
    return workspace_path / f"run_{run_number:04d}"


def compile_estimation_run(
    run_number: int,
    workspace_path: Path,
    output_path: Path,
) -> None:
    """Combine event checkpoints into one Parquet file."""

    ws_dir = run_workspace_dir(workspace_path, run_number)

    npz_files = sorted(ws_dir.glob("event_*.npz"))

    if not npz_files:
        print(
            f"No estimation checkpoints found for "
            f"run {run_number}. Skipping."
        )
        return

    results = []
    compiled_events = 0

    for npz_path in npz_files:
        try:
            with np.load(npz_path, allow_pickle=False) as npz:
                if "event_number" not in npz.files:
                    continue

                event_number = int(npz["event_number"])

                event_results = json.loads(
                    str(npz["results_json"].item())
                )

                # Restore the event number in every result.
                for result in event_results:
                    result.setdefault("event_number", event_number)

                results.extend(event_results)
                compiled_events += 1

        except Exception as exc:
            print(
                f"Failed to read checkpoint {npz_path}: {exc}",
                flush=True,
            )

    output_path.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_path / f"run_{run_number:04d}_estimation.parquet"
    )

    # Handle the case where no cluster produced a valid estimate.
    if results:
        df = pl.DataFrame(results)
    else:
        df = pl.DataFrame()

    df.write_parquet(output_file)

    print(
        f"Compiled {compiled_events} event checkpoints "
        f"containing {len(results)} estimates "
        f"for run {run_number} -> {output_file}",
        flush=True,
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python estimation_compile.py <run_number>"
        )

    run_number = int(sys.argv[1])

    compile_estimation_run(
        run_number,
        WORKSPACE_ESTIMATION_ASSETS_PATH,
        WORKSPACE_ESTIMATION_PATH,
    )


if __name__ == "__main__":
    main()
