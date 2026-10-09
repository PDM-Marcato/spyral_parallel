# estimation_phase.py

from pathlib import Path
import json
import sys

import h5py as h5
import numpy as np

from spyral.core.cluster import Cluster, Direction
from spyral.core.estimation import estimate_physics

from config import (
    WORKSPACE_CLUSTER_PATH,
    WORKSPACE_ESTIMATION_ASSETS_PATH,
    ESTIMATE_PARAMS,
    DETECTOR_PARAMS,
)


def run_workspace_dir(workspace_path: Path, run_number: int) -> Path:
    """Return the directory containing event checkpoints."""
    directory = workspace_path / f"run_{run_number:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def create_event_file(
    workspace_path: Path,
    run_number: int,
    idx: int,
) -> Path:
    """Return the checkpoint path for one event."""
    return (
        run_workspace_dir(workspace_path, run_number)
        / f"event_{idx:06d}.npz"
    )


def json_default(value):
    """Convert common NumPy values into JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def save_estimation_checkpoint(
    file_name: Path,
    idx: int,
    results: list[dict],
) -> None:
    """Save all successful cluster estimates for one event."""

    results_json = json.dumps(
        results,
        default=json_default,
        allow_nan=True,
    )

    np.savez_compressed(
        file_name,
        event_number=idx,
        nresults=len(results),
        results_json=results_json,
    )


def process_event(
    idx: int,
    event: h5.Group,
) -> list[dict]:
    """Estimate the physics parameters for every cluster in an event."""

    results = []

    nclusters = int(event.attrs["nclusters"])

    # IC information
    ic_amp = float(event.attrs["ic_amplitude"])
    ic_cent = float(event.attrs["ic_centroid"])
    ic_int = float(event.attrs["ic_integral"])
    ic_mult = float(event.attrs["ic_multiplicity"])

    # Original event identifiers
    orig_run = int(event.attrs["orig_run"])
    orig_event = int(event.attrs["orig_event"])

    for cidx in range(nclusters):
        cluster_name = f"cluster_{cidx}"

        if cluster_name not in event:
            continue

        local_cluster = event[cluster_name]

        cluster = Cluster(
            idx,
            local_cluster.attrs["label"],
            Direction(local_cluster.attrs["direction"]),
            local_cluster["cloud"][:].copy(),
        )

        res = estimate_physics(
            cidx,
            cluster,
            ic_amp,
            ic_cent,
            ic_int,
            ic_mult,
            orig_run,
            orig_event,
            ESTIMATE_PARAMS,
            DETECTOR_PARAMS,
        )

        if res is not None:
            results.append(vars(res))

    return results


def main():

    process_id = int(sys.argv[1])
    n_processes = int(sys.argv[2])
    run_number = int(sys.argv[3])


    cluster_path = (WORKSPACE_CLUSTER_PATH/ f"run_{run_number:04d}_cluster.h5")

    workspace_path = WORKSPACE_ESTIMATION_ASSETS_PATH

    with h5.File(cluster_path, "r") as cluster_file:
        cluster_group = cluster_file["cluster"]

        min_event = int(cluster_group.attrs["min_event"])
        max_event = int(cluster_group.attrs["max_event"])

        # Match the event assignment strategy used by clustering.
        event_indices = [idx for idx in range(min_event, max_event + 1) if f"event_{idx}" in cluster_group]

        processed = 0
        skipped = 0
        failed = 0

    for i, idx in enumerate(event_indices):

        if i % n_processes != process_id:
            continue

        checkpoint = create_event_file(workspace_path,run_number,idx)

        event_name = f"event_{idx}"

        try:
            event = cluster_group[event_name]
            results = process_event(idx, event)
            save_estimation_checkpoint(checkpoint,idx,results)

            except Exception as exc:
                failed += 1
                print(
                    f"[Worker {process_id}] "
                    f"Failed event {idx}: {exc}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
