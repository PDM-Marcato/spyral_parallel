#!/usr/bin/env python3
"""
Create a Spyral particle-ID (PID) gate from estimation Parquet files.

This is a terminal-friendly conversion of:
https://github.com/ATTPC/spyral_notebooks/blob/main/particle_id.ipynb

Run inside the environment containing spyral_utils, Spyral, Polars,
Matplotlib, and the relevant nuclear-data files.

Important:
- Set WORKSPACE below to your Spyral workspace.
- This script uses an interactive Matplotlib window to draw the polygon cut.
- Close the PID plot after drawing and closing the polygon to save the gate.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
import polars as pl

from spyral_utils.plot import CutHandler, Histogrammer
from spyral_utils.nuclear import NuclearDataMap
from spyral_utils.nuclear.particle_id import serialize_particle_id, ParticleID
from spyral.core.constants import DEG2RAD
from spyral.core.run_stacks import form_run_string


# ----------------------------------------------------------------------
# Configuration: edit these values for your experiment
# ----------------------------------------------------------------------

WORKSPACE = Path("/path/to/your/workspace")

RUN_MIN = 16
RUN_MAX = 16  # inclusive

IC_MIN_VAL = -1
IC_MAX_VAL = 4096.0

PID_NAME = "proton_cut"
PID_Z = 1
PID_A = 1

PID_X_AXIS = "sqrt_dEdx"
PID_Y_AXIS = "brho"

PID_PATH = Path("/path/to/some/gate.json")

# Histogram binning and ranges from the original notebook
PID_BINS = (400, 400)
PID_RANGE = ((-10.0, 200.0), (-0.1, 3.0))
IC_BINS = 4096
IC_RANGE = (0.0, 4096.0)
KINEMATICS_BINS = (720, 400)
KINEMATICS_RANGE = ((0.0, 180.0), (0.0, 3.0))

RAD2DEG = 1.0 / DEG2RAD


def load_estimation_data(estimation_path: Path) -> pl.DataFrame:
    """Load available run Parquet files and apply the IC gate."""
    dataframes = []

    for run in range(RUN_MIN, RUN_MAX + 1):
        run_path = estimation_path / f"{form_run_string(run)}.parquet"

        if not run_path.exists():
            print(f"Skipping missing estimation file: {run_path}")
            continue

        print(f"Reading {run_path}")
        df = pl.read_parquet(run_path)

        required = {
            PID_X_AXIS,
            PID_Y_AXIS,
            "polar",
            "brho",
            "ic_amplitude",
        }
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"{run_path} is missing required columns: {sorted(missing)}"
            )

        # Optional ion-chamber gate, matching the original notebook.
        df = df.filter(
            (pl.col("ic_amplitude") > IC_MIN_VAL)
            & (pl.col("ic_amplitude") < IC_MAX_VAL)
        )
        dataframes.append(df)

    if not dataframes:
        raise FileNotFoundError(
            f"No estimation Parquet files found for runs "
            f"{RUN_MIN} through {RUN_MAX} in {estimation_path}"
        )

    return pl.concat(dataframes, how="diagonal_relaxed")


def plot_histogram_1d(histogrammer: Histogrammer) -> None:
    ic = histogrammer.get_hist1d("ion_chamber")
    fig, ax = plt.subplots()
    ax.stairs(ic.counts, edges=ic.bins)
    ax.set_title("Ion Chamber")
    ax.set_xlabel("Amplitude (arb.)")
    ax.set_ylabel("Counts")
    fig.set_figwidth(8.0)


def plot_pid_and_select_cut(
    histogrammer: Histogrammer,
    handler: CutHandler,
) -> None:
    pid_hist = histogrammer.get_hist2d("particle_id")

    fig, ax = plt.subplots()
    selector = PolygonSelector(ax, handler.mpl_on_select)

    mesh = ax.pcolormesh(
        pid_hist.x_bins,
        pid_hist.y_bins,
        pid_hist.counts,
        norm="log",
    )
    fig.colorbar(mesh, ax=ax)
    ax.set_title("Particle ID")
    ax.set_xlabel(f"{PID_X_AXIS} Column")
    ax.set_ylabel(f"{PID_Y_AXIS} Column")
    fig.set_figheight(8.0)
    fig.set_figwidth(11.0)

    print(
        "\nDraw the particle-ID polygon by clicking its vertices. "
        "Close it by clicking the first vertex. "
        "When finished, close the plot window to continue."
    )
    plt.show()

    # Keep a reference to the selector until the window has closed.
    _ = selector


def plot_kinematics(histogrammer: Histogrammer) -> None:
    kine = histogrammer.get_hist2d("kinematics")
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(
        kine.x_bins, kine.y_bins, kine.counts, norm="log"
    )
    fig.colorbar(mesh, ax=ax)
    ax.set_title("Kinematics")
    ax.set_xlabel(r"Polar angle $\theta$ (deg)")
    ax.set_ylabel(r"$B\rho$ (Tm)")
    fig.set_figheight(8.0)
    fig.set_figwidth(11.0)


def fill_gated_histograms(
    histogrammer: Histogrammer,
    df: pl.DataFrame,
    pid: ParticleID,
) -> None:
    histogrammer.add_hist2d(
        "particle_id_gated",
        (400, 400),
        ((-10.0, 200.0), (0.0, 3.0)),
    )
    histogrammer.add_hist2d(
        "kinematics_gated",
        (720, 400),
        ((0.0, 180.0), (0.0, 3.0)),
    )

    gated = df.filter(
        pl.struct([PID_X_AXIS, PID_Y_AXIS]).map_batches(
            pid.cut.is_cols_inside
        )
    )

    gated = gated.filter(
        (pl.col("ic_amplitude") > IC_MIN_VAL)
        & (pl.col("ic_amplitude") < IC_MAX_VAL)
    )

    histogrammer.fill_hist2d(
        "particle_id_gated",
        gated.select(PID_X_AXIS).to_numpy(),
        gated.select(PID_Y_AXIS).to_numpy(),
    )
    histogrammer.fill_hist2d(
        "kinematics_gated",
        gated.select("polar").to_numpy() * RAD2DEG,
        gated.select("brho").to_numpy(),
    )


def plot_gated_histograms(histogrammer: Histogrammer) -> None:
    pid_gated = histogrammer.get_hist2d("particle_id_gated")
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(
        pid_gated.x_bins,
        pid_gated.y_bins,
        pid_gated.counts,
        norm="log",
    )
    fig.colorbar(mesh, ax=ax)
    ax.set_title("Particle ID Gated")
    ax.set_xlabel(f"{PID_X_AXIS} Column")
    ax.set_ylabel(f"{PID_Y_AXIS} Column")
    fig.set_figheight(8.0)
    fig.set_figwidth(11.0)

    kine_gated = histogrammer.get_hist2d("kinematics_gated")
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(
        kine_gated.x_bins,
        kine_gated.y_bins,
        kine_gated.counts,
        norm="log",
    )
    fig.colorbar(mesh, ax=ax)
    ax.set_title("Kinematics Gated")
    ax.set_xlabel(r"Polar angle $\theta$ (deg)")
    ax.set_ylabel(r"$B\rho$ (Tm)")
    fig.set_figheight(8.0)
    fig.set_figwidth(11.0)


def main() -> None:
    estimation_path = WORKSPACE / "Estimation"

    nuclear_map = NuclearDataMap()
    nucleus = nuclear_map.get_data(PID_Z, PID_A)

    histogrammer = Histogrammer()
    handler = CutHandler()

    histogrammer.add_hist2d(
        "particle_id", PID_BINS, PID_RANGE
    )
    histogrammer.add_hist1d(
        "ion_chamber", IC_BINS, IC_RANGE
    )
    histogrammer.add_hist2d(
        "kinematics", KINEMATICS_BINS, KINEMATICS_RANGE
    )

    df = load_estimation_data(estimation_path)

    histogrammer.fill_hist2d(
        "particle_id",
        df.select(PID_X_AXIS).to_numpy(),
        df.select(PID_Y_AXIS).to_numpy(),
    )
    histogrammer.fill_hist2d(
        "kinematics",
        df.select("polar").to_numpy() * RAD2DEG,
        df.select("brho").to_numpy(),
    )

    # The notebook plots one IC value per unique event.
    event_column = "event" if "event" in df.columns else None
    ic_df = df.unique(subset=[event_column]) if event_column else df
    histogrammer.fill_hist1d(
        "ion_chamber",
        ic_df.select("ic_amplitude").to_numpy(),
    )

    plot_histogram_1d(histogrammer)
    plot_pid_and_select_cut(histogrammer, handler)
    plot_kinematics(histogrammer)
    plt.show()

    if not handler.cuts:
        raise RuntimeError(
            "No polygon cut was registered. Draw and close a polygon "
            "on the Particle ID plot, then run the script again."
        )

    print(f"Available cuts: {list(handler.cuts.keys())}")
    cut_name = input(
        "Which cut should be saved? [cut_0]: "
    ).strip() or "cut_0"

    if cut_name not in handler.cuts:
        raise KeyError(
            f"Cut {cut_name!r} not found. "
            f"Available cuts: {list(handler.cuts.keys())}"
        )

    cut = handler.cuts[cut_name]
    cut.name = PID_NAME
    cut.x_axis = PID_X_AXIS
    cut.y_axis = PID_Y_AXIS

    pid = ParticleID(cut, nucleus)

    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    serialize_particle_id(PID_PATH, pid)
    print(f"Saved particle ID to: {PID_PATH}")

    # Show a quick validation of the saved cut on the current data.
    fill_gated_histograms(histogrammer, df, pid)
    plot_gated_histograms(histogrammer)
    plt.show()


if __name__ == "__main__":
    main()
