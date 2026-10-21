from pathlib import Path

from spyral import (
    PadParameters,
    GetParameters,
    FribParameters,
    DetectorParameters,
    ClusterParameters,
    HdbscanParameters,
    TripclustParameters,
    OverlapJoinParameters,
    ContinuityJoinParameters,
    SolverParameters,
    EstimateParameters,
    DEFAULT_MAP,
)

SPYRAL_MODIFICATION = Path("/home/danilo-marcato/Documents/Spyral_Modification")

import sys
sys.path.append(SPYRAL_MODIFICATION)

from clusterize_modification import RevaluationParameters


#SPYRAL_MODIFICATION = Path("/mnt/simulations/attpcroot/miniconda/spyral_modifications")
#TRACE_PATH = Path("/mnt/analysis/argonne2023/a2066/h5/")

TRACE_PATH = Path("/home/danilo-marcato/Documents/Spyral/traces")
WORKSPACE_PATH = Path("/home/danilo-marcato/Documents/spyral_parallel/workspace/")

WORKSPACE_POINTCLOUD_ASSETS_PATH = WORKSPACE_PATH / "Pointcloud_assets/"
WORKSPACE_POINTCLOUD_PATH = WORKSPACE_PATH / "Pointcloud/"

WORKSPACE_CLUSTER_ASSETS_PATH = WORKSPACE_PATH / "Cluster_assets/"
WORKSPACE_CLUSTER_PATH = WORKSPACE_PATH / "Cluster/"

WORKSPACE_ESTIMATION_ASSETS_PATH = WORKSPACE_PATH / "Estimation_assets/"
WORKSPACE_ESTIMATION_PATH = WORKSPACE_PATH / "Estimation/"

PAD_PARAMS = PadParameters(
    pad_geometry_path=DEFAULT_MAP,
    pad_time_path=DEFAULT_MAP,
    pad_scale_path=DEFAULT_MAP,
)

GET_PARAMS = GetParameters(
    baseline_window_scale=20.0,
    peak_separation=50.0,
    peak_prominence=20.0,
    peak_max_width=50.0,
    peak_threshold=40.0,
)

FRIB_PARAMS = FribParameters(
    baseline_window_scale=100.0,
    peak_separation=50.0,
    peak_prominence=20.0,
    peak_max_width=500.0,
    peak_threshold=100.0,
    ic_delay_time_bucket=1100,
    ic_multiplicity=5,
)

DET_PARAMS = DetectorParameters(
    magnetic_field=2.85,
    electric_field=40000.0,
    detector_length=1000.0,
    beam_region_radius=30.0,
    micromegas_time_bucket=10.0,
    window_time_bucket=344.0,
    get_frequency=3.125,
    garfield_file_path=Path("/path/to/some/garfield.txt"),
    do_garfield_correction=False,
)

REV_PARAMS = RevaluationParameters(
    min_cloud_size=30,
    threshold_circle = 25.0,
    threshold_line = 20.0,
    min_inliers = 20,
    n_iterations = 2000,
    origin_weight = 2.0,
    radius_cut = 30.0,
    inside_perc = 0.75,
    max_revaluation = 5,
    use_radius = True,
    ang_min = 2.5,
    ang_max = 7.5,
    overlap_tol = 0.5,
    zone_1 = 10.0,
    zone_2 = 25.0,
    dm = 10.0,
    reeavaluate_all = True,
    discartNoise = False,)
