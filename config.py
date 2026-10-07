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

TRACE_PATH = Path("/home/danilo-marcato/Documents/Spyral/traces")
WORKSPACE_PATH = Path("workspace/")

WORKSPACE_POINTCLOUD_ASSETS_PATH = WORKSPACE_PATH / "Pointcloud_assets/"
WORKSPACE_POINTCLOUD_PATH = WORKSPACE_PATH / "Pointcloud/"

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
    ic_delay_time_bucket=1000,
    ic_multiplicity=5,
)

DET_PARAMS = DetectorParameters(
    magnetic_field=2.85,
    electric_field=45000.0,
    detector_length=1000.0,
    beam_region_radius=30.0,
    micromegas_time_bucket=10.0,
    window_time_bucket=360.0,
    get_frequency=3.125,
    garfield_file_path=Path("/path/to/some/garfield.txt"),
    do_garfield_correction=False,
)

CLUSTER_PARAMS = ClusterParameters(
    min_cloud_size=30,
    hdbscan_parameters = None,
    continuity_join = ContinuityJoinParameters(
        join_radius_fraction=0.4,
        join_z_fraction=0.2),
    overlap_join=None,
    outlier_scale_factor=0.1,
    direction_threshold=0.5,
    tripclust_parameters=TripclustParameters(
         r=2, #6
         rdnn=True,
         k=19, #12
         n=2, #3
         a=0.03,
         s=0.3,
         sdnn=True,
         t=0.0,
         tauto=True,
         dmax=0.0,
         dmax_dnn=False,
         ordered=False,#True
         link=0,
         m=5,#50
         postprocess=False,
         min_depth=25,
     ),
)

ESTIMATE_PARAMS = EstimateParameters(
    min_total_trajectory_points=15, smoothing_factor=100.0
)

SLOVER_PARAMS = SolverParameters(
    gas_data_path=Path("/path/to/some/gas/data.json"),
    particle_id_filename=Path("/path/to/some/particle/id.json"),
    ic_min_val=900.0,
    ic_max_val=1350.0,
    n_time_steps=1000,
    interp_ke_min=0.1,
    interp_ke_max=70.0,
    interp_ke_bins=350,
    interp_polar_min=2.0,
    interp_polar_max=88.0,
    interp_polar_bins=166,
    fit_vertex_rho=True,
    fit_vertex_phi=True,
    fit_azimuthal=True,
    fit_method="lbfgsb",
)
