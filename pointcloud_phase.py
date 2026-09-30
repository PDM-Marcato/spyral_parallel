from pathlib import Path
import numpy as np
import sys

from spyral.trace.trace_reader import create_reader
from spyral.trace.frib_event import TriggerType
from spyral.core.point_cloud import point_cloud_from_get, calibrate_point_cloud_z, sort_point_cloud_in_z
from spyral.core.pad_map import PadMap
from spyral import PadParameters, GetParameters, FribParameters, DetectorParameters, DEFAULT_MAP

def run_workspace_dir(workspace_path: Path, run_number: int) -> Path:
    """Scratch folder that holds one .npz per processed event for this run."""
    d = workspace_path / f"run_{run_number}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def create_event_file(workspace_path: Path, run_number: int, idx: int)-> Path:
    file_name = run_workspace_dir(workspace_path, run_number) / f"event_{idx:06d}.npz"
    return file_name

def save_event_checkpoint(file_name: Path, run_number: int, idx: int, cloud, event, ic_peak, ic_mult, frib_params) -> None:

    has_ic = (
        ic_peak is not None
        and ic_mult > 0.0
        and ic_mult <= frib_params.ic_multiplicity
    )

    kwargs = dict(
        data=cloud.data,
        event_number=cloud.event_number,
        orig_run=event.original_run,
        orig_event=event.original_event,
        has_ic=has_ic,
    )
    if has_ic:
        kwargs.update(
            ic_amplitude=ic_peak.amplitude,
            ic_integral=ic_peak.integral,
            ic_centroid=ic_peak.centroid,
            ic_multiplicity=ic_mult,
        )
        
    np.savez(file_name, **kwargs)
    
def main(runs, pad_map, get_params, frib_params, rng):
    process_id = int(sys.argv[1])
    n_processes = int(sys.argv[2])
	
    trace_path = Path("/home/danilo-marcato/Documents/Spyral/traces")
    workspace_path = Path("workspace/")
    workspace_pointcloud_path = workspace_path / "Pointcloud_assets/"

    for run_number in runs:

        trace_file_path = trace_path / f"run_{run_number:04d}.h5"
        trace_reader = create_reader(trace_file_path, run_number)

        if trace_reader is None:
            continue

        for idx in trace_reader.event_range():

            if idx % n_processes != process_id:
                continue

            file_name = create_event_file(workspace_pointcloud_path,run_number,idx)
            
            event = trace_reader.read_event(idx, get_params, frib_params, rng)
            
            if event.get_pads is None:
                continue
			
            if event.frib is None:
                continue
            
            if (event.frib.trigger == TriggerType.IC_DOWNSCALE_TRIGGER):
                continue
            
            ic_mult = -1.0
            ic_peak = None
            cloud = point_cloud_from_get(event.get_pads, pad_map)
            
            corr_ic = 0.0
            if event.frib.get_good_ic_peak(frib_params) is not None:
                good_ic_count, good_peak = event.frib.get_good_ic_peak(frib_params)
                corr_ic = event.frib.correct_ic_time(good_peak, frib_params, det_params.get_frequency)
                ic_peak = good_peak
                ic_mult = good_ic_count
            else:
                ic_peak = event.frib.get_triggering_ic_peak(frib_params)
                ic_mult = event.frib.get_ic_multiplicity(frib_params)
            
            calibrate_point_cloud_z(cloud, det_params, corr_ic, None)
            sort_point_cloud_in_z(cloud)

            save_event_checkpoint(file_name, run_number, idx, cloud, event, ic_peak, ic_mult, frib_params)

pad_params = PadParameters(
    pad_geometry_path=DEFAULT_MAP,
    pad_time_path=DEFAULT_MAP,
    pad_scale_path=DEFAULT_MAP,
)

get_params = GetParameters(
    baseline_window_scale=20.0,
    peak_separation=50.0,
    peak_prominence=20.0,
    peak_max_width=50.0,
    peak_threshold=40.0,
)

frib_params = FribParameters(
    baseline_window_scale=100.0,
    peak_separation=50.0,
    peak_prominence=20.0,
    peak_max_width=500.0,
    peak_threshold=100.0,
    ic_delay_time_bucket=1000,
    ic_multiplicity=5,
)

det_params = DetectorParameters(
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

pad_map = PadMap(pad_params)
rng = np.random.default_rng()

runs = [46, 89, 95]

if __name__=="__main__":
    main(runs, pad_map, get_params, frib_params, rng)
