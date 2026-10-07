from pathlib import Path
import numpy as np
import sys

from spyral.trace.trace_reader import create_reader
from spyral.trace.frib_event import TriggerType
from spyral.core.point_cloud import point_cloud_from_get, calibrate_point_cloud_z, sort_point_cloud_in_z, discart_outside_cloud_in_z
from spyral.core.pad_map import PadMap

from config import PAD_PARAMS, GET_PARAMS, FRIB_PARAMS, DET_PARAMS

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
    run_number = int(sys.argv[3])
	
    pad_params = PAD_PARAMS
    get_params = GET_PARAMS
    frib_params = FRIB_PARAMS
    det_params = DET_PARAMS
    pad_map = PadMap(pad_params)PAD_MAP
    rng = np.random.default_rng()
    
    trace_path = TRACE_PATH
    workspace_path = WORKSPACE_PATH
    workspace_pointcloud_path = WORKSPACE_POINTCLOUD_ASSETS_PATH
    
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
            
        trigger_peak = event.frib.get_triggering_ic_peak(frib_params)
        if trigger_peak is None:
			continue
        
        ic_mult = event.frib.get_ic_multiplicity(frib_params)
        
        if ic_mult >  frib_params.ic_multiplicity:
			continue
        
        corr_ic = 0.0
        corr_ic += (trigger_peak.centroid-1120)*det_params.get_frequency/12.5
        
        ic_peak = None
        cloud = point_cloud_from_get(event.get_pads, pad_map)
        
        reaction_peak = event.frib.get_good_ic_peak(frib_params)
            
        if reaction_peak is not None:
            
            corr_ic += event.frib.correct_ic_time(reaction_peak, frib_params, det_params.get_frequency)
            ic_peak = reaction_peak
            
        else:
            ic_peak = trigger_peak
            
        calibrate_point_cloud_z(cloud, det_params, corr_ic, None)
        discart_outside_cloud_in_z(cloud)
        sort_point_cloud_in_z(cloud)

        save_event_checkpoint(file_name, run_number, idx, cloud, event, ic_peak, ic_mult, frib_params)

if __name__=="__main__":
    main()
