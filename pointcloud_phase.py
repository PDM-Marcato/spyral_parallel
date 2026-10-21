from pathlib import Path
import numpy as np
import sys

from spyral.trace.trace_reader import create_reader
from spyral.trace.frib_event import TriggerType
from spyral.core.point_cloud import point_cloud_from_get, calibrate_point_cloud_z, sort_point_cloud_in_z, discart_outside_cloud_in_z
from spyral.core.pad_map import PadMap

from config import PAD_PARAMS, GET_PARAMS, FRIB_PARAMS, DET_PARAMS, TRACE_PATH, WORKSPACE_PATH, WORKSPACE_POINTCLOUD_ASSETS_PATH, SPYRAL_MODIFICATION

sys.path.append(SPYRAL_MODIFICATION)

from helper_modification import run_workspace_dir, create_event_file

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
    
def main():
    process_id = int(sys.argv[1])
    n_processes = int(sys.argv[2])
    run_number = int(sys.argv[3])
    
    pad_map = PadMap(PAD_PARAMS)
    rng = np.random.default_rng()
    
    trace_file_path = TRACE_PATH / f"run_{run_number:04d}.h5"
    trace_reader = create_reader(trace_file_path, run_number)
    
    if trace_reader is None:
        return
    
    ic_within_mult_count = 0
    
    for idx in trace_reader.event_range():
        
        if idx % n_processes != process_id:
            continue
        
        file_name = create_event_file(WORKSPACE_POINTCLOUD_ASSETS_PATH,run_number,idx)
            
        event = trace_reader.read_event(idx, GET_PARAMS, FRIB_PARAMS, rng)
		
        if event.frib is None:
            continue
        
        ic_mult = event.frib.get_ic_multiplicity(FRIB_PARAMS)
        trigger_peak = event.frib.get_triggering_ic_peak(FRIB_PARAMS)
        
        if (event.frib.trigger == TriggerType.IC_DOWNSCALE_TRIGGER):
            if (ic_mult > 0 and ic_mult <= FRIB_PARAMS.ic_multiplicity and trigger_peak is not None):
                ic_within_mult_count += 1
            continue
        
        if event.get_pads is None:
            continue
        
        
        if ic_mult >  FRIB_PARAMS.ic_multiplicity:
            continue
        
        
        if trigger_peak is None:
            continue
        
        corr_ic = 0.0
        corr_ic += (trigger_peak.centroid-1120)*DET_PARAMS.get_frequency/12.5
        
        ic_peak = None
        reaction_peak = event.frib.get_good_ic_peak(FRIB_PARAMS)
            
        if reaction_peak is not None:
            num_reaction, good_peak = reaction_peak
            if num_reaction > 1:
                continue	
            corr_ic += event.frib.correct_ic_time(good_peak, FRIB_PARAMS, DET_PARAMS.get_frequency)
            ic_peak = good_peak
            
        else:
            ic_peak = trigger_peak
            
        cloud = point_cloud_from_get(event.get_pads, pad_map)
        calibrate_point_cloud_z(cloud, DET_PARAMS, corr_ic, None)
        discart_outside_cloud_in_z(cloud)
        sort_point_cloud_in_z(cloud)

        save_event_checkpoint(file_name, run_number, idx, cloud, event, ic_peak, ic_mult, FRIB_PARAMS)

    count_file = (run_workspace_dir(WORKSPACE_POINTCLOUD_ASSETS_PATH, run_number)/ f"gated_ic_count_{process_id:04d}.txt")
    count_file.write_text(str(ic_within_mult_count))

if __name__=="__main__":
    main()
