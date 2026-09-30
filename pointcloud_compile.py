from pathlib import Path
import numpy as np
import h5py as h5

from spyral.trace.trace_reader import create_reader
from spyral.core.run_stacks import form_run_string

def run_workspace_dir(workspace_path: Path, run_number: int) -> Path:
    """Scratch folder that holds one .npz per processed event for this run."""
    d = workspace_path / f"run_{run_number}"
    d.mkdir(parents=True, exist_ok=True)
    return d
    
def compile_run(run_number: int, trace_path: Path, workspace_path: Path,output_path: Path) -> None:
    
    ws_dir = run_workspace_dir(workspace_path, run_number)
    npz_files = sorted(ws_dir.glob("event_*.npz"))
    if not npz_files:
        print(f"No checkpoints found for run_{run_number}, skipping compile.")
        return
 
    # re-open the trace to recover first/last event metadata, same as before
    trace_file_path = trace_path / f"{form_run_string(run_number)}.h5"
    trace_reader = create_reader(trace_file_path, run_number)
    
    out_file_path = output_path / f"run_{run_number:04d}_pc.h5"
    with h5.File(out_file_path, "w") as point_file:
        cloud_group = point_file.create_group("cloud")
        cloud_group.attrs["min_event"] = trace_reader.first_event()
        cloud_group.attrs["max_event"] = trace_reader.last_event()
        compiled = 0
        for npz_path in npz_files:
            
            with np.load(npz_path) as npz:
                # Skip empty checkpoint files
                if "event_number" not in npz.files:
                    #print(f"Skipping empty checkpoint: {npz_path.name}")
                    continue
                
                event_number = int(npz["event_number"])
                pc_dataset = cloud_group.create_dataset(
                    f"cloud_{event_number}", data=npz["data"]
                )
                pc_dataset.attrs["orig_run"] = int(npz["orig_run"])
                pc_dataset.attrs["orig_event"] = int(npz["orig_event"])
                if bool(npz["has_ic"]):
                    pc_dataset.attrs["ic_amplitude"] = float(npz["ic_amplitude"])
                    pc_dataset.attrs["ic_integral"] = float(npz["ic_integral"])
                    pc_dataset.attrs["ic_centroid"] = float(npz["ic_centroid"])
                    pc_dataset.attrs["ic_multiplicity"] = float(npz["ic_multiplicity"])
                compiled += 1
                    
    print(f"Compiled {compiled} events for run_{run_number} -> {out_file_path}")

def main():
	
	trace_path = trace_path = Path("/home/danilo-marcato/Documents/Spyral/traces")
	workspace_path = Path("workspace/Pointcloud_assets/")
	output_path = Path("workspace/Pointcloud/")
	
	for run_number in runs:
		
		trace_file_path = trace_path / f"{form_run_string(run_number)}.h5"
		trace_reader = create_reader(trace_file_path, run_number)
		
		compile_run(run_number, trace_path, workspace_path, output_path)

runs = [46, 89, 95]
     
if __name__=="__main__":
    main()
