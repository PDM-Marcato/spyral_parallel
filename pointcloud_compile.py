from pathlib import Path
import numpy as np
import h5py as h5
import sys
from spyral.trace.trace_reader import create_reader

from config import TRACE_PATH, WORKSPACE_POINTCLOUD_ASSETS_PATH, WORKSPACE_POINTCLOUD_PATH, SPYRAL_MODIFICATION, FRIB_PARAMS
sys.path.append(SPYRAL_MODIFICATION)

from helper_modification import run_workspace_dir
    
def compile_run(run_number: int, trace_path: Path, workspace_path: Path,output_path: Path) -> None:
    
    ws_dir = run_workspace_dir(workspace_path, run_number)
    npz_files = sorted(ws_dir.glob("event_*.npz"))
    if not npz_files:
        print(f"No checkpoints found for run_{run_number}, skipping compile.")
        return
 
    # re-open the trace to recover first/last event metadata, same as before
    trace_file_path = trace_path / f"run_{run_number:04d}.h5"
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
    scalers = trace_reader.read_scalers()
    if scalers is not None:
        scalers.write_scalers(output_path / f"run_{run_number:04d}_scaler.parquet")
    else:
        print(f"Run {run_number} does not have scaler data")

    # --- gated IC count: sum the per-process files ---
    ic_within_mult_count = sum(
        int(p.read_text()) for p in ws_dir.glob("gated_ic_count_*.txt")
    )
    gated_ic_path = output_path / f"run_{run_number:04d}_gated_ic_scaler.txt"
    with open(gated_ic_path, "w") as f:
        f.write(f"IC counts with multiplicity <= {FRIB_PARAMS.ic_multiplicity}\n")
        f.write(f"{ic_within_mult_count}")
    print(f"Compiled {compiled} events for run_{run_number} -> {out_file_path}")


def main():
	
	run_number = int(sys.argv[1])
	
	WORKSPACE_POINTCLOUD_PATH.mkdir(parents=True, exist_ok=True)
		
	compile_run(run_number, TRACE_PATH, WORKSPACE_POINTCLOUD_ASSETS_PATH, WORKSPACE_POINTCLOUD_PATH)

     
if __name__=="__main__":
    main()
