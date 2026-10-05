from pathlib import Path
import numpy as np
import h5py as h5
import re
import sys

from spyral.core.run_stacks import form_run_string
from spyral.core.point_cloud import PointCloud
from spyral.core.clusterize import form_clusters, join_clusters, cleanup_clusters
from spyral import (
    ClusterParameters,
    HdbscanParameters,
    TripclustParameters,
    OverlapJoinParameters,
    ContinuityJoinParameters,
)

sys.path.append('/home/danilo-marcato/Documents/Spyral_Modification')
from clusterize_modification import New_Clustering_Method

cluster_params = ClusterParameters(
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


def run_workspace_dir(workspace_path: Path, run_number: int) -> Path:
    """Scratch folder that holds one .npz per processed event for this run."""
    d = workspace_path / f"run_{run_number}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def create_event_file(workspace_path, run_number, idx):
    file_name = run_workspace_dir(workspace_path, run_number) / f"event_{idx:06d}.npz"
    return file_name

def compile_cluster_run(run_number: int, point_path: Path,
                        workspace_path: Path, output_path: Path) -> None:
    """Assemble per-event cluster checkpoints into the final cluster h5."""
    ws_dir = run_workspace_dir(workspace_path, run_number)
    npz_files = sorted(ws_dir.glob("event_*.npz"))
    if not npz_files:
        print(f"No cluster checkpoints found for run_{run_number}, skipping compile.")
        return

    # min/max event come from the point cloud file, so no trace reader is needed
    with h5.File(point_path, "r") as point_file:
        pc_group = point_file["cloud"]
        min_event = int(pc_group.attrs["min_event"])
        max_event = int(pc_group.attrs["max_event"])

    out_file_path = output_path / f"run_{run_number:04d}_cluster.h5"
    compiled = 0
    with h5.File(out_file_path, "w") as cluster_file:
        cluster_group = cluster_file.create_group("cluster")
        cluster_group.attrs["min_event"] = min_event
        cluster_group.attrs["max_event"] = max_event

        for npz_path in npz_files:
            with np.load(npz_path) as npz:
                if "event_number" not in npz.files:
                    continue

                event_number = int(npz["event_number"])
                nclusters = int(npz["nclusters"])

                ev = cluster_group.create_group(f"event_{event_number}")
                ev.attrs["nclusters"] = nclusters
                ev.attrs["orig_run"] = int(npz["orig_run"])
                ev.attrs["orig_event"] = int(npz["orig_event"])
                if bool(npz["has_ic"]):
                    ev.attrs["ic_amplitude"] = float(npz["ic_amplitude"])
                    ev.attrs["ic_integral"] = float(npz["ic_integral"])
                    ev.attrs["ic_centroid"] = float(npz["ic_centroid"])
                    ev.attrs["ic_multiplicity"] = float(npz["ic_multiplicity"])

                for cidx in range(nclusters):
                    local = ev.create_group(f"cluster_{cidx}")
                    local.attrs["label"] = npz[f"cluster_{cidx}_label"].item()
                    local.attrs["direction"] = npz[f"cluster_{cidx}_direction"].item()
                    local.create_dataset("cloud", data=npz[f"cluster_{cidx}_data"])
                compiled += 1

    #print(f"Compiled {compiled} events for run_{run_number} -> {out_file_path}")
    
def main():
	
	run_number = int(sys.argv[1])
	
	workspace_path = Path("workspace/")

	workspace_cluster_path = Path("workspace/Cluster_assets/")
	output_path = Path("workspace/Cluster/")
	pointcloud_path = workspace_path / "Pointcloud" # this may change if you add custom phases!

	point_file_path = pointcloud_path / f"{form_run_string(run_number)}_pc.h5"
		
	compile_cluster_run(run_number, point_file_path, workspace_cluster_path, output_path)
           
if __name__=="__main__":
    main()
