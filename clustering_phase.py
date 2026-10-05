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

def save_cluster_checkpoint(file_name: Path, idx: int, cleaned, cloud_attrs) -> None:
    """Save one event's cleaned clusters to an npz checkpoint."""
    has_ic = "ic_amplitude" in cloud_attrs

    kwargs = dict(
        event_number=idx,
        nclusters=len(cleaned),
        orig_run=cloud_attrs["orig_run"],
        orig_event=cloud_attrs["orig_event"],
        has_ic=has_ic,
    )
    if has_ic:
        kwargs.update(
            ic_amplitude=cloud_attrs["ic_amplitude"],
            ic_integral=cloud_attrs["ic_integral"],
            ic_centroid=cloud_attrs["ic_centroid"],
            ic_multiplicity=cloud_attrs["ic_multiplicity"],
        )

    # One set of keys per cluster (npz can't hold ragged arrays in one key)
    for cidx, cluster in enumerate(cleaned):
        kwargs[f"cluster_{cidx}_data"] = cluster.data
        kwargs[f"cluster_{cidx}_label"] = cluster.label
        kwargs[f"cluster_{cidx}_direction"] = cluster.direction.value

    np.savez(file_name, **kwargs)


def main():
	
	process_id = int(sys.argv[1])
	n_processes = int(sys.argv[2])
	
	workspace_path = Path("workspace/")

	workspace_cluster_path = Path("workspace/Cluster_assets/")

	pointcloud_path = workspace_path / "Pointcloud"

	runs = [46,89, 95]
	rng = np.random.default_rng()

	for run_number in runs:
		
		point_file_path = pointcloud_path / f"{form_run_string(run_number)}_pc.h5"
		point_file = h5.File(point_file_path, 'r')
		
		cloud_group: h5.Group = point_file.get('cloud')
		min_event = cloud_group.attrs['min_event']
		max_event = cloud_group.attrs['max_event']
		idx_events = []
		for idx in range(min_event, max_event+1):
			event_name = event_name = f"cloud_{idx}"
			if event_name in cloud_group:
				idx_events.append(idx)
		
		for i, idx in enumerate(idx_events):
			
			if i % n_processes != process_id:
				continue
			
			file_name = create_event_file(workspace_cluster_path, run_number, idx)

			cloud_name = f"cloud_{idx}"
			cloud_data = cloud_group[cloud_name]
			cloud = PointCloud(idx, cloud_data[:].copy())
			
			if len(cloud)< 30:
				continue
			
			joined, labels = New_Clustering_Method(cloud)

			cleaned, _ = cleanup_clusters(joined, cluster_params, labels)
			
			save_cluster_checkpoint(file_name, idx, cleaned, dict(cloud_data.attrs))
			
if __name__=="__main__":
    main()
