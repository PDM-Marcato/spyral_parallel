from pathlib import Path
import numpy as np
import h5py as h5
import re
import sys

from spyral.core.run_stacks import form_run_string
from spyral.core.point_cloud import PointCloud
from spyral.core.clusterize import cleanup_clusters

from config import (
    WORKSPACE_CLUSTER_ASSETS_PATH,
    WORKSPACE_CLUSTER_PATH,
    WORKSPACE_POINTCLOUD_PATH,
    REV_PARAMS,
    SPYRAL_MODIFICATION,
)
sys.path.append(SPYRAL_MODIFICATION)
from clusterize_modification import New_Clustering_Method
from helper_modification import run_workspace_dir, create_event_file


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
	run_number = int(sys.argv[3])

	workspace_cluster_path = WORKSPACE_CLUSTER_ASSETS_PATH
	
	
	point_file_path = WORKSPACE_POINTCLOUD_PATH / f"run_{run_number:04d}_pc.h5"
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
		if idx != 3642:
			continue
		print(idx)
		
		file_name = create_event_file(WORKSPACE_CLUSTER_ASSETS_PATH, run_number, idx)
		cloud_name = f"cloud_{idx}"
		cloud_data = cloud_group[cloud_name]
		cloud = PointCloud(idx, cloud_data[:].copy())
			
		if len(cloud)< REV_PARAMS.min_cloud_size:
			continue
			
		cleaned = New_Clustering_Method(cloud, REV_PARAMS)
		
		save_cluster_checkpoint(file_name, idx, cleaned, dict(cloud_data.attrs))
			
if __name__=="__main__":
    main()
