import h5py as h5
import numpy as np
import polars as pl

from config import (
	TRACE_PATH,
	WORKSPACE_PATH,
	WORKSPACE_POINTCLOUD_ASSETS_PATH,
	WORKSPACE_POINTCLOUD_PATH,
	WORKSPACE_CLUSTER_ASSETS_PATH,
	WORKSPACE_CLUSTER_PATH,
	WORKSPACE_ESTIMATION_ASSETS_PATH,
	WORKSPACE_ESTIMATION_PATH,
)

def main():
	
	print(f"=======================================================================================================================")
	for run_number in runs:
		
		point_file_path = WORKSPACE_POINTCLOUD_PATH / f"run_{run_number:04d}_pc.h5"
		point_file = h5.File(point_file_path, 'r')
		
		cluster_file_path = WORKSPACE_CLUSTER_PATH / f"run_{run_number:04d}_cluster.h5"
		cluster_file = h5.File(cluster_file_path, "r")
		
		run_path = WORKSPACE_ESTIMATION_PATH / f"run_{run_number:04d}.parquet"
		
		cluster_group = cluster_file["cluster"]
		cloud_group =  point_file.get('cloud')
		
		min_event = cloud_group.attrs['min_event']
		max_event = cloud_group.attrs['max_event']
		
		print(f"RUN_{run_number}:")
		print(f"First event: {min_event} Last event: {max_event}")
		
		cloud_count = 0
		for idx in range(max_event+1):
			event_name = f"cloud_{idx}"
			if event_name in cloud_group:
				cloud_count += 1
		
		print(f"Number of Event Analyzed: {cloud_count}")
		
		event_cluster += 0
		tracks_count = 0
		for idx in range(max_event+1):
			event_name = f"event_{idx}"
			if event_name not in cluster_group:
				event_cluster += 1
				event_group = cluster_group[event_name]
				tracks_count = event_group.attrs["nclusters"]
				
		print(f"Number of Events Clusterized: {event_cluster}")
		print(f"Total Number of Tracks: {tracks_count}")
		
		ic_min_val = -1
		ic_max_val = 4096.0
		df = pl.scan_parquet(run_path)
		df = df.filter((pl.col('ic_amplitude') > ic_min_val) & (pl.col('ic_amplitude') < ic_max_val))
		df = df.collect()
		print(f"Total Number of Tracks Estimated: {df.shape[0]}")
		print(f"=======================================================================================================================")
