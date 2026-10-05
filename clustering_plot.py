from pathlib import Path
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import random

from spyral.core.run_stacks import form_run_string
from spyral.core.cluster import Cluster, Direction

DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Utility for syncing plot colors
def get_color(value: int) -> str:
    color_index = value
    if color_index >= len(DEFAULT_COLORS):
        color_index = color_index % len(DEFAULT_COLORS)
    elif color_index == -1:
        return "black"
    return DEFAULT_COLORS[color_index]
    
def Plot_Clusters(clusters, title = None):
    fig, axs = plt.subplot_mosaic(
    """
    AAB
    """,
    per_subplot_kw={
        "A": {
            "projection": "3d", 
            "box_aspect": (2,1,1),
            "aspect": "equalxy"
        }
    },
    figsize=(15.0, 5.0),
    constrained_layout=True
    )
    for cluster in clusters:
        axs["A"].scatter(cluster.data[:, 2], cluster.data[:, 0], cluster.data[:, 1], c = get_color(cluster.label), s=3, label=f"Cluster {cluster.label}")
        axs["B"].scatter(cluster.data[:, 0], cluster.data[:, 1], c = get_color(cluster.label), s=3, label = f"Cluster {cluster.label}")
    
    axs["A"].set_xlim3d(0., 1000.0)
    axs["A"].set_xlabel("Z(mm)")
    axs["A"].set_ylim3d(-300.0, 300.0)
    axs["A"].set_ylabel("X(mm)")
    axs["A"].set_zlim3d(-300.0, 300.0)
    axs["A"].set_zlabel("Y(mm)")
    
    axs["B"].set_xlim(-300.0, 300.0)
    axs["B"].set_xlabel("X(mm)")
    axs["B"].set_ylim(-300.0, 300.0)
    axs["B"].set_ylabel("Y(mm)")
    axs["B"].grid()
    axs["B"].legend()
    if title is not None:
        fig.suptitle(title)

    return fig

def Make_Cluster(idx, event):
    clusters = []
    nclusters =  event.attrs["nclusters"]
    
    for cidx in range(0, nclusters):
        local_cluster: h5.Group | None = None
        cluster_name = f"cluster_{cidx}"
        if cluster_name not in event:  # type: ignore
            continue
        else:
            local_cluster = event[cluster_name]  # type: ignore
    
        cluster = Cluster(
                        idx,
                        local_cluster.attrs["label"],  # type: ignore
                        Direction(local_cluster.attrs["direction"]), # type: ignore
                        local_cluster["cloud"][:].copy(),  # type: ignore
                )
        clusters.append(cluster)
    return clusters

def main():

	workspace_path = Path("workspace/")

	cluster_path = workspace_path / "Cluster" # This may change if you add custom phases!

	run_number = 46
	cluster_file_path = cluster_path / f"{form_run_string(run_number)}_cluster.h5"
	cluster_file = h5.File(cluster_file_path, "r")
	cluster_group = cluster_file["cluster"]
	min_event = cluster_group.attrs["min_event"]
	max_event = cluster_group.attrs["max_event"]

	idx_events = []
	for idx in range(max_event+1):
		event_name = f"event_{idx}"
		if event_name in cluster_group:
			idx_events.append(idx)

	print("Total number of events analized: ",len(idx_events))
	
	while True:

		user_input = input("\nEnter event number (or 'q' to quit): ").strip()

		if user_input.lower() == "q": 
			print("Exiting.")
			break
			
		try:
			idx = int(user_input)
		except ValueError:
			print("Please enter a valid event number.")
			continue

		try:
			event_name = f"event_{idx}"

			if event_name not in cluster_group:
				print(f"Event {idx} could not be read.")
				continue
				
			event = cluster_group[event_name]
			
			clusters = Make_Cluster(idx, event)

			Plot_Clusters(clusters)

		except Exception as e:
			print(f"Error reading event {idx}: {e}")

if __name__ == "__main__":
    main()
