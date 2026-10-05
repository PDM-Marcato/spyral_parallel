import h5py as h5
import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path

from spyral.trace.trace_reader import create_reader

from spyral.core.pad_map import PadMap
from spyral import PadParameters, GetParameters, FribParameters, DetectorParameters, DEFAULT_MAP

from spyral.core.point_cloud import point_cloud_from_get, calibrate_point_cloud_z, sort_point_cloud_in_z

def Plot_PC(cloud):
    fig, axs = plt.subplot_mosaic(
    """
    AAB
    """,
    per_subplot_kw={
        "A": {
            "projection": "3d", 
            "box_aspect": (1,0.5,0.5),
            "aspect": "equalxy"
        }
    },
    figsize=(15.0, 5.0),
    constrained_layout=True
    )
    s3d = axs["A"].scatter(cloud.data[:, 2], cloud.data[:, 0], cloud.data[:, 1], c=cloud.data[:, 3], s=3, label="Pointcloud")
    axs["A"].set_xlim3d(0., 1000.0)
    axs["A"].set_xlabel("Z(mm)")
    axs["A"].set_ylim3d(-300.0, 300.0)
    axs["A"].set_ylabel("X(mm)")
    axs["A"].set_zlim3d(-300.0, 300.0)
    axs["A"].set_zlabel("Y(mm)")
    s2d = axs["B"].scatter(cloud.data[:, 0], cloud.data[:, 1], c=cloud.data[:, 3], s=3)
    axs["B"].set_xlim(-300.0, 300.0)
    axs["B"].set_xlabel("Z(mm)")
    axs["B"].set_ylim(-300.0, 300.0)
    axs["B"].set_ylabel("Y(mm)")
    axs["B"].grid()
    fig.legend()

    plt.show()
    

def Plot_Signal(idx, event, good_peak):

    ic_signal = event.frib.get_ic_trace().trace
    si_signal = event.frib.get_si_trace().trace
    triggering_peak = event.frib.get_triggering_ic_peak(frib_params)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.axvline(
        1120,
        color="red",
        label="Trigger Reference"
    )

    ax.plot(
        ic_signal,
        label="IC signal"
    )

    ax.plot(
        si_signal,
        label="SI signal"
    )
    if good_peak is not None:
        ax.axvline(
            good_peak.centroid,
            color="blue",
            label="Reactio Peak"
        )
    if triggering_peak is not None:
        ax.axvline(
            triggering_peak.centroid,
            color="green",
            label="Trigger"
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")
    ax.set_title(f"Event {idx}")

    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()


def main():

	run_number = int(sys.argv[1])
	
	file_path = f"/home/danilo-marcato/Documents/pipeline/workspace/Pointcloud/run_{run_number:04d}_pc.h5"
	trace_path = Path("/home/danilo-marcato/Documents/Spyral/traces")
	trace_file_path = trace_path / f"run_{run_number:04d}.h5"
	
	point_file = h5.File(file_path, 'r')
	
	cloud_group = point_file.get("cloud")
	
	min_default = cloud_group.attrs["min_event"]
	max_default = cloud_group.attrs["max_event"]
	print(f"Event Range: {min_default} -> {max_default}")
	
	trace_reader = create_reader(trace_file_path,run_number)
	
	attributes = [
		"orig_run",
		"orig_event",
		"ic_amplitude",
		"ic_integral",
		"ic_centroid",
		"ic_multiplicity",
		]
	
	for idx in range(max_default+1):
		
		event_name = f"cloud_{idx}"
		if event_name in cloud_group:
			missing = [
				attr
				for attr in attributes
				if attr not in event_data.attrs
				]
				
			if missing:
				print(f"Parallel: {event_name} is missing: {missing}")
	

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
			event = trace_reader.read_event(idx, get_params, frib_params, rng)

			if event is None:
				print(f"Event {idx} could not be read.")
				continue
			
			ic_peak = None
            
			corr_ic = 0.0
			trigger_peak = event.frib.get_triggering_ic_peak(frib_params)
            
			if trigger_peak is None:
				print("No Trigger.")
				continue
				
            
			corr_ic += (trigger_peak.centroid-1120)*det_params.get_frequency/12.5
			reaction_peak = event.frib.get_good_ic_peak(frib_params)
			good_peak = None
			if reaction_peak is not None:
				good_ic_count, good_peak = reaction_peak
				corr_ic += event.frib.correct_ic_time(good_peak, frib_params, det_params.get_frequency)
				ic_peak = good_peak

			else:
				ic_peak = trigger_peak
			
			print(ic_peak)
			ic_mult = event.frib.get_ic_multiplicity(frib_params)

			Plot_Signal(idx, event, good_peak)
			
			cloud = point_cloud_from_get(event.get_pads, pad_map)
			calibrate_point_cloud_z(cloud, det_params, corr_ic, None)
			sort_point_cloud_in_z(cloud)
			
			Plot_PC(cloud)

		except Exception as e:
			print(f"Error reading event {idx}: {e}")

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

if __name__ == "__main__":
    main()
