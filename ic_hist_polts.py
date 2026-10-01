import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main():
	runs = [46, 89,95]
	with PdfPages("pdf/IC_Amplitude.pdf") as pdf:
		for run_number in runs:
			file_path = f"/home/danilo-marcato/Documents/pipeline/workspace/Pointcloud/run_00{run_number}_pc.h5"
			
			point_file = h5.File(file_path, 'r')
			
			cloud_group = point_file.get("cloud")
			
			min_event = cloud_group.attrs["min_event"]
			max_event = cloud_group.attrs["max_event"]
			print(f"Range of Events: {min_event} -> {max_event}")
			
			idx_event = []
			
			for idx in range(max_event+1):
				
				event_name = f"cloud_{idx}"
				if event_name in cloud_group:
					idx_event.append(idx)
				
					
			print(f"Number Events: {len(idx_event)}")
				
			attributes = [
				"orig_run",
				"orig_event",
				"ic_amplitude",
				"ic_integral",
				"ic_centroid",
				"ic_multiplicity",
				]
			
			ic_amplitude = []
			for idx in idx_event:

				event_name = f"cloud_{idx}"
				event_data = cloud_group[event_name]

				missing = [
					attr
					for attr in attributes
					if attr not in event_data.attrs
					]

				if missing:
					print(f"Parallel: {event_name} is missing: {missing}")
				else:
					ic_amplitude.append(event_data.attrs["ic_amplitude"])
			
			plt.figure(figsize=(8, 5))

			plt.hist(ic_amplitude, bins=100, range=(0, 1000))
			plt.xlim(0, 1000)

			plt.xlabel("IC Amplitude")
			plt.ylabel("Counts")
			plt.title(f"IC Amplitude Distribution run {run_number}: {len(ic_amplitude)} points.")
			pdf.savefig()
			plt.show()
			plt.close()
			


if __name__=="__main__":
    main()
