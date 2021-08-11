from pathlib import Path
import numpy as np
import nrrd
import h5py
import re
import os
from map_stack_to_z_brain import apply_volume_registration_to_stack
import time


# Optimally, this should be an environmental variable
reference_brain_path = Path("/Volumes/User-Data/Armin")

# resolutions of the size-reduced motion aligned lightsheet stack
dx = 0.7188675 * 2
dy = 0.7188675 * 2
dz = 9.9992850

for folder, subfolders, files in os.walk(r'/Volumes/User-Data/Armin/new_registration/control/stimulus'):
    for file in files:
        if re.search(r'aligned.hdf5', file):  # only apply it to the caiman aligned files

            filepath_aligned = Path(os.sep.join([folder, file]))
            root_path = filepath_aligned.parent  # sollte das gleiche sein wie folder, nur als Path
            stack_name = filepath_aligned.stem

            with open(r'/Volumes/User-Data/Armin/new_registration/control/stimulus/ANTs_List.txt', 'r') as ants_filelist:
                text = ants_filelist.read()

            if re.search(stack_name, text):
                print(f'\n{stack_name} is already registered\n')
            else:
                print(f'\n\nCurrently working on {stack_name}\n')

                with open(r'/Volumes/User-Data/Armin/new_registration/control/stimulus/ANTs_List.txt',
                          'a+') as ants_filelist:
                    ants_filelist.write(f'{stack_name}\n')

                f_hdf5_motion_aligned = h5py.File(filepath_aligned, 'r')  # read the caiman aligned hdf5 file

                t_size = f_hdf5_motion_aligned["TZYX"].shape[0]  # number of time steps in that file

				# Create a new hdf5 file with same number of time steps
				filepath_output = Path(os.sep.join([folder, 'zbrain_registered', 'over_time', f'{stack_name}_registered.hdf5']))
				f_hdf5_z_brain_registered = h5py.File(filepath_output, 'w')
				dset = f_hdf5_z_brain_registered.create_dataset("TZYX", (t_size, 138, 1406, 621), dtype='uint8')

				# convert each time point in the hdf5 file to XYZ and to a temporary nrrd file
				for t in range(t_size):
					print(f'\ntime step {t}')

					start = time.time()
					nparray = np.array(f_hdf5_motion_aligned["TZYX"][t], dtype=np.uint8)
					print("Loading \"stack\" Timestep: " + str(time.time()-start))
					start = time.time()
					stack = nparray.transpose(2, 1, 0)
					print("Transposing nparray to stack: " + str(time.time()-start))

					options = {'type': 'uint8',
							   'encoding': 'raw',
							   'endian': 'big',
							   'dimension': 3,
							   'sizes': stack.shape,
							   'space dimension': 3,
							   'space directions': [[dx, 0, 0], [0, dy, 0], [0, 0, dz]],
							   'space units': ['microns', 'microns', 'microns']}

					start = time.time()
					nrrd.write("temp_stack.nrrd", stack, options)  # save only as a temporary file
					print("Converting Numpy Array to temp_stack NRRD " + str(time.time()-start))

					start = time.time()

					readdata = apply_volume_registration_to_stack(
						registration_files_prefix_list=[root_path / 'zbrain_registered' /
														f'{stack_name}_time_averaged_to_Elavl3-H2BRFP'],
						source_stack_path='temp_stack.nrrd',
						target_stack_path=reference_brain_path / 'Elavl3-H2BRFP_z_brain.nrrd.nrrd',
						output_stack_path='temp_stack_registered.nrrd')

					print("apply_volume_registration_to_stack time " + str(time.time()-start))


					#start = time.time()
					#nrrd.write("temp_stack.nrrd", stack, options)  # save only as a temporary file
					#print("nrrd.write temp stack " + str(time.time()-start))

					# Read the registered data
					#readdata, header = nrrd.read('temp_stack_registered.nrrd')
					#readdata = readdata.astype(np.uint8)

					# Place it at the right place into the hdf5 file created above and bring it back to ZYX
					
					start = time.time()
					dset[t, :, :, :] = readdata.transpose((2, 1, 0))
					print("transpose data " + str(time.time()-start))

				f_hdf5_motion_aligned.close()
				f_hdf5_z_brain_registered.close()
