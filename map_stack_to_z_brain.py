from pathlib import Path
import numpy as np
import nrrd
import datetime
import subprocess
import platform
import tempfile
import uuid
import h5py
import os
import re
import time


ants_bin_path = "/Users/koester_lab/install/bin"
ants_use_threads = 8


def convert_path_to_linux(path_name):

    if platform.system() == "Windows":
        path_name_linux = "/mnt/" + str(path_name)

        path_name_linux = path_name_linux.replace("\\", '/')
        path_name_linux = path_name_linux.replace(" ", '\\ ')

        path_name_linux = path_name_linux.replace("C:", 'c')
        path_name_linux = path_name_linux.replace("D:", 'd')
        path_name_linux = path_name_linux.replace("E:", 'e')
        path_name_linux = path_name_linux.replace("F:", 'f')
        path_name_linux = path_name_linux.replace("G:", 'g')
        path_name_linux = path_name_linux.replace("X:", 'x')
        path_name_linux = path_name_linux.replace("Y:", 'y')
        path_name_linux = path_name_linux.replace("Z:", 'z')

        return path_name_linux
    else:
        return str(path_name)


def run_linux_command(command_list, stdin_file=None, stdout_file=None):

    if platform.system() in ["Linux", "Darwin"]:
        # If we are in linux or mac os, we can directly execute the commands
        print(command_list)
        subprocess.run(command_list,
                       stdin=open(stdin_file) if stdin_file is not None else None,
                       stdout=open(stdout_file, "w") if stdout_file is not None else None)

    elif platform.system() == "Windows":
        registration_commands = f"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={ants_use_threads}\n" \
                                "export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS\n\n"

        registration_commands += ' '.join(command_list)

        if stdin_file is not None:
            registration_commands += f" < {stdin_file}"

        if stdout_file is not None:
            registration_commands += f" > {stdout_file}"

        registration_commands_path = Path(tempfile.gettempdir()) / f"{str(uuid.uuid4())}_linux_commands.sh"
        registration_commands_path_linux = convert_path_to_linux(registration_commands_path)

        commands_file = open(registration_commands_path, 'wb')
        commands_file.write((registration_commands + '\n').encode())
        commands_file.close()

        print("Executing linux script inside windows shell....")
        print(registration_commands)

        subprocess.run(["bash", '-c', registration_commands_path_linux])

        registration_commands_path.unlink()


def compute_volume_registration(source_stack_path, target_stack_path, registration_files_prefix):
    print(datetime.datetime.now(), "Running compute_volume_registration.", locals())

    if type(target_stack_path) is not list:
        target_stack_path = [target_stack_path]
    if type(source_stack_path) is not list:
        source_stack_path = [source_stack_path]

    source_stack_path_linux = [source_stack_path[i] for i in range(len(source_stack_path))]
    target_stack_path_linux = [target_stack_path[i] for i in range(len(target_stack_path))]
    registration_files_prefix_linux = registration_files_prefix

    # Some parameters from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5597853/pdf/gix056.pdf
    # others from https://github.com/ANTsX/ANTs/wiki/Anatomy-of-an-antsRegistration-call
    # https://github.com/ANTsX/ANTs/wiki/Tips-for-improving-registration-results
    registration_commands_list = [f"{ants_bin_path}/antsRegistration",
                                  "-v", "1",
                                  "-d", "3",
                                  "--float", "1",
                                  "--winsorize-image-intensities", "[0.005, 0.995]",
                                  "–-use-histogram-matching", "1",
                                  "-o", f"{registration_files_prefix_linux}_"]

    # Only do one for the initial moving transform, take always the first from the list
    registration_commands_list += ["--initial-moving-transform"]
    registration_commands_list += [f"[{target_stack_path_linux[0]},{source_stack_path_linux[0]},1]"]

    registration_commands_list += ["-t", "rigid[0.1]"]
    for i in range(len(target_stack_path)):
        registration_commands_list += ["-m", f"GC[{target_stack_path_linux[i]},{source_stack_path_linux[i]},1,32,Regular,0.25]"]
    registration_commands_list += ["-c", "[200x200x200x0,1e-8,10]",
                                   "-f", "12x8x4x2",
                                   "-s", "4x3x2x1"]

    registration_commands_list += ["-t", "Affine[0.1]"]
    for i in range(len(target_stack_path)):
        registration_commands_list += ["-m", f"GC[{target_stack_path_linux[i]},{source_stack_path_linux[i]},1,32,Regular,0.25]"]
    registration_commands_list += ["-c", "[200x200x200x0,1e-8,10]",
                                   "-f", "12x8x4x2",
                                   "-s", "4x3x2x1vox"]

    registration_commands_list += ["-t", "SyN[0.05,6,0]"]       # 0.1 for live to live, 0.05 fixed to fixed
    for i in range(len(target_stack_path)):
        registration_commands_list += ["-m", f"CC[{target_stack_path_linux[i]},{source_stack_path_linux[i]},1,2]"]
    registration_commands_list += ["-c", "[200x200x200x100,1e-7,10]", "-f", "12x8x4x2", "-s", "4x3x2x1"]

    print(registration_commands_list)

    run_linux_command(registration_commands_list)


def apply_volume_registration_to_stack(registration_files_prefix_list, source_stack_path,
                                       target_stack_path, output_stack_path, use_inverted_transforms=False,
                                       interpolation_method="linear"):
    print(datetime.datetime.now(), "Running apply_volume_registration_to_stack.", locals())

    source_stack_path_linux = source_stack_path
    target_stack_path_linux = target_stack_path
    output_stack_path_linux = output_stack_path

    # get the brightness of the border frame of the source image
    readdata, header = nrrd.read(source_stack_path)
    pad_out = np.percentile(np.r_[readdata[:5].flatten(), readdata[-5:].flatten(), readdata[:, :5].flatten(), readdata[:, -5:].flatten()], 5)

    registration_commands_list = [f"{ants_bin_path}/antsApplyTransforms",
                                  "--float",    # use single precision (as for the registration)
                                  "-v", "1",
                                  "-d", "3",
                                  "-f", f"{pad_out}",
                                  "-i", f"{source_stack_path_linux}",
                                  "-r", f"{target_stack_path_linux}",
                                  "-n", f"{interpolation_method}"]

    if use_inverted_transforms is False:
        for registration_files_prefix in registration_files_prefix_list:
            registration_files_prefix_linux = registration_files_prefix

            registration_commands_list += ["--transform", f"{registration_files_prefix_linux}_1Warp.nii.gz",
                                           "--transform", f"{registration_files_prefix_linux}_0GenericAffine.mat"]
    else:
        for registration_files_prefix in registration_files_prefix_list[::-1]:
            registration_files_prefix_linux = registration_files_prefix

            registration_commands_list += ["--transform", f"[{registration_files_prefix_linux}_0GenericAffine.mat, 1]",
                                           "--transform", f"{registration_files_prefix_linux}_1InverseWarp.nii.gz"]

    registration_commands_list += ["-o", f"{output_stack_path_linux}"]     # nii.gz???
    start = time.time()
    print("Running Command: " + str(registration_commands_list))

    run_linux_command(registration_commands_list)
    print("Done registration_commands_list " + str(time.time()-start))

    # ANTS makes it a 32bit float tiff, convert it back to 16bit uint

    start = time.time()
    readdata, header = nrrd.read(str(output_stack_path))
    print("Done Reading output Tiff from Ant " + str(time.time()-start))

    start = time.time()
    readdata = readdata.astype(np.uint8)
    print(type(readdata))
    print("Done Converting F32 to UINT8 " + str(time.time()-start))

    #header["encoding"] = 'gzip'
    #header["type"] = 'uint16'

    #start = time.time()
    #nrrd.write(str(output_stack_path), readdata, header)
    #print("Done nrrd.write readdata to " + str(output_stack_path) + ": " + str(time.time()-start))
    return readdata


def convert_hdf5_file_to_nrrd(filepath):

    f_hdf5 = h5py.File(filepath, 'r')

    rootpath = filepath.parent
    stackname = filepath.stem

    dx = 0.7188675 * 2
    dy = 0.7188675 * 2
    dz = 9.9992850

    stack = np.array(f_hdf5["ZYX"], dtype=np.uint8).transpose((2, 1, 0))  # or just .T - that's the same

    # print(stack.shape)
    # sd
    options = {'type': 'uint8',
               'encoding': 'raw',
               'endian': 'big',
               'dimension': 3,
               'sizes': stack.shape,
               'space dimension': 3,
               'space directions': [[dx, 0, 0], [0, dy, 0], [0, 0, dz]],
               'space units': ['microns', 'microns', 'microns']}

    nrrd.write(str(rootpath / f"{stackname}.nrrd"), stack, options)
    # nrrd.write(str(os.sep.join([rootpath, f'{stackname}.nrrd'])), stack, options)

    f_hdf5.close()


if __name__ == '__main__':

    # import socket
    # computer_name = socket.gethostname()
    # if computer_name == ...:
    #   time_averaged_filelist = ...
    #   data_folder_for_walk = ...
    # elif computer_name == ...:
    #   time_averaged_filelist = ...
    #   data_folder_for_walk = ...

    time_averaged_filelist = open(r'/Volumes/User-Data/Armin/new_registration/Time_Averaged_List.txt', 'a+')  # only first time; then 'a+'
    reference_brain_path = Path("/Volumes/User-Data/Armin")

    for folder, subfolders, files in os.walk(r'/Volumes/User-Data/Armin/new_registration'):
        for file in files:
            if re.search(r'averaged.hdf5', file):  # only apply it to the time averaged files

                filepath_time_averaged = Path(os.sep.join([folder, file]))
                filestem = file.split('.')[-2]
                # to do: check whether file is already in the Time_Averaged_List and perform the code only if not
                time_averaged_filelist.write(f'{filestem}\n')

                print(f'Currently working on {filestem}')

                convert_hdf5_file_to_nrrd(filepath_time_averaged)
                root_path = filepath_time_averaged.parent  # sollte das gleiche sein wie folder, nur als Path
                stack_name = filepath_time_averaged.stem   # sollte das gleiche sein wie filestem, nur als Path

                compute_volume_registration(source_stack_path=root_path / f'{stack_name}.nrrd',
                                            target_stack_path=reference_brain_path / 'Elavl3-H2BRFP_z_brain.nrrd',
                                            registration_files_prefix=root_path / 'zbrain_registered' / f'{stack_name}_to_Elavl3-H2BRFP')
                # compute_volume_registration(
                #    source_stack_path=str(os.sep.join([root_path, f'{stack_name}.nrrd'])),
                #    target_stack_path=str(os.sep.join([reference_brain_path, 'Elavl3-H2BRFP.nrrd'])),
                #    registration_files_prefix=str(os.sep.join([root_path, f'{stack_name}_to_Elavl3-H2BRFP'])))

                apply_volume_registration_to_stack(
                    registration_files_prefix_list=[root_path / 'zbrain_registered' / f'{stack_name}_to_Elavl3-H2BRFP'],
                    source_stack_path=root_path / f"{stack_name}.nrrd",
                    target_stack_path=reference_brain_path / 'Elavl3-H2BRFP_z_brain.nrrd',
                    output_stack_path=root_path / 'zbrain_registered' / f'{stack_name}_registered.nrrd')

    time_averaged_filelist.close()
