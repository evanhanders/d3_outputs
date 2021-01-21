"""
Logic for merging files from dedalus simulations from d3_outputs; largely copied from dedalus/tools/post.py
"""
import glob
import pathlib
import h5py
import numpy as np
from mpi4py import MPI

from dedalus.tools.general import natural_sort

import logging
logger = logging.getLogger(__name__.split('.')[-1])

MPI_RANK = MPI.COMM_WORLD.rank
MPI_SIZE = MPI.COMM_WORLD.size

from dedalus.tools.post import get_assigned_sets

def merge_distributed_set(set_path, cleanup=False):
    """
    Merge a distributed analysis set from a FileHandler.

    Parameters
    ----------
    set_path : str of pathlib.Path
        Path to distributed analysis set folder
    cleanup : bool, optional
        Delete distributed files after merging (default: False)

    """
    set_path = pathlib.Path(set_path)
    logger.info("Merging set {}".format(set_path))

    set_stem = set_path.stem
    proc_paths = set_path.glob("{}_p*.h5".format(set_stem))
    proc_paths = natural_sort(proc_paths)
    joint_path = set_path.parent.joinpath("{}.h5".format(set_stem))

    # Create joint file, overwriting if it already exists
    with h5py.File(str(joint_path), mode='w') as joint_file:
        # Setup joint file based on first process file (arbitrary)
        merge_setup(joint_file, proc_paths)
        # Merge data from all process files
        for proc_path in proc_paths:
            merge_data(joint_file, proc_path)
    # Cleanup after completed merge, if directed
    if cleanup:
        for proc_path in proc_paths:
            proc_path.unlink()
        set_path.rmdir()


def merge_setup(joint_file, proc_paths):
    """
    Merge HDF5 setup from part of a distributed analysis set into a joint file.

    Parameters
    ----------
    joint_file : HDF5 file
        Joint file
    proc_path : str or pathlib.Path
        Path to part of a distributed analysis set

    """
    proc_path = proc_paths[0]
    proc_path = pathlib.Path(proc_path)

    min_process = 0
    with h5py.File(str(proc_path), mode='r') as proc_file:
        min_process = proc_file.attrs['min_process'][()]
    if min_process != 0:
        proc_path = proc_paths[min_process]
    logger.info("Merging setup from {}".format(proc_path))
    with h5py.File(str(proc_path), mode='r') as proc_file:
        # File metadata
        try:
            joint_file.attrs['set_number'] = proc_file.attrs['set_number']
        except KeyError:
            joint_file.attrs['set_number'] = proc_file.attrs['file_number']
        joint_file.attrs['handler_name'] = proc_file.attrs['handler_name']
        try:
            joint_file.attrs['writes'] = writes = proc_file.attrs['writes']
        except KeyError:
            joint_file.attrs['writes'] = writes = len(proc_file['scales']['write_number'])
        # Copy scales (distributed files all have global scales)
        proc_file.copy('scales', joint_file)
        # Tasks
        joint_tasks = joint_file.create_group('tasks')
        proc_tasks = proc_file['tasks']

        for taskname in proc_tasks:
            print('got here', taskname)
            # Setup dataset with automatic chunking
            proc_dset = proc_tasks[taskname]
            spatial_shape = proc_dset.attrs['global_shape']
            joint_shape = (writes,) + tuple(spatial_shape)
            print('about to create dataset')
            joint_dset = joint_tasks.create_dataset(name=proc_dset.name,
                                                    shape=joint_shape,
                                                    dtype=proc_dset.dtype,
                                                    chunks=True)
            if 'task_type' in proc_dset.attrs.keys():
                if proc_dset.attrs['task_type'] == 'sum':
                    joint_dset[:] = 0
            # Dataset metadata
            joint_dset.attrs['task_number'] = proc_dset.attrs['task_number']
            joint_dset.attrs['constant'] = proc_dset.attrs['constant']
            joint_dset.attrs['grid_space'] = proc_dset.attrs['grid_space']
            joint_dset.attrs['scales'] = proc_dset.attrs['scales']

def merge_data(joint_file, proc_path):
    """
    Merge data from part of a distributed analysis set into a joint file.

    Parameters
    ----------
    joint_file : HDF5 file
        Joint file
    proc_path : str or pathlib.Path
        Path to part of a distributed analysis set

    """
    proc_path = pathlib.Path(proc_path)
    logger.info("Merging data from {}".format(proc_path))

    with h5py.File(str(proc_path), mode='r') as proc_file:
        for taskname in proc_file['tasks']:
            joint_dset = joint_file['tasks'][taskname]
            proc_dset = proc_file['tasks'][taskname]
            # Merge across spatial distribution
            start = proc_dset.attrs['start']
            count = proc_dset.attrs['count']
            spatial_slices = tuple(slice(s, s+c) for (s,c) in zip(start, count))
            # Merge maintains same set of writes
            slices = (slice(None),) + spatial_slices
            if 'task_type' in proc_dset.attrs.keys():
                if proc_dset.attrs['task_type'] == 'slice':
                    joint_dset[slices] = proc_dset[:]
                elif proc_dset.attrs['task_type'] == 'sum':
                    joint_dset[slices] += proc_dset[:]

def merge_analysis(base_path, cleanup=False):
    """
    Merge distributed analysis sets from a FileHandler.

    Parameters
    ----------
    base_path : str or pathlib.Path
        Base path of FileHandler output
    cleanup : bool, optional
        Delete distributed files after merging (default: False)

    Notes
    -----
    This function is parallelized over sets, and so can be effectively
    parallelized up to the number of distributed sets.

    """
    set_path = pathlib.Path(base_path)
    logger.info("Merging files from {}".format(base_path))

    set_paths = get_assigned_sets(base_path, distributed=True)
    for set_path in set_paths:
        merge_distributed_set(set_path, cleanup=cleanup)
