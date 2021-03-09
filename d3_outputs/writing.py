import os
import hashlib
import pathlib
from collections import OrderedDict

import numpy as np
import h5py
from mpi4py import MPI

from dedalus.tools.parallel import Sync 
from dedalus.core.evaluator import FileHandler

from dedalus.tools.config import config
FILEHANDLER_MODE_DEFAULT = config['analysis'].get('FILEHANDLER_MODE_DEFAULT')
FILEHANDLER_PARALLEL_DEFAULT = config['analysis'].getboolean('FILEHANDLER_PARALLEL_DEFAULT')
FILEHANDLER_TOUCH_TMPFILE = config['analysis'].getboolean('FILEHANDLER_TOUCH_TMPFILE')

class d3FileHandler(FileHandler):

    def __init__(self, solver, filename, **kwargs):
        """ 
        An abstract class for writing to .h5 files, inspired by classic dedalus file handlers

        # Arguments
            solver (InitialValueSolver) :
                The dedalus solver used for the IVP
        """
        self.evaluator = solver.evaluator
        super(d3FileHandler, self).__init__(filename, self.evaluator.dist, self.evaluator.vars, **kwargs)
        self.task_dict = OrderedDict()
        self.local_file = None
        self.min_process = np.array([self.dist.comm_cart.rank,])

        self.evaluator.add_handler(self)

    def add_task(self, op, extra_op=None, extra_op_comm=False, scales=None, **kw):
        if scales is None:
            scales = op.evaluate().domain.dealias
        super(d3FileHandler, self).add_task(op, scales=scales, **kw)
        task = self.tasks[-1]
        task['extra_op'] = extra_op
        task['extra_op_comm'] = extra_op_comm
        self.task_dict[task['name']] = task

    def setup_file(self, file):
        """Largely copied from dedalus/core/evaluator.py, with addition of extra_op logic"""
        # Skip spatial scales for now
        skip_spatial_scales = True
        dist = self.dist

        # Metadeta
        file.attrs['set_number'] = self.set_num
        file.attrs['handler_name'] = self.base_path.stem
        file.attrs['writes'] = self.file_write_num
        if not self.parallel:
            file.attrs['mpi_rank'] = dist.comm_cart.rank
            file.attrs['mpi_size'] = dist.comm_cart.size
            file.attrs['min_process'] = self.min_process[0]

        # Scales
        scale_group = file.create_group('scales')
        # Start time scales with shape=(0,) to chunk across writes
        scale_group.create_dataset(name='sim_time', shape=(0,), maxshape=(None,), dtype=np.float64)
        scale_group.create_dataset(name='timestep', shape=(0,), maxshape=(None,), dtype=np.float64)
        scale_group.create_dataset(name='world_time', shape=(0,), maxshape=(None,), dtype=np.float64)
        scale_group.create_dataset(name='wall_time', shape=(0,), maxshape=(None,), dtype=np.float64)
        scale_group.create_dataset(name='iteration', shape=(0,), maxshape=(None,), dtype=np.int)
        scale_group.create_dataset(name='write_number', shape=(0,), maxshape=(None,), dtype=np.int)
        scale_group.create_dataset(name='constant', data=np.array([0.], dtype=np.float64))
        scale_group['constant'].make_scale()

        # Tasks
        task_group =  file.create_group('tasks')
        unique_bases = []
        for task_num, task in enumerate(self.tasks):
            op = task['operator']
            for axis in range(dist.dim):
                basis = op.domain.full_bases[axis]
                if basis not in unique_bases and basis is not None:
                    unique_bases.append(basis)
        for task_num, task in enumerate(self.tasks):
            op = task['operator']
            layout = task['layout']
            scales = task['scales']
            extra_op = task['extra_op']
            extra_op_comm = task['extra_op_comm']
            gnc_shape, gnc_start, write_shape, write_start, write_count = self.get_write_stats(extra_op, extra_op_comm, layout, scales, op.domain, op.tensorsig, index=0)
            if np.prod(write_shape) <= 1:
                # Start with shape[0] = 0 to chunk across writes for scalars
                file_shape = (0,) + tuple(write_shape)
            else:
                # Start with shape[0] = 1 to chunk within writes
                file_shape = (1,) + tuple(write_shape)
            file_max = (None,) + tuple(write_shape)
            dset = task_group.create_dataset(name=task['name'], shape=file_shape, maxshape=file_max, dtype=op.dtype)
            if not self.parallel:
                dset.attrs['global_shape'] = gnc_shape
                dset.attrs['start'] = gnc_start
                dset.attrs['count'] = write_count

            # Metadata and scales
            dset.attrs['task_number'] = task_num
            dset.attrs['constant'] = op.domain.constant
            dset.attrs['grid_space'] = layout.grid_space
            dset.attrs['scales'] = scales
            if extra_op is not None:
                if extra_op.task_type is not None:
                    dset.attrs['task_type'] = extra_op.task_type
                else:
                    dset.attrs['task_type'] = 'slice'
            else:
                dset.attrs['task_type'] = 'slice'

            # Time scales
            dset.dims[0].label = 't'
            for sn in ['sim_time', 'world_time', 'wall_time', 'timestep', 'iteration', 'write_number']:
                scale = scale_group[sn]
                dset.dims.create_scale(scale, sn)
                dset.dims[0].attach_scale(scale)

            # Spatial scales
            for axis in range(dist.dim):
                basis = op.domain.full_bases[axis]
                if basis is None:
                    sn = lookup = 'constant'
                else:
                    subaxis = axis - basis.axis
                    full_data = None
                    if layout.grid_space[axis]:
                        sn = basis.coordsystem.coords[subaxis].name
                        data = basis.local_grids(scales)[subaxis].ravel()
                        full_data = basis.global_grids(scales)[subaxis]
                    else:
                        sn = 'k' + basis.coordsystem.coords[subaxis].name
                        data = basis.local_elements()[subaxis].ravel()
                        full_data = np.arange(basis.global_shape(layout, scales)[subaxis])
                    lookup = 'hash_' + hashlib.sha1(data).hexdigest()
                    if lookup not in scale_group:
                        scale_group.create_dataset(name=lookup, data=data)
                        scale_group[lookup].make_scale()
                    if self.min_process[0] == self.dist.comm_cart.rank:
                        if len(unique_bases) > 1:
                            key = '{}_{}/{:.1f}'.format(sn, unique_bases.index(basis), scales[subaxis])
                        else:
                            key = '{}/{:.1f}'.format(sn, scales[subaxis])
                        if key not in scale_group:
                            scale_group[key] = full_data
                scale = scale_group[lookup]
                dset.dims[axis+1].label = sn
                dset.dims[axis+1].attach_scale(scale)

    def process(self, world_time=0, wall_time=0, sim_time=0, timestep=0, iteration=0, **kw):
        """ Save task outputs to HDF5 file.
            Largely copied from dedalus/core/evaluator.py, with addition of extra_op logic"""
        if self.local_file is None:
            # If there's no data on this file, don't even make a file.
            for task_num, task in enumerate(self.tasks):
                out = task['out']
                extra_op = task['extra_op']
                extra_op_comm = task['extra_op_comm']
                out.require_scales(task['scales'])
                out.require_layout(task['layout'])
                memory_space, file_space = self.get_hdf5_spaces(extra_op, extra_op_comm, out.layout, task['scales'], out.domain, out.tensorsig, 0)
                if np.prod(memory_space.shape) != 0:
                    self.local_file = True
            if self.local_file is None:
                self.min_process[0] = self.dist.comm_cart.size
                self.local_file = False
            self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.min_process, op=MPI.MIN)
        # HACK: fix world time and timestep inputs from solvers.py/timestepper.py
        file = self.get_file()
        self.total_write_num += 1
        self.file_write_num += 1
        index = self.file_write_num - 1
        if self.local_file:
            file.attrs['writes'] = self.file_write_num

            # Update time scales
            sim_time_dset = file['scales/sim_time']
            world_time_dset = file['scales/world_time']
            wall_time_dset = file['scales/wall_time']
            timestep_dset = file['scales/timestep']
            iteration_dset = file['scales/iteration']
            write_num_dset = file['scales/write_number']

            sim_time_dset.resize(index+1, axis=0)
            sim_time_dset[index] = sim_time
            world_time_dset.resize(index+1, axis=0)
            world_time_dset[index] = world_time
            wall_time_dset.resize(index+1, axis=0)
            wall_time_dset[index] = wall_time
            timestep_dset.resize(index+1, axis=0)
            timestep_dset[index] = timestep
            iteration_dset.resize(index+1, axis=0)
            iteration_dset[index] = iteration
            write_num_dset.resize(index+1, axis=0)
            write_num_dset[index] = self.total_write_num
        # Create task datasets
        for task_num, task in enumerate(self.tasks):
            out = task['out']
            extra_op = task['extra_op']
            extra_op_comm = task['extra_op_comm']
            out.require_scales(task['scales'])
            out.require_layout(task['layout'])
            memory_space, file_space = self.get_hdf5_spaces(extra_op, extra_op_comm, out.layout, task['scales'], out.domain, out.tensorsig, index)

            if self.local_file:
                dset = file['tasks'][task['name']]
                dset.resize(index+1, axis=0)

            if self.parallel:
                if extra_op is not None:
                    raise NotImplementedError('extra ops are not implemented for parallel file outputs')
                dset.id.write(memory_space, file_space, out.data, dxpl=self._property_list)
            else:
                if extra_op is not None:
                    out_data = extra_op(out, comm=extra_op_comm)
                else:
                    out_data = out.data
                if self.local_file:
                    dset.id.write(memory_space, file_space, out_data)

        if self.local_file:
            file.close()
        self.dist.comm_cart.Barrier()


    def get_write_stats(self, extra_op, extra_op_comm, layout, scales, domain, tensorsig, index):
        """ Determine write parameters for nonconstant subspace of a field.
            Largely copied from dedalus/core/evaluator.py, with addition of extra_op logic"""

        # References
        tensor_order = len(tensorsig)
        tensor_shape = tuple(cs.dim for cs in tensorsig)

        if extra_op is not None:
            gshape = tensor_shape + tuple(extra_op.global_shape)
            constant = np.array((False,)*tensor_order + tuple(extra_op.constant))
            if extra_op_comm:
                if self.dist.comm_cart.rank == 0:
                    lshape = tensor_shape + tuple(extra_op.global_shape)
                else:
                    lshape = tensor_shape + tuple([0 for s in extra_op.global_shape])
            else:
                lshape = tensor_shape + tuple(extra_op.local_shape)
            start = [0 for i in range(tensor_order)]
            if extra_op_comm:
                for i in extra_op.global_shape:
                    start += [0,]
            else:
                for elements in extra_op.local_elements:
                    if elements is not None:
                        start += [elements[0]]
                    else:
                        start += [None,]
            start = np.array(start)
        else:
            gshape = tensor_shape + layout.global_shape(domain, scales)
            lshape = tensor_shape + layout.local_shape(domain, scales)

            constant = np.array((False,)*tensor_order + domain.constant)
            start = np.array([0 for i in range(tensor_order)] + [elements[0] for elements in layout.local_elements(domain, scales)])
        first = (start == 0)

        # Build counts, taking just the first entry along constant axes
        write_count = np.array(lshape)
        write_count[constant & first] = 1
        write_count[constant & ~first] = 0

        # Collectively writing global data
        global_nc_shape = np.array(gshape)
        global_nc_shape[constant] = 1
        global_nc_start = np.array(start)
        global_nc_start[constant & ~first] = 1
        global_nc_start = np.array(global_nc_start, dtype=int)

        if self.parallel:
            # Collectively writing global data
            write_shape = global_nc_shape
            write_start = global_nc_start
        else:
            # Independently writing local data
            write_shape = write_count
            write_start = np.zeros(start.shape, dtype=int)
        return global_nc_shape, global_nc_start, write_shape, write_start, write_count

    def get_hdf5_spaces(self, extra_op, extra_op_comm, layout, scales, domain, tensorsig, index):
        """Create HDF5 space objects for writing nonconstant subspace of a field.
            Largely copied from dedalus/core/evaluator.py, with addition of extra_op logic"""

        # References
        tensor_order = len(tensorsig)
        tensor_shape = tuple(cs.dim for cs in tensorsig)
        if extra_op is not None:
            constant = np.array((False,)*tensor_order + tuple(extra_op.constant))
            if extra_op_comm:
                if self.dist.comm_cart.rank == 0:
                    lshape = tensor_shape + tuple(extra_op.global_shape)
                else:
                    lshape = tensor_shape + tuple([0 for s in extra_op.global_shape])
            else:
                lshape = tensor_shape + tuple(extra_op.local_shape)
            start = [0 for i in range(tensor_order)]
            if extra_op_comm:
                for i in extra_op.global_shape:
                    start += [0,]
            else:
                for elements in extra_op.local_elements:
                    if elements is not None:
                        start += [elements[0]]
                    else:
                        start += [None,]
            start = np.array(start)
        else:
            constant = np.array((False,)*tensor_order + domain.constant)
            lshape = tensor_shape + layout.local_shape(domain, scales)
            start = np.array([0 for i in range(tensor_order)] + [elements[0] for elements in layout.local_elements(domain, scales)])
        gnc_shape, gnc_start, write_shape, write_start, write_count = self.get_write_stats(extra_op, extra_op_comm, layout, scales, domain, tensorsig, index)

        # Build HDF5 spaces
        memory_shape = tuple(lshape)
        memory_start = tuple(np.zeros(start.shape, dtype=int))
        memory_count = tuple(write_count)
        memory_space = h5py.h5s.create_simple(memory_shape)
        memory_space.select_hyperslab(memory_start, memory_count)

        file_shape = (index+1,) + tuple(write_shape)
        file_start = (index,) + tuple(write_start)
        file_count = (1,) + tuple(write_count)
        file_space = h5py.h5s.create_simple(file_shape)
        file_space.select_hyperslab(file_start, file_count)

        return memory_space, file_space

    def check_file_limits(self):
        """Check if write or size limits have been reached."""
        write_limit = (self.file_write_num >= self.max_writes)
        size_limit = (self.current_path.stat().st_size >= self.max_size)
        if not self.parallel:
            # reduce(size_limit, or) across processes
            comm = self.dist.comm_cart
            self._sl_array[0] = size_limit
            comm.Barrier()
            comm.Allreduce(MPI.IN_PLACE, self._sl_array, op=MPI.LOR)
            size_limit = self._sl_array[0]
        return (write_limit or size_limit)

    def get_file(self):
        """Return current HDF5 file, creating if necessary."""
        # Create new file if necessary
        if os.path.exists(str(self.current_path)):
            if self.check_file_limits():
                self.set_num += 1
                self.create_current_file()
        else:
            self.create_current_file()
        if self.local_file:
            # Open current file
            if self.parallel:
                comm = self.dist.comm_cart
                h5file = h5py.File(str(self.current_path), 'r+', driver='mpio', comm=comm)
            else:
                h5file = h5py.File(str(self.current_path), 'r+')
            self.file_write_num = h5file['/scales/write_number'].shape[0]
            return h5file
        else:
            return None

    def create_current_file(self):
        """Generate new HDF5 file in current_path."""
        self.file_write_num = 0
        comm = self.dist.comm_cart
        if self.parallel:
            file = h5py.File(str(self.current_path), 'w-', driver='mpio', comm=comm)
        else:
            # Create set folder
            with Sync(comm):
                if comm.rank == 0:
                    self.current_path.parent.mkdir()
            if self.local_file:
                if FILEHANDLER_TOUCH_TMPFILE:
                    tmpfile = self.base_path.joinpath('tmpfile_p%i' %(comm.rank))
                    tmpfile.touch()
                file = h5py.File(str(self.current_path), 'w-')
                if FILEHANDLER_TOUCH_TMPFILE:
                    tmpfile.unlink()
        if self.local_file:
            self.setup_file(file)
            file.close()

    @property
    def current_path(self):
        comm = self.dist.comm_cart
        set_num = self.set_num
        if self.parallel:
            # Save in base directory
            file_name = '%s_s%i.hdf5' %(self.base_path.stem, set_num)
            return self.base_path.joinpath(file_name)
        else:
            # Save in folders for each filenum in base directory
            folder_name = '%s_s%i' %(self.base_path.stem, set_num)
            folder_path = self.base_path.joinpath(folder_name)
            if self.local_file:
                file_name = '%s_s%i_p%i.h5' %(self.base_path.stem, set_num, comm.rank)
            else:
                file_name = '%s_s%i_p%i.h5' %(self.base_path.stem, set_num, self.min_process[0])
            return folder_path.joinpath(file_name)


