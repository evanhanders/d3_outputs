import hashlib
import pathlib
from collections import OrderedDict

import numpy as np
import h5py
from mpi4py import MPI

from dedalus.tools.parallel import Sync 
from dedalus.core.evaluator import FileHandler

class d3FileHandler(FileHandler):

    def __init__(self, solver, *args, **kwargs):
        """ 
        An abstract class for writing to .h5 files, inspired by classic dedalus file handlers

        # Arguments
            solver (InitialValueSolver) :
                The dedalus solver used for the IVP
        """
        self.solver = solver
        self.evaluator = solver.evaluator
        super(d3FileHandler, self).__init__(*args, self.evaluator.dist, self.evaluator.vars, **kwargs)
        self.comm  = self.dist.comm_cart
        self.shapes = []
        self.buffs  = []
        self.current_file_name = None

        self.write_tasks = OrderedDict()
        self.extra_tasks = OrderedDict()
        self.writes     = 0

        self.evaluator.add_handler(self)

    def add_task(self, op, extra_op=None, **kw):
        super(d3FileHandler, self).add_task(op, **kw)
        self.tasks[-1]['extra_op'] = extra_op
        self.tasks[-1]['basis'] = op.domain.bases[-1]

    def define_extra_tasks(self):
        """ A function which should be implemented right before the loop, defining tasks to evaluate in terms of simulation fields. """
        pass
       
    def create_file(self):
        """ Creates and returns the output file """
        this_file_name = '{:s}_s{}.h5'.format(self.base_path.stem, int(self.writes/self.max_writes)+1)
        self.current_file_name = self.base_path.joinpath(this_file_name)
        file = h5py.File(self.current_file_name, 'w')
 
        # Scale group
        scale_group = file.create_group('scales')
        for k in ['sim_time', 'iteration', 'write_number', 'timestep']:
            scale_group.create_dataset(name=k,     shape=(0,), maxshape=(None,), dtype=np.float64)

        basis = self.tasks[-1]['basis']
        for i,k in enumerate(basis.coords.coords):
            grid=basis.global_grids()[i]
            scale_group.create_dataset(name='{}/1.0'.format(k), data=grid)
 
        task_group = file.create_group('tasks')
        for task in self.tasks:
            this_shape = self.shapes[task['shape_ind']]
            shape    = tuple([0] + [d for d in this_shape])
            maxshape = tuple([None] + [d for d in this_shape])
            task_group.create_dataset(name=task['name'], shape=shape, maxshape=maxshape, dtype=np.float64)
        for task_nm, task in self.extra_tasks.items():
            this_shape = task.shape
            shape    = tuple([0] + [d for d in this_shape])
            maxshape = tuple([None] + [d for d in this_shape])
            task_group.create_dataset(name=task_nm, shape=shape, maxshape=maxshape, dtype=np.float64)

        return file

    def _write_base_scales(self, file, world_time=0, wall_time=0, sim_time=0, timestep=0, iteration=0):
        """ Writes some basic scalar information to file. """
        for k in ['sim_time', 'iteration', 'write_number', 'timestep']:
            file['scales/{}'.format(k)].resize((self.writes-1) % self.max_writes + 1, axis=0)
        file['scales/sim_time'][-1] = sim_time
        file['scales/iteration'][-1] = iteration
        file['scales/write_number'][-1] = self.writes
        file['scales/timestep'][-1] = timestep
        return file

    def apply_operation(self):
        #Set up buffer space for communication on first write
        if len(self.shapes) == 0:
            for task_num, task in enumerate(self.tasks):
                out = task['out']
                out.require_scales(task['scales'])
                out.require_layout(task['layout'])
                operated_task = task['extra_op'](out, comm=False)
                have_shape = False
                for ind, shape in enumerate(self.shapes):
                    if len(operated_task.shape) == len(shape):
                        shapes_not_equal = [s1 != s2 for s1, s2 in zip(operated_task.shape, shape)]
                        if True not in shapes_not_equal:    
                            have_shape = True
                            task['shape_ind'] = ind
                            break
                if not have_shape:
                    self.shapes.append(operated_task.shape)
                    task['shape_ind'] = int(len(self.shapes)-1)
            for ind, shape in enumerate(self.shapes):
                buff_ind = 0
                for task_num, task in enumerate(self.tasks):
                    if task['shape_ind'] == ind:
                        task['buff_ind'] = buff_ind
                        buff_ind += 1
                self.buffs.append(np.zeros(tuple([buff_ind] + list(shape)), dtype=np.float64))

        for task_num, task in enumerate(self.tasks):
            out = task['out']
            out.require_scales(task['scales'])
            out.require_layout(task['layout'])
            self.buffs[task['shape_ind']][task['buff_ind'],:] = task['extra_op'](out, comm=False)

        for buff_ind, buff in enumerate(self.buffs):
            self.comm.Allreduce(MPI.IN_PLACE, self.buffs[buff_ind], op=MPI.SUM)
        for task_num, task in enumerate(self.tasks):
            self.write_tasks[task['name']] = self.buffs[task['shape_ind']][task['buff_ind'],:]
      
    def process(self, **kw):
        """ Write to file """
        self.apply_operation()
        self.define_extra_tasks()
        with Sync(self.comm):
            if self.comm.rank == 0:
                if self.writes % self.max_writes == 0:
                    file = self.create_file()
                else:
                    file = h5py.File(self.current_file_name, 'a')
                self.writes += 1
                file = self._write_base_scales(file, **kw)
                for k, task in self.write_tasks.items():
                    file['tasks/{}'.format(k)].resize((self.writes-1) % self.max_writes + 1, axis=0)
                    file['tasks/{}'.format(k)][-1] = task
                file.close()

class d3BallShellFileHandler(d3FileHandler):

    def apply_operation(self):
        if self.shape is None:
            self.base_tasks = []
            for i, task in enumerate(self.tasks):
                if '_shell' in task['name']: continue
                if '_ball' in task['name']:
                    self.base_tasks.append(task['name'].split('_ball')[0])
            self.shape = self.operation(self.fields[self.base_tasks[0]+'_ball']['g'], self.fields[self.base_tasks[0]+'_shell']['g'], comm=False).shape
            self.buff = np.zeros((len(self.base_tasks), *tuple(self.shape)))

        for i, task in enumerate(self.base_tasks):
            self.buff[i] = self.operation(self.fields[task+'_ball']['g'], self.fields[task+'_shell']['g'], comm=False)

        self.comm.Allreduce(MPI.IN_PLACE, self.buff, op=MPI.SUM)
        for i, task in enumerate(self.base_tasks):
            self.write_tasks[task] = self.buff[i]


class betterd3FileHandler(FileHandler):

    def __init__(self, solver, filename, **kwargs):
        """ 
        An abstract class for writing to .h5 files, inspired by classic dedalus file handlers

        # Arguments
            solver (InitialValueSolver) :
                The dedalus solver used for the IVP
        """
        self.evaluator = solver.evaluator
        super(betterd3FileHandler, self).__init__(filename, self.evaluator.dist, self.evaluator.vars, **kwargs)
        self.task_dict = OrderedDict()

        self.evaluator.add_handler(self)

    def add_task(self, op, extra_op=None, **kw):
        super(betterd3FileHandler, self).add_task(op, **kw)
        task = self.tasks[-1]
        task['extra_op'] = extra_op
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
        for task_num, task in enumerate(self.tasks):
            op = task['operator']
            layout = task['layout']
            scales = task['scales']
            extra_op = task['extra_op']
            gnc_shape, gnc_start, write_shape, write_start, write_count = self.get_write_stats(extra_op, layout, scales, op.domain, op.tensorsig, index=0)
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
                    if layout.grid_space[axis]:
                        sn = basis.coordsystem.coords[subaxis].name
                        data = basis.local_grids(scales)[subaxis].ravel()
                    else:
                        sn = 'k' + basis.coordsystem.coords[subaxis].name
                        data = basis.local_elements()[subaxis].ravel()
                    lookup = 'hash_' + hashlib.sha1(data).hexdigest()
                    if lookup not in scale_group:
                        scale_group.create_dataset(name=lookup, data=data)
                        scale_group[lookup].make_scale()
                scale = scale_group[lookup]
                dset.dims[axis+1].label = sn
                dset.dims[axis+1].attach_scale(scale)

    def process(self, world_time=0, wall_time=0, sim_time=0, timestep=0, iteration=0, **kw):
        """ Save task outputs to HDF5 file.
            Largely copied from dedalus/core/evaluator.py, with addition of extra_op logic"""
        # HACK: fix world time and timestep inputs from solvers.py/timestepper.py
        file = self.get_file()
        self.total_write_num += 1
        self.file_write_num += 1
        file.attrs['writes'] = self.file_write_num
        index = self.file_write_num - 1

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
            out.require_scales(task['scales'])
            out.require_layout(task['layout'])

            dset = file['tasks'][task['name']]
            dset.resize(index+1, axis=0)

            memory_space, file_space = self.get_hdf5_spaces(extra_op, out.layout, task['scales'], out.domain, out.tensorsig, index)
            if self.parallel:
                if extra_op is not None:
                    raise NotImplementedError('extra ops are not impelemented for parallel file outputs')
                dset.id.write(memory_space, file_space, out.data, dxpl=self._property_list)
            else:
                if extra_op is not None:
                    out_data = extra_op(out)
                else:
                    out_data = out.data
                dset.id.write(memory_space, file_space, out_data)

        file.close()

    def get_write_stats(self, extra_op, layout, scales, domain, tensorsig, index):
        """ Determine write parameters for nonconstant subspace of a field.
            Largely copied from dedalus/core/evaluator.py, with addition of extra_op logic"""

        # References
        tensor_order = len(tensorsig)
        tensor_shape = tuple(cs.dim for cs in tensorsig)

        if extra_op is not None:
            gshape = tensor_shape + tuple(extra_op.global_shape)
            lshape = tensor_shape + tuple(extra_op.local_shape)
            constant = np.array((False,)*tensor_order + tuple(extra_op.constant))
            start = [0 for i in range(tensor_order)]
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

    def get_hdf5_spaces(self, extra_op, layout, scales, domain, tensorsig, index):
        """Create HDF5 space objects for writing nonconstant subspace of a field.
            Largely copied from dedalus/core/evaluator.py, with addition of extra_op logic"""

        # References
        tensor_order = len(tensorsig)
        tensor_shape = tuple(cs.dim for cs in tensorsig)
        lshape = tensor_shape + layout.local_shape(domain, scales)
        constant = np.array((False,)*tensor_order + domain.constant)
        if extra_op is not None:
            start = [0 for i in range(tensor_order)]
            for elements in extra_op.local_elements:
                if elements is not None:
                    start += [elements[0]]
                else:
                    start += [None,]
            start = np.array(start)
        else:
            start = np.array([0 for i in range(tensor_order)] + [elements[0] for elements in layout.local_elements(domain, scales)])
        gnc_shape, gnc_start, write_shape, write_start, write_count = self.get_write_stats(extra_op, layout, scales, domain, tensorsig, index)

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
