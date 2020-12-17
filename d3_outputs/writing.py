import pathlib
from collections import OrderedDict

import numpy as np
import h5py
from mpi4py import MPI

from dedalus.tools.parallel import Sync 
from dedalus.core.evaluator import FileHandler

class d3FileHandler(FileHandler):

    def __init__(self, solver, operation, filename, **kwargs):
        """ 
        An abstract class for writing to .h5 files, inspired by classic dedalus file handlers

        # Arguments
            solver (InitialValueSolver) :
                The dedalus solver used for the IVP
            operation (OutputTask) :
                An output task to apply to each field in handler before output
        """
        self.operation = operation
        self.solver = solver
        self.evaluator = solver.evaluator
        super(d3FileHandler, self).__init__(filename, self.evaluator.dist, self.evaluator.vars, **kwargs)
        self.basis = self.operation.basis
        self.comm  = self.operation.dist.comm_cart
        self.shape = None
        self.buff  = None
        self.current_file_name = None

        self.write_tasks      = OrderedDict()
        self.writes     = 0

        self.evaluator.add_handler(self)

    def extra_tasks(self):
        """ A function which should be implemented right before the loop, defining tasks to evaluate in terms of simulation fields. """
        pass
       
    def create_file(self):
        """ Creates and returns the output file """
        self.current_file_name = '{:s}_s{}.h5'.format(self.base_path.stem, int(self.writes/self.max_writes)+1)
        file = h5py.File(self.current_file_name, 'w')
 
        # Scale group
        scale_group = file.create_group('scales')
        for k in ['sim_time', 'iteration', 'write_number', 'timestep']:
            scale_group.create_dataset(name=k,     shape=(0,), maxshape=(None,), dtype=np.float64)

        for i,k in enumerate(self.basis.coords.coords):
            grid=self.basis.global_grids()[i]
            scale_group.create_dataset(name='{}/1.0'.format(k), data=grid)
 
        task_group = file.create_group('tasks')
        for nm in self.write_tasks.keys():
            shape    = tuple([0] + [d for d in self.shape])
            maxshape = tuple([None] + [d for d in self.shape])
            task_group.create_dataset(name=nm, shape=shape, maxshape=maxshape, dtype=np.float64)

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
        for task_num, task in enumerate(self.tasks):
            out = task['out']
            out.require_scales(task['scales'])
            out.require_layout(task['layout'])
            if self.shape is None:
                self.shape = self.operation(out['g'], comm=False).shape
                self.buff = np.zeros((len(self.tasks), *tuple(self.shape)))

            self.buff[task_num] = self.operation(out['g'], comm=False)

        self.comm.Allreduce(MPI.IN_PLACE, self.buff, op=MPI.SUM)
        for task_num, task in enumerate(self.tasks):
            self.write_tasks[task['name']] = self.buff[task_num]
      
    def process(self, **kw):
        """ Write to file """
        self.apply_operation()
        self.extra_tasks()
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

