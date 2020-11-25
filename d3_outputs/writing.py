import pathlib
from collections import OrderedDict

import numpy as np
import h5py
from mpi4py import MPI

from dedalus.tools.parallel import Sync 

class HandlerWriter:

    def __init__(self, handler, operation, root_dir, filename, max_writes=np.inf):
        """ 
        An abstract class for writing to .h5 files, inspired by classic dedalus file handlers

        # Arguments
            handler (DictHandler) :
                A dictionary handle full of tasks to output
            operation (OutputTask) :
                An output task to apply to each field in handler before output
            root_dir (string) :
                The directory in which the folder filename/ will be created, for placement of this file.
            filename (string) :
                The name of the file and folder to be output.
            max_writes (int, optional) :
                The maximum number of writes allowed per file.
        """
        self.handler = handler
        self.operation = operation
        self.basis = self.operation.basis
        self.comm  = self.operation.dist.comm_cart
        self.shape = None
        self.buff  = None
        self.base_path  = pathlib.Path('{:s}/{:s}/'.format(root_dir, filename))
        self.filename   = filename
        if not self.base_path.exists():
            with Sync(self.comm):
                if self.comm.rank == 0:
                    self.base_path.mkdir()
        self.current_file_name = None

        self.last_sim_div  = self.handler.last_sim_div
        self.last_wall_div = self.handler.last_wall_div
        self.last_iter_div = self.handler.last_iter_div

        self.tasks      = OrderedDict()
        self.writes     = 0
        self.max_writes = max_writes

    def evaluate_tasks(self):
        """ A function which should be implemented right before the loop, defining tasks to evaluate in terms of simulation fields. """
        pass
       
    def create_file(self):
        """ Creates and returns the output file """
        self.current_file_name = '{:s}/{:s}_s{}.h5'.format(str(self.base_path), self.filename, int(self.writes/self.max_writes)+1)
        file = h5py.File(self.current_file_name, 'w')
 
        # Scale group
        scale_group = file.create_group('scales')
        for k in ['sim_time', 'iteration', 'write_number', 'timestep']:
            scale_group.create_dataset(name=k,     shape=(0,), maxshape=(None,), dtype=np.float64)

        for i,k in enumerate(self.basis.coords.coords):
            grid=self.basis.global_grids()[i]
            scale_group.create_dataset(name='{}/1.0'.format(k), data=grid)
 
        task_group = file.create_group('tasks')
        for nm in self.tasks.keys():
            shape    = tuple([0] + [d for d in self.shape])
            maxshape = tuple([None] + [d for d in self.shape])
            task_group.create_dataset(name=nm, shape=shape, maxshape=maxshape, dtype=np.float64)

        return file

    def _write_base_scales(self, solver, dt, file):
        """ Writes some basic scalar information to file. """
        for k in ['sim_time', 'iteration', 'write_number', 'timestep']:
            file['scales/{}'.format(k)].resize((self.writes-1) % self.max_writes + 1, axis=0)
        file['scales/sim_time'][-1] = solver.sim_time
        file['scales/iteration'][-1] = solver.iteration
        file['scales/write_number'][-1] = self.writes
        file['scales/timestep'][-1] = dt
        return file

    def pre_write_evaluations(self):
        if self.shape is None:
            self.shape = self.operation(self.handler.fields[self.handler.tasks[0]['name']]['g'], comm=False).shape
            self.buff = np.zeros((len(self.handler.tasks), *tuple(self.shape)))

        for i, task in enumerate(self.handler.tasks):
            self.buff[i] = self.operation(self.handler.fields[task['name']]['g'], comm=False)

        self.comm.Allreduce(MPI.IN_PLACE, self.buff, op=MPI.SUM)
        for i, task in enumerate(self.handler.tasks):
            self.tasks[task['name']] = self.buff[i]
      
    def process(self, solver, dt):
        """
        Checks to see if data needs to be written to file. If so, writes it.

        # Arguments:
            solver (Dedalus Solver):
                The solver object for the Dedalus IVP
        """
        no_write =  (self.last_sim_div  == self.handler.last_sim_div)\
                   *(self.last_wall_div == self.handler.last_wall_div)\
                   *(self.last_iter_div == self.handler.last_iter_div)
        if not no_write:
            self.last_sim_div  = self.handler.last_sim_div
            self.last_wall_div = self.handler.last_wall_div
            self.last_iter_div = self.handler.last_iter_div
            self.pre_write_evaluations()
            self.evaluate_tasks()
            with Sync(self.comm):
                if self.comm.rank == 0:
                    if self.writes % self.max_writes == 0:
                        file = self.create_file()
                    else:
                        file = h5py.File(self.current_file_name, 'a')
                    self.writes += 1
                    file = self._write_base_scales(solver, dt, file)
                    for k, task in self.tasks.items():
                        file['tasks/{}'.format(k)].resize((self.writes-1) % self.max_writes + 1, axis=0)
                        file['tasks/{}'.format(k)][-1] = task
                    file.close()

class BallShellHandlerWriter(HandlerWriter):

    def pre_write_evaluations(self):
        if self.shape is None:
            self.base_tasks = []
            for i, task in enumerate(self.handler.tasks):
                if '_shell' in task['name']: continue
                if '_ball' in task['name']:
                    self.base_tasks.append(task['name'].split('_ball')[0])
            self.shape = self.operation(self.handler.fields[self.base_tasks[0]+'_ball']['g'], self.handler.fields[self.base_tasks[0]+'_shell']['g'], comm=False).shape
            self.buff = np.zeros((len(self.base_tasks), *tuple(self.shape)))

        for i, task in enumerate(self.base_tasks):
            self.buff[i] = self.operation(self.handler.fields[task+'_ball']['g'], self.handler.fields[task+'_shell']['g'], comm=False)

        self.comm.Allreduce(MPI.IN_PLACE, self.buff, op=MPI.SUM)
        for i, task in enumerate(self.base_tasks):
            self.tasks[task] = self.buff[i]

