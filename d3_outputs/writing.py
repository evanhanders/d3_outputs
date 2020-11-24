import pathlib
from collections import OrderedDict

import numpy as np
import h5py
from mpi4py import MPI

from dedalus.tools.parallel import Sync 

class FileWriter:

    def __init__(self, basis, distributor, root_dir, filename='scalar', write_dt=1, write_iter=np.inf, max_writes=np.inf, dealias=1):
        """ 
        An abstract class for writing to .h5 files, inspired by classic dedalus file handlers

        # Arguments
            basis (BallBasis) :
                The 3D spherical basis on which the problem is being solved.
            distributor (Distributor) :
                The Dedalus distributor object for the simulation
            root_dir (string) :
                The directory in which the folder filename/ will be created, for placement of this file.
            filename (string, optional) :
                The name of the file and folder to be output.
            write_dt (float, optional) :
               The amount of simulation time to wait between outputs.
            write_iter (int, optional) :
               The amount of simulation iterations to wait between outputs.
            max_writes (int, optional) :
                The maximum number of writes allowed per file.
        """
        self.basis = basis
        self.comm  = distributor.comm_cart
        self.base_path  = pathlib.Path('{:s}/{:s}/'.format(root_dir, filename))
        self.filename   = filename
        if not self.base_path.exists():
            with Sync(self.comm):
                if self.comm.rank == 0:
                    self.base_path.mkdir()
        self.current_file_name = None
        self.dealias = dealias
        self.shape = None

        self.tasks      = OrderedDict()
        self.write_dt   = write_dt
        self.write_iter = write_iter
        self.last_time  = -write_dt
        self.last_iter  = -1
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
        scale_group.create_dataset(name='sim_time',     shape=(0,), maxshape=(None,), dtype=np.float64)
        scale_group.create_dataset(name='write_number', shape=(0,), maxshape=(None,), dtype=np.float64)
        scale_group.create_dataset(name='iteration', shape=(0,), maxshape=(None,), dtype=np.float64)
 
        task_group = file.create_group('tasks')
        for nm in self.tasks.keys():
            shape    = tuple([0] + [d for d in self.shape])
            maxshape = tuple([None] + [d for d in self.shape])
            task_group.create_dataset(name=nm, shape=shape, maxshape=maxshape, dtype=np.float64)

        return file



    def _write_base_scales(self, solver, file):
        """ Writes some basic scalar information to file. """
        file['scales/sim_time'].resize((self.writes-1) % self.max_writes + 1, axis=0)
        file['scales/sim_time'][-1] = solver.sim_time
        file['scales/iteration'].resize((self.writes-1) % self.max_writes + 1, axis=0)
        file['scales/iteration'][-1] = solver.iteration
        file['scales/write_number'].resize((self.writes-1) % self.max_writes + 1, axis=0)
        file['scales/write_number'][-1] = self.writes
        return file

    def pre_write_evaluations(self):
        """ 
        A function which is called before evaluate_tasks().

        This can be used to ensure that a field is evaluated ONLY when a write to file is about to occur.
        """
        pass
       
    def process(self, solver):
        """
        Checks to see if data needs to be written to file. If so, writes it.

        # Arguments:
            solver (Dedalus Solver):
                The solver object for the Dedalus IVP
        """
        if solver.sim_time - self.last_time > self.write_dt or solver.iteration - self.last_iter > self.write_iter:
            self.last_time = solver.sim_time
            self.pre_write_evaluations()
            self.evaluate_tasks()
            with Sync(self.comm):
                if self.comm.rank == 0:
                    if self.writes % self.max_writes == 0:
                        file = self.create_file()
                    else:
                        file = h5py.File(self.current_file_name, 'a')
                    self.writes += 1
                    file = self._write_base_scales(solver, file)

                    for k, task in self.tasks.items():
                        file['tasks/{}'.format(k)].resize((self.writes-1) % self.max_writes + 1, axis=0)
                        file['tasks/{}'.format(k)][-1] = task
                    file.close()

class HandlerWriter(FileWriter):

    def __init__(self, *args, **kwargs):
        super(HandlerWriter, self).__init__(*args, **kwargs)
        self.handler  = None
        self.averager = None

        self.buff  = None

    def add_handler(self, dict_handler):
        self.handler = dict_handler
        self.buff = np.zeros((len(self.handler.tasks), *tuple(self.shape)))

    def add_averager(self, averager):
        self.averager = averager
        
    def pre_write_evaluations(self):
        for i, task in enumerate(self.handler.tasks):
#            self.buff[i] = self.averager(self.handler.fields[task['name']]['g'], comm=True)
#            self.tasks[task['name']] = self.buff[i]
            self.buff[i] = self.averager(self.handler.fields[task['name']]['g'], comm=False)

        self.comm.Allreduce(MPI.IN_PLACE, self.buff, op=MPI.SUM)
        for i, task in enumerate(self.handler.tasks):
            self.tasks[task['name']] = self.buff[i]

class ScalarWriter(HandlerWriter):

    def __init__(self, *args, **kwargs):
        super(ScalarWriter, self).__init__(*args, filename='scalar', **kwargs)
        self.shape = []

class RadialProfileWriter(FileWriter):

    def __init__(self, *args, **kwargs):
        super(RadialProfileWriter, self).__init__(*args, filename='profiles', **kwargs)
        self.shape = self.basis.global_grid_radius(self.dealias).shape

    def create_file(self):
        file = super(RadialProfileWriter, self).create_file()
        scales_group = file['scales']
        scales_group.create_dataset(name='r/1.0', data=self.basis.global_grid_radius(self.dealias))
        return file

class EquatorialSliceWriter(HandlerWriter):
    
    def __init__(self, *args, **kwargs):
        super(EquatorialSliceWriter, self).__init__(*args, filename='eq_slice', **kwargs)
        r_shape = self.basis.global_grid_radius(self.dealias).shape
        φ_shape = self.basis.global_grid_azimuth(self.dealias).shape
        self.shape = [np.max((rl, φl)) for rl, φl in zip(r_shape, φ_shape)]

    def create_file(self):
        file = super(EquatorialSliceWriter, self).create_file()
        scales_group = file['scales']
        scales_group.create_dataset(name='r/1.0', data=self.basis.global_grid_radius(self.dealias))
        scales_group.create_dataset(name='φ/1.0', data=self.basis.global_grid_azimuth(self.dealias))
        return file

class SphericalShellWriter(FileWriter):
    
    def __init__(self, *args, **kwargs):
        super(SphericalShellWriter, self).__init__(*args, filename='shell_slice', **kwargs)
        φ_shape = self.basis.global_grid_azimuth(self.dealias).shape
        θ_shape = self.basis.global_grid_colatitude(self.dealias).shape
        self.shape = [np.max((φl, θl)) for φl, θl in zip(φ_shape, θ_shape)]

    def create_file(self):
        file = super(SphericalShellWriter, self).create_file()
        scales_group = file['scales']
        scales_group.create_dataset(name='φ/1.0', data=self.basis.global_grid_azimuth(self.dealias))
        scales_group.create_dataset(name='θ/1.0', data=self.basis.global_grid_colatitude(self.dealias))
        return file

class MeridionalSliceWriter(HandlerWriter):
    
    def __init__(self, *args, **kwargs):
        super(MeridionalSliceWriter, self).__init__(*args, filename='mer_slice', **kwargs)
        r_shape = self.basis.global_grid_radius(self.dealias).shape
        θ_shape = self.basis.global_grid_colatitude(self.dealias).shape
        self.shape = [np.max((rl, θl)) for rl, θl in zip(r_shape, θ_shape)]

    def create_file(self):
        file = super(MeridionalSliceWriter, self).create_file()
        scales_group = file['scales']
        scales_group.create_dataset(name='r/1.0', data=self.basis.global_grid_radius(self.dealias))
        scales_group.create_dataset(name='θ/1.0', data=self.basis.global_grid_colatitude(self.dealias))
        return file

class VolumeWriter(FileWriter):

    def __init__(self, *args, **kwargs):
        super(VolumeWriter, self).__init__(*args, filename='volumes', **kwargs)
        r_shape = self.basis.global_grid_radius(self.dealias).shape
        φ_shape = self.basis.global_grid_azimuth(self.dealias).shape
        θ_shape = self.basis.global_grid_colatitude(self.dealias).shape
        self.shape = [np.max((rl, φl, θl)) for rl, φl, θl in zip(r_shape, φ_shape, θ_shape)]

    def create_file(self):
        file = super(VolumeWriter, self).create_file()
        scales_group = file['scales']
        scales_group.create_dataset(name='r/1.0', data=self.basis.global_grid_radius(self.dealias))
        scales_group.create_dataset(name='φ/1.0', data=self.basis.global_grid_azimuth(self.dealias))
        scales_group.create_dataset(name='θ/1.0', data=self.basis.global_grid_colatitude(self.dealias))
        return file


