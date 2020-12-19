from collections import OrderedDict

import numpy as np
from mpi4py import MPI

from dedalus.extras.flow_tools import GlobalArrayReducer
from dedalus.tools.parallel import Sync 

class GridSlicer:

    def __init__(self, field):
        dist = field.dist
        base_slices = dist.layouts[-1].slices(field.domain, 1)
        self.slices = []
       
        for i in range(dist.dim):
            indices    = [0]*dist.dim
            indices[i] = slice(None)
            slice_axis = base_slices[i][tuple(indices)]
            self.slices.append(slice(slice_axis[0], slice_axis[-1]+1, 1))

    def __getitem__(self,index):
        return self.slices[index]

class OutputTask:
    
    task_type = None #'sum' or 'slice'
    
    def __init__(self, field):
        domain = field.domain
        self.basis = domain.bases[0]
        self.gslices = GridSlicer(field)
        self.dist = field.dist
        self.rank = self.dist.comm_cart.rank
        self.dealias = domain.dealias

        #Get local and global shapes of base field
        tensor_length = len(field.tensorsig)
        field_shape = field['g'].shape
        shape_length  = len(field_shape)
        self.local_shape = []
        for i in range(shape_length - tensor_length):
            self.local_shape.append(field_shape[i+tensor_length])
        self.global_shape = domain.grid_shape(self.dealias)
        self.local_elements = [np.arange(self.gslices[i].start, self.gslices[i].stop, self.gslices[i].step, dtype=int) for i in range(self.dist.dim)]
        self.local_slices = [self.gslices[i] for i in range(self.dist.dim)]
        self.constant = domain.constant


class PhiAverager(OutputTask):

    task_type = 'sum'

    def __init__(self, field):
        """
        Creates an object which averages over φ. Assumes that L_dealias is 1.

        # Arguments
            field (Field object) :
                a non-vector dedalus field.
        """
        super(PhiAverager, self).__init__(field)
        #Find integral weights
        self.φg        = self.basis.global_grid_azimuth(self.dealias[0])
        self.global_weight_φ = (np.ones_like(self.φg)*np.pi/((self.basis.Lmax+1)*self.dealias[0]))
        self.weight_φ = self.global_weight_φ[self.gslices[0],:,:].reshape((self.global_shape[0], 1, 1))
        self.t_weight_φ = np.expand_dims(self.weight_φ, axis=0)
        self.volume_φ = np.sum(self.global_weight_φ)
        self.local_shape = [1, self.local_shape[1], self.local_shape[2]]
        self.global_shape = [1, self.global_shape[1], self.global_shape[2]]
        self.local_elements = [np.zeros(1, dtype=int), self.local_elements[1], self.local_elements[2]]
        self.local_slices   = (slice(0,1,1), self.local_slices[1], self.local_slices[2])
        self.constant = (True, self.constant[1], self.constant[2])

        #Set up memory space
        self.local_profile = np.zeros(self.local_shape)
        self.local_t_profile = np.zeros([3,] + self.local_shape)
        self.global_profile = np.zeros(self.global_shape)
        self.global_t_profile = np.zeros([3,] + self.global_shape)

    def __call__(self, fd, comm=False):
        """ Takes the azimuthal average of the Dedalus field. """
        arr = fd['g']
        if len(fd.tensorsig) == 1: 
            self.local_t_profile[:] = np.sum(self.t_weight_φ*arr, axis=1).reshape(self.local_t_profile.shape)/self.volume_φ
        elif len(fd.tensorsig) == 0:
            self.local_profile[:] = np.sum(self.weight_φ*arr, axis=0).reshape(self.local_profile.shape)/self.volume_φ
        else:
            raise NotImplementedError("Only scalars and tensors are implemented")

        if not comm:
            if len(fd.tensorsig) == 1: 
                return self.local_t_profile
            elif len(fd.tensorsig) == 0:
                return self.local_profile

        if len(fd.tensorsig) == 1: 
            self.global_t_profile [:] = 0
            print(self.local_slices)
            self.global_t_profile[(slice(0,3,1),)+self.local_slices] = self.local_t_profile
            self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.global_t_profile, op=MPI.SUM)
            return self.global_t_profile
        elif len(fd.tensorsig) == 0:
            self.global_profile[:] = 0
            self.global_profile[self.local_slices] = self.local_profile
            self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.global_profile, op=MPI.SUM)
            return self.global_profile



class PhiThetaAverager(OutputTask):
    """
    Creates radial profiles that have been averaged over azimuth and colatitude.
    """

    task_type = 'sum'

    def __init__(self, field):
        super(PhiThetaAverager, self).__init__(field)
        self.phi_avg = PhiAverager(field)
        self.local_shape = [1, 1, self.local_shape[2]]
        self.global_shape = [1, 1, self.global_shape[2]]
        self.local_elements = [np.zeros(1, dtype=int), np.zeros(1, dtype=int), self.local_elements[2]]
        self.local_slices   = (slice(0,1,1), slice(0,1,1), self.local_slices[2])
        self.constant = (True, True, self.constant[2])
        self.weight_θ = self.basis.local_colatitude_weights(self.dealias[1])
        self.t_weight_θ = np.expand_dims(self.basis.local_colatitude_weights(self.dealias[1]), axis=0)

        global_weight_θ = self.basis.global_colatitude_weights(self.dealias[1])
        self.theta_vol = np.sum(global_weight_θ)

        self.local_profile = np.zeros(self.local_shape)
        self.local_t_profile = np.zeros([3,] + self.local_shape)
        self.global_profile = np.zeros(self.global_shape)
        self.global_t_profile = np.zeros([3,] + self.global_shape)
        
    def __call__(self, fd, comm=False):
        """ Takes the azimuthal and colatitude average of the Dedalus field. """
        arr = self.phi_avg(fd, comm=False)
        if len(fd.tensorsig) == 1: 
            self.local_t_profile[:] = np.sum(self.t_weight_θ*arr, axis=2).reshape(self.local_t_profile.shape)/self.theta_vol
        elif len(fd.tensorsig) == 0:
            self.local_profile[:] = np.sum(self.weight_θ*arr, axis=1).reshape(self.local_profile.shape)/self.theta_vol
        else:
            raise NotImplementedError("Only scalars and tensors are implemented")

        if not comm:
            if len(fd.tensorsig) == 1: 
                return self.local_t_profile[:]
            elif len(fd.tensorsig) == 0:
                return self.local_profile[:]

        if len(fd.tensorsig) == 1: 
            self.global_t_profile [:] = 0
            self.global_t_profile[(slice(0,3,1),) + self.local_slices] = self.local_t_profile
            if comm:
                self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.global_t_profile, op=MPI.SUM)
            return self.global_t_profile
        elif len(fd.tensorsig) == 0:
            self.global_profile [:] = 0
            self.global_profile[self.local_slices] = self.local_profile
            if comm:
                self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.global_profile, op=MPI.SUM)
            return self.global_profile


class Spherical3DVolumeAverager(OutputTask):

    task_type = 'sum'
    
    def __init__(self, field):
        super(Spherical3DVolumeAverager, self).__init__(field)
        self.local_shape = [1, 1, 1]
        self.global_shape = [1, 1, 1]
        self.local_elements = [np.zeros(1, dtype=int), np.zeros(1, dtype=int), np.zeros(1, dtype=int)]
        self.local_slices   = (0, 0, 0)
        self.constant = (True, True, True)
        self.profile = np.zeros(self.global_shape)
        self.t_profile = np.zeros([3,] + self.global_shape)

    def __call__(self, fd, comm=False):
        """
        Performs a volume average over the given field

        # Arguments
            fd (NumPy array) :
                A 3D NumPy array on the grid.
        """
        arr = fd['g']
        if len(fd.tensorsig) == 1:
            avg = np.sum(self.vol_correction*self.weight_r*self.weight_θ*arr.real, axis=(1,2,3))
            avg *= np.pi/(self.basis.Lmax+1)/self.dealias[1]
            avg /= self.volume
            self.t_profile[:,0,0,0] = avg
            if comm:
                self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.t_profile, op=MPI.SUM)
            return self.t_profile
        elif len(fd.tensorsig) == 0: 
            avg = np.sum(self.vol_correction*self.weight_r*self.weight_θ*arr.real)
            avg *= np.pi/(self.basis.Lmax+1)/self.dealias[1]
            avg /= self.volume
            self.profile[0,0,0] = avg
            if comm:
                self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.profile, op=MPI.SUM)
            return self.profile
        else:
            raise NotImplementedError("Only scalars and tensors are implemented")


class BallVolumeAverager(Spherical3DVolumeAverager):

    def __init__(self, field):
        """
        Initialize the averager.

        # Arguments
            field (Field) :
                A dummy field used to figure out integral weights, basis, etc.
        """
        super(BallVolumeAverager, self).__init__(field)
        self.weight_θ = self.basis.local_colatitude_weights(self.dealias[1])
        self.weight_r = self.basis.radial_basis.local_weights(self.dealias[2])
        self.reducer  = GlobalArrayReducer(self.dist.comm_cart)
        self.vol_test = np.sum(self.weight_r*self.weight_θ+0*field['g'])*np.pi/(self.basis.Lmax+1)/self.dealias[1]
        self.vol_test = self.reducer.reduce_scalar(self.vol_test, MPI.SUM)
        self.volume   = 4*np.pi*self.basis.radial_basis.radius**3/3
        self.vol_correction = self.volume/self.vol_test

class ShellVolumeAverager(Spherical3DVolumeAverager):

    def __init__(self, field):
        """
        Initialize the averager.

        # Arguments
            field (Field) :
                A dummy field used to figure out integral weights, basis, etc.
        """
        super(ShellVolumeAverager, self).__init__(field)
        self.r = self.basis.radial_basis.local_grid(self.dealias[2])
        self.weight_θ = self.basis.local_colatitude_weights(self.dealias[1])
        self.weight_r = self.basis.radial_basis.local_weights(self.dealias[2])*self.r**2
        self.reducer  = GlobalArrayReducer(self.dist.comm_cart)
        self.vol_test = np.sum(self.weight_r*self.weight_θ+0*field['g'])*np.pi/(self.basis.Lmax+1)/self.dealias[1]
        self.vol_test = self.reducer.reduce_scalar(self.vol_test, MPI.SUM)
        self.volume   = 4*np.pi*(self.basis.radial_basis.radii[1]**3 - self.basis.radial_basis.radii[0]**3)/3
        self.vol_correction = self.volume/self.vol_test


class BallShellVolumeAverager:

    def __init__(self, ball_field, shell_field):
        """
        Initialize the averager.

        # Arguments
            ball_field (Field) :
                A field used to figure out integral weights, etc, in the ball.
            shell_field (Field) :
                A field used to figure out integral weights, etc, in the shell.
        """
        self.ball_averager = BallVolumeAverager(ball_field)
        self.shell_averager = ShellVolumeAverager(shell_field)
        self.volume = self.ball_averager.volume + self.shell_averager.volume
        self.basis = self.ball_averager.basis
        self.dist  = self.ball_averager.dist

    def __call__(self, ball_fd, shell_fd, comm=False):
        """
        Performs a volume average over the given field

        # Arguments
            ball_fd (NumPy array) :
                A 3D NumPy array on the ball.
            shell_fd (NumPy array) :
                A 3D NumPy array on the shell.
        """
        ball_avg  = self.ball_averager(ball_fd, comm=comm)
        shell_avg = self.shell_averager(shell_fd, comm=comm)
        avg = (ball_avg*self.ball_averager.volume + shell_avg*self.shell_averager.volume)/self.volume
        return avg


class EquatorSlicer(OutputTask):
    """
    A class which slices out an array at the equator.
    """

    task_type = 'slice'

    def __init__(self, field):
        """
        Initialize the slice plotter.

        # Arguments
            field (Field) :
                A field used to figure out integral weights, basis, etc.
        """
        super(EquatorSlicer, self).__init__(field)
        θg    = self.basis.global_grid_colatitude(self.dealias[1])
        θl    = self.basis.local_grid_colatitude(self.dealias[1])
        θ_target   = θg[0,(self.basis.Lmax+1)//2,0]
        if θ_target in θl:
            self.i_θ = np.argmin(np.abs(θl[0,:,0] - θ_target))
            self.local_shape = [self.local_shape[0], 1, self.local_shape[2]]
            self.local_elements = [self.local_elements[0], np.zeros(1, dtype=int), self.local_elements[2]]
            self.local_slices   = (self.local_slices[0], slice(0,1,1), self.local_slices[2])
        else:
            self.i_θ = None
            self.local_shape = [self.local_shape[0], 0, self.local_shape[2]]
            self.local_elements = [self.local_elements[0], None, self.local_elements[2]]
            self.local_slices   = (self.local_slices[0], None, self.local_slices[2])
        self.global_shape = [self.global_shape[0], 1, self.global_shape[2]]
        self.constant = (self.constant[0], True, self.constant[2])

        self.local_equator = np.zeros(self.local_shape)
        self.local_t_equator = np.zeros([3,] + self.local_shape)
        self.global_equator = np.zeros(self.global_shape)
        self.global_t_equator = np.zeros([3,] + self.global_shape)

    def __call__(self, fd, comm=False):
        """ Communicate local plot data globally """
        arr = fd['g']
        if self.local_elements[1] is None:
            if len(fd.tensorsig) == 1:
                self.local_t_equator[:] = 0
            elif len(fd.tensorsig) == 0:
                self.local_t_equator[:] = 0
            else:
                raise NotImplementedError("Only scalars and tensors are implemented")
        else:
            if len(fd.tensorsig) == 1:
                self.local_t_equator[:] = arr[:,:,self.i_θ,:].real.reshape(self.local_t_equator.shape)
            elif len(fd.tensorsig) == 0:
                self.local_equator[:] = arr[:,self.i_θ,:].real.reshape(self.local_equator.shape)
            else:
                raise NotImplementedError("Only scalars and tensors are implemented")

        if not comm:
            if len(fd.tensorsig) == 1:
                return self.local_t_equator
            elif len(fd.tensorsig) == 0:
                return self.local_equator
            
        if self.local_elements[1] is None:
            if len(fd.tensorsig) == 1:
                self.global_t_equator[:] = 0
            elif len(fd.tensorsig) == 0:
                self.global_equator[:] = 0
        else:
            if len(fd.tensorsig) == 1:
                self.global_t_equator[:] = 0
                self.global_t_equator[(slice(0,3,1),)+self.local_slices] = self.local_t_equator
            elif len(fd.tensorsig) == 0:
                self.global_equator[:] = 0
                self.global_equator[self.local_slices] = self.local_equator

        if len(fd.tensorsig) == 1:
            self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.global_t_equator, op=MPI.SUM)
            return self.global_t_equator
        elif len(fd.tensorsig) == 0:
            self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.global_equator, op=MPI.SUM)
            return self.global_equator

class OutputRadialInterpolate(OutputTask):

    def __init__(self, field, interp_operator):
        """
        Initialize the slice plotter.

        # Arguments
            field (Field) :
                A dummy field used to figure out integral weights, basis, etc.
        """
        super(OutputRadialInterpolate, self).__init__(field)
        self.constant = (self.constant[0], self.constant[1], True)
        self.global_shape = [self.global_shape[0], self.global_shape[1], 1]
        output = interp_operator.evaluate()
        output_shape = output['g'].shape
        self.local_shape = output_shape[len(output.tensorsig):]
        if np.prod(self.local_shape) > 0:
            self.local_elements = [self.local_elements[0], self.local_elements[1], np.zeros(1, dtype=int)]
            self.local_slices = (self.local_slices[0], self.local_slices[1], slice(0,1,1))
        else:
            self.local_elements = [self.local_elements[0], self.local_elements[1], None]
            self.local_slices = (self.local_slices[0], self.local_slices[1], slice(0,0,1))
        self.full_local_slices = tuple([slice(0, output_shape[i], 1) for i in range(len(output.tensorsig))]) + self.local_slices

        self.local_shell = np.zeros(output_shape[:len(output.tensorsig)] + tuple(self.local_shape))
        self.global_shell = np.zeros(output_shape[:len(output.tensorsig)] + tuple(self.global_shape))

    def __call__(self, field, comm=False):
        """
        Returns the interpolated field on grid space
        """
        self.local_shell[:] = field['g']
        if not comm:
            return self.local_shell
        else:
            self.global_shell[:] = 0
            if self.local_elements[-1] is not None:
                self.global_shell[self.full_local_slices] = self.local_shell
            self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.global_shell, op=MPI.SUM)
            return self.global_shell
