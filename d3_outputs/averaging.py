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
    
    def __init__(self, field):
        domain = field.domain
        self.basis = domain.bases[0]
        self.gslices = GridSlicer(field)
        self.dist = field.dist
        self.rank = self.dist.comm_cart.rank
        self.dealias = domain.dealias
        self.shape = domain.grid_shape(self.dealias)

class PhiAverager(OutputTask):

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
        self.weight_φ = self.global_weight_φ[self.gslices[0],:,:].reshape((self.shape[0], 1, 1))
        self.t_weight_φ = np.expand_dims(self.weight_φ, axis=0)
        self.volume_φ = np.sum(self.global_weight_φ)

        #Set up memory space
        self.global_profile = np.zeros((1, *tuple(self.shape[1:])))
        self.global_tensor_profile = np.zeros([3, 1, *tuple(self.shape[1:])])

    def __call__(self, fd, comm=True):
        """ Takes the azimuthal average of the Dedalus field. """
        arr = fd['g']
        if len(fd.tensorsig) == 1:
            self.global_tensor_profile *= 0
            self.global_tensor_profile[:, :, self.gslices[1], self.gslices[2]] = np.expand_dims(np.sum(self.t_weight_φ*arr, axis=1), axis=1)/self.volume_φ
            if comm:
                self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.global_tensor_profile, op=MPI.SUM)
            return self.global_tensor_profile
        elif len(fd.tensorsig) == 0:
            self.global_profile *= 0
            self.global_profile[:, self.gslices[1], self.gslices[2]] = np.expand_dims(np.sum(self.weight_φ*arr, axis=0), axis=0)/self.volume_φ
            if comm:
                self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.global_profile, op=MPI.SUM)
            return self.global_profile
        else:
            raise NotImplementedError("Only scalars and tensors are implemented")

class PhiThetaAverager(PhiAverager):
    """
    Creates radial profiles that have been averaged over azimuth and colatitude.
    """
    def __init__(self, field):
        super(PhiThetaAverager, self).__init__(field)
        self.weight_θ = self.basis.local_colatitude_weights(self.dealias[1])
        self.t_weight_θ = np.expand_dims(self.basis.local_colatitude_weights(self.dealias[1]), axis=0)

        global_weight_θ = self.basis.global_colatitude_weights(self.dealias[1])
        self.theta_vol = np.sum(global_weight_θ)

        self.pt_global_profile = np.zeros((1, 1, self.shape[2]))
        self.pt_global_t_profile = np.zeros((3, 1, 1, self.shape[2]))
        
    def __call__(self, fd, comm=True):
        """ Takes the azimuthal and colatitude average of the Dedalus field. """
        arr = super(PhiThetaAverager, self).__call__(fd, comm=comm)
        if len(fd.tensorsig) == 1: 
            arr = arr[:,:,self.gslices[1], self.gslices[2]]
            self.pt_global_t_profile *= 0
            self.pt_global_t_profile[:,:,:, self.gslices[2]] = np.expand_dims(np.sum(self.t_weight_θ*arr, axis=2), axis=2)/self.theta_vol
            if comm:
                self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.pt_global_t_profile, op=MPI.SUM)
            return self.pt_global_t_profile
        elif len(fd.tensorsig) == 0:
            arr = arr[:,self.gslices[1], self.gslices[2]]
            self.pt_global_profile *= 0
            self.pt_global_profile[:,:, self.gslices[2]] = np.expand_dims(np.sum(self.weight_θ*arr, axis=1), axis=1)/self.theta_vol
            if comm:
                self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.pt_global_profile, op=MPI.SUM)
            return self.pt_global_profile
        else:
            raise NotImplementedError("Only scalars and tensors are implemented")

class Spherical3DVolumeAverager(OutputTask):
    
    def __init__(self, field):
        super(Spherical3DVolumeAverager, self).__init__(field)
        self.global_profile = np.zeros((1, 1, 1))
        self.global_t_profile = np.zeros((3, 1, 1, 1))

    def __call__(self, fd, comm=True):
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
            self.global_t_profile[:,0,0,0] = avg
            if comm:
                self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.global_t_profile, op=MPI.SUM)
            return self.global_t_profile
        elif len(fd.tensorsig) == 0: 
            avg = np.sum(self.vol_correction*self.weight_r*self.weight_θ*arr.real)
            avg *= np.pi/(self.basis.Lmax+1)/self.dealias[1]
            avg /= self.volume
            self.global_profile[0,0,0] = avg
            if comm:
                self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.global_profile, op=MPI.SUM)
            return self.global_profile
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

    def __call__(self, ball_fd, shell_fd, comm=True):
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
        if np.prod(θl.shape) != 0:
            self.i_θ = np.argmin(np.abs(θl[0,:,0] - θ_target))
            self.tplot = θ_target in θl
        else:
            self.i_θ = None
            self.tplot = False

        self.global_equator = np.zeros((self.shape[0], 1, self.shape[2]))
        self.global_t_equator = np.zeros((3, self.shape[0], 1, self.shape[2]))
        self.include_data = self.dist.comm_cart.gather(self.tplot)

    def __call__(self, fd, comm=True):
        """ Communicate local plot data globally """
        if len(fd.tensorsig) > 1:
            raise NotImplementedError("Only scalars and tensors are implemented")
        arr = fd['g']
        if not self.tplot:
            eq_slice = np.zeros_like(arr)
        else:
            if len(fd.tensorsig) == 1:
                eq_slice = arr[:,:,self.i_θ,:].real 
                self.global_t_equator[:,:,0,self.gslices[2]] = eq_slice
            elif len(fd.tensorsig) == 0:
                eq_slice = arr[:,self.i_θ,:].real 
                self.global_equator[:,0,self.gslices[2]] = eq_slice

        if len(fd.tensorsig) == 1:
            if comm:
                self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.global_t_equator, op=MPI.SUM)
            return self.global_t_equator
        elif len(fd.tensorsig) == 0:
            if comm:
                self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.global_equator, op=MPI.SUM)
            return self.global_equator


class SphericalShellCommunicator(OutputTask):

    def __init__(self, field):
        """
        Initialize the slice plotter.

        # Arguments
            field (Field) :
                A dummy field used to figure out integral weights, basis, etc.
        """
        super(SphericalShellCommunicator, self).__init__(field)
        self.buff = np.zeros((self.shape[0], self.shape[1], 1))
        self.t_buff = np.zeros((3, self.shape[0], self.shape[1], 1))
        self.tS2_buff = np.zeros((2, self.shape[0], self.shape[1], 1))

    def __call__(self, fd, comm=True):
        if len(fd.tensorsig) > 1:
            raise NotImplementedError("Only scalars and tensors are implemented")
        arr = fd['g']
        self.buff[:] = 0
        self.t_buff[:] = 0
        self.tS2_buff[:] = 0
        if np.prod(arr.shape) > 0:
            if len(fd.tensorsig) == 1:
                if arr.shape[0] == 2:
                    self.tS2_buff[:, self.gslices[0], self.gslices[1], :] = arr
                else:
                    self.t_buff[:, self.gslices[0], self.gslices[1], :] = arr
            elif len(fd.tensorsig) == 0:
                self.buff[self.gslices[0], self.gslices[1], :] = arr
        if len(fd.tensorsig) == 1:
            if arr.shape[0] == 2:
                if comm:
                    self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.tS2_buff, op=MPI.SUM)
                return self.tS2_buff
            else:
                if comm:
                    self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.t_buff, op=MPI.SUM)
                return self.t_buff
        elif len(fd.tensorsig) == 0:
            if comm:
                self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.buff, op=MPI.SUM)
            return self.buff
