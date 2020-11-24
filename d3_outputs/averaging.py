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

    def __call__(self, arr, comm=False, tensor=False):
        """ Takes the azimuthal average of the NumPy array arr. """
        if tensor:
            self.global_tensor_profile *= 0
            self.global_tensor_profile[:, :, self.gslices[1], self.gslices[2]] = np.expand_dims(np.sum(self.t_weight_φ*arr, axis=1), axis=1)/self.volume_φ
            if comm:
                self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.global_tensor_profile, op=MPI.SUM)
                if self.rank == 0:
                    return self.global_tensor_profile
                else:
                    return (np.nan, np.nan, np.nan)
            else:
                return self.global_tensor_profile

        else:
            self.global_profile *= 0
            self.global_profile[:, self.gslices[1], self.gslices[2]] = np.expand_dims(np.sum(self.weight_φ*arr, axis=0), axis=0)/self.volume_φ
            if comm:
                self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.global_profile, op=MPI.SUM)
                if self.rank == 0:
                    return self.global_profile
                else:
                    return np.nan 
            else:
                return self.global_profile

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
        
    def __call__(self, arr, tensor=False, comm=False):
        """ Takes the azimuthal and colatitude average of the NumPy array arr. """
        arr = super(PhiThetaAverager, self).__call__(arr, tensor=tensor, comm=comm)
        if tensor: 
            arr = arr[:,:,self.gslices[1], self.gslices[2]]
            self.pt_global_t_profile *= 0
            self.pt_global_t_profile[:,:,:, self.gslices[2]] = np.expand_dims(np.sum(self.t_weight_θ*arr, axis=2), axis=2)/self.theta_vol
            if comm:
                self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.pt_global_t_profile, op=MPI.SUM)
            return self.pt_global_t_profile
        else:
            arr = arr[:,self.gslices[1], self.gslices[2]]
            self.pt_global_profile *= 0
            self.pt_global_profile[:,:, self.gslices[2]] = np.expand_dims(np.sum(self.weight_θ*arr, axis=1), axis=1)/self.theta_vol
            if comm:
                self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.pt_global_profile, op=MPI.SUM)
            return self.pt_global_profile

class BallVolumeAverager(OutputTask):

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

    def __call__(self, arr, comm=True):
        """
        Performs a volume average over the given field

        # Arguments
            arr (NumPy array) :
                A 3D NumPy array on the grid.
        """
        avg = np.sum(self.vol_correction*self.weight_r*self.weight_θ*arr.real)
        avg *= np.pi/(self.basis.Lmax+1)/self.dealias[1]
        avg /= self.volume
        if comm:
            return self.reducer.reduce_scalar(avg, MPI.SUM)
        else:
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
                A dummy field used to figure out integral weights, basis, etc.
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
        self.include_data = self.dist.comm_cart.gather(self.tplot)

    def __call__(self, arr, comm=False):
        """ Communicate local plot data globally """
        if self.i_θ is None:
            eq_slice = np.zeros_like(arr)
        else:
            eq_slice = arr[:,self.i_θ,:].real 
        if comm:
            eq_slice = self.dist.comm_cart.gather(eq_slice, root=0)
            with Sync():
                data = []
                if self.rank == 0:
                    for s, i in zip(eq_slice, self.include_data):
                        if i: data.append(s)
                    data = np.array(data)
                    return np.expand_dims(np.transpose(data, axes=(1,0,2)).reshape((self.shape[0], self.shape[2])), axis=1)
                else:
                    return np.nan
        else:
            if self.tplot:
                self.global_equator[:,0,self.gslices[2]] = eq_slice
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

    def __call__(self, arr, comm=False):
        self.buff *= 0
        if np.prod(arr.shape) > 0:
            self.buff[self.gslices[0], self.gslices[1], :] = arr
        if comm:
            self.dist.comm_cart.Allreduce(MPI.IN_PLACE, self.buff, op=MPI.SUM)
        else:
            return self.buff
