from collections import OrderedDict

import numpy as np
from mpi4py import MPI

from dedalus.extras.flow_tools import GlobalArrayReducer
from dedalus.tools.parallel import Sync 

class PhiAverager:

    def __init__(self, basis, distributor, φ_ind=0, dealias=1):
        """
        Creates an object which averages over φ. Assumes that L_dealias is 1.

        # Arguments
            basis (SphericalBasis3D object) :
                The basis on which the averaging should be done.
            φ_ind (int, optional) :
                The index of the basis which is the azimuthal basis (0, 1, 2).
            dealias (float, optional) :
                Dealiasing factor for the grid
        """
        self.distributor = distributor
        self.rank = self.distributor.comm_cart.rank

        self.dealias = dealias
        self.φ_ind     = φ_ind
        self.basis     = basis
        self.Lmax      = basis.shape[1]-1
        self.φl        = basis.local_grid_azimuth(self.dealias)
        self.φg        = basis.global_grid_azimuth(self.dealias)
        φb = np.zeros_like(self.φg, dtype=bool)
        for φ in self.φl.flatten():
            φb[φ == self.φg] = 1
        self.global_weight_φ = (np.ones_like(self.φg)*np.pi/((self.Lmax+1)*self.dealias))
        self.weight_φ = self.global_weight_φ[φb].reshape(self.φl.shape)
        self.volume_φ = np.sum(self.global_weight_φ)

        rg = basis.global_grid_radius(self.dealias)
        θg = basis.global_grid_colatitude(self.dealias)
        rb = np.zeros_like(rg, dtype=bool)
        θb = np.zeros_like(θg, dtype=bool)
        for r in basis.local_grid_radius(self.dealias).flatten():
            rb[r == rg] = True
        for θ in basis.local_grid_colatitude(self.dealias).flatten():
            θb[θ == θg] = True
        self.global_bool = rb*θb
        self.local_profile  = np.zeros_like(rg*θg)
        self.global_profile = np.zeros_like(rg*θg)
        self.local_tensor_profile  = np.zeros([3, *tuple(self.global_profile.shape)])
        self.global_tensor_profile = np.zeros([3, *tuple(self.global_profile.shape)])

    def __call__(self, arr, comm=False, tensor=False):
        """ Takes the azimuthal average of the NumPy array arr. """
        if tensor:
            local_piece = np.expand_dims(np.sum(np.expand_dims(self.weight_φ, axis=0)*arr, axis=self.φ_ind+1), axis=self.φ_ind+1)/self.volume_φ
        else:
            local_piece = np.expand_dims(np.sum(self.weight_φ*arr, axis=self.φ_ind), axis=self.φ_ind)/self.volume_φ
        if comm:
            if tensor:
                self.local_tensor_profile *= 0
                self.local_tensor_profile[self.global_bool*np.ones((3,1,1,1), dtype=bool)] = local_piece.flatten()
                self.distributor.comm_cart.Allreduce(self.local_tensor_profile, self.global_tensor_profile, op=MPI.SUM)
                if self.rank == 0:
                    return self.global_tensor_profile
                else:
                    return (np.nan, np.nan, np.nan)
            else:
                self.local_profile *= 0
                self.local_profile[self.global_bool] = local_piece.flatten()
                self.distributor.comm_cart.Allreduce(self.local_profile, self.global_profile, op=MPI.SUM)
                if self.rank == 0:
                    return self.global_profile
                else:
                    return np.nan 
        else:
            return local_piece

class PhiThetaAverager(PhiAverager):
    """
    Creates radial profiles that have been averaged over azimuth and colatitude.
    """
    def __init__(self, basis, distributor, θ_ind=1, φ_ind=0, dealias=1):
        super(PhiThetaAverager, self).__init__(basis, distributor, φ_ind=φ_ind, dealias=dealias)
        self.weight_θ = basis.local_colatitude_weights(self.dealias)
        self.global_weight_θ = basis.global_colatitude_weights(self.dealias)
        self.θ_ind    = θ_ind
        self.distributor = distributor

        rl = basis.local_grid_radius(self.dealias)
        rg = basis.global_grid_radius(self.dealias)
        θg = basis.global_grid_colatitude(self.dealias)
        self.rb = np.zeros_like(rg, dtype=bool)
        for r in rl.flatten():
            self.rb[r == rg] = True
        self.local_profile = np.zeros_like(rg)
        self.global_profile = np.zeros_like(rg)
        self.local_t_profile  = np.zeros((3, *tuple(rg.shape)))
        self.global_t_profile = np.zeros((3, *tuple(rg.shape)))

        self.theta_vol = np.sum(self.global_weight_θ)
        
    def __call__(self, arr, tensor=False):
        """ Takes the azimuthal and colatitude average of the NumPy array arr. """
        arr = super(PhiThetaAverager, self).__call__(arr, tensor=tensor)
        if tensor: 
            local_sum = np.expand_dims(np.sum(np.expand_dims(self.weight_θ, axis=0)*arr, axis=self.θ_ind+1), axis=self.θ_ind+1)/self.theta_vol
            self.local_t_profile *= 0
            self.local_t_profile[:, self.rb] = local_sum.squeeze()
            self.distributor.comm_cart.Allreduce(self.local_t_profile, self.global_t_profile, op=MPI.SUM)
            return self.global_t_profile
        else:
            local_sum = np.expand_dims(np.sum(self.weight_θ*arr, axis=self.θ_ind), axis=self.θ_ind)/self.theta_vol
            self.local_profile *= 0
            self.local_profile[self.rb] = local_sum.squeeze()
            self.distributor.comm_cart.Allreduce(self.local_profile, self.global_profile, op=MPI.SUM)
            return self.global_profile


class VolumeAverager:

    def __init__(self, basis, distributor, dummy_field, dealias=1, radius=1):
        """
        Initialize the averager.

        # Arguments
            basis (BallBasis) :
                The basis on which the sphere is being solved.
            distributor (Distributor) :
                The Dedalus distributor object for the simulation
            dummy_field (Field) :
                A dummy field used to figure out integral weights.
            dealias (float, optional) :
                Angular dealiasing factor.
            radius (float, optional) :
                The radius of the simulation domain
        """
        self.basis    = basis
        self.Lmax     = basis.shape[1]-1
        self.dealias  = dealias

        self.weight_θ = basis.local_colatitude_weights(self.dealias)
        self.weight_r = basis.radial_basis.local_weights(self.dealias)
        self.reducer  = GlobalArrayReducer(distributor.comm_cart)
        self.vol_test = np.sum(self.weight_r*self.weight_θ+0*dummy_field['g'])*np.pi/(self.Lmax+1)/self.dealias
        self.vol_test = self.reducer.reduce_scalar(self.vol_test, MPI.SUM)
        self.volume   = 4*np.pi*radius**3/3
        self.vol_correction = self.volume/self.vol_test

        self.φavg = PhiAverager(self.basis, distributor)

        self.operations = OrderedDict()
        self.fields     = OrderedDict()
        self.values     = OrderedDict()

    def __call__(self, arr, comm=True):
        """
        Performs a volume average over the given field

        # Arguments
            arr (NumPy array) :
                A 3D NumPy array on the grid.
        """
        avg = np.sum(self.vol_correction*self.weight_r*self.weight_θ*arr.real)
        avg *= np.pi/(self.Lmax+1)/self.dealias
        avg /= self.volume
        if comm:
            return self.reducer.reduce_scalar(avg, MPI.SUM)
        else:
            return avg

class EquatorSlicer:
    """
    A class which slices out an array at the equator.
    """

    def __init__(self, basis, distributor, dealias=1):
        """
        Initialize the slice plotter.

        # Arguments
            basis (BallBasis) :
                The basis on which the sphere is being solved.
            distributor (Distributor) :
                The Dedalus distributor object for the simulation
        """
        self.basis = basis
        self.distributor = distributor
        self.rank = self.distributor.comm_cart.rank
        self.dealias = dealias

        self.θg    = self.basis.global_grid_colatitude(self.dealias)
        self.θl    = self.basis.local_grid_colatitude(self.dealias)
        self.nφ    = np.prod(self.basis.global_grid_azimuth(self.dealias).shape)
        self.nr    = np.prod(self.basis.global_grid_radius(self.dealias).shape)
        self.Lmax  = basis.shape[1] - 1
        self.Nmax  = basis.shape[-1] - 1
        θ_target   = self.θg[0,(self.Lmax+1)//2,0]
        if np.prod(self.θl.shape) != 0:
            self.i_θ   = np.argmin(np.abs(self.θl[0,:,0] - θ_target))
            self.tplot             = θ_target in self.θl
        else:
            self.i_θ = None
            self.tplot = False

        rg = self.basis.global_grid_radius(self.dealias)
        rl = basis.local_grid_radius(self.dealias)
        self.rb = np.zeros_like(rg, dtype=bool)
        for r in rl.flatten():
            self.rb[r == rg] = True
        self.rb = self.rb.flatten()
        self.global_equator = np.zeros((self.nφ, 1, self.nr))

        self.include_data = self.distributor.comm_cart.gather(self.tplot)

    def __call__(self, arr, comm=False):
        """ Communicate local plot data globally """
        if self.i_θ is None:
            eq_slice = np.zeros_like(arr)
        else:
            eq_slice = arr[:,self.i_θ,:].real 
        if comm:
            eq_slice = self.distributor.comm_cart.gather(eq_slice, root=0)
            with Sync():
                data = []
                if self.rank == 0:
                    for s, i in zip(eq_slice, self.include_data):
                        if i: data.append(s)
                    data = np.array(data)
                    return np.expand_dims(np.transpose(data, axes=(1,0,2)).reshape((self.nφ, self.nr)), axis=1)
                else:
                    return np.nan
        else:
            if self.tplot:
                self.global_equator[:,0,self.rb] = eq_slice
            return self.global_equator
