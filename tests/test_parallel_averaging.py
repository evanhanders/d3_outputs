"""
Unit tests which ensure that d3_outputs operations return the same answer in parallel and serial
"""
import pytest
import numpy as np
import functools

from mpi4py import MPI

from dedalus.core import coords, distributor, basis, field, operators
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer
import d3_outputs.averaging as averaging

def make_ball_basis(Nmax, Lmax, radius, dtype=np.float64, dealias=1, mesh=None, comm=MPI.COMM_WORLD):
    c    = coords.SphericalCoordinates('φ', 'θ', 'r')
    d    = distributor.Distributor((c,), mesh=mesh, comm=comm)
    b    = basis.BallBasis(c, (2*(Lmax+2), Lmax+1, Nmax+1), radius=radius, dtype=dtype)
    φ,  θ,  r  = b.local_grids((dealias, dealias, dealias))
    return c, d, b, φ, θ, r

def make_shell_basis(Nmax, Lmax, r_inner, r_outer, dtype=np.float64, dealias=1, mesh=None, comm=MPI.COMM_WORLD):
    c    = coords.SphericalCoordinates('φ', 'θ', 'r')
    d    = distributor.Distributor((c,), mesh=mesh, comm=comm)
    b    = basis.SphericalShellBasis(c, (2*(Lmax+2), Lmax+1, Nmax+1), radii=(r_inner, r_outer), dtype=dtype)
    φ,  θ,  r  = b.local_grids((dealias, dealias, dealias))
    return c, d, b, φ, θ, r

def make_ballShell_basis(NmaxB, NmaxS, Lmax, r_inner, r_outer, dtype=np.float64, dealias=1, mesh=None, comm=MPI.COMM_WORLD):
    c, d, bB, φB, θB, rB = make_ball_basis(NmaxB, Lmax, r_inner, dtype=dtype, dealias=dealias, mesh=mesh, comm=comm)
    bS   = basis.SphericalShellBasis(c, (2*(Lmax+2), Lmax+1, NmaxS+1), radii=(r_inner, r_outer), dtype=dtype)
    φS,  θS,  rS  = bS.local_grids((dealias, dealias, dealias))
    return c, d, bB, bS, φB, θB, rB, φS, θS, rS

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [14])
@pytest.mark.parametrize('radius', [1, 2])
@pytest.mark.parametrize('mesh', [[2,2], [1,4], [4,1]])
def test_ball_volume_average(Nmax, Lmax, radius, dtype, mesh):
    op_outs = []
    for this_mesh, comm in zip((mesh, None), (MPI.COMM_WORLD, MPI.COMM_SELF)):
        c, d, b, φ, θ, r = make_ball_basis(Nmax, Lmax, radius, dtype=dtype, mesh=this_mesh, comm=comm)
        f = field.Field(dist=d, bases=(b,), dtype=dtype)
        f['g'] = r**2
        vol_averager = averaging.BallVolumeAverager(f)
        op_avg      = vol_averager(f, comm=True)
        op_outs.append(op_avg)
    assert np.allclose(op_outs[0], op_outs[1])

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [14])
@pytest.mark.parametrize('r_inner', [0.1, 0.5])
@pytest.mark.parametrize('r_outer', [1, 2])
@pytest.mark.parametrize('mesh', [[2,2], [1,4], [4,1]])
def test_shell_volume_average(Nmax, Lmax, r_inner, r_outer, dtype, mesh):
    op_outs = []
    for this_mesh, comm in zip((mesh, None), (MPI.COMM_WORLD, MPI.COMM_SELF)):
        c, d, b, φ, θ, r = make_shell_basis(Nmax, Lmax, r_inner, r_outer, dtype=dtype, mesh=this_mesh, comm=comm)
        f = field.Field(dist=d, bases=(b,), dtype=dtype)
        f['g'] = r**2
        vol_averager = averaging.ShellVolumeAverager(f)
        op_avg      = vol_averager(f, comm=True)
        op_outs.append(op_avg)
    assert np.allclose(op_outs[0], op_outs[1])

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('NmaxB', [15])
@pytest.mark.parametrize('NmaxS', [15])
@pytest.mark.parametrize('Lmax', [14])
@pytest.mark.parametrize('r_inner', [1])
@pytest.mark.parametrize('r_outer', [1.5, 2])
@pytest.mark.parametrize('mesh', [[2,2], [1,4], [4,1]])
def test_ballShell_volume_average(NmaxB, NmaxS, Lmax, r_inner, r_outer, dtype, mesh):
    op_outs = []
    for this_mesh, comm in zip((mesh, None), (MPI.COMM_WORLD, MPI.COMM_SELF)):
        c, d, bB, bS, φB, θB, rB, φS, θS, rS = make_ballShell_basis(NmaxB, NmaxS, Lmax, r_inner, r_outer, dtype=dtype, mesh=this_mesh, comm=comm)
        fB = field.Field(dist=d, bases=(bB,), dtype=dtype)
        fS = field.Field(dist=d, bases=(bS,), dtype=dtype)
        fB['g'] = rB**2
        fS['g'] = rS**2
        vol_averager = averaging.BallShellVolumeAverager(fB, fS)
        op_avg      = vol_averager(fB, fS, comm=True)
        op_outs.append(op_avg)
    assert np.allclose(op_outs[0], op_outs[1])

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [14])
@pytest.mark.parametrize('radius', [1, 2])
@pytest.mark.parametrize('mesh', [[2,2], [1,4], [4,1]])
def test_ball_phi_average(Nmax, Lmax, radius, dtype, mesh):
    op_outs = []
    for this_mesh, comm in zip((mesh, None), (MPI.COMM_WORLD, MPI.COMM_SELF)):
        c, d, b, φ, θ, r = make_ball_basis(Nmax, Lmax, radius, dtype=dtype, mesh=this_mesh, comm=comm)
        f = field.Field(dist=d, bases=(b,), dtype=dtype)
        averager = averaging.PhiAverager(f)
        f['g'] = r**2 * np.sin(φ)**2*np.cos(θ)
        op_avg      = averager(f, comm=True)
        op_outs.append(op_avg)
    assert np.allclose(op_outs[0], op_outs[1])

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [14])
@pytest.mark.parametrize('r_inner', [1])
@pytest.mark.parametrize('r_outer', [1.5, 2])
@pytest.mark.parametrize('mesh', [[2,2], [1,4], [4,1]])
def test_shell_phi_average(Nmax, Lmax, r_inner, r_outer, dtype, mesh):
    op_outs = []
    for this_mesh, comm in zip((mesh, None), (MPI.COMM_WORLD, MPI.COMM_SELF)):
        c, d, b, φ, θ, r = make_shell_basis(Nmax, Lmax, r_inner, r_outer, dtype=dtype, mesh=this_mesh, comm=comm)
        f = field.Field(dist=d, bases=(b,), dtype=dtype)
        averager = averaging.PhiAverager(f)
        f['g'] = r**2 * np.sin(φ)**2*np.cos(θ)
        op_avg      = averager(f, comm=True)
        op_outs.append(op_avg)
    assert np.allclose(op_outs[0], op_outs[1])

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [14])
@pytest.mark.parametrize('radius', [1, 2])
@pytest.mark.parametrize('mesh', [[2,2], [1,4], [4,1]])
def test_ball_phi_theta_average(Nmax, Lmax, radius, dtype, mesh):
    op_outs = []
    for this_mesh, comm in zip((mesh, None), (MPI.COMM_WORLD, MPI.COMM_SELF)):
        c, d, b, φ, θ, r = make_ball_basis(Nmax, Lmax, radius, dtype=dtype, mesh=this_mesh, comm=comm)
        f = field.Field(dist=d, bases=(b,), dtype=dtype)
        averager = averaging.PhiThetaAverager(f)
        f['g'] = r**2 * np.sin(φ)**2 * 3 * np.cos(θ)**2
        op_avg      = averager(f, comm=True)
        op_outs.append(op_avg)
    assert np.allclose(op_outs[0], op_outs[1])

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [14])
@pytest.mark.parametrize('r_inner', [1])
@pytest.mark.parametrize('r_outer', [1.5, 2])
@pytest.mark.parametrize('mesh', [[2,2], [1,4], [4,1]])
def test_shell_phi_theta_average(Nmax, Lmax, r_inner, r_outer, dtype, mesh):
    op_outs = []
    for this_mesh, comm in zip((mesh, None), (MPI.COMM_WORLD, MPI.COMM_SELF)):
        c, d, b, φ, θ, r = make_shell_basis(Nmax, Lmax, r_inner, r_outer, dtype=dtype, mesh=this_mesh, comm=comm)
        f = field.Field(dist=d, bases=(b,), dtype=dtype)
        averager = averaging.PhiThetaAverager(f)
        f['g'] = r**2 * np.sin(φ)**2 * 3 * np.cos(θ)**2
        op_avg      = averager(f, comm=True)
        op_outs.append(op_avg)
    assert np.allclose(op_outs[0], op_outs[1])

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [14])
@pytest.mark.parametrize('radius', [1, 2])
@pytest.mark.parametrize('mesh', [[2,2], [1,4], [4,1]])
def test_ball_S2_outputter(Nmax, Lmax, radius, dtype, mesh):
    op_outs = []
    for this_mesh, comm in zip((mesh, None), (MPI.COMM_WORLD, MPI.COMM_SELF)):
        c, d, b, φ, θ, r = make_ball_basis(Nmax, Lmax, radius, dtype=dtype, mesh=this_mesh, comm=comm)
        f = field.Field(dist=d, bases=(b,), dtype=dtype)
        interp_op = f(r=0.5*radius)
        averager = averaging.OutputRadialInterpolate(f, interp_op)
        f['g'] = r**2 * np.sin(φ)**2 * 3 * np.cos(θ)**2
        op_avg      = averager(interp_op.evaluate(), comm=True)
        op_outs.append(op_avg)
    assert np.allclose(op_outs[0], op_outs[1])

