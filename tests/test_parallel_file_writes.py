import pytest
import numpy as np
import functools

from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer
import d3_outputs.averaging as averaging
from d3_outputs.writing import betterd3FileHandler
from d3_outputs import file_merging
import h5py

def make_ball_basis(Nmax, Lmax, radius, dtype=np.float64, dealias=1, mesh=None):
    c    = coords.SphericalCoordinates('φ', 'θ', 'r')
    d    = distributor.Distributor((c,), mesh=mesh)
    b    = basis.BallBasis(c, (2*(Lmax+2), Lmax+1, Nmax+1), radius=radius, dtype=dtype)
    φ,  θ,  r  = b.local_grids((dealias, dealias, dealias))
    return c, d, b, φ, θ, r

def make_shell_basis(Nmax, Lmax, r_inner, r_outer, dtype=np.float64, dealias=1):
    c    = coords.SphericalCoordinates('φ', 'θ', 'r')
    d    = distributor.Distributor((c,), mesh=None)
    b    = basis.SphericalShellBasis(c, (2*(Lmax+2), Lmax+1, Nmax+1), radii=(r_inner, r_outer), dtype=dtype)
    φ,  θ,  r  = b.local_grids((dealias, dealias, dealias))
    return c, d, b, φ, θ, r

def make_ballShell_basis(NmaxB, NmaxS, Lmax, r_inner, r_outer, dtype=np.float64, dealias=1):
    c, d, bB, φB, θB, rB = make_ball_basis(NmaxB, Lmax, r_inner, dtype=dtype, dealias=dealias)
    bS   = basis.SphericalShellBasis(c, (2*(Lmax+2), Lmax+1, NmaxS+1), radii=(r_inner, r_outer), dtype=dtype)
    φS,  θS,  rS  = bS.local_grids((dealias, dealias, dealias))
    return c, d, bB, bS, φB, θB, rB, φS, θS, rS

#Operators
#div       = lambda A: operators.Divergence(A, index=0)
#lap       = lambda A: operators.Laplacian(A, c)
#grad      = lambda A: operators.Gradient(A, c)
#dot       = lambda A, B: arithmetic.DotProduct(A, B)
#curl      = lambda A: operators.Curl(A)
#cross     = lambda A, B: arithmetic.CrossProduct(A, B)
#trace     = lambda A: operators.Trace(A)
#ddt       = lambda A: operators.TimeDerivative(A)
#transpose = lambda A: operators.TransposeComponents(A)
#radComp   = lambda A: operators.RadialComponent(A)
#angComp   = lambda A, index=1: operators.AngularComponent(A, index=index)

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('Nmax', [7])
@pytest.mark.parametrize('Lmax', [6])
@pytest.mark.parametrize('radius', [1])
@pytest.mark.parametrize('mesh', [[2,2], [1,4], [4,1]])
def test_ball_volume_average(Nmax, Lmax, radius, dtype, mesh):
    c, d, b, φ, θ, r = make_ball_basis(Nmax, Lmax, radius, dtype=dtype, mesh=mesh)
    f = field.Field(name='f', dist=d, bases=(b,), dtype=dtype)
    sim_f = field.Field(name='f', dist=d, bases=(b,), dtype=dtype)
    tau_f = field.Field(name='f', dist=d, bases=(b.S2_basis(),), dtype=dtype)
    x = r * np.sin(θ) * np.cos(φ)
    y = r * np.sin(θ) * np.sin(φ)
    z = r * np.cos(θ)
    f['g'] = (x**2 + y**2 + z**2) * (radius**2 - r**2)

    lap  = lambda A: operators.Laplacian(A, c)
    ddt = lambda A: operators.TimeDerivative(A)
    LiftTau = lambda A: operators.LiftTau(A, b, -1)
    # Problem
    problem = problems.IVP([sim_f, tau_f])
    problem.add_equation((ddt(sim_f) - lap(sim_f) + LiftTau(tau_f), f))
    problem.add_equation((sim_f(r=radius), 0))
    # Solver
    solver = solvers.InitialValueSolver(problem, timesteppers.SBDF1, matrix_coupling=(False, False, True))
    vol_averager = averaging.BallVolumeAverager(f)
    outputs = betterd3FileHandler(solver, './testing/', iter=1, max_writes=np.inf)
    outputs.add_task(sim_f, extra_op=vol_averager, name='sim_f', layout='g')
    times = []
    calculated_averages = []
    dt = 1
    for i in range(10):
        times.append(solver.sim_time)
        calculated_averages.append(np.copy(vol_averager(sim_f, comm=True)))
        solver.step(dt)
    file_merging.merge_analysis(outputs.base_path)
    d.comm_cart.Barrier()
    sim_time = np.array(times)
    sim_averages = np.array(calculated_averages)
    with h5py.File('./testing/testing_s1.h5', 'r') as f:
        file_time = f['scales/sim_time'][()]
        file_averages = f['tasks/sim_f'][()]
    assert np.allclose(sim_time, file_time)
    assert np.allclose(sim_averages, file_averages)
#
#@pytest.mark.parametrize('dtype', [np.float64])
#@pytest.mark.parametrize('Nmax', [15])
#@pytest.mark.parametrize('Lmax', [14])
#@pytest.mark.parametrize('r_inner', [0.1, 0.5])
#@pytest.mark.parametrize('r_outer', [1, 2])
#def test_shell_volume_average(Nmax, Lmax, r_inner, r_outer, dtype):
#    c, d, b, φ, θ, r = make_shell_basis(Nmax, Lmax, r_inner, r_outer, dtype=dtype)
#    f = field.Field(dist=d, bases=(b,), dtype=dtype)
#    f['g'] = r**2
#    vol_averager = averaging.ShellVolumeAverager(f)
#    volume      = (4/3)*np.pi*(r_outer**3 - r_inner**3)
#    true_avg    = (4/5)*np.pi*(r_outer**5 - r_inner**5) / volume
#    op_avg      = vol_averager(f, comm=True)
#    assert np.allclose(true_avg, op_avg)
#
#@pytest.mark.parametrize('dtype', [np.float64])
#@pytest.mark.parametrize('NmaxB', [15])
#@pytest.mark.parametrize('NmaxS', [15])
#@pytest.mark.parametrize('Lmax', [14])
#@pytest.mark.parametrize('r_inner', [1])
#@pytest.mark.parametrize('r_outer', [1.5, 2])
#def test_ballShell_volume_average(NmaxB, NmaxS, Lmax, r_inner, r_outer, dtype):
#    c, d, bB, bS, φB, θB, rB, φS, θS, rS = make_ballShell_basis(NmaxB, NmaxS, Lmax, r_inner, r_outer, dtype=dtype)
#    fB = field.Field(dist=d, bases=(bB,), dtype=dtype)
#    fS = field.Field(dist=d, bases=(bS,), dtype=dtype)
#    fB['g'] = rB**2
#    fS['g'] = rS**2
#    vol_averager = averaging.BallShellVolumeAverager(fB, fS)
#    volume      = ((4/3)*np.pi*r_outer**3)
#    true_avg    = (4/5)*np.pi*r_outer**5 / volume
#    op_avg      = vol_averager(fB, fS, comm=True)
#    assert np.allclose(true_avg, op_avg)
#
#@pytest.mark.parametrize('dtype', [np.float64])
#@pytest.mark.parametrize('Nmax', [15])
#@pytest.mark.parametrize('Lmax', [14])
#@pytest.mark.parametrize('radius', [1, 2])
#def test_ball_phi_average(Nmax, Lmax, radius, dtype):
#    c, d, b, φ, θ, r = make_ball_basis(Nmax, Lmax, radius, dtype=dtype)
#    f = field.Field(dist=d, bases=(b,), dtype=dtype)
#    averager = averaging.PhiAverager(f)
#    f['g'] = r**2 * np.sin(φ)*np.cos(θ)
#    op_avg      = averager(f, comm=True)
#    true_avg    = np.zeros_like(op_avg)
#    assert np.allclose(true_avg, op_avg)
#    f['g'] = r**2 * np.sin(φ)**2*np.cos(θ)
#    op_avg      = averager(f, comm=True)
#    true_avg = r**2 * np.cos(θ) * (np.pi / (2*np.pi))
#    assert np.allclose(true_avg, op_avg)
#
#@pytest.mark.parametrize('dtype', [np.float64])
#@pytest.mark.parametrize('Nmax', [15])
#@pytest.mark.parametrize('Lmax', [14])
#@pytest.mark.parametrize('r_inner', [1])
#@pytest.mark.parametrize('r_outer', [1.5, 2])
#def test_shell_phi_average(Nmax, Lmax, r_inner, r_outer, dtype):
#    c, d, b, φ, θ, r = make_shell_basis(Nmax, Lmax, r_inner, r_outer, dtype=dtype)
#    f = field.Field(dist=d, bases=(b,), dtype=dtype)
#    averager = averaging.PhiAverager(f)
#    f['g'] = r**2 * np.sin(φ)*np.cos(θ)
#    op_avg      = averager(f, comm=True)
#    true_avg    = np.zeros_like(op_avg)
#    assert np.allclose(true_avg, op_avg)
#    f['g'] = r**2 * np.sin(φ)**2*np.cos(θ)
#    op_avg      = averager(f, comm=True)
#    true_avg = r**2 * np.cos(θ) * (np.pi / (2*np.pi))
#    assert np.allclose(true_avg, op_avg)
#
#@pytest.mark.parametrize('dtype', [np.float64])
#@pytest.mark.parametrize('Nmax', [15])
#@pytest.mark.parametrize('Lmax', [14])
#@pytest.mark.parametrize('radius', [1, 2])
#def test_ball_phi_theta_average(Nmax, Lmax, radius, dtype):
#    c, d, b, φ, θ, r = make_ball_basis(Nmax, Lmax, radius, dtype=dtype)
#    f = field.Field(dist=d, bases=(b,), dtype=dtype)
#    averager = averaging.PhiThetaAverager(f)
#    f['g'] = r**2 * np.sin(φ)*np.cos(θ)
#    op_avg      = averager(f, comm=True)
#    true_avg    = np.zeros_like(op_avg)
#    assert np.allclose(true_avg, op_avg)
#    f['g'] = r**2 * np.sin(φ)**2 * 3 * np.cos(θ)**2
#    op_avg      = averager(f, comm=True)
#    true_avg = r**2 * (1/2)
#    assert np.allclose(true_avg, op_avg)
#
#@pytest.mark.parametrize('dtype', [np.float64])
#@pytest.mark.parametrize('Nmax', [15])
#@pytest.mark.parametrize('Lmax', [14])
#@pytest.mark.parametrize('r_inner', [1])
#@pytest.mark.parametrize('r_outer', [1.5, 2])
#def test_shell_phi_theta_average(Nmax, Lmax, r_inner, r_outer, dtype):
#    c, d, b, φ, θ, r = make_shell_basis(Nmax, Lmax, r_inner, r_outer, dtype=dtype)
#    f = field.Field(dist=d, bases=(b,), dtype=dtype)
#    averager = averaging.PhiThetaAverager(f)
#    f['g'] = r**2 * np.sin(φ)*np.cos(θ)
#    op_avg      = averager(f, comm=True)
#    true_avg    = np.zeros_like(op_avg)
#    assert np.allclose(true_avg, op_avg)
#    f['g'] = r**2 * np.sin(φ)**2 * 3 * np.cos(θ)**2
#    op_avg      = averager(f, comm=True)
#    true_avg = r**2 * (1/2)
#    assert np.allclose(true_avg, op_avg)
