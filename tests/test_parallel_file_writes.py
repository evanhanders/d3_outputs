"""
Unit tets which ensure that output operations properly write to file in parallel.
"""
import pytest
import numpy as np
import functools

from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer
import d3_outputs.averaging as averaging
from d3_outputs.writing import d3FileHandler
from d3_outputs import file_merging
import h5py

def make_ball_basis(Nmax, Lmax, radius, dtype=np.float64, dealias=1, mesh=None):
    c    = coords.SphericalCoordinates('φ', 'θ', 'r')
    d    = distributor.Distributor((c,), mesh=mesh)
    b    = basis.BallBasis(c, (2*(Lmax+2), Lmax+1, Nmax+1), radius=radius, dtype=dtype)
    φ,  θ,  r  = b.local_grids((dealias, dealias, dealias))
    return c, d, b, φ, θ, r

def create_simple_ball_ivp(Nmax, Lmax, radius, dtype, mesh, vector=False):
    c, d, b, φ, θ, r = make_ball_basis(Nmax, Lmax, radius, dtype=dtype, mesh=mesh)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    if vector:
        sim_f = field.Field(dist=d, bases=(b,), dtype=dtype, tensorsig=(c,))
        tau_f = field.Field(dist=d, bases=(b.S2_basis(),), dtype=dtype, tensorsig=(c,))
    else:
        sim_f = field.Field(dist=d, bases=(b,), dtype=dtype)
        tau_f = field.Field(dist=d, bases=(b.S2_basis(),), dtype=dtype)
    x = r * np.sin(θ) * np.cos(φ)
    y = r * np.sin(θ) * np.sin(φ)
    z = r * np.cos(θ)
    f['g'] = (radius**2 - x**2 + y**2 + z**2) * np.sin(2*np.pi*x)**2 * np.sin(2*np.pi*z)**2
    if vector:
        grad      = lambda A: operators.Gradient(A, c)
        forcing = grad(f)
    else:
        forcing = f
        
    lap  = lambda A: operators.Laplacian(A, c)
    ddt = lambda A: operators.TimeDerivative(A)
    LiftTau = lambda A: operators.LiftTau(A, b, -1)
    # Problem
    problem = problems.IVP([sim_f, tau_f])
    if vector:
        problem.add_equation((ddt(sim_f) - lap(sim_f) + LiftTau(tau_f), forcing), condition="nθ != 0")
        problem.add_equation((sim_f, 0), condition="nθ == 0")
        problem.add_equation((sim_f(r=radius), 0), condition="nθ != 0")
        problem.add_equation((tau_f, 0), condition="nθ == 0")
    else:
        problem.add_equation((ddt(sim_f) - lap(sim_f) + LiftTau(tau_f), forcing))
        problem.add_equation((sim_f(r=radius), 0))
    # Solver
    solver = solvers.InitialValueSolver(problem, timesteppers.SBDF1, matrix_coupling=(False, False, True))
    return c, b, d, φ, θ, r, x, y, z, sim_f, solver


@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('Nmax', [7])
@pytest.mark.parametrize('Lmax', [6])
@pytest.mark.parametrize('radius', [1])
@pytest.mark.parametrize('mesh', [[2,2], [1,4], [4,1]])
@pytest.mark.parametrize('vector', [True, False])
def test_ball_volume_average(Nmax, Lmax, radius, dtype, mesh, vector):
    c, b, d, φ, θ, r, x, y, z, sim_f, solver = create_simple_ball_ivp(Nmax, Lmax, radius, dtype, mesh, vector=vector)
    #Define averager and output tasks
    vol_averager = averaging.BallVolumeAverager(sim_f)
    outputs = d3FileHandler(solver, './testing/', iter=1, max_writes=np.inf)
    outputs.add_task(sim_f, extra_op=vol_averager, name='sim_f', layout='g')
    #Store outputs as the sim timesteps
    times = []
    sim_averages = []
    dt = 1
    for i in range(10):
        times.append(solver.sim_time)
        sim_averages.append(np.copy(vol_averager(sim_f, comm=True)))
        solver.step(dt)
    sim_time = np.array(times)
    sim_averages = np.array(sim_averages)
    #Merge files and pull out file writes
    file_merging.merge_analysis(outputs.base_path)
    d.comm_cart.Barrier()
    with h5py.File('./testing/testing_s1.h5', 'r') as f:
        file_time = f['scales/sim_time'][()]
        file_averages = f['tasks/sim_f'][()]
    #Compare
    assert np.allclose(sim_time, file_time)
    assert np.allclose(sim_averages, file_averages)

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('Nmax', [7])
@pytest.mark.parametrize('Lmax', [6])
@pytest.mark.parametrize('radius', [1])
@pytest.mark.parametrize('mesh', [[2,2], [1,4], [4,1]])
@pytest.mark.parametrize('vector', [True, False])
def test_ball_phi_average(Nmax, Lmax, radius, dtype, mesh, vector):
    c, b, d, φ, θ, r, x, y, z, sim_f, solver = create_simple_ball_ivp(Nmax, Lmax, radius, dtype, mesh, vector=vector)
    #Define averager and output tasks
    phi_averager = averaging.PhiAverager(sim_f)
    outputs = d3FileHandler(solver, './testing/', iter=1, max_writes=np.inf)
    outputs.add_task(sim_f, extra_op=phi_averager, name='sim_f', layout='g')
    #Store outputs as the sim timesteps
    times = []
    sim_averages = []
    dt = 1
    for i in range(10):
        times.append(solver.sim_time)
        sim_averages.append(np.copy(phi_averager(sim_f, comm=True)))
        solver.step(dt)
    sim_time = np.array(times)
    sim_averages = np.array(sim_averages)
    #Merge files and pull out file writes
    file_merging.merge_analysis(outputs.base_path)
    d.comm_cart.Barrier()
    with h5py.File('./testing/testing_s1.h5', 'r') as f:
        file_time = f['scales/sim_time'][()]
        file_averages = f['tasks/sim_f'][()]
    #Compare
    assert np.allclose(sim_time, file_time)
    assert np.allclose(sim_averages, file_averages)

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('Nmax', [7])
@pytest.mark.parametrize('Lmax', [6])
@pytest.mark.parametrize('radius', [1])
@pytest.mark.parametrize('mesh', [[2,2], [1,4], [4,1]])
@pytest.mark.parametrize('vector', [True, False])
def test_ball_phitheta_average(Nmax, Lmax, radius, dtype, mesh, vector):
    c, b, d, φ, θ, r, x, y, z, sim_f, solver = create_simple_ball_ivp(Nmax, Lmax, radius, dtype, mesh, vector=vector)
    #Define averager and output tasks
    phitheta_averager = averaging.PhiThetaAverager(sim_f)
    outputs = d3FileHandler(solver, './testing/', iter=1, max_writes=np.inf)
    outputs.add_task(sim_f, extra_op=phitheta_averager, name='sim_f', layout='g')
    #Store outputs as the sim timesteps
    times = []
    sim_averages = []
    dt = 1
    for i in range(10):
        times.append(solver.sim_time)
        sim_averages.append(np.copy(phitheta_averager(sim_f, comm=True)))
        solver.step(dt)
    sim_time = np.array(times)
    sim_averages = np.array(sim_averages)
    #Merge files and pull out file writes
    file_merging.merge_analysis(outputs.base_path)
    d.comm_cart.Barrier()
    with h5py.File('./testing/testing_s1.h5', 'r') as f:
        file_time = f['scales/sim_time'][()]
        file_averages = f['tasks/sim_f'][()]
    #Compare
    assert np.allclose(sim_time, file_time)
    assert np.allclose(sim_averages, file_averages)

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('Nmax', [7])
@pytest.mark.parametrize('Lmax', [6])
@pytest.mark.parametrize('radius', [1])
@pytest.mark.parametrize('mesh', [[2,2], [1,4], [4,1]])
@pytest.mark.parametrize('vector', [True, False])
def test_ball_equator_slicer(Nmax, Lmax, radius, dtype, mesh, vector):
    c, b, d, φ, θ, r, x, y, z, sim_f, solver = create_simple_ball_ivp(Nmax, Lmax, radius, dtype, mesh, vector=vector)
    #Define averager and output tasks
    equator_slicer = averaging.EquatorSlicer(sim_f)
    outputs = d3FileHandler(solver, './testing/', iter=1, max_writes=np.inf)
    outputs.add_task(sim_f, extra_op=equator_slicer, name='sim_f', layout='g')
    #Store outputs as the sim timesteps
    times = []
    sim_averages = []
    dt = 1
    for i in range(10):
        times.append(solver.sim_time)
        sim_averages.append(np.copy(equator_slicer(sim_f, comm=True)))
        solver.step(dt)
    sim_time = np.array(times)
    sim_averages = np.array(sim_averages)
    #Merge files and pull out file writes
    file_merging.merge_analysis(outputs.base_path)
    d.comm_cart.Barrier()
    with h5py.File('./testing/testing_s1.h5', 'r') as f:
        file_time = f['scales/sim_time'][()]
        file_averages = f['tasks/sim_f'][()]
    #Compare
    assert np.allclose(sim_time, file_time)
    assert np.allclose(sim_averages, file_averages)

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('Nmax', [7])
@pytest.mark.parametrize('Lmax', [6])
@pytest.mark.parametrize('radius', [1])
@pytest.mark.parametrize('mesh', [[2,2], [1,4], [4,1]])
@pytest.mark.parametrize('vector', [True, False])
def test_ball_shell_slicer(Nmax, Lmax, radius, dtype, mesh, vector):
    c, b, d, φ, θ, r, x, y, z, sim_f, solver = create_simple_ball_ivp(Nmax, Lmax, radius, dtype, mesh, vector=vector)
    #Define averager and output tasks
    interp_operation = sim_f(r=0.5*radius)
    shell_slicer = averaging.OutputRadialInterpolate(sim_f, interp_operation)
    outputs = d3FileHandler(solver, './testing/', iter=1, max_writes=np.inf)
    outputs.add_task(interp_operation, extra_op=shell_slicer, name='sim_f', layout='g')
    #Store outputs as the sim timesteps
    times = []
    sim_averages = []
    dt = 1
    for i in range(10):
        times.append(solver.sim_time)
        sim_averages.append(np.copy(shell_slicer(interp_operation.evaluate(), comm=True)))
        solver.step(dt)
    sim_time = np.array(times)
    sim_averages = np.array(sim_averages)
    #Merge files and pull out file writes
    file_merging.merge_analysis(outputs.base_path)
    d.comm_cart.Barrier()
    with h5py.File('./testing/testing_s1.h5', 'r') as f:
        file_time = f['scales/sim_time'][()]
        file_averages = f['tasks/sim_f'][()]
    #Compare
    assert np.allclose(sim_time, file_time)
    assert np.allclose(sim_averages, file_averages)


