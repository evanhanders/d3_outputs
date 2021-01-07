
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer
from scipy import sparse
import dedalus_sphere
import time
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.config import config
config['linear algebra']['MATRIX_FACTORIZER'] = 'SuperLUNaturalFactorizedTranspose'


# Parameters
radius = 1
Lmax = 14
L_dealias = 1
Nmax = 15
N_dealias = 1
dt = 1.5e-4
t_end = 0.5
ts = timesteppers.SBDF4
dtype = np.float64

comm = MPI.COMM_WORLD
rank = comm.rank
ncpu = comm.size
log2 = np.log2(ncpu)
if log2 == int(log2):
    mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
else:
    mesh = None
logger.info("running on processor mesh={}".format(mesh))

Ekman = 3e-4
Rayleigh = 95
Prandtl = 1

# Bases
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor((c,), mesh=mesh)
b = basis.BallBasis(c, (2*(Lmax+2), Lmax+1, Nmax+1), radius=radius, dtype=dtype)
b_S2 = b.S2_basis()
phi, theta, r = b.local_grids((1, 1, 1))

# Fields
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
p = field.Field(dist=d, bases=(b,), dtype=dtype)
T = field.Field(dist=d, bases=(b,), dtype=dtype)
tau_u = field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=dtype)
tau_T = field.Field(dist=d, bases=(b_S2,), dtype=dtype)

r_vec = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
r_vec['g'][2] = r

T['g'] = 0.5*(1-r**2) + 0.1/8*np.sqrt(35/np.pi)*r**3*(1-r**2)*(np.cos(3*phi)+np.sin(3*phi))*np.sin(theta)**3

T_source = field.Field(dist=d, bases=(b,), dtype=dtype)
T_source['g'] = 3

# Boundary conditions
u_r_bc = operators.RadialComponent(operators.interpolate(u,r=1))

stress = operators.Gradient(u, c) + operators.TransposeComponents(operators.Gradient(u, c))
u_perp_bc = operators.RadialComponent(operators.AngularComponent(operators.interpolate(stress,r=1), index=1))

# Parameters and operators
ez = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)
div = lambda A: operators.Divergence(A, index=0)
lap = lambda A: operators.Laplacian(A, c)
grad = lambda A: operators.Gradient(A, c)
curl = lambda A: operators.Curl(A)
dot = lambda A, B: arithmetic.DotProduct(A, B)
cross = lambda A, B: arithmetic.CrossProduct(A, B)
ddt = lambda A: operators.TimeDerivative(A)
LiftTau = lambda A: operators.LiftTau(A, b, -1)
angComp = lambda A, i=0: operators.AngularComponent(A, i)

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([p, u, T, tau_u, tau_T])

problem.add_equation(eq_eval("div(u) = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("p = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("Ekman*ddt(u) - Ekman*lap(u) + grad(p) + LiftTau(tau_u) = - Ekman*dot(u,grad(u)) + Rayleigh*r_vec*T - cross(ez, u)"), condition = "ntheta != 0")
problem.add_equation(eq_eval("u = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("Prandtl*ddt(T) - lap(T) + LiftTau(tau_T) = - Prandtl*dot(u,grad(T)) + T_source"))
problem.add_equation(eq_eval("u_r_bc = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("u_perp_bc = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("tau_u = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("T(r=1) = 0"))
print("Problem built")

# Solver
solver = solvers.InitialValueSolver(problem, ts)
solver.stop_sim_time = t_end

# Analysis
from d3_outputs import extra_ops
from d3_outputs.writing      import d3FileHandler
output_dir = './test_outputs/'
if MPI.COMM_WORLD.rank == 0:
    import os
    if not os.path.exists('{:s}/'.format(output_dir)):
        os.makedirs('{:s}/'.format(output_dir))
    logdir = os.path.join(output_dir,'logs')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
vol_averager       = extra_ops.BallVolumeAverager(p)
azimuthal_averager = extra_ops.PhiAverager(p)
radial_profile_averager = extra_ops.PhiThetaAverager(p)
eq_slicer = extra_ops.EquatorSlicer(p)
mer_slicer1 = extra_ops.MeridionSlicer(p, phi_target=0)
mer_slicer2 = extra_ops.MeridionSlicer(p, phi_target=np.pi)

z_vort = dot(ez, curl(u))
z_vort.store_last = True

scalars = d3FileHandler(solver, '{:s}/scalar'.format(output_dir), max_writes=np.inf, iter=10)
scalars.add_task(0.5*dot(u, u), extra_op=vol_averager, name='KE', layout='g', extra_op_comm=True) #extra_op_comm=True so we only write one scalar file rather than Ncpu files

ORI = extra_ops.OutputRadialInterpolate
slices = d3FileHandler(solver, '{:s}/slices'.format(output_dir), max_writes=40, sim_dt=0.05)
slices.add_task(T,      extra_op=eq_slicer, name='T_eq', layout='g', extra_op_comm=False)
slices.add_task(z_vort, extra_op=eq_slicer, name='z_vort_eq', layout='g', extra_op_comm=False)
slices.add_task(u,      extra_op=eq_slicer, name='u_eq', layout='g', extra_op_comm=False)
slices.add_task(T,      extra_op=azimuthal_averager, name='T_az_avg', layout='g', extra_op_comm=False)
slices.add_task(z_vort, extra_op=azimuthal_averager, name='z_vort_az_avg', layout='g', extra_op_comm=False)
slices.add_task(u,      extra_op=azimuthal_averager, name='u_az_avg', layout='g', extra_op_comm=False)
slices.add_task(T, extra_op=mer_slicer1, name='T(phi=0)', layout='g', extra_op_comm=False)
slices.add_task(T, extra_op=mer_slicer2, name='T(phi=pi)', layout='g', extra_op_comm=False)
slices.add_task(u, extra_op=mer_slicer1, name='u(phi=0)', layout='g', extra_op_comm=False)
slices.add_task(u, extra_op=mer_slicer2, name='u(phi=pi)', layout='g', extra_op_comm=False)
slices.add_task(dot(ez, curl(u)), extra_op=mer_slicer1, name='z_vort(phi=0)', layout='g', extra_op_comm=False)
slices.add_task(dot(ez, curl(u)), extra_op=mer_slicer2, name='z_vort(phi=pi)', layout='g', extra_op_comm=False)
slices.add_task(T(r=0.95), extra_op=ORI(T, T(r=0.95)), name='T_r0.95', layout='g')
slices.add_task(dot(ez, curl(u))(r=0.95), extra_op=ORI(T, dot(ez, curl(u))(r=0.95)), name='z_vort_r0.95', layout='g')
slices.add_task(angComp(u(r=0.95)), extra_op=ORI(T, angComp(u(r=0.95))), name='u_S2_r0.95', layout='g')
slices.add_task(u(r=0.95), extra_op=ORI(T, u(r=0.95)), name='u_r0.95', layout='g')

profiles = d3FileHandler(solver, '{:s}/profiles'.format(output_dir), max_writes=40, sim_dt=0.05)
profiles.add_task(T,      extra_op=radial_profile_averager, name='T_eq', layout='g', extra_op_comm=True)
profiles.add_task(z_vort, extra_op=radial_profile_averager, name='z_vort_eq', layout='g', extra_op_comm=True)
profiles.add_task(u,      extra_op=radial_profile_averager, name='u_eq', layout='g', extra_op_comm=True)

checkpoint = d3FileHandler(solver, '{:s}/checkpoint'.format(output_dir), max_writes=2, iter=1000)
checkpoint.add_task(T, name='T', scales=1, layout='c')
checkpoint.add_task(u, name='u', scales=1, layout='c')

analysis_tasks = [scalars, slices, profiles]


# Main loop
start_time = time.time()
while solver.ok:
    solver.step(dt)
    if solver.iteration % 10 == 0:
        E0 = vol_averager.volume*vol_averager(scalars.task_dict['KE']['out'], comm=True)
        logger.info("t = %f, E = %e" %(solver.sim_time, E0))
end_time = time.time()
print('Run time:', end_time-start_time)
print('mering data')

from d3_outputs import post
for t in analysis_tasks:
    post.merge_analysis(t.base_path)
