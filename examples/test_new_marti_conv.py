

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
Lmax = 15
L_dealias = 1
Nmax = 15
N_dealias = 1
dt = 8e-5
t_end = 1
ts = timesteppers.SBDF2
dtype = np.float64
mesh = None

Ekman = 3e-4
Rayleigh = 95
Prandtl = 1

# Bases
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor((c,), mesh=mesh)
b = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), radius=radius, dtype=dtype)
bk2 = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), k=2, radius=radius, dtype=dtype)
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
from d3_outputs.averaging    import BallVolumeAverager, PhiAverager, PhiThetaAverager, EquatorSlicer, SphericalShellCommunicator
from d3_outputs.writing      import d3FileHandler 
output_dir = './'
vol_averager       = BallVolumeAverager(p)
azimuthal_averager = PhiAverager(p)
radialProfile_averager = PhiThetaAverager(p)
eq_slicer = EquatorSlicer(p)
shell_comm = SphericalShellCommunicator(p)

scalars = d3FileHandler(solver, vol_averager, '{:s}/scalar'.format(output_dir), max_writes=np.inf, iter=10)
scalars.add_task(0.5*dot(u, u), name='KE', layout='g')

equatorial = d3FileHandler(solver, eq_slicer, '{:s}/eq_slice'.format(output_dir), max_writes=40, sim_dt=0.05)
meridional = d3FileHandler(solver, azimuthal_averager, '{:s}/mer_slice'.format(output_dir), max_writes=40, sim_dt=0.05)
profile    = d3FileHandler(solver, radialProfile_averager, '{:s}/profiles'.format(output_dir), max_writes=40, sim_dt=0.05)
for handler in [equatorial, meridional, profile]:
    handler.add_task(T, name='T', layout='g')
    handler.add_task(dot(ez, curl(u)), name='z_vort', layout='g')
    handler.add_task(u, name='u', layout='g')

shell = d3FileHandler(solver, shell_comm, '{:s}/shell_slice'.format(output_dir), max_writes=40, sim_dt=0.05)
shell.add_task(T(r=0.95), name='T_r0.95', layout='g')
shell.add_task(dot(ez, curl(u))(r=0.95), name='z_vort_r0.95', layout='g')
shell.add_task(angComp(u(r=0.95)), name='u_S2', layout='g')
shell.add_task(u(r=0.95), name='u_vector', layout='g')

# Main loop
start_time = time.time()
while solver.ok:
    solver.step(dt)
    if solver.iteration % 10 == 0:
        E0 = vol_averager.volume*scalars.write_tasks['KE']
        logger.info("t = %f, E = %e" %(solver.sim_time, E0))
end_time = time.time()
print('Run time:', end_time-start_time)
