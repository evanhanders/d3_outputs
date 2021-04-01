"""
Unit tests which ensure that d3_outputs operations work in serial.
"""
import pytest
import numpy as np
import functools

from dedalus.core import coords, distributor, basis, field, operators
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer
import d3_outputs.extra_ops as extra_ops

def make_ball_basis(Nmax, Lmax, radius, dtype=np.float64, dealias=(1,1,1)):
    c    = coords.SphericalCoordinates('φ', 'θ', 'r')
    d    = distributor.Distributor((c,), mesh=None)
    b    = basis.BallBasis(c, (2*(Lmax+2), Lmax+1, Nmax+1), radius=radius, dtype=dtype, dealias=dealias)
    φ,  θ,  r  = b.local_grids(dealias)
    return c, d, b, φ, θ, r

def make_shell_basis(Nmax, Lmax, r_inner, r_outer, dtype=np.float64, dealias=(1,1,1)):
    c    = coords.SphericalCoordinates('φ', 'θ', 'r')
    d    = distributor.Distributor((c,), mesh=None)
    b    = basis.SphericalShellBasis(c, (2*(Lmax+2), Lmax+1, Nmax+1), radii=(r_inner, r_outer), dtype=dtype, dealias=dealias)
    φ,  θ,  r  = b.local_grids(dealias)
    return c, d, b, φ, θ, r

def make_ballShell_basis(NmaxB, NmaxS, Lmax, r_inner, r_outer, dtype=np.float64, dealias=(1,1,1)):
    c, d, bB, φB, θB, rB = make_ball_basis(NmaxB, Lmax, r_inner, dtype=dtype, dealias=dealias)
    bS   = basis.SphericalShellBasis(c, (2*(Lmax+2), Lmax+1, NmaxS+1), radii=(r_inner, r_outer), dtype=dtype, dealias=dealias)
    φS,  θS,  rS  = bS.local_grids(dealias)
    return c, d, bB, bS, φB, θB, rB, φS, θS, rS

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [14])
@pytest.mark.parametrize('radius', [1, 2])
@pytest.mark.parametrize('dealias', [(1,1,1), (1.5,1.5,1.5), (1,1.5,1.5)])
def test_ball_volume_average(Nmax, Lmax, radius, dtype, dealias):
    c, d, b, φ, θ, r = make_ball_basis(Nmax, Lmax, radius, dtype=dtype, dealias=dealias)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.require_scales(dealias)
    f['g'] = r**2
    vol_averager = extra_ops.BallVolumeAverager(f)
    volume      = ((4/3)*np.pi*radius**3)
    true_avg    = (4/5)*np.pi*radius**5 / volume
    op_avg      = vol_averager(f, comm=True)
    assert np.allclose(true_avg, op_avg)

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [14])
@pytest.mark.parametrize('r_inner', [0.1, 0.5])
@pytest.mark.parametrize('r_outer', [1, 2])
@pytest.mark.parametrize('dealias', [(1,1,1), (1.5,1.5,1.5), (1,1.5,1.5)])
def test_shell_volume_average(Nmax, Lmax, r_inner, r_outer, dtype, dealias):
    c, d, b, φ, θ, r = make_shell_basis(Nmax, Lmax, r_inner, r_outer, dtype=dtype, dealias=dealias)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.require_scales(dealias)
    f['g'] = r**2
    vol_averager = extra_ops.ShellVolumeAverager(f)
    volume      = (4/3)*np.pi*(r_outer**3 - r_inner**3)
    true_avg    = (4/5)*np.pi*(r_outer**5 - r_inner**5) / volume
    op_avg      = vol_averager(f, comm=True)
    assert np.allclose(true_avg, op_avg)

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('NmaxB', [15])
@pytest.mark.parametrize('NmaxS', [15])
@pytest.mark.parametrize('Lmax', [14])
@pytest.mark.parametrize('r_inner', [1])
@pytest.mark.parametrize('r_outer', [1.5, 2])
@pytest.mark.parametrize('dealias', [(1,1,1), (1.5,1.5,1.5), (1,1.5,1.5)])
def test_ballShell_volume_average(NmaxB, NmaxS, Lmax, r_inner, r_outer, dtype, dealias):
    c, d, bB, bS, φB, θB, rB, φS, θS, rS = make_ballShell_basis(NmaxB, NmaxS, Lmax, r_inner, r_outer, dtype=dtype, dealias=dealias)
    fB = field.Field(dist=d, bases=(bB,), dtype=dtype)
    fS = field.Field(dist=d, bases=(bS,), dtype=dtype)
    fB.require_scales(dealias)
    fS.require_scales(dealias)
    fB['g'] = rB**2
    fS['g'] = rS**2
    vol_averager = extra_ops.BallShellVolumeAverager(fB, fS)
    volume      = ((4/3)*np.pi*r_outer**3)
    true_avg    = (4/5)*np.pi*r_outer**5 / volume
    op_avg      = vol_averager(fB, fS, comm=True)
    assert np.allclose(true_avg, op_avg)

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [14])
@pytest.mark.parametrize('radius', [1, 2])
@pytest.mark.parametrize('dealias', [(1,1,1), (1.5,1.5,1.5), (1,1.5,1.5)])
def test_ball_phi_average(Nmax, Lmax, radius, dtype, dealias):
    c, d, b, φ, θ, r = make_ball_basis(Nmax, Lmax, radius, dtype=dtype, dealias=dealias)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.require_scales(dealias)
    averager = extra_ops.PhiAverager(f)
    f['g'] = r**2 * np.sin(φ)*np.cos(θ)
    op_avg      = averager(f, comm=True)
    true_avg    = np.zeros_like(op_avg)
    assert np.allclose(true_avg, op_avg)
    f['g'] = r**2 * np.sin(φ)**2*np.cos(θ)
    op_avg      = averager(f, comm=True)
    true_avg = r**2 * np.cos(θ) * (np.pi / (2*np.pi))
    assert np.allclose(true_avg, op_avg)

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [14])
@pytest.mark.parametrize('r_inner', [1])
@pytest.mark.parametrize('r_outer', [1.5, 2])
@pytest.mark.parametrize('dealias', [(1,1,1), (1.5,1.5,1.5), (1,1.5,1.5)])
def test_shell_phi_average(Nmax, Lmax, r_inner, r_outer, dtype, dealias):
    c, d, b, φ, θ, r = make_shell_basis(Nmax, Lmax, r_inner, r_outer, dtype=dtype, dealias=dealias)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.require_scales(dealias)
    averager = extra_ops.PhiAverager(f)
    f['g'] = r**2 * np.sin(φ)*np.cos(θ)
    op_avg      = averager(f, comm=True)
    true_avg    = np.zeros_like(op_avg)
    assert np.allclose(true_avg, op_avg)
    f['g'] = r**2 * np.sin(φ)**2*np.cos(θ)
    op_avg      = averager(f, comm=True)
    true_avg = r**2 * np.cos(θ) * (np.pi / (2*np.pi))
    assert np.allclose(true_avg, op_avg)

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [14])
@pytest.mark.parametrize('radius', [1, 2])
@pytest.mark.parametrize('dealias', [(1,1,1), (1.5,1.5,1.5), (1,1.5,1.5)])
def test_ball_phi_theta_average(Nmax, Lmax, radius, dtype, dealias):
    c, d, b, φ, θ, r = make_ball_basis(Nmax, Lmax, radius, dtype=dtype, dealias=dealias)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.require_scales(dealias)
    averager = extra_ops.PhiThetaAverager(f)
    f['g'] = r**2 * np.sin(φ)*np.cos(θ)
    op_avg      = averager(f, comm=True)
    true_avg    = np.zeros_like(op_avg)
    assert np.allclose(true_avg, op_avg)
    f['g'] = r**2 * np.sin(φ)**2 * 3 * np.cos(θ)**2
    op_avg      = averager(f, comm=True)
    true_avg = r**2 * (1/2)
    assert np.allclose(true_avg, op_avg)

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [14])
@pytest.mark.parametrize('r_inner', [1])
@pytest.mark.parametrize('r_outer', [1.5, 2])
@pytest.mark.parametrize('dealias', [(1,1,1), (1.5,1.5,1.5), (1,1.5,1.5)])
def test_shell_phi_theta_average(Nmax, Lmax, r_inner, r_outer, dtype, dealias):
    c, d, b, φ, θ, r = make_shell_basis(Nmax, Lmax, r_inner, r_outer, dtype=dtype, dealias=dealias)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.require_scales(dealias)
    averager = extra_ops.PhiThetaAverager(f)
    f['g'] = r**2 * np.sin(φ)*np.cos(θ)
    op_avg      = averager(f, comm=True)
    true_avg    = np.zeros_like(op_avg)
    assert np.allclose(true_avg, op_avg)
    f['g'] = r**2 * np.sin(φ)**2 * 3 * np.cos(θ)**2
    op_avg      = averager(f, comm=True)
    true_avg = r**2 * (1/2)
    assert np.allclose(true_avg, op_avg)

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [14])
@pytest.mark.parametrize('radius', [1, 2])
@pytest.mark.parametrize('dealias', [(1,1,1), (1.5,1.5,1.5), (1,1.5,1.5)])
def test_ball_S2_outputter(Nmax, Lmax, radius, dtype, dealias):
    c, d, b, φ, θ, r = make_ball_basis(Nmax, Lmax, radius, dtype=dtype, dealias=dealias)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.require_scales(dealias)
    interp_op = f(r=0.5*radius)
    averager = extra_ops.OutputRadialInterpolate(f, interp_op)
    f['g'] = r**2 * np.sin(φ)*np.cos(θ)
    op_avg      = averager(interp_op.evaluate(), comm=True)
    true_avg    = interp_op.evaluate()['g']
    f.require_scales(dealias)
    assert np.allclose(true_avg, op_avg)
    f['g'] = r**2 * np.sin(φ)**2 * 3 * np.cos(θ)**2
    op_avg      = averager(interp_op.evaluate(), comm=True)
    true_avg = interp_op.evaluate()['g']
    assert np.allclose(true_avg, op_avg)

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [14])
@pytest.mark.parametrize('radius', [1, 2])
@pytest.mark.parametrize('dealias', [(1,1,1), (1.5,1.5,1.5), (1,1.5,1.5)])
def test_ball_equator_slice(Nmax, Lmax, radius, dtype, dealias):
    c, d, b, φ, θ, r = make_ball_basis(Nmax, Lmax, radius, dtype=dtype, dealias=dealias)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.require_scales(dealias)
    f['g'] = 3 * r**2 * np.sin(φ)**2 * (1 - np.cos(θ)**2)
    slicer = extra_ops.EquatorSlicer(f)
    op_slice      = slicer(f, comm=True)
    true_slice = 3 * r**2 * np.sin(φ)**2
    assert np.allclose(true_slice, op_slice)

@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [14])
@pytest.mark.parametrize('radius', [1, 2])
@pytest.mark.parametrize('dealias', [(1,1,1), (1.5,1.5,1.5), (1,1.5,1.5)])
def test_ball_meridion_slice(Nmax, Lmax, radius, dtype, dealias):
    c, d, b, φ, θ, r = make_ball_basis(Nmax, Lmax, radius, dtype=dtype, dealias=dealias)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.require_scales(dealias)
    f['g'] = 3 * r**2 * np.sin(φ)**2 * (1 - np.cos(θ)**2)
    for phi_target in [0, np.pi/2, np.pi, 3*np.pi/2]:
        slicer = extra_ops.MeridionSlicer(f, phi_target=phi_target)
        op_slice      = slicer(f, comm=True)
        true_slice = 3 * r**2 * np.sin(phi_target)**2 * (1 - np.cos(θ)**2)
        assert np.allclose(true_slice, op_slice)
