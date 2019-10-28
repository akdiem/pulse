import matplotlib.pyplot as plt
import dolfin
import pulse
import numpy as np
import sys

if len(sys.argv) > 1:
    gamma_space = sys.argv[1]
else:
    gamma_space = "R_0"

comm = dolfin.MPI.comm_world

# Step-wise loading (for plotting and convergence)
# gammas from pulse_adjoint optimization
pressures = [1.0, 10.0, 12.0, 0.5, 1.0]
gammas = [2.5608192030442395e-10, 0.18940608934909228, 0.22065280788488531,
            0.051917604971265185, 2.5608192030442395e-10]

phase = 10

#first ramp up pressure, then keep constant
iso_relaxation = np.linspace(pressures[-1], pressures[0], phase)
rapid_filling = np.linspace(pressures[0], pressures[1], 2*phase)
reduced_filling = np.linspace(pressures[1], pressures[2], 2*phase)
rapid_ejection = np.linspace(pressures[2], pressures[3], 2*phase)
reduced_ejection = np.linspace(pressures[3], pressures[4], phase)
pressures = np.concatenate((iso_relaxation, rapid_filling, reduced_filling,
                            rapid_ejection, reduced_ejection))

#zero active tension during filling, then increase linearly
active1 = np.linspace(gammas[-1], gammas[0], 3*phase)
active2 = np.linspace(gammas[0], gammas[1], phase)
active3 = np.linspace(gammas[1], gammas[2], phase)
active4 = np.linspace(gammas[2], gammas[3], phase)
active5 = np.linspace(gammas[3], gammas[4], 2*phase)
active =  np.concatenate((active1, active2, active3, active4, active5))

volumes = np.zeros_like(pressures)

geometry = pulse.HeartGeometry.from_file(pulse.mesh_paths['ellipsoid'])

activation = dolfin.Function(dolfin.FunctionSpace(geometry.mesh, "R", 0))
activation.assign(dolfin.Constant(0.0))
matparams = pulse.HolzapfelOgden.default_parameters()
matparams.update({'a': 2.1307174663282646})
material = pulse.HolzapfelOgden(activation=activation,
                                parameters=matparams,
                                active_model="active_stress",
                                eta=0.3,
                                f0=geometry.f0,
                                s0=geometry.s0,
                                n0=geometry.n0)

# LV Pressure
lvp = dolfin.Constant(0.0)
lv_marker = geometry.markers['ENDO'][0]
lv_pressure = pulse.NeumannBC(traction=lvp,
                              marker=lv_marker, name='lv')
neumann_bc = [lv_pressure]

# Add spring term at the base with stiffness 1.0 kPa/cm^2
base_spring = 1.0
robin_bc = [
    pulse.RobinBC(
        value=dolfin.Constant(base_spring), marker=geometry.markers["BASE"][0]
    )
]


# Fix the basal plane in the longitudinal direction
# 0 in V.sub(0) refers to x-direction, which is the longitudinal direction
def fix_basal_plane(W):
    V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
    bc = dolfin.DirichletBC(
        V.sub(0), dolfin.Constant(0.0), geometry.ffun, geometry.markers["BASE"][0]
    )
    return bc


dirichlet_bc = [fix_basal_plane]
# You can also use a built in function for this
# from functools import partial
# dirichlet_bc = partial(pulse.mechanicsproblem.dirichlet_fix_base_directional,
#                        ffun=geometry.ffun,
#                        marker=geometry.markers["BASE"][0])

# Collect boundary conditions
bcs = pulse.BoundaryConditions(dirichlet=dirichlet_bc,
                               neumann=neumann_bc,
                               robin=robin_bc)

# Create the problem
problem = pulse.MechanicsProblem(geometry, material, bcs,
                solver_parameters={'newton_solver': {'linear_solver': 'mumps'}})


us = []
lv_volumes = []
F_ref = None

def compute_volumes(u):
    lvv = geometry.cavity_volume(u=u, chamber="lv")
    lv_volumes.append(lvv)
    print("LVP: {}, LVV: {}".format(plv, lvv))

# us = []
lv_volumes = []
rv_volumes = []
F_ref = None

tf = 0.85
steps = pressure_steps + active_steps + relax_steps
dt = tf/steps
t = 0.0

P1 = dolfin.FunctionSpace(problem.geometry.mesh, "P", 1)
P2 = dolfin.FunctionSpace(problem.geometry.mesh, "P", 2)

ufile = dolfin.XDMFFile(comm, "u.xdmf")
pfile = dolfin.XDMFFile(comm, "pf.xdmf")
for i, (plv, g) in enumerate(zip(pressures, active)):

    pulse.iterate.iterate(problem, lvp, plv)
    pulse.iterate.iterate(problem, activation, g, max_iters=100)

    u, pu = problem.state.split()
    F = dolfin.variable(pulse.kinematics.DeformationGradient(u))
    E = dolfin.variable(pulse.kinematics.GreenLagrangeStrain(F))

    # p = dolfin.project(dolfin.tr(E), P1fine)
    # since solid is quadratic and pressure will be linear we can refine it
    p = dolfin.project(dolfin.inner(
                        dolfin.diff(
                            problem.material.strain_energy(F), F), F.T), P1)



    ufile.write_checkpoint(u, 'du', t)
    pfile.write_checkpoint(p, 'p', t)
    # us.append(u)
    compute_volumes(u)

    t += dt
    append = True

ufile.close()
pfile.close()

time = np.linspace(0, tf, len(lv_volumes))
np.savetxt('lv_volume.csv', lv_volumes, delimiter=',')
