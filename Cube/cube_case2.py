from dolfin import *
import numpy as np
from numpy import linalg as LA
import time
import sys
import argparse
import glob
import re
import math
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import matplotlib.pyplot as plt 
from mshr import *
from scipy.integrate import ode 
from petsc4py import PETSc
from mpi4py import MPI
import ufl
import os

#MPI definitions
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


#----------------------------------------------------------------------------------------------------
#FORM COMPILER SETTINGS
#----------------------------------------------------------------------------------------------------

parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['representation'] = "uflacs"
parameters['form_compiler']['quadrature_degree'] = 5
parameters['form_compiler']['optimize'] = True
ffc_options = {'optimize' : True,
               'eliminate_zeros' : True,
               'precompute_basis_const' : True,
               'precompute_ip_const' : True}

#----------------------------------------------------------------------------------------------------
#I/O DEFINITIONS
#----------------------------------------------------------------------------------------------------

output_dir = os.path.splitext(os.path.dirname(__file__))[0] + '/'
file_name = os.path.splitext(os.path.basename(__file__))[0]
file_name = file_name + '_10years'

#----------------------------------------------------------------------------------------------------
#MESH INITIALIZATION
#----------------------------------------------------------------------------------------------------

#Begin mesh values
x0 = 0.0
y0 = 0.0
z0 = 0.0

#End mesh values
x1 = 0.5
y1 = 0.5
z1 = 1.0

#mesh discretization
n_x = 8
n_y = 8
n_z = 20

bmesh = BoxMesh(comm,Point(x0,y0,z0),Point(x1,y1,z1),n_x,n_y,n_z)

File(comm, output_dir + file_name + "_box.pvd") << bmesh

#----------------------------------------------------------------------------------------------------
#FUNCTION SPACE INITIALIZATION
#----------------------------------------------------------------------------------------------------

n_elem = bmesh.num_cells()
Vs_cg = FunctionSpace(bmesh,'CG',2)
Vs_dg = FunctionSpace(bmesh,'DG',0)
Vv = VectorFunctionSpace(bmesh,'Lagrange',2)
Vt = TensorFunctionSpace(bmesh,'Lagrange',2)

local_range = Vs_cg.dofmap().ownership_range()
local_dim = local_range[1] - local_range[0]

#----------------------------------------------------------------------------------------------------
#BC INITIALIZATION
#----------------------------------------------------------------------------------------------------

class x_axis(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[1]) < DOLFIN_EPS

class y_axis(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0]) < DOLFIN_EPS

class bottom_face(SubDomain):
    def inside(self, x, on_boundary):
        return x[2] < DOLFIN_EPS and on_boundary

sub_domains = MeshFunction("size_t", bmesh, bmesh.topology().dim() - 1)

#symmetry bcs
#fix the y axis in the x direction
y_ax_fix = y_axis()
y_ax_fix.mark(sub_domains, 0)

#fix the x axis in the y direction
x_ax_fix = x_axis()
x_ax_fix.mark(sub_domains, 1)

#fixing to surface
zfix = bottom_face()
zfix.mark(sub_domains, 2)

file_BC_default = File(comm, output_dir + file_name + "_subdomains.pvd")
file_BC_default << sub_domains

# Symmetry BC: Fix y axis in the x direction
bcx = DirichletBC(Vv.sub(0), Constant(0.0), y_ax_fix)
# Symmetry BC: Fix x axis in the y direction
bcy = DirichletBC(Vv.sub(1), Constant(0.0), x_ax_fix)
# Fix bottom of geometry in z direction
bcz = DirichletBC(Vv.sub(2), Constant(0.0), zfix)
bcs_b = [bcx,bcy,bcz]

#----------------------------------------------------------------------------------------------------
#CHEMICAL SPECIES INITIALIZATION AND ODE SYSTEM DEFINITION
#----------------------------------------------------------------------------------------------------

#initialize vector for f_A
A_source_numpy = np.zeros((local_dim))
coordinate = Vs_cg.tabulate_dof_coordinates()

#gamma
growth_mult = 1.0
#growth rate initialization
growth_rate = Function(Vs_cg)
growth_rate.vector().set_local(A_source_numpy * growth_mult)
growth_rate.vector().apply('insert')

#definition of f_A
n = 2.0
b = 0.3
kd = 0.25
kda = 1.0
fs = 0.25
s0 = 0.075
a_sig = 5.0
b_sig = 0.4

for i in range (local_dim):
    x_point=coordinate[i][0]
    y_point=coordinate[i][1]
    top = pow(abs(x_point) + abs(y_point),n)
    A_source_numpy[i] = exp(-top / b)

def Stressfunc(ts):
    s = 1.0 + a_sig / (1.0 + exp(-(ts - b_sig) / s0))
    return s

# ODE for chemical species
def ODEfunc(t,y,fA,k):
    A = 0
    B = 1
    C = 2
    dydt = np.zeros(3)
    tr = ctrace.vector()[i]
    s = Stressfunc(tr)
    dydt[A] = k * ((fA * s) - (y[A] * y[B]) - (kda * y[A]))
    dydt[B] = k * (-(y[A] * y[B]) + fs - (kd * y[B]))
    dydt[C] = k * ((y[A] * y[B]) - (kd * y[C]))   
    return dydt

#----------------------------------------------------------------------------------------------------
#MATERIAL MODEL PARAMETERS
#----------------------------------------------------------------------------------------------------

#material parameters
rho = 1100e12
m_C1 = 10.0 / 6.0
m_C2 = -m_C1 / 2.0 #chosen so that the initial stress is 0
Kappa_incomp = 10.0

C1 = Function(Vs_dg)
C2 = Function(Vs_dg)
Kappa = Function(Vs_cg)

C1.vector().set_local(np.zeros(n_elem) + m_C1)
C2.vector().set_local(np.zeros(n_elem) + m_C2)
temp_array_Kappa = Kappa.vector().get_local() 
temp_array_Kappa[:] = Kappa_incomp
Kappa.vector().set_local(temp_array_Kappa)

#----------------------------------------------------------------------------------------------------
#MECHANICAL MODEL
#----------------------------------------------------------------------------------------------------

# Variational trial and test functions
du_b = TrialFunction(Vv)
vb = TestFunction(Vv)
ub = Function(Vv) # the most recently computed solution

#identity
I = ufl.variable(ufl.Identity(3))

# Hyperelasticity
# Deformation gradients and Jacobians
gradu = ufl.variable(ufl.grad(ub))
F_t = ufl.variable(I + gradu)
J_t = ufl.variable(ufl.det(F_t))
F_g = ufl.variable(I + as_tensor([[0, 0, 0 ],[0, 0, 0],[0, 0, growth_rate]]))
F_gi = ufl.variable(ufl.inv(F_g))
J_g = ufl.variable(ufl.det(F_g))
F_e = ufl.variable(F_t * F_gi)
J_e = ufl.variable(J_t * ufl.inv(J_g))

# Strain and stretch tensor
C_e = ufl.variable(ufl.dot(F_e.T, F_e))
I1 = ufl.variable(ufl.tr(C_e))
I2 =  ufl.variable(0.5 * (ufl.tr(C_e)**2 - ufl.tr(C_e * C_e)))
E_e = ufl.variable(0.5 * (ufl.dot(F_e.T, F_e) - I))

# Mooney-Rivlin Coupled hyperelastic solid
Wc = (I * C1) + (I * C2 * I1) - (C2 * C_e)
p = (2.0 / J_e) * (C1 + 2.0 * C2) - (Kappa / J_e ) * ufl.ln(J_e)
Wj = -1.0 * J_e * p * ufl.inv(C_e)

# Stress measurements
second_PK_stress_E = 2.0 * Wc + Wj
cauchy_stress = (1.0 / J_e) * F_e * second_PK_stress_E * F_e.T

# Making weak form
dx = dx(metadata={'quadrature_degree': 5})
Functional_b_isotropic = ufl.inner(ufl.dot(F_e,second_PK_stress_E),ufl.grad(vb)) * dx()
J_b_isotropic = derivative(Functional_b_isotropic,ub,du_b)

if rank == 0:
    print ('Assembling solver...')

problem_isotropic = NonlinearVariationalProblem(Functional_b_isotropic, ub, bcs_b, J_b_isotropic)
solver = NonlinearVariationalSolver(problem_isotropic)

file_isotropic_ub = File(comm, output_dir + file_name + 'displacement_multi.pvd')

#----------------------------------------------------------------------------------------------------
#SOLVER PARAMETERS MODEL
#----------------------------------------------------------------------------------------------------

prm = solver.parameters
prm["newton_solver"]["maximum_iterations"] = 200
prm["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = False
prm['newton_solver']['absolute_tolerance'] = 1E-5
prm['newton_solver']['relative_tolerance'] = 1E-5
prm['newton_solver']['linear_solver'] = 'gmres'
prm['newton_solver']['preconditioner'] = 'hypre_amg'
prm['newton_solver']['krylov_solver']['maximum_iterations'] = 20000
prm['newton_solver']['krylov_solver']['absolute_tolerance'] = 1e-5
prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1e-5
prm['newton_solver']['relaxation_parameter'] = 0.7
prm['newton_solver']['krylov_solver']['error_on_nonconvergence'] = True
prm['newton_solver']["convergence_criterion"] = "residual"

#----------------------------------------------------------------------------------------------------
#TIME STEPPING PARAMETERS
#----------------------------------------------------------------------------------------------------
num_cycle = 200
count = 0
ctime = 0.0
simulation_length = 10

year = 365.0 * 24.0 * 3600.0
tspan = simulation_length * year

#output every 0.5 years
plot_step = num_cycle / (simulation_length * 2)

dt = tspan / num_cycle

def years(t): 
    return t / 3.154e7

# inverse of 10 years in seconds so that reaction completes at 10 yrs
k = 1.0 / year

#----------------------------------------------------------------------------------------------------
#STATE VARIABLE FUNCTION INITIALIZATION
#----------------------------------------------------------------------------------------------------

CauchyS = Function(Vt) # the most recently computed solution
pout = Function(Vs_cg)
ctrace = Function(Vs_cg)
growth_out = Function(Vs_cg)
Jeout = Function(Vs_cg)
Jgout = Function(Vs_cg)
Jtout = Function(Vs_cg)

# Chemical species
init = np.zeros((local_dim,3))
init[:,0] = 0.0
init[:,1] = 0.0
init[:,2] = 0.0
A_s = Function(Vs_cg)
B_s = Function(Vs_cg)
C_s = Function(Vs_cg)
A_s.vector().set_local(init[:,0])
B_s.vector().set_local(init[:,1])
C_s.vector().set_local(init[:,2])
A_sp = project(A_s, Vs_cg, solver_type="gmres", preconditioner_type="hypre_amg")
B_sp = project(B_s, Vs_cg, solver_type="gmres", preconditioner_type="hypre_amg")
C_sp = project(C_s, Vs_cg, solver_type="gmres", preconditioner_type="hypre_amg")

#----------------------------------------------------------------------------------------------------
#STATE VARIABLE UPDATE FUNCTIONS
#----------------------------------------------------------------------------------------------------

def UpdateStressVals():
    CauchyS.assign(project(cauchy_stress, Vt, solver_type="gmres", preconditioner_type="hypre_amg"))
    CauchyS.vector().apply('insert')
    pout.assign(project(p, Vs_cg, solver_type="gmres", preconditioner_type="hypre_amg"))
    pout.vector().apply('insert')
    ctrace.assign(project(tr(cauchy_stress), Vs_cg, solver_type="gmres", preconditioner_type="hypre_amg"))
    ctrace.vector().apply('insert')
    growth_out.assign(project(growth_rate, Vs_cg, solver_type="gmres", preconditioner_type="hypre_amg"))
    growth_out.vector().apply('insert')    
    Jtout.assign(project(J_t, Vs_cg, solver_type="gmres", preconditioner_type="hypre_amg"))
    Jtout.vector().apply('insert')
    Jeout.assign(project(J_e, Vs_cg, solver_type="gmres", preconditioner_type="hypre_amg"))
    Jeout.vector().apply('insert')
    Jgout.assign(project(J_g, Vs_cg, solver_type="tfqmr", preconditioner_type="petsc_amg"))
    Jgout.vector().apply('insert')


def UpdateSpeciesVals(new_vals):
    A_sp.vector().set_local(new_vals[:,0])
    A_sp.vector().apply('insert')
    B_sp.vector().set_local(new_vals[:,1])
    B_sp.vector().apply('insert')
    C_sp.vector().set_local(new_vals[:,2])
    C_sp.vector().apply('insert')
    growth_rate.assign(C_sp * growth_mult)
    growth_rate.vector().apply('insert')

#----------------------------------------------------------------------------------------------------
#I/O FUNCTIONS
#----------------------------------------------------------------------------------------------------

xdmf_var_fields = { "displacement":ub, "Cauchy Stress":CauchyS, "Pressure":pout,\
                    "ctrace":ctrace, "growth rate":growth_out, "J_e": Jeout, "J_t": Jtout, "J_g": Jgout,\
                    "[A]":A_sp, "[B]":B_sp, "[C]": C_sp\
}

def init_xdmffile(f_xdmf,fields):
    for name, field in fields.items():
        field.rename(name, name)
        f_xdmf.write(field, years(ctime))

def write_xdmffile(f_xdmf,fields):
    for name, field in fields.items():
        field.rename(name, name)
        f_xdmf.write(field, years(ctime))

xdmf_filename = output_dir + file_name + '.xdmf'
xdmffile_u = XDMFFile(comm, xdmf_filename)
xdmffile_u.parameters["flush_output"] = True
xdmffile_u.parameters["functions_share_mesh"] = True

#----------------------------------------------------------------------------------------------------
#MODEL INITIALIZATION
#----------------------------------------------------------------------------------------------------

UpdateStressVals()
comm.barrier()
init_xdmffile(xdmffile_u,xdmf_var_fields)
start_time = time.perf_counter()

#----------------------------------------------------------------------------------------------------
#MAIN LOOP
#----------------------------------------------------------------------------------------------------

if rank == 0:
    print ('Entering loop:')

while count < num_cycle:
    if rank == 0:
        print("\nstep = ",count + 1,", start time = ", years(ctime)," end time = ",years(ctime+dt),"\n")

    #solve mechanical problem
    solver.solve()
    
    #update mechnical state variables
    UpdateStressVals()

    #solve for chemical equilibrium
    for i in range (len(A_source_numpy)):
        r = ode(ODEfunc).set_integrator('vode', method='adams', 
                                        order=10, rtol=0, atol=1e-9, 
                                        with_jacobian=True)
        args_iv = [[init[i,0],init[i,1],init[i,2]],ctime]
        args_params = [A_source_numpy[i],k]
        r.set_initial_value(*args_iv).set_f_params(*args_params)
        results = np.empty([0,3])
        while r.successful() and r.t < ctime + dt:
            r.integrate(r.t + dt)
            results = np.append(results,[r.y],axis = 0)
        init[i,0] = results[-1,0]
        init[i,1] = results[-1,1]
        init[i,2] = results[-1,2]

    #update chemical species
    UpdateSpeciesVals(init)

    #increment time, output results every 0.5 years
    count += 1
    ctime += dt

    comm.barrier()
    if count > 0 and count % plot_step == 0:
        if rank == 0:
            print("\nWriting step ",count," at time = ",years(ctime),"\n")
        write_xdmffile(xdmffile_u,xdmf_var_fields)

if rank == 0:
    print('Done!')