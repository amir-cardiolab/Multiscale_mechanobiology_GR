from dolfin import *
import numpy as np
from numpy import linalg as LA
import numpy
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

class Species:
    def __init__(self,y,n,tau,EC50,i):
        self.y = y
        self.n = n
        self.tau = tau
        self.EC50 = EC50
        self.i = i

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
z1 = 3.0

outer_tube = Cylinder(Point(x0, y0, z0),Point(x0,y0,z1),0.65,0.65)
inner_tube = Cylinder(Point(x0, y0, z0),Point(x0,y0,z1),0.49,0.49)
half_cut = Box(Point(-1.0,-1.0,z0),Point(0.0,1.0,z1))
quarter_cut = Box(Point(-1.0,-1.0,z0),Point(1.0,0.0,z1))
geometry = outer_tube - inner_tube - half_cut - quarter_cut

domain = CSGCGALDomain3D(geometry)
generator = TetgenMeshGenerator3D()
generator.parameters["mesh_resolution"] = 200.0
bmesh = refine(generator.generate(domain),True)

File(comm, output_dir + file_name + "_cylinder.pvd") << bmesh

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
        return (on_boundary and near(x[0],0.0))

class y_axis(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and near(x[1],0.0))

class z_faces(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and near(x[2], 3)) or (on_boundary and near(x[2],0))

class intimal_wall(SubDomain):
    def inside(self, x, on_boundary):
        inner_wall = abs(pow(x[0],2) + pow(x[1],2) - 0.25) < 0.025
        return (inner_wall and on_boundary)

sub_domains = MeshFunction("size_t", bmesh, bmesh.topology().dim() - 1, value=0)
sub_domains.set_all(0)

#symmetry bcs
#fix the y axis in the x direction
y_ax_fix = y_axis()
y_ax_fix.mark(sub_domains, 1)

#fix the x axis in the y direction
x_ax_fix = x_axis()
x_ax_fix.mark(sub_domains, 2)

#fixing to surface
zfix = z_faces()
zfix.mark(sub_domains, 3)

#Intimal wall for the pressurization BC
intima_pressure_bc = intimal_wall()
intima_pressure_bc.mark(sub_domains, 4)

file_BC_default = File(comm, output_dir + file_name + "_subdomains.pvd")
file_BC_default << sub_domains

n_elem = bmesh.num_cells()

#----------------------------------------------------------------------------------------------------
#CHEMICAL SPECIES INITIALIZATION AND ODE SYSTEM DEFINITION
#----------------------------------------------------------------------------------------------------

node_numbers = bmesh.num_vertices()
element_numbers = bmesh.num_cells()
connectivity_matrix = bmesh.cells()

local_range = Vs_cg.dofmap().ownership_range()
local_dim = local_range[1] - local_range[0]
vert = vertices(bmesh)

coordinate = Vs_cg.tabulate_dof_coordinates()
TGFB_source_np = np.zeros((local_dim))
growth_mult = 2.5
growth_rate = Function(Vs_cg)
growth_rate.vector().set_local(TGFB_source_np * growth_mult)
growth_rate.vector().apply('insert')

n 		= 1.2
tau 	= 1.0
Ymax 	= 1.0
w 		= 1.0
EC50	= 0.5
s0      = 0.05
a_sig = 1.00
b_sig = 0.2

def Stressfunc(ts):
    s = 1.0 + a_sig / (1.0 + exp(-(ts - b_sig)/s0))
    return s

def ODEfunc(t,y,fA,k):
    tr = ctrace.vector()[i]
    s = Stressfunc(tr)
    TGFB = Species(fA * s,n,tau,EC50,-1)
    MAPK = Species(y[0],n,tau,EC50,0)
    Smad = Species(y[1],n,tau,EC50,1)
    MMP = Species(y[2],n,tau,EC50,2)
    TIMP = Species(y[3],n,tau,EC50,3)

    dydt 		= np.zeros(4) 

    dydt[MAPK.i] 	= (k * ((w * activation(TGFB,MAPK) * Ymax) - MAPK.y) / MAPK.tau)

    dydt[Smad.i] 	= (k * ((w * activation(TGFB,Smad) * Ymax) - Smad.y) / Smad.tau)

    dydt[MMP.i] 	= (k * ((w * inhibition(TIMP,MMP)  * activation(MAPK,MMP) 
                                * Ymax) 
                             - MMP.y)
                         / MMP.tau
    )

    dydt[TIMP.i] 	= (k * ((w * activation(Smad,TIMP) * Ymax) - TIMP.y) / TIMP.tau)

    return dydt

#----------------------------------------------------------------------------------------------------
#MATERIAL MODEL PARAMETERS
#----------------------------------------------------------------------------------------------------

def activation(activator,activated): 
    # hill activation function with parameters w (weight), n (Hill coeff), EC50

    En = pow(activator.EC50, activator.n)
    Beta = (En - 1.0) / (2.0 * En - 1.0)
    Km = pow((Beta - 1.0), (1.0 / activator.n))
    cn = pow(activator.y, activator.n)
    Kn = pow(Km, activator.n)
    f_act = (Beta * cn) / (Kn + cn)

    return f_act
 
def inhibition(inhibitor,inhibited): 
    # inverse hill function with parameters w (weight), n (Hill coeff), EC50 
    f_inhib = 1 - activation(inhibitor,inhibited) 
    return f_inhib

######### Material parameters   ################
rho = 1100e12
m_C1 = 10.0 / 6.0
m_C2 = -m_C1 / 2.0 #chosen so that the initial stress is 0
Kappa_incomp = 16.111

C1 = Function(Vs_dg)
C2 = Function(Vs_dg)
Kappa = Function(Vs_cg)

C1.vector().set_local(np.zeros(n_elem) + m_C1)
C2.vector().set_local(np.zeros(n_elem) + m_C2)
temp_array_Kappa = Kappa.vector().get_local() 
temp_array_Kappa[:] = Kappa_incomp
Kappa.vector().set_local(temp_array_Kappa)


#fix motion in the x direction on nodes where y = 0
bcx = DirichletBC(Vv.sub(1), Constant(0.0), y_ax_fix)
#fix motion in the y direction on nodes where x = 0
bcy = DirichletBC(Vv.sub(0), Constant(0.0), x_ax_fix)
#fix motion in the z direction on nodes where z = 0
bcz = DirichletBC(Vv.sub(2), Constant(0.0), zfix)

bcs_b = [bcx, bcy, bcz]

# Trial and test functions
du_b = TrialFunction(Vv)   # displacement
vb = TestFunction(Vv)
ub = Function(Vv) # the most recently computed solution
vel0, a0 , ub0 = Function(Vv), Function(Vv), Function(Vv)
n_function = FacetNormal(bmesh)
h = CellDiameter(bmesh)

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
F_g = ufl.variable(I + as_tensor([[growth_rate, 0, 0 ],[0, growth_rate, 0],[0, 0, 0]]))
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

P_wall = Function(Vs_dg)

dx = dx(metadata={'quadrature_degree': 5})
ds = Measure('ds', domain = bmesh, subdomain_data = sub_domains)

Functional_b_isotropic = (inner(dot(F_e,second_PK_stress_E),grad(vb)) * dx() + P_wall * J_e * inner(inv(F_e.T) * n_function,vb) * ds(4))

J_b_isotropic = derivative(Functional_b_isotropic, ub, du_b)

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

'''Neuman Boundary Condition '''
Time_array = numpy.array([0,10])
P_aort_array = numpy.array([0,0.016])

#----------------------------------------------------------------------------------------------------
#TIME STEPPING PARAMETERS
#----------------------------------------------------------------------------------------------------
num_cycle = 160
count = 0
ctime = 0.0
simulation_length = 8

year = 365.0 * 24.0 * 3600.0
tspan = simulation_length * year

#output every 0.5 years
plot_step = num_cycle / (simulation_length * 2)

dt = tspan / num_cycle

init = np.zeros((local_dim,4))
k = 1.0 / year

def years(t): 
    return t / year

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
TGFB_sp = Function(Vs_cg)
TGFB_sp.vector().set_local(TGFB_source_np)
TGFB_sp.vector().apply('insert')
MAPK_sp = Function(Vs_cg)
Smad_sp = Function(Vs_cg)
MMP_sp  = Function(Vs_cg)
TIMP_sp = Function(Vs_cg)

# Intimal wall pressurization
def init_pressurization():
    count_init = 0
    while count_init < 10:
            count_init += 1
            temp_array_P_wall = P_wall.vector().get_local()
            P_current_wall = np.interp(count_init,Time_array,P_aort_array)
            temp_array_P_wall[:] = P_current_wall
            P_wall.vector().set_local(temp_array_P_wall)
            P_wall.vector().apply('insert')
            if rank ==0 :
                print('Pressurizing to ', P_current_wall, '. initialization step ',count_init)
            solver.solve()
            UpdateStressVals()

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
    TGFB_np = np.zeros((local_dim))
    for i in range (local_dim):
        tr = ctrace.vector()[i]
        s = Stressfunc(tr)
        TGFB_np[i] = TGFB_source_np[i] * s
    TGFB_sp.vector().set_local(TGFB_np)
    TGFB_sp.vector().apply('insert')
    MAPK_sp.vector().set_local(new_vals[:,0])
    MAPK_sp.vector().apply('insert')
    Smad_sp.vector().set_local(new_vals[:,1])
    Smad_sp.vector().apply('insert')
    MMP_sp.vector().set_local(new_vals[:,2])
    MMP_sp.vector().apply('insert')
    TIMP_sp.vector().set_local(new_vals[:,3])
    TIMP_sp.vector().apply('insert')
    growth_rate.assign(MMP_sp * growth_mult)
    growth_rate.vector().apply('insert')

#I/O FUNCTIONS
#----------------------------------------------------------------------------------------------------

xdmf_var_fields = { "displacement":ub, "Cauchy Stress":CauchyS, "Pressure":pout,\
                    "ctrace":ctrace, "growth rate":growth_out, "J_e": Jeout, "J_t": Jtout, "J_g": Jgout,\
                    "[TGFB]":TGFB_sp, "[MAPK]":MAPK_sp, "[Smad]": Smad_sp, "[MMP]": MMP_sp, "[TIMP]": TIMP_sp\
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

    #solve for chemical equilibrium
    for i in range (len(TGFB_source_np)):
        r = ode(ODEfunc).set_integrator('vode', method='adams', 
                                        order=15, rtol=1e-12, atol=1e-12,
                                        with_jacobian=True)
        args_iv = [[init[i,0],init[i,1],init[i,2],init[i,3]],ctime]
        args_params = [TGFB_source_np[i],k]
        r.set_initial_value(*args_iv).set_f_params(*args_params)
        results = np.empty([0,4])
        while r.successful() and r.t < ctime + dt:
            r.integrate(r.t + dt)
            results = np.append(results,[r.y],axis = 0)
        init[i,0]   = results[-1,0]
        init[i,1]   = results[-1,1]
        init[i,2]   = results[-1,2]
        init[i,3]   = results[-1,3]

    #update chemical species
    UpdateSpeciesVals(init)

    #solve mechanical problem
    solver.solve()
    
    #update mechnical state variables
    UpdateStressVals()

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