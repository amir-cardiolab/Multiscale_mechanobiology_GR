from dolfin import *
from dolfin.la import solver
import numpy as np
from numpy import linalg as LA
import numpy
import time
import sys
import argparse
import glob
import re
import math
import matplotlib.pyplot as plt 
from mshr import *
from mpi4py import MPI
import ufl
import os

#MPI definitions
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

n_Hill	= 1.2
tau_Hill 	= 1.0
Ymax_Hill 	= 1.0
w_Hill 		= 1.0
EC50_Hill	= 0.5
En_Hill = pow(EC50_Hill,n_Hill)
Beta_Hill = (En_Hill - 1.0) / (2.0 * En_Hill - 1.0)
Km_Hill = pow((Beta_Hill - 1.0), (1.0 / n_Hill))
Kn_Hill = pow(Km_Hill,n_Hill)

class Species:
    def __init__(self):
        self.c = Function(Vs_cg)
        
    def update(self,u_i):
        self.c.assign(project(u_i,Vs_cg,solver_type="gmres"))
        
    def cn(self):
        return Expression('max(pow(c,n),0.0)',c=self.c,n=Constant(n_Hill), degree=2)

    def act(self,activator):
        return Expression('(Beta * cn) / (Kn + cn)',Beta=Constant(Beta_Hill), cn=activator.cn(), Kn=Constant(Kn_Hill), degree=2)

    def inh(self,inhibitor):
        return Expression('max(1.0 - fa,0.0)',fa=self.act(inhibitor), degree=2)

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

#----------------------------------------------------------------------------------------------------
#FLAGS
#----------------------------------------------------------------------------------------------------

bool_reference_growth_dir = True

#----------------------------------------------------------------------------------------------------
#MESH INITIALIZATION
#----------------------------------------------------------------------------------------------------

if rank == 0:
    print ('reading mesh')

bmesh = Mesh('./Tube_Mesh.xml')

sub_domain_ref = MeshFunction("bool", bmesh, bmesh.topology().dim() - 1)
sub_domain_ref.set_all(0)

#----------------------------------------------------------------------------------------------------
#FUNCTION SPACE INITIALIZATION AND GLOBAL/LOCAL NODE INTERFACE FOR PARALLEL RUNS
#----------------------------------------------------------------------------------------------------

# displacement function spaces
Vs_cg = FunctionSpace(bmesh,'CG',2)
Vs_dg = FunctionSpace(bmesh,'DG',0)
Vv = VectorFunctionSpace(bmesh,'Lagrange',2)
Vt = TensorFunctionSpace(bmesh,'Lagrange',2)
# chemical element and function spaces
P1 = FiniteElement('P', bmesh.ufl_cell(), 1)
chem_element = MixedElement([P1, P1, P1, P1])
Vc = FunctionSpace(bmesh, chem_element)

# global node numbers (better for 1 processor)
global_node_num= bmesh.num_vertices()

#local node number (better for more than 1 processor)
local_range = Vs_cg.dofmap().ownership_range()
local_dim = local_range[1] - local_range[0]

# element numbers
n_elem = bmesh.num_cells()
# coordinates of each node. 
coord = Vs_cg.tabulate_dof_coordinates()

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
        return ((on_boundary and near(x[2], 3.0,0.01)) or (on_boundary and near(x[2], 0.0,0.01)))

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

# Symmetry BC: Fix y axis in the x direction
bcx = DirichletBC(Vv.sub(1), Constant(0.0), y_ax_fix)
# Symmetry BC: Fix x axis in the y direction
bcy = DirichletBC(Vv.sub(0), Constant(0.0), x_ax_fix)
# Symmetry BC: Fix motion in the z direction on nodes where z = 0
bcz = DirichletBC(Vv.sub(2), Constant(0.0), zfix)

bcs_b = [bcx,bcy,bcz]

#----------------------------------------------------------------------------------------------------
#TIME STEPPING PARAMETERS
#----------------------------------------------------------------------------------------------------
num_cycle = 240
count = 0
ctime = 0.0
simulation_length = 8.0

year = 365.0 * 24.0 * 3600.0
tspan = simulation_length * year

#output every 0.5 years
plot_step = num_cycle / (simulation_length * 2)

dt = tspan / num_cycle

def years(t): 
    return t / 3.154e7

# convert y^-1 to s^-1
k = Constant(1.0 / year)
k_r = Function(Vs_cg)

#----------------------------------------------------------------------------------------------------
#CHEMICAL SPECIES INITIALIZATION AND ODE SYSTEM DEFINITION
#----------------------------------------------------------------------------------------------------

#initialize vector for growth rate and previous growth rate.
growth_rate = Function(Vs_cg)

#----------------------------------------------------------------------------------------------------
#MATERIAL MODEL PARAMETERS
#----------------------------------------------------------------------------------------------------

# neo-Hookean material parameters
C1 = Function(Vs_dg)
C2 = Function(Vs_dg)
Kappa = Function(Vs_dg)

# Define trial/test functions
u_s = Function(Vc)
u_n = Function(Vc)
u_1, u_2, u_3, u_4 = split(u_s)
u_n1, u_n2, u_n3, u_n4 = split(u_n)
v_1, v_2 , v_3, v_4 = TestFunctions(Vc)
# Define expressions used in variational forms
ht = Constant(dt)
kd = Constant(1.0)
D = Constant(1e-20)

k_bp = Function(Vs_cg)
phi_me = Function(Vs_cg)

f0 = Function(Vv)
f_TGFB = Expression('0.5*exp(-abs(pow(x[2],3.0))/3.0)',degree=3)
TGFB = Species()
MAPK = Species()
Smad = Species()
MMP = Species()
TIMP = Species()

# initial BCs
if count == 0:
    C1.assign(Constant(1.6667))
    C2.assign(Constant(0.0))
    k_r.assign(Constant(1.0/year))
    Kappa.assign(Constant(1.0))
    u_n.assign(project(Expression(('0.0','0.0','0.0','0.0'),degree=1), Vc, solver_type="gmres"))
    growth_rate.assign(project(Expression('1.0',degree=1), Vs_cg, solver_type="gmres"))
    TGFB.update(f_TGFB)
    MAPK.update(u_n1)
    Smad.update(u_n2)
    MMP.update(u_n3)
    TIMP.update(u_n4)
    k_bp.assign(project(Expression('1.0',degree=1), Vs_cg, solver_type="gmres"))
    phi_me.assign(project(Expression('0.0',degree=1), Vs_cg, solver_type="gmres"))
    f0.assign(project(Expression(('0.0','0.0','1.0'),degree=1), Vv, solver_type="gmres"))

#----------------------------------------------------------------------------------------------------
#MECHANICAL MODEL
#----------------------------------------------------------------------------------------------------

# Variational trial and test functions
du_b = TrialFunction(Vv)
vb = TestFunction(Vv)
ub = Function(Vv) # the most recently computed solution
n_function = FacetNormal(bmesh)
h = CellDiameter(bmesh)
dx = ufl.Measure('dx', domain=bmesh, metadata={'quadrature_degree': 5})
ds = Measure('ds', domain = bmesh, subdomain_data = sub_domains)

#kinematics
d = len(ub)
#identity
I = ufl.variable(ufl.Identity(3))

# Hyperelasticity
# Deformation gradients and Jacobians
gradu = ufl.variable(ufl.grad(ub))
F = ufl.variable(I + gradu)
J = ufl.variable(ufl.det(F))
E = ufl.variable(0.5 * (ufl.dot(F.T, F) - I))

if bool_reference_growth_dir:
    theta_normal = f0
    theta_normal = theta_normal / sqrt(ufl.dot(theta_normal,theta_normal))
else:
    theta_normal = F * f0
    theta_normal = theta_normal / sqrt(ufl.dot(theta_normal,theta_normal))

nxn = ufl.variable(ufl.outer(theta_normal,theta_normal))
F_g = ufl.variable((sqrt(growth_rate) * I) + (1.0 - sqrt(growth_rate)) * nxn)
F_gi = ufl.variable(ufl.inv(F_g))
J_g = ufl.variable(ufl.det(F_g))
F_e = ufl.variable(F * F_gi)
J_e = ufl.variable(J * ufl.inv(J_g))
F_i = ufl.variable(ufl.inv(F))

# Strain and stretch tensor
C_e = ufl.variable(ufl.dot(F_e.T, F_e))
I1 = ufl.variable(ufl.tr(C_e))
I2 =  ufl.variable(0.5 * (ufl.tr(C_e)**2 - ufl.tr(C_e * C_e)))

# Mooney-Rivlin Coupled hyperelastic solid
Wc = (I * C1) + (I * C2 * I1) - (C2 * C_e)
p = (2.0 / J_e) * (C1 + 2.0 * C2) - (Kappa / J_e ) * ufl.ln(J_e)
Wj = -1.0 * J_e * p * ufl.inv(C_e)

# Stress measurements
second_PK_stress_E = 2.0 * Wc + Wj
second_PK_stress = J_g * F_gi * second_PK_stress_E * F_gi.T
first_PK_stress = F * second_PK_stress
cauchy_stress = (1.0 / J) * first_PK_stress * F.T
cauchy_stress_I1 = ufl.variable(ufl.tr(cauchy_stress))
cauchy_stress_I2 = ufl.variable(0.5 * (ufl.tr(cauchy_stress)**2-ufl.tr(cauchy_stress * cauchy_stress)))
cauchy_stress_I3 = ufl.variable(det(cauchy_stress))

P_wall = Function(Vs_dg)
# Making weak form
dx = dx(metadata={'quadrature_degree': 5})
Functional_b_isotropic = ufl.inner(first_PK_stress,ufl.grad(vb)) * dx() + P_wall * J * inner(inv(F.T) * n_function,vb) * ds(4)
J_b_isotropic = derivative(Functional_b_isotropic,ub,du_b)

if rank == 0:
    print ('Assembling solver...')

problem_isotropic = NonlinearVariationalProblem(Functional_b_isotropic, ub, bcs_b, J_b_isotropic)
solver = NonlinearVariationalSolver(problem_isotropic)

# LHS - RHS
# partial derivatives
# sources
# sinks
# reactions

Fc   = (((u_1 - u_n1) / ht) * v_1) * dx + (D * dot(grad(u_1),grad(v_1))) * dx - (k_r * MAPK.act(TGFB) * v_1) * dx + (k_r * kd * u_1 * v_1) * dx\
    + (((u_2 - u_n2) / ht) * v_2) * dx + (D * dot(grad(u_2),grad(v_2))) * dx - (k_r * Smad.act(TGFB) * v_2) * dx + (k_r * kd * u_2 * v_2) * dx\
    + (((u_3 - u_n3) / ht) * v_3) * dx + (D * dot(grad(u_3),grad(v_3))) * dx - (k_r * MMP.inh(TIMP) * MMP.act(MAPK) * v_3) * dx + (k_r * kd * u_3 * v_3) * dx\
    + (((u_4 - u_n4) / ht) * v_4) * dx + (D * dot(grad(u_4),grad(v_4))) * dx - (k_r * TIMP.act(Smad) * v_4) * dx + (k_r * kd * u_4 * v_4) * dx

#----------------------------------------------------------------------------------------------------
#SOLVER PARAMETERS MODEL
#----------------------------------------------------------------------------------------------------

prm = solver.parameters
prm["newton_solver"]["maximum_iterations"] = 200
prm["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = False
prm['newton_solver']['absolute_tolerance'] = 1E-5
prm['newton_solver']['relative_tolerance'] = 1E-5
prm['newton_solver']['linear_solver'] = 'gmres'
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
#STATE VARIABLE FUNCTION INITIALIZATION
#----------------------------------------------------------------------------------------------------

# output functions for visualization.
CauchyS = Function(Vt)
LagrangeS = Function(Vt)
pressure = Function(Vs_cg)
Cauchy1stPRout = Function(Vv)
CauchyI1 = Function(Vs_cg)
CauchyI2 = Function(Vs_cg)
CauchyI3 = Function(Vs_cg)
Jout = Function(Vs_cg)
Jeout = Function(Vs_cg)
Jgout = Function(Vs_cg)
growth_out = Function(Vs_cg)
k_out = Function(Vs_cg)
phi_out = Function(Vs_cg)
f0_out = Function(Vv)
growth_tens_out = Function(Vs_cg)
normal_out = Function(Vs_cg)


#----------------------------------------------------------------------------------------------------
#STATE VARIABLE UPDATE FUNCTIONS
#----------------------------------------------------------------------------------------------------

def UpdateStressVals():
    CauchyS.assign(project(cauchy_stress, Vt, solver_type="gmres"))
    CauchyS.vector().apply('insert')
    LagrangeS.assign(project(E, Vt, solver_type="gmres"))
    LagrangeS.vector().apply('insert')
    pressure.assign(project(-(1.0 / 3.0) * tr(cauchy_stress), Vs_cg, solver_type="gmres"))
    pressure.vector().apply('insert')
    CauchyI1.assign(project(cauchy_stress_I1, Vs_cg, solver_type="gmres"))
    CauchyI1.vector().apply('insert')
    CauchyI2.assign(project(cauchy_stress_I2, Vs_cg, solver_type="gmres"))
    CauchyI2.vector().apply('insert')
    CauchyI3.assign(project(cauchy_stress_I3, Vs_cg, solver_type="gmres"))
    CauchyI3.vector().apply('insert')
    growth_out.assign(project(growth_rate, Vs_cg, solver_type="gmres"))
    growth_out.vector().apply('insert')    
    Jout.assign(project(J, Vs_cg, solver_type="gmres"))
    Jout.vector().apply('insert')
    Jeout.assign(project(J_e, Vs_cg, solver_type="gmres"))
    Jeout.vector().apply('insert')
    Jgout.assign(project(J_g, Vs_cg, solver_type="gmres"))
    Jgout.vector().apply('insert')
    k_out.assign(project(k_bp, Vs_cg, solver_type="gmres"))
    phi_out.assign(project(phi_me, Vs_cg, solver_type="gmres"))
    f0_out.assign(project(f0, Vv, solver_type="gmres"))
    f0_out.vector().apply('insert')
    normal_out.assign(project(theta_normal, Vv, solver_type="gmres"))
    normal_out.vector().apply('insert')
    growth_tens_out.assign(project(F_g, Vt, solver_type="gmres"))
    growth_tens_out.vector().apply('insert')

def UpdateSpeciesVals():
    _u_1, _u_2, _u_3, _u_4 = split(u_s)
    _u_n1, _u_n2, _u_n3, _u_n4 = split(u_n)
    MAPK.update(_u_1)
    Smad.update(_u_2)
    MMP.update(_u_3)
    TIMP.update(_u_4)
    sigmoid_act = Expression('1.0/(1.0+exp(-(theta-t_a)/gamma))', theta=Function(Vs_cg), t_a=Constant(-3.0), gamma=Constant(0.3), degree=1)
    sigmoid_inh = Expression('1.0/(1.0+exp(-(theta+t_a)/gamma))', theta=Function(Vs_cg), t_a=Constant(-3.0), gamma=Constant(0.3), degree=1)
    s_act = Function(Vs_cg)
    s_inh = Function(Vs_cg)
    sigmoid_act.theta = growth_rate
    sigmoid_inh.theta = growth_rate
    s_act.interpolate(sigmoid_act)
    s_inh.interpolate(sigmoid_inh)
    K_bandpass = Expression('theta_min + theta_max * (s_act - s_inh)', theta_min=Constant(0.0), theta_max=Constant(1.0), s_act=Function(Vs_cg), s_inh=Function(Vs_cg), degree=1)
    K_bandpass.s_act = s_act
    K_bandpass.s_inh = s_inh
    k_bp.interpolate(K_bandpass)
    dCdt = Expression('phi_a*(C-Cn)/dt', phi_a=Constant(6.5), C=Function(Vs_cg), Cn=Function(Vs_cg), dt=Constant(dt), degree=1)
    dCdt.C = project(_u_3,Vs_cg, solver_type="gmres")
    dCdt.Cn = project(_u_n3,Vs_cg, solver_type="gmres")
    phi_me.interpolate(dCdt)
    growth_rate.assign(project((growth_rate + (k_bp * phi_me * dt)),Vs_cg, solver_type="gmres"))
    u_n.assign(u_s)
    
#----------------------------------------------------------------------------------------------------
def init_pressurization():
    prm['newton_solver']['absolute_tolerance'] = 1E-9
    prm['newton_solver']['relative_tolerance'] = 1E-9
    prm['newton_solver']['krylov_solver']['absolute_tolerance'] = 1e-9
    prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1e-9
    count_init = 0
    count_end = 10
    P_init = 0.016
    while count_init < 10:
            count_init += 1
            P_current_wall = P_init * count_init / count_end
            if rank ==0 :
                print("Initialization step ",count_init, "pressurizing to ",P_current_wall,".\n")
            P_wall.assign(Constant(P_current_wall))
            solver.solve()
            UpdateStressVals()
    prm['newton_solver']['absolute_tolerance'] = 1E-5
    prm['newton_solver']['relative_tolerance'] = 1E-5
    prm['newton_solver']['krylov_solver']['absolute_tolerance'] = 1e-5
    prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1e-5

#----------------------------------------------------------------------------------------------------
#I/O FUNCTIONS
#----------------------------------------------------------------------------------------------------
xdmf_var_fields = {"displacement":ub, "Lagrange Strain":LagrangeS, "Cauchy Stress":CauchyS, "Pressure":pressure,\
                    "stress I1": CauchyI1, "stress I2": CauchyI2, "stress I3":CauchyI3,\
                    "growth rate":growth_out, "J_e": Jeout, "J_t": Jout, "J_g": Jgout,\
                    "[TGFB]":TGFB.c, "[MAPK]":MAPK.c, "[Smad]": Smad.c, "[MMP]": MMP.c, "[TIMP]": TIMP.c,\
                    "k_bp":k_out, "phi":phi_out,\
                    "f0":f0_out, "growth_tens":growth_tens_out, "normal vector":normal_out\
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

init_pressurization()
UpdateStressVals()
init_xdmffile(xdmffile_u,xdmf_var_fields)
start_time = time.perf_counter()

#----------------------------------------------------------------------------------------------------
#MAIN LOOP
#----------------------------------------------------------------------------------------------------

if rank == 0:
    start = time.time()
    print("Started solution")
    print ('Entering loop:',flush=True)

while count < num_cycle:
    if rank == 0:
        print("\nstep = ",count + 1,", start time = ", years(ctime)," end time = ",years(ctime+dt),"\n",flush=True)

    solve(Fc == 0, u_s, solver_parameters={"newton_solver":{"relative_tolerance":1e-20, "absolute_tolerance": 1e-20}})

    #update chemical species
    if rank == 0:
        print("\nUpdating chemical species\n")
    UpdateSpeciesVals()

    #solve mechanical problem
    solver.solve()

    #update mechnical state variables
    if rank == 0:
        print("\nUpdating mechanical dof\n")
    UpdateStressVals()
    
    #increment time, output results every 0.5 years
    count += 1
    ctime += dt

    if count > 0 and count % plot_step == 0:
        if rank == 0:
            print("\nWriting step ",count," at time = ",years(ctime),"\n",flush=True)
        write_xdmffile(xdmffile_u,xdmf_var_fields)

if rank == 0:
    print('Done!')
    print("Finished solution in %f seconds"%(time.time()-start))