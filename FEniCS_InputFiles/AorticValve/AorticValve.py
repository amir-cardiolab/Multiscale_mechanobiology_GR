
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
# import matplotlib.pyplot as plt 

from scipy.integrate import ode 
from scipy.integrate import odeint


import Pres_final_edition as Pressure_BC
# import Pres as Pres
import Pressure_trans_shift as P_trans_shift

parser = argparse.ArgumentParser()
parser.add_argument("-pd", "--polynomial_degree",
                    help="polynomial degree of the ansatz functions",
                    default=1,
                    type=int)
args = parser.parse_args()


set_log_level(30)
# set_log_level(LogLevel.TRACE)
# set_log_level(16)
#---------------------------------
# Optimization options for the form compiler
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['representation'] = "uflacs"
parameters['form_compiler']['quadrature_degree'] = 4
parameters['form_compiler']['optimize'] = True
ffc_options = {'optimize' : True,
               'eliminate_zeros' : True,
               'precompute_basis_const' : True,
               'precompute_ip_const' : True}



Flag_mpi = True
Flag_quasi=False
Flag_fenics2017=False
Flag_own_computer=False
Flag_generalized_alpha = True 
Humphrey_flag = True

###########################################################


'''Fenics 2017 '''
if Flag_fenics2017:
	rank = MPI.rank(mpi_comm_world())
###Fenics 2018'''
else:
	rank = MPI.rank(MPI.comm_world)


###########################################################

''' Loading Geometry (contains direction, material stiffness, and boundaries) '''

if Flag_own_computer:
### own _computer
	root_dir='/Users/reza/research/growth_model/thick_wall/geo/'
	output_dir = '/Users/reza/research/growth_model/thick_wall/geo/run_test2/'
else:
### Monsoon Cluster
	root_dir = '/scratch/ms3459/Data/35years_old_geomtry/thick_geo/' 
	output_dir = '/scratch/ms3459/valve_ode/results_9/'


mesh_filename=root_dir+'whole_model.h5'
bmesh = Mesh()
if (rank == 0):
    print ('Loading mesh:', mesh_filename)
hdf = HDF5File(bmesh.mpi_comm(), mesh_filename, "r")
hdf.read(bmesh, "mesh",False)
V0=FunctionSpace(bmesh,'CG',1)
V00=FunctionSpace(bmesh,'DG',0)
e_circum=Function(VectorFunctionSpace(bmesh,'DG',0))
e_normal = Function(VectorFunctionSpace(bmesh,'CG',1))
e_normal_loc = Function(VectorFunctionSpace(bmesh,'CG',1))
Tag_leaflet_element = Function(FunctionSpace(bmesh, 'DG', 0))
facet_domains = MeshFunction('size_t', bmesh,2, bmesh.domains())
hdf.read(e_normal, "e_normal")
hdf.read(e_normal_loc, "e_normal_loc_2la")
hdf.read(facet_domains,'facet_domains')
hdf.read(e_circum,'e_circum')
hdf.read(Tag_leaflet_element,'Tag_leaflet')
calcif = Function(FunctionSpace(bmesh, 'DG', 0))
hdf.read(calcif,'calcif')
if Flag_own_computer:
    file_calcif = File(output_dir + 'calcif.pvd')
    file_calcif << calcif
ds = Measure('ds', domain = bmesh, subdomain_data = facet_domains)

C10Y = Function(V00)
C0=Function(V00)
C0_calc=Function(V00)
local_range = V0.dofmap().ownership_range()
local_dim = local_range[1] - local_range[0]
normal_numpy_flatten=e_normal.vector().get_local()
normal_numpy=np.reshape(normal_numpy_flatten,(local_dim,3))
###########################################################
''' cell wise leaflet tags '''
element_numbers = bmesh.num_cells()
# Leaflet_tag_element = root_dir + 'Leaflet_elements_tag.xml'
# Tags_load_element = Function(V00,Leaflet_tag_element)
Tags_element =  np.zeros((element_numbers, 1),dtype="intc")
for gol in range(element_numbers):
  Tags_element[gol] = Tag_leaflet_element.vector()[gol]


fib_tag_file = root_dir + 'Leaflet_fibrosa_tag.xml'


# element_numbers = bmesh.num_cells()
# V0 = FunctionSpace(bmesh, 'DG', 0)



Tags_fibrosa =  np.zeros((element_numbers, 1),dtype="intc")
Tags_load_f = Function(V00,fib_tag_file)
Tags_fibrosa = Tags_load_f.vector().get_local()


# print (Tags_fibrosa)


count_f = 0

for kk in range(element_numbers):
   if (Tags_fibrosa[kk] == 1 ):
     count_f  = count_f  + 1


# print(count_f)


Fibrosa_IDs = np.zeros((count_f, 1),dtype="intc") #only stores the IDs of fibrosa elements
cc = 0
for kk in range(element_numbers):
  if (Tags_fibrosa[kk] ==1 ):
     Fibrosa_IDs[cc] = kk
     cc = cc + 1




if Flag_quasi:

	def update(ub,ub0):
	  ub_vec, ub0_vec = ub.vector() , ub0.vector()
	  ub0.vector()[:] = ub.vector()
else:
	def update(ub,ub0,vel0,a0,beta2,gamma,dt):
	  ub_vec, ub0_vec = ub.vector(), ub0.vector()
	  vel0_vec, a0_vec = vel0.vector(), a0.vector()
	    #update acceleration and vel
	  a_vec = (1.0/(2.0*beta2))*(  (ub_vec -ub0_vec - vel0_vec*dt) / (0.5*dt*dt) - (1.0 -2.0*beta2)*a0_vec)
	  v_vec = dt*((1.0-gamma) * a0_vec + gamma*a_vec ) + vel0_vec
	  #update
	  vel0.vector()[:] , a0.vector()[:] = v_vec , a_vec
	  ub0.vector()[:] = ub.vector()

###########################################################

'''Some Function which is used in our solution'''

def Dev(mytensor, C_e):
    return mytensor - 1.0/3.0*(inner(mytensor, C_e))*inv(C_e)
mesh_edge_size = 5e-4
TOL = mesh_edge_size/3.0
class Pinpoint(SubDomain):
    
    def __init__(self, coords):
        self.coords = np.array(coords)
        SubDomain.__init__(self)
    def move(self, coords):
        self.coords[:] = np.array(coords)
    def inside(self, x, on_boundary):
        return np.linalg.norm(x-self.coords) < TOL

###########################################################

'''Time values:'''

beta2 = 0.25 #Newmark
gamma = 0.5 #Newmark
dt = Constant(5e-5) #2e-4
dt_fine = 2.5e-4   #5e-5 fails at the start of the steep acceleration starting at the end of 1st cycle
dt_finer = 1e-6
dt_coarse = 2e-3
dt_coarser=5e-3
dt_med = 7.5e-4
T = .6 #0.84   #integration time
t = 0.0
#Generalized alpha method
rho_inf = 0.0 # 0.5 #0 annihilates high freq in one step. 1: high freq preserved

alpha_m =  (2.0* rho_inf - 1.0 ) / (1.0 + rho_inf)
alpha_f = rho_inf / ( 1.0 + rho_inf)
if (Flag_generalized_alpha):
  gamma = 0.5 - alpha_m + alpha_f
  beta2 = 0.25*( ( 1.0 - alpha_m + alpha_f )**2 )
  if rank == 0:
    print ('Using Generalized-alpha!!!!! ')


###########################################################

'''Material parameters'''


rho = Constant(1100.0)  #material density
beta = 1e7 #1e7  #penalty for Nitsche
Kappa_incomp = 1e6 #5e8 #5e6
C10Y_leaflet = 0.476e3 #The exp model in Votta etal16
C0_leaflet = 5.93e3
C10Y_aorta = 2e6/ 6.  
C0_aorta = 0.0 #Aorta is neo-Hookean 
C1 = 1.866
C2 = 4.451
delta = 0.558
C0_aorta = 0.0 #Aorta is neo-Hookean
###Humphrey-alpha model (Auricchio, Ferrara, Morganti Ann. Solid Struct Mech 2012)
Humphrey_flag = True
C1_H = 5.81  #Auricchio et al 2012 paper
C2_H = 24.97  #Auricchio et al 2012 paper
C0_H_s = 0.062e6 #Auricchio et al 2012 paper
C_H_s = 0.022e6
damp=Constant(1e6)

node_numbers = bmesh.num_vertices()
element_numbers = bmesh.num_cells()
connectivity_matrix = bmesh.cells()
growth_local_rate_numpy=np.zeros((node_numbers))
# C10Y_numpy = np.zeros(local_dim)
# C0_numpy = np.zeros(local_dim)

if (Humphrey_flag == False):
 C10Y_values = [C10Y_leaflet, C10Y_aorta ]
else:
 C10Y_values = [0 , C10Y_aorta ]

if (Humphrey_flag == False):
  C0_values = [C0_leaflet, C0_aorta ]
else:
  C0_values = [1.0,  C0_aorta ]


sss = element_numbers
C10Y_numpy = np.zeros(sss)
C0_numpy = np.zeros(sss)
C0_numpy_calc=np.zeros(sss)
damp_values=[4e5,5e9]
damp_regions_numpy=np.zeros(sss)
for cell_no in range(sss):
    if (Tags_element[cell_no] == 1 ):
     subdomain_no = 0
    else:
     subdomain_no = 1
    C10Y_numpy[cell_no] = C10Y_values[subdomain_no]
    C0_numpy[cell_no] = C0_values[subdomain_no]
    C0_numpy_calc[cell_no] = C0_values[subdomain_no]
    damp_regions_numpy[cell_no]=damp_values[subdomain_no]

damp_co=Function(V00)
damp_co.vector().set_local(damp_regions_numpy)
damp_co.vector().apply('insert')
C10Y.vector().set_local(C10Y_numpy)
C0.vector().set_local(C0_numpy)
C0_calc.vector().set_local(C0_numpy)
C10Y.vector().apply('insert')
C0.vector().apply('insert')
C0_calc.vector().apply('insert')


if rank == 0:
 print ('number of elements:', element_numbers)
 print ('number of nodes:', node_numbers)
C_H = Function(V00)
C0_H = Function(V00)
C_H_temp = C_H.vector().get_local()
C_H_temp[:] = C_H_s
C_H.vector().set_local(C_H_temp)
C_H.vector().apply('insert')
C0_H_temp = C0_H.vector().get_local()
C0_H_temp[:] = C0_H_s
C0_H.vector().set_local(C0_H_temp)
C0_H.vector().apply('insert')





# print(len(temp_strain_calc))

# sys.exit()
###########################################################

'''define a discrete function space Vb over bmesh'''

Vb = VectorFunctionSpace(bmesh, 'Lagrange', 2)
Vb_scalar = FunctionSpace(bmesh, 'DG', 0)

###########################################################

'''Drichlet Boundary conditions'''
bc1 = DirichletBC(Vb.sub(1), 0.0 , facet_domains, 6) #wall_bottom
bc2 = DirichletBC(Vb.sub(1), 0.0 , facet_domains, 7) #wall_top
bc3z_top = DirichletBC(Vb,(0.0,0.0,0.0), Pinpoint([.0108, .043, 0]), 'pointwise')
bc4x_top = DirichletBC(Vb, (0.0,0.0,0.0), Pinpoint([0.0054, 0.043, -0.00935307]), 'pointwise')
bc3z_bot = DirichletBC(Vb, (0.0,0.0,0.0), Pinpoint([0.00599995, -.015, -0.0103922]), 'pointwise')
bc4x_bot = DirichletBC(Vb, (0.0,0.0,0.0), Pinpoint([0.0120001, -0.015, 6.899e-10]), 'pointwise')
bcs_b = [bc1,bc2,bc3z_top,bc4x_top,bc3z_bot,bc4x_bot]

###########################################################


'''Trial and test functions'''

du_b = TrialFunction(Vb)   # displacement
vb = TestFunction(Vb)
ub = Function(Vb) # the most recently computed solution
vel0, a0 , ub0 = Function(Vb), Function(Vb), Function(Vb)
n_function = FacetNormal(bmesh)
h = CellDiameter(bmesh)


###########################################################

######################################
# anisotropic problem
######################################
loc_flag=Constant(0.0)
I = Identity(3)   # define second order identity
if (Flag_generalized_alpha ): #generalized alpha integrator
 F_t = I + grad( (1.0 - alpha_f)* ub + alpha_f * ub0)
else: #Newmark or typical integrators
 F_t = I + grad(ub)
dir_tensor=outer(e_normal,e_normal)
dir_tensor_loc=outer(e_normal_loc,e_normal_loc)
growth_constant=Constant(0.0)
growth_constant_loc=Constant(0.0)
avalu=Function(V0)
growth_rate_local=Function(V0)
GROW=Constant(0.0)
F_growth=(avalu+GROW*growth_rate_local)*dir_tensor+I

F_growth_inv=inv(F_growth)
F_e = F_t*F_growth_inv
C_e = dot(F_e.T,F_e)
J_e = det(F_e)
J_t=det(F_t)
I1 = tr(C_e)  # first invariant of C
I2 =  0.5*( tr(C_e)**2 - tr(C_e*C_e) )  # second invariant of C
E = 0.5*(C_e - I)
F_iso = J_e**(-1.0/3.0)*F_e
C_iso = dot(F_iso.T, F_iso)
I1_iso = tr(C_iso)
I2_iso =  0.5*( tr(C_iso)**2 - tr(C_iso*C_iso) )
Dir_tensor = outer(e_circum ,e_circum)
I4_iso = inner(C_iso, Dir_tensor) 
E_output  = 0.5*(C_e - I)
strain_eps = inner(E_output, Dir_tensor)
Kappa = Function(Vb_scalar)
temp_array_Kappa = Kappa.vector().get_local() 
temp_array_Kappa[:] = Kappa_incomp
Kappa.vector().set_local(temp_array_Kappa)
dU_dJ = Kappa*(J_e-1)
S_vol = J_e*dU_dJ*inv(C_e)
P = Function(Vb_scalar)
P_wall = Function(Vb_scalar)
P_wall1 = Function(Vb_scalar)
K_ramp_value = Constant(1.) #ramp the anisotropic part
K_ramp = Function(V00)
temp_array_K_ramp = K_ramp.vector().get_local()
temp_array_K_ramp[:] = K_ramp_value
K_ramp.vector().set_local(temp_array_K_ramp)


###########################################################

'''Making weak form'''


d_psi_bar_d_C_bar = ( C10Y + C0_calc*C0 * C_H * C1_H * exp(C1_H*(I1_iso - 3.0)) ) * I
d_psi_bar_d_C_bar = d_psi_bar_d_C_bar + ( C0*C0_calc * C0_H * C2_H * (1.0 - (1./(I4_iso**.5)) ) * exp(C2_H* ( (I4_iso**.5) - 1.0)**2)  ) *Dir_tensor
S_isc = 2.0*J_e**(-2.0/3.0)*Dev(d_psi_bar_d_C_bar, C_e)
second_PK_stress =  S_vol + S_isc
vel = (gamma/(beta2*dt))*(ub - ub0) - (gamma/beta2 - 1.0)*vel0 - dt*(gamma/(2.0*beta2) - 1.0)*a0 #vel and acceleration at t_(n+1)
a  = (1.0/(beta2*dt**2))*(ub - ub0 - dt*vel0) - (1.0/(2.0*beta2) - 1.0) * a0
if (Flag_generalized_alpha ): #generalized alpha integrator
 a = (1.0 - alpha_m) * a + alpha_m * a0
 vel = (1.0 - alpha_f) * vel + alpha_f * vel0
Side_flag = Constant(1.0)
Functional_b_isotropic = inner(dot(F_e,second_PK_stress),grad(vb))*dx() + Side_flag* P*J_e*inner(inv(F_e.T)*n_function,vb)*ds(1) #+P.. will apply pressure into the surface p is in line 690
Side_flag2 = Constant(0.0)
Functional_b_isotropic = Functional_b_isotropic + Side_flag2 * P*J_e* inner(inv(F_e.T)*n_function,vb)*(ds(2))#presssure on the aortic side of Leaflet
Functional_b_isotropic = Functional_b_isotropic +  P_wall*J_e*inner(inv(F_e.T)*n_function,vb)* (ds(4)) 
Functional_b_isotropic = Functional_b_isotropic +  P_wall1*J_e*inner(inv(F_e.T)*n_function,vb)* (ds(3)) 
Functional_b_isotropic = Functional_b_isotropic + rho * dot(a,vb) * dx() +damp*dot(vel,vb)*dx()
Functional_b_isotropic = Functional_b_isotropic + beta/h * dot(ub,n_function) * dot(vb,n_function)* ds(5) + beta/h * dot(ub,n_function) * dot(vb,n_function)* ds(8)
J_b_isotropic = derivative(Functional_b_isotropic, ub, du_b)
if rank == 0:
    print ('Assembling solver...')
problem_isotropic = NonlinearVariationalProblem(Functional_b_isotropic, ub, bcs_b, J_b_isotropic)
solver_isotropic = NonlinearVariationalSolver(problem_isotropic)


###########################################################

###########################################################

'''saving everything to see they are impoerted correctly'''

file_BC = File(output_dir + 'facet_domains_main.pvd')
# file_BC << facet_domains
file_circum = File(output_dir + 'circum_main.pvd')
# file_circum << e_circum
file_normal_direction_loc = File(output_dir + 'normal_loc.pvd')
#file_normal_direction_loc << e_normal_loc
file_normal_direction = File(output_dir + 'normal_main.pvd')
#file_normal_direction << e_normal
file_leaflet = File(output_dir + 'leaflet_main.pvd')
# file_leaflet << Tag_leaflet_node
Tags=Tag_leaflet_element.vector().get_local()
# file_material_3 = File(output_dir + 'tags.pvd')
file_material = File(output_dir + 'C10y.pvd')
# file_material << C10Y
file_material_2 = File(output_dir + 'C0.pvd')
# file_material_2 << C0

file_material_3= File(output_dir + 'C0_H.pvd')
file_isotropic_ub = File(output_dir + 'displacement_multi.pvd')
file_isotropic_ub_dynamic=File(output_dir + 'displacement_multi_dynamic.pvd')
file_isotropic_ub_dynamic_Global_growth=File(output_dir + 'displacement_multi_dynamic_global_growth.pvd')
file_sigma = File(output_dir + 'stress_aniso.pvd')
file_sigma1 = File(output_dir + 'stress_cycle.pvd')
Filename_ub = 'disp_anisoHumphYin_'
Filename_strain = 'strain_'
########################################
calcif_tags=np.zeros((element_numbers))
# C_0_temp=np.zeros((element_numbers))
# C_0_temp[:]=C0.vector().get_local()[:]
# for hh in range (element_numbers):
#   if calcif_tags[hh]==1:
#     C_0_temp[hh]=C_0_temp[hh]*100
# C0.vector().set_local(C_0_temp[:])
# file_material_2<<C0
# sys.exit()
###########################################################
if rank == 0:
    print ('Entering loop:')
###########################################################
'''manual projection:'''

w_m = TrialFunction(V00)
v_m = TestFunction(V00)
a_m = w_m * v_m * dx
L_m = I4_iso * v_m * dx
I_4_proj = Function(V00)
# #manual projection 2:
w_m2 = TrialFunction(V00)
v_m2 = TestFunction(V00)
a_m2 = w_m2 * v_m2 * dx
L_m2 = strain_eps * v_m2 * dx
strain_eps_proj = Function(V00)

###########################################################

'''solver parameter settiing'''

#solver.parameters.update(snes_solver_parameters)
prm = solver_isotropic.parameters
#print prm["newton_solver"]["maximum_iterations"]
prm["newton_solver"]["maximum_iterations"] = 200
#prm["newton_solver"]["relaxation_parameter"] = 0.6 #.7
# prm["newton_solver"]["monitor_convergence"] = True
# prm["newton_solver"]["error_on_nonconvergence"] = False
prm["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = True
#prm["newton_solver"]["krylov_solver"]["absolute_tolerance"] = 1e-9 
prm['newton_solver']['absolute_tolerance'] = 1E-9#1E-5
#prm['newton_solver']['relative_tolerance'] = 1E-1
prm['newton_solver']['linear_solver'] = 'tfqmr' #'gmres' bicgstab
prm['newton_solver']['preconditioner'] = 'petsc_amg'
prm["newton_solver"]["krylov_solver"]["monitor_convergence"] = False
prm['newton_solver']['krylov_solver']['maximum_iterations'] = 20000


###########################################################
q_degree = 5
dx = dx(metadata={'quadrature_degree': q_degree})
###########################################################

'''Neuman Boundary Condition '''
Time_array = Pressure_BC.time_BC()
# Time_array1=Pressure_BC.time_BC()
Time_array2=P_trans_shift.time_BC()
P_shift=P_trans_shift.P_trans_added()
P_trans_array = Pressure_BC.P_trans_BC()
P_aort_array = Pressure_BC.P_aort_BC()
P_vent_array = Pressure_BC.P_vent_BC()
F_interp_trans = np.interp(t,Time_array,P_trans_array)
F_interp_aort = np.interp(t,Time_array,P_aort_array)
F_interp_vent = np.interp(t,Time_array,P_vent_array)

###########################################################
count = 0
Flag_ramp = False #if true ramps the anisotropic term
t_prev = 0.
count_2=0
N_cycles = 34
strain_peak_out = Function(V0)
count=0
cycle_number=0
# sys.exit()
count_saving_file1=0
count_saving_file2=0

###########################################################
Flag_global_growth=True
Flag_local_growth=False
Flag_cardiac_cycle=False

###########################################################


stiffness_factore_local=50
dt_growth=0.00001
C_0_temp=np.zeros((element_numbers))
C_0_temp_calc=np.zeros((element_numbers))

###########################################################
xdmffile_c0 = XDMFFile(output_dir + 'stiffness.xdmf')
# for iiii in range (30):
  




#### Constants initial phase from Amir's paper
V=27e-8	
WSS0=1
sci=1  #### rate of ldl influx
d_ldl=3e-4  ### LDL oxidation
d_l=2.4e-5   ### LDL diffusing out
C_ldl_in=9.44e-5 ## Influx of LDL into the leaflets in abscense of flow
tou_0=1   ### refrence WSS
k_l=1.2e-18    ### macrophage reaction with ox-LDL
d_m=1.15e-6    #### monoyte diffrentiation to macrophage
m_d=2.76e-6		#### monocyte apoptosis rate
m_r=6.6e-12		### monocyte penetration rate at zero wss
C_m1=5.5e11		## monocyte concentration at the lumen wall
alpha=5.76e-6	## = M_r1/M_r2
M_r1=2.83e-11	##### ox-LDL leading to foam cell	
M_r2=9.25e-24	##### foam cell formation
k_f=1.7e-7		### foam cell efferocytosis and defradation rate	



##### calcification phase constants from Amir's paper
Volume=27e-8		### total aortic valve volume
lamda=1.28e-4		##### latent TGF_B degradation rate
lamda_a=5.78e-3		#### active TGF_B activation rate by macrophage
a_F=5.2e-15			### TGF_B activation rate by macrophages
k_prime=4.3e-6  #### 	TGF to calcium
k_c=4.3e-6      ####   calcification rate
gamma1=0.09			####  calcification unit conversion
d_Ca=1				####   calcification removal rate
strain_peak=0.2
### constant functions
def F_l_tou(wss):
	FL_tou=C_ldl_in/ ( 1 + wss/WSS0 )
	return FL_tou

def F_m(wss):
	V=27e-8	
	a=m_r/(1+wss/WSS0)*C_m1
	return a

def F_t(Cf):
	return (3.3e-7*Cf*Volume)/(Cf*Volume+2.84e4)


def h(CA):
	# epsil=(0.2*CA+30)/(CA+333.33)
	return 4.435e4*exp(6.404*CA)-4.435e4

def ode_system(t,x,strain1):

	
	C_ldl=x[0]
	C_oxldl=x[1]
	C_m=x[2]
	C_M=x[3]
	C_f=x[4]
	C_F=C_f+C_M
	TGF=x[5]
	TGF_act=x[6]
	Ca=x[7]
	strain_threshhold=0.07

	
	if strain1>=strain_threshhold:
		factor=1

	else:
		factor=0
	calcification_Agatston2 =  Ca*( gamma1 )
	strain = ( strain1* calcification_Agatston2 + 30 ) /  (calcification_Agatston2 + 333.33);


	dydt=np.zeros(8)
	wssq = 36 / (181 + calcification_Agatston2 );
	# wssq=36/(Ca+181)

	dydt[0]= F_l_tou(0.5)-d_ldl*C_ldl-d_l*C_ldl
	dydt[1]= d_ldl*C_ldl-k_l*C_oxldl*C_M
	dydt[2]=F_m(wssq)*C_oxldl - d_m*C_m-m_d*C_m
	dydt[3]=d_m*C_m - alpha*k_l*C_oxldl*C_M
	dydt[4]=alpha*k_l*C_oxldl*C_M
	dydt[5]=F_t(C_F)-lamda*TGF-a_F*C_F*TGF
	dydt[6]=a_F*C_F*TGF-lamda_a*TGF_act
	dydt[7]=(gamma1*k_prime*TGF_act*(1+h(strain)))*factor#-d_Ca*Ca#)-d_Ca*Ca
	# ode_sys=[dC_ldl_dt,dC_oxldl_dt,dC_m_dt,dC_M_dt,dC_f_dt,dTGF_dt,dTGF_act_dt,dCa_dt]
	return dydt

speciesNames = ['C_ldl','C_oxldl','C_m','C_M','C_f','TGF','TGF_act','Ca'] 
# print(speciesNames[0])
def initial(z):
	y=z
	return y
tot_time=365*24*3600*5
dt_o=24*3600.*365./12

# t_ode=[]
# TIME=[]
# while r.successful() and r.t <= tot_time:
# 	if r.t<=2000:
# 		dt=0.1
# 	else:
# 		dt=24*3600.*365.//2.//200

# 	r.integrate(r.t + dt)
# 	results = np.append(results,[r.y],axis=0)
# 	t_ode.append(r.t)
# 	year=t[-1]/3600./24./365.
# 	TIME.append(year)
# 	print(year)





LL=1

FFF=np.zeros((local_dim))
par_updat=np.zeros((local_dim,8))

# print(par_updat[0,:])

TIME=[]

# calcif_tags
#   xdmffile_c0.write(C0, iiii)
# sys.exit()
###########################################################
temp_strain_calc=np.zeros((element_numbers))
temp_strain_calc1=np.zeros((element_numbers))
total_strain_np=np.zeros((element_numbers))
count_timestep=0
strain_threshhold=0.07
strain_NODE_dol=Function(V0)
strain_node_np=np.zeros(node_numbers)

strain_ode_np=np.zeros(local_dim)



growth_rate_dol=Function(V0)


new_numpy_growth=np.zeros((local_dim))
grow_rate_dol_tonp=np.zeros(local_dim)

t_ode=[]
# dt_s=0.
while cycle_number <40:
	tt=0
	vel0.vector()[:] = vel0.vector() * 0.
	a0.vector()[:] = a0.vector() * 0.
	xdmffile_u = XDMFFile(output_dir + 'displacement_'+str(cycle_number)+'.xdmf')
	xdmffile_s = XDMFFile(output_dir + 'stress_'+str(cycle_number)+'.xdmf')
	xdmffile_strain = XDMFFile(output_dir + 'strain_'+str(cycle_number)+'.xdmf')
	xdmffile_growth_rate = XDMFFile(output_dir + 'GR_'+str(cycle_number)+'.xdmf')
	xdmffile_strain_ave = XDMFFile(output_dir + 'ave_strain_'+str(cycle_number)+'.xdmf')
	xdmffile_C0_glob = XDMFFile(output_dir + 'c0glob_'+str(cycle_number)+'.xdmf')
	xdmffile_c0_loc = XDMFFile(output_dir + 'c0_loc_'+str(cycle_number)+'.xdmf')
	xdmffile_strain_node=XDMFFile(output_dir + 'strain_node'+str(cycle_number)+'.xdmf')
	xdmffile_u.parameters["flush_output"] = True
	xdmffile_u.parameters["functions_share_mesh"] = True

	# xdmf_gr_checkpoint=XDMFFile(output_dir + 'growth_checkpoint'+str(cycle_number)+'.xdmf')

	# xdmffile_u.parameters["flush_output"] = True
	# xdmffile_u.parameters["functions_share_mesh"] = True
  ###########################################################
  ###cardiac cycle loop
	if cycle_number%2==0 :
		Flag_local_growth=False
		Flag_ode_solver=False
		Flag_cardiac_cycle=True
		count_timestep=0
		temp_strain_calc=np.zeros((element_numbers))
	else:
		Flag_ode_solver=True
		Flag_local_growth=True
		Flag_cardiac_cycle=False
	if (Flag_cardiac_cycle):
		rho.assign(1100)
		while tt<=0.42:
			if(1):
				if tt<0.0:
					dt_s=dt_coarser
				elif tt<0.04:
					dt_s=dt_fine
				elif tt< 0.25:
					dt_s=dt_med
				else:
					dt_s=dt_coarse
			t_prev=tt
			tt+=dt_s
			dt.assign(dt_s)
			temp_array_P = P.vector().get_local()
			temp_array_P_wall = P_wall.vector().get_local()
			temp_array_P_wall1 = P_wall1.vector().get_local()
			T_alpha = (1.0 - alpha_f) * tt + (alpha_f * t_prev)
			P_current_trans = np.interp(T_alpha,Time_array,P_trans_array)
			P_current_wall = np.interp(T_alpha,Time_array,P_aort_array)
			P_current_wall1 = np.interp(T_alpha,Time_array,P_vent_array)
			shift=np.interp(T_alpha,Time_array2,P_shift)
			if (P_current_trans < 0 ):
				Side_flag.assign(0.0)
				Side_flag2.assign(1.0)
				P_current_trans_in = -P_current_trans
				vent_side=False
			else:
				P_current_trans_in=P_current_trans#+cycle_number*shift
				Side_flag.assign(1.0)
				Side_flag2.assign(0.0)
				vent_side=True
			if rank==0:
				print('cardiac_cycle')
				print ('cycle=', cycle_number)
				print ('time=', tt)
				print ("tot_pressure", P_current_trans_in)
				if vent_side:
					print ("pressure_in_vent=", P_current_trans)
					print ("shifting_pressure", shift)
				else:
					print ("pressure_in aort=", P_current_trans)
			temp_array_P[:] = P_current_trans_in
			temp_array_P_wall1[:] = P_current_wall1
			temp_array_P_wall[:] = P_current_wall
			P.vector().set_local(temp_array_P)
			P_wall.vector().set_local(temp_array_P_wall)
			P_wall1.vector().set_local(temp_array_P_wall1)
			solver_isotropic.solve()
			update(ub,ub0,vel0,a0,beta2,gamma,dt_s)
			solve(a_m2 == L_m2, strain_eps_proj)
			temp_strain_calc+=strain_eps_proj.vector().get_local()
			# temp_strain_calc=strain_eps_proj.vector().get_local()
			if count_saving_file1%4==0:
				stress_tensor = (1./J_e)* F_e*second_PK_stress*F_e.T
				TTTT=TensorFunctionSpace(bmesh, 'DG', 0)
				eqSigma = Function(TTTT)
				eqSigma.assign(project(stress_tensor,TTTT))
				eqSigma.rename("stress","Label")
				ub.rename("displacement","Label")
				strain_eps_proj.rename("strain","Label")
				xdmffile_u.write(ub, tt)
				xdmffile_u.write(eqSigma, tt)
				xdmffile_u.write(strain_eps_proj,tt)
			count_saving_file1+=1
			count_timestep+=1
		temp_strain_calc1=temp_strain_calc/count_timestep
		# strain_ave.vector().set_local(temp_strain_calc)
		# strain_ave.vector().apply('insert')
		# xdmffile_strain_ave.write(strain_ave)
		avalu.vector().set_local(new_numpy_growth)
		avalu.vector().apply('insert')

		for kkk in range(count_f):
			point_1=(V0.dofmap().cell_dofs(Fibrosa_IDs[kkk])[0])
			point_2=(V0.dofmap().cell_dofs(Fibrosa_IDs[kkk])[1])
			point_3=(V0.dofmap().cell_dofs(Fibrosa_IDs[kkk])[2])
			point_4=(V0.dofmap().cell_dofs(Fibrosa_IDs[kkk])[3])
			strain_node_np[point_1]=abs(temp_strain_calc1[Fibrosa_IDs[kkk]])
			strain_node_np[point_2]=abs(temp_strain_calc1[Fibrosa_IDs[kkk]])
			strain_node_np[point_3]=abs(temp_strain_calc1[Fibrosa_IDs[kkk]])
			strain_node_np[point_4]=abs(temp_strain_calc1[Fibrosa_IDs[kkk]])
			if  abs(temp_strain_calc1[Fibrosa_IDs[kkk]])>strain_threshhold:
				calcif_tags[Fibrosa_IDs[kkk]]+=1
				point_1=(V0.dofmap().cell_dofs(Fibrosa_IDs[kkk])[0])
				point_2=(V0.dofmap().cell_dofs(Fibrosa_IDs[kkk])[1])
				point_3=(V0.dofmap().cell_dofs(Fibrosa_IDs[kkk])[2])
				point_4=(V0.dofmap().cell_dofs(Fibrosa_IDs[kkk])[3])
				growth_local_rate_numpy[point_1]=1
				growth_local_rate_numpy[point_2]=1
				growth_local_rate_numpy[point_3]=1
				growth_local_rate_numpy[point_4]=1
		strain_NODE_dol.vector().set_local(strain_node_np[:])
		strain_NODE_dol.vector().apply('insert')
		xdmffile_strain_node.write(strain_NODE_dol)
		strain_ode_np=strain_NODE_dol.vector().get_local()
		growth_rate_dol.vector().set_local(growth_local_rate_numpy[:])
		growth_rate_dol.vector().apply('insert')
		grow_rate_dol_tonp=growth_rate_dol.vector().get_local()

	if Flag_ode_solver:
		for j in range(local_dim):
			r = ode(ode_system).set_integrator('vode', method='bdf', order=5, rtol=0, atol=1e-4, with_jacobian=True)
			r.set_initial_value(initial(par_updat[j,:]),0).set_f_params(strain_ode_np[j])
			results = np.empty([0,len(speciesNames)])
			while r.successful() and r.t <= tot_time:
				r.integrate(r.t + dt_o)
				results = np.append(results,[r.y],axis=0)
				t_ode.append(r.t)
				year=t_ode[-1]/3600./24./365.
				TIME.append(year)
			FFF[j]=results[-1,-1]/2000
			# print(FFF[i],i)
			par_updat[j,:]=results[-1,:]
		# print(max(FFF))
		if rank==0:
			print('ODE_DONE!', )
			print(max(FFF))
	if (Flag_local_growth):
		loc_flag.assign(1.0)
		rho.assign(0.0)
		C_0_temp_calc[:]=C0_calc.vector().get_local()[:]
		for hh in range (element_numbers):
			if calcif_tags[hh]==1:
				if C_0_temp_calc[hh]<=2:
					C_0_temp_calc[hh]=C_0_temp_calc[hh]*stiffness_factore_local
		C0_calc.vector().set_local(C_0_temp_calc[:])
		C0_calc.rename("c0loc","Label")
		xdmffile_c0_loc.write(C0_calc)

		growth_rate_local.vector().set_local(FFF)
		growth_rate_local.vector().apply('insert')
		for qq in range (151):
			GROW.assign(0.03*qq)
			if rank == 0:
				print('GROW',qq)
			solver_isotropic.solve()
			if qq%30==0:
				ub.rename("displacement_growth","Label")
				xdmffile_growth_rate.write(ub,qq)
		new_numpy_growth+=float(GROW)*FFF
		if rank ==0:
			print('growth_max', max(new_numpy_growth))		
	cycle_number=cycle_number+1
	if rank ==0:
		print("cycle_number",cycle_number)






sys.exit()





























