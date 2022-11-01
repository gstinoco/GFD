import math
from scipy.io import loadmat
from sys import path
path.insert(0, 'General/')
import Errors
import Graph
import Diffusion_2D

# Region data is loaded.
# Triangulation and unstructured cloud of points to work in.
nube = 'CAB_1'
# This region can be changed for any other tringulation or unestructured cloud of points on Regions/Clouds/ or with any other region with the same file data structure.

# Number of Time Steps
t   = 1000

# Diffusion coefficient
nu = 0.4

# All data is loaded from the file
mat = loadmat('Regions/Clouds/' + nube + '.mat')

# Node data is saved
p   = mat['p']
pb  = mat['pb']
vec = mat['vec']
tt  = mat['t']

# Mesh to work in.
malla = 'CAB21'
# This region can be changed for any other mesh on Regions/Meshes/ or with any other region with the same file data structure.

# All data is loaded from the file
mat = loadmat('Regions/Meshes/' + malla + '.mat')

# Node data is saved
x  = mat['x']
y  = mat['y']

# Boundary conditions
# The boundary conditions are defined as
#   f = e^{-2*\pi^2vt}\cos(\pi x)cos(\pi y)

def fDIF(x, y, t, v):
    fun = math.exp(-2*math.pi**2*v*t)*math.cos(math.pi*x)*math.cos(math.pi*y)
    return fun

# Diffusion 2D computed in a logically rectangular mesh
u_ap, u_ex = Diffusion_2D.Diffusion_Mesh(x, y, fDIF, nu, t)
er = Errors.Mesh_Transient(x, y, u_ap, u_ex)
print('El error máximo cometido para el método es de: ', er.max())
Graph.Error(er)
Graph.Mesh_Transient(x, y, u_ap, u_ex)

# Diffusion 2D computed in a triangulation
u_ap, u_ex, vec = Diffusion_2D.Diffusion_Tri(p, tt, pb, fDIF, nu, t)
er = Errors.Cloud_Transient(p, vec, u_ap, u_ex)
print('El error máximo cometido para el método es de: ', er.max())
Graph.Error(er)
Graph.Cloud_Transient(p, u_ap, u_ex)

# Diffusion 2D computed in an unstructured cloud of points
u_ap, u_ex, vec = Diffusion_2D.Diffusion_Cloud(p, pb, vec, fDIF, nu, t)
er = Errors.Cloud_Transient(p, vec, u_ap, u_ex)
print('El error máximo cometido para el método es de: ', er.max())
Graph.Error(er)
Graph.Cloud_Transient(p, u_ap, u_ex)