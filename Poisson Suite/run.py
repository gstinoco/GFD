# All the codes presented below were developed by:
#   Dr. Gerardo Tinoco Guerrero
#   Universidad Michoacana de San Nicolás de Hidalgo
#   gerardo.tinoco@umich.mx
#
# With the funding of:
#   National Council of Science and Technology, CONACyT (Consejo Nacional de Ciencia y Tecnología, CONACyT). México.
#   Coordination of Scientific Research, CIC-UMSNH (Coordinación de la Investigación Científica de la Universidad Michoacana de San Nicolás de Hidalgo, CIC-UMSNH). México
#   Aula CIMNE Morelia. México
#
# Date:
#   November, 2022.
#
# Last Modification:
#   November, 2022.

import math
from scipy.io import loadmat
from sys import path
path.insert(0, 'General/')
import Errors
import Graph
import Poisson_2D

# Region data is loaded.
# Triangulation and unstructured cloud of points to work in.
nube = 'CAB_1'
# This region can be changed for any other tringulation or unestructured cloud of points on Regions/Clouds/ or with any other region with the same file data structure.

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
#   \phi = 2e^{2x+y}
#
#   f = 10e^{2x+y}

def phi(x,y):
    fun = 2*math.exp(2*x+y)
    return fun

def f(x,y):
    fun = 10*math.exp(2*x+y)
    return fun

# Poisson 2D computed in a logically rectangular mesh
phi_ap, phi_ex = Poisson_2D.Poisson_Mesh(x, y, phi, f)
er = Errors.Mesh_Static(x, y, phi_ap, phi_ex)
print('The mean square error in the mesh', malla, 'is: ', er)
Graph.Mesh_Static(x, y, phi_ap, phi_ex)

# Poisson 2D computed in a triangulation
phi_ap, phi_ex, vec = Poisson_2D.Poisson_Tri(p, pb, tt, phi, f)
er = Errors.Cloud_Static(p, vec, phi_ap, phi_ex)
print('The mean square error in the triangulation', nube, 'is: ', er)
Graph.Cloud_Static(p, phi_ap, phi_ex)

# Poisson 2D computed in an unstructured cloud of points
phi_ap, phi_ex = Poisson_2D.Poisson_Cloud(p, pb, vec, phi, f)
er = Errors.Cloud_Static(p, vec, phi_ap, phi_ex)
print('The mean square error in the unstructured cloud of points', nube, 'is: ', er)
Graph.Cloud_Static(p, phi_ap, phi_ex)