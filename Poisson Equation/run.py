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
#   December, 2022.

import math
from scipy.io import loadmat
from sys import path
path.insert(0, 'General/')
import Errors
import Graph
import Poisson_2D

# Region data is loaded.
# Triangulation or unstructured cloud of points to work in.
region = 'CAB'
cloud = '1'
# Mesh size to work in.
mesh = '21'
# This region can be changed for any other triangulation, unstructured cloud of points or mesh on Regions/ or with any other region with the same file data structure.

# All data is loaded from the file
mat = loadmat('Regions/Clouds/' + region + '_' + cloud + '.mat')

# Node data is saved
p   = mat['p']
pb  = mat['pb']
vec = mat['vec']
tt  = mat['t']
if tt.min() == 1:
    tt -= 1

# All data is loaded from the file
mat = loadmat('Regions/Meshes/' + region + mesh + '.mat')

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
phi_ap, phi_ex = Poisson_2D.Mesh(x, y, phi, f)
er = Errors.Mesh_Static(x, y, phi_ap, phi_ex)
print('The mean square error in the mesh', region, 'with', mesh, 'points per side is: ', er)
Graph.Mesh_Static(x, y, phi_ap, phi_ex)

# Poisson 2D computed in a triangulation
phi_ap, phi_ex, vec = Poisson_2D.Triangulation(p, pb, tt, phi, f)
er = Errors.Cloud_Static(p, vec, phi_ap, phi_ex)
print('The mean square error in the triangulation', region, 'with size', cloud, 'is: ', er)
Graph.Cloud_Static(p, tt, phi_ap, phi_ex)

# Poisson 2D computed in an unstructured cloud of points
phi_ap, phi_ex, vec = Poisson_2D.Cloud(p, pb, phi, f)
er = Errors.Cloud_Static(p, vec, phi_ap, phi_ex)
print('The mean square error in the unstructured cloud of points', region, 'with size', cloud, 'is: ', er)
Graph.Cloud_Static(p, tt, phi_ap, phi_ex)