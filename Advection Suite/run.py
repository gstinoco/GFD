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
import Advection_2D

# Region data is loaded.
# Triangulation and unstructured cloud of points to work in.
nube = 'CAB_1'
# This region can be changed for any other tringulation or unestructured cloud of points on Regions/Clouds/ or with any other region with the same file data structure.

# A, bmber of Time Steps
t   = 2000

# Advection coefficient
a = 0.3
b = 0.3

# All data is loaded from the file
mat = loadmat('Regions/Clouds/' + nube + '.mat')

# Node data is saved
p   = mat['p']
pb  = mat['pb']
vec = mat['vec']
tt  = mat['t']

# Mesh to work in.
malla = 'CAB41'
# This region can be changed for any other mesh on Regions/Meshes/ or with any other region with the same file data structure.

# All data is loaded from the file
mat = loadmat('Regions/Meshes/' + malla + '.mat')

# Node data is saved
x  = mat['x']
y  = mat['y']

# Boundary conditions
# The boundary conditions are defined as
#   f = e^{-2*\pi^2vt}\cos(\pi x)cos(\pi y)

def fCAB(x, y, t, a, b):
    fun = 0.2*math.exp((-(x-.5-a*t)**2-(y-.5-b*t)**2)/.001)
    return fun

# Advection 2D computed in a logically rectangular mesh
#u_ap, u_ex = Advection_2D.Advection_Mesh(x, y, fCAB, a, b, t)
#er = Errors.Mesh_Transient(x, y, u_ap, u_ex)
#print('The maximum mean square error in the mesh', malla, 'is: ', er.max())
#Graph.Error(er)
#Graph.Mesh_Transient(x, y, u_ap, u_ex)

# Advection 2D computed in a triangulation
#u_ap, u_ex, vec = Advection_2D.Advection_Tri(p, pb, tt, fCAB, a, b, t)
#er = Errors.Cloud_Transient(p, vec, u_ap, u_ex)
#print('The maximum mean square error in the triangulation', nube, 'is: ', er.max())
#Graph.Error(er)
#Graph.Cloud_Transient(p, u_ap, u_ex)

# Advection 2D computed in an unstructured cloud of points
#u_ap, u_ex, vec = Advection_2D.Advection_Cloud(p, pb, vec, fCAB, a, b, t)
#er = Errors.Cloud_Transient(p, vec, u_ap, u_ex)
#print('The maximum mean square error in the unstructured cloud of points', nube, 'is: ', er.max())
#Graph.Error(er)
#Graph.Cloud_Transient(p, u_ap, u_ex)

# Advection 2D computed in a logically rectangular mesh with Matrix Formulation
#u_ap, u_ex = Advection_2D.Advection_Mesh_K(x, y, fCAB, a, b, t)
#er = Errors.Mesh_Transient(x, y, u_ap, u_ex)
#print('The maximum mean square error in the mesh', malla, 'is: ', er.max())
#Graph.Error(er)
#Graph.Mesh_Transient(x, y, u_ap, u_ex)