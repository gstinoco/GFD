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
import Advection_2D
import Advection_2D_LW

# Region data is loaded.
# Triangulation or unstructured cloud of points to work in.
region = 'CAB'
cloud = '2'
# Mesh size to work in.
mesh = '41'
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

# Number of Time Steps
t   = 2000

# Advection coefficients
a = 0.3
b = 0.3

# Boundary conditions
# The boundary conditions are defined as
#   f = e^{-2*\pi^2vt}\cos(\pi x)cos(\pi y)

def fCAB(x, y, t, a, b):
    fun = 0.2*math.exp((-(x-.5-a*t)**2-(y-.5-b*t)**2)/.001)
    return fun

print('###################################### PURE ADVECTION ######################################')
# Advection 2D computed in a logically rectangular mesh
u_ap, u_ex = Advection_2D.Advection_Mesh(x, y, fCAB, a, b, t)
er = Errors.Mesh_Transient(x, y, u_ap, u_ex)
print('The maximum mean square error in the mesh', region, 'with', mesh, 'points per side is: ', er.max())
Graph.Error(er)
Graph.Mesh_Transient(x, y, u_ap, u_ex)

# Advection 2D computed in a triangulation
u_ap, u_ex, vec = Advection_2D.Advection_Tri(p, pb, tt, fCAB, a, b, t)
er = Errors.Cloud_Transient(p, vec, u_ap, u_ex)
print('The maximum mean square error in the triangulation', region, 'with size', cloud, 'is: ', er.max())
Graph.Error(er)
Graph.Cloud_Transient(p, u_ap, u_ex)

# Advection 2D computed in an unstructured cloud of points
u_ap, u_ex, vec = Advection_2D.Advection_Cloud(p, pb, fCAB, a, b, t)
er = Errors.Cloud_Transient(p, vec, u_ap, u_ex)
print('The maximum mean square error in the unstructured cloud of points', region, 'with size', cloud, 'is: ', er.max())
Graph.Error(er)
Graph.Cloud_Transient(p, u_ap, u_ex)

# Advection 2D computed in a logically rectangular mesh with Matrix Formulation
u_ap, u_ex = Advection_2D.Advection_Mesh_K(x, y, fCAB, a, b, t)
er = Errors.Mesh_Transient(x, y, u_ap, u_ex)
print('The maximum mean square error in the mesh', region, 'with', mesh, 'points per side is: ', er.max())
Graph.Error(er)
Graph.Mesh_Transient(x, y, u_ap, u_ex)

print('####################################### LAX-WENDROFF #######################################')

# Advection 2D computed in a logically rectangular mesh
u_ap, u_ex = Advection_2D_LW.Advection_Mesh(x, y, fCAB, a, b, t)
er = Errors.Mesh_Transient(x, y, u_ap, u_ex)
print('The maximum mean square error in the mesh', region, 'with', mesh, 'points per side is: ', er.max())
Graph.Error(er)
Graph.Mesh_Transient(x, y, u_ap, u_ex)

# Advection 2D computed in a triangulation
u_ap, u_ex, vec = Advection_2D_LW.Advection_Tri(p, pb, tt, fCAB, a, b, t)
er = Errors.Cloud_Transient(p, vec, u_ap, u_ex)
print('The maximum mean square error in the triangulation', region, 'with size', cloud, 'is: ', er.max())
Graph.Error(er)
Graph.Cloud_Transient(p, u_ap, u_ex)

# Advection 2D computed in an unstructured cloud of points
u_ap, u_ex, vec = Advection_2D_LW.Advection_Cloud(p, pb, fCAB, a, b, t)
er = Errors.Cloud_Transient(p, vec, u_ap, u_ex)
print('The maximum mean square error in the unstructured cloud of points', region, 'with size', cloud, 'is: ', er.max())
Graph.Error(er)
Graph.Cloud_Transient(p, u_ap, u_ex)

# Advection 2D computed in a logically rectangular mesh with Matrix Formulation
u_ap, u_ex = Advection_2D_LW.Advection_Mesh_K(x, y, fCAB, a, b, t)
er = Errors.Mesh_Transient(x, y, u_ap, u_ex)
print('The maximum mean square error in the mesh', region, 'with', mesh, 'points per side is: ', er.max())
Graph.Error(er)
Graph.Mesh_Transient(x, y, u_ap, u_ex)