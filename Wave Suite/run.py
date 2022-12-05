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
import Wave_2D
import Wave_2D_Implicit

# Region data is loaded.
# Triangulation and unstructured cloud of points to work in.
nube = 'CAB'
size = '1'
# This region can be changed for any other tringulation or unestructured cloud of points on Regions/Clouds/ or with any other region with the same file data structure.

# Number of Time Steps
t   = 500

# Wave coefficient
c = 1

# Lambda
lam = 0.9

# All data is loaded from the file
mat = loadmat('Wave Suite/Data/' + nube + '/' + nube + '_' + size + '.mat')

# Node data is saved
p   = mat['p']
pb  = mat['pb']
vec = mat['vec']
tt  = mat['t']
tt  -= 1

# Boundary conditions
# The boundary conditions are defined as
#     f = \cos(\pi ct\sqrt{2})\sin(\pi x)\sin(\pi y)
#
# And its derivative
#     g = -(\pi c\sqrt{2})\sin(\pi x)\sin(\pi y)\sin(\pi ct\sqrt{2})

def fWAV(x, y, t, c):
    fun = math.cos(math.sqrt(2)*math.pi*c*t)*math.sin(math.pi*x)*math.sin(math.pi*y);
    return fun

def gWAV(x, y, t, c):
    fun = -math.sqrt(2)*math.pi*c*math.sin(math.pi*x)*math.sin(math.pi*y)*math.sin(math.sqrt(2)*math.pi*c*t)
    return fun

# Wave Equation in 2D computed on triangulations.
u_ap, u_ex = Wave_2D.Wave_Tri(p, pb, tt, fWAV, gWAV, t, c)
er = Errors.Cloud_Transient(p, vec, u_ap, u_ex)
print('The maximum mean square error in the triangulation with the explicit scheme is: ', er.max())
Graph.Error(er)
Graph.Cloud_Transient(p, tt, u_ap, u_ex)

# Wave Equation in 2D computed on a unstructured cloud of points.
u_ap, u_ex = Wave_2D.Wave_Cloud(p, pb, vec, fWAV, gWAV, t, c)
er = Errors.Cloud_Transient(p, vec, u_ap, u_ex)
print('The maximum mean square error in the unstructured cloud of points with the explicit scheme is: ', er.max())
Graph.Error(er)
Graph.Cloud_Transient(p, tt, u_ap, u_ex)

# Wave Equation in 2D computed with an implicit algorithm on triangulations.
u_ap, u_ex = Wave_2D_Implicit.Wave_Tri(p, pb, tt, fWAV, gWAV, t, c, lam)
er = Errors.Cloud_Transient(p, vec, u_ap, u_ex)
print('The maximum mean square error in the triangulation with the implicit scheme is: ', er.max())
Graph.Error(er)
Graph.Cloud_Transient(p, tt, u_ap, u_ex)

# Wave Equation in 2D computed with an implicit algorithm on a unstructured cloud of points.
u_ap, u_ex = Wave_2D_Implicit.Wave_Cloud(p, pb, vec, fWAV, gWAV, t, c, lam)
er = Errors.Cloud_Transient(p, vec, u_ap, u_ex)
print('The maximum mean square error in the unstructured cloud of points with the implicit scheme is: ', er.max())
Graph.Error(er)
Graph.Cloud_Transient(p, tt, u_ap, u_ex)