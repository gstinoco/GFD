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

import numpy as np
from scipy.io import loadmat
from sys import path
path.insert(0, 'General/')
import Graph
import Wave_Cloud_1

# Region data is loaded.
# Triangulation or unstructured cloud of points to work in.
region = 'CUA'
cloud = '41'
# This region can be changed for any other triangulation or unstructured cloud of points on Regions/ or with any other region with the same file data structure.

# All data is loaded from the file
mat = loadmat('Regions/Clouds/New/Holes/' + region + cloud + '.mat')

# Node data is saved
p   = mat['p']
tt  = mat['tt']
if tt.min() == 1:
    tt -= 1

# Number of Time Steps
t = 2000

# Wave coefficient
c = 1

# Lambda
lam = 0.5

# Boundary conditions
# The boundary conditions are defined as
#     f = \cos(\pi ct\sqrt{2})\sin(\pi x)\sin(\pi y)
#
# And its derivative
#     g = -(\pi c\sqrt{2})\sin(\pi x)\sin(\pi y)\sin(\pi ct\sqrt{2})

def fWAV(x, y, t, c):
    #fun = np.cos(np.sqrt(2)*np.pi*c*t)*np.sin(np.pi*x)*np.sin(np.pi*y);
    fun = (1/2)*np.exp(-400*((x - c*t - 0.5)**2 + (y - c*t - 0.5)**2))
    return fun

def gWAV(x, y, t, c):
    #fun = -np.sqrt(2)*np.pi*c*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.sqrt(2)*np.pi*c*t)
    fun = c*(400*x + 400*y - 800*c*t - 240)*np.exp(-400*((x - c*t - 0.5)**2 + (y - c*t - 0.5)**2))
    return fun

# Wave Equation in 2D computed on a unstructured cloud of points.
u_ap, u_ex, vec = Wave_Cloud_1.Cloud(p, fWAV, gWAV, t, c)
#er = Errors.Cloud_Transient(p, vec, u_ap, u_ex)
#print('The maximum mean square error in the unstructured cloud of points with the explicit scheme is: ', er.max())
#Graph.Error(er)
Graph.Cloud_Transient(p, tt, u_ap, u_ex)
#Graph.Cloud_Transient_Vid(p, tt, u_ap, u_ex, 'CAB1')