# On going, not yet finished

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
from sys import path
path.insert(0, 'General/')
import Gammas
import Neighbors

def Mesh(x, y, f, nu, t):
    # 2D Diffusion Equation implemented in Logically Rectangular Meshes.
    # 
    # This routine calculates an approximation to the solution of Diffusion equation in 2D using a Generalized Finite Differences scheme in logically rectangular meshes.
    # 
    # The problem to solve is:
    # 
    # \frac{\partial q}{\partial t} + \frac{\partial r}{\partial x} + \frac{\partial s}{\partial y} = 0
    #
    # \frac{\partial r}{\partial t} + \frac{2r}{a}\frac{\partial r}{\partial x} + ga\frac{\partial q}{\partial x} + \frac{r}{a}\frac{\partial s}{\partial y} + \frac{s}{a}\frac{\partial r}{\partial y} - fs + c_f \frac{r}{a^2}\sqrt{r^2+s^2} = 0
    #
    # \frac{\partial s}{\partial t} + \frac{r}{a}\frac{\partial s}{\partial x} + \frac{s}{a}\frac{\partial r}{\partial x} + \frac{2s}{a}\frac{\partial s}{\partial y} + ga\frac{\partial q}{\partial y} + fs + c_f \frac{s}{a^2}\sqrt{r^2+s^2} = 0
    # 
    # Input parameters
    #   x           m x n           Array           Array with the coordinates in x of the nodes.
    #   y           m x n           Array           Array with the coordinates in y of the nodes.
    #   f                           Function        Function declared with the boundary condition.
    #   nu                          Real            Diffusion coefficient.
    #   t                           Integer         Number of time steps considered.
    # 
    # Output parameters
    #   u_ap        m x n x t       Array           Array with the approximation computed by the routine.
    #   u_ex        m x n x t       Array           Array with the theoretical solution.

    # Variable initialization
    me   = x.shape                                                                  # The size of the mesh is found.
    m    = me[0]                                                                    # The number of nodes in x.
    n    = me[1]                                                                    # The number of nodes in y.
    T    = np.linspace(0,1,t)                                                       # Time discretization.
    dt   = T[1] - T[0]                                                              # dt computation.
    u_ap = np.zeros([m, n, t])                                                      # u_ap initialization with zeros.
    u_ex = np.zeros([m, n, t])                                                      # u_ex initialization with zeros.

    # Boundary conditions
    for k in np.arange(t):
        for i in np.arange(m):                                                      # For each of the nodes on the x boundaries.
            u_ap[i, 0,   k] = f(x[i, 0], y[i, 0], T[k], nu)                         # The boundary condition is assigned at the first y.
            u_ap[i, n-1, k] = f(x[i, n-1], y[i, n-1], T[k], nu)                     # The boundary condition is assigned at the last y.
        for j in np.arange(n):                                                      # For each of the nodes on the y boundaries.
            u_ap[0,   j, k] = f(x[0, j], y[0, j], T[k], nu)                         # The boundary condition is assigned at the first x.
            u_ap[m-1, j, k] = f(x[m-1, j], y[m-1, j], T[k], nu)                     # The boundary condition is assigned at the last x.
  
    # Initial condition
    for i in np.arange(m):                                                          # For each of the nodes on x.
        for j in np.arange(n):                                                      # For each of the nodes on y.
            u_ap[i, j, 0] = f(x[i, j], y[i, j], T[0], nu)                           # The initial condition is assigned.

    # Computation of Gamma values
    L = np.vstack([[0], [0], [2*nu*dt], [0], [2*nu*dt]])                            # The values of the differential operator are assigned.
    Gamma = Gammas.Mesh(x, y, L)                                                    # Gamma computation.

    # A Generalized Finite Differences Method
    for k in np.arange(1,t):                                                        # For each time step.
        for i in np.arange(1,m-1):                                                  # For each of the nodes on the x axis.
            for j in np.arange(1,n-1):                                              # For each of the nodes on the y axis.
                u_ap[i, j, k] = u_ap[i, j, k-1] + (\
                    Gamma[i, j, 0]*u_ap[i    , j    , k-1] + \
                    Gamma[i, j, 1]*u_ap[i + 1, j    , k-1] + \
                    Gamma[i, j, 2]*u_ap[i + 1, j + 1, k-1] + \
                    Gamma[i, j, 3]*u_ap[i    , j + 1, k-1] + \
                    Gamma[i, j, 4]*u_ap[i - 1, j + 1, k-1] + \
                    Gamma[i, j, 5]*u_ap[i - 1, j    , k-1] + \
                    Gamma[i, j, 6]*u_ap[i - 1, j - 1, k-1] + \
                    Gamma[i, j, 7]*u_ap[i    , j - 1, k-1] + \
                    Gamma[i, j, 8]*u_ap[i + 1, j - 1, k-1])                         # u_ap es calculated at the central node.

    # Theoretical Solution
    for k in np.arange(t):                                                          # For all the time steps.
        for i in np.arange(m):                                                      # For all the nodes on x.
            for j in np.arange(n):                                                  # For all the nodes on y.
                u_ex[i, j, k] = f(x[i, j], y[i, j], T[k], nu)                       # The theoretical solution is computed.

    return u_ap, u_ex