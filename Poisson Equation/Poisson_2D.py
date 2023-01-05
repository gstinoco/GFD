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

def Mesh(x, y, phi, f):
    # 2D Poisson Equation implemented in Logically Rectangular Meshes.
    # 
    # This routine calculates an approximation to the solution of Poisson's equation in 2D using a Generalized Finite Differences scheme in logically rectangular meshes.
    # 
    # The problem to solve is:
    # 
    # \nabla^2 \phi = f
    # 
    # Input parameters
    #   x           m x n           Array               Array with the coordinates in x of the nodes.
    #   y           m x n           Array               Array with the coordinates in y of the nodes.
    #   phi                         function            Function declared with the boundary condition.
    #   f                           function            Function declared with the right side of the equation.
    # 
    # Output parameters
    #   u_ap        m x n           Array               Array with the approximation computed by the routine.
    #   u_ex        m x n           Array               Array with the theoretical solution.

    # Variable initialization
    me   = x.shape                                                                  # The size of the mesh is found.
    m    = me[0]                                                                    # The number of nodes in x.
    n    = me[1]                                                                    # The number of nodes in y.
    err  = 1                                                                        # err initialization in 1.
    tol  = np.finfo(float).eps                                                      # The tolerance is defined as eps
    u_ap = np.zeros([m,n])                                                          # u_ap initialization with zeros.
    u_ex = np.zeros([m,n])                                                          # u_ex initialization with zeros.

    # Boundary conditions
    for i in range(m):                                                              # For each of the nodes on the x boundaries.
        u_ap[i, 0]   = phi(x[i, 0],   y[i, 0])                                      # The boundary condition is assigned at the first y.
        u_ap[i, n-1] = phi(x[i, n-1], y[i, n-1])                                    # The boundary condition is assigned at the last y.
    for j in range(n):                                                              # For each of the nodes on the y boundaries.
        u_ap[0,   j] = phi(x[0,   j], y[0,   j])                                    # The boundary condition is assigned at the first x.
        u_ap[m-1, j] = phi(x[m-1, j], y[m-1, j])                                    # The boundary condition is assigned at the last x.

    # Computation of Gamma values
    L = np.vstack([[0], [0], [2], [0], [2]])                                        # The values of the differential operator are assigned.
    Gamma = Gammas.Mesh(x, y, L)                                                    # Gamma computation.

    # A Generalized Finite Differences Method
    while err >= tol:                                                               # As long as the error is greater than the tolerance.
        err = 0                                                                     # Error becomes zero to be able to update.
        for i in range(1,m-1):                                                      # For each of the nodes on the x axis.
            for j in range(1,n-1):                                                  # For each of the nodes on the y axis.
                t = (f(x[i, j], y[i, j]) - (              \
                    Gamma[i, j, 1]*u_ap[i + 1, j    ] + \
                    Gamma[i, j, 2]*u_ap[i + 1, j + 1] + \
                    Gamma[i, j, 3]*u_ap[i    , j + 1] + \
                    Gamma[i, j, 4]*u_ap[i - 1, j + 1] + \
                    Gamma[i, j, 5]*u_ap[i - 1, j    ] + \
                    Gamma[i, j, 6]*u_ap[i - 1, j - 1] + \
                    Gamma[i, j, 7]*u_ap[i    , j - 1] + \
                    Gamma[i, j, 8]*u_ap[i + 1, j - 1]))/Gamma[i, j, 0]              # u_ap is calculated at the central node.
                err = max(err, abs(t - u_ap[i, j]));                                # Error computation.
                u_ap[i,j] = t;                                                      # The previously computed value is assigned.
    
    # Theoretical Solution
    for i in range(m):                                                              # For all the nodes on x.
        for j in range(n):                                                          # For all the nodes on y.
            u_ex[i,j] = phi(x[i,j], y[i,j])                                         # The theoretical solution is computed.
    
    return u_ap, u_ex

def Triangulation(p, pb, tt, phi, f):
    # 2D Poisson Equation implemented in Triangulations.
    # 
    # This routine calculates an approximation to the solution of Poisson's equation in 2D using a Generalized Finite Differences scheme in triangulations.
    # 
    # The problem to solve is:
    # 
    # \nabla^2 \phi = f
    # 
    # Input parameters
    #   p           m x 2           Array           Array with the coordinates of the nodes.
    #   pb          b x 2           Array           Array with the coordinates of the boundary nodes.
    #   tt          n x 3           Array           Array with the correspondence of the n triangles.
    #   phi                         function        Function declared with the boundary condition.
    #   f                           function        Function declared with the right side of the equation.
    # 
    # Output parameters
    #   u_ap        m x 1           Array           Array with the approximation computed by the routine.
    #   u_ex        m x 1           Array           Array with the theoretical solution.
    
    # Variable initialization
    m    = len(p[:,0])                                                              # The total number of nodes is calculated.
    mf   = len(pb[:,0])                                                             # The number of boundary nodes is calculated.
    nvec = 9                                                                        # The maximum number of nodes.
    err  = 1                                                                        # err initialization in 1.
    tol  = np.finfo(float).eps                                                      # The tolerance is defined as eps
    u_ap = np.zeros([m])                                                            # u_ap initialization with zeros.
    u_ex = np.zeros([m])                                                            # u_ex initialization with zeros.
    
    # Boundary conditions
    for i in np.arange(mf):                                                         # For each of the boundary nodes.
        u_ap[i]   = phi(pb[i, 0], pb[i, 1])                                         # The boundary condition is assigned.
    
    # Neighbor search for all the nodes.
    vec = Neighbors.Triangulation(p, tt, nvec)                                      # Neighbor search with the proper routine.

    # Computation of Gamma values
    L = np.vstack([[0], [0], [2], [0], [2]])                                        # The values of the differential operator are assigned.
    Gamma = Gammas.Cloud(p, pb, vec, L)                                             # Gamma computation.

    # A Generalized Finite Differences Method
    while err >= tol:                                                               # As long as the error is greater than the tolerance.
        err = 0                                                                     # Error becomes zero to be able to update.
        for i in np.arange(mf, m):                                                  # For each of the interior nodes.
            utemp = 0                                                               # utemp is initialized with zero.
            nvec = sum(vec[i,:] != -1)                                              # The number of neighbors of the central node.
            for j in np.arange(1,nvec+1):                                           # For each of the neighbor nodes.
                utemp = utemp + Gamma[i, j]*u_ap[int(vec[i, j-1])]                  # utemp is computed.
            t = (f(p[i, 0], p[i, 1]) - utemp)/Gamma[i,0]                            # The central node is added to the approximation.
            err = max(err, abs(t - u_ap[i]));                                       # Error computation.
            u_ap[i] = t;                                                            # The previously computed value is assigned.
    
    # Theoretical Solution
    for i in range(m):                                                              # For all the nodes.
        u_ex[i] = phi(p[i,0], p[i,1])                                             # The theoretical solution is computed.

    return u_ap, u_ex, vec

def Cloud(p, pb, phi, f):
    # 2D Poisson Equation implemented in unstructured clouds of points.
    # 
    # This routine calculates an approximation to the solution of Poisson's equation in 2D using a Generalized Finite Differences scheme in unstructured clouds of points.
    # 
    # The problem to solve is:
    # 
    # \nabla^2 \phi = f
    # 
    # Input parameters
    #   p           m x 2           Array           Array with the coordinates of the nodes.
    #   pb          b x 2           Array           Array with the coordinates of the boundary nodes.
    #   phi                         function        Function declared with the boundary condition.
    #   f                           function        Function declared with the right side of the equation.
    # 
    # Output parameters
    #   u_ap        m x 1           Array           Array with the approximation computed by the routine.
    #   u_ex        m x 1           Array           Array with the theoretical solution.

    # Variable initialization
    m    = len(p[:,0])                                                              # The total number of nodes is calculated.
    mf   = len(pb[:,0])                                                             # The number of boundary nodes is calculated.
    err  = 1                                                                        # err initialization in 1.
    tol  = np.finfo(float).eps                                                      # The tolerance is defined as eps
    u_ap = np.zeros([m])                                                            # u_ap initialization with zeros.
    u_ex = np.zeros([m])                                                            # u_ex initialization with zeros.

    # Boundary conditions
    for i in np.arange(mf):                                                         # For each of the boundary nodes.
        u_ap[i] = phi(pb[i, 0],   pb[i, 1])                                         # The boundary condition is assigned.
    
    # Neighbor search for all the nodes.
    vec = Neighbors.Cloud(p, pb, 9)                                                 # Neighbor search with the proper routine.

    # Computation of Gamma values
    L = np.vstack([[0], [0], [2], [0], [2]])                                        # The values of the differential operator are assigned.
    Gamma = Gammas.Cloud(p, pb, vec, L)                                             # Gamma computation.

    # A Generalized Finite Differences Method
    while err >= tol:                                                               # As long as the error is greater than the tolerance.
        err = 0                                                                     # Error becomes zero to be able to update.
        for i in np.arange(mf, m):                                                  # For each of the interior nodes.
            utemp = 0                                                               # utemp is initialized with zero.
            nvec = sum(vec[i,:] != -1)                                              # The number of neighbors of the central node.
            for j in np.arange(1,nvec+1):                                           # For each of the neighbor nodes.
                utemp = utemp + Gamma[i, j]*u_ap[int(vec[i, j-1])]                  # utemp is computed.
            t = (f(p[i, 0], p[i, 1]) - utemp)/Gamma[i,0]                            # The central node is added to the approximation.
            err = max(err, abs(t - u_ap[i]));                                       # Error computation.
            u_ap[i] = t;                                                            # The previously computed value is assigned.
    
    # Theoretical Solution
    for i in range(m):                                                              # For all the nodes.
        u_ex[i] = phi(p[i,0], p[i,1])                                             # The theoretical solution is computed.

    return u_ap, u_ex, vec