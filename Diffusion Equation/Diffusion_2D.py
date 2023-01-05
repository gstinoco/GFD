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
    # \frac{\partial u}{\partial t}= \nu\nabla^2 u
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

def Mesh_K(x, y, f, nu, t):
    # 2D Diffusion implemented in Logically Rectangular Meshes.
    # 
    # This routine calculates an approximation to the solution of Diffusion equation in 2D using an Explicit Generalized Finite Differences scheme in logically rectangular meshes.
    # For this routine, a matrix formulation is used compute the approximation.
    # 
    # The problem to solve is:
    # 
    # \frac{\partial u}{\partial t}= \nu\nabla^2 u
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
    urr  = np.zeros([m*n,1])                                                        # u_rr initialization with zeros.

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

    # Computation of K with Gammas
    L  = np.vstack([[0], [0], [2*nu*dt], [0], [2*nu*dt]])                           # The values of the differential operator are assigned.
    K  = Gammas.K(x, y, L)                                                          # K computation that include the Gammas.
    Kp = np.identity(m*n) + K                                                       # Kp with an explicit formulation.

    # A Generalized Finite Differences Method
    for k in np.arange(1,t):                                                        # For each time step.
        R = Gammas.R(u_ap, m, n, k)                                                 # R Matrix is computed.
        for i in np.arange(m):                                                      # For each of the nodes on x.
            for j in np.arange(n):                                                  # For each of the nodes on y.
                urr[i + j*m, 0] = u_ap[i, j, k-1]                                   # urr values' assignation.
        
        un = (Kp@urr) + R                                                           # un is Kp*urr + R. 

        for i in np.arange(1,m-1):                                                  # For each of the interior nodes on x.
            for j in np.arange(1,n-1):                                              # For each of the interior nodes on y.
                u_ap[i, j, k] = un[i + (j)*m]                                       # u_ap values' are assigned.

    # Theoretical Solution
    for k in np.arange(t):                                                          # For all the time steps.
        for i in np.arange(m):                                                      # For all the nodes on x.
            for j in np.arange(n):                                                  # For all the nodes on y.
                u_ex[i, j, k] = f(x[i, j], y[i, j], T[k], nu)                       # The theoretical solution is computed.

    return u_ap, u_ex

def Triangulation(p, pb, tt, f, nu, t):
    # 2D Diffusion Equation implemented in Triangulations.
    # 
    # This routine calculates an approximation to the solution of Diffusion equation in 2D using a Generalized Finite Differences scheme in triangulations.
    # 
    # The problem to solve is:
    # 
    # \frac{\partial u}{\partial t}= \nu\nabla^2 u
    # 
    # Input parameters
    #   p           m x 2           Array           Array with the coordinates of the nodes.
    #   pb          b x 2           Array           Array with the coordinates of the boundary nodes.
    #   tt          n x 3           Array           Array with the correspondence of the n triangles.
    #   f                           Function        Function declared with the boundary condition.
    #   nu                          Real            Diffusion coefficient.
    #   t                           Integer         Number of time steps considered.
    # 
    # Output parameters
    #   u_ap        m x 1           Array           Array with the approximation computed by the routine.
    #   u_ex        m x 1           Array           Array with the theoretical solution.
    #   vec         m x o           Array           Array with the correspondence of the o neighbors of each node.

    # Variable initialization
    m    = len(p[:,0])                                                              # The total number of nodes is calculated.
    mf   = len(pb[:,0])                                                             # The number of boundary nodes is calculated.
    T    = np.linspace(0,1,t)                                                       # Time discretization.
    dt   = T[1] - T[0]                                                              # dt computation.
    u_ap = np.zeros([m,t])                                                          # u_ap initialization with zeros.
    u_ex = np.zeros([m,t])                                                          # u_ex initialization with zeros.

    # Boundary conditions
    for k in np.arange(t):
        for i in np.arange(mf):                                                     # For each of the boundary nodes.
            u_ap[i, k] = f(pb[i, 0], pb[i, 1], T[k], nu)                            # The boundary condition is assigned.
  
    # Initial condition
    for i in np.arange(m):                                                          # For each of the nodes.
        u_ap[i, 0] = f(p[i, 0], p[i, 1], T[0], nu)                                  # The initial condition is assigned.
    
    # Neighbor search for all the nodes.
    vec = Neighbors.Triangulation(p, tt, 9)                                         # Neighbor search with the proper routine.

    # Computation of Gamma values
    L = np.vstack([[0], [0], [2*nu*dt], [0], [2*nu*dt]])                            # The values of the differential operator are assigned.
    Gamma = Gammas.Cloud(p, pb, vec, L)                                             # Gamma computation.

    # A Generalized Finite Differences Method
    for k in np.arange(1,t):                                                        # For each of the time steps.
        for i in np.arange(mf, m):                                                  # For each of the interior nodes.
            utemp = 0                                                               # utemp initialization in 0.
            nvec = sum(vec[i,:] != -1)                                              # Number of neighbor nodes of the central node.
            for j in np.arange(1,nvec+1):                                           # For each of the neighbor nodes.
                utemp = utemp + Gamma[i,j]*u_ap[int(vec[i, j-1]), k-1]              # utemp computation with the neighbors.
            utemp = utemp + Gamma[i,0]*u_ap[i, k-1]                                 # The central node is added to the approximation.
            u_ap[i,k] = u_ap[i, k-1] + utemp                                        # u_ap value is assigned.

    # Theoretical Solution
    for k in np.arange(t):                                                          # For all the time steps.
        for i in np.arange(m):                                                      # For each of the nodes.
            u_ex[i,k] = f(p[i,0], p[i,1], T[k], nu)                                 # The theoretical solution is computed.

    return u_ap, u_ex, vec

def Cloud(p, pb, f, nu, t):
    # 2D Diffusion Equation implemented in Unstructured Clouds of Points.
    # 
    # This routine calculates an approximation to the solution of Diffusion equation in 2D using a Generalized Finite Differences scheme on unstructured clouds of points.
    # 
    # The problem to solve is:
    # 
    # \frac{\partial u}{\partial t}= \nu\nabla^2 u
    # 
    # Input parameters
    #   p           m x 2           Array           Array with the coordinates of the nodes.
    #   pb          b x 2           Array           Array with the coordinates of the boundary nodes.
    #   f                           Function        Function declared with the boundary condition.
    #   nu                          Real            Diffusion coefficient.
    #   t                           Integer         Number of time steps to be considered.
    # 
    # Output parameters
    #   u_ap        m x 1           Array           Array with the approximation computed by the routine.
    #   u_ex        m x 1           Array           Array with the theoretical solution.
    #   vec         m x o           Array           Array with the correspondence of the o neighbors of each node.

    # Variable initialization
    m    = len(p[:,0])                                                              # The total number of nodes is calculated.
    mf   = len(pb[:,0])                                                             # The number of boundary nodes is calculated.
    T    = np.linspace(0,1,t)                                                       # Time discretization.
    dt   = T[1] - T[0]                                                              # dt computation.
    u_ap = np.zeros([m,t])                                                          # u_ap initialization with zeros.
    u_ex = np.zeros([m,t])                                                          # u_ex initialization with zeros.

    # Boundary conditions
    for k in np.arange(t):
        for i in np.arange(mf):                                                     # For each of the boundary nodes.
            u_ap[i, k] = f(pb[i, 0], pb[i, 1], T[k], nu)                            # The boundary condition is assigned.
  
    # Initial condition
    for i in np.arange(m):                                                          # For each of the nodes.
        u_ap[i, 0] = f(p[i, 0], p[i, 1], T[0], nu)                                  # The initial condition is assigned.
    
    # Neighbor search for all the nodes.
    vec = Neighbors.Cloud(p, pb, 9)                                                 # Neighbor search with the proper routine.

    # Computation of Gamma values
    L = np.vstack([[0], [0], [2*nu*dt], [0], [2*nu*dt]])                            # The values of the differential operator are assigned.
    Gamma = Gammas.Cloud(p, pb, vec, L)                                             # Gamma computation.

    # A Generalized Finite Differences Method
    for k in np.arange(1,t):                                                        # For each of the time steps.
        for i in np.arange(mf, m):                                                  # For each of the interior nodes.
            utemp = 0                                                               # utemp initialization in 0.
            nvec = sum(vec[i,:] != -1)                                              # Number of neighbor nodes of the central node.
            for j in np.arange(1,nvec+1):                                           # For each of the neighbor nodes.
                utemp = utemp + Gamma[i,j]*u_ap[int(vec[i, j-1]), k-1]              # utemp computation with the neighbors.
            utemp = utemp + Gamma[i,0]*u_ap[i, k-1]                                 # The central node is added to the approximation.
            u_ap[i,k] = u_ap[i, k-1] + utemp                                        # u_ap value is assigned.

    # Theoretical Solution
    for k in np.arange(t):                                                          # For all the time steps.
        for i in np.arange(m):                                                      # For each of the nodes.
            u_ex[i,k] = f(p[i,0], p[i,1], T[k], nu)                                 # The theoretical solution is computed.

    return u_ap, u_ex, vec