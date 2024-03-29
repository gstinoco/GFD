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

def Triangulation(p, tt, f, g, t, c, cho, r):
    # Wave Equation 2D implemented on Triangulations
    # 
    # This routine calculates an approximation to the solution of wave equation in 2D using a Generalized Finite Differences scheme on triangulations.
    # 
    # The problem to solve is:
    # 
    # \frac{\partial^2 u}{\partial t^2} = c^2\nabla^2 u$
    # 
    # Input parameters
    #   p           m x 2           Array           Array with the coordinates of the nodes.
    #   tt          n x 3           Array           Array with the correspondence of the n triangles.
    #   fWAV                        Function        Function declared with the boundary condition.
    #   gWAV                        Function        Function declared with the boundary condition.
    #   t                           Integer         Number of time steps to be considered.
    #   c                           Real            Wave propagation velocity,
    #   cho                         Integer         Approximation Type.
    #                                               (0 for cero boundary condition)
    #                                               (1 for function boundary condition)
    #
    # Output parameters
    #   u_ap        m x n x t       Array           Array with the approximation computed by the routine.
    #   u_ex        m x n x t       Array           Array with the theoretical solution.
    #   vec         m x o           Array           Array with the correspondence of the o neighbors of each node.

    # Variable initialization
    m    = len(p[:,0])                                                              # The total number of nodes is calculated.
    T    = np.linspace(0,1,t)                                                       # Time discretization.
    dt   = T[1] - T[0]                                                              # dt computation.
    u_ap = np.zeros([m,t])                                                          # u_ap initialization with zeros.
    u_ex = np.zeros([m,t])                                                          # u_ex initialization with zeros.
    cdt  = (c**2)*(dt**2)                                                           # cdt is equals to c^2 dt^2

    # Boundary conditions
    if cho == 1:                                                                    # Approximation Type selection.
        for k in np.arange(t):                                                      # For each time step.
            for i in np.arange(m):                                                  # For each of the nodes.
                if p[i,2] == 1:                                                     # If the node is in the boundary.
                    u_ap[i, k] = f(p[i, 0], p[i, 1], T[k], c, cho, r)               # The boundary condition is assigned.
  
    # Initial condition
    for i in np.arange(m):                                                          # For each of the nodes.
        u_ap[i, 0] = f(p[i, 0], p[i, 1], T[0], c, cho, r)                           # The initial condition is assigned.
    
    # Neighbor search for all the nodes.
    vec = Neighbors.Triangulation(p, tt, 9)                                         # Neighbor search with the proper routine.

     # Computation of Gamma values
    L = np.vstack([[0], [0], [2], [0], [2]])                                        # The values of the differential operator are assigned.
    Gamma = Gammas.Cloud(p, vec, L)                                                 # Gamma computation.

    # A Generalized Finite Differences Method
    ## Second time step computation.
    for k in np.arange(1,2):                                                        # For the second time step.
        for i in np.arange(m):                                                      # For all the nodes.
            if p[i,2] == 0:                                                         # If the node is an inner node.
                utemp = 0                                                           # utemp is initialized with 0.
                nvec = sum(vec[i,:] != -1)                                          # Number of neighbor nodes of the central node.
                for j in np.arange(1,nvec+1):                                       # For each of the neighbor nodes.
                    utemp = utemp + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1]), k-1]      # utemp computation with the neighbors.
                utemp = utemp + cdt*Gamma[i,0]*u_ap[i, k-1]                         # The central node is added to the approximation.
                u_ap[i,k] = u_ap[i, k-1] + (1/2)*utemp + \
                            dt*g(p[i, 0], p[i, 1], T[k], c, cho, r)                 # The second time step is completed.
    
    ## Other time steps computation.
    for k in np.arange(2,t):                                                        # For all the other time steps.
        for i in np.arange(m):                                                      # For all the nodes.
            if p[i,2] == 0:                                                         # If the node is an inner node.
                utemp = 0                                                           # utemp is initialized with 0.
                nvec = sum(vec[i,:] != -1)                                          # Number of neighbor nodes of the central node.
                for j in np.arange(1,nvec+1):                                       # For each of the neighbor nodes.
                    utemp = utemp + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1]), k-1]      # utemp computation with the neighbors.
                utemp = utemp + cdt*Gamma[i,0]*u_ap[i, k-1]                         # The central node is added to the approximation.
                u_ap[i,k] = 2*u_ap[i, k-1] - u_ap[i, k-2] + utemp                   # The time step is completed.

    # Theoretical Solution
    for k in np.arange(t):                                                          # For all the time steps.
        for i in np.arange(m):                                                      # For each of the nodes.
            u_ex[i,k] = f(p[i,0], p[i,1], T[k], c, cho, r)                          # The theoretical solution is computed.

    return u_ap, u_ex, vec

def Cloud(p, f, g, t, c, cho, r):
    # Wave Equation 2D implemented on Unstructured Clouds of Points
    # 
    # This routine calculates an approximation to the solution of wave equation in 2D using a Generalized Finite Differences scheme on unstructured clouds of points.
    # 
    # The problem to solve is:
    # 
    # \frac{\partial^2 u}{\partial t^2} = c^2\nabla^2 u$
    # 
    # Input parameters
    #   p           m x 2           Array           Array with the coordinates of the nodes.
    #   f                           Function        Function declared with the boundary condition.
    #   g                           Function        Function declared with the boundary condition.
    #   t                           Integer         Number of time steps to be considered.
    #   c                           Real            Wave propagation velocity.
    #   cho                         Integer         Approximation Type.
    #                                               (0 for cero boundary condition)
    #                                               (1 for function boundary condition)
    #
    # Output parameters
    #   u_ap        m x n x t       Array           Array with the approximation computed by the routine.
    #   u_ex        m x n x t       Array           Array with the theoretical solution.
    #   vec         m x o           Array           Array with the correspondence of the o neighbors of each node.

    # Variable initialization
    m    = len(p[:,0])                                                              # The total number of nodes is calculated.
    T    = np.linspace(0,1,t)                                                       # Time discretization.
    dt   = T[1] - T[0]                                                              # dt computation.
    u_ap = np.zeros([m,t])                                                          # u_ap initialization with zeros.
    u_ex = np.zeros([m,t])                                                          # u_ex initialization with zeros.
    cdt  = (c**2)*(dt**2)                                                           # cdt is equals to c^2 dt^2

    # Boundary conditions
    if cho == 1:                                                                    # Approximation Type selection.
        for k in np.arange(t):                                                      # For each time step.
            for i in np.arange(m):                                                  # For each of the nodes.
                if p[i,2] == 1:                                                     # If the node is in the boundary.
                    u_ap[i, k] = f(p[i, 0], p[i, 1], T[k], c, cho, r)               # The boundary condition is assigned.
  
    # Initial condition
    for i in np.arange(m):                                                          # For each of the nodes.
        u_ap[i, 0] = f(p[i, 0], p[i, 1], T[0], c, cho, r)                           # The initial condition is assigned.
    
    # Neighbor search for all the nodes.
    vec = Neighbors.Cloud(p, 9)                                                     # Neighbor search with the proper routine.

     # Computation of Gamma values
    L = np.vstack([[0], [0], [2], [0], [2]])                                        # The values of the differential operator are assigned.
    Gamma = Gammas.Cloud(p, vec, L)                                                 # Gamma computation.

    # A Generalized Finite Differences Method
    ## Second time step computation.
    for k in np.arange(1,2):                                                        # For the second time step.
        for i in np.arange(m):                                                      # For all the nodes.
            if p[i,2] == 0:                                                         # If the node is an inner node.
                utemp = 0                                                           # utemp is initialized with 0.
                nvec = sum(vec[i,:] != -1)                                          # Number of neighbor nodes of the central node.
                for j in np.arange(1,nvec+1):                                       # For each of the neighbor nodes.
                    utemp = utemp + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1]), k-1]      # utemp computation with the neighbors.
                utemp = utemp + cdt*Gamma[i,0]*u_ap[i, k-1]                         # The central node is added to the approximation.
                u_ap[i,k] = u_ap[i, k-1] + (1/2)*utemp + \
                            dt*g(p[i, 0], p[i, 1], T[k], c, cho, r)                 # The second time step is completed.
    
    ## Other time steps computation.
    for k in np.arange(2,t):                                                        # For all the other time steps.
        for i in np.arange(m):                                                      # For all the nodes.
            if p[i,2] == 0:                                                         # If the node is an inner node.
                utemp = 0                                                           # utemp is initialized with 0.
                nvec = sum(vec[i,:] != -1)                                          # Number of neighbor nodes of the central node.
                for j in np.arange(1,nvec+1):                                       # For each of the neighbor nodes.
                    utemp = utemp + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1]), k-1]      # utemp computation with the neighbors.
                utemp = utemp + cdt*Gamma[i,0]*u_ap[i, k-1]                         # The central node is added to the approximation.
                u_ap[i,k] = 2*u_ap[i, k-1] - u_ap[i, k-2] + utemp                   # The time step is completed.

    # Theoretical Solution
    for k in np.arange(t):                                                          # For all the time steps.
        for i in np.arange(m):                                                      # For each of the nodes.
            u_ex[i,k] = f(p[i,0], p[i,1], T[k], c, cho, r)                          # The theoretical solution is computed.

    return u_ap, u_ex, vec