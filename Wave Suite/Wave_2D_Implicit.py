# All the codes presented below were developed by:
#   Dr. Gerardo Tinoco Guerrero
#   Universidad Michoacana de San Nicolás de Hidalgo
#   gerardo.tinoco@umich.mx
#
# With the financing of:
#   National Council of Science and Technology, CONACyT (Consejo Nacional de Ciencia y Tecnología, CONACyT). México.
#   Coordination of Scientific Research, CIC-UMSNH (Coordinación de la Investigación Científica de la Universidad Michoacana de San Nicolás de Hidalgo, CIC-UMSNH). México
#   Aula CIMNE Morelia. México
#
# Date:
#   November, 2022.
#
# Last Modification:
#   November, 2022.

import numpy as np
from sys import path
path.insert(0, 'General/')
import Gammas
import Neighbors

def Wave_Tri(p, pb, tt, f, g, t, c, lam):
    # Wave Equation 2D implemented on Triangulations
    # 
    # This routine calculates an approximation to the solution of wave equation in 2D using an Implicit Generalized Finite Differences scheme on triangulations.
    # 
    # The problem to solve is:
    # 
    # \frac{\partial^2 u}{\partial t^2} = c^2\nabla^2 u$
    # 
    # Input parameters
    #   p           m x 2           Array           Array with the coordinates of the nodes.
    #   pb          b x 2           Array           Array with the coordinates of the boundary nodes.
    #   tt          n x 3           Array           Array with the correspondence of the n triangles.
    #   fWAV                        Function        Function declared with the boundary condition.
    #   gWAV                        Function        Function declared with the boundary condition.
    #   t                           Integer         Number of time steps to be considered.
    #   c                           Real            Wave propagation velocity,
    #   lam                         Real            lambda parameter to be considered by the implicit scheme.
    #
    # Output parameters
    #   u_ap        m x n x t       Array           Array with the approximation computed by the routine.
    #   u_ex        m x n x t       Array           Array with the theoretical solution.

    # Variable initizalization
    m    = len(p[:,0])                                                              # The total number of nodes is calculated.
    mf   = len(pb[:,0])                                                             # The number of boundary nodes is calculated.
    T    = np.linspace(0,3,t)                                                       # Time discretization.
    dt   = T[1] - T[0]                                                              # dt computation.
    u_ap = np.zeros([m,t])                                                          # u_ap initialization with zeros.
    u_ex = np.zeros([m,t])                                                          # u_ex initialization with zeros.
    tol  = np.finfo(float).eps                                                      # The tolerance for the predictor-Corrector scheme.
    cdt  = (c**2)*(dt**2)                                                           # cdt is equals to c^2 dt^2

    # Boundary conditions
    for k in np.arange(t):
        for i in np.arange(mf):                                                     # For each of the boundary nodes.
            u_ap[i, k] = f(pb[i, 0], pb[i, 1], T[k], c)                             # The boundary condition is assigned.
  
    # Initial condition
    for i in np.arange(m):                                                          # For each of the nodes.
        u_ap[i, 0] = f(p[i, 0], p[i, 1], T[0], c)                                   # The initial condition is assigned.
    
    # Neighbor search for all the nodes.
    vec = Neighbors.Neighbors_Tri(p, tt, 9)                                         # Neighbor search with the proper routine.

     # Computation of Gamma values
    L = np.vstack([[0], [0], [2], [0], [2]])                                        # The values of the differential operator are assigned.
    Gamma = Gammas.Cloud(p, pb, vec, L)                                             # Gamma computation.

    # A Generalized Finite Differences Method

    ## Second time step computation.
    for k in np.arange(1,2):                                                        # For the second time step.
        er = 1                                                                      # er is initializaded in 1.
        # Prediction
        for i in np.arange(mf, m):                                                  # For all the interior nodes.
            utemp = 0                                                               # utemp is initialized with 0.
            nvec = sum(vec[i,:] != 0)                                               # Number of neighbor nodes of the central node.
            for j in np.arange(1,nvec+1):                                           # For each of the neighbor nodes.
                utemp = utemp + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1])-1, k-1]        # utemp computation with the neighbors.

            utemp = utemp + cdt*Gamma[i,0]*u_ap[i, k-1]                             # The central node is added to the approximation.
            u_ap[i,k] = u_ap[i, k-1] + (1/2)*utemp + \
                        dt*g(p[i, 0], p[i, 1], T[k], c)                             # The Prediction is completed.
        
        # Correction
        while er >= tol:                                                            # While the error is greater than the tolerance.
            Z = u_ap[:,k]                                                           # Temporal Z saves the value of u_ap
            for i in np.arange(mf,m):                                               # For all the interior nodes.
                utemp1 = 0                                                          # utemp1 is initialized with 0. For the approximation in k-1.
                utemp2 = 0                                                          # utemp2 is initialized with 0. For the approximation in k.
                nvec = sum(vec[i,:] != 0)                                           # Number of neighbor nodes of the central node.
                for j in np.arange(1,nvec+1):                                       # For each of the neighbor nodes.
                    utemp1 = utemp1 + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1])-1, k-1]  # utemp1 computation with the neighbors.
                    utemp2 = utemp2 + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1])-1, k]    # utemp2 computation with the neighbors.
                
                utemp1 = utemp1 + cdt*Gamma[i,0]*u_ap[i, k-1]                       # The central node is added to the approximation in k.
                u_ap[i,k] = (u_ap[i,k-1] + dt*g(p[i, 0], p[i, 1], T[k], c) + \
                            (1/2)*(lam*utemp1 + (1-lam)*utemp2))/ \
                            (1 - (1/2)*(1-lam)*Gamma[i,0]*cdt)                      # The correction is done.
            
            er = np.max(abs(u_ap[:,k] - Z))                                         # Error is computed.
    
    ## Other time steps computation.
    for k in np.arange(2,t):                                                        # For all the other time steps.
        er = 1                                                                      # er is initializaded in 1.
         # Prediction
        for i in np.arange(mf, m):                                                  # For all the interior nodes.
            utemp = 0                                                               # utemp is initialized with 0.
            nvec = sum(vec[i,:] != 0)                                               # Number of neighbor nodes of the central node.
            for j in np.arange(1,nvec+1):                                           # For each of the neighbor nodes.
                utemp = utemp + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1])-1, k-1]        # utemp computation with the neighbors.

            utemp = utemp + cdt*Gamma[i,0]*u_ap[i, k-1]                             # The central node is added to the approximation.
            u_ap[i,k] = 2*u_ap[i, k-1] - u_ap[i, k-2] + utemp                       # The Prediction is completed.
        
        # Correction
        while er >= tol:                                                            # While the error is greater than the tolerance.
            Z = u_ap[:,k]                                                           # Temporal Z saves the value of u_ap
            for i in np.arange(mf,m):                                               # For all the interior nodes.
                utemp1 = 0                                                          # utemp1 is initialized with 0. For the approximation in k-1.
                utemp2 = 0                                                          # utemp2 is initialized with 0. For the approximation in k.
                nvec = sum(vec[i,:] != 0)                                           # Number of neighbor nodes of the central node.
                for j in np.arange(1,nvec+1):                                       # For each of the neighbor nodes.
                    utemp1 = utemp1 + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1])-1, k-1]  # utemp1 computation with the neighbors.
                    utemp2 = utemp2 + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1])-1, k]    # utemp1 computation with the neighbors.
                
                utemp1 = utemp1 + cdt*Gamma[i,0]*u_ap[i, k-1]                       # The central node is added to the approximation in k.
                u_ap[i,k] = (2*u_ap[i, k-1] - u_ap[i, k-2] + \
                            lam*utemp1 + (1-lam)*utemp2)/ \
                            (1 - (1-lam)*Gamma[i,0]*cdt)                            # The correction is done.

            er = np.max(abs(u_ap[:,k] - Z))                                         # Error is computed.

    # Theoretical Solution
    for k in np.arange(t):                                                          # For all the time steps.
        for i in np.arange(m):                                                      # For each of the nodes.
            u_ex[i,k] = f(p[i,0], p[i,1], T[k], c)                                  # The theoretical solution is computed.

    return u_ap, u_ex

def Wave_Cloud(p, pb, vec, f, g, t, c, lam):
    # Wave Equation 2D implemented on Unstructured Clouds of Points
    # 
    # This routine calculates an approximation to the solution of wave equation in 2D using an Implicit Generalized Finite Differences scheme on unstructured clouds of points.
    # 
    # The problem to solve is:
    # 
    # \frac{\partial^2 u}{\partial t^2} = c^2\nabla^2 u$
    # 
    # Input parameters
    #   p           m x 2           Array           Array with the coordinates of the nodes.
    #   pb          b x 2           Array           Array with the coordinates of the boundary nodes.
    #   vec         m x o           Array           Array with the correspondence of the o neighbors of each node.
    #   fWAV                        Function        Function declared with the boundary condition.
    #   gWAV                        Function        Function declared with the boundary condition.
    #   t                           Integer         Number of time steps to be considered.
    #   c                           Real            Wave propagation velocity,
    #   lam                         Real            lambda parameter to be considered by the implicit scheme.
    #
    # Output parameters
    #   u_ap        m x n x t       Array           Array with the approximation computed by the routine.
    #   u_ex        m x n x t       Array           Array with the theoretical solution.

    # Variable initizalization
    m    = len(p[:,0])                                                              # The total number of nodes is calculated.
    mf   = len(pb[:,0])                                                             # The number of boundary nodes is calculated.
    T    = np.linspace(0,3,t)                                                       # Time discretization.
    dt   = T[1] - T[0]                                                              # dt computation.
    u_ap = np.zeros([m,t])                                                          # u_ap initialization with zeros.
    u_ex = np.zeros([m,t])                                                          # u_ex initialization with zeros.
    tol  = np.finfo(float).eps                                                      # The tolerance for the predictor-Corrector scheme.
    cdt  = (c**2)*(dt**2)                                                           # cdt is equals to c^2 dt^2

    # Boundary conditions
    for k in np.arange(t):
        for i in np.arange(mf):                                                     # For each of the boundary nodes.
            u_ap[i, k] = f(pb[i, 0], pb[i, 1], T[k], c)                             # The boundary condition is assigned.
  
    # Initial condition
    for i in np.arange(m):                                                          # For each of the nodes.
        u_ap[i, 0] = f(p[i, 0], p[i, 1], T[0], c)                                   # The initial condition is assigned.

     # Computation of Gamma values
    L = np.vstack([[0], [0], [2], [0], [2]])                                        # The values of the differential operator are assigned.
    Gamma = Gammas.Cloud(p, pb, vec, L)                                             # Gamma computation.

    # A Generalized Finite Differences Method

    ## Second time step computation.
    for k in np.arange(1,2):                                                        # For the second time step.
        er = 1                                                                      # er is initializaded in 1.
        # Prediction
        for i in np.arange(mf, m):                                                  # For all the interior nodes.
            utemp = 0                                                               # utemp is initialized with 0.
            nvec = sum(vec[i,:] != 0)                                               # Number of neighbor nodes of the central node.
            for j in np.arange(1,nvec+1):                                           # For each of the neighbor nodes.
                utemp = utemp + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1])-1, k-1]        # utemp computation with the neighbors.

            utemp = utemp + cdt*Gamma[i,0]*u_ap[i, k-1]                             # The central node is added to the approximation.
            u_ap[i,k] = u_ap[i, k-1] + (1/2)*utemp + \
                        dt*g(p[i, 0], p[i, 1], T[k], c)                             # The Prediction is completed.
        
        # Correction
        while er >= tol:                                                            # While the error is greater than the tolerance.
            Z = u_ap[:,k]                                                           # Temporal Z saves the value of u_ap
            for i in np.arange(mf,m):                                               # For all the interior nodes.
                utemp1 = 0                                                          # utemp1 is initialized with 0. For the approximation in k-1.
                utemp2 = 0                                                          # utemp2 is initialized with 0. For the approximation in k.
                nvec = sum(vec[i,:] != 0)                                           # Number of neighbor nodes of the central node.
                for j in np.arange(1,nvec+1):                                       # For each of the neighbor nodes.
                    utemp1 = utemp1 + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1])-1, k-1]  # utemp1 computation with the neighbors.
                    utemp2 = utemp2 + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1])-1, k]    # utemp2 computation with the neighbors.
                
                utemp1 = utemp1 + cdt*Gamma[i,0]*u_ap[i, k-1]                       # The central node is added to the approximation in k.
                u_ap[i,k] = (u_ap[i,k-1] + dt*g(p[i, 0], p[i, 1], T[k], c) + \
                            (1/2)*(lam*utemp1 + (1-lam)*utemp2))/ \
                            (1 - (1/2)*(1-lam)*Gamma[i,0]*cdt)                      # The correction is done.
            
            er = np.max(abs(u_ap[:,k] - Z))                                         # Error is computed.
    
    ## Other time steps computation.
    for k in np.arange(2,t):                                                        # For all the other time steps.
        er = 1                                                                      # er is initializaded in 1.
         # Prediction
        for i in np.arange(mf, m):                                                  # For all the interior nodes.
            utemp = 0                                                               # utemp is initialized with 0.
            nvec = sum(vec[i,:] != 0)                                               # Number of neighbor nodes of the central node.
            for j in np.arange(1,nvec+1):                                           # For each of the neighbor nodes.
                utemp = utemp + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1])-1, k-1]        # utemp computation with the neighbors.

            utemp = utemp + cdt*Gamma[i,0]*u_ap[i, k-1]                             # The central node is added to the approximation.
            u_ap[i,k] = 2*u_ap[i, k-1] - u_ap[i, k-2] + utemp                       # The Prediction is completed.
        
        # Correction
        while er >= tol:                                                            # While the error is greater than the tolerance.
            Z = u_ap[:,k]                                                           # Temporal Z saves the value of u_ap
            for i in np.arange(mf,m):                                               # For all the interior nodes.
                utemp1 = 0                                                          # utemp1 is initialized with 0. For the approximation in k-1.
                utemp2 = 0                                                          # utemp2 is initialized with 0. For the approximation in k.
                nvec = sum(vec[i,:] != 0)                                           # Number of neighbor nodes of the central node.
                for j in np.arange(1,nvec+1):                                       # For each of the neighbor nodes.
                    utemp1 = utemp1 + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1])-1, k-1]  # utemp1 computation with the neighbors.
                    utemp2 = utemp2 + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1])-1, k]    # utemp1 computation with the neighbors.
                
                utemp1 = utemp1 + cdt*Gamma[i,0]*u_ap[i, k-1]                       # The central node is added to the approximation in k.
                u_ap[i,k] = (2*u_ap[i, k-1] - u_ap[i, k-2] + \
                            lam*utemp1 + (1-lam)*utemp2)/ \
                            (1 - (1-lam)*Gamma[i,0]*cdt)                            # The correction is done.

            er = np.max(abs(u_ap[:,k] - Z))                                         # Error is computed.

    # Theoretical Solution
    for k in np.arange(t):                                                          # For all the time steps.
        for i in np.arange(m):                                                      # For each of the nodes.
            u_ex[i,k] = f(p[i,0], p[i,1], T[k], c)                                  # The theoretical solution is computed.

    return u_ap, u_ex