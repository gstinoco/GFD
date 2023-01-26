import numpy as np

def Neighbors(p, nvec):
    # Clouds
    # Routine to find the neighbor nodes in a cloud of points generated with dmsh on Python.
    #
    # Input parameters
    #   p           m x 2           double          Array with the coordinates of the nodes.
    #   pm          m x 2           double          Array with the coordinates of the boundary nodes.
    #   nvec                        integer         Maximum number of neighbors.
    # 
    # Output parameters
    #   vec         m x nvec        double          Array with matching neighbors of each node.

    # Variable initialization
    m    = len(p[:,0])                                                              # The size if the triangulation is obtained.
    vec  = np.zeros([m, nvec],dtype=int)-1                                                    # The array for the neighbors is initialized.
    dist = 0                                                                        # Maximum distance between the boundary nodes.

    # Search for the maximum distance between the boundary nodes
    xb = []
    yb = []
    for i in np.arange(m):
        if p[i,2] == 1:
            xb.append(p[i,0])
            yb.append(p[i,1])
    xb = np.array(xb)
    yb = np.array(yb)
    m2 = len(xb)                                                                    # Total number of boundary nodes.
    for i in np.arange(m2-1):
        d    = np.sqrt((xb[i] - xb[i+1])**2 + (yb[i] - yb[i+1])**2)                 # The distance between a boundary node and the next one.
        dist = max(dist,d)                                                          # Maximum distance search.
    d    = np.sqrt((xb[m2-1] - xb[0])**2 + (yb[m2-1] - yb[0])**2)                   # The distance between the last and the first boundary nodes.
    dist = (8/7)*max(dist,d)                                                        # Maximum distance search.

    # Search of the neighbor nodes
    for i in np.arange(m):                                                          # For each of the nodes.
        x    = p[i,0]                                                               # x coordinate of the central node.
        y    = p[i,1]                                                               # y coordinate of the central node.
        temp = 0                                                                    # Temporal variable as a counter.
        for j in np.arange(m):                                                      # For all the interior nodes.
            if i != j:                                                              # Check that we are not working with the central node.
                x1 = p[j,0]                                                         # x coordinate of the possible neighbor.
                y1 = p[j,1]                                                         # y coordinate of the possible neighbor.
                d  = np.sqrt((x - x1)**2 + (y - y1)**2)                           # Distance from the possible neighbor to the central node.
                if d < dist:                                                        # If the distance is smaller or equal to the tolerance distance.
                    if temp < nvec:                                                # If the number of neighbors is smaller than nvec.
                        vec[i,temp] = j                                             # Save the neighbor.
                        temp       += 1                                             # Increase the counter by 1.
                    else:                                                           # If the number of neighbors is greater than nvec.
                        x2 = p[vec[i,:],0]                                          # x coordinates of the current neighbor nodes.
                        y2 = p[vec[i,:],1]                                          # y coordinates of the current neighbor nodes.
                        d2 = np.sqrt((x - x2)**2 + (y - y2)**2)                   # The total distance from all the neighbors to the central node.
                        I  = np.argmax(d2)                                          # Look for the greatest distance.
                        if d < d2[I]:                                               # If the new node is closer than the farthest neighbor.
                            vec[i,I] = j                                            # The new neighbor replace the farthest one.
    return vec

def Gammas(p, vec, L):
    # Unstructured Clouds of Points and Triangulations Gammas Computation.
    # 
    # This routine computes the Gamma values for unstructured clouds of points and triangulations.
    # 
    # Input parameters
    #   p           m x 2           Array           Array with the coordinates of the nodes.
    #   pb          b x 2           Array           Array with the coordinates of the boundary nodes.
    #   vec         m x o           Array           Array with the correspondence of the o neighbors of each node.
    #   L           5 x 1           Array           Array with the values of the differential operator.
    # 
    # Output parameters
    #   Gamma       m x n x 9       Array           Array with the computed gamma values.

    nvec  = len(vec[:,1])                                                           # The maximum number of neighbors.
    m     = len(p[:,0])                                                             # The total number of nodes.
    Gamma = np.zeros([m, nvec])                                                     # Gamma initialization with zeros.

    for i in np.arange(m):                                                          # For each of the boundary nodes.
        if p[i,2] == 0:                                                                 # If the node is in the boundary.
            nvec = sum(vec[i,:] != -1)                                                  # The total number of neighbors of the node.
            dx = np.zeros([nvec])                                                       # dx initialization with zeros.
            dy = np.zeros([nvec])                                                       # dy initialization with zeros.

            for j in np.arange(nvec):                                                   # For each of the neighbor nodes.
                vec1 = int(vec[i, j])                                                   # The neighbor index is found.
                dx[j] = p[vec1, 0] - p[i,0]                                             # dx is computed.
                dy[j] = p[vec1, 1] - p[i,1]                                             # dy is computed.

            M = np.vstack([[dx], [dy], [dx**2], [dx*dy], [dy**2]])                      # M matrix is assembled.
            M = np.linalg.pinv(M)                                                       # The pseudoinverse of matrix M.
            YY = M@L                                                                    # M*L computation.
            Gem = np.vstack([-sum(YY), YY]).transpose()                                 # Gamma values are found.
            for j in np.arange(nvec+1):                                                 # For each of the Gamma values.
                Gamma[i,j] = Gem[0,j]                                                   # The Gamma value is stored.
    
    return Gamma

def Cloud(p, f, g, t, c):
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
    #   pb          b x 2           Array           Array with the coordinates of the boundary nodes.
    #   fWAV                        Function        Function declared with the boundary condition.
    #   gWAV                        Function        Function declared with the boundary condition.
    #   t                           Integer         Number of time steps to be considered.
    #   c                           Real            Wave propagation velocity,
    #
    # Output parameters
    #   u_ap        m x n x t       Array           Array with the approximation computed by the routine.
    #   u_ex        m x n x t       Array           Array with the theoretical solution.
    #   vec         m x o           Array           Array with the correspondence of the o neighbors of each node.

    # Variable initialization
    m    = len(p[:,0])                                                              # The total number of nodes is calculated.
    T    = np.linspace(0,3,t)                                                       # Time discretization.
    dt   = T[1] - T[0]                                                              # dt computation.
    u_ap = np.zeros([m,t])                                                          # u_ap initialization with zeros.
    u_ex = np.zeros([m,t])                                                          # u_ex initialization with zeros.
    cdt  = (c**2)*(dt**2)                                                           # cdt is equals to c^2 dt^2

    # Boundary conditions
    for k in np.arange(t):
        for i in np.arange(m):                                                      # For each of the boundary nodes.
            if p[i,2] == 1:                                                         # If the node is in the boundary.
                u_ap[i, k] = f(p[i, 0], p[i, 1], T[k], c)                           # The boundary condition is assigned.
  
    # Initial condition
    for i in np.arange(m):                                                          # For each of the nodes.
        u_ap[i, 0] = f(p[i, 0], p[i, 1], T[0], c)                                   # The initial condition is assigned.
    
    # Neighbor search for all the nodes.
    vec = Neighbors(p, 8)                                                           # Neighbor search with the proper routine.

    # Computation of Gamma values
    L = np.vstack([[0], [0], [2], [0], [2]])                                        # The values of the differential operator are assigned.
    Gamma = Gammas(p, vec, L)                                                       # Gamma computation.

    # A Generalized Finite Differences Method
    ## Second time step computation.
    for k in np.arange(1,2):                                                        # For the second time step.
        for i in np.arange(m):                                                      # For all the interior nodes.
            if p[i,2] == 0:
                utemp = 0                                                           # utemp is initialized with 0.
                nvec = sum(vec[i,:] != -1)                                          # Number of neighbor nodes of the central node.
                for j in np.arange(1,nvec+1):                                       # For each of the neighbor nodes.
                    utemp = utemp + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1]), k-1]      # utemp computation with the neighbors.

                utemp = utemp + cdt*Gamma[i,0]*u_ap[i, k-1]                         # The central node is added to the approximation.
                u_ap[i,k] = u_ap[i, k-1] + (1/2)*utemp + \
                            dt*g(p[i, 0], p[i, 1], T[k], c)                         # The second time step is completed.
    
    ## Other time steps computation.
    for k in np.arange(2,t):                                                        # For all the other time steps.
        for i in np.arange(m):                                                      # For all the interior nodes.
            if p[i,2] == 0:
                utemp = 0                                                           # utemp is initialized with 0.
                nvec = sum(vec[i,:] != -1)                                          # Number of neighbor nodes of the central node.
                for j in np.arange(1,nvec+1):                                       # For each of the neighbor nodes.
                    utemp = utemp + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1]), k-1]      # utemp computation with the neighbors.

                utemp = utemp + cdt*Gamma[i,0]*u_ap[i, k-1]                         # The central node is added to the approximation.
                u_ap[i,k] = 2*u_ap[i, k-1] - u_ap[i, k-2] + utemp                   # The time step is completed.

    # Theoretical Solution
    for k in np.arange(t):                                                          # For all the time steps.
        for i in np.arange(m):                                                      # For each of the nodes.
            u_ex[i,k] = f(p[i,0], p[i,1], T[k], c)                                  # The theoretical solution is computed.

    return u_ap, u_ex, vec