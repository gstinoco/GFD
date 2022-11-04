import numpy as np
from sys import path
path.insert(0, 'General/')
import Gammas
import Neighbors

def Diffusion_Mesh(x, y, f, nu, t):
    # 2D Diffusion implemented in Logically Rectangular Meshes.
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

    me   = x.shape                                                                  # Se encuentra el tamaño de la malla.
    m    = me[0]                                                                    # Se encuentra el número de nodos en x.
    n    = me[1]                                                                    # Se encuentra el número de nodos en y.
    T    = np.linspace(0,1,t)                                                       # Se hace la malla en el tiempo.
    dt   = T[1] - T[0]                                                              # Se calcula dt.
    u_ap = np.zeros([m, n, t])                                                      # Se inicializa u_ap con ceros.
    u_ex = np.zeros([m, n, t])                                                      # Se inicializa u_ex con ceros.

    ## Condiciones de Frontera
    for k in np.arange(t):
        for i in np.arange(m):                                                      # Para cada uno de los nodos en las fronteras en x.
            u_ap[i, 0,   k] = f(x[i, 0], y[i, 0], T[k], nu)                         # Se agrega la condición de frontera.
            u_ap[i, n-1, k] = f(x[i, n-1], y[i, n-1], T[k], nu)                     # Se agrega la condición de frontera.
        for j in np.arange(n):                                                      # Para cada uno de los nodos en las fronteras en y.
            u_ap[0,   j, k] = f(x[0, j], y[0, j], T[k], nu)                         # Se agrega la condición de frontera.
            u_ap[m-1, j, k] = f(x[m-1, j], y[m-1, j], T[k], nu)                     # Se agrega la condición de frontera.
  
    ## Condición Inicial
    for i in np.arange(m):
        for j in np.arange(n):
            u_ap[i, j, 0] = f(x[i, j], y[i, j], T[0], nu)                           # Se agrega la condición inicial

    # Ahora, se calculan las Gammas
    L = np.vstack([[0], [0], [2*nu*dt], [0], [2*nu*dt]])                            # Se asignan los valores para el operador diferencial.
    Gamma = Gammas.Mesh(x, y, L)                                                    # Se calculan las Gammas.

    for k in np.arange(1,t):
        for i in np.arange(1,m-1):
            for j in np.arange(1,n-1):
                u_ap[i, j, k] = u_ap[i, j, k-1] + (\
                    Gamma[i, j, 0]*u_ap[i    , j    , k-1] + \
                    Gamma[i, j, 1]*u_ap[i + 1, j    , k-1] + \
                    Gamma[i, j, 2]*u_ap[i + 1, j + 1, k-1] + \
                    Gamma[i, j, 3]*u_ap[i    , j + 1, k-1] + \
                    Gamma[i, j, 4]*u_ap[i - 1, j + 1, k-1] + \
                    Gamma[i, j, 5]*u_ap[i - 1, j    , k-1] + \
                    Gamma[i, j, 6]*u_ap[i - 1, j - 1, k-1] + \
                    Gamma[i, j, 7]*u_ap[i    , j - 1, k-1] + \
                    Gamma[i, j, 8]*u_ap[i + 1, j - 1, k-1])

    for k in np.arange(t):
        for i in np.arange(m):
            for j in np.arange(n):
                u_ex[i, j, k] = f(x[i, j], y[i, j], T[k], nu)                                    # Se coloca la solución exacta.

    return u_ap, u_ex

def Diffusion_Tri(p, tt, pb, f, nu, t):
    m    = len(p[:,0])                                                              # Se encuentra el número de nodos.
    mf   = len(pb[:,0])                                                             # Se encuentra el número de nodos de frontera.
    T    = np.linspace(0,1,t)                                                       # Se hace la malla en el tiempo.
    dt   = T[1] - T[0]                                                              # Se calcula dt.
    tol  = np.finfo(float).eps                                                      # La tolerancia será eps.
    u_ap = np.zeros([m,t])                                                          # Se inicializa u_ap con ceros.
    u_ex = np.zeros([m,t])                                                          # Se inicializa u_ex con ceros.

    ## Condiciones de Frontera
    for k in np.arange(t):
        for i in np.arange(mf):                                                     # Para cada uno de los nodos de frontera.
            u_ap[i, k] = f(pb[i, 0], pb[i, 1], T[k], nu)                            # Se agrega la condición de frontera.
  
    ## Condición Inicial
    for i in np.arange(m):                                                          # Para cada uno de los nodos.
        u_ap[i, 0] = f(p[i, 0], p[i, 1], T[0], nu)                                  # Se agrega la condición inicial.
    
    ## Se hace la búsqueda de vecinos
    vec = Neighbors.Neighbors_Tri(p, tt, 9)                                         # Se manda llamar la función para buscar vecinos.

    ## Ahora, se calculan las Gammas
    L = np.vstack([[0], [0], [2*nu*dt], [0], [2*nu*dt]])                            # Se asignan los valores para el operador diferencial.
    Gamma = Gammas.Cloud(p, pb, vec, L)                                             # Se calculan las Gammas.

    ## Se calcula una aproximación
    for k in np.arange(1,t):                                                        # Para los pasos de tiempo de 1 a t.
        for i in np.arange(mf, m):                                                  # Para cada uno de los nodos interiores.
            utemp = 0                                                               # Se inicializa utemp en 0.
            nvec = sum(vec[i,:] != 0)                                               # Se calcula el número de vecinos que tiene el nodo.
            for j in np.arange(1,nvec+1):                                           # Para cada uno de los vecinos del nodo actual.
                utemp = utemp + Gamma[i,j]*u_ap[int(vec[i, j-1])-1, k-1]            # Se calcula utemp con los vecinos.
            utemp = utemp + Gamma[i,0]*u_ap[i, k-1]                                 # Se agrega el nodo central a la aporixmación.
            u_ap[i,k] = u_ap[i, k-1] + utemp                                        # Se asigna el valor de u_ap.

    ## Se calcula la solución exacta
    for k in np.arange(t):                                                          # Para todos los pasos de tiempo.
        for i in np.arange(m):                                                      # Para cada uno de los nodos.
            u_ex[i,k] = f(p[i,0], p[i,1], T[k], nu)                                 # Se coloca la solución exacta.

    return u_ap, u_ex, vec

def Diffusion_Cloud(p, pb, vec, f, nu, t):
    m    = len(p[:,0])                                                             # Se encuentra el número de nodos.
    mf   = len(pb[:,0])                                                            # Se encuentra el número de nodos de frontera.
    T    = np.linspace(0,1,t)                                                      # Se hace la malla en el tiempo.
    dt   = T[1] - T[0]                                                             # Se calcula dt.
    tol  = np.finfo(float).eps                                                     # La tolerancia será eps.
    u_ap = np.zeros([m,t])                                                         # Se inicializa u_ap con ceros.
    u_ex = np.zeros([m,t])                                                         # Se inicializa u_ex con ceros.

    ## Condiciones de Frontera
    for k in np.arange(t):
        for i in np.arange(mf):                                                    # Para cada uno de los nodos de frontera.
            u_ap[i, k] = f(pb[i, 0], pb[i, 1], T[k], nu)                           # Se agrega la condición de frontera.
  
    ## Condición Inicial
    for i in np.arange(m):                                                         # Para cada uno de los nodos
        u_ap[i, 0] = f(p[i, 0], p[i, 1], T[0], nu)                                 # Se agrega la condición inicial

    # Ahora, se calculan las Gammas
    L = np.vstack([[0], [0], [2*nu*dt], [0], [2*nu*dt]])                           # Se asignan los valores para el operador diferencial.
    Gamma = Gammas.Cloud(p, pb, vec, L)                                            # Se calculan las Gammas.


    for k in np.arange(1,t):
        for i in np.arange(mf, m):
            utemp = 0
            nvec = sum(vec[i,:] != 0)                                              # Se calcula el número de vecinos que tiene el nodo.
            for j in np.arange(1,nvec+1):
                utemp = utemp + Gamma[i,j]*u_ap[int(vec[i, j-1])-1, k-1]
            utemp = utemp + Gamma[i,0]*u_ap[i, k-1]
            u_ap[i,k] = u_ap[i, k-1] + utemp

    for k in np.arange(t):
        for i in np.arange(m):                                                     # Para cada uno de los nodos.
            u_ex[i,k] = f(p[i,0], p[i,1], T[k], nu)                                # Se coloca la solución exacta.

    return u_ap, u_ex, vec

def Diffusion_Mesh_K(x, y, f, nu, t):
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

    me   = x.shape                                                                  # Se encuentra el tamaño de la malla.
    m    = me[0]                                                                    # Se encuentra el número de nodos en x.
    n    = me[1]                                                                    # Se encuentra el número de nodos en y.
    T    = np.linspace(0,1,t)                                                       # Se hace la malla en el tiempo.
    dt   = T[1] - T[0]                                                              # Se calcula dt.
    u_ap = np.zeros([m, n, t])                                                      # Se inicializa u_ap con ceros.
    u_ex = np.zeros([m, n, t])                                                      # Se inicializa u_ex con ceros.
    urr  = np.zeros([m*n,1])                                                        # Se inicializa urr con ceros.


    ## Condiciones de Frontera
    for k in np.arange(t):
        for i in np.arange(m):                                                      # Para cada uno de los nodos en las fronteras en x.
            u_ap[i, 0,   k] = f(x[i, 0], y[i, 0], T[k], nu)                         # Se agrega la condición de frontera.
            u_ap[i, n-1, k] = f(x[i, n-1], y[i, n-1], T[k], nu)                     # Se agrega la condición de frontera.
        for j in np.arange(n):                                                      # Para cada uno de los nodos en las fronteras en y.
            u_ap[0,   j, k] = f(x[0, j], y[0, j], T[k], nu)                         # Se agrega la condición de frontera.
            u_ap[m-1, j, k] = f(x[m-1, j], y[m-1, j], T[k], nu)                     # Se agrega la condición de frontera.

    ## Condición Inicial
    for i in np.arange(m):
        for j in np.arange(n):
            u_ap[i, j, 0] = f(x[i, j], y[i, j], T[0], nu)                           # Se agrega la condición inicial

    ## Se hace el cálculo de K y matrices relacionadas
    L  = np.vstack([[0], [0], [2*nu*dt], [0], [2*nu*dt]])                           # Se asignan los valores para el operador diferencial.
    K  = Gammas.K(x, y, L)
    Kp = np.identity(m*n) + K

    for k in np.arange(1,t):
        R = Gammas.R(u_ap, m, n, k)
        for i in np.arange(m):
            for j in np.arange(n):
                urr[i + j*m, 0] = u_ap[i, j, k-1]
        
        un = (Kp@urr)
        un = un + R

        for i in np.arange(1,m-1):
            for j in np.arange(1,n-1):
                u_ap[i, j, k] = un[i + (j)*m]

    for k in np.arange(t):
        for i in np.arange(m):
            for j in np.arange(n):
                u_ex[i, j, k] = f(x[i, j], y[i, j], T[k], nu)                       # Se coloca la solución exacta.

    return u_ap, u_ex