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

## Cálculo de Gammas para diferentes códigos
# En este archivo se definen diferentes funciones para el cálculo de Gammas, el cálculo de Gammas se define para los siguientes casos:
# 
#     1.   El problema se resuelve en una malla lógicamente rectangular.
#     2.   El problema se resuelve en una triangulación o en una nube de puntos.
# 
# En todos los casos, es necesario introducir la región en $x$ y $y$.

import numpy as np
import math
 
def Mesh(x, y, L):
    me       = x.shape                                                              # Se encuentra el tamaño de la malla.
    m        = me[0]                                                                # Se encuentra el tamaño en x.
    n        = me[1]                                                                # Se encuentra el tamaño en y.
    Gamma    = np.zeros([m,n,9])                                                    # Se inicializa Gamma en cero.

    for i in range(1,m-1):                                                          # Para cada uno de los nodos en x.
        for j in range(1,n-1):                                                      # Para cada uno de los nodos en y.
            dx = np.array([x[i + 1, j]   - x[i, j], x[i + 1, j + 1] - x[i, j], \
                           x[i, j + 1]   - x[i, j], x[i - 1, j + 1] - x[i, j], \
                           x[i - 1, j]   - x[i, j], x[i - 1, j - 1] - x[i, j], \
                           x[i, j - 1]   - x[i, j], x[i + 1, j - 1] - x[i, j]])     # Se calcula dx.
            
            dy = np.array([y[i + 1, j]   - y[i, j], y[i + 1, j + 1] - y[i, j], \
                           y[i, j + 1]   - y[i, j], y[i - 1, j + 1] - y[i, j], \
                           y[i - 1, j]   - y[i, j], y[i - 1, j - 1] - y[i, j], \
                           y[i, j - 1]   - y[i, j], y[i + 1, j - 1] - y[i, j]])     # Se calcula dy
            
            M = np.vstack([[dx], [dy], [dx**2], [dx*dy], [dy**2]])                  # Se hace la matriz M.
            M = np.linalg.pinv(M)                                                   # Se calcula la pseudoinversa.
            YY = M@L                                                                # Se calcula M*L.
            Gem = np.vstack([-sum(YY), YY])                                         # Se encuentran los balores Gamma.
            for k in np.arange(9):
                Gamma[i,j,k] = Gem[k]                                             # Se guardan los valores Gamma correspondientes.

    return Gamma

def Cloud(p, pb, vec, L):
    nvec  = len(vec[:,1])                                                           # Se encuentra el número máximo de vecinos.
    m     = len(p[:,0])                                                             # Se encuentra el número de nodos.
    mf    = len(pb[:,0])                                                            # Se encuentra el número de nodos frontera.
    Gamma = np.zeros([m, nvec])                                                     # Se inicializa el arreglo para guardar las Gammas.

    for i in np.arange(mf, m):                                                      # Para cada uno de los nodos internos.
        nvec = sum(vec[i,:] != -1)                                                   # Se calcula el número de vecinos que tiene el nodo.
        dx = np.zeros([nvec])                                                       # Se inicializa dx en 0.
        dy = np.zeros([nvec])                                                       # Se inicializa dy en 0.

        for j in np.arange(nvec):                                                   # Para cada uno de los nodos vecinos.
            vec1 = int(vec[i, j])                                                   # Se obtiene el índice del vecino.
            dx[j] = p[vec1, 0] - p[i,0]                                             # Se calcula dx.
            dy[j] = p[vec1, 1] - p[i,1]                                             # Se calcula dy.

        M = np.vstack([[dx], [dy], [dx**2], [dx*dy], [dy**2]])                      # Se hace la matriz M.
        M = np.linalg.pinv(M)                                                       # Se calcula la pseudoinversa.
        YY = M@L                                                                    # Se calcula M*L.
        Gem = np.vstack([-sum(YY), YY]).transpose()                                 # Se encuentran los valores Gamma.
        for j in np.arange(nvec+1):                                                 # Para cada uno de los vecinos.
            Gamma[i,j] = Gem[0,j]                                                   # Se guarda el Gamma correspondiente.
    
    return Gamma

def K(x, y, L):
    me = x.shape                                                                    # Se encuentra el tamaño de la malla.
    m  = me[0]                                                                      # Se encuentra el número de nodos en x.
    n  = me[1]                                                                      # Se encuentra el número de nodos en y.
    K  = np.zeros([(m)*(n), (m)*(n)])                                               # Se define la matríz K que guardará la estructura de las Diferencias Finitas Generalizadas.

    for i in np.arange(1,m-1):
        for j in np.arange(1,n-1):
            u = np.array(x[i-1:i+2, j-1:j+2])
            v = np.array(y[i-1:i+2, j-1:j+2])
            dx = np.hstack([u[0,0] - u[1,1], u[1,0] - u[1,1], \
                            u[2,0] - u[1,1], u[0,1] - u[1,1], \
                            u[2,1] - u[1,1], u[0,2] - u[1,1], \
                            u[1,2] - u[1,1], u[2,2] - u[1,1]])
            dy = np.hstack([v[0,0] - v[1,1], v[1,0] - v[1,1], \
                            v[2,0] - v[1,1], v[0,1] - v[1,1], \
                            v[2,1] - v[1,1], v[0,2] - v[1,1], \
                            v[1,2] - v[1,1], v[2,2] - v[1,1]])
            M = np.vstack([[dx], [dy], [dx**2], [dx*dy], [dy**2]])                  # Se hace la matriz M.
            M = np.linalg.pinv(M)                                                   # Se calcula la pseudoinversa.
            YY = M@L                                                                # Se calcula M*L.
            Gamma = np.vstack([-sum(YY), YY])                                       # Se encuentran los balores Gamma.
            p = m*(j) + i
            K[p, p-1]   = Gamma[4]
            K[p, p]     = Gamma[0]
            K[p, p+1]   = Gamma[5]
            K[p, p-1-m] = Gamma[1]
            K[p, p-m]   = Gamma[2]
            K[p, p+1-m] = Gamma[3]
            K[p, p-1+m] = Gamma[6]
            K[p, p+m]   = Gamma[7]
            K[p, p+1+m] = Gamma[8]
    
    for j in np.arange(n):
        K[m*j, m*j] = 0
    
    for i in np.arange(1,m-1):
        p = i+(n-1)*m
        K[i, i]                 = 0
        K[p, p] = 0
    
    return K

def R(u, m , n, k):
    R = np.zeros([m*n, 1])

    for i in np.arange(1,m-1):
        R[i, 0]           = u[i, 0, k]   - u[i, 0, k-1]
        R[i + (n-1)*m, 0] = u[i, n-1, k] - u[i, n-1, k-1]
    
    for j in np.arange(n):
        R[(j)*m, 0] = u[0,   j, k] - u[0, j, k-1]
        R[m*(j+1)-1, 0] = u[m-1, j, k] - u[m-1, j, k-1]
    
    return R