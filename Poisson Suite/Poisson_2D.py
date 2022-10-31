import numpy as np
from sys import path
path.insert(0, 'Generales/')
import Gammas

def BuscarVecinos(p, tt, nvec):                                                     # Script to find the neighbors in a triangulation.
    m   = len(p[:,0])                                                               # The size if the triangulation is obtained.
    vec = np.zeros([m, nvec*2])                                                     # The array for the neighbors is initialized.
    for i in np.arange(m):                                                          # For each of the nodes.
        kn   = np.argwhere(tt == i+1)                                               # Se busca en qué triángulos aparece el nodo.
        vec2 = np.setdiff1d(tt[kn[:,0]], i+1)                                       # Se guardan los vecinos dentro de vec2.
        vec2 = np.vstack([vec2])                                                    # Se convierte vec2 en una columna.
        nvec = sum(vec2[0,:] != 0)                                                  # Se calcula el número de vecinos del nodo.
        for j in np.arange(nvec):                                                   # Para cada uno de los vecinos.
            vec[i,j] = vec2[0,j]                                                    # Se guardan los vecinos.
    return vec

def Poisson_Mesh(x, y, phi, f):
    ## Se inicializan las variables
    me       = x.shape                                                              # Se encuentra el tamaño de la malla.
    m        = me[0]                                                                # Se encuentra el número de nodos en x.
    n        = me[1]                                                                # Se encuentra el número de nodos en y.
    err      = 1                                                                    # Se inicializa el error en 1.
    tol      = np.finfo(float).eps                                                  # La tolerancia sera eps.
    phi_ap   = np.zeros([m,n])                                                      # Se inicializa phi_ap con ceros.
    phi_ex   = np.zeros([m,n])                                                      # Se inicializa phi_ex con ceros.
    ## Se fijan las condiciones de Frontera
    for i in range(m):                                                              # Para cada uno de los nodos en las fronteras en x.
        phi_ap[i, 0]   = phi(x[i, 0],   y[i, 0])                                    # Se asigna la condición de frontera en el primer y.
        phi_ap[i, n-1] = phi(x[i, n-1], y[i, n-1])                                  # Se asigna la condición de frontera en el último y.
    for j in range(n):                                                              # Para cada uno de los nodos en las fronteras en y.
        phi_ap[0,   j] = phi(x[0,   j], y[0,   j])                                  # Se asigna la condición de frontera en el primer x.
        phi_ap[m-1, j] = phi(x[m-1, j], y[m-1, j])                                  # Se asigna la condición de frontera en el último x.
    ## Se calculan los valores de los Gamma
    L = np.vstack([[0], [0], [2], [0], [2]])                                        # Se asignan los valores del operador dierencial.
    Gamma = Gammas.Mesh(x, y, L)                                                    # Se calculan las Gammas.
    ## Se hace el Método de Diferencias Finitas Generalizadas
    while err >= tol:                                                               # Mientras que el error sea mayor que la tolerancia.
        err = 0                                                                     # Error se hace cero para poder actualizar.
        for i in range(1,m-1):                                                      # Para cada uno de los nodos en x.
            for j in range(1,n-1):                                                  # Para cada uno de los valores en y.
                t = (f(x[i, j], y[i, j]) - (              \
                    Gamma[i, j, 1]*phi_ap[i + 1, j    ] + \
                    Gamma[i, j, 2]*phi_ap[i + 1, j + 1] + \
                    Gamma[i, j, 3]*phi_ap[i    , j + 1] + \
                    Gamma[i, j, 4]*phi_ap[i - 1, j + 1] + \
                    Gamma[i, j, 5]*phi_ap[i - 1, j    ] + \
                    Gamma[i, j, 6]*phi_ap[i - 1, j - 1] + \
                    Gamma[i, j, 7]*phi_ap[i    , j - 1] + \
                    Gamma[i, j, 8]*phi_ap[i + 1, j - 1]))/Gamma[i, j, 0]            # Se calcula phi_ap en el nodo central.
                err = max(err, abs(t - phi_ap[i, j]));                              # Se calcula el error.
                phi_ap[i,j] = t;                                                    # Se asigna el valor calculado previamente.
    ## Se guarda la solución exacta
    for i in range(m):                                                              # Para todos los nodos en x.
        for j in range(n):                                                          # Para todos los nodos en y.
            phi_ex[i,j] = phi(x[i,j], y[i,j])                                       # Se coloca la solución exacta.
    return phi_ap, phi_ex

def Poisson_Tri(p, tt, pb, phi, f):
    m        = len(p[:,0])                                                          # Se encuentra el número de nodos.
    mf       = len(pb[:,0])                                                         # Se encuentra el número de nodos de frontera.
    nvec     = 9                                                                    # Se establece un número máximo de vecinos.
    err      = 1                                                                    # Se inicializa el error con 1.
    tol      = np.finfo(float).eps                                                  # La tolerancia será eps.
    phi_ap   = np.zeros([m])                                                        # Se inicializa phi_ap con ceros.
    phi_ex   = np.zeros([m])                                                        # Se inicializa phi_ex con ceros.
    ## Conduciones de Frontera
    for i in np.arange(mf):                                                         # Para cada uno de los nodos de frontera.
        phi_ap[i]   = phi(pb[i, 0],   pb[i, 1])                                     # Se agrega la condición de frontera.
    ## Primero se calculan los nodos vecinos
    vec = BuscarVecinos(p, tt, nvec)                                                # Se hace encuentran los vecinos.
    ## Ahora, se calculan las Gammas
    L = np.vstack([[0], [0], [2], [0], [2]])                                        # Se asignan los valores para el operador diferencial.
    Gamma = Gammas.Cloud(p, pb, vec, L)                                             # Se calculan las Gammas.
    while err >= tol:                                                               # Mientras que el error sea mayor que la tolerancia.
        err = 0                                                                     # Error se hace cero para poder actualizar.
        for i in np.arange(mf, m):                                                  # Para cada uno de los nodos interiores.
            phitemp = 0                                                             # phitemp se hace cero.
            nvec = sum(vec[i,:] != 0)                                               # Se calcula el número de vecinos que tiene el nodo.
            for j in np.arange(1,nvec+1):                                           # Para cada uno de los vecinos.
                phitemp = phitemp + Gamma[i, j]*phi_ap[int(vec[i, j-1])-1]          # Se calcula hacen diferencias finitas.
            t = (f(p[i, 0], p[i, 1]) - phitemp)/Gamma[i,0]                          # Se calcula el valor de phi_ap en el nodo central.
            err = max(err, abs(t - phi_ap[i]));                                     # Se calcula el error.
            phi_ap[i] = t;                                                          # Se asigna el valor calculado previamente.
        for i in range(m):                                                          # Para cada uno de los nodos.
            phi_ex[i] = phi(p[i,0], p[i,1])                                         # Se coloca la solución exacta.
    return phi_ap, phi_ex, vec

def Poisson_Cloud(p, pb, vec, phi, f):
    m        = len(p[:,0])                                                          # Se encuentra el número de nodos.
    mf       = len(pb[:,0])                                                         # Se encuentra el número de nodos de frontera.
    mm       = len(p[np.arange(mf,m),0])                                            # Se encuentra el número de nodos interiores.
    err      = 1                                                                    # Se inicializa el error con 1.
    tol      = np.finfo(float).eps                                                  # La tolerancia será eps.
    phi_ap   = np.zeros([m])                                                        # Se inicializa phi_ap con ceros.
    phi_ex   = np.zeros([m])                                                        # Se inicializa phi_ex con ceros.
    for i in np.arange(mf):                                                         # Para cada uno de los nodos de frontera.
        phi_ap[i]   = phi(pb[i, 0],   pb[i, 1])                                     # Se agrega la condición de frontera.
    # Ahora, se calculan las Gammas
    L = np.vstack([[0], [0], [2], [0], [2]])                                        # Se asignan los valores para el operador diferencial.
    Gamma = Gammas.Cloud(p, pb, vec, L)                                             # Se calculan las Gammas.
    while err >= tol:                                                               # Mientras que el error sea mayor que la tolerancia.
        err = 0                                                                     # Error se hace cero para poder actualizar.
        for i in np.arange(mf, m):                                                  # Para cada uno de los nodos interiores.
            phitemp = 0                                                             # phitemp se hace cero.
            nvec = sum(vec[i,:] != 0)                                               # Se calcula el número de vecinos que tiene el nodo.
            for j in np.arange(1,nvec+1):                                           # Para cada uno de los vecinos.
                phitemp = phitemp + Gamma[i, j]*phi_ap[int(vec[i, j-1])-1]          # Se calcula hacen diferencias finitas.
            t = (f(p[i, 0], p[i, 1]) - phitemp)/Gamma[i,0]                          # Se calcula el valor de phi_ap en el nodo central.
            err = max(err, abs(t - phi_ap[i]));                                     # Se calcula el error.
            phi_ap[i] = t;                                                          # Se asigna el valor calculado previamente.
        for i in range(m):                                                          # Para cada uno de los nodos.
            phi_ex[i] = phi(p[i,0], p[i,1])                                         # Se coloca la solución exacta.
    return phi_ap, phi_ex