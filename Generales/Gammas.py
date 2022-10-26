# %% [markdown]
# # Cálculo de Gammas para diferentes códigos
# En este archivo se definen diferentes funciones para el cálculo de Gammas, el cálculo de Gammas se define para los siguientes casos:
# 
#     1.   El problema se resuelve en una malla lógicamente rectangular.
#     2.   El problema se resuelve en una triangulación o en una nube de puntos.
# 
# En todos los casos, es necesario introducir la región en $x$ y $y$.

# %%
import numpy as np
import math

# %%
def Gammas_mesh(x, y, L):
  me       = x.shape                                                             # Se encuentra el tamaño de la malla.
  m        = me[0]                                                               # Se encuentra el tamaño en x.
  n        = me[1]                                                               # Se encuentra el tamaño en y.
  Gamma    = np.zeros([m,n,9])                                                   # Se inicializa Gamma en cero.

  for i in range(1,m-1):                                                         # Para cada uno de los nodos en x.
    for j in range(1,n-1):                                                       # Para cada uno de los nodos en y.
      dx = np.array([x[i + 1, j]   - x[i, j], x[i + 1, j + 1] - x[i, j], \
                     x[i, j + 1]   - x[i, j], x[i - 1, j + 1] - x[i, j], \
                     x[i - 1, j]   - x[i, j], x[i - 1, j - 1] - x[i, j], \
                     x[i, j - 1]   - x[i, j], x[i + 1, j - 1] - x[i, j]])        # Se calcula dx.
      
      dy = np.array([y[i + 1, j]   - y[i, j], y[i + 1, j + 1] - y[i, j], \
                     y[i, j + 1]   - y[i, j], y[i - 1, j + 1] - y[i, j], \
                     y[i - 1, j]   - y[i, j], y[i - 1, j - 1] - y[i, j], \
                     y[i, j - 1]   - y[i, j], y[i + 1, j - 1] - y[i, j]])        # Se calcula dy
      
      M = np.vstack([[dx], [dy], [dx**2], [dx*dy], [dy**2]])                     # Se hace la matriz M.
      M = np.linalg.pinv(M)                                                      # Se calcula la pseudoinversa.
      YY = M@L                                                                   # Se calcula M*L.
      Gem = np.vstack([-sum(YY), YY])                                            # Se encuentran los balores Gamma.
      Gamma[i,j,:] = Gem.transpose()                                             # Se guardan los valores Gamma correspondientes.

  return Gamma

# %%
def Gammas_cloud(p, pb, vec, L):
  nvec  = len(vec[:,1])                                                          # Se encuentra el número máximo de vecinos.
  m     = len(p[:,0])                                                            # Se encuentra el número de nodos.
  mf    = len(pb[:,0])                                                           # Se encuentra el número de nodos frontera.
  Gamma = np.zeros([m, nvec])                                                    # Se inicializa el arreglo para guardar las Gammas.

  for i in np.arange(mf, m):                                                     # Para cada uno de los nodos internos.
    nvec = sum(vec[i,:] != 0)                                                    # Se calcula el número de vecinos que tiene el nodo.
    dx = np.zeros([nvec])                                                        # Se inicializa dx en 0.
    dy = np.zeros([nvec])                                                        # Se inicializa dy en 0.

    for j in np.arange(nvec):                                                    # Para cada uno de los nodos vecinos.
      vec1 = int(vec[i, j])-1                                                    # Se obtiene el índice del vecino.
      dx[j] = p[vec1, 0] - p[i,0]                                                # Se calcula dx.
      dy[j] = p[vec1, 1] - p[i,1]                                                # Se calcula dy.

    M = np.vstack([[dx], [dy], [dx**2], [dx*dy], [dy**2]])                       # Se hace la matriz M.
    M = np.linalg.pinv(M)                                                        # Se calcula la pseudoinversa.
    YY = M@L                                                                     # Se calcula M*L.
    Gem = np.vstack([-sum(YY), YY]).transpose()                                  # Se encuentran los valores Gamma.
    for j in np.arange(nvec+1):                                                  # Para cada uno de los vecinos.
      Gamma[i,j] = Gem[0,j]                                                      # Se guarda el Gamma correspondiente.
  
  return Gamma


