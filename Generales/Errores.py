# %% [markdown]
# # Cálculo del Error
# En este archivo se definen diferentes funciones para calcular el error cometido con los métodos de Diferencias Finitas Generalizadas. Se agregan 3 formas para calcular:
# 
# \begin{align}
#     \| e \|^2 = \left(\sqrt{\sum_{i} (u_{i}^{k} - U_{i}^{k})}\right) A_{i}
# \end{align}
# 
#     1.   El problema se resolvió en mallas lógicamente rectangulares.
#     2.   El problema se resolvió en una triangulación o en una nube de puntos.
# 
# En todos los casos, es necesario introducir la solución exacta $U_{i}^{k}$ y la solución aproximada $u_{i}^{k}$.

# %% [markdown]
# ## Importación de Módulos
# En esta sección se importan todos los módulos necesarios para que se ejecuten correctamente los códigos.

# %%
import numpy as np
import math

# %% [markdown]
# ## Cálculo del error
# En esta sección se implementan los códigos para calcular el Error Cuadrático Medio (ECM) sobre mallas lógicamentes rectangulares y sobre nubes de puntos. Se implementan las siguientes funciones:
# 
#     1.  PolyArea(x, y): Se calcula el área del polígono formado por el nodo central y sus vecinos.
#     2.  ECM_mesh_transient(x, y, u_ap, u_ex): Se calcula el ECM en mallas lógicamente rectangulares para problemas que involucran el tiempo.
#     3.  ECM_mesh_static(x, y, u_ap, u_ex): Se calcula el ECM en mallas lógicamente rectangulares para problemas que no involucran el tiempo.
#     4.  ECM_cloud_transient(x, y, u_ap, u_ex): Se calcula el ECM en nubes de puntos para problemas que involucran el tiempo.
#     5.  ECM_cloud_static(x, y, u_ap, u_ex): Se calcula el ECM en nubes de puntos para problemas que no involucran el tiempo.

# %%
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

# %%
def ECM_mesh_transient(x, y, u_ap, u_ex):
    me   = x.shape                                                                  # Se calcula el tamaño de la malla.
    m    = me[0]                                                                    # Se calcula el número de nodos en x.
    n    = me[1]                                                                    # Se calcula el número de nodos en y.
    t    = len(u_ap[1,:])                                                           # Se encuentra la cantidad de pasos en el tiempo.
    er   = np.zeros(t)                                                              # Se inicializa la variable para guardar el error.
    area = np.zeros([m,n])                                                          # Se inicializa la variable para guardar el área.

    for i in np.arange(1,m-1):                                                      # Para cada uno de los nodos en x.
        for j in np.arange(1,n-1):                                                  # Para cada uno de los nodos en y.
            px = np.array([x[i+1, j], x[i+1, j+1], x[i, j+1], x[i-1, j+1], \
                          x[i-1, j], x[i-1, j-1], x[i, j-1], x[i+1, j-1], \
                          x[i, j]])                                                 # Se guardan los valores x del polígono.
            py = np.array([y[i+1, j], y[i+1, j+1], y[i, j+1], y[i-1, j+1], \
                          y[i-1, j], y[i-1, j-1], y[i, j-1], y[i+1, j-1], \
                          y[i, j]])                                                 # Se guardan los valores y del polígono.
            area[i,j] = PolyArea(px,py)                                             # Se calcula el área.

    for k in np.arange(t):                                                          # Para cada uno de los pasos de tiempo.
        for i in np.arange(1,m-1):                                                  # Para cada uno de los nodos en x.
            for j in np.arange(1,n-1):                                              # Para cada uno de los nodos en y.
                er[k] = er[k] + area[i,j]*(u_ap[i,j,k] - u_ex[i,j,k])**2            # Se calcula el error en el nodo.
        er[k] = math.sqrt(er[k])                                                    # Se calcula la raiz cuadrada.
    
    return er

# %%
def ECM_mesh_static(x, y, u_ap, u_ex):
    me   = x.shape                                                                  # Se calcula el tamaño de la malla.
    m    = me[0]                                                                    # Se calcula el número de nodos en x.
    n    = me[1]                                                                    # Se calcula el número de nodos en y.
    er   = 0                                                                        # Se inicializa la variable para guardar el error.
    area = np.zeros([m,n])                                                          # Se inicializa la variable para guardar el área.

    for i in np.arange(1,m-1):                                                      # Para cada uno de los nodos en x.
        for j in np.arange(1,n-1):                                                  # Para cada uno de los nodos en y.
            px = np.array([x[i+1, j], x[i+1, j+1], x[i, j+1], x[i-1, j+1], \
                          x[i-1, j], x[i-1, j-1], x[i, j-1], x[i+1, j-1], \
                          x[i, j]])                                                 # Se guardan los valores x del polígono.
            py = np.array([y[i+1, j], y[i+1, j+1], y[i, j+1], y[i-1, j+1], \
                          y[i-1, j], y[i-1, j-1], y[i, j-1], y[i+1, j-1], \
                          y[i, j]])                                                 # Se guardan los valores y del polígono.
            area[i,j] = PolyArea(px,py)                                             # Se calcula el área.

    for i in np.arange(1,m-1):                                                      # Para cada uno de los nodos en x.
        for j in np.arange(1,n-1):                                                  # Para cada uno de los nodos en y.
            er = er + area[i,j]*(u_ap[i,j] - u_ex[i,j])**2                          # Se calcula el error en el nodo.
        er = math.sqrt(er)                                                          # Se calcula la raiz cuadrada.
    
    return er

# %%
def ECM_cloud_transient(p, vec, u_ap, u_ex):
    m    = len(p[:,0])                                                              # Se encuentra el tamaño de la triangulación.
    t    = len(u_ap[1,:])                                                           # Se encuentra la cantidad de pasos en el tiempo.
    er   = np.zeros(t)                                                              # Se inicializa la variable para guardar el error.
    area = np.zeros(m)                                                              # Se inicializa la variable para guardar el área.

    for i in np.arange(m):
        nvec = sum(vec[i,:] != 0)                                                   # Se calcula el número de vecinos que tiene el nodo.
        polix = np.zeros([nvec])                                                    # Se hace un arreglo para las coordenadas x del polígono.
        poliy = np.zeros([nvec])                                                    # Se hace un arreglo para las coordenadas y del polígono.
        for j in np.arange(nvec):                                                   # Para cada uno de los nodos vecinos.
            vec1 = int(vec[i,j])-1                                                  # Se encuentra el índice del nodo.
            polix[j] = p[vec1,0]                                                    # Se guarda la coordenada x del nodo.
            poliy[j] = p[vec1,1]                                                    # Se guarda la coordenada y del nodo.
        area[i] = PolyArea(polix, poliy)                                            # Se calcula el área.

    for k in np.arange(t):                                                          # Para cada uno de los pasos de tiempo.
        for i in np.arange(m):                                                      # Para cada uno de los nodos de la malla.
            er[k] = er[k] + area[i]*(u_ap[i,k] - u_ex[i,k])**2                      # Se calcula el error en el nodo.
        er[k] = math.sqrt(er[k])                                                    # Se calcula la raiz cuadrada.
    
    return er

# %%
def ECM_cloud_static(p, vec, u_ap, u_ex):
    m    = len(p[:,0])                                                              # Se encuentra el tamaño de la triangulación.
    er   = 0                                                                        # Se inicializa la variable para guardar el error.
    area = np.zeros(m)                                                              # Se inicializa la variable para guardar el área.

    for i in np.arange(m):
        nvec = sum(vec[i,:] != 0)                                                   # Se calcula el número de vecinos que tiene el nodo.
        polix = np.zeros([nvec])                                                    # Se hace un arreglo para las coordenadas x del polígono.
        poliy = np.zeros([nvec])                                                    # Se hace un arreglo para las coordenadas y del polígono.
        for j in np.arange(nvec):                                                   # Para cada uno de los nodos vecinos.
            vec1 = int(vec[i,j])-1                                                  # Se encuentra el índice del nodo.
            polix[j] = p[vec1,0]                                                    # Se guarda la coordenada x del nodo.
            poliy[j] = p[vec1,1]                                                    # Se guarda la coordenada y del nodo.
        area[i] = PolyArea(polix, poliy)                                            # Se calcula el área.

    for i in np.arange(m):                                                          # Para cada uno de los nodos de la malla.
        er = er + area[i]*(u_ap[i] - u_ex[i])**2                                    # Se calcula el error en el nodo.
    er = math.sqrt(er)                                                              # Se calcula la raiz cuadrada.
    
    return er


