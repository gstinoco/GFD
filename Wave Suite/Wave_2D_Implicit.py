import numpy as np
from sys import path
path.insert(0, 'General/')
import Gammas

def Wave_Cloud(p, pb, vec, t, f, g, c, lam):
    
    # Ecuación de Onda en 2D implícito implementado en nubes de puntos
    # 
    # Función que calcula una aproximación a la solución de la ecuación de Onda en 2D utilizando un esquema implícito de Diferencias Finitas Generalizadas esobrenubes de puntos.
    # 
    # El problema que se resuelve es:
    # 
    # <center>$\frac{\partial^2 u}{\partial t^2} = c^2\nabla^2 u$</center>
    # 
    # El número de vecinos varia dependiendo del nodo.
    # 
    # ### Parámetros de entrada
    #     p           m x 2       double      Matriz con las coordenadas de los m nodos.
    #     pb          o x 2       double      Matriz con las coordenadas de los nodos frontera.
    #     vec         mxnvec      double      Matriz con la correspondencia de los vecinos de cada nodo.
    #     fWAV                    function    Función declarada con la condición inicial y de frontera.
    #     gWAV                    function    Función con la derivada temporal de f.
    #     lambda                  real        Coeficiente lambda del esquema.
    # 
    # ### Parámetros de salida
    #     u_ap        m x 1       double      Vector con la aproximación calculada por el método.
    #     u_ex        m x 1       double      Vector con la solución exacta del problema.
  
    m    = len(p[:,0])                                                             # Se encuentra el número de nodos.
    mf   = len(pb[:,0])                                                            # Se encuentra el número de nodos de frontera.
    T    = np.linspace(0,3,t)                                                       # Se hace la malla en el tiempo.
    dt   = T[1] - T[0]                                                             # Se calcula dt.
    tol  = np.finfo(float).eps                                                     # La tolerancia será eps.
    u_ap = np.zeros([m,t])                                                         # Se inicializa u_ap con ceros.
    u_ex = np.zeros([m,t])                                                         # Se inicializa u_ex con ceros.
    cdt  = (c**2)*(dt**2)                                                          # Se hace c^2 dt^2
    tol  = np.finfo(float).eps                                                     # La tolerancia será eps.

    ## Condiciones de Frontera
    for k in np.arange(t):
        for i in np.arange(mf):                                                      # Para cada uno de los nodos de frontera.
            u_ap[i, k] = f(pb[i, 0], pb[i, 1], T[k], c)                                # Se agrega la condición de frontera.
    
    ## Condición Inicial
    for i in np.arange(m):                                                         # Para cada uno de los nodos
        u_ap[i, 0] = f(p[i, 0], p[i, 1], T[0], c)                                    # Se agrega la condición inicial

    ## Ahora, se calculan las Gammas
    L = np.vstack([[0], [0], [2], [0], [2]])                               # Se asignan los valores para el operador diferencial.
    Gamma = Gammas.Cloud(p, pb, vec, L)                                                  # Se calculan las Gammas.

    ## Se calcula el segundo nivel de tiempo
    for k in np.arange(1,2):
        er = 1

        # Primero se calcula una predicción
        for i in np.arange(mf, m):
            utemp = 0
            nvec = sum(vec[i,:] != 0)                                                  # Se calcula el número de vecinos que tiene el nodo.
            for j in np.arange(1,nvec+1):
                utemp = utemp + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1])-1, k-1]

            utemp = utemp + cdt*Gamma[i,0]*u_ap[i, k-1]
            u_ap[i,k] = u_ap[i, k-1] + (1/2)*utemp + \
                        dt*g(p[i, 0], p[i, 1], T[k], c)
        
        # Ahora hacemos la corrección
        while er >= tol:
            Z = u_ap[:,k]
            for i in np.arange(mf,m):
                utemp1 = 0
                utemp2 = 0
                nvec = sum(vec[i,:] != 0)                                                # Se calcula el número de vecinos que tiene el nodo.
                for j in np.arange(1,nvec+1):
                    utemp1 = utemp1 + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1])-1, k-1]
                    utemp2 = utemp2 + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1])-1, k]
                
                utemp1 = utemp1 + cdt*Gamma[i,0]*u_ap[i, k-1]

                u_ap[i,k] = (u_ap[i,k-1] + dt*g(p[i, 0], p[i, 1], T[k], c) + \
                            (1/2)*(lam*utemp1 + (1-lam)*utemp2))/ \
                            (1 - (1/2)*(1-lam)*Gamma[i,0]*cdt)
            
            er = np.max(abs(u_ap[:,k] - Z))
    
    ## Se calculan los tiempos siguientes
    for k in np.arange(2,t):
        er = 1

        # Primero se calcula una predicción
        for i in np.arange(mf, m):
            utemp = 0
            nvec = sum(vec[i,:] != 0)                                                  # Se calcula el número de vecinos que tiene el nodo.
            for j in np.arange(1,nvec+1):
                utemp = utemp + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1])-1, k-1]

            utemp = utemp + cdt*Gamma[i,0]*u_ap[i, k-1]
            u_ap[i,k] = 2*u_ap[i, k-1] - u_ap[i, k-2] + utemp
        
        # Ahora hacemos la corrección
        while er >= tol:
            Z = u_ap[:,k]
            for i in np.arange(mf,m):
                utemp1 = 0
                utemp2 = 0
                nvec = sum(vec[i,:] != 0)                                                # Se calcula el número de vecinos que tiene el nodo.
                for j in np.arange(1,nvec+1):
                    utemp1 = utemp1 + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1])-1, k-1]
                    utemp2 = utemp2 + cdt*Gamma[i,j]*u_ap[int(vec[i, j-1])-1, k]
                
                utemp1 = utemp1 + cdt*Gamma[i,0]*u_ap[i, k-1]
                u_ap[i,k] = (2*u_ap[i, k-1] - u_ap[i, k-2] + \
                            lam*utemp1 + (1-lam)*utemp2)/ \
                            (1 - (1-lam)*Gamma[i,0]*cdt)

            er = np.max(abs(u_ap[:,k] - Z))


    ## Se calcula la solución exacta
    for k in np.arange(t):
        for i in np.arange(m):                                                       # Para cada uno de los nodos.
            u_ex[i,k] = f(p[i,0], p[i,1], T[k], c)                                    # Se coloca la solución exacta.

    return u_ap, u_ex