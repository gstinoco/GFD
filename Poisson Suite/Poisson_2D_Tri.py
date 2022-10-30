## Poisson 2D implementado en Triangulaciones.
# 
# Función que calcula una aproximación a la solución de la ecuación de Poisson en 2D utilizando Diferencias Finitas Generalizadas en traingulaciones realizadas en DistMesh.
# 
# El problema que se resuelve es:
# 
# \nabla^2 \phi = f
# 
# El número de vecinos varia dependiendo del nodo.
# 
# Parámetros de entrada
#     p           m x 2       double      Matriz con las coordenadas de los nodos.
#     tt          n x 3       double      Matriz con la correspondencia de los n triángulos.
#     pb          o x 2       double      Matrix con las coordenadas de los nodos de frontera.
#     phi                     function    Función declarada con la condición de frontera.
#     f                       function    Función declarada con el lado derecho de la ecuación.
# 
# Parámetros de salida
#     phi_ap      m x 1       double      Vector con la aproximación calculada por el método.
#     phi_ex      m x 1       double      Vector con la solución exacta del problema.


## Importación de Librerias
# 
#    numpy.         Para poder hacer la mayor parte de los cálculos numéricos.
#    math.          Permite hacer uso de diferentes funciones matemáticas.
#    matplotlib.    Para hacer las gráficas necesarias.
#    scipy.io.      Para usar una gran cantidad de funciones matemáticas.
#    sys.           Para poder usar códigos en otras carpetas.
#    Gammas.        Rutinas para el cálculo de Gammas del Método de Diferencias Finias Generalizadas.
#    Errores.       Rutinas para calcular el Error Cuadrático Medio.
#    Graficar.      Rutinas para graficar los resultados.

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io
import sys
sys.path.insert(0, 'Generales/')
from Gammas import Cloud as Gammas
from Errores import Cloud_Static as ECM
from Graficar import Cloud_Static as Graph
from scipy.io import loadmat

## Se cargan los datos de la region
# Nube para trabajar
nube = 'CAB_1'

# Se cargan todos los datos desde el archivo
mat = scipy.io.loadmat('Regiones/Nubes/' + nube + '.mat')

# Se guardan los datos de los nodos
p   = mat['p']
tt  = mat['t']
pb  = mat['pb']

## Se definen las condiciones de frontera 
# Las condiciones de frontera setán definidas como:
#   \phi = 2e^{2x+y}
# 
#   f = 10e^{2x+y}

def phi(x,y):
  fun = 2*math.exp(2*x+y)
  return fun

def f(x,y):
  fun = 10*math.exp(2*x+y)
  return fun

## Diferencias Finitas Generalizadas
### Busqueda de vecinos. 
# Los vecinos que se encuentran son los vecinos obtenidos por medio de la triangución obtenida de DistMesh.

def BuscarVecinos(p, tt, nvec):
    m   = len(p[:,0])                                                              # Se encuentra el tamaño de la triangulación.
    vec = np.zeros([m, nvec*2])                                                      # Se inicializa el arreglo para guardar los vecinos.

    for i in np.arange(m):                                                         # Para cada uno de los nodos.
        kn   = np.argwhere(tt == i+1)                                                # Se busca en qué triángulos aparece el nodo.
        #print('Los triángulos donde aparece el nodo', i, ' son:', kn[:,0])
        vec2 = np.setdiff1d(tt[kn[:,0]], i+1)                                        # Se guardan los vecinos dentro de vec2.
        #print('Los vecinos del nodo', i+1, ' son:', vec2)
        vec2 = np.vstack([vec2])                                                     # Se convierte vec2 en una columna.
        nvec = sum(vec2[0,:] != 0)                                                   # Se calcula el número de vecinos del nodo.
        for j in np.arange(nvec):                                                    # Para cada uno de los vecinos.
            vec[i,j] = vec2[0,j]                                                       # Se guardan los vecinos.

    return vec

### Diferencias Finitas Generalizadas. 
# Se resuelve el problema de Diferencias Finitas Generalizadas.

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
    Gamma = Gammas(p, pb, vec, L)                                                   # Se calculan las Gammas.

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

## Ejecución
## En esta parte se ejecutan los códigos necesarios para resolver el problema y se grafican las soluciones.

phi_ap, phi_ex, vec = Poisson_Tri(p, tt, pb, phi, f)
er = ECM(p, vec, phi_ap, phi_ex)
print('El error cometido para el método es de: ', er)
Graph(p, phi_ap, phi_ex)