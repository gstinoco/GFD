
## Poisson 2D implementado en Mallas Lógicamente Rectangulares.
# 
# Función que calcula una aproximación a la solución de la ecuación de Poisson en 2D utilizando Diferencias Finitas Generalizadas en mallas lógicamente rectangulares.
# 
# El problema que se resuelve es:
# 
# \nabla^2 \phi = f
# 
# El número de vecinos varia dependiendo del nodo.
# 
# ### Parámetros de entrada
#     x           m x n       double      Matriz con las coordenadas en x de los nodos.
#     y           m x n       double      Matriz con las coordenadas en y de los nodos.
#     phi                     function    Función declarada con la condición de frontera.
#     f                       function    Función declarada con el lado derecho de la ecuación.
# 
# ### Parámetros de salida
#     phi_ap      m x 1       double      Vector con la aproximación calculada por el método.
#     phi_ex      m x 1       double      Vector con la solución exacta del problema.

## Importación de Librerias
# En esta parte se importan las librerias necesarias para ejecutar todo el código. En particual se importan:
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
from Gammas import Mesh as Gammas
from Errores import Mesh_Static as ECM
from Graficar import Mesh_Static as Graph
from scipy.io import loadmat


## Se cargan los datos de la region
# Malla para trabajar
malla = 'CAB21'

# Se cargan todos los datos desde el archivo
mat = scipy.io.loadmat('Regiones/Mallas/' + malla + '.mat')

# Se guardan los datos de los nodos
x  = mat['x']
y  = mat['y']

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
# Se resuelve el problema de Diferencias Finitas Generalizadas.

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
    Gamma = Gammas(x, y, L)                                                         # Se calculan las Gammas.

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

## Ejecución
# En esta parte se ejecutan los códigos necesarios para resolver el problema y se grafican las soluciones.

phi_ap, phi_ex = Poisson_Mesh(x, y, phi, f)
er = ECM(x, y, phi_ap, phi_ex)
print('El error cometido para el método es de: ', er)
Graph(x, y, phi_ap, phi_ex)