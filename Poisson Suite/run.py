import math
from scipy.io import loadmat
from sys import path
path.insert(0, 'Generales/')
import Errores
import Graficar
import Poisson_2D

## Se cargan los datos de la region
# Nube para trabajar
nube = 'CAB_1'

# Se cargan todos los datos desde el archivo
mat = loadmat('Regiones/Nubes/' + nube + '.mat')

# Se guardan los datos de los nodos
p   = mat['p']
pb  = mat['pb']
vec = mat['vec']
tt  = mat['t']

# Malla para trabajar
malla = 'CAB21'

# Se cargan todos los datos desde el archivo
mat = loadmat('Regiones/Mallas/' + malla + '.mat')

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

phi_ap, phi_ex = Poisson_2D.Poisson_Cloud(p, pb, vec, phi, f)
er = Errores.Cloud_Static(p, vec, phi_ap, phi_ex)
print('El error cometido para la nube', nube, ' es de: ', er)
Graficar.Cloud_Static(p, phi_ap, phi_ex)

phi_ap, phi_ex, vec = Poisson_2D.Poisson_Tri(p, tt, pb, phi, f)
er = Errores.Cloud_Static(p, vec, phi_ap, phi_ex)
print('El error cometido para la triangulación', nube,  ' es de: ', er)
Graficar.Cloud_Static(p, phi_ap, phi_ex)

phi_ap, phi_ex = Poisson_2D.Poisson_Mesh(x, y, phi, f)
er = Errores.Mesh_Static(x, y, phi_ap, phi_ex)
print('El error cometido para la malla', malla, 'es de: ', er)
Graficar.Mesh_Static(x, y, phi_ap, phi_ex)