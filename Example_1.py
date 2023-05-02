import numpy as np
import time
from GFD import GFD_Transient as GFD
from scipy.io import loadmat

def fDIF(x, y, t, c):
    fun = np.exp(-2*np.pi**2*c*t)*np.cos(np.pi*x)*np.cos(np.pi*y)
    return fun

def fADIF(x, y, t, c):
    fun = (1/(4*t+1))*np.exp(-(x-c[0]*t-0.5)**2/(c[2]*(4*t+1)) - (y-c[1]*t-0.5)**2/(c[2]*(4*t+1)))
    return fun

def fADV(x, y, t, c):
    fun = 0.2*np.exp((-(x-.5-c[0]*t)**2-(y-.5-c[1]*t)**2)/.001)
    return fun

mat = loadmat('Regions/Clouds/CAB_2.mat')
p   = mat['p']
tt  = mat['tt']
if tt.min() == 1:
    tt -= 1

solu = GFD(points = p, triangulation = tt, time_interval = [0,1], time_steps = 2000)

solu.Solver('Advection', fADV, [0.2, 0.3], triangulation = True, implicit = True)
solu.Graph()

solu.Solver('Diffusion', fDIF, 0.2, triangulation = True, implicit = True)
solu.Graph()

solu.Solver('Advection-Diffusion', fADIF, [0.2, 0.3, 0.05], triangulation = True, implicit = True)
solu.Graph()

time.sleep(2)