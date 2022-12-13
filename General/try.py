import math
from scipy.io import loadmat
from sys import path
path.insert(0, 'General/')
import Neighbors

nube = 'CAB_1'
mat = loadmat('Regions/Clouds/' + nube + '.mat')

p   = mat['p']
pb  = mat['pb']
vec = mat['vec']
tt  = mat['t']
nvec = 9

vec = Neighbors.Clouds(p, pb, nvec)
print('Done')