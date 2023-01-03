import numpy as np
from numpy import linalg

def normals(pb, p):
    nb  = len(pb[:,0])
    q   = p[nb, :]
    z   = q - pb[nb-1,:]
    w   = q - pb[0,:]
    a   = z[0]*w[1] - z[1]*w[0]
    nor = np.zeros([nb,2])

    for i in np.arange(nb-1):
        z = q - pb[i,:]
        w = q - pb[i+1,:]
        a = a + z[0]*w[1] - z[1]*w[0]
    
    if a > 0:
        rota = np.array([[0, 1],[-1, 0]])
    else:
        rota = np.array([[0, -1],[1, 0]])
    
    for i in np.arange(1,nb-1):
        v        = pb[i+1,:] - pb[i-1,:];
        nor[i,:] = np.transpose(np.dot(rota, v))
        nor[i,:] = nor[i,:]/linalg.norm(nor[i,:])
    
    v        = pb[1,:] - pb[nb-1,:]
    nor[1,:] = np.transpose(np.dot(rota, v))
    nor[1,:] = nor[1,:]/linalg.norm(nor[1,:])

    v           = pb[0,:] - pb[nb-2,:]
    nor[nb-1,:] = np.transpose(np.dot(rota, v))
    nor[nb-1,:] = nor[nb-1,:]/linalg.norm(nor[nb-1,:])

    vecs = pb + nor

    return vecs