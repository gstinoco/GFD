import numpy as np
import random
from numpy import linalg

def normals(p, b_interior = False):
    if max(p[:,2]) == 1:
        b_interior = False
    elif max(p[:,2] == 2):
        b_interior = True

    msc    = (p[:, 2] == 1)
    pb_o   = np.vstack([p[msc]])
    a      = 0
    nb     = len(pb_o[:, 0])
    nor_o  = np.zeros([nb, 2])
    q      = p[nb, 0:2]
    
    for i in np.arange(nb):
        z  = q - pb_o[i-1, 0:2]
        w  = q - pb_o[i, 0:2]
        a += z[0]*w[1] - z[1]*w[0]
    
    if a > 0:
        rota = np.array([[0, 1], [-1, 0]])
    else:
        rota = np.array([[0, -1], [1, 0]])
    
    for i in np.arange(nb-1):
        v          = pb_o[i+1,0:2] - pb_o[i-1,0:2]
        nor_o[i,:] = np.transpose(np.dot(rota, v))
        nor_o[i,:] = nor_o[i,:]/linalg.norm(nor_o[i,:])

    v              = pb_o[0,0:2] - pb_o[nb-2,0:2]
    nor_o[nb-1,:]  = np.transpose(np.dot(rota, v))
    nor_o[nb-1,:]  = nor_o[nb-1,:]/linalg.norm(nor_o[nb-1,:])
    pb             = pb_o
    vecs           = pb_o[:,0:2] + nor_o

    if b_interior == True:
        msc    = (p[:, 2] == 2)
        pb_i   = np.vstack([p[msc]])
        a      = 0
        nb     = len(pb_i[:, 0])
        nor_i  = np.zeros([nb, 2])
        q      = p[nb, 0:2]
    
        for i in np.arange(nb):
            z  = q - pb_i[i-1, 0:2]
            w  = q - pb_i[i, 0:2]
            a += z[0]*w[1] - z[1]*w[0]
    
        if a < 0:
            rota = np.array([[0, 1], [-1, 0]])
        else:
            rota = np.array([[0, -1], [1, 0]])
    
        for i in np.arange(nb-1):
            v          = pb_i[i+1,0:2] - pb_i[i-1,0:2]
            nor_i[i,:] = np.transpose(np.dot(rota, v))
            nor_i[i,:] = nor_i[i,:]/linalg.norm(nor_i[i,:])

        v              = pb_i[0,0:2] - pb_i[nb-2,0:2]
        nor_i[nb-1,:]  = np.transpose(np.dot(rota, v))
        nor_i[nb-1,:]  = nor_i[nb-1,:]/linalg.norm(nor_i[nb-1,:])
        pb             = np.concatenate([pb, pb_i])
        vec_i          = pb_i[:,0:2] + nor_i
        vecs           = np.concatenate([vecs, vec_i])

    return pb, vecs