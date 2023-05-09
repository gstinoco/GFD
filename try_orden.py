from scipy.io import loadmat
from scipy.io import savemat
import General.CloudGen as CloudGen
import General.normals as normals
import matplotlib.pyplot as plt
import numpy as np


# Region data is loaded.
regions = ['CAB']#,'CUA','CUI','DOW','ENG','GIB','HAB','MIC','PAT','ZIR']
sizes = ['1']#, '2','3']

for reg in regions:
    regi = reg

    for me in sizes:
        mesh = me

        mat = loadmat('Regions/Clouds/' + regi + '_' + mesh + '.mat')
        p   = mat['p']
        tt  = mat['tt']
        if tt.min() == 1:
            tt -= 1

        p        = CloudGen.OrdenNodes(p)
        pb, vecs = normals.normals(p)

        plt.figure(figsize=(8,5))
        plt.title('Vectores normales')
        plt.scatter(pb[:,0], pb[:,1])
        for i in np.arange(len(pb[:,0])):
            x = [pb[i,0], vecs[i,0]]
            y = [pb[i,1], vecs[i,1]]
            plt.plot(x, y, 'k')
            plt.text(pb[i,0], pb[i,1], str(i), color='red')

        plt.axis('equal')
        plt.show()


