from scipy.io import loadmat
from scipy.io import savemat
import General.CloudGen as CloudGen
import General.normals as normals
import matplotlib.pyplot as plt
import numpy as np


# Region data is loaded.
#regions = ['CAB','CUA','CUI','DOW','ENG','GIB','HAB','MIC','PAT','ZIR']
sizes = [1]

regions =['CUA']

## To generate the clouds

for reg in regions:
    regi = reg

    for me in sizes:
        mesh = str(me)

        # All data is loaded from the file
        mat  = loadmat('Regions/Meshes/' + regi + '_1.mat')
        nom = 'Regions/Clouds/' + regi + '_' + mesh + '_n.mat'
        print('Trabajando en la malla ' + regi + '_' + mesh + '.')

        # The cloud is generated
        x  = mat['x']
        y  = mat['y']
        p, tt = CloudGen.GridToCloud(x,y, holes = False, num = me)

        #p = CloudGen.OrdenNodes(p, b_interior = True)

        # The cloud is saved
        #mdic = {"p": p, "tt": tt}
        #savemat(nom, mdic)

        nom = regi + '_' + mesh
        CloudGen.GraphCloud(p, nom)

        #pb, vecs = normals.normals(p, b_interior = True)
        #print('Done')

        #plt.figure(figsize=(10,6))
        #plt.title('Vectors')
        #plt.scatter(pb[:,0], pb[:,1])
        #for i in np.arange(len(pb[:,0])):
        #    x = [pb[i,0], vecs[i,0]]
        #    y = [pb[i,1], vecs[i,1]]
        #    plt.plot(x, y, 'k')
        #    plt.text(pb[i,0], pb[i,1], str(i), color='red')

        #plt.axis('equal')
        #plt.show()
