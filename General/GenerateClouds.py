from scipy.io import loadmat
from scipy.io import savemat
import CloudGen as CloudGen

# Region data is loaded.
regions = ['CAB','CUA','CUI','DOW','ENG','GIB','HAB','MIC','PAT','SWA','ZIR']
sizes = ['21', '41','81']


## To generate the clouds

#for reg in regions:
#    region = reg
#
#    for me in sizes:
#        mesh = me
#
#        # All data is loaded from the file
#        mat  = loadmat('Regions/Meshes/' + region + mesh + '.mat')
#        nom = 'Regions/Clouds/New/' + region + mesh + '.mat'
#        print('Trabajando en la malla ' + region + mesh + '.')
#
#        # The cloud is generated
#        x  = mat['x']
#        y  = mat['y']
#        p, tt = CloudGen.GridToCloud(x,y)
#
#        # The cloud is saved
#        mdic = {"p": p, "tt": tt}
#        savemat(nom, mdic)
#
#        nom = region + mesh
#        CloudGen.GraphCloud(p, tt, nom)

## To graph the clouds:
for reg in regions:
    region = reg

    for me in sizes:
        mesh = me

        # All data is loaded from the file
        nom  = 'Regions/Clouds/New/' + region + mesh + '.mat'
        mat  = loadmat(nom)

        p   = mat['p']
        tt  = mat['tt']
        
#
#        # The cloud is generated
#        x  = mat['x']
#        y  = mat['y']
#        p, tt = CloudGen.GridToCloud(x,y)
#
#        # The cloud is saved
#        mdic = {"p": p, "tt": tt}
#        savemat(nom, mdic)
#
        nom = region + mesh
        CloudGen.GraphCloud(p, tt, nom)