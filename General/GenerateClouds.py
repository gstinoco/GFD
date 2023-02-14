from scipy.io import loadmat
from scipy.io import savemat
import CloudGen as CloudGen

# Region data is loaded.
regions = ['ZIR']#['CAB','CUA','CUI','DOW','ENG','GIB','HAB','MIC','PAT','SWA','ZIR']
sizes = ['21', '41','81']


## To generate the clouds

for reg in regions:
    regi = reg

    for me in sizes:
        mesh = me

        # All data is loaded from the file
        mat  = loadmat('Regions/Meshes/' + regi + mesh + '.mat')
        nom = 'Regions/Clouds/New/Holes/' + regi + mesh + '.mat'
        print('Trabajando en la malla ' + regi + mesh + '.')

        # The cloud is generated
        x  = mat['x']
        y  = mat['y']
        p, tt = CloudGen.GridToCloud(x,y)

        # The cloud is saved
        mdic = {"p": p, "tt": tt}
        savemat(nom, mdic)

        nom = regi + mesh
        CloudGen.GraphCloud(p, tt, nom)