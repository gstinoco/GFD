from scipy.io import loadmat
import General.Graph as Graph

# Names of the regions
regions = ['CAB','CUA','CUI','DOW','ENG','GIB','HAB','MIC','PAT','ZIR']

# Sizes of the clouds
sizes = ['1', '2', '3']

for reg in regions:
    regi = reg
    ss = 1
    for me in sizes:
        size = me

        mat = loadmat('Regions/Meshes/' + regi + '_' + size + '.mat')
        x  = mat['x']
        y  = mat['y']

        mat = loadmat('Regions/Holes/' + regi + '_' + size + '.mat')
        p   = mat['p']
        tt  = mat['tt']
        if tt.min() == 1:
            tt -= 1
        nom = regi + '_' + str(ss)
        
        Graph.GraphMesh(x, y, nom)
        Graph.GraphCloud(p, tt, nom)
        Graph.GraphTriangles(p, tt, nom)
        Graph.GraphHoles(p, tt, nom)

        ss += 1