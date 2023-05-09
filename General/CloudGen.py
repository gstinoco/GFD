import dmsh
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from scipy.spatial import Delaunay

def CreateCloud(xb, yb):
    dist = 0

    # Find the maximum distance between the boundary nodes.
    m = len(xb)
    for i in range(m):
        d = np.sqrt((xb[i] - xb[i-1])**2 + (yb[i] - yb[i-1])**2)
        dist = max(dist,d)
        
    # Create the Triangulation
    pb = np.hstack([xb,yb])
    geo = dmsh.Polygon(pb)
    X, cells = dmsh.generate(geo, dist)

    # Create a polygon
    poly = Polygon(pb).buffer(-dist/4)

    points = []
    for point in X:
        points.append(Point(point[0], point[1]))

    # Check if point os within the buffer zones
    pbx = []
    pby = []
    for i in points:
        if i.within(poly) == False:
            pbx.append([i.x])
            pby.append([i.y])
    
    bond = np.hstack([np.array(pbx),np.array(pby)])

    m    = len(X[:,0])
    n    = len(bond[:,0])
    X    = np.hstack([X,np.zeros([m,1])])

    for i in range(m):
        for j in range(n):
            if X[i,0] == bond[j,0] and X[i,1] == bond[j,1]:
                X[i,2] = 1

    return X, cells

def CreateCloud_Holes(xb, yb):
    dist = 0

    # Find the maximum distance between the boundary nodes.
    m = len(xb)
    for i in range(m):
        d = np.sqrt((xb[i] - xb[i-1])**2 + (yb[i] - yb[i-1])**2)
        dist = max(dist,d)
        
    # For holes, the coordinates and the radium
    x1 = 0.5
    y1 = 0.3
    x2 = 3#1
    y2 = 3#0.8
    x3 = 3#1.5
    y3 = 3#0.5
    ra = 0.05
    # Create the Triangulation
    pb = np.hstack([xb,yb])
    geo = dmsh.Polygon(pb) - dmsh.Circle([x1,y1],ra) - dmsh.Circle([x2,y2],ra) - dmsh.Circle([x3,y3],ra)
    X, cells = dmsh.generate(geo, dist)

    # Create a polygon
    poly = Polygon(pb).buffer(-dist/4)
    circ1 = Point(x1,y1).buffer(ra).buffer(dist/4)
    circ2 = Point(x2,y2).buffer(ra).buffer(dist/4)
    circ3 = Point(x3,y3).buffer(ra).buffer(dist/4)

    points = []
    for point in X:
        points.append(Point(point[0], point[1]))

    # Check if point os within the buffer zones
    pbx_o = []
    pby_o = []
    pbx_i = []
    pby_i = []
    for i in points:
        if i.within(poly) == False:
            pbx_o.append([i.x])
            pby_o.append([i.y])
        elif i.within(circ1) == True:
            pbx_i.append([i.x])
            pby_i.append([i.y])
        elif i.within(circ2):
            pbx_i.append([i.x])
            pby_i.append([i.y])
        elif i.within(circ3):
            pbx_i.append([i.x])
            pby_i.append([i.y])
        
    bond_o = np.hstack([np.array(pbx_o),np.array(pby_o)])
    bond_i = np.hstack([np.array(pbx_i),np.array(pby_i)])

    m      = len(X[:,0])
    n_o    = len(bond_o[:,0])
    n_i    = len(bond_i[:,0])
    X      = np.hstack([X,np.zeros([m,1])])

    for i in range(m):
        for j in range(n_o):
            if X[i,0] == bond_o[j,0] and X[i,1] == bond_o[j,1]:
                X[i,2] = 1
        for j in range(n_i):
            if X[i,0] == bond_i[j,0] and X[i,1] == bond_i[j,1]:
                X[i,2] = 2

    return X, cells

def GridToCloud(x, y, holes = False):
    # First, the region is scaled to fit in [0,1]X[0,1].
    mm = max(x.max(), y.max())
    x  = x-x.min()
    y  = y-y.min()
    x  = x/mm
    y  = y/mm

    # The dimensions of the Mesh.
    m    = len(x[:,0])
    n    = len(x[:,1])

    # The boundaries of the Mesh.
    xb = np.vstack([np.array([x[m-1,:]]).transpose(), np.flip([x[1:m-1,n-1]]).transpose(), np.flip([x[0,:]]).transpose(), np.array([x[1:m-1,0]]).transpose()])
    yb = np.vstack([np.array([y[m-1,:]]).transpose(), np.flip([y[1:m-1,n-1]]).transpose(), np.flip([y[0,:]]).transpose(), np.array([y[1:m-1,0]]).transpose()])

    # The cloud is created with the boundary.
    if holes == False:
        X, cells = CreateCloud(xb, yb)
    else:
        X, cells = CreateCloud_Holes(xb, yb)

    return X, cells

def GraphCloud(X, nom):
    nomm  = nom + '.png'
    color = ['blue' if x == 0 else 'red' for x in X[:,2]]
    plt.rcParams["figure.figsize"] = (12,12)
    plt.scatter(X[:,0], X[:,1], c=color)
    plt.title(nom + 'Cloud')
    #plt.show()
    plt.savefig(nomm)
    plt.close()

def OrdenNodes(X, b_interior = False):
    """
    Reordena los nodos de una triangulación para que los nodos de frontera se coloquen primero en sentido antihorario y
    después se agreguen los nodos interiores. Esto facilita el cálculo de las normales en las fronteras.

    Args:
    -----------
    X : numpy.ndarray
        Matriz de tamaño (m, 3) que contiene las coordenadas de los nodos de la triangulación y una bandera que indica si el nodo es de frontera o no. Las columnas de X corresponden a:
        0: Coordenadas en x.
        1: Coordenadas en y.
        2: 0 para nodos interiores y 1 para nodos de frontera.

    Returns:
    --------
    X : numpy.ndarray
        Una matriz de tamaño (m, 3) que contiene las mismas coordenadas que la matriz de entrada X, pero reordenadas de tal manera que primero aparecen los nodos de frontera y después los interiores.
    """
    if max(X[:,2]) == 1:
        b_interior = False
    elif max(X[:,2] == 2):
        b_interior = True

    idx_inter   = np.where(X[:, 2] == 0)[0]                                             # Índices de los nodos interiores
    idx_bound_o = np.where(X[:, 2] == 1)[0]                                             # Índices de los nodos de frontera exterior
    inter       = X[idx_inter, :]                                                       # Subconjunto de nodos interiores
    bound_o     = X[idx_bound_o, :]                                                     # Subconjunto de nodos de frontera exterior
    
    center      = np.mean(bound_o[:, 0:2], axis = 0)                                    # Centro de masa de los nodos de frontera exterior
    angles      = np.arctan2(bound_o[:, 1] - center[1], bound_o[:, 0] - center[0])      # Ángulos de los nodos respecto al centro de masa
    idx_bound_o = np.argsort(angles)                                                    # Índices ordenados según los ángulos
    bound_o     = bound_o[idx_bound_o, :]                                               # Nodos de frontera exterior ordenados antihorario

    if b_interior == True:
        idx_bound_i = np.where(X[:, 2] == 2)[0]                                         # Índices de los nodos de frontera interior
        bound_i     = X[idx_bound_i, :]                                                 # Subconjunto de nodos de frontera interior
        center      = np.mean(bound_i[:, 0:2], axis = 0)                                # Centro de masa de los nodos de frontera interior
        angles      = np.arctan2(bound_i[:, 1] - center[1], bound_i[:, 0] - center[0])  # Ángulos de los nodos respecto al centro de masa
        idx_bound_i = np.argsort(-angles)                                               # Índices ordenados según los ángulos
        bound_i     = bound_i[idx_bound_i, :]                                           # Nodos de frontera interior ordenados horario
        X           = np.concatenate([bound_o, bound_i, inter])                         # Concatenamos los nodos de frontera e interiores
        
    else:
        X           = np.concatenate([bound_o, inter])                                  # Concatenamos los nodos de frontera e interiores

    return X
