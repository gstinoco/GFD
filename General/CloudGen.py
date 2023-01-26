import dmsh
import numpy as np
from scipy.io import loadmat
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

def CreateCloud(xb,yb):
    dist = 0
    # For holes, the coordinates and the radium
    xc = 0.5
    yc = 0.5
    ra = 0.05

    # Find the maximum distance between the boundary nodes.
    m = len(xb)
    for i in range(m-1):
        d = np.sqrt((xb[i] - xb[i+1])**2 + (yb[i] - yb[i+1])**2)
        dist = max(dist,d)
    d = np.sqrt((xb[m-1] - xb[0])**2 + (yb[m-1] - yb[0])**2)
    dist = max(dist,d)

    # Create the Triangulation
    pb = np.hstack([xb,yb])
    geo = dmsh.Polygon(pb) - dmsh.Circle([xc,yc],ra)
    X, cells = dmsh.generate(geo, dist)

    # Create a polygon
    poly = Polygon(pb).buffer(-0.005)
    circ = Point(xc,yc).buffer(ra).buffer(0.005)

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
        elif i.within(circ):
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

    color = ['blue' if x == 0 else 'red' for x in X[:,2]]
    plt.scatter(X[:,0], X[:,1], c=color)

    return X, cells

def GridToCloud(x,y):
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
    X, cells = CreateCloud(xb,yb)

    return X, cells