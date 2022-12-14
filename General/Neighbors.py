# All the codes presented below were developed by:
#   Dr. Gerardo Tinoco Guerrero
#   Universidad Michoacana de San Nicolás de Hidalgo
#   gerardo.tinoco@umich.mx
#
# With the funding of:
#   National Council of Science and Technology, CONACyT (Consejo Nacional de Ciencia y Tecnología, CONACyT). México.
#   Coordination of Scientific Research, CIC-UMSNH (Coordinación de la Investigación Científica de la Universidad Michoacana de San Nicolás de Hidalgo, CIC-UMSNH). México
#   Aula CIMNE Morelia. México
#
# Date:
#   November, 2022.
#
# Last Modification:
#   December, 2022.

# Routines to find neighboring nodes in a triangulation or in a cloud of points.

import numpy as np
import math

def Triangulation(p, tt, nvec):
    # Triangulation
    # Routine to find the neighbor nodes in a triangulation generated with dmsh on Python.
    #
    # Input parameters
    #   p           m x 2           double          Array with the coordinates of the nodes.
    #   tt          n x 3           double          Array with the correspondence of the n triangles.
    #   nvec                        integer         Maximum number of neighbors.
    # 
    # Output parameters
    #   vec         m x nvec        double          Array with matching neighbors of each node.

    # Variable initialization
    m   = len(p[:,0])                                                               # The size if the triangulation is obtained.
    vec = np.zeros([m, nvec])-1                                                     # The array for the neighbors is initialized.

    # Neighbor search
    for i in np.arange(m):                                                          # For each of the nodes.
        kn   = np.argwhere(tt == i)                                                 # Search in which triangles the node appears.
        vec2 = np.setdiff1d(tt[kn[:,0]], i)                                         # Neighbors are stored inside vec2.
        vec2 = np.vstack([vec2])                                                    # Convert vec2 to a column.
        nvec = sum(vec2[0,:] != -1)                                                 # The number of neighbors of the node is calculated.
        for j in np.arange(nvec):                                                   # For each of the nodes.
            vec[i,j] = vec2[0,j]                                                    # Neighbors are saved.   
    return vec

def Cloud(p, pb, nvec):
    # Clouds
    # Routine to find the neighbor nodes in a cloud of points generated with dmsh on Python.
    #
    # Input parameters
    #   p           m x 2           double          Array with the coordinates of the nodes.
    #   pm          m x 2           double          Array with the coordinates of the boundary nodes.
    #   nvec                        integer         Maximum number of neighbors.
    # 
    # Output parameters
    #   vec         m x nvec        double          Array with matching neighbors of each node.

    # Variable initialization
    m    = len(p[:,0])                                                              # The size if the triangulation is obtained.
    vec  = np.zeros([m, nvec])-1                                                    # The array for the neighbors is initialized.
    dist = 0                                                                        # Maximum distance between the boundary nodes.

    # Search for the maximum distance between the boundary nodes
    xb = pb[:,0]                                                                    # x coordinates of the boundary nodes.
    yb = pb[:,1]                                                                    # y coordinates of the boundary nodes.
    m2 = len(xb)                                                                    # Total number of boundary nodes.
    for i in np.arange(m2-1):
        d    = math.sqrt((xb[i] - xb[i+1])**2 + (yb[i] - yb[i+1])**2)               # The distance between a boundary node and the next one.
        dist = max(dist,d)                                                          # Maximum distance search.
    d    = math.sqrt((xb[m2-1] - xb[0])**2 + (yb[m2-1] - yb[0])**2)                 # The distance between the last and the first boundary nodes.
    dist = (8/7)*max(dist,d)                                                        # Maximum distance search.

    # Search of the neighbor nodes
    for i in np.arange(m):                                                          # For each of the nodes.
        x    = p[i,0]                                                               # x coordinate of the central node.
        y    = p[i,1]                                                               # y coordinate of the central node.
        temp = 0                                                                    # Temporal variable as a counter.
        for j in np.arange(m):                                                      # For all the interior nodes.
            if i != j:                                                              # Check that we are not working with the central node.
                x1 = p[j,0]                                                         # x coordinate of the possible neighbor.
                y1 = p[j,1]                                                         # y coordinate of the possible neighbor.
                d  = math.sqrt((x - x1)**2 + (y - y1)**2)                           # Distance from the possible neighbor to the central node.
                if d < dist:                                                        # If the distance is smaller or equal to the tolerance distance.
                    if temp <= nvec:                                                # If the number of neighbors is smaller than nvec.
                        vec[i,temp] = j                                             # Save the neighbor.
                        temp       += 1                                             # Increase the counter by 1.
                    else:                                                           # If the number of neighbors is greater than nvec.
                        x2 = p[vec[i,:],0]                                          # x coordinates of the current neighbor nodes.
                        y2 = p[vec[i,:],1]                                          # y coordinates of the current neighbor nodes.
                        d2 = math.sqrt((x - x2)**2 + (y - y2)**2)                   # The total distance from all the neighbors to the central node.
                        I  = np.argmax(d2)                                          # Look for the greatest distance.
                        if d < d2[I]:                                               # If the new node is closer than the farthest neighbor.
                            vec[i,I] = j                                            # The new neighbor replace the farthest one.
    return vec

def Cloud_Adv(p, pb, nvec, a, b):
    # Clouds_Adv
    # Routine to find the neighbor nodes in a cloud of points for the advection equation generated with dmsh on Python.
    #
    # Input parameters
    #   p           m x 2           double          Array with the coordinates of the nodes.
    #   pm          m x 2           double          Array with the coordinates of the boundary nodes.
    #   nvec                        integer         Maximum number of neighbors.
    #   a                           real            Advection velocity on the x direction.
    #   b                           real            Advection velocity on the y direction.
    # 
    # Output parameters
    #   vec         m x nvec        double          Array with matching neighbors of each node.

    # Variable initialization
    m    = len(p[:,0])                                                              # The size if the triangulation is obtained.
    vec  = np.zeros([m, nvec])-1                                                    # The array for the neighbors is initialized.
    dist = 0                                                                        # Maximum distance between the boundary nodes.

    # Search for the maximum distance between the boundary nodes
    xb = pb[:,0]                                                                    # x coordinates of the boundary nodes.
    yb = pb[:,1]                                                                    # y coordinates of the boundary nodes.
    m2 = len(xb)                                                                    # Total number of boundary nodes.
    for i in np.arange(m2-1):
        d    = math.sqrt((xb[i] - xb[i+1])**2 + (yb[i] - yb[i+1])**2)               # The distance between a boundary node and the next one.
        dist = max(dist,d)                                                          # Maximum distance search.
    d    = math.sqrt((xb[m2-1] - xb[0])**2 + (yb[m2-1] - yb[0])**2)                 # The distance between the last and the first boundary nodes.
    dist = (8/7)*max(dist,d)                                                        # Maximum distance search.

    # Search of the neighbor nodes
    for i in np.arange(m):                                                          # For each of the nodes.
        x    = p[i,0]                                                               # x coordinate of the central node.
        y    = p[i,1]                                                               # y coordinate of the central node.
        temp = 0                                                                    # Temporal variable as a counter.
        for j in np.arange(m):                                                      # For all the interior nodes.
            if i != j:                                                              # Check that we are not working with the central node.
                x1 = p[j,0]                                                         # x coordinate of the possible neighbor.
                y1 = p[j,1]                                                         # y coordinate of the possible neighbor.
                xt = x1 - x                                                         # The "new" origin on x.
                yt = y1 - y                                                         # The "new" origin on y.
                di = np.sign(np.dot([xt, yt],[a, b]))                               # Check the direction between (xt, yt) and (a,b).
                d  = math.sqrt((x - x1)**2 + (y - y1)**2)                           # Distance from the possible neighbor to the central node.
                if di == -1 and d < dist:                                           # If the direction is opposite and the distance is smaller or equal to the tolerance distance.
                    if temp <= nvec:                                                # If the number of neighbors is smaller than nvec.
                        vec[i,temp] = j                                             # Save the neighbor.
                        temp       += 1                                             # Increase the counter by 1.
                    else:                                                           # If the number of neighbors is greater than nvec.
                        x2 = p[vec[i,:],0]                                          # x coordinates of the current neighbor nodes.
                        y2 = p[vec[i,:],1]                                          # y coordinates of the current neighbor nodes.
                        d2 = math.sqrt((x - x2)**2 + (y - y2)**2)                   # The total distance from all the neighbors to the central node.
                        I  = np.argmax(d2)                                          # Look for the greatest distance.
                        if d < d2[I]:                                               # If the new node is closer than the farthest neighbor.
                            vec[i,I] = j                                            # The new neighbor replace the farthest one.
    return vec