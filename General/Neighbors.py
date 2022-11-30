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
#   November, 2022.

# Routines to find neighboring nodes in a triangulation or in a cloud of points.

import numpy as np

def Neighbors_Tri(p, tt, nvec):
    # Neighbors_Tri
    # Routine to find the neighbor nodes in a triangulation.
    #
    # Input parameters
    #   p           m x 2           double          Array with the coordinates of the nodes.
    #   tt          n x 3           double          Array with the correspondence of the n triangles.
    #   nvec                        integer         Maximum number of neighbors.
    # 
    # Output parameters
    #   vec         m x nvec        double          Array with matching neighbors of each node.

    # Variable initizalization
    m   = len(p[:,0])                                                               # The size if the triangulation is obtained.
    vec = np.zeros([m, nvec*2])                                                     # The array for the neighbors is initialized.

    # Neighbor search
    for i in np.arange(m):                                                          # For each of the nodes.
        kn   = np.argwhere(tt == i+1)                                               # Search in which triangles the node appears.
        vec2 = np.setdiff1d(tt[kn[:,0]], i+1)                                       # Neighbors are stored inside vec2.
        vec2 = np.vstack([vec2])                                                    # Convert vec2 to a column.
        nvec = sum(vec2[0,:] != 0)                                                  # The number of neighbors of the node is calculated.
        for j in np.arange(nvec):                                                   # For each of the nodes.
            vec[i,j] = vec2[0,j]                                                    # Neighbors are saved.
            
    return vec

def Neighbors_Tri_Adv(p, nvec):
    # Neighbors_Tri_Adv
    # Routine to find the neighbor nodes for the Advection Equation Routines.
    #
    # Input parameters
    #   p           m x 2           double          Array with the coordinates of the nodes.
    #   nvec                        integer         Maximum number of neighbors.
    # 
    # Output parameters
    #   vec         m x nvec        double          Array with matching neighbors of each node.

    # Variable initizalization
    m   = len(p[:,0])                                                               # The size if the triangulation is obtained.
    vec = np.zeros([m, nvec*2])                                                     # The array for the neighbors is initialized.

    # Neighbor search
    for i in np.arange(m):                                                          # For each of the nodes.
        kn   = np.argwhere(tt == i+1)                                               # Search in which triangles the node appears.
        vec2 = np.setdiff1d(tt[kn[:,0]], i+1)                                       # Neighbors are stored inside vec2.
        vec2 = np.vstack([vec2])                                                    # Convert vec2 to a column.
        nvec = sum(vec2[0,:] != 0)                                                  # The number of neighbors of the node is calculated.
        for j in np.arange(nvec):                                                   # For each of the nodes.
            vec[i,j] = vec2[0,j]                                                    # Neighbors are saved.
            
    return vec