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

# Graphics
# Some routines are defined in here in order to correctly Graph different kinds of results.

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import matplotlib.tri as mtri
import cv2
from mpl_toolkits import mplot3d
from matplotlib import cm

def Mesh_Static(x, y, u_ap, u_ex):
    min  = u_ex.min()
    max  = u_ex.max()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(8, 4))
    
    ax1.set_title('Approximation')
    ax1.set_zlim([min, max])
    ax1.plot_surface(x, y, u_ap, cmap=cm.coolwarm)
    
    ax2.set_title('Theoretical Solution')
    ax2.set_zlim([min, max])
    ax2.plot_surface(x, y, u_ex, cmap=cm.coolwarm)

    plt.show()

def Cloud_Static(p, tt, u_ap, u_ex):
    if tt.min() == 1:
        tt -= 1
    
    min  = u_ex.min()
    max  = u_ex.max()

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw = {"projection": "3d"}, figsize=(8, 4))
    
    ax1.plot_trisurf(p[:,0], p[:,1], u_ap[:], triangles=tt, cmap=cm.coolwarm)
    ax1.set_zlim([min, max])
    ax1.set_title('Approximation')
    
    ax2.plot_trisurf(p[:,0], p[:,1], u_ex[:], triangles=tt, cmap=cm.coolwarm)
    ax2.set_zlim([min, max])
    ax2.set_title('Theoretical Solution')

    plt.show()

def Mesh_Transient(x, y, u_ap, u_ex):
    t    = len(u_ex[0,0,:])
    step = math.ceil(t/50)
    min  = u_ex.min()
    max  = u_ex.max()
    T    = np.linspace(0,1,t)

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw = {"projection": "3d"}, figsize=(8, 4))
    
    for k in range(0,t,step):
        ax1.clear()
        ax2.clear()
        tin = float(T[k])
        plt.suptitle('Solution at t = %1.3f s.' %tin)
        
        ax1.plot_surface(x, y, u_ap[:,:,k], cmap=cm.coolwarm)
        ax1.set_zlim([min, max])
        ax1.set_title('Approximation')

        ax2.plot_surface(x, y, u_ex[:,:,k], cmap=cm.coolwarm)
        ax2.set_zlim([min, max])
        ax2.set_title('Theoretical Solution')

        plt.pause(0.01)
    
    ax1.clear()
    ax2.clear()
    tin = float(T[t-1])
    plt.suptitle('Solution at t = %1.3f s.' %tin)
    
    ax1.plot_surface(x, y, u_ap[:,:,t-1], cmap=cm.coolwarm)
    ax1.set_zlim([min, max])
    ax1.set_title('Approximation')

    ax2.plot_surface(x, y, u_ex[:,:,t-1], cmap=cm.coolwarm)
    ax2.set_zlim([min, max])
    ax2.set_title('Theoretical Solution')

    plt.pause(0.1)

def Cloud_Transient(p, tt, u_ap, u_ex):
    if tt.min() == 1:
        tt -= 1
    t    = len(u_ex[0,:])
    step = math.ceil(t/50)
    min  = u_ex.min()
    max  = u_ex.max()
    T    = np.linspace(0,1,t)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw = {"projection": "3d"}, figsize=(8, 4))

    for k in range(0,t,step):
        ax1.clear()
        ax2.clear()
        tin = float(T[k])
        plt.suptitle('Solution at t = %1.3f s.' %tin)

        ax1.plot_trisurf(p[:,0], p[:,1], u_ap[:,k], triangles=tt, cmap=cm.coolwarm)
        ax1.set_zlim([min, max])
        ax1.set_title('Approximation')
        
        ax2.plot_trisurf(p[:,0], p[:,1], u_ex[:,k], triangles=tt, cmap=cm.coolwarm)
        ax2.set_zlim([min, max])
        ax2.set_title('Theoretical Solution')

        plt.pause(0.01)

    ax1.clear()
    ax2.clear()
    tin = float(T[t-1])
    plt.suptitle('Solution at t = %1.3f s.' %tin)

    ax1.plot_trisurf(p[:,0], p[:,1], u_ap[:,t-1], triangles=tt, cmap=cm.coolwarm)
    ax1.set_zlim([min, max])
    ax1.set_title('Approximation')
    
    ax2.plot_trisurf(p[:,0], p[:,1], u_ex[:,t-1], triangles=tt, cmap=cm.coolwarm)
    ax2.set_zlim([min, max])
    ax2.set_title('Theoretical Solution')

    plt.pause(0.1)
        
def Cloud_Transient_Vid(p, tt, u_ap, u_ex, nam):
    if tt.min() == 1:
        tt -= 1
    t    = len(u_ex[0,:])
    step = math.ceil(t/1000)
    min  = u_ex.min()
    max  = u_ex.max()
    T    = np.linspace(0,1,t)
    nom  = nam + '.avi'

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw = {"projection": "3d"}, figsize=(8, 4))

    for k in range(0,t,step):
        ax1.clear()
        ax2.clear()
        tin = float(T[k])
        plt.suptitle('Solution at t = %1.3f s.' %tin)

        ax1.plot_trisurf(p[:,0], p[:,1], u_ap[:,k], triangles=tt, cmap=cm.coolwarm)
        ax1.set_zlim([min, max])
        ax1.set_title('Approximation')
        
        ax2.plot_trisurf(p[:,0], p[:,1], u_ex[:,k], triangles=tt, cmap=cm.coolwarm)
        ax2.set_zlim([min, max])
        ax2.set_title('Theoretical Solution')

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        ima = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  
        if k == 0:
            height, width, layers = ima.shape
            size = (width,height)
            out = cv2.VideoWriter(nom,cv2.VideoWriter_fourcc(*'DIVX'), 25, size)
  
        out.write(ima)
        plt.pause(0.1)
    
    ax1.clear()
    ax2.clear()
    tin = float(T[t-1])
    plt.suptitle('Solution at t = %1.3f s.' %tin)

    ax1.plot_trisurf(p[:,0], p[:,1], u_ap[:,t-1], triangles=tt, cmap=cm.coolwarm)
    ax1.set_zlim([min, max])
    ax1.set_title('Approximation')
    
    ax2.plot_trisurf(p[:,0], p[:,1], u_ex[:,t-1], triangles=tt, cmap=cm.coolwarm)
    ax2.set_zlim([min, max])
    ax2.set_title('Theoretical Solution')

    out.release()

def Error(er):
    t = len(er)
    T = np.linspace(0,1,t)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(T, er)
    ax1.set_title('Linear')
    ax1.set(xlabel='Time Step', ylabel='Error')

    ax2.semilogy(T, er)
    ax2.set_title('Logarithmic')
    ax2.set(xlabel='Time Step', ylabel='Error')

    plt.suptitle('Quadratic Mean Error')
    plt.show()