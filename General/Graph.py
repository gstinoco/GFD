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

# Graficadores
# Aquí se definen los diferentes graficadores que se utilizarán en los diferentes códigos de Python con la finalidad de no tener que definirlos en cada código por separado.

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from mpl_toolkits import mplot3d
import matplotlib.tri as mtri
import cv2

def Mesh_Static(x, y, u_ap, u_ex):
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    plt.rcParams["figure.figsize"] = (10,5)
    
    ax1.set_title('Approximation')
    ax1.plot_surface(x, y, u_ap)
    
    ax2.set_title('Theoretical Solution')
    ax2.plot_surface(x, y, u_ex)
    plt.show()

def Cloud_Static(p, u_ap, u_ex):
    min  = u_ex.min()
    max  = u_ex.max()

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw = {"projection": "3d"})

    plt.rcParams["figure.figsize"] = (10,5)
    
    ax1.scatter(p[:,0], p[:,1], u_ap[:])
    ax1.set_zlim([min, max])
    ax1.set_title('Approximation')
    
    ax2.scatter(p[:,0], p[:,1], u_ex[:])
    ax2.set_zlim([min, max])
    ax2.set_title('Theoretical Solution')

    plt.show()

def Mesh_Transient(x, y, u_ap, u_ex):
    t = len(u_ex[0,0,:])
    step = math.ceil(t/100)
    min  = u_ex.min()
    max  = u_ex.max()
    T    = np.linspace(0,1,t)

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw = {"projection": "3d"})
    plt.rcParams["figure.figsize"] = (10,5)
    
    for k in range(0,t,step):
        ax1.clear()
        ax2.clear()
        tin = float(T[k])
        plt.suptitle('Solution at t = %1.3f s.' %tin)
        
        ax1.plot_surface(x, y, u_ap[:,:,k])
        ax1.set_zlim([min, max])
        ax1.set_title('Approximation')

        ax2.plot_surface(x, y, u_ex[:,:,k])
        ax2.set_zlim([min, max])
        ax2.set_title('Theoretical Solution')

        plt.pause(0.1)

def Cloud_Transient(p, u_ap, u_ex):
    t = len(u_ex[0,:])
    step = math.ceil(t/100)
    min  = u_ex.min()
    max  = u_ex.max()
    T    = np.linspace(0,1,t)
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw = {"projection": "3d"})
    plt.rcParams["figure.figsize"] = (10,5)

    for k in range(0,t,step):
        ax1.clear()
        ax2.clear()
        tin = float(T[k])
        plt.suptitle('Solution at t = %1.3f s.' %tin)

        ax1.scatter(p[:,0], p[:,1], u_ap[:, k])
        ax1.set_zlim([min, max])
        ax1.set_title('Approximation')
        
        ax2.scatter(p[:,0], p[:,1], u_ex[:, k])
        ax2.set_zlim([min, max])
        ax2.set_title('Theoretical Solution')

        plt.pause(0.1)
        
def Cloud_Transient_Vid(p, u_ap, u_ex, nube):
    t = len(u_ex[0,:])
    step = math.ceil(t/1000)
    min  = u_ex.min()
    max  = u_ex.max()
    T    = np.linspace(0,1,t)
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw = {"projection": "3d"})
    plt.rcParams["figure.figsize"] = (10,5)
    nom  = nube + '.avi'

    for k in range(0,t,step):
        ax1.clear()
        ax2.clear()
        tin = float(T[k])
        plt.suptitle('Solution at t = %1.3f s.' %tin)
        
        ax1.scatter(p[:,0], p[:,1], u_ap[:, k])
        ax1.set_zlim([min, max])
        ax1.set_title('Approximation')
        
        ax2.scatter(p[:,0], p[:,1], u_ex[:, k])
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
    
    out.release()

def Error(er):
    t = len(er)
    T = np.linspace(0,1,t);
    plt.plot(T, er)
    plt.ylabel('Error')
    plt.xlabel('Time step')
    plt.title('Cuadratic Mean Error')
    plt.show()