# %% [markdown]
# # Graficadores
# Aquí se definen los diferentes graficadores que se utilizarán en los diferentes códigos de Python con la finalidad de no tener que definirlos en cada código por separado.

# %%
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from mpl_toolkits import mplot3d
import cv2

# %%
def graph_mesh_static(x, y, u_ap, u_ex):
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    
    ax1.set_title('Aproximación')
    ax1.plot_surface(x, y, u_ap)
    
    ax2.set_title('Solución Exacta')
    ax2.plot_surface(x, y, u_ex)
    
    plt.show()

# %%
def graph_mesh_transient(x, y, u_ap, u_ex):
    t = len(u_ex[0,0,:])
    step = math.ceil(t/1000)
    min  = u_ex.min()
    max  = u_ex.max()
    T    = np.linspace(0,1,t)
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    
    for k in range(o,t,step):
        tin = float(T[k])
        plt.suptitle('Solución al tiempo t = %1.3f seg.' %tin)
        
        ax1.set_title('Aproximación')
        ax1.plot_surface(x, y, u_ap[:,:,k])
    
        ax2.set_title('Solución Exacta')
        ax2.plot_surface(x, y, u_ex[:,:,k])
    
        plt.show()
        
        if k < t-step:
            ax1.cla()
            ax2.cla()

# %%
def graph_cloud_static(p, u_ap, u_ex):
    min  = u_ex.min()
    max  = u_ex.max()

    fig  = plt.figure(figsize =(15, 5))
    
    ax1  = fig.add_subplot(1,2,1, projection='3d')
    ax2  = fig.add_subplot(1,2,2, projection='3d')
    
    plt.suptitle('Ecuación de Poisson')
    
    tri1 = ax1.plot_trisurf(p[:,0], p[:,1], u_ap[:], cmap='viridis', edgecolor='none')
    tri2 = ax2.plot_trisurf(p[:,0], p[:,1], u_ex[:], cmap='viridis', edgecolor='none')
    
    ax1.set_zlim([min, max])
    ax1.set_title('Solución Aproximada')
    
    ax2.set_zlim([min, max])
    ax2.set_title('Solución Exacta')
    
    fig.canvas.draw()

# %%
def graph_cloud_transient(u_ap, u_ex):
    t = len(u_ex[0,:])
    step = math.ceil(t/1000)
    min  = u_ex.min()
    max  = u_ex.max()
    T    = np.linspace(0,1,t)
    fig  = plt.figure(figsize =(15, 5))
    ax1  = fig.add_subplot(1,2,1, projection='3d')
    ax2  = fig.add_subplot(1,2,2, projection='3d')

    for k in range(0,t,step):
        tin = float(T[k])
        plt.suptitle('Solución al tiempo t = %1.3f seg.' %tin)
        tri1 = ax1.plot_trisurf(p[:,0], p[:,1], u_ap[:,k])
        tri2 = ax2.plot_trisurf(p[:,0], p[:,1], u_ex[:,k])
        ax1.set_zlim([min, max])
        ax1.set_title('Solución Aproximada')
        ax2.set_zlim([min, max])
        ax2.set_title('Solución Exacta')
        fig.canvas.draw()

        if k < t-step:
            ax1.cla()
            ax2.cla()

# %%
def graph_cloud_transient_vid(u_ap, u_ex, nube):
    t = len(u_ex[0,:])
    step = math.ceil(t/1000)
    min  = u_ex.min()
    max  = u_ex.max()
    T    = np.linspace(0,1,t)
    fig  = plt.figure(figsize =(15, 5))
    ax1  = fig.add_subplot(1,2,1, projection='3d')
    ax2  = fig.add_subplot(1,2,2, projection='3d')
    nom  = nube + '.avi'

    for k in range(0,t,step):
        tin = float(T[k])
        plt.suptitle('Solución al tiempo t = %1.3f seg.' %tin)
        tri1 = ax1.plot_trisurf(p[:,0], p[:,1], u_ap[:,k])
        tri2 = ax2.plot_trisurf(p[:,0], p[:,1], u_ex[:,k])
        ax1.set_zlim([min, max])
        ax1.set_title('Solución Aproximada')
        ax2.set_zlim([min, max])
        ax2.set_title('Solución Exacta')
        fig.canvas.draw()
  
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        ima = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  
        if k == 0:
            height, width, layers = ima.shape
            size = (width,height)
            out = cv2.VideoWriter(nom,cv2.VideoWriter_fourcc(*'DIVX'), 25, size)
  
        out.write(ima)

        if k < t-step:
            ax1.cla()
            ax2.cla()
    
    out.release()


