"""
All the codes presented below were developed by:
    Dr. Gerardo Tinoco Guerrero
    Universidad Michoacana de San Nicolás de Hidalgo
    gerardo.tinoco@umich.mx

With the funding of:
    National Council of Science and Technology, CONACyT (Consejo Nacional de Ciencia y Tecnología, CONACyT). México.
    Coordination of Scientific Research, CIC-UMSNH (Coordinación de la Investigación Científica de la Universidad Michoacana de San Nicolás de Hidalgo, CIC-UMSNH). México
    Aula CIMNE-Morelia. México

Date:
    April, 2023.

Last Modification:
    March, 2023.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class GFD_Transient:
    def __init__(self, points, triangulation, time_interval, time_steps):
        self.p    = points
        self.tt   = triangulation
        self.t    = time_steps
        self.m    = len(points[:,0])
        self.L    = np.vstack([[0], [0], [0], [0], [0]])
        self.K    = np.zeros([self.m,self.m])
        self.T    = np.linspace(time_interval[0],time_interval[1],time_steps)
        self.dt   = self.T[1] - self.T[0]
        self.u_ap = np.zeros([self.m,time_steps])
        self.u_ex = np.zeros([self.m,time_steps])
        
    
    def Boundary(self, f, coefficient):
        for k in np.arange(self.t):
            for i in np.arange(self.m):
                if self.p[i,2] == 1:
                    self.u_ap[i,k] = f(self.p[i, 0], self.p[i, 1], self.T[k], coefficient)
    
    def Initial(self, f, coefficient):
        for i in np.arange(self.m):
            self.u_ap[i,0] = f(self.p[i, 0], self.p[i, 1], self.T[0], coefficient)
    
    def Neighbors_Triangulation(self, nvec):
        self.vec = np.zeros([self.m, nvec], dtype=int)-1
        for i in np.arange(self.m):
            kn    = np.argwhere(self.tt == i)
            vec2  = np.setdiff1d(self.tt[kn[:,0]], i)
            vec2  = np.vstack([vec2])
            nvec2 = sum(vec2[0,:] != -1)
            nnvec = np.minimum(nvec, nvec2)
            for j in np.arange(nnvec):
                self.vec[i,j] = vec2[0,j]
    
    def Neighbors_Clouds(self, nvec):
        self.vec = np.zeros([self.m, nvec], dtype=int)-1
        dmin = np.zeros([self.m,1])+1
        for i in np.arange(self.m):
            x    = p[i,0]
            y    = p[i,1]
            for j in np.arange(self.m):
                if i != j:
                    x1 = p[j,0]
                    y1 = p[j,1]
                    d  = np.sqrt((x - x1)**2 + (y - y1)**2)
                    dmin[i] = min(dmin[i],d)
        dist = (3/2)*max(max(dmin))

        for i in np.arange(self.m):
            x    = self.p[i,0]
            y    = self.p[i,1]
            temp = 0
            for j in np.arange(self.m):
                if i != j:
                    x1 = self.p[j,0]
                    y1 = self.p[j,1]
                    d  = np.sqrt((x - x1)**2 + (y - y1)**2)
                    if d < dist:
                        if temp < nvec:
                            self.vec[i,temp] = j
                            temp            += 1
                        else:
                            x2 = p[self.vec[i,:], 0]
                            y2 = p[self.vec[i,:], 1]
                            d2 = np.sqrt((x - x2)**2 + (y - y2)**2)
                            I  = np.argmax(d2)
                            if d < d2[I]:
                                self.vec[i,I] = j

    def Operator(self, equation, coefficient):
        if equation == 'Advection':
            self.L = np.vstack([[-self.dt*coefficient[0]], [-self.dt*coefficient[1]], [0], [0], [0]])
        if equation == 'Diffusion':
            self.L = np.vstack([[0], [0], [2*coefficient*self.dt], [0], [2*coefficient*self.dt]])
        if equation == 'Advection-Diffusion':
            self.L = np.vstack([[-self.dt*coefficient[0]], [-self.dt*coefficient[1]], [2*coefficient[2]*self.dt], [0], [2*coefficient[2]*self.dt]])

    def Gammas(self):
        nvec  = len(self.vec[0,:])
        
        for i in np.arange(self.m):
            if self.p[i,2] == 0:
                nvec = sum(self.vec[i,:] != -1)
                dx   = np.zeros([nvec])
                dy   = np.zeros([nvec])
                for j in np.arange(nvec):
                    vec1  = int(self.vec[i, j])
                    dx[j] = self.p[vec1, 0] - self.p[i,0]
                    dy[j] = self.p[vec1, 1] - self.p[i,1]
                M     = np.vstack([[dx], [dy], [dx**2], [dx*dy], [dy**2]])
                M     = np.linalg.pinv(M)
                YY    = M@self.L
                Gamma = np.vstack([-sum(YY), YY]).transpose()
                self.K[i,i] = Gamma[0,0]
                for j in np.arange(nvec):
                    self.K[i, self.vec[i,j]] = Gamma[0,j+1]

            if self.p[i,2] == 1:
                self.K[i,i] = 0 
                for j in np.arange(nvec):
                    self.K[i, self.vec[i,j]] = 0
    
    def Solution(self, implicit, lam):
        if implicit == False:
            K2 = np.identity(self.m) + self.K
        else:
            K2 = np.linalg.pinv(np.identity(self.m) - (1-lam)*self.K)@(np.identity(self.m) + lam*self.K)

        for k in np.arange(1,self.t):
            un = K2@self.u_ap[:,k-1]
            for i in np.arange(self.m):
                if self.p[i,2] == 0:
                    self.u_ap[i,k] = un[i]
    
    def Exact(self, f, coefficient):
        for k in np.arange(self.t):
            for i in np.arange(self.m):
                self.u_ex[i,k] = f(self.p[i, 0], self.p[i, 1], self.T[k], coefficient)
    
    def Graph(self):
        if self.tt.min() == 1:
            self.tt -= 1
        step = int(np.ceil(self.t/50))
        min  = self.u_ex.min()
        max  = self.u_ex.max()
        fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw = {"projection": "3d"}, figsize=(8, 4))

        for k in np.arange(0,self.t,step):
            tin = float(self.T[k])
            plt.suptitle('Solution at t = %1.3f s.' %tin)

            ax1.plot_trisurf(self.p[:,0], self.p[:,1], self.u_ap[:,k], triangles=self.tt, cmap=cm.coolwarm)
            ax1.set_zlim([min, max])
            ax1.set_title('Approximation')
        
            ax2.plot_trisurf(self.p[:,0], self.p[:,1], self.u_ex[:,k], triangles=self.tt, cmap=cm.coolwarm)
            ax2.set_zlim([min, max])
            ax2.set_title('Theoretical Solution')

            plt.pause(0.01)
            ax1.clear()
            ax2.clear()

        tin = float(self.T[self.t-1])
        plt.suptitle('Solution at t = %1.3f s.' %tin)

        ax1.plot_trisurf(self.p[:,0], self.p[:,1], self.u_ap[:,self.t-1], triangles=self.tt, cmap=cm.coolwarm)
        ax1.set_zlim([min, max])
        ax1.set_title('Approximation')
    
        ax2.plot_trisurf(self.p[:,0], self.p[:,1], self.u_ex[:,self.t-1], triangles=self.tt, cmap=cm.coolwarm)
        ax2.set_zlim([min, max])
        ax2.set_title('Theoretical Solution')

        plt.pause(0.1)
    
    def Solver(self, equation, f, coefficient, triangulation = False, implicit = False, lam = 0.5):
        self.Boundary(f, coefficient)
        self.Initial(f, coefficient)
        if triangulation == True:
            self.Neighbors_Triangulation(8)
        else:
            self.Neighbors_Clouds(8)
        self.Operator(equation,coefficient)
        self.Gammas()
        self.Solution(implicit = implicit, lam = lam)
        self.Exact(f, coefficient)