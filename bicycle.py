import numpy as np
import scipy



class Bicycle:
    def __init__(self,lr=1.8,v=20.0,x_range = np.array([[-20.0,20.0] , [-20.0,20.0] , [-0.5,0.5] , [-0.1,0.1]]) ,u_range=np.array([[-0.5,0.5]])):
        self.lr = lr
        self.v = v
        self.x_range = x_range
        self.u_range = u_range

        self.x_dim = self.x_range.shape[0]
        self.u_dim = self.u_range.shape[0]

    def diff(self,x,u):
        return np.stack((self.v*np.cos(x[...,2] + u[...,0]) , self.v*np.sin(x[...,2] + u[...,0]) , self.v/self.lr*np.sin(u[...,0])),axis=-1)

    def step(self,x,u,dt=0.1):
        new_x = x + dt*self.diff(x,u)
        return new_x
    
    def step_op(self,x,u,op,dt=0.1):
        new_x = x + dt*np.array([self.v*op.cos(x[2] + u[0]) , self.v*op.sin(x[2] + u[0]) , self.v/self.lr*op.sin(u[0])])
        return new_x
        
    def step_ode(self,x,u,dt=0.1,N=50):
        diff_ode = lambda _,x: np.concatenate(self.diff(x[:self.x_dim],x[self.x_dim:]),np.zeros((self.u_dim,)))
        return scipy.integrate.solve_ivp(diff_ode,(0,dt),np.concatenate((x,u)),t_eval= np.linspace(0,dt,N))['y']
