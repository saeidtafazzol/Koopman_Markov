import numpy as np
import scipy


class Controlled_Vanderpole:
    def __init__(self,mu=1,x_range = np.array([[-3.0,3.0],[-6.0,6.0]]) ,u_range=np.array([[-5.0,5.0]])):
        self.mu = mu
        self.x_range = x_range
        self.u_range = u_range

        self.x_dim = self.x_range.shape[0]
        self.u_dim = self.u_range.shape[0]

    def diff(self,x,u):
        return np.stack((self.mu * (x[...,0] - (1/3)*x[...,0]**3 - x[...,1]) , 1/self.mu * x[...,0] +u[...,0]),axis=-1)

    def step(self,x,u,dt=0.1):
        new_x = x + dt*self.diff(x,u)
        return new_x
    
    def step_op(self,x,u,op,dt=0.1):
        new_x = x + dt*np.array([self.mu * (x[0] - (1/3)*x[0]**3 - x[1]) , 1/self.mu * x[0] +u[0]])
        return new_x
    
    def step_ode(self,x,u,dt=0.1,N=50):
        diff_ode = lambda _,x: np.concatenate(self.diff(x[:self.x_dim],x[self.x_dim:]),np.zeros((self.u_dim,)))
        return scipy.integrate.solve_ivp(diff_ode,(0,dt),np.concatenate((x,u)),t_eval= np.linspace(0,dt,N))['y']



class Uncontrolled_Vanderpol:
    def __init__(self,mu=1,x_range = np.array([[-3.0,3.0],[-6.0,6.0]])):
        self.mu = mu
        self.x_range = x_range

        self.x_dim = self.x_range.shape[0] 

    def diff(self,x):
        return np.stack((self.mu * (x[...,0] - (1/3)*x[...,0]**3 - x[...,1]) , 1/self.mu * x[...,0]),axis=-1)

    def step(self,x,dt=0.1):
        new_x = x + dt*self.diff(x)
        return new_x
        
    def step_ode(self,x,u,dt=0.1,N=50):
        diff_ode = lambda _,x: self.diff(x)
        return scipy.integrate.solve_ivp(diff_ode,(0,dt),np.concatenate((x,u)),t_eval= np.linspace(0,dt,N))['y']