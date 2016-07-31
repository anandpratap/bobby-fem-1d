import numpy as np

class Equation(object):
    def __init__(self, a=1):
        self.var_names = ["Advection"]
        self.nvar = 1
        self.a = a
        self.flux = [lambda u : a*u]
        self.dflux = [[lambda u : a*np.ones_like(u)]]
        self.wavespeed = [lambda u : a*np.ones_like(u)]
        self.ddflux = [lambda u : np.zeros_like(u)]
        self.dwavespeed = [lambda u : np.zeros_like(u)]

    def exact(func_initial, x, t):
        return func_initial(x - self.a*t)


    def step_fd(self, x, u_fd, dt, periodic):
        u_fd_new = np.zeros_like(u_fd)
        if periodic:
            start = 1
            u_fd_new[:start*self.nvar] = u_fd[:start*self.nvar]
        else:
            start = 1
            u_fd_new[:start*self.nvar] = u_fd[:start*self.nvar]

        
        n = x.size
        
        for i in range(start, n):
            for ivar in range(self.nvar):
                uavg = u_fd[i*self.nvar:i*self.nvar+self.nvar]
                w = self.wavespeed[ivar](uavg)
                idx_p = (i+1)*self.nvar
                idx = i*self.nvar
                idx_m = (i-1)*self.nvar
                if w >= 0:
                    u_fd_new[i*self.nvar+ivar] = u_fd[i*self.nvar + ivar] - \
                                                 (self.flux[ivar](u_fd[idx:idx+self.nvar]) - self.flux[ivar](u_fd[idx_m:idx_m+self.nvar]))/(x[i] - x[i-1])*dt
                else:
                    u_fd_new[i*self.nvar+ivar] = u_fd[i] - \
                                                 (self.flux[ivar](u_fd[idx_p:idx_p+self.nvar]) - self.flux[ivar](u_fd[idx:idx+self.nvar]))/(x[i+1] - x[i])*dt
        if periodic:
            for ivar in range(self.nvar):
                u_fd_new[ivar] = u_fd_new[-self.nvar+ivar]
                        
        return u_fd_new


class BurgersEquation(Equation):
    def __init__(self, a=1):
        self.var_names = ["Burgers"]

        self.nvar = 1
        self.a = 0.5
        self.flux = [lambda u : a*u*u]
        self.dflux = [[lambda u : a*2*u]]
        self.wavespeed = [lambda u : a*2*u]
        self.ddflux = [lambda u : a*2*np.ones_like(u)]
        self.dwavespeed = [lambda u : a*2*np.ones_like(u)]

    
class SystemEquation(Equation):
    def __init__(self, a=1):
        self.var_names = ["Advection a = 1", "Burgers", "Advection a = 2"]
        self.nvar = 3
        self.a = a
        b = 0.5
        self.flux = [lambda u : a*u[0], lambda u : b*u[1]*u[1], lambda u: 2*a*u[2]]
        self.dflux = [[lambda u : a*np.ones_like(u[0]), lambda u : a*np.zeros_like(u[0]), lambda u : a*np.zeros_like(u[0])], 
                      [lambda u : a*np.zeros_like(u[1]), lambda u : 2*b*u[1], lambda u : a*np.zeros_like(u[1])],
                      [lambda u : a*np.zeros_like(u[2]), lambda u : np.zeros_like(u[2]), lambda u : 2*a*np.ones_like(u[2])],
        ]
        self.wavespeed = [lambda u : a*np.ones_like(u[0]), lambda u : 2*b*u[1], lambda u : 2*a*np.ones_like(u[2])]
        self.ddflux = [lambda u : np.zeros_like(u)]
        self.dwavespeed = [lambda u : np.zeros_like(u)]


class EulerEquation(Equation):
    def __init__(self, a=1):
        self.var_names = ["rho", "rhoU", "rhoE"]
        self.nvar = 3
        gamma = 1.4

        def p(u):
            pp = (gamma-1)*(u[2] - 0.5*u[1]**2)/u[0]
            return pp

        self.flux = [lambda u : u[1], 
                     lambda u : 0.5*(3 - gamma)*u[1]**2/u[0] + (gamma - 1)*u[2], 
                     lambda u: gamma*u[1]*u[2]/u[0] - 0.5*(gamma-1)*u[1]**2/u[0]**2]

        self.dflux = [[lambda u : np.zeros_like(u[0]), lambda u : np.ones_like(u[0]), lambda u : np.zeros_like(u[0])], 
                      [lambda u : -0.5*(gamma-3)*(u[1]**2/u[0]**2), lambda u : (3 - gamma)*u[1]/u[0], lambda u : (gamma-1)*np.ones_like(u[0])],
                      [lambda u : -gamma*u[1]*u[2]/u[0]**2 + (gamma-1)*(u[1]/u[0])**3, lambda u : gamma*u[2]/u[0] - 1.5*(gamma-1)*(u[1]/u[0])**2, lambda u : gamma*u[1]/u[0]],
                  ]
        self.wavespeed = [lambda u : u[1]/u[0] + np.sqrt(gamma*p(u)/u[0]), lambda u : u[1]/u[0] + np.sqrt(gamma*p(u)/u[0]), lambda u : u[1]/u[0] + np.sqrt(gamma*p(u)/u[0])]
        self.ddflux = [lambda u : np.zeros_like(u)]
        self.dwavespeed = [lambda u : np.zeros_like(u)]
