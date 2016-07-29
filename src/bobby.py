import numpy as np
import matplotlib.pyplot as plt

from quadratures import GaussianQuadrature
from shapefunctions import ShapeFunction

class Bobby1D(object):
    def __init__(self, x, u, tf, cfl = 0.5, periodic = True, step_count_max = 2):
        self.x = x.copy()
        self.u = u.copy()
        self.tf = tf
        self.cfl = cfl
        self.periodic = periodic
        self.step_count_max = step_count_max

        dx = self.x[1:] - self.x[0:-1]
        self.dt = dx.min()*self.cfl

        self.dudt = np.zeros_like(self.x)

        self.n = self.x.size        
        self.nelements = self.n - 1
        self.t = 0.0
        self.step_count = 0
        self.global_rhs = np.zeros([self.n])
        self.global_lhs = np.zeros([self.n, self.n])

        self.quadrature = GaussianQuadrature()
        self.shape_function = ShapeFunction()
        self.N = [lambda chi: self.shape_function.value(chi, 0), lambda chi: self.shape_function.value(chi, 1)]

    def get_dchidx(self, el):
        dxdchi = self.get_dxdchi(el)
        return 1.0/dxdchi
        
    def get_dxdchi(self, el):
        h = self.x[el+1] - self.x[el]
        dxdchi = h/2.0
        return dxdchi
        
    def get_volume(self, el):
        h = self.x[el+1] - self.x[el]
        return h/2.0
        
    def get_local_variables(self, var, el):
        return np.array([var[el], var[el+1]])

    def global_assembly(self, local_rhs, local_lhs, el):
        self.global_lhs[el, el] += local_lhs[0,0]
        self.global_lhs[el, el+1] += local_lhs[0,1]
        
        self.global_lhs[el+1, el] += local_lhs[1,0]
        self.global_lhs[el+1, el+1] += local_lhs[1,1]

        self.global_rhs[el] += local_rhs[0]
        self.global_rhs[el+1] += local_rhs[1]

    def step(self):
        self.global_rhs[:] = 0.0
        self.global_lhs[:,:] = 0.0

        for el in range(self.nelements):
            self.elemental(el)
        self.post_assembly()

    def elemental(self, el):
        pass

    def post_assembly(self):
        start = 0
        self.global_rhs[-1] -= self.u[-1]

        if self.periodic:
            self.global_lhs[1,-1] = self.global_lhs[1,0]
            start = 1
        #print self.global_lhs
        #print self.u[-1]
        self.dudt[start:] = np.linalg.solve(self.global_lhs[start:,start:], self.global_rhs[start:])
        self.u[start:] += self.dudt[start:]*self.dt
        #print self.u[-1]
        #print self.dudt[:]*self.dt
        if self.periodic:
            self.u[0] = self.u[-1]
            self.dudt[0] = self.dudt[-1]
        
    def solve(self):
        while 1:
            self.step()
            self.plot()
            self.t += self.dt
            self.step_count += 1
            if self.t > self.tf or self.step_count >= self.step_count_max: 
                break
                
    def plot(self):
        pass
