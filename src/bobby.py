import numpy as np
import matplotlib.pyplot as plt

from quadratures import GaussianQuadrature
from shapefunctions import ShapeFunction
from equations import Equation

class Bobby1D(object):
    def __init__(self, x, u, tf, cfl = 0.5, periodic = True, step_count_max = 2, implicit = False, equation=Equation(), func_initial = None, solve_fd=False):
        self.x = x.copy()
        self.u = u.copy()
        self.tf = tf
        self.cfl = cfl
        self.periodic = periodic
        self.step_count_max = step_count_max
        self.implicit = implicit
        self.equation = equation
        self.func_initial = func_initial
        dx = self.x[1:] - self.x[0:-1]
        self.dxmin = dx.min()
        self.u_old = self.u.copy()
        self.dudt = np.zeros_like(self.x)
        self.du = np.zeros_like(self.x)

        self.n = self.x.size        
        self.nelements = self.n - 1
        self.t = 0.0
        self.step_count = 0
        self.global_rhs = np.zeros([self.n])
        self.global_lhs = np.zeros([self.n, self.n])
        self.global_mass = np.zeros([self.n, self.n])
        self.global_linearization = np.zeros([self.n, self.n])

        self.quadrature = GaussianQuadrature()
        self.shape_function = ShapeFunction()
        self.N = [lambda chi: self.shape_function.value(chi, 0), lambda chi: self.shape_function.value(chi, 1)]
        self.setup(equation)

        self.solve_fd = solve_fd
        if self.solve_fd:
            self.u_fd = self.u.copy()
            self.u_fd_new = self.u.copy()

    def step_fd(self):
        if self.periodic:
            start = 1
        else:
            start = 0

        for i in range(start, self.n):
            uavg = self.u_fd[i]
            w = self.wavespeed(uavg)
            if w >= 0:
                self.u_fd_new[i] = self.u_fd[i] - \
                (self.flux(self.u_fd[i]) - self.flux(self.u_fd[i-1]))/(self.x[i] - self.x[i-1])*self.dt
            else:
                self.u_fd_new[i] = self.u_fd[i] - \
                (self.flux(self.u_fd[i+1]) - self.flux(self.u_fd[i]))/(self.x[i+1] - self.x[i])*self.dt
        if self.periodic:
            self.u_fd_new[0] = self.u_fd_new[-1]
        self.u_fd[:] = self.u_fd_new[:]

    def setup(self, equation):
        pass

    def calc_dt(self):
        self.dt = 0.0

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

    def global_assembly(self, local_rhs, local_mass, local_linearization, el):
        self.global_mass[el, el] += local_mass[0,0]
        self.global_mass[el, el+1] += local_mass[0,1]
        
        self.global_mass[el+1, el] += local_mass[1,0]
        self.global_mass[el+1, el+1] += local_mass[1,1]

        self.global_linearization[el, el] += local_linearization[0,0]
        self.global_linearization[el, el+1] += local_linearization[0,1]
        
        self.global_linearization[el+1, el] += local_linearization[1,0]
        self.global_linearization[el+1, el+1] += local_linearization[1,1]

        self.global_rhs[el] += local_rhs[0]
        self.global_rhs[el+1] += local_rhs[1]

    def step(self):
        self.calc_dt()
        self.global_mass[:,:] = 0.0
        self.global_linearization[:,:] = 0.0
        self.global_rhs[:] = 0.0
        self.global_lhs[:,:] = 0.0

        for el in range(self.nelements):
            self.elemental(el)
        self.post_assembly()

    def elemental(self, el):
        pass

    def post_assembly(self):
        if self.periodic:
            start = 1
        else:
            start = 0

        if self.implicit:
            self.global_lhs[:,:] = self.global_mass[:,:]/self.dt + self.global_linearization[:,:]
            self.global_rhs[:] -= np.dot(self.global_mass, (self.u - self.u_old)/self.dt)
            #self.global_lhs[:,:] += self.global_mass[:,:]/self.dt
        else:
            self.global_lhs[:,:] = self.global_mass[:,:]


        if self.periodic:
            self.global_rhs[-1] += self.global_rhs[0]
            self.global_lhs[1,-1] = self.global_lhs[1,0]
            self.global_lhs[-1,1] = self.global_lhs[0,1]
            self.global_lhs[-1,-1] += self.global_lhs[0,0]
        else:
            self.global_rhs[-1] -= self.flux(self.u[-1])
            if self.implicit:
                self.global_lhs[-1,-1] += self.dflux(self.u[-1])
            self.global_rhs[0] += self.flux(self.u[0])
            if self.implicit:
                self.global_lhs[0,0] -= self.dflux(self.u[0])

    def step_solve(self):
        #print self.global_lhs
        #print self.u[-1]
        if self.periodic:
            start = 1
        else:
            start = 0

        if self.implicit:
            self.u_old = self.u.copy()
            print 10*"#"
            for i in range(30):
                self.du[start:] = np.linalg.solve(self.global_lhs[start:,start:], self.global_rhs[start:])
                self.u[start:] = self.u[start:] + self.du[start:]
                self.dudt[start:] = self.du[start:]/self.dt
                if self.periodic:
                    self.u[0] = self.u[-1]
                    self.dudt[0] = self.dudt[-1]
                    
                self.step()
                
                if i == 0:
                    dunorm_1 = np.linalg.norm(self.du)
                    print "dunorm = ", dunorm_1
                else:
                    dunorm = np.linalg.norm(self.du)
                    print "dunorm = ", dunorm
                    tol = abs(dunorm)
                    if tol < 1e-3:
                        break
        else:
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
            self.step_solve()
            if self.solve_fd:
                self.step_fd()
            self.plot()
            self.t += self.dt
            self.step_count += 1
            if self.t > self.tf or self.step_count >= self.step_count_max: 
                break
                
    def plot(self):
        pass
