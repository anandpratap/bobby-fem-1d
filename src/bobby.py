import numpy as np
import matplotlib.pyplot as plt

from quadratures import GaussianQuadrature
from shapefunctions import ShapeFunction
from equations import Equation

class Bobby1D(object):
    def __init__(self, x, u, tf, cfl = 0.5, periodic = True, step_count_max = 2, implicit = False, equation=Equation(), func_initial = None, solve_fd=False):
        self.setup(equation)
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
        self.dudt = np.zeros_like(self.u)
        self.du = np.zeros_like(self.u)

        self.n = self.x.size        
        self.nelements = self.n - 1
        self.t = 0.0
        self.step_count = 0
        self.global_rhs = np.zeros([self.n*self.nvar])
        self.global_lhs = np.zeros([self.n*self.nvar, self.n*self.nvar])
        self.global_mass = np.zeros([self.n*self.nvar, self.n*self.nvar])
        self.global_linearization = np.zeros([self.n*self.nvar, self.n*self.nvar])

        self.quadrature = GaussianQuadrature()
        self.shape_function = ShapeFunction()
        self.N = [lambda chi: self.shape_function.value(chi, 0), lambda chi: self.shape_function.value(chi, 1)]

        self.solve_fd = solve_fd
        if self.solve_fd:
            self.u_fd = self.u.copy()
            self.u_fd_new = self.u.copy()

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
        var_tmp = np.zeros([2, self.nvar])
        for ivar in range(self.nvar):
            var_tmp[:,ivar] = np.array([var[el*self.nvar + ivar], var[(el+1)*self.nvar + ivar]])
        return var_tmp

    def global_assembly(self, local_rhs, local_mass, local_linearization, el):
        nvar = self.nvar
        for ivar in range(self.nvar):
            for jvar in range(self.nvar):
                self.global_mass[el*nvar+ivar, el*nvar+jvar] += local_mass[0+2*ivar,0+2*jvar]
                self.global_mass[el*nvar+ivar, (el+1)*nvar+jvar] += local_mass[0+2*ivar,1+2*jvar]
                
                self.global_mass[(el+1)*nvar+ivar, el*nvar+jvar] += local_mass[1+2*ivar,0+2*jvar]
                self.global_mass[(el+1)*nvar+ivar, (el+1)*nvar+jvar] += local_mass[1+2*ivar,1+2*jvar]
                
                self.global_linearization[el*nvar+ivar, el*nvar+jvar] += local_linearization[0+2*ivar,0+2*jvar]
                self.global_linearization[el*nvar+ivar, (el+1)*nvar+jvar] += local_linearization[0+2*ivar,1+2*jvar]
                
                self.global_linearization[(el+1)*nvar+ivar, el*nvar+jvar] += local_linearization[1+2*ivar,0+2*jvar]
                self.global_linearization[(el+1)*nvar+ivar, (el+1)*nvar+jvar] += local_linearization[1+2*ivar,1+2*jvar]
            self.global_rhs[el*nvar + ivar] += local_rhs[0 + 2*ivar]
            self.global_rhs[(el+1)*nvar + ivar] += local_rhs[1 + 2*ivar]

        #print local_mass
        #print self.global_mass
        

    def step(self):
        self.u = np.maximum(self.u, 1e-14)
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
            start = 1*self.nvar
        else:
            start = 0

        if self.implicit:
            self.global_lhs[:,:] = self.global_mass[:,:]/self.dt + self.global_linearization[:,:]
            self.global_rhs[:] -= np.dot(self.global_mass, (self.u - self.u_old)/self.dt)
            #self.global_lhs[:,:] += self.global_mass[:,:]/self.dt
        else:
            self.global_lhs[:,:] = self.global_mass[:,:]


        if self.periodic:
            for ivar in range(self.nvar):
                self.global_rhs[ivar-self.nvar] += self.global_rhs[ivar]
                self.global_lhs[1*self.nvar+ivar,-1*self.nvar+ivar] = self.global_lhs[1*self.nvar+ivar,ivar]
                self.global_lhs[-1*self.nvar+ivar,1*self.nvar+ivar] = self.global_lhs[ivar,1*self.nvar+ivar]
                self.global_lhs[-1*self.nvar+ivar,-1*self.nvar+ivar] += self.global_lhs[ivar,ivar]
        else:
            for ivar in range(self.nvar):
                self.global_rhs[ivar-self.nvar] -= self.flux[ivar](self.u[-self.nvar:])
                if self.implicit:
                    self.global_lhs[ivar-self.nvar,ivar-self.nvar] += self.dflux[ivar][ivar](self.u[-self.nvar:])
                    
                self.global_rhs[ivar] += self.flux[ivar](self.u[:self.nvar])
                if self.implicit:
                    self.global_lhs[ivar,ivar] -= self.dflux[ivar][ivar](self.u[:self.nvar])

    def step_solve(self):
        #print self.global_lhs
        #print self.u[-1]
        if self.periodic:
            start = 1*self.nvar
        else:
            start = 0

        if self.implicit:
            self.u_old = self.u.copy()
            print 10*"#"
            for i in range(30):
                self.du[start:] = np.linalg.solve(self.global_lhs[start:,start:], self.global_rhs[start:])
                self.u[start:] = self.u[start:] + self.du[start:]
                self.dudt[start:] = (self.u[start:]-self.u_old[start:])/self.dt
                if self.periodic:
                    for ivar in range(self.nvar):
                        self.u[ivar] = self.u[ivar-self.nvar]
                        self.dudt[ivar] = self.dudt[ivar-self.nvar]
                    
                self.u = np.maximum(self.u, 1e-14)
                self.step()
                if i == 0:
                    dunorm_1 = np.linalg.norm(self.du)
                    print "dunorm = ", dunorm_1
                else:
                    dunorm = np.linalg.norm(self.du)
                    print "dunorm = ", dunorm
                    tol = abs(dunorm)
                    if tol < 1e-2:
                        break
        else:
            self.dudt[start:] = np.linalg.solve(self.global_lhs[start:,start:], self.global_rhs[start:])
            self.u[start:] += self.dudt[start:]*self.dt
        #print self.u[-1]
        #print self.dudt[:]*self.dt
        if self.periodic:
            for ivar in range(self.nvar):
                self.u[ivar] = self.u[ivar-self.nvar]
                self.dudt[ivar] = self.dudt[ivar-self.nvar]
                    
        
    def solve(self):
        while 1:
            self.step()
            self.step_solve()
            if self.solve_fd:
                self.u_fd[:] = self.equation.step_fd(self.x, self.u_fd, self.dt, self.periodic)
            self.plot()
            self.t += self.dt
            self.step_count += 1
            if self.t > self.tf or self.step_count >= self.step_count_max: 
                break
                
    def plot(self):
        pass
