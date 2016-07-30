import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from src.shapefunctions import ShapeFunction
from src.quadratures import GaussianQuadrature
from src.bobby import Bobby1D
from src.equations import Equation

class OneDimensionalWave(Bobby1D):
    def setup(self, equation):
        self.flux = equation.flux
        self.dflux = equation.dflux
        self.wavespeed = equation.wavespeed
        self.ddflux = equation.ddflux
        self.dwavespeed = equation.dwavespeed

    def elemental(self, el):
        u = self.u.copy()
        local_rhs = np.zeros([2])
        local_mass = np.zeros([2, 2])
        local_linearization = np.zeros([2, 2])

        dchidx = self.get_dchidx(el)
        dxdchi = self.get_dxdchi(el)
        self.dN = [lambda chi: self.shape_function.derivative(chi, 0)*dchidx, lambda chi: self.shape_function.derivative(chi, 1)*dchidx]
        integral = self.quadrature.integral

        u_local = self.get_local_variables(u, el)
        dudt_local = self.get_local_variables(self.dudt, el)
        
        local_mass[0,0] = integral(self.N[0], self.N[0])*dxdchi
        local_mass[0,1] = integral(self.N[0], self.N[1])*dxdchi
        local_mass[1,0] = integral(self.N[1], self.N[0])*dxdchi
        local_mass[1,1] = integral(self.N[1], self.N[1])*dxdchi

        func_u_chi = lambda chi: self.N[0](chi)*u_local[0] + self.N[1](chi)*u_local[1] + 1e-14
        func_fu_chi = lambda chi: self.flux(func_u_chi(chi))
        

        local_rhs[0] += integral(self.dN[0], func_fu_chi)*dxdchi
        local_rhs[1] += integral(self.dN[1], func_fu_chi)*dxdchi

        func_residual_chi = lambda chi : (self.dN[0](chi)*u_local[0] + self.dN[1](chi)*u_local[1])*self.dflux(func_u_chi(chi))
        #+ dudt_local[0]*self.N[0](chi) + + dudt_local[1]*self.N[1](chi)
        func_tau_chi = lambda chi: 1.0/np.sqrt((4*abs(self.wavespeed(func_u_chi(chi))))*dchidx**2)
        func_tau_times_residual_chi = lambda chi : func_tau_chi(chi)*func_residual_chi(chi)


        galerkin_correction = np.zeros(2)
        galerkin_correction[0] = integral(self.dN[0], func_tau_times_residual_chi)*dxdchi
        galerkin_correction[1] = integral(self.dN[1], func_tau_times_residual_chi)*dxdchi
        local_rhs[0] -= galerkin_correction[0]
        local_rhs[1] -= galerkin_correction[1]
        

        if self.implicit:
            tmp_func = lambda chi : 1#self.dflux(func_u_chi(chi))
            local_linearization[0,0] -= integral(self.dN[0], tmp_func, self.N[0])*dxdchi
            local_linearization[0,1] -= integral(self.dN[0], tmp_func, self.N[1])*dxdchi
            local_linearization[1,0] -= integral(self.dN[1], tmp_func, self.N[0])*dxdchi
            local_linearization[1,1] -= integral(self.dN[1], tmp_func, self.N[1])*dxdchi
            

            tmp_func_1 = lambda chi: self.ddflux(func_u_chi(chi))*(self.dN[0](chi)*u_local[0] + self.dN[1](chi)*u_local[1])
            tmp_func_2 = lambda chi: 1#self.dflux(func_u_chi(chi))
            
            local_linearization[0,0] += integral(self.dN[0], func_tau_chi, tmp_func_2, self.dN[0])*dxdchi
            local_linearization[0,1] += integral(self.dN[0], func_tau_chi, tmp_func_2, self.dN[1])*dxdchi
            local_linearization[1,0] += integral(self.dN[1], func_tau_chi, tmp_func_2, self.dN[0])*dxdchi
            local_linearization[1,1] += integral(self.dN[1], func_tau_chi, tmp_func_2, self.dN[1])*dxdchi

            # local_linearization[0,0] += integral(self.dN[0], func_tau_chi, tmp_func_1, self.N[0])*dxdchi
            # local_linearization[0,1] += integral(self.dN[0], func_tau_chi, tmp_func_1, self.N[1])*dxdchi
            # local_linearization[1,0] += integral(self.dN[1], func_tau_chi, tmp_func_1, self.N[0])*dxdchi
            # local_linearization[1,1] += integral(self.dN[1], func_tau_chi, tmp_func_1, self.N[1])*dxdchi

            
            # tmp_func = lambda chi: 1.0/np.sqrt((4*abs(self.wavespeed(func_u_chi(chi))))*dchidx**2)**3 * (4.0*dchidx**2*abs(self.dwavespeed(func_u_chi(chi))))
            # local_linearization[0,0] += integral(self.dN[0], func_residual_chi, tmp_func, self.N[0])*dxdchi
            # local_linearization[0,1] += integral(self.dN[0], func_residual_chi, tmp_func, self.N[1])*dxdchi
            # local_linearization[1,0] += integral(self.dN[1], func_residual_chi, tmp_func, self.N[0])*dxdchi
            # local_linearization[1,1] += integral(self.dN[1], func_residual_chi, tmp_func, self.N[1])*dxdchi


        self.global_assembly(local_rhs, local_mass, local_linearization, el)

    def calc_dt(self):
        u_speed = self.wavespeed(self.u)
        u_max_speed = abs(u_speed).max()
        self.dt = self.dxmin/u_max_speed*self.cfl

    def plot(self):
        plt.ion()
        plt.figure(1)
        if self.step_count > 0:
         #   if self.func_initial is not None:
         #       del plt.gca().lines[1]
            if self.solve_fd is True:
                del plt.gca().lines[1]

            del plt.gca().lines[0]
        plt.title("t = %.2f cfl = %.3f implicit = %s"%(self.t, self.cfl, str(self.implicit)))
        plt.plot(self.x, self.u, "r.-", label="FEM")
        #if self.func_initial is not None:
        #    plt.plot(self.x, self.func_initial(self.x - self.wavespeed(self.u)*(self.t + self.dt)), "g-")
        if self.solve_fd:
            plt.plot(self.x, self.u_fd, "b--", label="FD")
        plt.legend(loc=3)
        plt.pause(0.0001)
        plt.savefig("figures/solution_4_%s.png"%(str(self.step_count).zfill(10)))
        plt.ioff()


if __name__ == "__main__":
    pass
