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
        self.nvar = equation.nvar

    def elemental(self, el):
        u = self.u.copy()
        nvar = self.nvar
        local_rhs = np.zeros([2*nvar])
        local_mass = np.zeros([2*nvar, 2*nvar])
        local_linearization = np.zeros([2*nvar, 2*nvar])

        dchidx = self.get_dchidx(el)
        dxdchi = self.get_dxdchi(el)
        self.dN = [lambda chi: self.shape_function.derivative(chi, 0)*dchidx, lambda chi: self.shape_function.derivative(chi, 1)*dchidx]
        integral = self.quadrature.integral

        u_local = self.get_local_variables(u, el)
        dudt_local = self.get_local_variables(self.dudt, el)

        for ivar in range(nvar):
            local_mass[0+2*ivar,0+2*ivar] = integral(self.N[0], self.N[0])*dxdchi
            local_mass[0+2*ivar,1+2*ivar] = integral(self.N[0], self.N[1])*dxdchi
            local_mass[1+2*ivar,0+2*ivar] = integral(self.N[1], self.N[0])*dxdchi
            local_mass[1+2*ivar,1+2*ivar] = integral(self.N[1], self.N[1])*dxdchi

            
        func_u_chi = lambda chi: self.N[0](chi)*u_local[0][:] + self.N[1](chi)*u_local[1][:] + 1e-14

        for ivar in range(nvar):
            func_fu_chi = lambda chi: self.flux[ivar](func_u_chi(chi))
            local_rhs[0+2*ivar] += integral(self.dN[0], func_fu_chi)*dxdchi
            local_rhs[1+2*ivar] += integral(self.dN[1], func_fu_chi)*dxdchi


            func_residual_chi = lambda chi : 0*(self.N[0](chi)*dudt_local[0][ivar] + self.N[1](chi)*dudt_local[1][ivar]) + sum([(self.dN[0](chi)*u_local[0][jvar] + self.dN[1](chi)*u_local[1][jvar])*self.dflux[ivar][jvar](func_u_chi(chi)) for jvar in range(nvar)])
            #+ dudt_local[0]*self.N[0](chi) + + dudt_local[1]*self.N[1](chi)
            func_tau_chi = lambda chi: self.equation.get_tau(func_u_chi(chi), self.dt, dchidx, ivar)
            func_tau_times_residual_chi = lambda chi : func_tau_chi(chi)*func_residual_chi(chi)

            A_ = lambda chi : sum([self.dflux[ivar][jvar](func_u_chi(chi)) for jvar in range(nvar)])
            galerkin_correction = np.zeros(2)
            galerkin_correction[0] = integral(A_, self.dN[0], func_tau_times_residual_chi)*dxdchi
            galerkin_correction[1] = integral(A_, self.dN[1], func_tau_times_residual_chi)*dxdchi
            local_rhs[0+2*ivar] -= galerkin_correction[0]
            local_rhs[1+2*ivar] -= galerkin_correction[1]
        

        if self.implicit:
            for ivar in range(self.nvar):
                for jvar in range(self.nvar):
                    tmp_func = lambda chi : self.dflux[ivar][jvar](func_u_chi(chi))
                    local_linearization[0+2*ivar,0+2*jvar] -= integral(self.dN[0], tmp_func, self.N[0])*dxdchi
                    local_linearization[0+2*ivar,1+2*jvar] -= integral(self.dN[0], tmp_func, self.N[1])*dxdchi
                    local_linearization[1+2*ivar,0+2*jvar] -= integral(self.dN[1], tmp_func, self.N[0])*dxdchi
                    local_linearization[1+2*ivar,1+2*jvar] -= integral(self.dN[1], tmp_func, self.N[1])*dxdchi
                    

                    # #tmp_func_1 = lambda chi: self.ddflux(func_u_chi(chi))*(self.dN[0](chi)*u_local[0] + self.dN[1](chi)*u_local[1])
                    func_tau_chi = lambda chi: self.equation.get_tau(func_u_chi(chi), self.dt, dchidx, ivar)

                    # #func_tau_chi = lambda chi: 1.0/np.sqrt((4*abs(self.wavespeed[ivar](func_u_chi(chi))))*dchidx**2)
                    tmp_func_2 = lambda chi: self.dflux[ivar][jvar](func_u_chi(chi))
                    A_ = lambda chi :  sum([self.dflux[ivar][jvar](func_u_chi(chi)) for jvar in range(nvar)])

                    local_linearization[0+2*ivar,0+2*jvar] += integral(A_, self.dN[0], func_tau_chi, tmp_func_2, self.dN[0])*dxdchi
                    local_linearization[0+2*ivar,1+2*jvar] += integral(A_, self.dN[0], func_tau_chi, tmp_func_2, self.dN[1])*dxdchi
                    local_linearization[1+2*ivar,0+2*jvar] += integral(A_, self.dN[1], func_tau_chi, tmp_func_2, self.dN[0])*dxdchi
                    local_linearization[1+2*ivar,1+2*jvar] += integral(A_, self.dN[1], func_tau_chi, tmp_func_2, self.dN[1])*dxdchi

            # Local_linearization[0,0] += integral(self.dN[0], func_tau_chi, tmp_func_1, self.N[0])*dxdchi
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
        dt = np.zeros(self.nvar)
        for var in range(self.nvar):
            u_tmp = self.u.reshape(self.nvar, self.x.size, order="F")
            u_speed = self.wavespeed[var](u_tmp)
            u_max_speed = abs(u_speed).max()
            dt[var] = self.dxmin/u_max_speed*self.cfl
        self.dt = dt.min()
        print self.dt

    def plot(self):
        plt.ion()
        plt.figure(1)
        nvar = self.nvar

        if self.step_count > 0:
         #   if self.func_initial is not None:
         #       del plt.gca().lines[1]
            if self.solve_fd is True:
                pass
                #for ivar in range(nvar):
                 #   del plt.gca().lines[ivar]

                 #for ivar in range(nvar):
                #de#l plt.gca().lines[ivar]
            pass
        plt.clf()
        plt.title("t = %.5f cfl = %.3f implicit = %s"%(self.t, self.cfl, str(self.implicit)))
        for ivar in range(nvar):
            if ivar == 1:
                plt.plot(self.x, self.u[ivar::nvar]/self.u[0::nvar], ".-", label="FEM %s"%(self.equation.var_names[ivar]))
            else:
                plt.plot(self.x, self.u[ivar::nvar], ".-", label="FEM %s"%(self.equation.var_names[ivar]))
        #if self.func_initial is not None:
        #    plt.plot(self.x, self.func_initial(self.x - self.wavespeed(self.u)*(self.t + self.dt)), "g-")
        if self.solve_fd:
            for ivar in range(nvar):
                plt.plot(self.x, self.u_fd[ivar::nvar], "--", label="FD %s"%(self.equation.var_names[ivar]))
        plt.legend(loc="best")
        plt.pause(0.0001)
        plt.savefig("figures/solution_system_%s.png"%(str(self.step_count).zfill(10)))
        plt.ioff()


if __name__ == "__main__":
    pass
