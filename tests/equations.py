import sys, os
import numpy as np
import unittest
import numpy.testing as npt

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.equations import Equation, BurgersEquation, SystemEquation, EulerEquation

class TestGaussianQuadrature(unittest.TestCase):
    def test_jacobian(self):
        for equation in [Equation, BurgersEquation, SystemEquation, EulerEquation]:
            self.jacobian(equation())

    def jacobian(self, eqn):
        eps = 1e-12
        u = np.ones(eqn.nvar, dtype=np.complex)
        dRdQ_approx_complex = np.zeros([eqn.nvar, eqn.nvar])
        dRdQ_approx_fd = np.zeros([eqn.nvar, eqn.nvar])
        dRdQ_exact = np.zeros([eqn.nvar, eqn.nvar])


        for i in range(eqn.nvar):
            for j in range(eqn.nvar):
                dRdQ_exact[i,j] = eqn.dflux[i][j](u).astype(np.float64)

        R = np.array([eqn.flux[i](u) for i in range(eqn.nvar)])
        for i in range(eqn.nvar):
            u[i] += eps*1j
            for j in range(eqn.nvar):
                dRdQ_approx_complex[j,i] = np.imag(eqn.flux[j](u))/eps
            u[i] -= eps*1j

            
        eps = 1e-6
        for i in range(eqn.nvar):
            u[i] += eps
            for j in range(eqn.nvar):
                R_p = np.array([eqn.flux[k](u) for k in range(eqn.nvar)])
                dRdQ_approx_fd[j,i] = np.real((-R[j] + R_p[j])/eps)
            u[i] -= eps
            


        print dRdQ_exact
        print dRdQ_approx_complex
        print dRdQ_approx_fd
        npt.assert_almost_equal(dRdQ_exact, dRdQ_approx_complex, 12)
        npt.assert_almost_equal(dRdQ_exact, dRdQ_approx_fd, 5)

if __name__ == "__main__":
    unittest.main()
