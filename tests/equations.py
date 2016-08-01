import sys, os
import numpy as np
import unittest
import numpy.testing as npt

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.equations import EulerEquation

class TestGaussianQuadrature(unittest.TestCase):
    def test_jacobian_integral(self):
        equation = EulerEquation()
        eps = 1e-12
        u = np.ones(3, dtype=np.complex)
        dRdQ_approx_complex = np.zeros([3, 3])
        dRdQ_approx_fd = np.zeros([3, 3])
        dRdQ_exact = np.zeros([3, 3])


        for i in range(3):
            for j in range(3):
                dRdQ_exact[i,j] = equation.dflux[i][j](u).astype(np.float64)

        R = np.array([equation.flux[i](u) for i in range(3)])
        for i in range(3):
            u[i] += eps*1j
            for j in range(3):
                dRdQ_approx_complex[j,i] = np.imag(equation.flux[j](u))/eps
            u[i] -= eps*1j

            
        eps = 1e-6
        for i in range(3):
            u[i] += eps
            for j in range(3):
                R_p = np.array([equation.flux[k](u) for k in range(3)])
                dRdQ_approx_fd[j,i] = np.real((-R[j] + R_p[j])/eps)
            u[i] -= eps
            


        print dRdQ_exact
        print dRdQ_approx_complex
        print dRdQ_approx_fd
        npt.assert_almost_equal(dRdQ_exact, dRdQ_approx_complex, 12)
        npt.assert_almost_equal(dRdQ_exact, dRdQ_approx_fd, 5)

if __name__ == "__main__":
    unittest.main()
