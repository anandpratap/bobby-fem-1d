import sys, os
import numpy as np
import unittest
import numpy.testing as npt

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.shapefunctions import ShapeFunction
from src.quadratures import GaussianQuadrature

class TestGaussianQuadrature(unittest.TestCase):
    def test_zero_integral(self):
        quadrature = GaussianQuadrature()
        func = lambda chi : 0.0
        npt.assert_almost_equal(quadrature.integral(func), 0.0, 13)

        func = lambda chi : 1.0
        npt.assert_almost_equal(quadrature.integral(func), 2.0, 13)

        func = lambda chi : -1.0
        npt.assert_almost_equal(quadrature.integral(func), -2.0, 13)

    def test_linear_integral(self):
        quadrature = GaussianQuadrature()
        func = lambda chi : chi
        npt.assert_almost_equal(quadrature.integral(func), 0.0, 13)

        func = lambda chi : chi + 1
        npt.assert_almost_equal(quadrature.integral(func), 2.0, 13)

        func = lambda chi : chi - 1
        npt.assert_almost_equal(quadrature.integral(func), -2.0, 13)

    def test_multiple_integral(self):
        quadrature = GaussianQuadrature()
        func_1 = lambda chi : 1
        func_2 = lambda chi : chi
        npt.assert_almost_equal(quadrature.integral(func_1, func_2), 
                                0.0, 13)

        func_1 = lambda chi : chi
        func_2 = lambda chi : chi
        npt.assert_almost_equal(quadrature.integral(func_1, func_2), 
                                2.0/3.0, 13)

        func_1 = lambda chi : chi + 1
        func_2 = lambda chi : chi
        npt.assert_almost_equal(quadrature.integral(func_1, func_2), 
                                2.0/3.0, 13)

        func_1 = lambda chi : chi
        func_2 = lambda chi : chi + 1
        npt.assert_almost_equal(quadrature.integral(func_1, func_2), 
                                2.0/3.0, 13)

        func_1 = lambda chi : chi
        func_2 = lambda chi : chi + 1
        func_3 = lambda chi : chi - 1
        npt.assert_almost_equal(quadrature.integral(func_1, func_2, func_3), 
                                0.0, 13)

        func_1 = lambda chi : chi
        func_2 = lambda chi : chi + 1
        func_3 = lambda chi : chi + 1
        npt.assert_almost_equal(quadrature.integral(func_1, func_2, func_3), 
                                4.0/3.0, 13)

if __name__ == "__main__":
    unittest.main()
