import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from src.shapefunctions import ShapeFunction
from src.quadratures import GaussianQuadrature
from src.bobby import Bobby1D
from src.equations import Equation, BurgersEquation
from recipes.one_dimensional_wave import OneDimensionalWave

n = 160
x = np.linspace(-2*np.pi, 2*np.pi, n)
#u = 0*np.ones_like(x)
#u[np.where(abs(x) < np.pi)] = 1.0
u = np.sin(x) + 2
u[np.where(abs(x) > np.pi)] = 2.0

tf = 2*np.pi*20
cfl = 0.5
equation = BurgersEquation(a = 0.5)
adv = OneDimensionalWave(x, u, tf=tf, cfl=cfl, periodic=True, step_count_max = 100000, implicit=True, func_initial=np.sin, equation=equation, solve_fd = True)
plt.figure(1)
adv.solve()
plt.show()
