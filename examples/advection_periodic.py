import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from src.shapefunctions import ShapeFunction
from src.quadratures import GaussianQuadrature
from src.bobby import Bobby1D
from src.equations import Equation
from recipes.one_dimensional_wave import OneDimensionalWave

n = 100
x = np.linspace(-np.pi, np.pi, n)
u = np.sin(x)
#u[np.where(x>0)] = 0
tf = 2*np.pi*20
cfl = 0.1
equation = Equation(a = 1)
adv = OneDimensionalWave(x, u, tf=tf, cfl=cfl, periodic=True, step_count_max = 100000, implicit=True, func_initial=np.sin, equation=equation, solve_fd=True)
plt.figure(1)
adv.solve()
plt.show()
