import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from src.shapefunctions import ShapeFunction
from src.quadratures import GaussianQuadrature
from src.bobby import Bobby1D
from src.equations import Equation, SystemEquation
from recipes.one_dimensional_wave import OneDimensionalWave

n = 100
x = np.linspace(-np.pi, np.pi+20, n)
u = np.zeros(n*3)
u[::2] = 0#np.sin(x) + 4
u[1::3] = 1
u[2::3] = 2#np.sin(x) + 2
u[::3][np.where(abs(x)<np.pi/2)] = 1
u[1::3][np.where(abs(x)<np.pi/2)] = 2
u[2::3][np.where(abs(x)<np.pi/2)] = 3


tf = 2*np.pi*20
cfl = 0.5
equation = SystemEquation()
adv = OneDimensionalWave(x, u, tf=tf, cfl=cfl, periodic=True, step_count_max = 100000, implicit=True, func_initial=np.sin, equation=equation, solve_fd=True)
plt.figure(1)
adv.solve()
plt.show()
