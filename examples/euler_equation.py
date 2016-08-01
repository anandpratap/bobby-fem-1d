import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from src.shapefunctions import ShapeFunction
from src.quadratures import GaussianQuadrature
from src.bobby import Bobby1D
from src.equations import Equation, SystemEquation, EulerEquation
from src.smooth import smooth
from recipes.one_dimensional_wave import OneDimensionalWave

n = 400
x = np.linspace(0.2, 0.8, n)
u = np.zeros(n*3)
gamma = 1.4
def get_conservative_vars(rho, u, p):
    q = np.zeros(3)
    q[0] = rho
    q[1] = rho*u
    q[2] = p/(gamma - 1.0) + 0.5*q[1]**2/q[0]
    return q

rhol = 1.0
pl = 1.0
ul = 0.0#1.1*np.sqrt(gamma*pl/rhol)


rhor = 0.125#1.1691*rhol
pr = 0.1#1.2450*pl
ur = 0.0#0.9118*np.sqrt(gamma*pr/rhor)

ql = get_conservative_vars(rhol, ul, pl)
qr = get_conservative_vars(rhor, ur, pr)

u[::3] = ql[0]
u[1::3] = ql[1]
u[2::3] = ql[2]

u[::3][np.where(x>0.5)] = qr[0]
u[1::3][np.where(x>0.5)] = qr[1]
u[2::3][np.where(x>0.5)] = qr[2]

u[::3] = smooth(u[::3], 10)
u[1::3] = smooth(u[1::3], 10)
u[2::3] = smooth(u[2::3], 10)
#print u[::

tf = 2*np.pi*20
cfl = 0.01
equation = EulerEquation()
adv = OneDimensionalWave(x, u, tf=tf, cfl=cfl, periodic=False, step_count_max = 100000, implicit=False, func_initial=np.sin, equation=equation, solve_fd=False)
plt.figure(1)
adv.solve()
plt.show()
