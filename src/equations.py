import numpy as np

class Equation(object):
    def __init__(self, a=1):
        self.a = a
        self.flux = lambda u : a*u
        self.dflux = lambda u : a*np.ones_like(u)
        self.wavespeed = lambda u : a*np.ones_like(u)
        self.ddflux = lambda u : np.zeros_like(u)
        self.dwavespeed = lambda u : np.zeros_like(u)

    def exact(func_initial, x, t):
        return func_initial(x - self.a*t)


class BurgersEquation(object):
    def __init__(self, a=1):
        self.a = 0.5
        self.flux = lambda u : a*u*u
        self.dflux = lambda u : a*2*u
        self.wavespeed = lambda u : a*2*u
        self.ddflux = lambda u : a*2*np.ones_like(u)
        self.dwavespeed = lambda u : a*2*np.ones_like(u)
