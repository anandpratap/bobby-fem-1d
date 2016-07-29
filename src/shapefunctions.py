import numpy as np

class ShapeFunction(object):
    def __init__(self):
        self.__chi_min = -1.0
        self.__chi_max = 1.0
        self.__function = [lambda chi: 0.5*(1.0 - chi), 
                           lambda chi: 0.5*(1.0 + chi)]

        self.__dfunction = [lambda chi: -0.5, lambda chi: 0.5]

    def value(self, chi, mode=0):
        return self.__function[mode](chi)

    def derivative(self, chi, mode=0):
        return self.__dfunction[mode](chi)
