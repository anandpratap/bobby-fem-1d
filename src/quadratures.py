import numpy as np

class GaussianQuadrature(object):
    def __init__(self):
        self.__eval_points = np.array([-np.sqrt(1.0/3.0), np.sqrt(1.0/3.0)])
        self.__eval_weights = np.array([1.0, 1.0])
        self.__neval_points = self.__eval_points.size

    def integral(self, *args):
        integral_value = 0.0
        for i in range(self.__neval_points):
            prod = 1.0
            chi = self.__eval_points[i]
            weight = self.__eval_weights[i]
            for count, func in enumerate(args):
                prod *= func(chi)
            integral_value += prod*weight
        return integral_value
