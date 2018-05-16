
from cvx_sym.operations.functions import ScalarFunction, VectorFunction
from cvx_sym.operations.functions import norms
from cvx_sym.symbolic import Constant
import numpy as np

class quad_over_lin(VectorFunction):
    """
        Quadratic over Linear   : x.T * x / y
                                | x is a vector (type: list)
                                | y >= 0 scalar (type: int, float, Symbol)
    """
    name = 'quad_over_lin'
    curvature = +1

    def __init__(self, x, y):
        super().__init__(x,y)

        self.args = x + [y]  # having this enables composition of functions
        self.x = x
        self.y = y

    def graph_form(self, scalar):
        epigraph_axes  = [self.y + scalar, self.y - scalar]
        epigraph_axes += [2 * x_i for x_i in self.x]
        return epigraph_axes, len(self.x) + 2

    def value(self, parameters):
        """ Parameters supplied from assign_values """

        p_args = [parameters[v.name] for v in self.args]

        x = np.array([self.args[:-1]])
        return x.T * x / self.args[-1]

class square(ScalarFunction):

    """ Square of a scalar input """

    name = 'square'
    curvature = +1

    def __init__(self, *args):
        super().__init__(*args)

        self.args = list(args)  # make the args mutable
        # self.args always length 1 because input always scalar

    def graph_form(self, scalar):

        # epigraph_axes, dimensions
        return quad_over_lin(self.args, 1).graph_form(scalar)

    def value(self, parameters):
        """ Parameters supplied from assign_values """

        return quad_over_lin(self.args, 1).value(parameters)

class inv_pos(ScalarFunction):

    """ Positive reciporical/"inverse" of scalar input : 1/x"""

    name = 'inv_pos'
    curvature = +1

    def __init__(self, *args):
        super().__init__(*args)

        self.args = list(args)  # make the args mutable
        # self.args always length 1 because input always scalar

    def graph_form(self, scalar):

        # epigraph_axes, dimensions
        return quad_over_lin([1], self.args[0]).graph_form(scalar)

    def value(self, parameters):
        """ Parameters supplied from assign_values """

        return quad_over_lin([1], self.args[0]).value(parameters)


def sum_squares(*args):
    return square(norms.norm(*args, kind = 2))
