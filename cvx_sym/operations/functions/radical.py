
from cvx_sym.operations.functions import PureScalarFunction, ScalarFunction
from cvx_sym.symbolic import Constant
import numpy as np

class geo_mean(PureScalarFunction):
    """
        Geometric Mean          : sqrt(x*y)
                                | x is a scalar (type: int, float, Symbol)
                                | y >= 0 scalar (type: int, float, Symbol)
    """
    name = 'geo_mean'
    curvature = -1

    def __init__(self, x, y):
        super().__init__(x, y)

        self.args = [x, y]  # having this enables composition of functions
        self.x = x
        self.y = y

        # Remove constant x's from x to ensure sensical constraints...
        # Otherwise we can end up with silly things like 2 <= 0
        if not self.parametric:
            rm = [n for n, x_n in enumerate(self.x)
                    if type(x_n) in [float, int, Constant]]

            for n in rm:
                self.x.pop(n)

    def graph_form(self, scalar):
        epigraph_axes  = [self.y + self.x, self.y - self.x]
        epigraph_axes += [2 * scalar]
        return epigraph_axes, 3

    def value(self, parameters):
        """ Parameters supplied from assign_values """

        p_args = [parameters[v.name] for v in self.args]
        return np.sqrt(pargs[0] * pargs[1])

class sqrt(ScalarFunction):

    """ Square root of a scalar input """

    name = 'sqrt'
    curvature = -1

    def __init__(self, *args):
        super().__init__(*args)
        self.args = list(args)  # make the args mutable

    def graph_form(self, scalar):

        # epigraph_axes, dimensions
        return geo_mean(self.args[0], 1).graph_form(scalar)

    def value(self, parameters):
        """ Parameters supplied from assign_values """

        return geo_mean(self.args[0], 1).value(parameters)
