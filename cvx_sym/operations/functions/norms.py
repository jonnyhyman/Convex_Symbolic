
from cvx_sym.operations.functions import VectorFunction
from cvx_sym.conventions import scalar_shape
from cvx_sym.symbolic import Symbol
from . import vectors
from . import scalars
import numpy as np

class norm2(VectorFunction):
    """ Euclidean 2-norm : sqrt(sum of squares) """

    name = 'norm2'
    curvature = +1

    def __init__(self, *args):
        super().__init__(*args)

        self.args = list(args)  # make the args mutable
        self.n = max(self.args[0].shape[0], self.args[0].shape[1])

    def graph_form(self, scalar):

        # epigraph_axes, dimensions
        return [scalar] + self.args, (self.n + 1)

    def get_context(self):
        """ Provide more context to templates """
        return {'n' : self.n}

    def value(self, parameters):
        """ Parameters supplied from assign_values """

        p_args = [parameters[v.name] for v in self.args[0]]
        return np.linalg.norm(p_args)

def norm_inf(*args):
    return vectors.maximum(scalars.abs(*args))

def norm1(*args):
    abs = scalars.abs(*args)

    if abs.shape == scalar_shape:
        return abs

    else:
        return vectors.sum(abs)

def norm(*args, kind = 2):

    if kind == 2:
        return norm2(*args)

    if kind == 1:
        return norm1(*args)

    elif kind == 'inf':
        return norm_inf(*args)

    else:
        return NotImplemented
