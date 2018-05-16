
from cvx_sym.operations.functions import VectorFunction
from cvx_sym.operations.atoms import sums
import numpy as np

class maximum(VectorFunction):
    """ Max value element of vector """

    name = 'max'
    curvature = +1

    def __init__(self, *args):
        super().__init__(*args)
        self.args = list(args)

    def graph_form(self, scalar):
        return [scalar - self.args[0]], 1

class minimum(VectorFunction):
    """ Min value element of vector """

    name = 'min'
    curvature = -1

    def __init__(self, *args):
        super().__init__(*args)
        self.args = list(args)

    def graph_form(self, scalar):

        # TODO: Clear this up
        """ There's a bit of fishiness with this function...

        The theory differs quite significantly from the practice,
            likely because of some negative multiplication being done
            somewhere... For now we leave in a functional state.

        Specifically, this should be self.args[0] - scalar. However,
        that has the effect of making the problem unbounded.

        The problem matrix will show that the signs are the negative of what
        they should be, but the cause of this remains unknown """

        return [- self.args[0] + scalar], 1

def sum(args):
    """ Sum value elements in vector - not to be confused with
        atoms.sums.sum, which this function makes use of """

    return sums.sum(*args)
