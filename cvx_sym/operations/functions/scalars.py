
from cvx_sym.operations.functions import ScalarFunction
import numpy as np

class abs(ScalarFunction):
    """ Absolute value of scalar """

    name = 'abs'
    curvature = +1

    def __init__(self, *args):
        super().__init__(*args)
        self.args = list(args)


    def graph_form(self, scalar):
        return [scalar, self.args[0]], 2
