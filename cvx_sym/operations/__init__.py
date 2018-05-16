
from cvx_sym.conventions import scalar_shape
from cvx_sym import symbolic as sym
from .functions import matrix
from .atoms import muls

def mul(a, b):
    """ Select what "kind" of multiplication to do, based on shape! """

    if type(a) in [float, int]:
        a = sym.Constant(a)

    if type(b) in [float, int]:
        b = sym.Constant(b)

    if a.shape == scalar_shape or b.shape == scalar_shape:
        return atoms.muls.smul(a, b)

    else:
        return matrix.matmul(a, b)
