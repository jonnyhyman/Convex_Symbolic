
from cvx_sym.conventions import scalar_shape
from cvx_sym import symbolic as sym
from .functions import matrix
from .atoms import muls
from .atoms import divs

def mul(a, b):
    """ Select what "kind" of multiplication to do, based on shape! """

    if type(a) in [float, int]:
        a = sym.Constant(a)

    if type(b) in [float, int]:
        b = sym.Constant(b)

    if a.shape == scalar_shape or b.shape == scalar_shape:
        return muls.smul(a, b)

    else:
        return matrix.matmul(a, b)

def div(a, b):
    """ Select what "kind" of division to do! For now just one kind """
    return divs.div(a, b)
