from cvx_sym.operations.functions import quads
from cvx_sym.conventions import scalar_shape
from cvx_sym.symbolic import Vector

def div(numer, denom):
    return numer * quads.inv_pos(denom)


    """

    if (numer.shape != scalar_shape
        and denom.shape == scalar_shape):

        return Vector([quads.inv_pos(n, denom) for n in numer])

    elif (numer.shape == scalar_shape
        and denom.shape != scalar_shape):

        return Vector([quads.inv_pos(numer, d) for d in numer])

    else:
        raise(TypeError("Division must be able to occur elementwise."
                " Got shapes "+str(numer.shape)+" and "+str(denom.shape)))
    """
