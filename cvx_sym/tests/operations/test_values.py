""" Test the 'value' numerical method of functions """

from cvx_sym.operations.functions.norms import norm
from cvx_sym.symbolic import Parameter, reset_symbols
import numpy as np

def test_norm_value_expr():

    p = Parameter((3,1), name = 'p')

    expr = norm(p)

    value = expr.value({'p[0][0]' : 3, 'p[1][0]' : 2, 'p[2][0]' : 1})

    assert(np.isclose(value, np.linalg.norm([[3],[2],[1]]) ))

    reset_symbols()

    p = Parameter((1,3), name = 'p')

    expr = norm(p)

    value = expr.value({'p[0][0]' : 3, 'p[0][1]' : 2, 'p[0][2]' : 1})

    assert(np.isclose(value, np.linalg.norm([3,2,1]) ))

    reset_symbols()
