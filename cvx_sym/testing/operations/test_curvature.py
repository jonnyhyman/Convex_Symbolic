from cvx_sym.operations.functions.norms import norm, norm2
from cvx_sym.symbolic import Variable, Parameter, reset_symbols

def test_symbol_curvature():

    v = Variable(name = 'v')
    assert(v.curvature == 0)

    p = Parameter(name = 'p')
    assert(p.curvature == 0)

    reset_symbols()

def check_curvy_function(function, asserted_curvature):
    print('Testing Curvature of', function.name)

    v = Variable(name = 'v')

    e = function(v)

    if asserted_curvature > 0:
        assert(e.curvature > 0)
    else:
        assert(e.curvature < 0)

    e = -1 * function(v)

    if asserted_curvature > 0:
        assert(e.curvature < 0)
    else:
        assert(e.curvature > 0)

    reset_symbols()

def test_functions_curvature():
    check_curvy_function(norm2, +1)
