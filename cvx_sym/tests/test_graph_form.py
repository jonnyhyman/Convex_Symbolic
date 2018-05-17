from cvx_sym.symbolic import (Variable, Parameter, Constant, Symbol,
                                reset_symbols)
from cvx_sym.operations.functions.quads import square, quad_over_lin, inv_pos
from cvx_sym.operations.functions.vectors import maximum, minimum
from cvx_sym.operations.functions.radical import geo_mean, sqrt
from cvx_sym.operations.functions.norms import norm
from cvx_sym.canonicalize import Canonicalize

from cvx_sym.constraints import SecondOrderConeConstraint, eq, le

"""
    Test the Graph Forms of the nonlinear atoms:
        norm
        quad_over_lin(x,1)
        quad_over_lin(1,x)
        quad_over_lin(x,y)
        square(x) --> quad_over_lin(x,1)
        max(x)
"""

def test_norm_graph():

    v0 = Variable(name = 'v0')
    s0 = Symbol(name = 's0')

    expr = norm(v0)

    constr = (expr <= s0)  # effectively in relaxed smith form

    print('Graph form of', end = '')
    print(constr)
    constr = constr.graph_form()
    print('-->', constr)
    print()

    n = max(v0.shape[0], v0.shape[1])
    assert(constr.dims == n + 1)
    assert(type(constr) is SecondOrderConeConstraint)
    assert(all([type(c) is le for c in constr.constraints]))
    assert(constr.constraints[1].expr.args[0] is v0)
    assert(constr.constraints[0].expr.args[0].args[1] is s0)

    reset_symbols()

def test_quad_over_lin_y1_graph():

    v0 = Variable(name = 'v0')
    v1 = Variable(name = 'v1')
    s0 = Symbol(name = 's0')

    x = [v0, v1]
    expr = quad_over_lin(x, 1)

    constr = (expr <= s0)  # effectively in relaxed smith form

    print('Graph form of', end = '')
    print(constr)
    constr = constr.graph_form()
    print('-->', constr)
    print()

    n = len(x)
    assert(constr.dims == n + 2)
    assert(type(constr) is SecondOrderConeConstraint)
    assert(all([type(c) is le for c in constr.constraints]))

    c = constr # keeps below names same as other tests

    string_equals = [
        """(-1.0 + (-1.0 * s0)) <= 0""",
        """(1.0 + (-1.0 * s0)) <= 0""",
        """((2.0 * v0)) <= 0""",
        """((2.0 * v1)) <= 0""",
    ]

    for n, string in enumerate(string_equals):
        print(c.constraints[n], '==', string, '?')
        assert(str(c.constraints[n]) == str(string))
        print('SUCCESS!')

    reset_symbols()

def test_quad_over_lin_x1_graph():

    v0 = Variable(name = 'v0')
    s0 = Symbol(name = 's0')

    x = v0
    expr = quad_over_lin([1], x)

    constr = (expr <= s0)  # effectively in relaxed smith form

    print('Graph form of', end = '')
    print(constr)
    constr = constr.graph_form()
    print('-->', constr)
    print()

    assert(constr.dims == 3)
    assert(type(constr) is SecondOrderConeConstraint)
    assert(all([type(c) is le for c in constr.constraints]))

    c = constr # keeps below names same as other tests

    string_equals = [
        """((-1.0 * v0) + (-1.0 * s0)) <= 0""",
        """((1.0 * v0) + (-1.0 * s0)) <= 0""",
        """(2.0) <= 0"""  # although this constraint is weird, it is required
    ]

    for n, string in enumerate(string_equals):
        print(c.constraints[n], '==', string, '?')
        assert(str(c.constraints[n]) == str(string))
        print('SUCCESS!')

    reset_symbols()

def test_quad_over_lin_comp_graph():

    v0 = Variable(name = 'v0')
    v1 = Variable(name = 'v1')
    s0 = Symbol(name = 's0')

    x = [(3-v0), (1+v1)]
    expr = quad_over_lin(x, 1)

    constr = (expr <= s0)  # effectively in relaxed smith form

    print('Graph form of', end = '')
    print(constr)
    constr = constr.graph_form()
    print('-->', constr)
    print()

    n = len(x)
    assert(constr.dims == n + 2)
    assert(type(constr) is SecondOrderConeConstraint)
    assert(all([type(c) is le for c in constr.constraints]))

    c = constr # keeps below names same as other tests

    string_equals = [
        """(-1.0 + (-1.0 * s0)) <= 0""",
        """(1.0 + (-1.0 * s0)) <= 0""",
        """(6.0 + (-2.0 * v0)) <= 0""",
        """(2.0 + (2.0 * v1)) <= 0""",
    ]

    for n, string in enumerate(string_equals):
        print(c.constraints[n], '==', string, '?')
        assert(str(c.constraints[n]) == str(string))
        print('SUCCESS!')

    reset_symbols()

def test_square_graph():

    v0 = Variable(name = 'v0')
    s0 = Symbol(name = 's0')

    expr = square(v0)

    constr = (expr <= s0)  # effectively in relaxed smith form

    print('Graph form of', end = '')
    print(constr)
    constr = constr.graph_form()
    print('-->', constr)
    print()

    n = max(v0.shape[0], v0.shape[1])
    assert(constr.dims == n + 2)
    assert(type(constr) is SecondOrderConeConstraint)
    assert(all([type(c) is le for c in constr.constraints]))

    c = constr # keeps below names same as other tests

    string_equals = [
        """(-1.0 + (-1.0 * s0)) <= 0""",
        """(1.0 + (-1.0 * s0)) <= 0""",
        """((2.0 * v0)) <= 0"""
    ]

    for n, string in enumerate(string_equals):
        print(c.constraints[n], '==', string, '?')
        assert(str(c.constraints[n]) == str(string))
        print('SUCCESS!')

    reset_symbols()

def test_max_graph():

    v0 = Variable((3,1), name = 'v0')
    s0 = Symbol((3,1),   name = 's0')

    expr = maximum(v0)

    constr = (expr <= s0)  # effectively in relaxed smith form

    print('Graph form of', end = '')
    print(constr)
    constr = constr.graph_form()
    print('-->', constr)
    print()
    print('Expanded:')
    constrs = constr.expand()
    for con in constrs:
        print('-->', con)
    print()

    assert(constr.dims == 1)
    assert(type(constr) is SecondOrderConeConstraint)
    assert(all([type(c) is le for c in constr.constraints]))

    string_equals = [
        ["""((-1.0 * s0[0][0]) + (1.0 * v0[0][0])) <= 0"""],
        ["""((-1.0 * s0[1][0]) + (1.0 * v0[1][0])) <= 0"""],
        ["""((-1.0 * s0[2][0]) + (1.0 * v0[2][0])) <= 0"""],
    ]

    for i, c in enumerate(constrs):
        for n, string in enumerate(string_equals[i]):
            print(c.constraints[n], '==', string, '?')
            assert(str(c.constraints[n]) == str(string))
            print('SUCCESS!')

    reset_symbols()

def test_min_graph():

    v0 = Variable((3,1), name = 'v0')
    s0 = Symbol(name = 's0')

    expr = minimum(v0)  # concave

    constr = (expr >= s0)  # effectively in relaxed smith form

    print('Graph form of', end = ' ')
    print(constr)
    constr = constr.graph_form()
    print('-->', constr)
    print()
    print('Expanded:')
    constrs = constr.expand()
    for con in constrs:
        print('-->', con)
    print()

    assert(constr.dims == 1)
    assert(type(constr) is SecondOrderConeConstraint)
    assert(all([type(c) is le for c in constr.constraints]))

    string_equals = [
        ["""((1.0 * v0[0][0]) + (-1.0 * s0)) <= 0"""],
        ["""((1.0 * v0[1][0]) + (-1.0 * s0)) <= 0"""],
        ["""((1.0 * v0[2][0]) + (-1.0 * s0)) <= 0"""],
    ]

    for i, c in enumerate(constrs):
        for n, string in enumerate(string_equals[i]):
            print(c.constraints[n], '==', string, '?')
            assert(str(c.constraints[n]) == str(string))
            print('SUCCESS!')

    reset_symbols()

def test_inv_pos_graph():

    v0 = Variable(name = 'v0')
    s0 = Symbol(name = 's0')

    expr = inv_pos(v0)

    constr = (expr <= s0)  # effectively in relaxed smith form

    print('Graph form of', end = ' ')
    print(constr)
    constr = constr.graph_form()
    print('-->', constr)

    c = constr

    assert(constr.dims == 3)
    assert(type(constr) is SecondOrderConeConstraint)
    assert(all([type(c) is le for c in constr.constraints]))

    string_equals = ["""((-1.0 * v0) + (-1.0 * s0)) <= 0""",
                     """((1.0 * v0) + (-1.0 * s0)) <= 0""",
                     """(2.0) <= 0"""]

    for n, string in enumerate(string_equals):
        print(c.constraints[n], '==', string, '?')
        assert(str(c.constraints[n]) == str(string))
        print('SUCCESS!')

    reset_symbols()

def test_geo_mean_graph():

    v0 = Variable(name = 'v0')
    s0 = Symbol(name = 's0')

    expr = geo_mean(v0, 2)  # concave

    constr = (expr >= s0)  # effectively in relaxed smith form

    print('Graph form of', end = ' ')
    print(constr)
    constr = constr.graph_form()
    print('-->', constr)

    assert(constr.dims == 3)
    assert(type(constr) is SecondOrderConeConstraint)
    assert(all([type(c) is le for c in constr.constraints]))

    c = constr

    string_equals = [
        """(-2.0 + (-1.0 * v0)) <= 0""",
        """(2.0 + (-1.0 * v0)) <= 0""",
        """((2.0 * s0)) <= 0""",
    ]

    for n, string in enumerate(string_equals):
        print(c.constraints[n], '==', string, '?')
        assert(str(c.constraints[n]) == str(string))
        print('SUCCESS!')

    reset_symbols()

def test_sqrt_graph():

    v0 = Variable(name = 'v0')
    s0 = Symbol(name = 's0')

    expr = sqrt(v0)  # concave

    constr = (expr >= s0)  # effectively in relaxed smith form

    print('Graph form of', end = ' ')
    print(constr)
    constr = constr.graph_form()
    print('-->', constr)

    assert(constr.dims == 3)
    assert(type(constr) is SecondOrderConeConstraint)
    assert(all([type(c) is le for c in constr.constraints]))

    c = constr

    string_equals = [
        """(-1.0 + (-1.0 * v0)) <= 0""",
        """(1.0 + (-1.0 * v0)) <= 0""",
        """((2.0 * s0)) <= 0""",
    ]

    for n, string in enumerate(string_equals):
        print(c.constraints[n], '==', string, '?')
        assert(str(c.constraints[n]) == str(string))
        print('SUCCESS!')

    reset_symbols()
