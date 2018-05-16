from cvx_sym.symbolic import Variable, Parameter, Constant, reset_symbols
from cvx_sym.operations.functions.norms import norm
from cvx_sym.constraints import eq, ge, le
from cvx_sym.operations.atoms import sums, muls

def test_constraints():

    v0 = Variable(name = 'v0')
    v1 = Variable(name = 'v1')

    p0 = Parameter(name = 'p0')
    p1 = Parameter(name = 'p1')

    c = (v0 == 0)

    print(c)
    assert(type(c) == eq)
    assert(c.expr.args[0] is v0)

    c = (v0 <= 0)

    print(c)
    assert(type(c) == le)
    assert(c.expr.args[0] is v0)

    c = (v0 >= 0)

    print(c)
    assert(type(c) == le)
    assert(c.expr.args[0].args[0].value == -1)
    assert(c.expr.args[0].args[1] is v0)

    c = (v0 >= p0)

    print(c)
    assert(type(c) == le)
    assert(c.expr.args[0].args[0].value == -1)
    assert(c.expr.args[0].args[1] is v0)
    assert(c.expr.args[1].args[0] is p0)

    # TODO: ADD TESTS FOR ATOMS AND FUNCTIONS IN CONSTRAINTS!

    # ATOMS in Constraints

    c = (v1 + v0 == p0)

    print(c)
    assert(type(c) == eq)
    assert(str(c) == '(p0 + (-1.0 * v1) + (-1.0 * v0)) == 0')

    # FUNCTIONS in Constraints

    c = (norm(v1) <= v0)

    print(c)
    assert(type(c) == le)
    assert(c.expr.nterms == 2)
    assert(c.expr.args[1].args[1] is v0)


    reset_symbols()
