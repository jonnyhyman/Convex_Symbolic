from cvx_sym.symbolic import (Variable, Parameter, Constant, Symbol,
                                reset_symbols)
from cvx_sym.problem import Minimize, Problem
from cvx_sym.canonicalize import Canonicalize

from cvx_sym.operations.functions.quads import square
from cvx_sym.operations.functions.norms import norm
from cvx_sym.operations.atoms import sums, muls
from cvx_sym.constraints import eq, le

"""
    Test the Relaxed Forms of the examples:
        Minimize(square(v0))
        Minimize(square(norm(v0)))
        Minimize(square(norm(v0))) with v0 >= p0
        Minimize(square(norm(v0))) with v0 >= square(v1)
        Minimize(square(norm(F*x - g))) with x >= p0
"""


def test_relax_sq():
    """ Relaxed Form of Minimize(square(v0)) """

    v0 = Variable(name = 'v0')
    v1 = Variable(name = 'v1')

    p0 = Parameter(name = 'p0')
    p1 = Parameter(name = 'p1')

    obj = square(v0)
    con = []

    p = Problem(Minimize(obj), con)
    c = Canonicalize(p, verbose=True, only='relax')

    assert( len(c.constraints) == 1 )
    assert( c.objective is not obj   )
    assert( 'sym0' == c.objective.name )

    assert(type(c.constraints[0]) is le)

    assert(c.constraints[0].expr.args[0].curvature == 0)
    assert(c.constraints[0].expr.args[1].curvature == +1)
    assert(c.constraints[0].expr.args[1].symbol_groups()[0][0].value == 1.0)
    assert(c.constraints[0].expr.args[1].symbol_groups()[2][0].name == 'square')

    reset_symbols()

def test_relax_sq2norm():
    """ Relaxed Form of Minimize(square(norm(v0))) """

    v0 = Variable(name = 'v0')
    v1 = Variable(name = 'v1')

    p0 = Parameter(name = 'p0')
    p1 = Parameter(name = 'p1')

    obj = square(norm(v0))
    con = []

    p = Problem(Minimize(obj), con)
    c = Canonicalize(p, verbose=True, only='relax')

    assert( c.objective is not obj   )
    assert( 'sym0' == c.objective.name )

    # Let up on the testing rigor a bit, now that we checked core fundamentals

    string_equals = [
        """((-1.0 * sym1) + (1.0 * norm2(v0))) <= 0""",
        """((-1.0 * sym0) + (1.0 * square(sym1))) <= 0""",
    ]

    for n, string in enumerate(string_equals):
        print(c.constraints[n],'==',string,'?')
        assert(str(c.constraints[n]) == string)
        print('SUCCESS!')

    reset_symbols()

def test_relax_sq2norm_constr():
    """ Relaxed Form of Minimize(square(norm(v0))) with v0 >= p0"""

    v0 = Variable(name = 'v0')
    v1 = Variable(name = 'v1')

    p0 = Parameter(name = 'p0')
    p1 = Parameter(name = 'p1')

    obj = square(norm(v0))
    con = [v0 >= p0]  # should become -v0 + p0 <= 0

    p = Problem(Minimize(obj), con)
    c = Canonicalize(p, verbose=True, only='relax')

    assert( type(c.constraints[-1]) is le)
    assert( type(c.constraints[-1].expr) is sums.sum)
    assert( type(c.constraints[-1].expr.args[0]) is muls.smul)
    assert( type(c.constraints[-1].expr.args[1]) is muls.smul)
    assert( c.objective is not obj   )
    assert( 'sym0' == c.objective.name )

    string_equals = [
        """((-1.0 * sym1) + (1.0 * norm2(v0))) <= 0""",
        """((-1.0 * sym0) + (1.0 * square(sym1))) <= 0""",
        """((-1.0 * v0) + (p0 * 1.0)) <= 0""",
    ]

    for n, string in enumerate(string_equals):
        print(c.constraints[n],'==',string,'?')
        assert(str(c.constraints[n]) == string)
        print('SUCCESS!')

    reset_symbols()

def test_relax_sq2norm_sqcon():
    """ Relaxed Form : Minimize(square(norm(v0))) with v0 >= square(v1) """

    v0 = Variable(name = 'v0')
    v1 = Variable(name = 'v1')

    p0 = Parameter(name = 'p0')
    p1 = Parameter(name = 'p1')

    obj = square(norm(v0 + v1))
    con = [v0 >= square(v1)]  # should become -v0 + square(v1) <= 0

    p = Problem(Minimize(obj), con)
    c = Canonicalize(p, verbose=True, only='relax')

    assert( type(c.constraints[-1]) is le)
    assert( type(c.constraints[-1].expr) is sums.sum)
    assert( type(c.constraints[-1].expr.args[0]) is muls.smul)
    assert( type(c.constraints[-1].expr.args[1]) is muls.smul)
    assert( c.objective is not obj   )
    assert( 'sym0' == c.objective.name )

    string_equals = [
        """(sym2 + (-1.0 * v0) + (-1.0 * v1)) == 0""",
        """((-1.0 * sym1) + (1.0 * norm2(sym2))) <= 0""",
        """((-1.0 * sym0) + (1.0 * square(sym1))) <= 0""",
        """((-1.0 * sym3) + (1.0 * square(v1))) <= 0""",
        """((-1.0 * v0) + (1.0 * sym3)) <= 0""",
    ]

    for n, string in enumerate(string_equals):
        print(c.constraints[n],' ???? ',string, end=' ')
        assert(str(c.constraints[n]) == string)
        print('... SUCCESS!')

    reset_symbols()


def test_relax_least_squares_constr():
    """ Relaxed Form : Minimize(square(norm(F*x - g))) with x >= p0 """

    x = Variable ((3,1),name='x')
    F = Parameter((3,3),name='F')
    g = Parameter((3,1),name='g')

    objective = square(norm( F*x - g ))

    objective = Minimize(objective)
    problem   = Problem(objective, [x >= 0])

    c = Canonicalize(problem, verbose=True, only='relax')

    string_equals = [
        """(sym2 + (-1.0 * matmul(F, x))) == 0""",
        """(sym3 + (-1.0 * sym2) + (g * 1.0)) == 0""",
        """((-1.0 * sym1) + (1.0 * norm2(sym3))) <= 0""",
        """((-1.0 * sym0) + (1.0 * square(sym1))) <= 0""",
        """((-1.0 * x)) <= 0""",
    ]

    for n, string in enumerate(string_equals):
        print(c.constraints[n],' ???? ',string, end=' ')
        assert(str(c.constraints[n]) == string)
        print('... SUCCESS!')

    reset_symbols()
