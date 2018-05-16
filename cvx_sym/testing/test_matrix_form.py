from cvx_sym.symbolic import (Variable, Parameter, Constant, Symbol,
                                reset_symbols)
from cvx_sym.problem import Minimize, Problem
from cvx_sym.canonicalize import Canonicalize

from cvx_sym.operations.functions.quads import square
from cvx_sym.operations.functions.norms import norm
from cvx_sym.operations.atoms import sums, muls
from cvx_sym.constraints import eq, le
from cvx_sym.sparse import COO_to_CS

"""
    Test the Matrix Forms of the examples:
        Minimize(square(v0))
        Minimize(square(norm(v0)))
        Minimize(square(norm(v0))) with v0 >= p0
        Minimize(square(norm(v0))) with v0 >= square(v1)
        Minimize(square(norm(F*x - g))) with x >= p0
"""

def test_matrix_sq():
    """ Matrix Form of Minimize(square(v0)) """

    v0 = Variable(name = 'v0')
    v1 = Variable(name = 'v1')

    p0 = Parameter(name = 'p0')
    p1 = Parameter(name = 'p1')

    obj = square(v0)
    con = []

    p = Problem(Minimize(obj), con)
    c = Canonicalize(p, verbose=True)

    assert(c.c == [0, 0, 1])
    assert(c.A is None)
    assert(c.b is None)

    assert_G =  {
                    'row':[0, 1, 2],
                    'col':[2, 2, 0],
                    'val':[Constant(-1.0), Constant(-1.0), Constant(2.0)]
                }


    assert_h = [Constant(1.0), Constant(-1.0), Constant(0.0)]
    assert_G = COO_to_CS(assert_G, (len(assert_h), len(c.c)), 'col')

    assert(c.G == assert_G)
    assert(all([c.h[n].value == t.value for n, t in enumerate(assert_h)]))

    assert(c.dims == {'q': [3], 'l': 0})

    reset_symbols()


def test_matrix_sq2norm():
    """ Matrix Form of Minimize(square(norm(v0))) """

    v0 = Variable(name = 'v0')
    v1 = Variable(name = 'v1')

    p0 = Parameter(name = 'p0')
    p1 = Parameter(name = 'p1')

    obj = square(norm(v0))
    con = []

    p = Problem(Minimize(obj), con)
    c = Canonicalize(p, verbose=True)


    assert_c = [0.0, 0.0, 1.0, 0.0]

    assert_A = COO_to_CS({'row':[],'col':[],'val':[]}, (0,4), 'col')

    assert_G =  {'row':[0, 1, 2, 3, 4],
                 'col':[3, 0, 2, 2, 3],
                 'val':[Constant(-1.0), Constant(1.0), Constant(-1.0),
                        Constant(-1.0), Constant(2.0)]}

    assert_h = [Constant(0.0), Constant(0.0), Constant(1.0),
                Constant(-1.0), Constant(0.0)]

    assert_G = COO_to_CS(assert_G, (len(assert_h), len(c.c)), 'col')

    assert(c.c == assert_c)
    assert(c.A is None)
    assert(c.b is None)

    assert(c.G == assert_G)
    assert(all([c.h[n].value == t.value for n, t in enumerate(assert_h)]))

    assert(c.dims == {'q': [2,3], 'l': 0})

    reset_symbols()


def test_matrix_sq2norm_constr():
    """ Matrix Form of Minimize(square(norm(v0))) with v0 >= p0"""

    v0 = Variable(name = 'v0')
    v1 = Variable(name = 'v1')

    p0 = Parameter(name = 'p0')
    p1 = Parameter(name = 'p1')

    obj = square(norm(v0))
    con = [v0 >= p0]  # should become -v0 + p0 <= 0

    p = Problem(Minimize(obj), con)
    c = Canonicalize(p, verbose=True)


    assert_c = [0.0, 0.0, 1.0, 0.0]

    assert_A = COO_to_CS({'row':[],'col':[],'val':[]}, (0,4), 'col')

    assert_G =  {'row': [0, 1, 2, 3, 4, 5], 'col': [0, 3, 0, 2, 2, 3],
                'val': [Constant(-1.0), Constant(-1.0), Constant(1.0),
                        Constant(-1.0), Constant(-1.0), Constant(2.0)]}

    assert_h = [(-1.0 * p0), Constant(0.0), Constant(0.0),
                Constant(1.0), Constant(-1.0), Constant(0.0)]

    assert_G = COO_to_CS(assert_G, (len(assert_h), len(c.c)), 'col')

    assert(c.c == assert_c)
    assert(c.A is None)
    assert(c.b is None)

    assert(c.G == assert_G)
    assert(all([c.h[n].value == t.value for n, t in enumerate(assert_h)]))

    assert(c.dims == {'q': [2, 3], 'l': 1})

    reset_symbols()

def test_matrix_sq2norm_sqcon():
    """ Relaxed Form : Minimize(square(norm(v0))) with v0 >= square(v1) """

    v0 = Variable(name = 'v0')
    v1 = Variable(name = 'v1')

    p0 = Parameter(name = 'p0')
    p1 = Parameter(name = 'p1')

    obj = square(norm(v0 + v1))
    con = [v0 >= square(v1)]  # should become -v0 + square(v1) <= 0

    p = Problem(Minimize(obj), con)
    c = Canonicalize(p, verbose=True)

    sym3 = [v for vn, v in c.vars.items() if vn == 'sym3'][0]

    assert_A = {'row': [0, 0, 0],
         'col': [0, 1, 4],
         'val': [Constant(-1.0), Constant(-1.0), Constant(1.0)]}

    assert_b = [Constant(0.0)]
    #assert_G = {'row': [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
    #     'col': [0, 5, 3, 4, 2, 2, 3, 5, 5, 1],
    #     'val': [Constant(-1.0), Constant(0.0), Constant(-1.0), Constant(1.0),
    #             Constant(-1.0), Constant(-1.0), Constant(2.0), Constant(-1.0),
    #             Constant(-1.0), Constant(2.0)]}

    assert_G = {'row': [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
         'col': [0, 5, 3, 4, 2, 2, 3, 5, 5, 1],
         'val': [Constant(-1.0), Constant(1.0), Constant(-1.0), Constant(1.0),
                 Constant(-1.0), Constant(-1.0), Constant(2.0), Constant(-1.0),
                 Constant(-1.0), Constant(2.0)]}

    assert_h = [ (-1.0 * sym3), Constant(0.0), Constant(0.0), Constant(1.0),
                Constant(-1.0), Constant(0.0), Constant(1.0),
                Constant(-1.0), Constant(0.0)]

    assert_h = [Constant(0.0), Constant(0.0), Constant(0.0), Constant(1.0),
                Constant(-1.0), Constant(0.0), Constant(1.0), Constant(-1.0),
                Constant(0.0)]

    assert_dims = {'q': [2, 3, 3], 'l': 1}

    assert_A = COO_to_CS(assert_A, (len(assert_b), len(c.c)), 'col')
    assert_G = COO_to_CS(assert_G, (len(assert_h), len(c.c)), 'col')

    assert_c = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    assert(c.c == assert_c)
    assert(c.A == assert_A)
    assert(c.b == assert_b)

    assert(c.G == assert_G)
    assert(all([c.h[n].value == t.value for n, t in enumerate(assert_h)]))

    assert(c.dims == assert_dims)

    reset_symbols()


def test_matrix_least_squares_constr():
    """ Minimize(square(norm(F*x - g))) with x >= p0 """

    x = Variable ((3,1),name='x')
    F = Parameter((3,3),name='F')
    g = Parameter((3,1),name='g')

    objective = square(norm( F*x - g ))

    objective = Minimize(objective)
    problem   = Problem(objective, [x >= 1])

    c = Canonicalize(problem, verbose=True)

    # Test for errors, not asserts, and for solution outcome (via ecos_solution)

    reset_symbols()
