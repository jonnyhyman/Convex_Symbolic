
from cvx_sym.symbolic import Variable, Parameter, Symbol, Vector, reset_symbols
from cvx_sym.operations.functions.matrix import matmul
from cvx_sym.operations.functions.quads import square

from cvx_sym.operations.atoms import sums
from cvx_sym.operations import mul
from cvx_sym.constraints import le

from cvx_sym.utilities import reshape

def list_shape(L):
    if type(L[0]) is list:
        return len(L), len(L[0])
    else:
        return len(L), 1

def print_matrix(L):
    print('[')
    for row in L:
        print('    [ ', end='')
        for n, val in enumerate(row):
            if n < len(row) - 1:
                print(val, end=',    ')
            else:
                print(val, end='')

        print(' ],')
    print(']')

def test_mul_redirect():
    """ Test that mul redirects properly to matmul """

    a = Parameter((10,1), name = 'a')
    b = Variable((1,10), name = 'b')

    m = a * b

    assert(type(m) is matmul)
    assert(m.args[0] is a)
    assert(m.args[1] is b)

    reset_symbols()

def test_matmat():
    """ Test Matrix * Matrix product """

    a = Parameter((3,2), name = 'a')
    b = Variable((2,3), name = 'b')

    m = a * b

    print(m, 'with shapes', a.shape,'x', b.shape)

    expanded = m.expand()

    print_matrix(expanded)

    assert(list_shape(expanded) == m.shape)

    # Holds all the indices we assert must be in each ELEMENT of each atom
    ind = [[ [[(0,0),(0,0)], [(0,1),(1,0)]], [[(0,0),(0,1)], [(0,1),(1,1)]], [[(0,0),(0,2)], [(0,1),(1,2)]] ],
           [ [[(1,0),(0,0)], [(1,1),(1,0)]], [[(1,0),(0,1)], [(1,1),(1,1)]], [[(1,0),(0,2)], [(1,1),(1,2)]] ],
           [ [[(2,0),(0,0)], [(2,1),(1,0)]], [[(2,0),(0,1)], [(2,1),(1,1)]], [[(2,0),(0,2)], [(2,1),(1,2)]] ]]

    for n, row in enumerate(ind):
        for m, col in enumerate(row):
            for s, sumarg in enumerate(col):
                for a, mularg_shape in enumerate(sumarg):
                    assert(expanded[n][m].args[s].args[a].index == mularg_shape)

    reset_symbols()

def test_matvec():
    """ Test Matrix * Vector product """

    a = Parameter((3,3), name = 'a')
    x = Variable((3,1), name = 'x')

    m = a * x

    print(m, 'with shapes', a.shape,'x', x.shape)

    expanded = m.expand()

    print_matrix(expanded)

    assert(list_shape(expanded) == m.shape)

    # Holds all the indices we assert must be in each ELEMENT of each atom
    ind = [[ [[(0,0),(0,0)], [(0,1),(1,0)], [(0,2),(2,0)]], ],
           [ [[(1,0),(0,0)], [(1,1),(1,0)], [(1,2),(2,0)]], ],
           [ [[(2,0),(0,0)], [(2,1),(1,0)], [(2,2),(2,0)]], ]]

    for n, row in enumerate(ind):
        for m, col in enumerate(row):
            for s, sumarg in enumerate(col):
                for a, mularg_shape in enumerate(sumarg):
                    assert(expanded[n][m].args[s].args[a].index == mularg_shape)

    reset_symbols()

def test_matvec_transpose():
    """ Test Matrix * Vector product but where Vector is transposed """

    a = Parameter((3,3), name = 'a')
    x = Variable((1,3), name = 'x')

    m = a * x.T

    print(m, 'with shapes', a.shape,'x', (x.T).shape)

    expanded = m.expand()

    print_matrix(expanded)

    assert(list_shape(expanded) == m.shape)

    # Holds all the indices we assert must be in each ELEMENT of each atom
    ind = [[ [[(0,0),(0,0)], [(0,1),(0,1)], [(0,2),(0,2)]], ],
           [ [[(1,0),(0,0)], [(1,1),(0,1)], [(1,2),(0,2)]], ],
           [ [[(2,0),(0,0)], [(2,1),(0,1)], [(2,2),(0,2)]], ]]

    for n, row in enumerate(ind):
        for m, col in enumerate(row):
            for s, sumarg in enumerate(col):
                for a, mularg_shape in enumerate(sumarg):
                    print(expanded[n][m].args[s].args[a].index,'?',mularg_shape)
                    assert(expanded[n][m].args[s].args[a].index == mularg_shape)

    reset_symbols()

def test_matmul_constraint():
    """ Test that constraints with matmul are expanded elementwise in canon """

    p = Parameter((3,1), name = 'p')
    a = Parameter((3,3), name = 'a')
    x = Variable((3,1), name = 'x')

    expr = a * x

    print('expr', expr)
    constr = (expr <= p)

    print('constr', constr)
    constr = constr.expand()
    print('-->', constr)
    print()

    assert( len(constr) == 3 )
    assert( all([type(a) is le for a in constr]) )

    reset_symbols()

def test_elementwisesymbol_into_function():
    """ Test a shaped symbol input to a scalar function and
        the elementwise output """

    x = Variable((3,1), name = 'x')

    expr = square(x)

    print(expr)

    constr = (expr <= 0)

    print(constr)

    expand = constr.expand()

    print(expand)

    assert( type(constr.expr) is sums.sum )
    assert( type(constr.expr.args[0]) is Vector )
    assert( len(constr.expr.args[0].args) == 3 )
    assert( len(expand) == 3 )
    assert( all([type(a) is le for a in expand]) )

    reset_symbols()

def test_elementwiselist_into_function():
    """ Test a list input to a scalar function and the elementwise output """

    x = Variable(name = 'x')
    y = Variable(name = 'y')
    z = Variable(name = 'z')
    a = Variable(name = 'a')
    b = Variable(name = 'b')
    c = Variable(name = 'c')

    expr = square([x, y, z, a, b, c])

    print(expr)

    constr = (expr <= 0)

    print(constr)

    assert( type(constr.expr) is sums.sum )
    assert( type(constr.expr.args[0]) is Vector )
    assert( len(constr.expr.args[0].args) == 6 )

    reset_symbols()

def test_reshaping():
    """ Test reshaping a symbol """

    x = Symbol((6), name = 'x')

    print(x, x.shape, '-->', (3,2), end = ' --> ')

    x32 = reshape(x, (3,2))
    assert(x32.shape == (3,2))

    print(x32, x32.shape)

    # Must do string comparison because 'is' will not be true,
    # and '==' will try making a constraint
    assert(str(x32[0,0]) == str(x[0][0]))
    assert(str(x32[0,1]) == str(x[1][0]))
    assert(str(x32[1,0]) == str(x[2][0]))
    assert(str(x32[1,1]) == str(x[3][0]))
    assert(str(x32[2,0]) == str(x[4][0]))
    assert(str(x32[2,1]) == str(x[5][0]))

    print(x, x.shape, '-->', (2,3), end = ' --> ')

    x23 = reshape(x, (2,3))
    assert(x23.shape == (2,3))

    print(x23, x23.shape)

    # Must do string comparison because 'is' will not be true,
    # and '==' will try making a constraint
    assert(str(x23[0,0]) == str(x[0][0]))
    assert(str(x23[0,1]) == str(x[1][0]))
    assert(str(x23[0,2]) == str(x[2][0]))
    assert(str(x23[1,0]) == str(x[3][0]))
    assert(str(x23[1,1]) == str(x[4][0]))
    assert(str(x23[1,2]) == str(x[5][0]))

    y = Symbol((2,2), name = 'y')

    print(y, y.shape, '-->', (4,1), end = ' --> ')

    y_flat = reshape(y, (4,1) )

    print(y_flat, y_flat.shape)

    assert(str(y_flat[0,0]) == str(y[0][0]))
    assert(str(y_flat[1,0]) == str(y[0][1]))
    assert(str(y_flat[2,0]) == str(y[1][0]))
    assert(str(y_flat[3,0]) == str(y[1][1]))

    print(y, y.shape, '-->', (1,4), end = ' --> ')

    y_flat = reshape(y, (1,4) )

    print(y_flat, y_flat.shape)

    assert(str(y_flat[0,0]) == str(y[0][0]))
    assert(str(y_flat[0,1]) == str(y[0][1]))
    assert(str(y_flat[0,2]) == str(y[1][0]))
    assert(str(y_flat[0,3]) == str(y[1][1]))

    reset_symbols()

def test_reshaping_vector():
    """ Test reshaping a vector (slice of a symbol in this case)"""

    p = Parameter((2, (2*2)), name = 'p')

    sliced = p[1,:]

    print(sliced)

    shaped = reshape(sliced, (2,2))

    print(shaped)

    assert(str(shaped[0,0]) == str(p[1][0]))
    assert(str(shaped[0,1]) == str(p[1][1]))
    assert(str(shaped[1,0]) == str(p[1][2]))
    assert(str(shaped[1,1]) == str(p[1][3]))

    reset_symbols()
