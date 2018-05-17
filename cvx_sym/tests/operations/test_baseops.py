
from cvx_sym.symbolic import (Variable, Parameter, Constant, Symbol,
                                symbols, reset_symbols)
from cvx_sym.operations.atoms import sums, muls

debug = 1

def test_summation():

    v0 = Variable(name = 'v0')
    v1 = Variable(name = 'v1')

    p0 = Parameter(name = 'p0')
    p1 = Parameter(name = 'p1')

    s = v0

    print(s)
    assert( type(s)  is Variable )
    assert( s is v0 )

    s = v0 + 0

    print(s)
    assert( type(s)  is sums.sum )
    assert( s.args[0] is v0 )
    assert( s.nterms == 1)

    s = v0 + v1

    print(s)
    assert( s.nterms == 2 )
    assert( type(s)  is sums.sum )
    assert( s.vars   == [v0, v1] )

    assert( s.coeffs['v0'].value == +1 )
    assert( s.coeffs['v1'].value == +1 )

    s = v0 + v1 + 1

    print(s)
    assert( s.nterms == 3 )
    assert( type(s)  is sums.sum )
    assert( s.vars   == [v0, v1, None] )
    assert( s.coeffs['v0'].value == +1 )
    assert( s.coeffs['v1'].value == +1 )
    assert( s.offset[0].value == 1)

    s = v0 + (-v1 + (1 + p1))

    print(s)
    assert( s.nterms == 4 )
    assert( type(s)  is sums.sum )
    assert( s.vars   == [v0, v1, None, None] )
    assert( s.coeffs['v0'].value == +1 )
    assert( s.coeffs['v1'].value == -1 )
    assert( (s.offset[0]).value == 1 )
    assert( (s.offset[1]) == p1 )


    s = 3*v0 + 2*v1 + 4

    print(s)
    assert( s.nterms == 3 )
    assert( type(s)  is sums.sum )
    assert( s.vars   == [v0, v1, None] )

    assert( s.coeffs['v0'].value == 3 )
    assert( s.coeffs['v1'].value == 2 )
    assert( s.offset[0].value == 4)

    s = 3*(v0 + 2*v1 + 4)

    print(s)
    assert( s.nterms == 3 )
    assert( type(s)  is sums.sum )
    assert( s.vars   == [v0, v1, None] )
    assert( s.coeffs['v0'].value == 3 )
    assert( s.coeffs['v1'].value == 6 )
    assert( s.offset[0].value == 12)

    s = p0*(v0 + p1*v1 + 4)

    print(s)
    assert( s.nterms == 3 )
    assert( type(s)  is sums.sum )
    assert( s.vars   == [v0, v1, None] )
    assert( s.coeffs['v0'] == p0 )
    assert( s.coeffs['v1'].args[0] == p0 )
    assert( s.coeffs['v1'].args[1] == p1 )
    assert( (s.offset[0]).args[0].value == 4.0 )
    assert( (s.offset[0]).args[1] == p0 )

    reset_symbols()

def test_scalarmul():

    s0 = Symbol(name = 's0')

    v0 = Variable(name = 'v0')
    v1 = Variable(name = 'v1')

    p0 = Parameter(name = 'p0')
    p1 = Parameter(name = 'p1')

    m = 1 * v0

    if debug:
        print(m)


    assert(m == v0)

    m = 2 * v0

    if debug:
        print(m)


    assert(m.coefficient_of(v0).value == 2)

    m = 3 * (2 * v0)

    if debug:
        print(m)


    assert(m.coefficient_of(v0).value == 6)

    m = p0 * (2 * v0)

    if debug:
        print(m)

    assert(type(m.coefficient_of(v0)) == muls.smul)
    assert(p0 in m.coefficient_of(v0).args)
    assert(2.0 in [arg.value for arg in m.coefficient_of(v0).args
                                     if hasattr(arg,'value')])

    m = (-1 * s0)

    if debug:
        print(m)

    assert(hasattr(m.args[0],'value'))
    assert(m.args[0].value == -1)
    assert(m.args[1] is s0)

    m = 3 * (2 * (4* v0))

    if debug:
        print(m)


    assert(m.coefficient_of(v0).value == 24)

    m = 3 * (2 * (4* v0 + 1*v1))

    if debug:
        print(m)


    assert(m.coefficient_of(v0).value == 24)
    assert(m.coefficient_of(v1).value == 6)

    m = 2*(3*(p0 + p1*v1) + 3*(2*p0 + 3*v0))

    v0_coeff = m.coefficient_of(v0)
    v1_coeff = m.coefficient_of(v1)

    if debug:
        print(m)


    assert(m.nterms == 4)
    assert(type(v0_coeff) == Constant)
    assert(type(v1_coeff) == muls.smul)
    assert(v0_coeff.value == 18)
    assert(p1 in v1_coeff.args)
    assert(6.0 in [arg.value for arg in v1_coeff.args if hasattr(arg,'value')])

    # Test factorization of terms
    m = 2*(3*(p0 + p1*v1) + 3*(2*p0 + 3*v0)) + v1

    v0_coeff = m.coefficient_of(v0)
    v1_coeff = m.coefficient_of(v1)

    if debug:
        print(m)

    assert(m.nterms == 5)
    assert(type(v0_coeff) == Constant)
    assert(type(v1_coeff) == sums.sum)
    assert(v0_coeff.value == 18)
    assert(v1_coeff.args[0].args[0].value == 6.0)
    assert(v1_coeff.args[0].args[1] == p1)
    assert(v1_coeff.args[1].value == 1.0)

    reset_symbols()

if __name__ == '__main__':
    test_summation()
    test_scalarmul()
