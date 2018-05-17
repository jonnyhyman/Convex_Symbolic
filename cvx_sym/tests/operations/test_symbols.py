
from cvx_sym.symbolic import (Variable, Parameter, Constant, symbols,
                                reset_sym_counter, reset_symbols)

def test_live_symbols():

    v = Variable((10,), name = 'v')

    assert(v.shape == (10,1))
    assert(v.name == 'v')

    assert('v' in symbols.keys())

    reset_symbols()

    assert('v' not in symbols.keys())

    v = Variable((10,))

    assert(v.shape == (10,1))
    assert(v.name == 'sym0')

    assert('sym0' in symbols.keys())

    reset_symbols()

    assert('sym0' not in symbols.keys())

def test_shape():

    v = Variable(name = 'v')

    assert(v.shape == (1,1))
    assert(v.name == 'v')

    v.clean()

    v = Variable((10,), name = 'v')

    assert(v.shape == (10,1))
    assert(v.name == 'v')

    v.clean()

    v = Variable((10,10), name = 'v')

    assert(v.shape == (10,10))
    assert(v.name == 'v')

    v.clean()

def test_const():

    v = Constant(0)
    assert(v.value == 0.0)
