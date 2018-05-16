from cvx_sym.conventions import shape_conform, scalar_shape, index_string
from cvx_sym import operations as ops
from cvx_sym import utilities as util
from cvx_sym import constraints
import copy

symbols = {}
unnamed = -1

def get_name(desired):
    global symbols, unnamed

    if desired in ['b','c','h']:
        print("WARNING: Symbol named " + str(desired) + " conflicts with "
                "canonical matrix name, renaming to " + str(desired) + '_')

        desired = desired + '_'

    if desired == 'sym':
        unnamed += 1
        return 'sym' + str(unnamed)

    elif desired in symbols.keys():
        raise(NameError("Symbol named " + str(desired) + " already exists"))

    return desired

def reset_sym_counter():
    global unnamed
    unnamed = -1

def reset_symbols():

    global symbols, unnamed

    for key in list(symbols.keys()):
        del symbols[key]

    unnamed = -1

class AtomicSymbol:
    """ Base defining all mathematical operations on symbols """

    curvature = 0

    def __add__(self, other):
        return ops.atoms.sums.sum(self, other)

    def __radd__(self, other):
        return ops.atoms.sums.sum(other, self)

    def __pos__(self):
        return self

    def __neg__(self):
        return (-1 * self)

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return other + (-1 * self)

    def __mul__(self, other):
        return ops.mul(self, other)

    def __rmul__(self, other):
        return ops.mul(other, self)

    def __div__(self, other):
        return ops.atoms.divs.div(self, other)

    def __rdiv__(self, other):
        return divs.div(other, self)

    def __eq__(self, other):
        return constraints.eq(self, other)

    def __le__(self, other):
        return constraints.le(self, other)

    def __ge__(self, other):
        return constraints.le(-1 * self, -1 * other)

    def simplify(self):
        return self

class Symbol(AtomicSymbol):
    """ Base for all mathematical symbols """

    # whether or not an instance of the symbol should be a solver variable
    is_var = True

    def __init__(self, shape = (), name = 'sym', index = None):

        self.shape = shape_conform(shape)
        self.value = NotImplemented
        self.name  = get_name(name)
        self.index = index

        # This COO sparse matrix holds elements within symbol
        self.matrix = {'row':[], 'col':[], 'val':[]}

        # This string holds the shape as a string for use in templates
        self.shape_string = index_string(self.shape)

        global symbols
        symbols[self.name] = self

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'Symbol ' + self.name + ''

    def __getitem__(self, index):

        if self.shape == scalar_shape:
            return self

        if type(index) is int or len(index) == 1:
            if self.shape[1] != 1:
                index = (index, slice(None,None,None))
            else:
                index = (index, 0)

        if any([ (type(i) is slice) for i in index]):

            for n, i in enumerate(index):
                if (type(i) is slice):

                    start, stop, step = i.start, i.stop, i.step

                    if (start == stop == step == None):
                        start, stop, step = 0, self.shape[n], 1

                    sliced = []
                    for m in range(start, stop, step):

                        here_index = list(index)
                        here_index[n] = m

                        sliced.append(self[here_index])

                    return Vector(sliced)

        # We may have previously put the element into the matrix
        # Try to go find it and return it
        for n, value in enumerate(self.matrix['val']):

            i = self.matrix['row'][n]
            j = self.matrix['col'][n]

            if i == index[0] and j == index[1]:
                return value

        else:
            # We have not previously put the element into the self.matrix
            # If shape of index is not out of bounds, add it to self.matrix

            if (index[0] < self.shape[0] and index[1] < self.shape[1]):

                element_name  = self.name
                element_name += index_string(index)
                element = type(self)(name = element_name, index = index)

                self.matrix['row'].append(index[0])
                self.matrix['col'].append(index[1])
                self.matrix['val'].append(element)

                # Then recurse to actually go grab that element
                return self[index]

            else:
                raise(IndexError('Index '+ str(index) +
                                ' out of bounds of shape '
                                + str(self.shape)))

    def __iter__(self):

        # Fill in the matrix of values entirely
        for n in range(self.shape[0]):
            for m in range(self.shape[1]):
                self[n, m]

        return iter(self.matrix['val'])

    @property
    def T(self):
        """ Transpose this symbol by returning a copy with swapped rows/cols"""

        # First build out the entire self.matrix, preserves naming of elements
        self.__iter__()

        transpose = copy.deepcopy(self)

        # Swap rows and columns
        transpose.matrix['row'], transpose.matrix['col'] = (
        transpose.matrix['col'], transpose.matrix['row'])

        # Broadcast modified shape
        transpose.shape = (transpose.shape[1], transpose.shape[0])

        return transpose


    def is_var(self):
        return (True if type(self) is Variable else False)

    def is_param(self):
        return (True if type(self) is Parameter else False)

    @property
    def curvature(self):
        return 0  # affine

    def clean(self):
        global symbols
        del symbols[self.name]

class Variable(Symbol):

    is_var = True

    def __repr__(self):
        return 'Variable ' + self.name + ''

class Parameter(Symbol):
    """ Parameters are symbols which are not variables.
        Assigned value at embedded code runtime only."""

    is_var = False

    def __repr__(self):
        return 'Parameter ' + self.name + ''

class Constant(AtomicSymbol):
    """ Constants are symbols which are assigned values immediately"""

    is_var = False
    shape = scalar_shape

    def __init__(self, value):
        self.value = float(value)

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return 'Constant(' + str(self.value) + ')'

    """ Below, override the rich compare methods for Constant comparisons """

    def __eq__(self, other):
        if type(other) is Constant:
            return self.value == other.value
        else:
            return super().__eq__(other)

    def __le__(self, other):
        if type(other) is Constant:
            return self.value <= other.value
        else:
            return super().__le__(other)

    def __ge__(self, other):
        if type(other) is Constant:
            return self.value >= other.value
        else:
            return super().__ge__(other)

    def __getitem__(self, index):
        return self

class Vector(AtomicSymbol):
    """ Holds objects inside which are going to be applied elementwise """

    is_var = False
    value  = NotImplemented

    def __init__(self, args):
        self.args = args
        self.shape = (len(args), 1)
        self.trans = False
        self.parametric = util.are_args_parametric(*args)
        self.curvature = util.all_args_curvature(*args)

    def __str__(self):
        string = '<' + ''.join([str(a)+', ' for a in self.args])[:-2] + '>'
        return string

    def __repr__(self):
        return 'Vector ' + str(self)

    def __getitem__(self, index):

        if not self.trans:  # Not a transpose. Act normal!
            i0 = index[0]
            i1 = index[1] if len(index) > 1 else 0
        else:
            i0 = index[1]  # Is a transpose. Swap row/col
            i1 = index[0] if len(index) > 1 else 0

        item = self.args[i0]

        if type(item) in [tuple, list]:
            return item[i1]

        else:
            return item

    def __iter__(self):
        return iter(self.args)

    @property
    def T(self):
        """ Transpose this vector by returning a copy with swapped rows/cols"""

        transpose = copy.deepcopy(self)

        # Broadcast modified shape
        transpose.shape = (transpose.shape[1], transpose.shape[0])
        transpose.trans = True

        return transpose
