from cvx_sym.conventions import scalar_shape
from cvx_sym.errors import ShapeError
from cvx_sym import symbolic as sym
import copy

def determine_shape(args):

    largest_shape = scalar_shape

    for arg in args:
        if arg.shape != scalar_shape:

            if (largest_shape != scalar_shape and largest_shape != arg.shape):
                raise(ShapeError('Shapes ' + str(largest_shape)
                                 + ' and ' + str(arg.shape) +' incompatible '
                                 'for args ' + str(args)))

            largest_shape = arg.shape

    return largest_shape

def list_pprint(L):

    if L is None:
        return str(L)

    line_count = 0
    string     = '['
    endl       = '\n'
    wrapped    = False

    for n, val in enumerate(L):

        line_count += len(str(val))

        string += str(val)
        if n < len(L)-1: string += ', '

        if line_count >= 70:
            string += endl
            line_count = 0
            wrapped = True

    string += ']'

    if wrapped:
        string += endl

    return string

def are_args_parametric(*args):

    if type(args) is sym.Vector and args.parametric:
        return True

    elif type(args) is sym.Parameter:
        return True

    elif type(args) in [tuple, list]:

        if all([type(a) is sym.Vector and a.parametric for a in args]):
            return True

        elif all([type(a) is sym.Parameter for a in args]):
            return True

def all_args_curvature(*args):

    if all([arg.curvature > 0 for arg in args]):
        return +1

    elif all([arg.curvature < 0 for arg in args]):
        return -1

    elif all([arg.curvature == 0 for arg in args]):
        return 0

    else:
        return None

def reshape_symbol(input, desired_shape):
    """ Reshape some form of symbol """
    # Since symbols always create their self.matrices in row-major order,
    # we can simply reassign the row and column lists with the new shape

    # First build out the entire self.matrix, preserves naming of elements
    input.__iter__()
    reshaped = copy.deepcopy(input)

    reshaped.matrix['row'] = []
    reshaped.matrix['col'] = []

    for i in range(desired_shape[0]):
        for j in range(desired_shape[1]):
            reshaped.matrix['row'] += [i]
            reshaped.matrix['col'] += [j]

    reshaped.shape = desired_shape
    return reshaped

def reshape_vector(input, desired_shape):
    """ Reshape some form of a vector """

    # First flatten, then shape from there

    elements = []

    for v in input:  # row major by definition

        if issubclass(type(v), sym.Vector):
            if v.shape[0] != 1 and v.shape[1] != 1:
                raise(ShapeError('Cannot have more than 2 dimensional Vector.'
                                ' Could unpack vector of shape'+str(v.shape)))
            elements += v.args

        else:
            elements += [v]

    # Now, shape the elements into Vectors of Vectors

    n = 0
    reshaped = []
    for i in range(desired_shape[0]):

        row_args = []

        for j in range(desired_shape[1]):

            row_args += [elements[n]]
            n += 1

        reshaped += [ sym.Vector(row_args) ]

    reshaped = sym.Vector(reshaped)

    # Override vector shape attribute so that it sees the full shape
    reshaped.shape = desired_shape

    return reshaped



def reshape(input, desired_shape):
    """ Reshape input matrix into the desired shape """

    # Ensure the input can be reshaped as desired

    if (not issubclass(type(input), sym.Symbol) and
        not issubclass(type(input), sym.Vector)) :
        raise(TypeError("Cannot reshape non-symbolic input of type ",
                        str(type(input))))

    in_space = 0
    for s in input.shape:
        in_space *= s

    out_space = 0
    for s in desired_shape:
        out_space *= s

    if (in_space != out_space):
        raise(ShapeError("Cannot reshape "  + str(input.shape) + " into "
                                            + str(desired_shape)))

    if issubclass(type(input), sym.Symbol):
        return reshape_symbol(input, desired_shape)

    elif issubclass(type(input), sym.Vector):
        return reshape_vector(input, desired_shape)
