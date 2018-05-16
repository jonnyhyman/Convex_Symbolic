
from cvx_sym.conventions import scalar_shape
from cvx_sym.symbolic import AtomicSymbol, Vector, Parameter
from cvx_sym import utilities as util

class Function(AtomicSymbol):
    """ Base class defining fundamental functional methods """

    is_var = False
    value  = NotImplemented
    name   = 'undefined'

    def __str__(self):

        if len(self.args) > 1:
            str_args = [ str(arg)+', ' for n, arg in enumerate(self.args) ]
            str_args[-1] = str_args[-1][:-2]  # goodbye trailing comma

        else:
            str_args = str(self.args[0])

        return self.name + '(' + ''.join(str_args) + ')'

    def __repr__(self):
        return "Function " + str(self)

    def __init__(self, *args):
        self.parametric = util.are_args_parametric(*args)

    def get_context(self):
        """ Provide templates with more context about a function """
        return {}

class VectorFunction(Function):
    """
        Input  : Vector
        Output : Scalar
    """

    shape = scalar_shape

class ScalarFunction(Function):
    """
        Input  : Scalar
        Output : Scalar
            -- : Vector inputs should be applied elementwise.
    """

    shape = scalar_shape

    def __new__(cls, args):

        # Vector input as : [list of things] OR shaped_symbol
        if type(args) in [tuple, list]:
            return Vector([ cls(arg) for arg in args ])

        elif args.shape != scalar_shape:
            return Vector([ cls(elem) for elem in args ])

        else:
            return super().__new__(cls)

class PureScalarFunction(Function):
    """
        Input  : Scalar
        Output : Scalar
            -- : Vectors should NOT BE applied elementwise ("PURE SCALAR")
    """

    shape = scalar_shape

    def __new__(cls, *args):
        return super().__new__(cls)
