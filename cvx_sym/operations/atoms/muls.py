
from cvx_sym.operations import functions
from cvx_sym.errors import AlgebraError, ShapeError
from cvx_sym.conventions import scalar_shape
from cvx_sym import operations as ops
from cvx_sym import utilities as util
from cvx_sym import symbolic as sym
from . import Atom

def distribute(coeff, into):
    return ops.atoms.sums.sum(*[ (coeff*arg) for arg in into.args])

def distribute_vector(coeff, into):
    return sym.Vector([ (coeff*arg) for arg in into.args])

class smul(Atom):
    """ 1*1, 2*x, you  know, scalar multiplication!
        Define standard form: ((const) * (params))*(var)  """

    def __new__(cls, *args, simple = False):
        """
            Base level simplifications:
                0 * x => 0
                1 * x => x
        """

        # 0 * Something
        if any([arg.value == 0 for arg in args]):
            return sym.Constant(0)

        # 1 * Something
        elif any([arg.value == 1 for arg in args]):
            for n, arg in enumerate(args):
                if arg.value == 1:
                    return args[not n]

        # Constant * Constant
        elif all([type(arg) is sym.Constant for arg in args]):
            return sym.Constant(args[0].value * args[1].value)

        # Something * (Sum of Things) --> Something*Thing0 + Something*Thing1...
        elif any([type(arg) is ops.atoms.sums.sum for arg in args]):

            if type(args[0]) is ops.atoms.sums.sum:
                return distribute(args[1], args[0])

            elif type(args[1]) is ops.atoms.sums.sum:
                return distribute(args[0], args[1])
        elif any([type(arg) is sym.Vector for arg in args]):

            if type(args[0]) is sym.Vector:
                return distribute_vector(args[1], args[0])

            elif type(args[1]) is sym.Vector:
                return distribute_vector(args[0], args[1])
        else:
            return super().__new__(cls)

    def __init__(self, a, b, simple = False):

        self.args = [a, b]

        # Determine and cache the shape
        self.shape = util.determine_shape(self.args)

        # maintains parity with sum when getting offsets
        self.offset = sym.Constant(0)

        if not simple:
            self.simplify()

    def __str__(self):
        return '(' + str(self.args[0]) + ' * ' + str(self.args[1]) + ')'

    def __repr__(self):
        return '(smul) ' + str(self)

    def __getitem__(self, index):

        if self.shape != scalar_shape:

            # Find the argument with shape.
            # Must only be one of them else we would be matrix multiplying!
            for n, arg in enumerate(self.args):
                if arg.shape != scalar_shape:
                    # Return the shaped argument along with its coefficient
                    return self.args[not n] * arg[index]

        return self

    @property
    def curvature(self):
        """ Get the curvature of the expression, disregarding parameter sign """

        const,_,var = self.symbol_groups()

        if var == []:
            return 0 # AFFINE

        if const != []:
            const = const[0]
        else:
            const = sym.Constant(1)  # only used for curvature analysis

        v = var[0]  # we know that vars is at most of length 1
                     # because var * var is not valid DCP!

        if const.value > 0:
            return v.curvature  # POSITIVE * (CONVEX/CONCAVE)

        elif const.value < 0:
            if v.curvature > 0:  # NEGATIVE * CONVEX
                return -1

            elif v.curvature < 0:  # NEGATIVE * CONCAVE
                return +1

        return 0

    def graph_form(self, scalar):
        """ Get the graph form the same way we got the curvature.

            Otherwise, non-1 constants or any parameters ain't able to multiply
            nonlinear functions (see Smith Form), so if there are, raise hell

            Since by design we must represent all inequalities as (<=),
                representing t >= concave(x) with only (<= 0) expressions means we
                will end up with a non-1 constant on concave: t - concave(x) <= 0

                To overcome this, we specifically check for -1 constant on functions.
                If found, we activate the function's graph form, given the negative
                of the scalar t,
                    because: t - concave(x) <= 0 >>>> -t + concave(x) >= 0,
                    and the epigraph form of a concave function expects the latter.
        """

        const,parms,var = self.symbol_groups()

        if parms != []:
            raise(AssertionError(str(self) + ' has a parameter in args'))

        if var != [] and var[0].curvature < 0 and const[0].value == -1.0:
            return var[0].graph_form(- scalar)

        if const[0].value != 1.0:
            raise(AssertionError(str(self) + ' has a non-1 constant in args'))

        return var[0].graph_form(scalar)

    def vars_in_args(self):
        """ Get variables involved in this multiplier, or None """

        for arg in self.args:

            if arg.is_var:
                return arg

            elif issubclass( type(arg), Atom ):  # sum or mul

                vars = arg.vars_in_args()
                if vars is not None:
                    return vars

    def symbol_groups(self):

        c, p, v = [], [], []
        for arg in self.args:

            if type(arg) is sym.Constant:
                c.append(arg)

            elif type(arg) is sym.Parameter:
                p.append(arg)

            elif arg.is_var:
                v.append(arg)

            # Treat functions like vars, at least as far as standard form goes
            # . Unless they are parametric functions, then treat like parameter
            elif issubclass(type(arg), functions.Function):

                if not arg.parametric:
                    v.append(arg)
                else:
                    p.append(arg)

            elif type(arg) is smul:

                ci, pi, vi = arg.symbol_groups()

                c += ci
                p += pi
                v += vi

            elif type(arg) == sum:
                raise(AlgebraError(str(self)+" is an smul with a sum argument."
                    " Distribution did not occur in the __new__ method."))

        return c, p, v

    def simplify(self):
        """ Simplify into standard form: ((const) * (params))*(var) """

        c, p, v = self.symbol_groups()

        coeffs = c[0] if len(c) > 0 else sym.Constant(1)
        vars   = v[0] if len(v) > 0 else None

        for const in c[1:]:
            coeffs = smul(coeffs, const, simple = True)

        for param in p:
            coeffs = smul(coeffs, param, simple = True)

        for variable in v[1:]:
            vars = smul(vars, variable, simple = True)

        if vars is not None:
            self.args[0] = coeffs
            self.args[1] = vars

        else:

            # Adopt inner arguments if no variables
            if hasattr(coeffs, 'args'):
                self.args[0] = coeffs.args[0]
                self.args[1] = coeffs.args[1]

            else:
                self.args[0] = coeffs
                self.args[1] = sym.Constant(1)

        return self  # no guarantees that this return will have an audience

    def coefficient_of(self, var):
        """ Get the coefficient of this variable. Assumes standard form. """
        """ If you're expecting a factored coefficient, you're doing it wrong!
                Instead, use the factor_coefficients() method of sums.sum """

        if var is self.args[1]:
            return self.args[0]
        else:
            return sym.Constant(0)

    def assign_values(self, parameters):
        """ Assign any parameters with real values """

        news = []
        for n, val in enumerate(self.args):

            #print('    val',val,type(val), hasattr(val,'index'))

            if type(val) is sym.Constant:
                self.args[n] = (val.value)

            elif type(val) in [float, int]:
                news.append(val)

            elif type(val) is sym.Parameter:
                try:
                    #print('>>',parameters[val.name])
                    self.args[n] = (parameters[val.name])
                except KeyError:
                    raise(KeyError('Parameter named '+ val.name +
                                    'not supplied'))

            elif issubclass(type(val), Atom):

                new = val.assign_values(parameters)
                if new:
                    self.args[n] = (new)

            elif issubclass(type(val), functions.Function):

                    if hasattr(val, 'parametric') and val.parametric:
                        news[n].append(val.value(parameters))

                    else:
                        raise(Exception(str(val) + 'non-parametric function '
                                                    'was found in a matrix'))

            else:
                raise(Exception('WARNING: No assignment category found for',val))
                self.args[n] = (val)

        return eval(str(self))
