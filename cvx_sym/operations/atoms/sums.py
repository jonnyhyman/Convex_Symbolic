
from cvx_sym.conventions import scalar_shape
from cvx_sym import operations as ops
from cvx_sym import utilities as util
from cvx_sym import symbolic as sym
from . import Atom

class sum(Atom):
    """ 1+1, 2+2+3, you know, the simple stuff! """

    def __new__(cls, *args, simple = False):
        """ Base level simplifications:
                (x) => x     """

        if len(args) == 1:
            return args[0]
        else:
            return super().__new__(cls)

    def __init__(self, *args, simple = False):

        self.args = list(args)

        for n, arg in enumerate(self.args):
            if type(arg) in [int, float]:
                self.args[n] = sym.Constant(arg)

        # Determine and cache the shape
        self.shape = util.determine_shape(self.args)

        if not simple:
            self.simplify()

    def __str__(self):
        sumstr  = '('

        for n, arg in enumerate(self.args):
            sumstr += str(arg)
            sumstr += ' + '

        sumstr = sumstr[:-3]  # no trailing plus
        sumstr += ')'
        return sumstr

    def __repr__(self):
        return '(sum) ' + str(self)

    def __getitem__(self, index):

        if self.shape != scalar_shape:

            # Get an element of an elementwise sum

            # Gather the indexed element of all shaped things,
            # and the unshaped things alone, then return the sum of them

            indexed = []

            for n, arg in enumerate(self.args):
                if arg.shape != scalar_shape:
                    indexed.append( arg[index] )

                else:
                    indexed.append( arg )

            return sum(*indexed)

        return self

    def take_stock(self, omit = None):
        """ Count all the things and compile the lists of coeffs and vars """
        self.nterms = len(self.args)
        self.vars   = self.vars_in_args()
        self.factor = self.factor_coefficients()
        self.coeffs = self.coefficients()
        self.offset = self.offsets()

    def take_stock_factors(self):
        """ Do everything take_stock does but omit self.coefficients(),
            because this run occurs from inside that function """
            
        self.nterms = len(self.args)
        self.vars   = self.vars_in_args()
        self.factor = self.factor_coefficients()
        self.offset = self.offsets()

    def simplify(self):
        """ Simplify the sum by simplifying the insides and flattening """

        # Ensure all constants are Constant symbols
        to_remove = []
        for n, arg in enumerate(self.args):

            if type(arg) in [ float, int ] and arg != 0:
                self.args[n] = sym.Constant(arg)

            elif type(arg) is sym.Constant and arg.value == 0:
                to_remove.append(n)

        for rm in to_remove:
            self.args.pop(rm)

        # Inner simplify
        for n, arg in enumerate(self.args):
            self.args[n] = arg.simplify()

        # Flatten into this sum
        new_args = []
        for arg in self.args:
            if type(arg) is sum:
                new_args += arg.args
            else:
                new_args += [arg]

        self.args = new_args
        self.take_stock()

        return self  # no guaranteed audience for this

    def factor_coefficients(self):
        """ Factoring for variables. At this point there are no sub-sums

            Cache the factors for later retrieval. Do not replace args.

            This is because replacing args would require smul to handle
                variable distribution differently than other distribution.

            In other words, smul's standard form of ((const) * (params))*(var)
                cannot capture factorization for variables.
        """

        factors = { n:[] for n,var in enumerate(self.vars) if var is not None}

        for n, var in enumerate(self.vars):
            #print('>>>',var)
            for arg in self.args:
                #print('...',arg)
                if type(arg) == ops.atoms.muls.smul:

                    coeff = arg.coefficient_of(var)

                    if coeff.value != 0:
                        #print('   -->',coeff)
                        factors[n].append(coeff)

                elif arg is var:
                    #print('   -->',1)
                    factors[n].append(sym.Constant(1))

        return factors

    def vars_in_args(self):
        """ Get variables involved in each term, or None """

        vars = []

        for arg in self.args:

            if arg.is_var:
                vars.append( arg )

            elif issubclass(type(arg), Atom):  # sum or mul
                vars.append( arg.vars_in_args() )

            else:
                vars.append( None )

        return vars

    def find_var(self, var):
        """ Find the args indices where this var might be found, including
            if it is found within a sub-sum or sub-mul """

        for n, arg in enumerate(self.args):

            if arg is var:
                return n

            elif issubclass(type(arg), Atom):

                vars = arg.vars_in_args()

                if var is vars:
                    return n

                elif type(vars) == list and var in vars:
                    return n

        return None
        raise(ValueError( "Variable "+str(var)+" not found in "+str(self) ))

    def coefficient_of(self, var, failraise = True):
        """ Find all the coefficients of a specific variable. """

        self.take_stock_factors()
        var_index = self.find_var(var)

        if var_index is not None:
            factors = self.factor[var_index]
            return sum(*factors)

        else:
            return sym.Constant(0)

    def coefficients(self):
        """ Coefficients of each variable in each term """

        coeffs = {}

        for n, var in enumerate(self.vars):
            if var is not None:
                if type(var) == dict:  # means there's vars below, in a sub sum
                    raise(NotImplemented('No sum coefficient recursion yet'))

                else:  # means the var was in self, the top level sum
                    coeffs[var.name] = self.coefficient_of(var)

        return coeffs

    def offsets(self):

        offsum = []

        for n, var in enumerate(self.vars):
            if var is None:
                offsum.append(self.args[n])

        if offsum != []:
            return offsum
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
