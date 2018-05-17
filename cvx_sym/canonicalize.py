
from cvx_sym.operations.functions import Function, VectorFunction
from cvx_sym.conventions import index_string
from cvx_sym.operations.atoms import Atom
from cvx_sym.utilities import list_pprint
from cvx_sym.problem import Problem
from cvx_sym import symbolic as sym
from cvx_sym import sparse
import copy

from cvx_sym.constraints import (Constraint, eq, le, ge,
                                    SecondOrderConeConstraint)

class Canonicalize(Problem):
    """
        Canonicalizes the problem. Specifically,
            - Convert to Smith Form
            - Convert to Relaxed Smith Form
            - Convert to Graph Form (epigraph representations)
            - Convert to Canonical Matrix Form
                : Convention: (Ax == b) and (Gx <= h) matrices
    """

    def __init__(self, problem, verbose = False, only = ''):

        self.problem = problem
        self.verbose = verbose

        if self.verbose:
            print(self.problem)

        # May want to do this in the future to easily retrieve actual soln
        #self.gather_symbols()
        #self.original_vars = copy.deepcopy(self.vars)

        self.constraints = []
        self.smith_form()
        if (only == 'smith'): return

        self.relax_form()
        if (only == 'relax'): return

        self.graph_form()
        if (only == 'graph'): return

        self.canon_form()

    def gather_symbols(self):
        """ Rebuild self.vars, and self.parms """

        # All rise to praise the holy python gods for enabling argument order
        # preservation in Python3.6+ dictionaries.
        self.vars = { n:s for n, s in sym.symbols.items()
                        if ((type(s) in [sym.Variable, sym.Symbol]) and
                            s.matrix['val'] == [])  # hide parent symbols
                    }

        self.parms = { n:s for n, s in sym.symbols.items()
                        if (type(s) is sym.Parameter) }

    def smith(self, input, with_aux = 0, debug=0):
        """
            Traverse the expression tree, set aux vars equal at each node

            input    : the thing which is to be smith-formified
            with_aux : return aux variable (if applic.) instead of modif. input
            debug    : enable/disable verbose printing
        """

        if issubclass(type(input), sym.Symbol):
            if debug == 2: print('... Symbol', input)
            return input

        elif issubclass(type(input), sym.Vector):
            if debug == 2: print('... Vector', input)
            aux = sym.Symbol(input.shape)
            self.constraints += eq(aux, input).expand()
            # expand the constraint to ensure any scalar functions are
            #   applied elementwise from the beginning

            return aux

        elif issubclass(type(input), Function):
            if debug == 2: print('... Function', input)

            for n, arg in enumerate(input.args):

                # Non-Parametric Function/Atom/Symbol
                if (issubclass(type(arg), Function) or
                    issubclass(type(arg), Atom) or
                    issubclass(type(arg), sym.Symbol)):

                    if debug == 2: print('... ... With Func/Atom/Sym Arg')

                    aux = sym.Symbol(input.shape)
                    new = self.smith(arg, with_aux=1, debug=debug)

                    input.args[n] = new
                    self.constraints += [ eq(aux, input) ]

                    return aux

                # Non-Parametric Vector
                elif (issubclass(type(arg), sym.Vector) and not arg.parametric):

                    if debug == 2: print('... ... With Vector Arg')

                    aux = sym.Symbol(input.shape)
                    new = self.smith(arg, with_aux=1, debug=debug)

                    input.args[n] = new
                    self.constraints += [ eq(aux, input) ]

                    return aux

                else:
                    print('... ... Parametric Function',input)
                    return input

        elif issubclass(type(input), Atom):

            if debug == 2: print('... Atom', input)

            for n, arg in enumerate(input.args):
                new = self.smith(arg, debug=debug)
                input.args[n] = new

                if debug == 2: print('... ... Replace', arg,'with',new,'in',input)

            if with_aux:
                # will replace this atom in above expressions
                aux = sym.Symbol(input.shape)
                self.constraints += [ eq(aux, input) ]
                return aux
            else:
                return input

        elif issubclass(type(input), Constraint):

            if debug == 2: print('... Constraint', input)

            new = self.smith(input.expr, debug=debug)
            input.expr = new
            self.constraints += [ input ]

        else:
            if debug == 2: print('...', type(input))
            return input

    def smith_form(self):
        """ Convert to Smith Form """

        # Objective first
        self.objective = self.smith(self.problem.objective, debug = self.verbose)

        # Constraints next
        for constr in self.problem.constraints:
            self.smith(constr, debug = self.verbose)

        if self.verbose:
            print()
            print('----Smith Form----')
            print(self)

        self.gather_symbols()

    def relax_form(self):
        """ Convert to Relaxed Smith Form """

        for n, constr in enumerate(self.constraints):

            # If constraint args == 2, that means we *might* be looking at
            # a nonlinear equality constraint, eg. aux + (-1 * norm2()) == 0
            if len(constr.expr.args) == 2:

                lhs = constr.expr.args[0]
                rhs = (-1 * constr.expr.args[1])
                # evaluate rhs curvature as if it were *actually* on rhs
                #  (constraint.expr has it on the left hand side with -1 coeff)

                lhs_curve = lhs.curvature
                rhs_curve = rhs.curvature

                if lhs_curve > 0 and (rhs_curve < 0 or rhs_curve == 0):
                    # Convex <= Concave or Affine
                    self.constraints[n] = (lhs <= rhs)

                elif (lhs_curve > 0 or lhs_curve == 0) and rhs_curve < 0:
                    # Convex or Affine <= Concave
                    self.constraints[n] = (lhs <= rhs)

                # The next two are equivalent to the top two but >=, not <=

                elif (lhs_curve < 0 or lhs_curve == 0) and rhs_curve > 0:
                    # Concave or Affine >= Convex
                    self.constraints[n] = (lhs >= rhs)

                elif lhs_curve < 0 and (rhs_curve > 0 or rhs_curve == 0):
                    # Concave >= Convex or Affine
                    self.constraints[n] = (lhs >= rhs)

        if self.verbose:
            print()
            print('----Relax Form----')
            print(self)

    def graph_form(self):
        """ Convert to Graph Form (epigraph representations) """

        for n, constr in enumerate(self.constraints):
            self.constraints[n] = constr.graph_form()

        if self.verbose:
            print()
            print('----Graph Form----')
            print(self)

    def stuff(self, n, constr, matrix, vector):
        """ Generalized form of matrix stuffing """

        for m, v in enumerate(self.vars.values()):

            if constr.expr.has_var(v):
                matrix['row'].append(n)
                matrix['col'].append(m)
                matrix['val'].append(constr.expr.coefficient_of(v))

                if self.verbose == 1:
                    print(constr.expr,'>>>',v,'>>>', matrix['val'][-1])

        # -1 * offset because in constr.expr it's on the left hand side
        if constr.expr.offset is not None:

            if type(constr.expr.offset) in [sym.Constant, int, float]:
                vector.append(-1 * constr.expr.offset)

            elif len(constr.expr.offset) > 1:
                vector.append(-1 * sum(constr.expr.offset))
            else:
                vector.append(-1 * constr.expr.offset[0])
        else:
            vector.append( 0.0 )

    def stuff_form(self):
        """ Stuff matrices with their coefficients / factors """

        self.matrix_names = ['c', 'A', 'b', 'G', 'h']
        self.dims = {'q':[], 'l':0}

        self.c = []
        self.A = {'row':[],'col':[],'val':[]}  # COO: row, col, val sparse
        self.G = {'row':[],'col':[],'val':[]}  # COO: row, col, val sparse
        self.b = []
        self.h = []

        # Build the c vector : all the coefficients making up the objective
        for v_name, v in self.vars.items():

            if (hasattr(self.objective,'coefficient_of') and
                self.objective.coefficient_of(v)):

                # Will add a Constant(0) if v not in self.objective
                self.c += [self.objective.coefficient_of(v)]

            elif v is self.objective:
                self.c += [1.0]

            else:
                self.c += [0.0]

        # Build Ax = b, and the first l rows of Gx = h
        ns = {'eq':0, 'le':0}
        for constr in self.constraints:

            if type(constr) is eq:
                self.stuff(ns['eq'], constr, self.A, self.b)  # Ax = b
                ns['eq'] += 1

            elif type(constr) is le:

                # Gx <= h, with l += 1
                self.stuff(ns['le'], constr, self.G, self.h)
                ns['le'] += 1

        self.dims['l'] = int(ns['le'])

        for constr in self.constraints:
            if type(constr) is SecondOrderConeConstraint:

                # Gx <= h, with q += [cone dims]
                # Yo dawg, we heard you like constraints...

                self.dims['q'] += [constr.dims]

                for cone_constr in constr.constraints:
                    self.stuff(ns['le'], cone_constr, self.G, self.h) # Gx <= h
                    ns['le'] += 1

        self.ns = dict(ns)  # save for later use

    def sparse_form(self):
        """ Convert COO Matrices to CSC / CCS form """

        n = len(self.vars)
        p = len(self.b)
        m = len(self.h)

        self.A = sparse.COO_to_CS(self.A, (p, n), 'col')
        self.G = sparse.COO_to_CS(self.G, (m, n), 'col')

        if self.A is None:
            self.b  =  None

    def canon_form(self):
        """ Convert to Canonical Matrix Form """

        # First, expand any linear stuff: matrix multiplies or elementwise ops
        #print('about to expand...')
        expand_at = {}
        for n, constr in enumerate(self.constraints):
            expand_at[n] = constr.expand()
            #print('n',n,constr, expand_at[n])

        self.constraints = []  # reset and refill

        # Apply the expansions
        for n, expanded in expand_at.items():
            self.constraints += expanded

        if self.verbose:
            print()
            print('----Canon Form----')
            print(self)

        self.gather_symbols()

        # Then, stuff the problem into the canonical matrices
        self.stuff_form()

        if self.verbose:
            print()
            print('----Matrices----')
            print('n', len(self.vars))
            print('v', list_pprint(list(self.vars.values())))
            print('c', list_pprint(self.c))
            print('A row', list_pprint(self.A['row']))
            print('A col', list_pprint(self.A['col']))
            print('A val', list_pprint(self.A['val']))
            print('b', list_pprint(self.b))
            print('G row', list_pprint(self.G['row']))
            print('G col', list_pprint(self.G['col']))
            print('G val', list_pprint(self.G['val']))
            print('h', list_pprint(self.h))
            print('dims', self.dims)

        self.sparse_form()

        if self.verbose:
            print()
            print('----Sparse----')
            print('n', len(self.vars))
            print('v', list_pprint(list(self.vars.values())))
            print('c', list_pprint(self.c))
            print('A', self.A)
            print('b', list_pprint(self.b))
            print('G', self.G)
            print()
            print('h', list_pprint(self.h))
            print('dims', self.dims)


    def assign_values(self, parameters, debug=0):
        """ Assign constants & parameters real values as given in parameters """

        # Preprocess parameters, replacing arrays with a bunch of elements

        elements = {}
        for name, item in parameters.items():
            if type(item) not in [float, int] and hasattr(item, '__getitem__'):

                # Only other reasonable thing is a numpy array

                for n in range(item.shape[0]):

                    if len(item.shape) > 1:
                        for m in range(item.shape[1]):

                            elem_name = name + index_string((n, m))
                            elements[elem_name] = item[n, m]

                    else:
                        elem_name = name + index_string((n, 0))
                        elements[elem_name] = item[n]

        parameters.update(elements)

        # Replace all constants with actual values

        matrices = [self.G.A, self.h, self.c]

        if self.A is not None:
            matrices += [self.A.A, self.b]

        news = [[],[],[],[],[]]
        for n, matrix in enumerate(matrices):

            if debug: print('---- IN MATRIX',n)

            for val in matrix:

                if debug: print('____________ VAL IS',val)

                if type(val) is sym.Constant:
                    news[n].append(val.value)

                elif type(val) in [float, int]:
                    news[n].append(val)

                elif type(val) is sym.Parameter:

                    try:
                        news[n].append(parameters[val.name])

                    except KeyError:
                        raise(KeyError('Parameter named '+ val.name +
                                        ' not supplied'))

                elif issubclass(type(val), Atom):

                    new = val.assign_values(parameters)
                    news[n].append(new)

                elif issubclass(type(val), Function):

                    if hasattr(val, 'parametric') and val.parametric:
                        news[n].append(val.value(parameters))

                    else:
                        raise(Exception(str(val) + 'non-parametric function '
                                                    'was found in a matrix'))

                else:
                    print('WARNING: No assignment category found for',val)
                    news[n].append(val)


        self.G.A, self.h, self.c = news[0:3]

        if self.A is not None:
            self.A.A, self.b = news[3:]
