from cvx_sym.conventions import scalar_shape
from cvx_sym.operations.atoms import sums
from cvx_sym import symbolic as sym

class SecondOrderConeConstraint:

    """ Base class for any second order cone constraint """

    def __init__(self, epigraph_axes, dimensions):

        """
            Mathematical Background:
                Let Q[p] = {(t,x) in R x R[p-1] | norm2(x) <= t}

                This thing, Q is known as a "second order cone" of dimension p.
                We see that t is a scalar which bounds the 2 norm of x

                How does the solver know how to embed t and x in the same cone?
                What does that mean mathematically?

                We can see from the above statements that we should constrain:
                    t >= 0
                    x_i <= 0 for x_i in range(x)
        """

        self.dims = dimensions

        # First constraint always >= 0, because cone of dimension 1 is cone R+,
        # the cone of non-negative reals

        self.constraints = [ epigraph_axes[0] >= sym.Constant(0) ]

        # Rest of constraints always <= 0
        self.constraints += [ ax <= sym.Constant(0) for ax in epigraph_axes[1:]]

    def __str__(self):
        endl = '\n'
        tabs = '    '
        cons = ''.join([tabs*2 + str(c) + ',' + endl for c in self.constraints])
        return str(self.dims)+'D Cone' + endl + cons[:-2]

    def __repr__(self):
        return 'SecondOrderConeConstraint'# + ' ' + str(self)

    def expand(self, debug=0):
        """ Second Order Cone Constraints cannot be applied elementwise,
            but their content constraints can!"""

        new_constraints = []

        for c, constraint in enumerate(self.constraints):

            if constraint.expr.shape != scalar_shape:

                if debug: print('expanding',constraint.expr)

                for n in range(constraint.expr.shape[0]):
                    for m in range(constraint.expr.shape[1]):

                        exp_expr = [arg[n, m] for arg in constraint.expr.args]
                        exp_expr = sums.sum(*exp_expr)

                        new_constraints += [type(constraint)(exp_expr, 0)]

                        if debug: print('    ', (n,m), new_constraints[-1])

            else:
                new_constraints += [constraint]

            if c == 0:  # Handle multiple elementwise Q1 constraints in SOC
                if len(new_constraints) > 1:
                    # new.expr started as a >=, was turned into <=, and must be
                    #   turned back into its native format by negating
                    return [ SecondOrderConeConstraint([-new.expr],  1)
                                            for new in new_constraints]

        self.constraints = new_constraints  # refresh with expanded versions

        return [self]

class Constraint:

    """ Base class for all (in)equality constraints, eq, le """

    def __init__(self, lhs, rhs):
        """ lhs (compare to) rhs --> lhs - rhs (compare to) zero"""
        self.expr = lhs - rhs  # compared to zero

    def __str__(self):
        return str(self.expr) +' '+ self.op +' 0'

    def __repr__(self):
        return 'Constraint ' + str(self)

    def graph_form(self):
        """ Graph Form Representation

            If the constraint holds a nonlinear smith form constraint like
                [ (-sym + norm2(x)) <= 0 ], which always will have 2 args,

            Then return the epigraph representation of the constraint
                ... else self
        """

        if len(self.expr.args) == 2:
            for n, arg in enumerate(self.expr.args):
                if arg.curvature != 0:  # NON-AFFINE

                    # -1 * scalar because by graph form definitions,
                    # scalar is on the other side of the constraint
                    # ie. scalar >= convex or scalar <= concave
                    scalar = -1 * self.expr.args[not n]
                    epigraph_axes, dims = arg.graph_form(scalar)
                    soc = SecondOrderConeConstraint(epigraph_axes, dims)

                    return soc

        return self

    def expand(self, debug=0):
        """ Expanded Representation

            If the constraint holds something elementwise, return a list of
            the same constraint but applied elementwise.

            Each new constraint.expr should have scalar shape.
        """

        if self.expr.shape != scalar_shape:

            expanded = []

            if debug: print('expanding',self)

            for n in range(self.expr.shape[0]):
                for m in range(self.expr.shape[1]):

                    exp_expr = sums.sum(*[arg[n, m] for arg in self.expr.args])
                    expanded.append( type(self)(exp_expr, 0) )

                    if debug: print('    ',(n,m), expanded[-1])

            if debug: print('...return')
            return expanded

        return [self]

class eq(Constraint):
    op = '=='

class le(Constraint):
    op = '<='

class ge(Constraint):

    def __new__(cls, lhs, rhs):
        return le(-1 * lhs, -1 * rhs)
