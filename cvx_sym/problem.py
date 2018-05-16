from cvx_sym.conventions import scalar_shape
from cvx_sym.operations.atoms import sums
from cvx_sym.errors import ShapeError

class Minimize(sums.sum):

    def __init__(self, args):

        if not hasattr(args, 'shape') or args.shape != scalar_shape:
            raise(ShapeError('Objectives must resolve to a scalar. Got: '
                            + str(args) + ' with shape ' + str(args.shape)))

        # Minimize is an explicit sum of terms
        super().__init__(args)


class Problem:
    def __init__(self, objective, constraints):

        self.objective   = objective
        self.constraints = constraints

    def __str__(self):

        endl = '\n'
        tabs = '    '
        cons = ''.join([tabs + str(c) + ',' + endl for c in self.constraints])

        return (endl + 'Objective:' + endl +
                tabs + str(self.objective) + endl*2 +
                'Constraints:' + endl + str(cons))

    def solve(self):
        raise(TypeError("Must use cvxpy, generated code, or ecos-python for "
                        "obtaining solution values, not cvx_sym"))
