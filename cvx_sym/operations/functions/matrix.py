from cvx_sym.operations.functions import Function
from cvx_sym.conventions import scalar_shape
from cvx_sym.operations.atoms import sums
from cvx_sym.errors import ShapeError

class matmul(Function):

    name = 'matmul'
    curvature = 0

    def __new__(cls, a, b):

        # Do some shape checking : inner & outer dims match, and 2D shapes only

        if (a.shape[1] != b.shape[0] or len(a.shape) > 2 or len(b.shape) > 2):
            raise(ShapeError('Cannot matmul objects with shapes '+
                                str(a.shape) + ' and ' + str(b.shape)))

        # If we're making a matmul with output shape (1,1), just return
        # the (0,0) index of the output matrix! (the scalar outcome)

        n = a.shape[0]
        m = a.shape[1]  # = b.shape[0]
        p = b.shape[1]

        shape = (n,p)  # output shape
        mat = super().__new__(cls)
        mat.__init__(a,b)

        if shape == scalar_shape:
            return mat[0,0]

        else:
            return mat

    def __init__(self, a, b):
        super().__init__(a, b)

        self.args = [a, b]

        self.n = a.shape[0]
        self.m = a.shape[1]  # = b.shape[0]
        self.p = b.shape[1]

        self.shape = (self.n, self.p)  # output shape

    def __getitem__(self, index, alt_args = None):
        """ Retrieve an element c[i, j] of the matrix multiply outcome """

        if type(index) not in [tuple, list] or len(index) == 1 or len(index)>2:
            raise(IndexError('Must index matmul with two index arguments, got '
                                                                + str(index)))

        i, j = index
        a, b = self.args

        if alt_args is not None:
            a, b = alt_args

        c_ij = sums.sum(*[ a[i,k] * b[k,j] for k in range(self.m) ])

        return c_ij

    def expand(self):
        """ Return the symbolic value of the matrix multiply, which is
            really just a list of atoms """

        mat = [[None for j in range(self.p)] for i in range(self.n)]

        for i in range(self.n):
            for j in range(self.p):

                mat[i][j] = self[i, j]

        return mat


    def value(self):
        """ Return the symbolic value of the matrix multiply, which is
            really just a list of atoms """

        p_args = [[[ parameters[v.name] for v in row]
                                        for row in arg]
                                        for arg in self.args]

        mat = [[None for j in range(self.p)] for i in range(self.n)]

        for i in range(self.n):
            for j in range(self.p):

                mat[i][j] = self.__getitem__([i, j], alt_args = p_args)

        return mat
