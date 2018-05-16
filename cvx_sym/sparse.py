from cvx_sym.utilities import list_pprint

class COO_to_CS:

    def __new__(cls, COO, shape, format):

        if any([s == 0 for s in shape]):
            return None

        elif format.lower() not in ['csr','row','csc','ccs','col']:
            raise(ValueError('Format '+str(format)+' is not valid'))

        else:
            return super().__new__(cls)

    def __init__(self, COO, shape, format):

        self.shape = shape

        if format.lower() in ['csr', 'row']:
            self.form = 'Row'
            major = 'row'
            minor = 'col'

        elif format.lower() in ['csc', 'ccs', 'col']:
            self.form = 'Column'
            major  = 'col'
            minor  = 'row'

        ''' For an (m x n) dense matrix defined by COO, CSR format is:

        A = all nonzero entries. Needs to be sorted in "ROW-MAJOR" order
        JA = column index of each element of A
        IA has length m+1, and is recursively defined by:

            IA[0] = 0
            IA[i] = IA[i − 1] + (number of nonzero elements on the (i-1)-th row
                                                        in the original matrix)

            (Thus, the first m elements of IA store the index into A of the
            first nonzero element in each row of M, and the last element IA[m]
            stores NNZ, the number of elements in A, which can be also thought
            of as the index in A of first element of a phantom row just beyond
            the end of the matrix M.

            The values of the i-th row of the original
            matrix is read from the elements A[IA[i]] to A[IA[i + 1] − 1]
            (inclusive on both ends), i.e. from the start of one row to the
            last index just before the start of the next)

        '''

        # ------------------ Sort values and their columns by row index
        # ------------------ (row major order)

        self.A     = [x for n, x in sorted(zip(COO[major], COO['val']),
                                            key = lambda pair: pair[0])]

        maj_min = [(n, x) for n, x in sorted(zip(COO[major], COO[minor]),
                                            key = lambda pair: pair[0])]

        majs = [x[0] for x in maj_min]
        mins = [x[1] for x in maj_min]

        self.JA = mins
        self.IA = [0]

        for i in  range(self.shape[['row','col'].index(major)]):

            val_idxs_in_maj = [n for n, r in enumerate(majs) if r == i]
            self.IA.append( self.IA[-1] + len(val_idxs_in_maj)  )

    def __str__(self):

        endl = '\n'
        tabs = '    '

        string  = 'Compressed Sparse ' + self.form + ' ' + str(self.shape)
        string += endl + tabs + 'A  ' + list_pprint(self.A)
        string += endl + tabs + 'IA ' + list_pprint(self.IA)
        string += endl + tabs + 'JA ' + list_pprint(self.JA)
        return string

    def __repr__(self):
        return 'Compressed Sparse ' + self.form + ' ' + str(self.shape)

    def __eq__(self, other):
        """ Method to prove equality this object and other of the same type """

        if type(other) is type(self):
            if (other.form == self.form and
                other.shape == self.shape and
                self.A == other.A and
                self.IA == other.IA and
                self.JA == other.JA):
                return True

        return False

    def to_scipy(self):
        """ Returns the args tuple which can be directly unpacked into
            the initializer of a scipy.sparse.csc/csr_matrix """

        return (self.A, self.JA, self.IA), self.shape
