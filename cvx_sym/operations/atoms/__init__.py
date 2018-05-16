
from cvx_sym import operations as ops
from cvx_sym import symbolic as sym

class Atom:

    value  = NotImplemented
    is_var = False

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
        return ops.atoms.div(self, other)

    def __rdiv__(self, other):
        return ops.atoms.div(other, self)

    def simplify(self):
        return self

    def graph_form(self):
        return self

    def has_var(self, var_to_find):

        for arg in self.args:

            if arg.is_var:

                if arg is var_to_find:
                    return True

            elif issubclass( type(arg), Atom ):  # sum or mul
                if arg.has_var(var_to_find):
                    return True

        return False
