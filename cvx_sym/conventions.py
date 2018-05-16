
scalar_shape = (1,1)

def shape_conform(shape, tranpose = False):
    """ Shape conventions:
            If scalar : (1, 1)
            If vector : (N, 1) or (1, N)
            If matrix : (N, M)
    """

    # Scalar
    if shape == () or shape == (1, 1) or shape == (1,):
        return scalar_shape

    if type(shape) is int:
        shape = (shape,)

    shape = list(shape)

    if len(shape) == 1:
        if not tranpose:
            shape = [shape[0], 1]
        else:
            shape = [1, shape[0]]

    return tuple(shape)


class index_string:

    style = 'c'

    def __new__(cls, index):
        """
            Given some index, what should print/codegen it as?
            Driven largely by the language we're targeting.
        """

        if cls.style == 'math':
            math_style = ''.join([str(i) for i in index])
            return math_style

        elif cls.style == 'py':
            py_style = str(list(index))
            return py_style

        elif cls.style == 'c':
            c_style = ''.join(['['+str(i)+']' for i in index])
            return c_style
