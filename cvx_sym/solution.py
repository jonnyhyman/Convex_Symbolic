import importlib
import numpy as np

def solve(canon, verbose = False, fail_return = True):

    if importlib.util.find_spec("ecos"):
        import ecos
        import scipy  # assumed that if you have ecos, you have scipy
    else:
        print("WARNING: Module 'ecos' not found - can not solve")
        return None

    G = scipy.sparse.csc_matrix(*canon.G.to_scipy())
    c = np.array(canon.c)
    h = np.array(canon.h)

    if canon.A is not None:
        A = scipy.sparse.csc_matrix(*canon.A.to_scipy())
        b = np.array(canon.b)
    else:
        A = None
        b = None

    return ecos.solve(c, G, h, canon.dims, A, b, verbose=verbose)
