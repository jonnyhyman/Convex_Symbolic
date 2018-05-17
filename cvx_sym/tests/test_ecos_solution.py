
"""
    Test the Matrices in ECOS and assert against their solution values:
        Minimize(square(norm(F*x - g))) ; F, g parameters
"""

from cvx_sym.symbolic import reset_symbols, Variable, Parameter
from cvx_sym.canonicalize import Canonicalize

import cvx_sym as cvx

import numpy as np

"""
    The following trivial tests are straightforward convex optimization
    problems which prove that their member functions work. Graded by outcome.
"""

def test_ecos_trivial_minimum():

    x = cvx.Variable(name='x')

    constraints   = [x >= -1]
    constraints  += [x <= +1]

    objective = cvx.minimum(x)
    objective = cvx.Minimize(objective)
    problem   = cvx.Problem(objective, constraints)

    canon = Canonicalize(problem, verbose=True)
    canon.assign_values({})

    solution = cvx.solve(canon, verbose = True)

    if solution:
        print('Solution obj:', solution['info']['pcost'])
        print('Solution   x:', solution['x'])

        assert( np.allclose(solution['x'], [-1.0, -1.0]) )

    reset_symbols()

def test_ecos_trivial_norm1():

    x = cvx.Variable((3,1),name='x')

    constraints   = [x >= -1]
    constraints  += [x <= +1]

    objective = cvx.norm(x, kind=1)
    objective = cvx.Minimize(objective)
    problem   = cvx.Problem(objective, constraints)

    canon = Canonicalize(problem, verbose=True)
    canon.assign_values({})

    solution = cvx.solve(canon, verbose = True)

    if solution:
        print('Solution obj:', solution['info']['pcost'])
        print('Solution   x:', solution['x'])

        assert( np.allclose(solution['x'][0:3], [0.0, 0.0, 0.0]) )

    reset_symbols()

def test_ecos_trivial_invpos():

    x = cvx.Variable( (2), name='x')

    constraints  = [x[0] >= 0]

    constraints += [x[1] >= x[0]]
    constraints += [x[1] >= cvx.inv_pos(x[0])]

    objective = x[1]
    objective = cvx.Minimize(objective)
    problem   = cvx.Problem(objective, constraints)

    canon = Canonicalize(problem, verbose=True)
    canon.assign_values({})

    solution = cvx.solve(canon, verbose = True)

    if solution:

        print('Solution obj:', solution['info']['pcost'])
        print('Solution   x:', solution['x'][0:2])

        assert( np.allclose(solution['x'][0:2], [1.0, 1.0]) )

    reset_symbols()


def test_ecos_trivial_geomean():

    x = cvx.Variable( (2), name='x')

    constraints  = [x[0] >= 0]

    constraints += [x[1] <= cvx.geo_mean(x[0], 5)]
    constraints += [x[1] >= 5]

    objective = x[0] + x[1]  # farthest down and left
    objective = cvx.Minimize(objective)
    problem   = cvx.Problem(objective, constraints)

    canon = Canonicalize(problem, verbose=True)
    canon.assign_values({})

    solution = cvx.solve(canon, verbose = True)

    if solution:

        print('Solution obj:', solution['info']['pcost'])
        print('Solution   x:', solution['x'][0:2])

        assert( np.allclose(solution['x'][0:2], [5.0, 5.0]) )

    reset_symbols()


def test_ecos_trivial_geomean():

    x = cvx.Variable( (2), name='x')

    constraints  = [x[0] >= 0]

    constraints += [x[1] <= cvx.sqrt(x[0])]
    constraints += [x[1] == 2]

    objective = x[0] + x[1]  # farthest down and left
    objective = cvx.Minimize(objective)
    problem   = cvx.Problem(objective, constraints)

    canon = Canonicalize(problem, verbose=True)
    canon.assign_values({})

    solution = cvx.solve(canon, verbose = True)

    if solution:

        print('Solution obj:', solution['info']['pcost'])
        print('Solution   x:', solution['x'][0:2])

        assert( np.allclose(solution['x'][0:2], [4.0, 2.0]) )

    reset_symbols()

"""
    The following non-trivial tests are QPs, LPs, and SOCPs designed to
    prove integration of components. We grade on outcome, not format.
"""

def test_ecos_leastsquares():

    x = cvx.Variable (name='x')
    F = cvx.Parameter(name='F')
    g = cvx.Parameter(name='g')

    objective = cvx.square(cvx.norm( F*x - g ))

    objective = cvx.Minimize(objective)
    problem   = cvx.Problem(objective, [])

    canon = Canonicalize(problem, verbose=True)

    # Set values of parameters
    parameters = {
        'F' : 42,
        'g' : 42,
    }

    canon.assign_values(parameters)

    solution = cvx.solve(canon, verbose = True)

    if solution:
        print('Solution obj:', solution['info']['pcost'])
        print('Solution   x:', solution['x'][0])

        assert( np.isclose(solution['x'][0], 1.0) )

    reset_symbols()

def test_ecos_leastsquares_constr():

    x = cvx.Variable ((3,1),name='x')
    F = cvx.Parameter((3,3),name='F')
    g = cvx.Parameter((3,1),name='g')

    U = cvx.Parameter((3,1),name='U')
    L = cvx.Parameter((3,1),name='L')

    constraints  = []
    objective = cvx.square(cvx.norm( F*x - g ))

    constraints += [ x <= U ]
    constraints += [ L <= x ]

    objective = cvx.Minimize(objective)
    problem   = cvx.Problem(objective, constraints)

    canon = Canonicalize(problem, verbose=True)

    # Set values of parameters
    parameters = {
        'F' : np.array([[1,2,3],[4,5,6],[7,8,9]]),
        'g' : np.array([[1],[2],[3]]),

        'U' : np.array([[42],[42],[42]]),
        'L' : np.array([[1],[2],[3]])
    }

    canon.assign_values(parameters)

    solution = cvx.solve(canon, verbose = True)

    if solution:
        print('Solution obj:', solution['info']['pcost'])

        # Gather the OG solution variables

        v_indices = [n for n, vn in enumerate(canon.vars.keys()) if 'x' in vn]
        x_solution = [solution['x'][i] for i in v_indices]

        print('Solution   x:', x_solution)
        print('Solution vec:', solution['x'])

        assert( np.allclose(x_solution, [1,2,3] ) )

    reset_symbols()

def test_ecos_chebyshevcenter():

    import cvx_sym as cvx

    n = 2
    m = 3

    r = cvx.Variable((1,1),name='r')
    x = cvx.Variable((n,1),name='x')

    A = cvx.Parameter((m,n),name='A')
    B = cvx.Parameter((m,1),name='B')

    objective = -r

    constraints  = [A[i,:].T * x + r * cvx.norm(A[i,:]) <= B[i] for i in range(m)]
    constraints += [r >= 0]

    objective = cvx.Minimize(objective)
    problem   = cvx.Problem(objective, constraints)

    canon = Canonicalize(problem, verbose=True)

    # Set values of parameters
    parameters = {
        'A' : np.array([[-1,1],[1,1],[0,-1]]),
        'B' : np.array([[3],[3],[0]]),
    }

    canon.assign_values(parameters)

    solution = cvx.solve(canon, verbose = True)

    if solution:
        print('Solution obj:', solution['info']['pcost'])

        # Gather the OG solution variables

        v_indices = [n for n, vn in enumerate(canon.vars.keys()) if 'x' in vn]
        x_solution = [solution['x'][i] for i in v_indices]

        print('Solution   x:', x_solution)
        print('Solution vec:', solution['x'])

        assert( np.allclose(x_solution, [0, 1.242641] ) )

    reset_symbols()

def test_ecos_polyhedradist():
    import cvx_sym as cvx

    n = 2  # number of dimensions
    m = 3  # number of lines defining polyhedron 1
    p = 3  # number of lines defining polyhedron 2

    x1 = cvx.Variable ((n,1),name='x1')
    x2 = cvx.Variable ((n,1),name='x2')

    A1 = cvx.Parameter((m,n),name='A1')
    A2 = cvx.Parameter((p,n),name='A2')
    B1 = cvx.Parameter((m,1),name='B1')
    B2 = cvx.Parameter((p,1),name='B2')

    objective = cvx.square(cvx.norm(x1 - x2))

    constraints  = [ A1[i,:].T * x1 <= B1[i] for i in range(m) ]
    constraints += [ A2[i,:].T * x2 <= B2[i] for i in range(p) ]

    objective = cvx.Minimize(objective)
    problem   = cvx.Problem(objective, constraints)

    canon = Canonicalize(problem, verbose=True)

    # Set values of parameters
    parameters = {
        'A1' : np.array([[-1,1],[1,1],[0,-1]]),
        'B1' : np.array([[3],[3],[0]]),

        'A2' : np.array([[.5,-1],[0,1],[+1,0]]),
        'B2' : np.array([[-3],[3],[5]]),
    }

    canon.assign_values(parameters)

    solution = cvx.solve(canon, verbose = True)

    if solution:
        print('Solution obj:', solution['info']['pcost'])

        # Gather the OG solution variables

        v_indices = [n for n, vn in enumerate(canon.vars.keys()) if 'x' in vn]
        x_solution = [solution['x'][i] for i in v_indices]

        print('Solution   x:', x_solution)
        print('Solution vec:', solution['x'])

        # Solution is found at intersection of polyhedra
        assert( np.allclose(x_solution, [0,3, 0,3],  atol=1e-4) )

        # So therefore the objective should be near zero
        assert( np.allclose(solution['info']['pcost'], 0.0 ) )

    reset_symbols()

def test_ecos_robustlp():

    import cvx_sym as cvx

    n = 2  # number of dimensions
    m = 3  # number of elementwise elements

    x = cvx.Variable ((n,1),name='x')

    A = cvx.Parameter((m,n),name='A')
    B = cvx.Parameter((m,1),name='B')
    C = cvx.Parameter((n,1),name='C')
    P = cvx.Parameter((m,n),name='P')

    objective = C.T * x

    constraints = [A[i].T * x + cvx.norm(P[i].T * x) <= B[i] for i in range(m)]

    objective = cvx.Minimize(objective)
    problem   = cvx.Problem(objective, constraints)

    canon = Canonicalize(problem, verbose=True)

    # Set values of parameters
    parameters = {
        'A' : np.array([[1,1],[1,1],[1,1]]),
        'B' : np.array([[3],[3],[3]]),

        'C' : np.array([[.1],[.2]]),
        'P' : np.array([[1,2],[3,4],[5,6]])
    }

    canon.assign_values(parameters)

    solution = cvx.solve(canon, verbose = True)

    if solution:
        print('Solution obj:', solution['info']['pcost'])

        # Gather the OG solution variables

        v_indices = [n for n, vn in enumerate(canon.vars.keys()) if 'x' in vn]
        x_solution = [solution['x'][i] for i in v_indices]

        print('Solution   x:', x_solution)
        print('Solution vec:', solution['x'])

        assert( np.allclose(x_solution, [3,-3] ) )

    reset_symbols()

def test_ecos_control():

    import cvx_sym as cvx


    n = 2
    m = 2
    T = 5

    x = cvx.Variable((n, T+1), name='x')
    u = cvx.Variable((m, T), name='u')

    x_0 = cvx.Parameter((n,1), name='x_0')
    A = cvx.Parameter((n,n), name='A')
    B = cvx.Parameter((n,m), name='B')

    states = []
    constraints  = [ x[:,T] == 0 ]
    constraints += [ x[:,0] == x_0[:,0] ]

    for t in range(T):


        constraints += [ x[:,t+1] == A*x[:,t] + B*u[:,t] ]
        constraints += [ cvx.norm(u[:,t], kind = 'inf') <= 1 ]

        cost = cvx.sum_squares(x[:,t+1]) + cvx.sum_squares(u[:,t])
        states.append( cost )

    # sums problem objectives and concatenates constraints.
    objective = cvx.sum(states)
    objective = cvx.Minimize(objective)
    problem   = cvx.Problem(objective, constraints)

    canon = Canonicalize(problem, verbose = 1 )

    np.random.seed(1)
    alpha = 0.2
    beta = 5

    A_set = np.eye(n) + alpha*np.random.randn(n,n)
    B_set = np.random.randn(n,m)
    x_0_set = beta*np.random.randn(n,1)

    # Set values of parameters
    parameters = {
        'A' : A_set,
        'B' : B_set,
        'x_0' : x_0_set,
    }

    canon.assign_values(parameters)

    solution = cvx.solve(canon, verbose = True)

    if solution:

        obj = solution['info']['pcost']

        # Gather the OG solution variables

        v_indices = {n:vn for n, vn in enumerate(canon.vars.keys()) if 'x' in vn}
        x_solution = [solution['x'][i] for i in v_indices.keys()]

        print('Solution obj:', obj)
        print('Solution   x:',x_solution)
        print('Solution vec:', solution['x'])

        assert( np.isclose(obj, 2.770957684125057) )

    reset_symbols()

def full_scale_ecos_control():

    """ Takes a long time to canonicalize, so don't run as routine test """

    import cvx_sym as cvx

    n = 8
    m = 2
    T = 50

    x = cvx.Variable((n, T+1), name='x')
    u = cvx.Variable((m, T), name='u')

    x_0 = cvx.Parameter((n,1), name='x_0')
    A = cvx.Parameter((n,n), name='A')
    B = cvx.Parameter((n,m), name='B')

    states = []
    constraints  = [ x[:,T] == 0 ]
    constraints += [ x[:,0] == x_0[:,0] ]

    for t in range(T):


        constraints += [ x[:,t+1] == A*x[:,t] + B*u[:,t] ]
        constraints += [ cvx.norm(u[:,t], kind = 'inf') <= 1 ]

        cost = cvx.sum_squares(x[:,t+1]) + cvx.sum_squares(u[:,t])
        states.append( cost )

    # sums problem objectives and concatenates constraints.
    objective = cvx.sum(states)
    objective = cvx.Minimize(objective)
    problem   = cvx.Problem(objective, constraints)

    canon = Canonicalize(problem, verbose = 1 )

    np.random.seed(1)
    alpha = 0.2
    beta = 5

    A_set = np.eye(n) + alpha*np.random.randn(n,n)
    B_set = np.random.randn(n,m)
    x_0_set = beta*np.random.randn(n,1)

    # Set values of parameters
    parameters = {
        'A' : A_set,
        'B' : B_set,
        'x_0' : x_0_set,
    }

    canon.assign_values(parameters)

    solution = cvx.solve(canon, verbose = True)

    if solution:

        obj = solution['info']['pcost']

        # Gather the OG solution variables

        v_indices = [n for n, vn in enumerate(canon.vars.keys()) if 'x' in vn]
        x_solution = [solution['x'][i] for i in v_indices]

        print('Solution obj:', obj)
        print('Solution   x:', x_solution)
        print('Solution vec:', solution['x'])

        assert( np.isclose(obj, 64470.59019495451) )

    reset_symbols()

if __name__ == '__main__':
    from time import time
    start = time()
    """
    test_ecos_leastsquares()
    test_ecos_leastsquares_constr()
    test_ecos_polyhedradist()
    test_ecos_robustlp()
    """
    #test_ecos_chebyshevcenter()
    test_ecos_control()

    print('Time to solve:', time()-start)
