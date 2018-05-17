
import cvxpy as cvx
import numpy as np
import scipy

from time import time
start = time()

x = cvx.Variable ((3,1),name='x')
F = cvx.Parameter((3,3),name='F')
g = cvx.Parameter((3,1),name='g')

U = cvx.Parameter((3,1),name='U')
L = cvx.Parameter((3,1),name='L')

constraints = [x >= L, x <= U]

objective = cvx.square(cvx.norm( F*x - g ))
objective = cvx.Minimize(objective)
problem   = cvx.Problem(objective, constraints)

F.value = np.array([[1,2,3],[4,5,6],[7,8,9]])
g.value = np.array([[1],[2],[3]])

U.value = np.array([[42],[42],[42]])
L.value = np.array([[1],[2],[3]])

""" C :
F[0][0] = 1;
F[1][0] = 4;
F[2][0] = 7;

F[0][1] = 2;
F[1][1] = 5;
F[2][1] = 8;

F[0][2] = 3;
F[1][2] = 6;
F[2][2] = 9;

g[0][0] = 1;
g[1][0] = 1;
g[2][0] = 1;

U[0][0] = 42;
U[1][0] = 42;
U[2][0] = 42;

L[0][0] = 1;
L[1][0] = 2;
L[2][0] = 3;
"""

data = problem.get_problem_data('ECOS')[0]

for matrix, value in data.items():
    print(matrix)
    if type(value) == scipy.sparse.csc_matrix:
        print(value.todense())

print('c =',data['c'])
print('dims', 'soc:',data['dims'].soc,'l:',data['dims'].nonpos)

obj = problem.solve(solver='ECOS', verbose=True)
sol = x.value


print('Solution obj :', obj)
print('Solution x   :', sol)
print('Time to solve:', time()-start)
