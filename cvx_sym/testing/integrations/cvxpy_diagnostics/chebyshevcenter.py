
import cvxpy as cvx
import numpy as np
import scipy

'''
        Chebyshev centering optimization problem
        maximize r
        subject: (a[i].T * x) + (r * norm(a[i])) <= b[i]   : for i in [0,m]
        vars: x, r
'''

from time import time
start = time()

n = 2
m = 3

r = cvx.Variable (name='r')
x = cvx.Variable ((n,1),name='x')

A = cvx.Parameter((m,n),name='A')
B = cvx.Parameter((m,1),name='B')

objective = -r

constraints = [ A[i].T * x + r * cvx.norm(A[i]) <= B[i] for i in range(m) ]
constraints += [r >= 0]

objective = cvx.Minimize(objective)
problem   = cvx.Problem(objective, constraints)

A.value = np.array([[-1,1],[1,1],[0,-1]])
B.value = np.array([[3],[3],[0]])

"""
A[0][0] = -1.0;
A[0][1] = +1.0;
A[1][0] = +1.0;
A[1][1] = +1.0;
A[2][0] = 0.0;
A[2][1] = -1.0;

B[0][0] = 3;
B[1][0] = 3;
B[2][0] = 0;
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
