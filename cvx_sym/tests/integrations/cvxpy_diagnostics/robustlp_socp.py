
import cvxpy as cvx
import numpy as np
import scipy

'''
        Robust Linear Program optimization problem

        minimize c.T * x
        subject: a[i].T * x  +  norm(P[i].T * x) <= b[i]   ; i in [0,m]
'''

from time import time
start = time()

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

A.value = np.array([[1,1],[1,1],[1,1]])
B.value = np.array([[3],[3],[3]])
C.value = np.array([[.1],[.2]])

P.value = np.array([[1,2],[3,4],[5,6]])

"""
A[0][0] = +1.0;
A[0][1] = +1.0;
A[1][0] = +1.0;
A[1][1] = +1.0;
A[2][0] = +1.0;
A[2][1] = +1.0;

B[0][0] = 3;
B[1][0] = 3;
B[2][0] = 3;

C[0][0] = 0.1;
C[1][0] = 0.2;

P[0][0] = +1.0;
P[0][1] = +2.0;
P[1][0] = +3.0;
P[1][1] = +4.0;
P[2][0] = +5.0;
P[2][1] = +6.0;
"""

data = problem.get_problem_data('ECOS')[0]

for matrix, value in data.items():
    print(matrix)
    if type(value) == scipy.sparse.csc_matrix:
        print(value.todense())

print('c =',data['c'])
print('dims', 'soc:',data['dims'].soc,'l:',data['dims'].nonpos)

obj = problem.solve(solver='ECOS', verbose=True)

print('Solution obj :', obj)
print('Solution x  :', x.value)
print('Time to solve:', time()-start)
