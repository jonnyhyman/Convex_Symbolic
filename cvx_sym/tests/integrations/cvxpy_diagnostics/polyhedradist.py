
import cvxpy as cvx
import numpy as np
import scipy

'''
        Polyhedra distance optimization problem

        minimize square(norm(x1 - x2))
        subject: A1*x1 <= b1 (elementwise)
                 A2*x2 <= b2 (elementwise)
'''

from time import time
start = time()

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

A1.value = np.array([[-1,1],[1,1],[0,-1]])
B1.value = np.array([[3],[3],[0]])

A2.value = np.array([[.5,-1],[0,1],[+1,0]])
B2.value = np.array([[-3],[3],[5]])

"""
A1[0][0] = -1.0;
A1[0][1] = +1.0;
A1[1][0] = +1.0;
A1[1][1] = +1.0;
A1[2][0] = +0.0;
A1[2][1] = -1.0;

B1[0][0] = 3;
B1[1][0] = 3;
B1[2][0] = 0;

A2[0][0] = +0.5;
A2[0][1] = -1.0;
A2[1][0] = +0.0;
A2[1][1] = +1.0;
A2[2][0] = +1.0;
A2[2][1] = +0.0;

B2[0][0] = -3;
B2[1][0] = 3;
B2[2][0] = 5;
"""

data = problem.get_problem_data('ECOS')[0]

for matrix, value in data.items():
    print(matrix)
    if type(value) == scipy.sparse.csc_matrix:
        print(value.todense())

print('c =',data['c'])
print('dims', 'soc:',data['dims'].soc,'l:',data['dims'].nonpos)

obj = problem.solve(solver='ECOS', verbose=True)
sol = np.sqrt(obj)

print('Solution obj :', obj)
print('Solution x1  :', x1.value)
print('Solution x2  :', x2.value)
print('Time to solve:', time()-start)
