
import cvxpy as cvx
import scipy

from time import time
start = time()

x = cvx.Variable (name='x')
F = cvx.Parameter(name='F')
g = cvx.Parameter(name='g')

objective = cvx.square(cvx.norm( F*x - g ))
objective = cvx.Minimize(objective)
problem   = cvx.Problem(objective, [])

F.value = 42
g.value = 42

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
