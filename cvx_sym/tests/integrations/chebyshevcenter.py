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

generator = cvx.Generate(problem,  name = 'chebyshevcenter',
                                    folder = 'integrated_projects',
                                    verbose = 1)
