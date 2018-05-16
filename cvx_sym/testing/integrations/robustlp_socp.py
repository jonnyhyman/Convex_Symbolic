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

generator = cvx.Generate(problem,  name = 'robustlp_socp',
                                    folder = 'integrated_projects',
                                    verbose = 1)
