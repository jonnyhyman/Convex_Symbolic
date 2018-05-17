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

constraints  = [ A1[i].T * x1 <= B1[i] for i in range(m) ]
constraints += [ A2[i].T * x2 <= B2[i] for i in range(p) ]

objective = cvx.Minimize(objective)
problem   = cvx.Problem(objective, constraints)

generator = cvx.Generate(problem,  name = 'polyhedradist',
                                    folder = 'integrated_projects',
                                    verbose = 1)
