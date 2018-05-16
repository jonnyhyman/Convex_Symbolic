import cvx_sym as cvx

x = cvx.Variable ((3,1),name='x')
F = cvx.Parameter((3,3),name='F')
g = cvx.Parameter((3,1),name='g')

U = cvx.Parameter((3,1),name='U')
L = cvx.Parameter((3,1),name='L')

constraints  = []

constraints += [ x <= U ]
constraints += [ L <= x ]

objective = cvx.square(cvx.norm( F*x - g ))
problem = cvx.Problem( cvx.Minimize(objective), constraints )

generator = cvx.Generate(problem,  name = 'least_squares_constr',
                                    folder = 'integrated_projects',
                                    verbose = True)
