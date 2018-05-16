
import cvx_sym as cvx
import numpy as np

'''
        Least squares optimization problem
'''

x = cvx.Variable (name='x')
F = cvx.Parameter(name='F')
g = cvx.Parameter(name='g')

constraints = []

objective = cvx.square(cvx.norm( F*x - g ))

objective = cvx.Minimize(objective)
problem   = cvx.Problem(objective, constraints)
generator = cvx.Generate(problem,  name = 'least_squares_unconstr',
                                    folder = 'integrated_projects',
                                    verbose = True)
