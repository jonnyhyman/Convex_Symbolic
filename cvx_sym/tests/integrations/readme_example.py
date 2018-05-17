from cvx_sym import *

# Problem size.
m = 30
n = 20

# Construct the problem.
A = Parameter((m, n), name = 'A')
B = Parameter((m),    name = 'B')
x = Variable((n),     name = 'x')

objective = Minimize(sum_squares(A*x - B))
constraints = [0 <= x, x <= 1]
problem  = Problem(objective, constraints)

gen = Generate(problem,
                            name = 'readme_example',
                            folder = 'examples',
                            verbose = True  # show each stage of the process
)

import numpy
numpy.random.seed(1)

canon = Canonicalize(problem)
canon.assign_values({ # Set values of parameters
                      'A' : numpy.random.randn(m, n),
                      'B' : numpy.random.randn(m)
                    })

solution = solve(canon, verbose = True)  # returns what ecos.solve(...) returns
print(solution['x'])
