from time import time
start = time()

import cvx_sym as cvx

n = 8
m = 2
T = 50

x = cvx.Variable((n, T+1), name='x')
u = cvx.Variable((m, T), name='u')

x_0 = cvx.Parameter((n), name='x_0')
A = cvx.Parameter((n,n), name='A')
B = cvx.Parameter((n,m), name='B')

states = []
constraints  = [ x[:,T] == 0 ]
constraints += [ x[:,0] == x_0[:,0] ]

for t in range(T):


    constraints += [ x[:,t+1] == A*x[:,t] + B*u[:,t] ]
    constraints += [ cvx.norm(u[:,t], kind = 'inf') <= 1 ]

    cost = cvx.sum_squares(x[:,t+1]) + cvx.sum_squares(u[:,t])
    states.append( cost )

# sums problem objectives and concatenates constraints.
objective = cvx.sum(states)
objective = cvx.Minimize(objective)
problem   = cvx.Problem(objective, constraints)

problem = cvx.Problem(objective, constraints)
generator = cvx.Generate(problem,  name = 'control',
                                    folder = 'integrated_projects',
                                    verbose = 1)
