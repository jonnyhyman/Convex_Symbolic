# Form and solve control problem.
# Generate data for control problem.
import cvxpy as cvx
import numpy as np
import scipy

from time import time
start = time()

np.random.seed(1)

n = 8
m = 2
T = 50
alpha = 0.2
beta = 5

x = cvx.Variable((n, T+1), name='x')
u = cvx.Variable((m, T), name='u')

x_0 = cvx.Parameter((n), name='x_0')
A = cvx.Parameter((n,n), name='A')
B = cvx.Parameter((n,m), name='B')

A.value = np.eye(n) + alpha*np.random.randn(n,n)
B.value = np.random.randn(n,m)
x_0.value = beta*np.random.randn(n)
"""
for i in range(A.value.shape[0]):
    for j in range(A.value.shape[1]):
        print('A['+str(i)+']['+str(j)+'] =',A.value[i,j],';')

for i in range(B.value.shape[0]):
    for j in range(B.value.shape[1]):
        print('B['+str(i)+']['+str(j)+'] =',B.value[i,j],';')

for i in range(x_0.value.shape[0]):
    print('x_0['+str(i)+'][0] =',x_0.value[i],';')


A[0][0] = 1.3248690727326484 ;
A[0][1] = -0.12235128273001508 ;
A[0][2] = -0.10563435045269115 ;
A[0][3] = -0.21459372443123412 ;
A[0][4] = 0.17308152586493572 ;
A[0][5] = -0.46030773937605657 ;
A[0][6] = 0.348962352843296 ;
A[0][7] = -0.15224138017902056 ;
A[1][0] = 0.06380781921141972 ;
A[1][1] = 0.950125924904518 ;
A[1][2] = 0.2924215874089948 ;
A[1][3] = -0.41202814189953085 ;
A[1][4] = -0.0644834408027015 ;
A[1][5] = -0.07681087093368313 ;
A[1][6] = 0.2267538884670875 ;
A[1][7] = -0.2199782534628062 ;
A[2][0] = -0.03448564151008715 ;
A[2][1] = -0.17557168358427436 ;
A[2][2] = 1.0084427493431185 ;
A[2][3] = 0.11656304274316445 ;
A[2][4] = -0.22012383544258426 ;
A[2][5] = 0.22894474196792283 ;
A[2][6] = 0.18031814411855912 ;
A[2][7] = 0.10049886778037365 ;
A[3][0] = 0.18017118985288239 ;
A[3][1] = -0.13674557183486663 ;
A[3][2] = -0.024578045103729636 ;
A[3][3] = 0.8128461131481862 ;
A[3][4] = -0.05357761592520319 ;
A[3][5] = 0.1060710933476372 ;
A[3][6] = -0.13833215034506183 ;
A[3][7] = -0.07935070537119548 ;
A[4][0] = -0.1374345400239199 ;
A[4][1] = -0.16904112829974394 ;
A[4][2] = -0.13424922616736382 ;
A[4][3] = -0.0025329197837802723 ;
A[4][4] = 0.7765379302729445 ;
A[4][5] = 0.046883139563418434 ;
A[4][6] = 0.3319604354219741 ;
A[4][7] = 0.14840883211546713 ;
A[5][0] = -0.03836711047232299 ;
A[5][1] = -0.17752579281696726 ;
A[5][2] = -0.14943165875016753 ;
A[5][3] = 0.33849092020554933 ;
A[5][4] = 0.010161550955205795 ;
A[5][5] = 0.8726008706861293 ;
A[5][6] = 0.038183096933493206 ;
A[5][7] = 0.4200510272957685 ;
A[6][0] = 0.024031790496325832 ;
A[6][1] = 0.12344062194148385 ;
A[6][2] = 0.0600340639911655 ;
A[6][3] = -0.07044996929870373 ;
A[6][4] = -0.22850363960442804 ;
A[6][5] = -0.0698685444825755 ;
A[6][6] = 0.9582211533250444 ;
A[6][7] = 0.11732463823643953 ;
A[7][0] = 0.167796682774901 ;
A[7][1] = 0.18622041626071148 ;
A[7][2] = 0.05711746505085176 ;
A[7][3] = 0.17702823285414562 ;
A[7][4] = -0.15087958819933056 ;
A[7][5] = 0.2505736310466576 ;
A[7][6] = 0.10258596408360177 ;
A[7][7] = 0.9403814329794569 ;
B[0][0] = 0.48851814653749703 ;
B[0][1] = -0.07557171302105573 ;
B[1][0] = 1.131629387451427 ;
B[1][1] = 1.5198168164221988 ;
B[2][0] = 2.1855754065331614 ;
B[2][1] = -1.3964963354881377 ;
B[3][0] = -1.4441138054295894 ;
B[3][1] = -0.5044658629464512 ;
B[4][0] = 0.16003706944783047 ;
B[4][1] = 0.8761689211162249 ;
B[5][0] = 0.31563494724160523 ;
B[5][1] = -2.022201215824003 ;
B[6][0] = -0.3062040126283718 ;
B[6][1] = 0.8279746426072462 ;
B[7][0] = 0.2300947353643834 ;
B[7][1] = 0.7620111803120247 ;
x_0[0][0] = -1.1116407130517962 ;
x_0[1][0] = -1.0037903446499872 ;
x_0[2][0] = 0.9328069549414215 ;
x_0[3][0] = 2.0502582360412815 ;
x_0[4][0] = 0.9914986006338488 ;
x_0[5][0] = 0.5950432290372941 ;
x_0[6][0] = -3.353311431445153 ;
x_0[7][0] = 1.887818931604597 ;

printf("Solution was :\n");
printf("   Objective : %f \n", solver_work->info->pcost);
"""

states = []
constraints  = [ x[:,T] == 0 ]
constraints += [ x[:,0] == x_0 ]

for t in range(T):

    cost = cvx.sum_squares(x[:,t+1]) + cvx.sum_squares(u[:,t])

    constraints += [ x[:,t+1] == A*x[:,t] + B*u[:,t] ]
    constraints += [ cvx.norm(u[:,t], 'inf') <= 1 ]

    states.append( cost )

# sums problem objectives and concatenates constraints.
objective = cvx.sum(states)
objective = cvx.Minimize(objective)

problem = cvx.Problem(objective, constraints)

data = problem.get_problem_data('ECOS')[0]

"""
print('c =',data['c'])
print('dims', 'soc:',data['dims'].soc,'l:',data['dims'].nonpos)

for matrix, value in data.items():
    print(matrix)
    if type(value) == scipy.sparse.csc_matrix:
        print(value.todense())
"""

obj = problem.solve(solver='ECOS', verbose=True)

print('Time to solve:', time()-start)
print('Solution obj :', obj)
#print('Solution x  :', x.value)
