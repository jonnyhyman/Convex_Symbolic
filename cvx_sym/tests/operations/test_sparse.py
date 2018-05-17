
from cvx_sym.sparse import COO_to_CS

def test_CS_row():

    # with some intentional mis-sorting
    COOL = {'row':[ 0,  0,   1,  1,  2,  2,  3,  2],
            'col':[ 0,  1,   1,  3,  2,  3,  5,  4],
            'val':[ 10, 20, 30, 40, 50, 60,  80, 70 ]}

    CSR  = COO_to_CS(COOL, (4,6), 'row')
    print('CSR')
    print(CSR)

    assert(CSR.A == [10, 20, 30, 40, 50, 60, 70, 80])
    assert(CSR.IA== [0, 2, 4, 7, 8])
    assert(CSR.JA== [0, 1, 1, 3, 2, 3, 4, 5])

def test_CS_col():

    # with some intentional mis-sorting
    COOL = {'row':[ 0,  0,   1,  1,  2,  2,  3,  2],
            'col':[ 0,  1,   1,  3,  2,  3,  5,  4],
            'val':[ 10, 20, 30, 40, 50, 60,  80, 70 ]}

    CSC  = COO_to_CS(COOL, (4,6), 'col')

    print('CSC/CCS')
    print('   A', CSC.A)
    print('   IA', CSC.IA)
    print('   JA', CSC.JA)

    assert(CSC.A  == [10, 20, 30, 50, 40, 60, 70, 80])
    assert(CSC.IA == [0, 1, 3, 4, 6, 7, 8])
    assert(CSC.JA == [0, 0, 1, 2, 1, 2, 2, 3])
