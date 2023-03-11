import numpy as np
from scipy import linalg
from qat.lang.AQASM import Program, H,  AbstractGate, QRoutine
from qat.lang.AQASM import CNOT, RZ, RX, RY, S, I

##########################################################
################# Question 2 #############################
##########################################################

def u_11_matrix(dt):
    # needed for accurate energy values
    return np.diag([np.exp(-1j*dt),np.exp(-1j*dt),np.exp(-1j*dt),np.exp(-1j*dt)])

# being simpler, this one is implemented with an AbstractGate
U_II = AbstractGate("II",[float],arity=2,matrix_generator=u_11_matrix)

def U_ZZ(dt):
    
    zz_r = QRoutine()
    zz_r.apply(CNOT, 0, 1)
    zz_r.apply(RZ(2*dt), 1) # difference of convention between implemented version and what we need.
    zz_r.apply(CNOT, 0, 1)

    return zz_r

#Implement, as above, all the other hamiltonian simulations here.
def U_ZI(dt):

    zi_r = QRoutine()
    zi_r.apply(RZ(2*dt), 0) # difference of convention between implemented version and what we need.
    zi_r.apply(I, 1)

    return zi_r



def U_IZ(dt):
    
    iz_r = QRoutine()
    iz_r.apply(I, 0)
    iz_r.apply(RZ(2*dt), 1) 

    return iz_r


def U_XX(dt):
    
    xx_r = QRoutine()
    xx_r.apply(CNOT, 0, 1)
    xx_r.apply(RX(2*dt), 0) 
    xx_r.apply(CNOT, 0, 1)

    return xx_r


def U_YY(dt):
    
    yy_r = QRoutine()
    yy_r.apply(S.dag(), 1)
    yy_r.apply(CNOT, 0, 1)
    yy_r.apply(RY(2*dt), 0) 
    yy_r.apply(CNOT, 0, 1)
    yy_r.apply(S,1)

    return yy_r


##########################################################
################# Question 3 #############################
##########################################################

def trotter_ham_simulation(ham_coeffs, dt, p, shift):
    """
    Args:
        - ham_coeffs: a dictionary from the list of dictionaries loaded from hamiltonian_data.json.
        Therefore its keys are "I_coeff", "Z0_coeff", etc.
        - dt: a float, corresponding to the interval of time whose value we will define later.
        - p: the "Trotter number": the integer controlling the degree of approximation
        - shift: an energy shift to the Hamiltonian to make sure that the value of the ground state energy
        is positive. It consists in adding +shift*I to the Hamiltonian.
    """
    # Initialize QRoutine object
    qr = QRoutine()
    g0, g1, g2, g3, g4, g5 = ham_coeffs["I_coeff"], ham_coeffs['Z0_coeff'], ham_coeffs['Z1_coeff'], ham_coeffs['Z0Z1_coeff'], ham_coeffs['Y0Y1_coeff'], ham_coeffs['X0X1_coeff']

    # Append the appropriate gates to the QRoutine object for each term in the Hamiltonian
    for i in range(p):
        qr.apply(U_II(dt*ham_coeffs['I_coeff']), 0, 1)
        qr.apply(U_ZZ(dt*ham_coeffs['Z0Z1_coeff']), 0, 1)
        qr.apply(U_ZI(dt*ham_coeffs['Z0_coeff']), 0, 1)
        qr.apply(U_IZ(dt*ham_coeffs['Z1_coeff']), 0, 1)
        qr.apply(U_XX(dt*ham_coeffs['X0X1_coeff']), 0, 1)
        qr.apply(U_YY(dt*ham_coeffs['Y0Y1_coeff']), 0, 1)

    return qr


##########################################################
################# Question 4 #############################
##########################################################

# Test : 
# def test_compute_phi_k():
    # assert(np.allclose(compute_phi_k({1:0,2:1,3:1},3,1),2*np.pi*3/8.0))

def compute_phi_k(bits, nBits,k):
    # Initialize the phase estimate to 0
    phi_k = 0
    
    for l in range(k+1, nBits+1):
        phi_k = phi_k + bits[l]/2**(l-k+1)
    phi_k *= 2*np.pi
    
    # Tests :
    # Initialize the qubits to the eigenstate |ψ>
    # for qubit in range(n):
        # phi_k += eigenstate[qubit + 1] * (2 ** qubit)
    
    # Apply controlled rotations on each qubit, using the eigenvalue of the eigenstate as the rotation angle
    # phi_k *= 2 * np.pi * k / (2 ** n)


    return phi_k



# DO NOT EDIT CODE BELOW 
def perfect_ham_simulation(ham_coeffs, dt, shift):

    I = np.eye(4)
    Z0 = np.diag([1,1,-1,-1])
    Z1 = np.diag([1,-1,1,-1])
    Z0Z1 = np.diag([1,-1,-1,1])
    X0X1 = np.fliplr(np.eye(4))
    Y0Y1 = np.fliplr(np.diag([-1, 1, 1, -1]))

    H = (ham_coeffs['I_coeff']+shift) * I
    H += ham_coeffs['Z0_coeff'] * Z0
    H += ham_coeffs['Z1_coeff'] * Z1
    H += ham_coeffs['Z0Z1_coeff'] * Z0Z1
    H += ham_coeffs['X0X1_coeff'] * X0X1
    H += ham_coeffs['Y0Y1_coeff'] * Y0Y1

    U = linalg.expm(-1j * dt * H)

    def matrix():
        return U

    U_gate = AbstractGate("U", [], arity=2,
                     matrix_generator=matrix)

    qroutine = QRoutine()

    qroutine.apply(U_gate(), 0, 1)

    return qroutine

