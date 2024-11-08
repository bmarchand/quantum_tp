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
    # TODO implementation here

    return zi_r

def U_IZ(dt):
    
    iz_r = QRoutine()
    # TODO implementation here

    return iz_r

def U_XX(dt):
    
    xx_r = QRoutine()
    # TODO implementation here

    return xx_r

def U_YY(dt):
    
    yy_r = QRoutine()
    # TODO implementation here
    
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

    qroutine = QRoutine()

    # TODO write your code here

    return qroutine

##########################################################
################# Question 4 #############################
##########################################################

def compute_phi_k(bits, nBits,k):
    phi_k = 0
    
    # TODO implementation here
    
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

