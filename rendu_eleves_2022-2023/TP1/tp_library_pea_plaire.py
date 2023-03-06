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
    zi_r.apply(CNOT,0,1) # The 2 CNOT are useless but otherwise this gate was considered as a 1 qubit gate and would not pass the tests
    zi_r.apply(CNOT,0,1)
    zi_r.apply(RZ(2*dt),0)

    return zi_r

def U_IZ(dt):
    
    iz_r = QRoutine()
    # TODO implementation here
    iz_r.apply(RZ(2*dt),1)
    return iz_r

def U_XX(dt):
    
    xx_r = QRoutine()
    # TODO implementation here
    xx_r.apply(H,0)
    xx_r.apply(H,1)
    xx_r.apply(CNOT,0,1)
    xx_r.apply(RZ(2*dt),1)
    xx_r.apply(CNOT,0,1)
    xx_r.apply(H,1)
    xx_r.apply(H,0)

    return xx_r

def U_YY(dt):
    
    yy_r = QRoutine()
    # TODO implementation here
    yy_r.apply(RX(np.pi/2),0)
    yy_r.apply(RX(np.pi/2),1)
    yy_r.apply(CNOT,0,1)
    yy_r.apply(RZ(2*dt),1)
    yy_r.apply(CNOT,0,1)
    yy_r.apply(RX(-np.pi/2),0)
    yy_r.apply(RX(-np.pi/2),1)
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
    for i in range(p):
        qroutine.apply(U_II(ham_coeffs["I_coeff"]*dt/p + shift),0,1)
        qroutine.apply(U_ZI(ham_coeffs["Z0_coeff"]*dt/p),0,1)
        qroutine.apply(U_IZ(ham_coeffs["Z1_coeff"]*dt/p),0,1)
        qroutine.apply(U_ZZ(ham_coeffs["Z0Z1_coeff"]*dt/p),0,1)
        qroutine.apply(U_YY(ham_coeffs["Y0Y1_coeff"]*dt/p),0,1)
        qroutine.apply(U_XX(ham_coeffs["X0X1_coeff"]*dt/p),0,1)

    return qroutine

##########################################################
################# Question 4 #############################
##########################################################

def compute_phi_k(bits, nBits,k):
    phi_k = 0
    
    # TODO implementation here
    for l in range(k+1,nBits+1):
        phi_k += 2*np.pi*bits[l]/2**(l-k+1)
    
    return phi_k
