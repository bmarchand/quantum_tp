import numpy as np
from qat.lang.AQASM import AbstractGate, QRoutine
from qat.lang.AQASM import H, X
from qat.lang.AQASM.qftarith import QFT

def u_zz_matrix(dt):
    return np.diag([np.exp(-1j*dt),np.exp(1j*dt),np.exp(1j*dt),np.exp(-1j*dt)])

U_ZZ = AbstractGate("ZZ",[float],arity=2,matrix_generator=u_zz_matrix)

def u_xx_matrix(dt):
    M = np.diag([np.cos(dt),np.cos(dt),np.cos(dt),np.cos(dt)])

    N = np.diag([-1j*np.sin(dt),-1j*np.sin(dt),-1j*np.sin(dt),-1j*np.sin(dt)])

    return M+np.fliplr(N)

U_XX = AbstractGate("XX",[float],arity=2,matrix_generator=u_xx_matrix)

def u_yy_matrix(dt):
    M = np.diag([np.cos(dt),np.cos(dt),np.cos(dt),np.cos(dt)])

    N = np.diag([-1j*np.sin(dt),-1j*np.sin(dt),-1j*np.sin(dt),-1j*np.sin(dt)])

    return M+np.fliplr(N)

U_YY = AbstractGate("YY",[float],arity=2,matrix_generator=u_yy_matrix)

def u_xy_matrix(dt):
    M = np.diag([np.cos(dt),np.cos(dt),np.cos(dt),np.cos(dt)])

    N = np.diag([np.sin(dt),-np.sin(dt),np.sin(dt),-np.sin(dt)])

    return M+np.fliplr(N)

U_XY = AbstractGate("YY",[float],arity=2,matrix_generator=u_xy_matrix)

def u_z1_matrix(dt):
    return np.diag([np.exp(-1j*dt),np.exp(-1j*dt),np.exp(1j*dt),np.exp(1j*dt)])

U_ZI = AbstractGate("ZI",[float],arity=2,matrix_generator=u_z1_matrix)

def u_1z_matrix(dt):
    return np.diag([np.exp(1j*dt),np.exp(-1j*dt),np.exp(1j*dt),np.exp(-1j*dt)])

U_IZ = AbstractGate("IZ",[float],arity=2,matrix_generator=u_1z_matrix)

def u_11_matrix(dt):
    return np.diag([np.exp(-1j*dt),np.exp(-1j*dt),np.exp(-1j*dt),np.exp(-1j*dt)])

U_II = AbstractGate("II",[float],arity=2,matrix_generator=u_11_matrix)


####################
####################

def ham_simulation(ham_coeffs, dt, p):

    gates = []

    gates.append(U_II(ham_coeffs["I_coeff"]*dt / float(p)))
    gates.append(U_ZI(ham_coeffs["Z0_coeff"]*dt / float(p)))
    gates.append(U_IZ(ham_coeffs["Z1_coeff"]*dt / float(p)))
    gates.append(U_ZZ(ham_coeffs["Z0Z1_coeff"]*dt / float(p)))
    gates.append(U_XX(ham_coeffs["X0X1_coeff"]*dt / float(p)))
    gates.append(U_YY(ham_coeffs["Y0Y1_coeff"]*dt / float(p)))

    qroutine = QRoutine()

    for _ in range(p):
        for g in gates:
            qroutine.apply(g, 0, 1)

    return qroutine

def ansatz(theta):

    qroutine = QRoutine()
    
    qroutine.apply(X, 1)

    qroutine.apply(U_XY(theta), 0, 1)

    return qroutine

def pea(ham_coeffs, p, t, theta):

    qroutine = QRoutine()

    for k in range(t):

        qroutine.apply(H, k)

    qroutine.apply(ansatz(theta), t, t+1)

    for k in range(t):

        dt = 2**k * ham_coeffs["t0"]

        qroutine.apply(ham_simulation(ham_coeffs, dt, p).ctrl(), k, t, t+1)

    qroutine.apply(QFT(t).dag(), range(t))

    return qroutine

