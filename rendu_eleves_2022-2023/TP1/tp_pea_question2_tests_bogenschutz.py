from Library_Antoine_Bogenschutz import U_ZZ, U_IZ, U_ZI, U_XX, U_YY
import numpy as np
from qat.lang.AQASM import Program, H, CNOT, S
from qat.qpus import get_default_qpu # Here, the default qpu is a numerical simulator

qpu = get_default_qpu()

def uniform_state_prep():
    print("ok1")
    
    prog = Program()
    q = prog.qalloc(2)

    prog.apply(H, q[0])
    prog.apply(H, q[1])

    return prog, q

def test_uzz():
    print("ok2")
    prog, q = uniform_state_prep()

    test_uzz = U_ZZ(0.5*np.pi) 
    prog.apply(test_uzz, q)

    circ = prog.to_circ()
    job = circ.to_job()
    result = qpu.submit(job)
    statevector = np.array([s.amplitude for s in result])
    
    assert(np.allclose(statevector,0.5*np.array([-1,1,-1j,1j])))

def test_uzi():
    print("ok3")
    prog, q = uniform_state_prep()

    test_uzi = U_ZI(0.5*np.pi)
    prog.apply(test_uzi, q)

    circ = prog.to_circ()
    job = circ.to_job()
    result = qpu.submit(job)
    statevector = np.array([s.amplitude for s in result])

    assert(np.allclose(statevector,0.5*np.array([-1j,-1j,1j,1j])))

def test_uiz():
    print("ok4")
    prog, q = uniform_state_prep()

    test_uiz = U_IZ(0.5*np.pi)
    prog.apply(test_uiz, q)

    circ = prog.to_circ()
    job = circ.to_job()
    result = qpu.submit(job)
    statevector = np.array([s.amplitude for s in result])

    assert(np.allclose(statevector,0.5*np.array([-1j,1j,-1j,1j])))

def test_uxx():
    print("ok5")
    prog, q = uniform_state_prep()

    prog.apply(S,q[0])
    test_uxx = U_XX(0.5*np.pi)
    prog.apply(test_uxx, q)

    circ = prog.to_circ()
    job = circ.to_job()
    result = qpu.submit(job)
    statevector = np.array([s.amplitude for s in result])

    print(statevector)
    assert(np.allclose(statevector,0.5*np.array([1,1,-1j,-1j])))

def test_uyy():
    print("ok6")
    prog, q = uniform_state_prep()

    prog.apply(S,q[0])
    test_uyy = U_YY(0.5*np.pi)
    prog.apply(test_uyy, q)

    circ = prog.to_circ()
    job = circ.to_job()
    result = qpu.submit(job)
    statevector = np.array([s.amplitude for s in result])

    print(statevector)
    assert(np.allclose(statevector,0.5*np.array([-1,1,-1j,1j])))

