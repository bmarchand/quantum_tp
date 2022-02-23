#!/usr/bin/env python
# coding: utf-8

# ## Question 1

# In[23]:


import time 
import numpy as np
from qat.lang.AQASM import Program, H,  AbstractGate, QRoutine, CNOT
from scipy import linalg # for linalg.expm, the matrix exponential.
from qat.qpus import get_default_qpu # numerical simulator for small quantum circuits.

# PUT YOUR IMPLEMENTATION HERE. Take inspiration from the "minimal notebook" that was sent to you. 


# In[24]:


prog = Program() # The object we use to "accumulate" the gates when building the circuits

q = prog.qalloc(2) # Allocation of a register of 2 qubits called q. It is addressable with [.] like an array.
                   # We will only work with one register in this session, but defining several is possible !
    
prog.apply(H, q[0]) # The first instruction of the program is the application of an Hadamard gate onto q[0]

def matrix(theta):
    X = np.array([[0,1],[1,0]])
    return linalg.expm(-1j * theta * X)

ham_x = AbstractGate("ham_X", [float], arity=1, matrix_generator=matrix) # definition of a custom parametrized gate
prog.apply(ham_x(0.3).ctrl(), [q[0],q[1]]) # The third instuction is the application of our custom gate onto q[0]

circ = prog.to_circ() # The program is exported into a circuit.  
# displaying the circuit:
get_ipython().run_line_magic('qatdisplay', '--svg circ')


# In[25]:


qpu = get_default_qpu()

job = circ.to_job() # options could be fed here. choosing the default, like here, means
# that we want to compute the amplitude/probability of all possible states.

result = qpu.submit(job)

st = -1
for sample in result:
    # print results
    print(sample.amplitude, sample.state, sample.state.int, sample.probability)
    st = sample.state

print(st.int)


# ### Hamiltonian data
# 
# The purpose of the TP is to reproduce, using numerical simulation, Figure 3.(a) of https://arxiv.org/abs/1512.06860.
# 
# On this figure, the ground state energy of a dihydrogen molecule is plotted against the distance $R$ separating the hydrogen atoms. It allows to predict the **equilibrium bond length** of the molecule.
# 
# *Note*: In more complicated settings with larger molecules, energy may be plotted against various distances and angles, forming an *energy landscape* that proves useful in predicting chemical reaction mechanisms, rates, etc.
# 
# The cell below imports the data of Table I of https://arxiv.org/abs/1512.06860.

# In[26]:


#importing Hamiltonian data.
import json 

with open('hamiltonian_data.json','r') as f:
    ham_data = json.load(f)

for coeffs in ham_data:
    print(coeffs)


# ### Question 2:
# Following the QRoutine minimal example below, implement QRoutines for each of the Hamiltonian evolutions we need.
# 

# #### QRoutine: minimal example

# In[28]:


from qat.lang.AQASM import RZ
def f(dt):
    
    routine = QRoutine()
    routine.apply(RZ(dt), 0)            # like a Program, except that gates are applied to "wires" 
    routine.apply(CNOT, 0, 1)      # numbered from 0 to the max number that has been seen.
    routine.apply(H, 0)
    
    return routine

#Pasting it into a circuit

prog = Program()

q = prog.qalloc(4)

a = f(0.1)

prog.apply(a, q[:2])
prog.apply(f(0.2), q[1:3])
prog.apply(f(0.3).ctrl(), q[1:]) #Controlled version

circ = prog.to_circ()

get_ipython().run_line_magic('qatdisplay', '--svg circ')


# #### Hamiltonian evolutions to implement

# In order to evaluate the routines, we need to compute the  

# To find the function U_YY, we evaluate:
# $Y \otimes X = Y \otimes . (S^{\dagger} . Y . S) = CNOT_{0\longrightarrow1} .Y \otimes I.CNOT_{0\longrightarrow1}\\
# \Rightarrow Y \otimes Y =S . CNOT_{0\longrightarrow1} .Y \otimes I.CNOT_{0\longrightarrow1} S^{\dagger}$

# In[30]:


from qat.lang.AQASM import CNOT, RZ, RX, RY, S, I

def u_11_matrix(dt):
    # needed for accurate energy values.
    return np.diag([np.exp(-1j*dt),np.exp(-1j*dt),np.exp(-1j*dt),np.exp(-1j*dt)])

U_II = AbstractGate("II",[float],arity=2,matrix_generator=u_11_matrix)
    
def U_ZZ(dt):
    zz_r = QRoutine()
    zz_r.apply(CNOT, 0, 1)
    zz_r.apply(RZ(2*dt), 1) # difference of convention between implemented version and what we need.
    zz_r.apply(CNOT, 0, 1)

    return zz_r

#Implement, as above, all the other hamiltonian simulations here.
def U_ZI(dt):
    #your code goes here
    zi_r = QRoutine()
    zi_r.apply(RZ(2*dt), 0)
    zi_r.apply(I, 1)
    return zi_r

def U_IZ(dt):
    #and here
    
    iz_r = QRoutine()
    iz_r.apply(I,0)
    iz_r.apply(RZ(2*dt), 1)
    
    return iz_r

def U_XX(dt):
    #...
    xx_r = QRoutine()
    xx_r.apply(CNOT, 0, 1)
    xx_r.apply(RX(2*dt), 0) 
    xx_r.apply(CNOT, 0, 1)
    
    return xx_r

def U_YY(dt):
    
    yy_r= QRoutine()
    yy_r.apply(S.dag(),1)
    yy_r.apply(CNOT, 0, 1)
    yy_r.apply(RY(2*dt),0)
    yy_r.apply(CNOT, 0, 1)
    yy_r.apply(S,1)
    
    return yy_r
    
    
check = True # turn true to plot and see what you do

if check:
    prog = Program()
    q = prog.qalloc(2)
    prog.apply(U_YY(3.), q)
    circ = prog.to_circ()

    #uncomment following line to plot circuit
    get_ipython().run_line_magic('qatdisplay', '--svg circ')


# ### Question 3:
# Implement a function returning a Qroutine implementing a Trotterized evolution generated by our Hamiltonian.

# In[31]:


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
    for k in range(p):
        
        qroutine.apply( U_XX(ham_coeffs['X0X1_coeff']*dt/p), q)
        qroutine.apply( U_YY(ham_coeffs['Y0Y1_coeff'] * dt/p), q)
        qroutine.apply( U_ZZ(ham_coeffs['Z0Z1_coeff']*dt/p), q)
        qroutine.apply( U_IZ(ham_coeffs['Z1_coeff']*dt/p), q)
        qroutine.apply( U_ZI(ham_coeffs['Z0_coeff']*dt/p), q)
        qroutine.apply( U_II((ham_coeffs['I_coeff']+shift)*dt/p), q) 

    # Hint: gates are regular python objects, they can be put in a list.
            
    return qroutine


# In[32]:


# IDEAL HAMILTONIAN SIMULATION: we will use it to compare to the Trotterized version.
def perfect_ham_simulation(ham_coeffs, dt, shift):

    I = np.eye(4)
    Z0 = np.diag([1,1,-1,-1])
    Z1 = np.diag([1,-1,1,-1])
    Z0Z1 = np.diag([1,-1,-1,1])
    X0X1= np.fliplr(np.eye(4))
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


# ### Question 4: Implement iterative phase estimation
# As a function taking as input an Hamiltonian and execution parameters, and returning a phase.

# In[33]:


from qat.lang.AQASM import X

E_max = 3
E_min = -2
    
dt = (2 * np.pi) / float(E_max)

st=-1
#We define a function that will give the most likely state of a simulation
def max_state_calc(result):
    List_proba=[]
    List_state= []
    for sample in result:
        List_proba.append(sample.probability)
        List_state.append(sample.state)
    max_state= List_state[List_proba.index(max(List_proba))]
    
    return max_state


def phase(coeffs, trotterization=False, trotter_number=4, shift=-E_min, nBits = 10):
    """
    Given Hamiltonian coefficients, compute phi, s.t U|\psi\rangle = e^{-2i\pi\phi}|\psi\rangle
    
    Args:
        - coeffs: a dictionary of coefficients as extracted from the list of dictionaries loaded
        from hamiltonian_data.json
        - trotterization: Boolean flag specifying whether to use the Trotterized evolution or the
        ideal "cheat mode" which exponentiates the Hamiltonian.
        - trotter_number: the "p" controlling the degree of approximation of the Trotterization.
        - shift: the energy shift that we use to make sure that the phase we compute is 0 < phi < 1
        - nBits: The number of precision bits we compute.
        
    Returns:
        - phi, a real number that should fall between 0 and 1.
    """
    
    
    bits = {}
    
    for k in range(nBits, 0, -1):
        
        #we initialize an Progam and allocate 3 qubits for control
        prog=Program()
        q = prog.qalloc(3)
        # CIRCUIT CREATION
        # Put your implementation here. Paste here the QRoutines 
        phi_k=0
        if k!=nBits:
            for l in range(k+1, nBits+1):
                 phi_k+= bits[l]/(2**(l-k+1))
            phi_k*=2*np.pi

        prog.apply(X, q[1])
        prog.apply(H, q[0])
          
        #we use the trotterization function or not depending on the flag
        if trotterization:    
            prog.apply(trotter_ham_simulation(coeffs, dt*(2**(k-1)), trotter_number, shift).ctrl(),q)
        else: 
            prog.apply(perfect_ham_simulation(coeffs, dt*(2**(k-1)), shift).ctrl(), q)
            
        prog.apply(RZ(phi_k), q[0])
        prog.apply(H,q[0])
            
        
        # CIRCUIT SIMULATION
        job = prog.to_circ().to_job(qubits=[0])

        result_loc = qpu.submit(job)

        # SELECTION OF MOST LIKELY RESULT 
        # Put your implementation here
        
        max_state=max_state_calc(result_loc)
        bits[k] = max_state.int
    # recompute phi
    phi=0
    for i in range(1, nBits+1):
        phi+= bits[i]/(2**i)
        
    return phi


# ### Question 5: Plot dissociation curves
# Call the function you defined above to compute phases for each values of R. Convert them back to energies, and plot the result for two different Trotter number values: 4 and 10. Both should be wiggly approximations to the ideal curve, but 10 should be closer to it.

# In[34]:


vals_perfect = []
vals_trotter_4 = []
vals_trotter_10 = []
Rs = []

shift = -E_min

start= time.time()

for coeffs in ham_data:
    phi_perfect = phase(coeffs)
    phi_trotter_4 = phase(coeffs, trotterization=True, trotter_number=4)
    phi_trotter_10 = phase(coeffs, trotterization=True, trotter_number=10)

    # CONVERT PHASES BACK TO ENERGY
    E=phi_perfect*2*np.pi/dt
    E_trotter_4=phi_trotter_4*2*np.pi/dt
    E_trotter_10=phi_trotter_10*2*np.pi/dt
    
    print("R", coeffs['R'])
    Rs.append(coeffs['R'])
    
    vals_perfect.append(E)
    vals_trotter_4.append(E_trotter_4)
    vals_trotter_10.append(E_trotter_10)
    
end=time.time()
print(end-start)


# In[35]:


import matplotlib.pylab as plt

plt.plot(Rs, vals_perfect, label="perfect")
plt.plot(Rs, vals_trotter_4, label="p=4")
plt.plot(Rs, vals_trotter_10, label="p=10")
plt.legend();


# We notice that our perfect curve has the predicted tendancy, the curve for p=4 oscillates aroud the perfect one and the curve for p=10 fits better, which is what we expected. We also notice the +2 shift in energy. 

# In[36]:


phi_perf=[i*dt/(2*np.pi) for i in vals_perfect]
phi_4=[i*dt/(2*np.pi) for i in vals_trotter_4]
phi_10=[i*dt/(2*np.pi) for i in vals_trotter_10]

plt.plot(Rs, phi_perf, label="phiperf")
plt.plot(Rs, phi_4, label="phi4")
plt.plot(Rs, phi_10, label="phi10")
plt.legend();


# In[ ]:




