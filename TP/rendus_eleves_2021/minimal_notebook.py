#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports you will need: 
from qat.lang.AQASM import Program, H, CNOT, RZ,  AbstractGate, S
import numpy as np
from scipy import linalg
import matplotlib.pylab as plt


# ## Programming a simple circuit:

# In[2]:


prog = Program() # The object we use to "accumulate" the gates when building the circuits

q = prog.qalloc(2) # Allocation of a register of 2 qubits called q. It is addressable with [.] like an array.
                   # We will only work with one register in this session, but defining several is possible !

prog.apply(H, q[0]) # The first instruction of the program is the application of an Hadamard gate onto q[0]
prog.apply(S.dag(), q[1])
prog.apply(CNOT, [q[0],q[1]]) # The second is a CNOT

def matrix(theta):
    X = np.array([[0,1],[1,0]])
    return linalg.expm(-1j * theta * X)

ham_x = AbstractGate("ham_X", [float], arity=1, matrix_generator=matrix) # definition of a custom parametrized gate

prog.apply(ham_x(0.45), q[0]) # The third instuction is the application of our custom gate onto q[0]

prog.apply(H.ctrl(), q) # .ctrl() can be used on any 1-qubit gate to get a 2-qubit controlled version

circ = prog.to_circ() # The program is exported into a circuit.  

prog.apply(CNOT, q)
circ2 = prog.to_circ()

# displaying the circuit:
get_ipython().run_line_magic('qatdisplay', '--svg circ')
get_ipython().run_line_magic('qatdisplay', '--svg circ2')


# ## Simulating it:

# In[24]:


from qat.qpus import get_default_qpu # Here, the default qpu is a numerical simulator
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


# ### Feel free to consult https://myqlm.github.io/ and https://myqlm.github.io/myqlm_specific/notebooks.html for more details.

# In[ ]:




