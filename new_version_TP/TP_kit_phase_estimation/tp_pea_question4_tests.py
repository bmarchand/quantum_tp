from tp_library_pea import compute_phi_k
import numpy as np

def test_compute_phi_k():
    assert(np.allclose(compute_phi_k({1:0,2:1,3:1},3,1),2*np.pi*3/8.0))
