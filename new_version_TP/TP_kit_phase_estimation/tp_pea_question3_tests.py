from tp_library_pea import trotter_ham_simulation

def test_num_gates():
    
    keys = ["I_coeff","Z0_coeff","Z1_coeff","Z0Z1_coeff","X0X1_coeff","Y0Y1_coeff"]
    ham_coeffs = {k:1.0 for k in keys}

    qr = trotter_ham_simulation(ham_coeffs, 0.1, 10, 1)

    assert(len(qr.op_list)==10*6)
