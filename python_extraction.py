

with open("pea_table_formatted.tex",'r') as f:
    lines = f.readlines()

data = []

for l in lines:
    data_point = {}

    l = l.split('&')
    try:
        print(l)
        data_point['R'] = float(l[0])
        data_point['I_coeff'] = float(l[1])
        data_point['Z0_coeff'] = float(l[2])
        data_point['Z1_coeff'] = float(l[3])
        data_point['Z0Z1_coeff'] = float(l[4])
        data_point['X0X1_coeff'] = float(l[5])
        data_point['Y0Y1_coeff'] = float(l[6])
        data_point['t0'] = float(l[6])
   

    except ValueError:
        continue

    data.append(data_point)

print(data)

import json

with open('hamiltonian_data.json','w') as f:
    json.dump(data, f)
