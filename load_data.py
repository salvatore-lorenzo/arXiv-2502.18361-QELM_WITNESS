import json
import numpy as np
import pandas as pd

import definitions as defs
import isometry as isometry


folders = ['01_21','11_21','12_12','01_09','09_02','11_29']
experiments = ['E1','E2','E3','E4','E5','E6']
Nruns = [ 4, 6, 6, 5, 3, 5]
Nstates =[300, 166, 300, 300, 300, 236]

def select_experiment(num):
    # Select the index for the folder, number of runs, and number of states
    select = 0

    # Get the folder name, number of repetitions, and number of states for the selected index
    fold = folders[select]
    repetitions = Nruns[select]
    num_states = Nstates[select]

    # Initialize lists to store data for separable and entangled states
    data_sep = [None] * repetitions
    dataRAW_sep = [None] * repetitions
    data_ent = [None] * repetitions
    dataRAW_ent = [None] * repetitions

    # Load data for separable states from JSON files
    for rep in range(repetitions):
        with open(f'./data/Separabili_{fold}_rep_'+str(rep)+'.json') as f:
            data_sep[rep] = pd.DataFrame(json.load(f))

    # Load data for entangled states from JSON files
    for rep in range(repetitions):
        with open(f'./data/Entangled_{fold}_rep_'+str(rep)+'.json') as f:
            data_ent[rep] = pd.DataFrame(json.load(f))

    # Load raw data for separable states from JSON files
    for rep in range(repetitions):
        with open(f'./data/SeparabiliRAW_{fold}_rep_'+str(rep)+'.json') as f:
            dataRAW_sep[rep] = pd.DataFrame(json.load(f))

    # Load raw data for entangled states from JSON files
    for rep in range(repetitions):
        with open(f'./data/EntangledRAW_{fold}_rep_'+str(rep)+'.json') as f:
            dataRAW_ent[rep] = pd.DataFrame(json.load(f))

    # Initialize lists to store counts for separable and entangled states for each repetition
    counts_rep_sep = [None] * repetitions
    counts_rep_ent = [None] * repetitions
    counts_rep = [None] * repetitions

    # Loop over each repetition to process the data
    for rep in range(repetitions):
        counts_rep_sep[rep] = np.stack(dataRAW_sep[rep].groupby('State')['doubles'].apply(np.array).values)
        counts_rep_ent[rep] = np.stack(dataRAW_ent[rep].groupby('State')['doubles'].apply(np.array).values)
        counts_rep[rep] = np.vstack((counts_rep_sep[rep], counts_rep_ent[rep]))
    # Sum the counts for the first two repetitions to get the total counts
    counts_tot = np.sum([counts_rep[rep] for rep in range(2)], axis=0)

    # Initialize dictionaries to store angles and state references for different folders
    anglesQW1 = {}
    anglesQW2 = {}
    sep_state_ref = {}
    ent_state_ref = {}

    # Define angles and state references for folder '01_09'
    anglesQW1['01_09'] = [0.3316125578789226, 1.5707963267948966, 1.0070977275570985, 1.5570696348907915, 2.908160255954825, 1.3439035240356338, 3.141592653589793, 4.539779498163959, 1.5707963284790725]
    anglesQW2['01_09'] = [5.8643062867009474, 1.5707963267948966, 1.095755783171672, 1.5937676381596888, 2.89289820797498, 2.844886680750757, 3.141592653589793, 4.54821017223903, 1.5707963277463037]
    sep_state_ref['01_09'] = [1/2, 1/2, -1/2, -1/2]
    ent_state_ref['01_09'] = [1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]

    # Define angles and state references for folder '01_21'
    anglesQW1['01_21'] = [0.3316125578789226, 1.5707963267948966, 1.0070977275570985, 1.5570696348907915, 2.908160255954825, 1.3439035240356338, 3.141592653589793, 4.539779498163959, 1.5707963284790725]
    anglesQW2['01_21'] = [5.8643062867009474, 1.5707963267948966, 1.095755783171672, 1.5937676381596888, 2.89289820797498, 2.844886680750757, 3.141592653589793, 4.54821017223903, 1.5707963277463037]
    sep_state_ref['01_21'] = [1/2, -1/2, -1/2, 1/2]
    ent_state_ref['01_21'] = [1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]

    # Define angles and state references for folder '11_21'
    anglesQW1['11_21'] = [0.3316125578789226, 1.5707963267948966, 2.908160255954825, 1.5570696348907915, 1.0070977275570985, 1.3439035240356338, 3.141592653589793, 4.539779498163959, 1.5707963284790725]
    anglesQW2['11_21'] = [5.8643062867009474, 1.5707963267948966, 2.844886680750757, 1.5937676381596888, 2.89289820797498, 1.095755783171672, 3.141592653589793, 4.54821017223903, 1.5707963277463037]
    sep_state_ref['11_21'] = [1/2, 1/2, -1/2, -1/2]
    ent_state_ref['11_21'] = [1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]

    # Define angles and state references for folder '12_12'
    anglesQW1['12_12'] = [0.3316125578789226, 1.5707963267948966, 3.03163293212240909, 0.26739621035255735, 1.201407919651270299, 1.3439035240356338, 3.141592653589793, 1.988305525776593563, 0.292390408389419524]
    anglesQW2['12_12'] = [5.8643062867009474, 1.5707963267948966, 1.32290702615153614, 1.80951555206664139, 0.65633642655426307, 2.844886680750757, 3.141592653589793, 0.685928880617840962, 2.303692086342170878]
    sep_state_ref['12_12'] = [1/2, -1/2, -1/2, 1/2]
    ent_state_ref['12_12'] = [1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]

    # Define angles and state references for folder '09_02'
    anglesQW1['09_02'] = [0.3316125578789226, 1.5707963267948966, 2.908160255954825, 1.5570696348907915, 1.0070977275570985, 1.3439035240356338, 3.141592653589793, 4.539779498163959, 1.5707963284790725]
    anglesQW2['09_02'] = [5.8643062867009474, 1.5707963267948966, 2.844886680750757, 1.5937676381596888, 2.89289820797498, 1.095755783171672, 3.141592653589793, 4.54821017223903, 1.5707963277463037]
    sep_state_ref['09_02'] = [1/2, 1/2, -1/2, -1/2]
    ent_state_ref['09_02'] = [1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]

    # Define angles and state references for folder '11_29'
    anglesQW1['11_29'] = [0.3316125578789226, 1.5707963267948966, 2.908160255954825, 1.5570696348907915, 1.0070977275570985, 1.3439035240356338, 3.141592653589793, 4.539779498163959, 1.5707963284790725]
    anglesQW2['11_29'] = [5.8643062867009474, 1.5707963267948966, 2.844886680750757, 1.5937676381596888, 2.89289820797498, 1.095755783171672, 3.141592653589793, 4.54821017223903, 1.5707963277463037]
    sep_state_ref['11_29'] = [1/2, 1/2, -1/2, -1/2]
    ent_state_ref['11_29'] = [1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]

    if fold == '09_02':
        HWP1_sep = data_sep[0].groupby('State')['Angle_HWP'].apply(defs.first).values
        HWP2_sep = data_sep[0].groupby('State')['Angle_HWP'].apply(defs.first).values
        HWP1_ent = data_ent[0].groupby('State')['Angle_HWP'].apply(defs.first).values
        HWP2_ent = data_ent[0].groupby('State')['Angle_HWP'].apply(defs.first).values

        QWP1_sep = data_sep[0].groupby('State')['Angle_QWP'].apply(defs.first).values
        QWP2_sep = data_sep[0].groupby('State')['Angle_QWP'].apply(defs.first).values
        QWP1_ent = data_ent[0].groupby('State')['Angle_QWP'].apply(defs.first).values
        QWP2_ent = data_ent[0].groupby('State')['Angle_QWP'].apply(defs.first).values
    else:
        HWP1_sep = data_sep[0].groupby('State')['Angle_HWP1'].apply(defs.first).values
        HWP2_sep = data_sep[0].groupby('State')['Angle_HWP2'].apply(defs.first).values
        HWP1_ent = data_ent[0].groupby('State')['Angle_HWP1'].apply(defs.first).values
        HWP2_ent = data_ent[0].groupby('State')['Angle_HWP2'].apply(defs.first).values

        QWP1_sep = data_sep[0].groupby('State')['Angle_QWP1'].apply(defs.first).values
        QWP2_sep = data_sep[0].groupby('State')['Angle_QWP2'].apply(defs.first).values
        QWP1_ent = data_ent[0].groupby('State')['Angle_QWP1'].apply(defs.first).values
        QWP2_ent = data_ent[0].groupby('State')['Angle_QWP2'].apply(defs.first).values

    # Convert angles from degrees to radians and stack them for separable and entangled states
    angles_sep = np.pi/180 * np.vstack((HWP2_sep, QWP2_sep, HWP1_sep, QWP1_sep)).T
    angles_ent = np.pi/180 * np.vstack((HWP2_ent, QWP2_ent, HWP1_ent, QWP1_ent)).T

    # Generate states from angles for separable and entangled states
    sep_states = [defs.state_from_angles(angles, sep_state_ref[fold]) for angles in angles_sep]
    ent_states = [defs.state_from_angles(angles, ent_state_ref[fold]) for angles in angles_ent]

    # Stack the states for separable and entangled states
    states = np.vstack((sep_states, ent_states))

    # Define Pauli matrices
    pauli2 = defs.pauli2

    # Calculate expectation values for Pauli matrices
    expval_pauli = []
    for op in pauli2:
        expval_pauli += [[st.conj() @ op @ st for st in states]]
    expval_pauli = np.array(expval_pauli).T.real

    # Extract specific expectation values for Pauli matrices
    expval_pauli1 = np.array([expval_pauli[:, 0], expval_pauli[:, 1], expval_pauli[:, 2], expval_pauli[:, 3]]).T
    expval_pauli2 = np.array([expval_pauli[:, 0], expval_pauli[:, 4], expval_pauli[:, 8], expval_pauli[:, 12]]).T

    # Define witness operators
    witnesses = defs.witnesses

    # Calculate expectation values for witness operators
    expval_witnesses = []
    for wit in witnesses:
        expval_witnesses += [[(st).conj() @ wit @ (st) for st in states]]
    expval_witnesses = np.array(expval_witnesses).T.real

    # Calculate probabilities using isometry
    V1 = isometry.isometry(anglesQW1[fold])
    V2 = isometry.isometry(anglesQW2[fold])
    VV = np.kron(V1,V2)
    probs = np.array([(np.abs(VV @ st)**2) for st in states])
    print('OK')
    return [expval_pauli, expval_witnesses, probs, counts_tot, VV, num_states]
