#-----------------------------------------------------------------#
#-------------DEFINITIONS FOR THE ISOMETRIES----------------------#
#-----------------------------------------------------------------#

import numpy as np

ns = 5

def basis(s, n):    
    if n >= ns or n < 0:
        b = np.zeros((2 * ns,))  
    else:
        b = np.eye(2 * ns)[2 * n + s]  
    return b

# Define QP
def QP2(alpha, delta):
    result = 0
    for n in range(ns):
        term1 = np.cos(delta / 2) * (np.outer(basis(0,n), basis(0,n)) + np.outer(basis(1,n), basis(1,n)))
        term2 = 1j * np.sin(delta / 2) * (
            np.exp(2j * alpha) * np.outer(basis(1,n), basis(0,n + 1)) +
            np.exp(-2j * alpha) * np.outer(basis(0,n), basis(1,n - 1))
        )
        result += term1 + term2
    return result

# Define HWP
def HWP2(theta):
    return np.kron(np.eye(ns), np.array([[0, np.exp(2j * theta)], [np.exp(-2j * theta), 0]]))

# Define QWP
def QWP2(phi):
    return np.kron(np.eye(ns), np.array([
        [1/2 + 1j/2, (1/2 - 1j/2) * np.exp(2j * phi)],
        [(1/2 - 1j/2) * np.exp(-2j * phi), 1/2 + 1j/2]
    ]))

RLtoHV = np.kron(np.eye(ns), np.array([[1, 1],[-1j, 1j]]) / np.sqrt(2))

def isometry(angles):
    [alpha1, delta1,zeta2,theta2,phi2,alpha2,delta2,theta_p,phi_p] = angles
    U = QWP2(phi_p) @ HWP2(theta_p) @ QP2(alpha2, delta2)@QWP2(phi2) @ HWP2(theta2) @ QWP2(zeta2) @ QP2(alpha1, delta1)
    return (1 / np.sqrt(2)) * (U[0::2][:,ns-1:ns+1]+ U[1::2][:,ns-1:ns+1])