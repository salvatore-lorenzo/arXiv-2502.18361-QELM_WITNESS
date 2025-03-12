import numpy as np

def first(v):
    """
    Returns the first element of the input array.
    
    Parameters:
    v (array-like): Input array.
    
    Returns:
    element: First element of the input array.
    """
    return np.array(v)[0]

def MSE(a, b):
    """
    Computes the Mean Squared Error between two arrays.
    
    Parameters:
    a, b (array-like): Input arrays.
    
    Returns:
    float: Mean Squared Error between the input arrays.
    """
    return np.mean(np.abs(a - b)**2)

def trDistance(rho, sigma):
    """
    Computes the trace distance between two density matrices.
    
    Parameters:
    rho, sigma (ndarray): Input density matrices.
    
    Returns:
    float: Trace distance between the input density matrices.
    """
    return  np.linalg.norm(rho - sigma, ord=1)

def HWP(th):
    """
    Returns the matrix representation of a Half-Wave Plate (HWP) with angle th.
    
    Parameters:
    th (float): Angle in radians.
    
    Returns:
    ndarray: 2x2 matrix representing the HWP.
    """
    return np.array([[0, np.exp(2*1j*th)], [np.exp(-2*1j*th), 0]])

def QWP(th):
    """
    Returns the matrix representation of a Quarter-Wave Plate (QWP) with angle th.
    
    Parameters:
    th (float): Angle in radians.
    
    Returns:
    ndarray: 2x2 matrix representing the QWP.
    """
    return np.array([[(1+1j)/2, (1-1j)/2*np.exp(2*1j*th)], [(1-1j)/2*np.exp(-2*1j*th), (1+1j)/2]])

def state_from_angles(angles, initial_state):
    """
    Computes the state from given angles and initial state.
    
    Parameters:
    angles (list of float): List of angles in radians.
    initial_state (ndarray): Initial state vector.
    
    Returns:
    ndarray: Resulting state vector.
    """
    return np.kron(QWP(angles[3]) @ HWP(angles[2]), QWP(angles[1]) @ HWP(angles[0])) @ initial_state


def to_dm(st):
    return np.outer(st,st.conj())

# Pauli matrices and their Kronecker products
pauli = np.array([[[1,0],[0,1]],[[0,1],[1,0]],[[0,-1j],[1j,0]], [[1,0],[0,-1]]])
pauli2 = np.array([np.kron(op1, op2) for op1 in pauli for op2 in pauli])

# Bell states
Psip = np.array([[0,0,0,0],[0,1/2,1/2,0],[0,1/2,1/2,0],[0,0,0,0]])
Psim = np.array([[0,0,0,0],[0,1/2,-1/2,0],[0,-1/2,1/2,0],[0,0,0,0]])
Phip = np.array([[1/2,0,0,1/2],[0,0,0,0],[0,0,0,0],[1/2,0,0,1/2]])
Phim = np.array([[1/2,0,0,-1/2],[0,0,0,0],[0,0,0,0],[-1/2,0,0,1/2]])

def witness(state):
    """
    Computes the witness operator for a given state.
    
    Parameters:
    state (ndarray): Input state matrix.
    
    Returns:
    ndarray: Witness operator.
    """
    return 1/2 * np.eye(4) - state

# List of witness operators for the Bell states
witnesses = [witness(Psip), witness(Psim), witness(Phip), witness(Phim)]

def my_confusion_matrix(y_true, y_pred):
    """
    Computes a custom confusion matrix for binary classification.
    
    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    
    Returns:
    ndarray: Confusion matrix.
    """
    # Initialize counts
    tp = np.sum((y_true == 1) & (y_pred == 1))   # True Positives
    fn = np.sum((y_true == 1) & (y_pred == -1))  # False Negatives
    fp = np.sum((y_true == -1) & (y_pred == 1))  # False Positives
    tn = np.sum((y_true == -1) & (y_pred == -1)) # True Negatives

    if (tp + fp) == 0:
        cm = np.array([[0, tn/(fn+tn)],
                       [0, fn/(fn+tn)]])
    elif (fn + tn) == 0:
        cm = np.array([[fp/(tp+fp), 0],
                       [tp/(tp+fp), 0]])
    else:
        cm = np.array([[fp/(tp+fp), tn/(fn+tn)],
                       [tp/(tp+fp), fn/(fn+tn)]])
    return cm

def my_confusion_matrix2(y_true, y_pred):
    """
    Computes an alternative custom confusion matrix for binary classification.
    
    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    
    Returns:
    ndarray: Confusion matrix.
    """
    # Initialize counts
    tp = np.sum((y_true == 1) & (y_pred == 1))   # True Positives
    fn = np.sum((y_true == 1) & (y_pred == -1))  # False Negatives
    fp = np.sum((y_true == -1) & (y_pred == 1))  # False Positives
    tn = np.sum((y_true == -1) & (y_pred == -1)) # True Negatives

    if (tn + fn) == 0:
        cm = np.array([[0, tp/(fp+tp)],
                       [0, fp/(fp+tp)]])
    elif (fp + tp) == 0:
        cm = np.array([[fn/(tn+fn), 0],
                       [tn/(tn+fn), 0]])
    else:
        cm = np.array([[fn/(tn+fn), tp/(fp+tp)],
                       [tn/(tn+fn), fp/(fp+tp)]])
    return cm

def dirtprobs(v, stat):
    probs = stat * v
    return np.random.poisson(probs, size=len(probs)) / stat