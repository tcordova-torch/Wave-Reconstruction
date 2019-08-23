import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy.linalg as la

# Functions #
###############################################################################
def zonalReconstruction(Sx, Sy, ds):
    n, n = np.shape(Sx)
    S = np.reshape(Sx, (1, n*n))
    S2 = np.reshape(Sy, (1, n*n))
    S = np.transpose(np.append(S, S2, axis=1))
    E = getE(n)
    U, D, V = la.svd(E, full_matrices=False)
    D = np.diag(D)
    D = la.pinv(D)
    C = getC(n)
    W = np.transpose(V) @ D @ np.transpose(U) @ C @ S
    W = np.reshape(np.transpose(W), (n, n))/ds
    return W

###############################################################################
def getE(n):
    E = np.zeros((2*n*(n - 1), n*n))
    for i in range(0, n):
        for j in range(0, n-1):
            E[i*(n - 1) + j, i*n + j] = -1
            E[i*(n - 1) + j, i*n + j + 1] = 1
            E[(n + i)*(n - 1) + j, i + j*n] = -1
            E[(n + i)*(n - 1) + j, i + (j + 1)*n] = 1
    return E

###############################################################################
def getC(n):
    C = np.zeros((2*n*(n - 1), 2*n*n))
    for i in range(0, n):
        for j in range(0, n-1):
            C[i*(n - 1) + j, i*n + j] = 0.5
            C[i*(n - 1) + j, i*n + j + 1] = 0.5
            C[(n + i)*(n - 1) + j, n*(n + j) + i] = 0.5
            C[(n + i)*(n - 1) + j, n*(n + (j + 1)) + i] = 0.5
    return C
