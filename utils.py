import numpy as np
import scipy

def cartesian_product_simple_transpose(arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[i, ...] = a
    return arr.reshape(la, -1).T



def pinv_Koopman(x1,u1,x2):
    XU = np.concatenate((x1,u1),-1)
    return scipy.linalg.lstsq(XU,x2)[0].T


def pinv_bilinear_Koopman(x1,u1,x2):
    N = x1.shape[0]
    
    XU = np.einsum('Nx,Nu->Nux',x1,u1)
    XU = XU.reshape((N,-1))
    XU = np.concatenate((x1,XU),axis=-1)

    return scipy.linalg.lstsq(XU,x2)


def linear_step(T,x,u):
    return T@np.concatenate((x,u),axis=-1)


def bilinear_step(T,x,u):
    XU = np.einsum('x,u->xu',x,u).reshape(-1)
    XU = np.concatenate((x,XU),axis=-1)
    return T@XU
