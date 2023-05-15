import jax.numpy as jnp
import numpy as np
import pickle

from jax.numpy import array
from jax.numpy.linalg import pinv


def complex_sign(x):
    return jnp.exp(1j * jnp.angle(x))

def hpinv(A):
    return pinv(A, hermitian=True)

def hmean(a, axis=0, weights=None):
    if not isinstance(a, jnp.ndarray):
        a = array(a)
    if weights is not None and not isinstance(weights, jnp.ndarray):
        weights = array(weights)

    if jnp.all(a >= 0):
        # Harmonic mean only defined if greater than or equal to zero.
        return 1.0 / jnp.average(1.0 / a, axis=axis, weights=weights)
    else:
        raise ValueError("Harmonic mean only defined if all elements greater "
                         "than or equal to zero")
    
def load_obj(name, data_dir):
    with open(data_dir+name+'.pkl', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name, data_dir):
    with open(data_dir+name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def partial_coherence(P):
    R=np.zeros_like(P)
    for k in range(P.shape[-1]):
        g=np.diag(np.diag(P[...,k]))
        try:
            ginv=np.sqrt(np.linalg.pinv(g))
            R[...,k]=-ginv@P[...,k]@ginv
        except Exception as e:
            print("Frequency {}:".format(k))
            print(e)
            
    return R
