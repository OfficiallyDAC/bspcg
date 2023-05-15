#adapted from pyproximal

import jax
import jax.numpy as jnp

from jaxopt import Bisection
from jax.lax import while_loop

@jax.jit
def BoxProj(lower, upper, x):
    """Box orthogonal projection. 
    (see https://pyproximal.readthedocs.io/en/stable/api/generated/pyproximal.projection.BoxProj.html)
    
    INPUT
    =====
    lower : float or jnp.ndarray, lower bound
    upper : float or jnp.ndarray, upper bound
    x : jnp.ndarray, vector

    OUTPUT
    ======
    Projection.
    """
    x = jnp.minimum(jnp.maximum(x, lower), upper)
    return x

@jax.jit
def fun(mu, x, coeffs, scalar, lower, upper):
    return jnp.dot(coeffs, BoxProj(lower, upper, x - mu * coeffs)) - scalar

@jax.jit
def HyperPlaneBoxProj(x, coeffs, scalar, lower=-jnp.inf, upper=jnp.inf, maxiter=100, xtol=1e-5):
    """
    Orthogonal projection of the intersection between a Hyperplane and a Box.
    (see https://pyproximal.readthedocs.io/en/stable/api/generated/pyproximal.projection.HyperPlaneBoxProj.html)

    INPUT
    =====
    x : jnp.ndarray, vector
    coeffs : jnp.ndarray, vector of coefficients used in the definition of the hyperplane
    scalar : float, scalar used in the definition of the hyperplane
    lower : float or jnp.ndarray, lower bound of Box
    upper : float or jnp.ndarray, upper bound of Box
    maxiter : int, maximum number of iterations used by :func:`jaxopt.Bisection`
    xtol : float, absolute tolerance of :func:`jaxopt.Bisection`
    
    OUTPUT
    ======
    Projection.
    """

    xshape = x.shape
    x = x.ravel()

    # identify brackets for bisect ensuring that the evaluated fun
    # has different sign
    
    f_mu = lambda mu: fun(mu, x, coeffs, scalar, lower, upper)
    f_mu_low = lambda mu: fun(mu, x, coeffs, scalar, lower, upper)<0
    f_mu_up = lambda mu: fun(mu, x, coeffs, scalar, lower, upper)>0
    f_double = lambda mu: mu*2

    bisect_lower=while_loop(f_mu_low, f_double, -1.)
    bisect_upper=while_loop(f_mu_up, f_double, 1.)
    
    # find optimal mu
    bisect = Bisection(f_mu, bisect_lower, bisect_upper, maxiter, xtol, check_bracket=False)
    mu = bisect.run().params

    # compute projection
    y = BoxProj(lower, upper, x - mu * coeffs)
    return y.reshape(xshape)

jax.jit
def SimplexProj(x, coeffs, radius, maxiter=100, xtol=1e-5):
    """
    Simplex projection. 
    (see https://pyproximal.readthedocs.io/en/stable/api/generated/pyproximal.projection.SimplexProj.html)

    INPUTS
    ======
    x : jnp.ndarray, vector
    coeffs : jnp.ndarray, vector of coefficients used in the definition of the hyperplane. It must be set to jnp.ones(x.shape[0])
    radius : float, radius
    maxiter : int, maximum number of iterations used by :func:`jaxopt.Bisection`
    xtol : float, absolute tolerance of :func:`jaxopt.Bisection`
    
    OUTPUT
    ======
    Projection.
    """
    
    return HyperPlaneBoxProj(x, coeffs, radius, lower=0., upper=jnp.inf, maxiter=maxiter, xtol=xtol)

@jax.jit
def L1BallProj(x, coeffs, radius, maxiter=100, xtol=1e-5):
    """
    L1 ball projection.
    (see https://pyproximal.readthedocs.io/en/stable/api/generated/pyproximal.projection.L1BallProj.html)

    INPUTS
    ======
    x : jnp.ndarray, vector
    coeffs : jnp.ndarray, vector of coefficients used in the definition of the hyperplane. It must be set to jnp.ones(x.shape[0])
    radius : float, radius
    maxiter : int, maximum number of iterations used by :func:`jaxopt.Bisection`
    xtol : float, absolute tolerance of :func:`jaxopt.Bisection`
    
    OUTPUT
    ======
    Projection.
    """
    return jnp.exp(1j * jnp.angle(x)) * SimplexProj(jnp.abs(x), coeffs, radius, maxiter, xtol)

@jax.jit
def L1Ballprox(x, n, radius, maxiter=100, xtol=1e-5):
    """
    L1 ball proximal operator.
    (see https://pyproximal.readthedocs.io/en/stable/api/generated/pyproximal.L1Ball.html#pyproximal.L1Ball)

    INPUTS
    ======
    x : jnp.ndarray, vector
    coeffs : jnp.ndarray, vector of coefficients used in the definition of the hyperplane. It must be set to jnp.ones(x.shape[0])
    radius : float, radius
    maxiter : int, maximum number of iterations used by :func:`jaxopt.Bisection`
    xtol : float, absolute tolerance of :func:`jaxopt.Bisection`
    
    OUTPUT
    ======
    Proximal operator.
    """
    return L1BallProj(x, n, radius, maxiter, xtol)

@jax.jit
def L1Ballproxdual(x, tau, coeffs, radius, maxiter=100, xtol=1e-5):
    """
    L1 ball dual proximal operator, computed by Moreau decomposition, which corresponds to the L-inf proximal operator.
    (see https://pyproximal.readthedocs.io/en/stable/api/generated/pyproximal.utils.moreau.moreau.html)

    INPUTS
    ======
    x : jnp.ndarray, vector
    tau : float, positive scalar weight
    coeffs : jnp.ndarray, vector of coefficients used in the definition of the hyperplane. It must be set to jnp.ones(x.shape[0])
    radius : float, radius
    maxiter : int, maximum number of iterations used by :func:`jaxopt.Bisection`
    xtol : float, absolute tolerance of :func:`jaxopt.Bisection`
    
    OUTPUT
    ======
    Proximal operator.
    """
    #this is also valid when x is complex
    pdual = x - tau * L1Ballprox(x / tau, coeffs, radius, maxiter, xtol)
    return pdual
