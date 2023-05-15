import jax 
import jax.numpy as jnp

from copy import deepcopy
from jax.numpy import absolute, diag, einsum, expand_dims, eye, kron, ones, trace, zeros_like
from jaxopt.prox import prox_group_lasso, prox_lasso
from jax.numpy.linalg import norm, eigh, slogdet
from src.utils import complex_sign

from jax.config import config 
config.update("jax_enable_x64", True)

@jax.jit
def diag_t(d):
    return diag(diag(d))

@jax.jit
def jitted_prox_lasso(v, lmbd, scaling):
    return complex_sign(v)*prox_lasso(absolute(v), lmbd, scaling)    

@jax.jit
def jitted_prox_group_lasso(v, lmbd, scaling):
    return complex_sign(v)*prox_group_lasso(absolute(v), lmbd, scaling)

@jax.jit
def jitted_prox_group_lasso_custom(v, alpha, lmbd, rho):
    st = jitted_prox_lasso(v, alpha*lmbd, 1./rho)
    den_l21 = norm(st, ord=2)
    thresh=(1-alpha)*lmbd/rho
    return jnp.maximum(1-thresh/den_l21,0)

@jax.jit
def fro_norm(m):
    return norm(m, axis=(0,1), ord='fro')

@jax.jit
def jitted_obj(F_hat, P, W, U, rho, alpha, lmbd):
    _,t1=jax.vmap(slogdet, 2, 0)(P)
    _,t2=jax.vmap(slogdet, 2, 0)(P.conj())
    prod1=einsum("ijk,jlk->ilk", F_hat, P)
    prod2=einsum("ijk,jlk->ilk", F_hat.conj(), P.conj())
    t3 = jax.vmap(trace, 2, 0)(prod1+prod2)
    t4 = jax.vmap(fro_norm, 2, 0)(P-W+U)
    r1 = alpha*lmbd*absolute(W).sum()
    r2 = (1-alpha)*lmbd*norm(W, axis=-1, ord=2).sum()
    loss=(.5*(-t1-t2+t3+rho*t4*t4)).sum() + r1 + r2
    return loss


def TSGLASSO(F_hat, lmbd, alpha=0., rho=2., tolp_abs=1.e-3, tolp_rel=1.e-3, told_abs=1.e-3, told_rel=1.e-3, max_iter=1000, step=100, penalize_diag=False, varying_rho=False):
    """
    Ref. 
    [1] Tugnait, Jitendra K. "On sparse high-dimensional graphical model learning for dependent time series." Signal Processing 197 (2022): 108539.  
    [2] Boyd, Stephen, et al. "Distributed optimization and statistical learning via the alternating direction method of multipliers." Foundations and Trends® in Machine learning 3.1 (2011): 1-122.

    INPUT
    =====
    - F_hat: jnp.ndarray, size (N,N,M) estimation of the spectral density F
    - lmbd: float, regularization strength
    - alpha: float in [0.,1.], it balances l1 and l21 norms regularizations, i.e., alpha*lmbd*l1 + (1-alpha)*lmbd*l21 
    - rho: float >0., regularization parameter for Augmented Lagrangian
    - tolp_abs: float, absolute tolerance for primal residual, see sec. 3.1.1 in [2].
    - tolp_rel: float, relative tolerance for primal residual, see sec. 3.1.1 in [2].
    - told_abs: float, absolute tolerance for dual residual, see sec. 3.1.1 in [2].
    - told_rel: float, relative tolerance for dual residual, see sec. 3.1.1 in [2].
    - max_iter: int, maximum number of self.iteration. If primal and dual residual is not achieved in max_iter, the algo will stop.
    - step: int, defines the logging of the algo. It prints opt information each "step" self.iterations.
    - penalize_diag: bool, if True the algo penalizes the diagonal terms as well.
    - varying_rho: bool, if True the algo uses varying stepsize according to the rule in [1], otherwise the stepsize is kept fixed.

    OUTPUT
    ======
    - results: dict, it contains optimization process results
    """
    
    _, N, M = F_hat.shape
    dims = N*jnp.sqrt(M)
    iteration = 0.

    P = kron(ones([1,1,M]),expand_dims(eye(N),2))
    W = zeros_like(F_hat)
    U = zeros_like(F_hat)

    Rp1, Rd1 = jnp.inf, jnp.inf
    Max1p, Max1d = 0.,0.

    check_feasibility_p = lambda x,y: x < dims*tolp_abs +  tolp_rel*y
    check_feasibility_d = lambda x,y: x < dims*told_abs +  told_rel*y

    def all_feasibility(Rp1, Rd1, Max1p, Max1d, iteration):
        all_cond = not (check_feasibility_p(Rp1, Max1p) and check_feasibility_d(Rd1, Max1d)) and iteration<max_iter
        return all_cond

    def fd_tilde(d, rho):
        c=1./(2.*rho)
        q=jnp.sqrt(d*d+4*rho)
        return c*(-d+q)

    def f_P(d_tilde, V):
        return V@diag(d_tilde)@V.conj().T
    
    while all_feasibility(Rp1, Rd1, Max1p, Max1d, iteration):
        #Store previous values
        oldW = deepcopy(W)
            
        iteration+=1

        #P
        A_p= W - U
        S=F_hat-rho*A_p
        d,V = jax.vmap(eigh, 2, (1,2))(S)
        d_tilde = jax.vmap(fd_tilde, (1, None), 1)(d, rho)
        P = jax.vmap(f_P, (1,2), 2)(d_tilde, V)

        #W
        A_w= P+U
        W1 = jitted_prox_lasso(A_w, alpha*lmbd, 1./rho)
        vA_w=A_w.reshape((N*N,1,M), order='F')
        W2 = (jax.vmap(jitted_prox_group_lasso_custom, (0, None, None, None), 0)(vA_w, alpha, lmbd, rho)).reshape((N,N,1), order='F')
        W = W1*W2
        
        if not penalize_diag:
            W = W - jax.vmap(diag_t, 2, 2)(W) + jax.vmap(diag_t, 2, 2)(A_w)
        
        #U
        U+=rho*(P-W)

        #loss
        obj=jitted_obj(F_hat, P, W, U, rho, alpha, lmbd)

        #primal res
        Rp1=norm((P-W).flatten(), ord=2)
        #dual res
        Rd1=norm((rho*(W-oldW)).flatten(), ord=2)

        Max1p=max(norm(P.flatten(), ord=2), norm(W.flatten(), ord=2))
        Max1d=norm(U.flatten(), ord=2)/rho

        if varying_rho:
            if Rp1>10*Rd1:
                rho*=2
            elif Rd1>10*Rp1:
                rho/=2
            else:
                pass

        if iteration%step==0:
                #update here the conditions for primal and dual residuals
                #primal
                cond1p = check_feasibility_p(Rp1,Max1p)
                #dual
                cond1d = check_feasibility_d(Rd1,Max1d)
                
                print("\n\nIteration: {}".format(iteration))
                print("objective: {}".format(obj))
                print("Residual for primal feasibility: {} vs {}".format(jnp.round(Rp1,3), jnp.round(dims*tolp_abs +  tolp_rel*Max1p,3)))
                print("Residual for dual feasibility: {} vs {}".format(jnp.round(Rd1,3), jnp.round(dims*told_abs +  told_rel*Max1d,3)))
                print("Primal feasibility: {}, dual feasibility: {}".format(cond1p, cond1d))
                print("ADMM stepsize value: {}".format(rho))
                print('\n\n')
    
    # Primal and dual conditions
    cond1p = check_feasibility_p(Rp1,Max1p)
    cond1d = check_feasibility_d(Rd1,Max1d)

    return {'P':P, 'objective value': obj, 'iterations':iteration, 'cond1p':cond1p, 'cond1d':cond1d}
