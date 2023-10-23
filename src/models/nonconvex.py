import jax 
import jax.numpy as jnp

from copy import deepcopy
from functools import partial

from jaxopt import Bisection
from jaxopt.prox import prox_group_lasso
from jax.numpy import absolute, argsort, array, append, diag, diff, einsum, expand_dims, eye, insert, kron, ones, tril, unique, zeros, zeros_like
from jax.numpy.fft import rfft
from jax.numpy.linalg import pinv, norm, eigh
from jax.random import PRNGKey, normal
from jax.scipy.linalg import block_diag
from jax.lax import cond, while_loop
from src.models.Linf_prox import L1Ballproxdual
from src.utils import complex_sign, hpinv

from jax.config import config 
config.update("jax_enable_x64", True)

def top_k_mask(m,k,N):
    
    m=m.flatten(order='F')
    #initialize flattend mask
    mask=zeros(N*N)
    
    #1) unique values and their counts
    uv, cuv= unique(m, return_counts=True)
    #2) index of sorted uv
    iuv = argsort(uv)
    
    if k==0:
        mask=eye(N)
    elif k<uv.shape[0]:
        k1 = cuv[iuv[-k:]].sum()
        idx=argsort(m)
        top_k = idx[-k1:]
        mask=mask.at[top_k].set(1.)
        mask=mask.reshape((N,N), order='F')
        mask+=(mask.T+eye(N))
    else:
        #take all elements
        mask=ones((N,N))
   
    return mask

def CF_method(F_hat, k, K=None):
    """
    Closed-form (CF) method for learning K-sparse inverse spectral density when prior domain knowledge about the sparsity measure is available. 

    INPUTS
    ======
    - F: jnp.ndarray, estimation of the periodogram, hermitian matrix of size (N,N,M)
    - k: jnp.ndarray, number of unique off-diagonal values of the l2-norm of P. 
        Hence, Psk is made of the fibers of P corresponding to those k unique values (plus the diagonal of P) 
    - K: jnp.ndarray of dtype jnp.int, splitting points for frequencies to define the blocks.
        Example: M=50, K=[M//10, M//4] --> The groups will be {[0, M//10); [M//10, M//4); [M//4, M]} 

    OUTPUT
    ======
    Psk: jnp.ndarray, K-sparse inverse spectral density.   
    """

    assert isinstance(F_hat, jnp.ndarray) and F_hat.dtype in (jnp.complex64, jnp.complex128) and F_hat.shape[0]==F_hat.shape[1], 'F_hat must be a hermitian tensor size (N,N,M).'
    assert isinstance(k, jnp.ndarray) and k.dtype in (jnp.int16, jnp.int32) and k.ndim==1, 'k must be an one dimensional array of integers.'

    N,_,M=F_hat.shape
    
    if K is None:
            K = array([0,M])
    else:
        assert isinstance(K, jnp.ndarray) and K.dtype in (jnp.int32, jnp.int64) and K.ndim==1, 'indexes of frequencies must be a one dim array of integers.'
        assert ((0<K) & (K<M)).sum()==len(K), 'Splitting point must be in [1,2,...,M-1].' 
        K = append(insert(K,0,0), M)

    if k.shape[-1]==1:
        k=jnp.ones(K.shape[-1]-1, dtype=jnp.int32)*k
    else:
        assert k.shape[-1]==K.shape[-1]-1, 'The lenght of k must be equal the number of considered intervals.'

    vhpinv=jax.vmap(hpinv, (-1), -1)
    P=vhpinv(F_hat)
    Psk=zeros_like(P)
    
    #compute the l2 norms of the blocks
    for g in range(len(K)-1):
        Pl2=tril(norm(P[...,K[g]:K[g+1]], axis=-1, ord=2),k=-1)
        mask = expand_dims(top_k_mask(jnp.round(Pl2,5), k[g], N),2)
        kval = mask*P[...,K[g]:K[g+1]]
        Psk = Psk.at[...,K[g]:K[g+1]].set(kval)
    
    return Psk

class IA_method:
    def __init__(self, Y=None, K=None, F_hat=None, P_init='identity', penalize_diag=False, seed=0):
        """
        Iterative approximate (IA) method for the general case where prior domain knowledge is not available.

        INPUT
        =====
        - Y: jnp.ndarray, time series dataset of size (N,T)
        - K: jnp.array of type np.int, splitting points for frequencies to define groups for block soft-thresholding.
            Example: T=100, K=[T//10, T//4] --> The groups will be {[0, T//10); [T//10, T//4); [T//4, T//2]}
        - F: jnp.ndarray, intial value for the periodogram, size (N, N, M=T//2+1)
        - P: jnp.ndarray, intial value for the inverse periodogram, size (N, N, M)  
        - P_init: str, one among ('inverse', 'identity', 'random') 
        - penalize_diag: bool, if True the algo penalizes the diagonal terms as well.
        """
        
        assert isinstance(penalize_diag,bool), 'penalize_diag must be True or False'
        assert not (Y is  None and F_hat is None), 'You must provide either Y or F_hat'

        if Y is not None: 
            assert isinstance(Y, jnp.ndarray) and Y.dtype in (jnp.float32,jnp.float64), 'Y must be a jnp.ndarray of float values'
            self.T = Y.shape[1]

        if F_hat is None:
            self.YF=rfft(Y, axis=-1, norm="ortho")
            #biased periodogram (only positive frequencies), matrix form
            self.bF = 1/self.T*einsum("nk,mk->nmk", self.YF, self.YF.conj())
        else:
            self.bF = deepcopy(F_hat)

        self.N,self.T1 = self.bF.shape[-2:]
        self.M = self.N*(self.N+1)//2
        self.N2 = self.N*self.N
        
        if penalize_diag: pen_d=0 
        else: pen_d=1.

        #creation of selection matrix
        self.O = ones((self.N,self.N), dtype=jnp.float64)-pen_d*eye(self.N, dtype=jnp.float64)
        self.vO = self.O.flatten(order='F')
        self.S = diag(self.vO)
        self.aS = kron(ones([1,1,self.T1]), expand_dims(self.S,2))

        if K is None:
            self.K = array([0,self.T1])
        else:
            assert isinstance(K, jnp.ndarray) and K.dtype in (jnp.int32, jnp.int64) and K.ndim==1, 'indexes of frequencies must be a one dim array of integers.'
            assert ((0<K) & (K<self.T1)).sum()==len(K), 'Splitting point must be in [1,2,...,max_freq-1].' 
            self.K = append(insert(K,0,0),self.T1)

        self.Ndims = diff(self.K) #number of frequencies within the block, i.e., size of the groups

        assert isinstance(seed,int), "The seed must be an integer."
        self.seed = seed

        self.P_init=P_init

        self.I = eye(self.N)
        self.vIn = self.I.reshape((self.N2,1), order='F')

    def initialization(self, F=None, P=None, check_init=True):
        #initialize F and P
        if F is None: 
            F=deepcopy(self.bF)
        else:
            F = deepcopy(F)
            assert isinstance(F, jnp.ndarray) and F.dtype in (jnp.complex64, jnp.complex128) and F.shape[0]==F.shape[1], 'F must be a power spectral density tensor (hermitian) of size (N,N,T1).'

        if P is None: 
            assert self.P_init in ('inverse', 'identity','random'), 'Please, provide P_init in "(inverse, identity,random)"'
            P=zeros_like(F, dtype=jnp.complex128)

            if  self.P_init=='inverse': 
                vhpinv=jax.vmap(hpinv, (-1), -1)
                P+=vhpinv(self.bF)
            elif  self.P_init=='identity': P+= kron(ones([1,1,self.T1]), expand_dims(eye(self.N),2))
            elif  self.P_init=='random': 
                key = PRNGKey(self.seed)
                A=normal(key, shape=self.bf.shape, dtype=jnp.complex128)
                P+=(A+A.conj().T)/2.

        else:
            P = deepcopy(P)
            assert isinstance(P, jnp.ndarray) and P.dtype in (jnp.complex64, jnp.complex128) and P.shape[0]==P.shape[1], 'P must be a valid inverse power spectral density (hermitian) tensor of size (N,N,T1).'

        aI = kron(jnp.ones([1,1,self.T1]), jnp.expand_dims(self.I,2))
        prod = einsum("ijk,jmk->imk", F,P)

        #vectorization
        vF=F.reshape((self.N2,1,self.T1), order='F')
        vP=P.reshape((self.N2,1,self.T1), order='F')
        vU = (prod-aI).reshape((self.N2,1,self.T1), order='F')
        vX = (prod-aI).reshape((self.N2,1,self.T1), order='F')
        vV = einsum('ijk,jlk->ilk', self.aS, vP)
        vW = deepcopy(vP)
        vL = deepcopy(vF)
        
        #initialize dual var alpha, beta, mu, phi
        alpha = zeros([self.N2,1,self.T1], dtype=jnp.complex128) #associated with inf-norm of subproblem in F 
        mdelta = zeros([self.N2,1,self.T1], dtype=jnp.complex128) #associated with indicator function of subproblem in F
        beta = zeros([self.T1], dtype=jnp.float64) #associated with the constraint concerning the inverse Fourier transform of the power spectrum 
        mu = zeros([self.N2,1,self.T1], dtype=jnp.complex128) #associated with inf-norm of subproblem in P
        momega = zeros([self.N2,1,self.T1], dtype=jnp.complex128) #associated with indicator function of subproblem in P
        phi = zeros([self.N2,1,self.T1], dtype=jnp.complex128) #associated with the l21 regularization

        if check_init:
            #check init separately for each freq
            for k in range(self.T1):
                AF=kron(P[...,k].T, self.I)
                AP=kron(self.I,F[...,k])

                assert jnp.allclose(AF@vF[...,k]-self.vIn, vU[...,k]), "Inizialization violates feasibility for U at frequency {}".format(k)
                assert jnp.allclose(AP@vP[...,k]-self.vIn, vX[...,k]), "Inizialization violates feasibility for X at frequency {}".format(k)
                assert jnp.allclose(self.S@vP[...,k], vV[...,k]), "Inizialization violates feasibility for V {}".format(k)
               
        return F, P, vF, vP, vU, vX, vV, vL, vW, alpha, mdelta, beta, mu, momega, phi
    
    @partial(jax.jit, static_argnums=0)
    def hprojection(self, wf, vf):
        return jnp.maximum(wf,self.epsilon)*vf.reshape(-1,1)@vf.reshape(-1,1).T.conj()
    
    #here we assume to pass a single slice
    @partial(jax.jit, static_argnums=0)
    def compute_F(self, P, oldF, oldU, oldL, bF, alpha, mdelta, beta, AF_H, rho, delta):

        invGamma2 = kron(self.I, pinv((self.tauF/2+beta+delta/2)*self.I + rho/2*P.T.conj()@P))
        f_val = invGamma2@(self.tauF/2*oldF + delta/2*(oldL-mdelta/delta)+ rho/2 * AF_H@(self.vIn + oldU - alpha/rho)+ beta*bF.reshape((self.N2,1), order='F'))
        return f_val
    
    @partial(jax.jit, static_argnums=0)
    def fbeta(self, beta, P, oldF, oldU, oldL, bF, alpha, mdelta, AF_H, rho, delta):

        f_val = self.compute_F( P, oldF, oldU, oldL, bF, alpha, mdelta, beta, AF_H, rho, delta)  
        contr = f_val-bF.reshape((self.N2,1), order='F')
        return beta*(norm(contr.flatten(), ord=2)**2 - self.eta/self.T1)
    
    @partial(jax.jit, static_argnums=0)
    def cmpl_slcknss_beta_null(self, P, oldF, oldU, oldL, bF, alpha, mdelta, AF_H, rho, delta):
        beta=0.
        f_val=self.compute_F( P, oldF, oldU, oldL, bF, alpha, mdelta, beta, AF_H, rho, delta)
        contr = f_val-bF.reshape((self.N2,1), order='F')
        return f_val, contr, beta
    
    @partial(jax.jit, static_argnums=0)
    def cmpl_slcknss_beta_notnull(self, P, oldF, oldU, oldL, bF, alpha, mdelta, AF_H, f_val_c, rho, delta, beta0):
        bisec = Bisection(self.fbeta, 1./beta0, beta0, maxiter=1000, check_bracket=False, tol=1.e-5)
        beta = bisec.run(P=P, oldF=oldF, oldU=oldU, oldL=oldL, bF=bF, alpha=alpha, mdelta=mdelta, AF_H=AF_H, rho=rho, delta=delta).params
        f_val=self.compute_F( P, oldF, oldU, oldL, bF, alpha, mdelta, beta, AF_H, rho, delta)
        return f_val, beta
    
    @partial(jax.jit, static_argnums=0)
    def do_nothing(self, P, oldF, oldU, oldL, bF, alpha, mdelta, AF_H, f_val_c, rho, delta, beta0):
        return f_val_c, 0.
    
    @partial(jax.jit, static_argnums=0)
    def check_interval(self, P, oldF, oldU, oldL, bF, alpha, mdelta, AF_H, rho, delta):
        beta0 = 1.e2
        f_increase = lambda x: x*10.
        f_opposite = lambda x: self.fbeta(x, P, oldF, oldU, oldL, bF, alpha, mdelta, AF_H, rho, delta)*self.fbeta(1./x, P, oldF, oldU, oldL, bF, alpha, mdelta, AF_H, rho, delta)>=0.
        f_magnitude = lambda x: x<1.e8
        def f_check(x):
            return f_opposite(x) & f_magnitude(x)     
        beta0 = while_loop(f_check, f_increase, beta0)
        cnd1=self.fbeta(beta0, P, oldF, oldU, oldL, bF, alpha, mdelta, AF_H, rho, delta)*self.fbeta(1./beta0, P, oldF, oldU, oldL, bF, alpha, mdelta, AF_H, rho, delta)<0
        return cnd1, beta0
    
    @partial(jax.jit, static_argnums=0)
    def do_False(self, P, oldF, oldU, oldL, bF, alpha, mdelta, AF_H, rho, delta):
        return False, 0.
    
    @partial(jax.jit, static_argnums=0)
    def solve_F(self, P, oldF, oldU, oldL, bF, alpha, mdelta, AF_H, rho, delta):
        
        #1) first evaluate at beta=0
        f_val, contr, _ = self.cmpl_slcknss_beta_null(P, oldF, oldU, oldL, bF, alpha, mdelta, AF_H, rho, delta)
        #2) check if primal feasibility is not satisfied
        cnd0=norm(contr.flatten(), ord=2)**2 - self.eta/self.T1>1.e-10
        #3) if not satisfied, find an interval for bisection, else do nothing
        cnd1, beta0=cond(cnd0, self.check_interval, self.do_False, P, oldF, oldU, oldL, bF, alpha, mdelta, AF_H, rho, delta)
        #4) if the interval has been found apply bisection, else return solution in 0
        f_val, beta = cond(cnd1, self.cmpl_slcknss_beta_notnull, self.do_nothing, P, oldF, oldU, oldL, bF, alpha, mdelta, AF_H, f_val, rho, delta, beta0)
                
        return f_val, beta

    @partial(jax.jit, static_argnums=0)
    def jitted_l2(self, x):
        return norm(x, ord=2)

    @partial(jax.jit, static_argnums=0)
    def jitted_prox_group_lasso(self,x, scaling):
        return prox_group_lasso(x, self.lmbd, scaling=scaling)
    
    def objective(self, vF, oldF, vP, oldP, vU, vX, vV):
        
        gsparsity=0.
        
        #Here we need to compute l21 norm.
        #Since we multiply vP by the mask,
        #the intra-slice diagonal elements of vV are 
        #already zero.
        
        for g in range(len(self.K)-1):
            gsparsity+=(jax.vmap(self.jitted_l2, 0, 0)(jnp.squeeze(vV)[...,self.K[g]:self.K[g+1]])).sum()

        tU=jnp.max(absolute(vU), axis=(0,1)).sum()
        tX=jnp.max(absolute(vX), axis=(0,1)).sum()
        tF=self.tauF/2*(norm(vF-oldF, axis=0)**2).sum()
        tP=self.tauP/2*(norm(vP-oldP, axis=0)**2).sum()
        tl21=self.lmbd*gsparsity

        loss=tU+ tX+tF+tP+tl21
    
        return loss

    @partial(jax.jit, static_argnums=0)
    def blockpinv(self, eqcontr, ei, theta):
        return pinv(eqcontr + theta/2*self.I*ei)
    
    @partial(jax.jit, static_argnums=0)
    def parallel_update(self, F,P,vF,vP,vU,vX,vV,vW,vL,alpha,mdelta,beta,mu,momega,phi, oldF, oldU, bF, oldP, oldX, oldV, oldW, oldL, stepsize, rho, sigma, theta, omega, delta, iteration):
        
        AF=kron(P.T, self.I)
        AP=kron(self.I,F)
        AF_H=AF.T.conj()
        AP_H=AP.T.conj()

        eqcontr = self.tauP/2*self.I + omega/2*self.I + sigma/2*F.T.conj()@F
        vblockpinv = jax.vmap(self.blockpinv, (None,1,None),0)
        binvGamma1 = vblockpinv(eqcontr, self.O, theta)
        
        invGamma1 = block_diag(*binvGamma1)
        
        #F 
        f_val, beta = self.solve_F(P, oldF, oldU, oldL, bF, alpha, mdelta, AF_H, rho, delta)
        vF = oldF + stepsize*(f_val-oldF)
        AFnewF = AF@vF

        #U
        newu = AFnewF  - self.vIn + alpha/rho
        vU = L1Ballproxdual(newu, 1./rho, ones(self.N2), rho)

        #L
        Mdelta = mdelta.reshape((self.N,self.N), order='F')
        MF = vF.reshape((self.N,self.N), order='F')
        L_k = MF + Mdelta
        L_k_H = (L_k + L_k.T.conj())/2.
        wl, vl = eigh(L_k_H, symmetrize_input=False)
        L_k_p = zeros_like(L_k_H, dtype=complex)

        vhprojectionl = jax.vmap(self.hprojection, (0 ,1), 2)
        L_k_p = vhprojectionl(wl, vl).sum(axis=-1)
        vL = L_k_p.reshape((self.N2,1), order='F')
        
        #alpha
        rp1 = AFnewF - self.vIn - vU
        alpha += rho*rp1

        #mdelta
        rp5 = vF - vL
        mdelta += delta*rp5
        
        #beta
        dspectrum = vF-bF.reshape((self.N2,1), order='F')
        
        #P
        p_val = invGamma1@(self.tauP/2*oldP + omega/2*(oldW - momega/omega)+ sigma/2*AP_H @(self.vIn + oldX - mu/sigma)+theta/2*self.S@(oldV - phi/theta))
        vP = oldP + stepsize*(p_val-oldP)
        APnewP = AP@vP
        
        #X
        newx = APnewP - self.vIn + mu/sigma
        vX = L1Ballproxdual(newx, 1./sigma, ones(self.N2), sigma)

        #W
        Momega = momega.reshape((self.N,self.N), order='F')
        MP = vP.reshape((self.N,self.N), order='F')
        W_k = MP + Momega
        W_k_H = (W_k + W_k.T.conj())/2.
        ww, vw = eigh(W_k_H, symmetrize_input=False)
        W_k_p = zeros_like(W_k_H, dtype=complex)

        vhprojectionw = jax.vmap(self.hprojection, (0 ,1), 2)
        W_k_p = vhprojectionw(ww, vw).sum(axis=-1)
        vW = W_k_p.reshape((self.N2,1), order='F')
        
        #mu
        rp2 = APnewP - self.vIn - vX
        mu += sigma*(rp2)

        #momega
        rp6 = vP - vW
        momega += omega*rp6
        
        #Contribution to dual residual
        rd1 = delta/2*(vL-oldL) + rho/2*AF_H@(vU-oldU)
        rd2 = omega/2*(vW-oldW) + sigma/2*AP_H@(vX-oldX)
        
        #update the tensors
        F = deepcopy(vF.reshape((self.N,self.N), order='F'))
        P = deepcopy(vP.reshape((self.N,self.N), order='F'))

        #primal
        max1_=AFnewF
        max2_=APnewP
        max7_=dspectrum
        
        #dual
        maxd1_=.5*AF_H@alpha
        maxd2_=AP_H@mu
        maxd3_=beta*dspectrum

        return vF, beta, vU, vL, rp1, alpha, rp5, mdelta,\
             vP, vX, vW, rp2, mu, rp6, momega,\
                 rd1, rd2, F, P,\
                     max1_, max2_, max7_, maxd1_, maxd2_, maxd3_

    def solve_vmap(self, F, P, vF, vP, vU, vX, vV, vL, vW, alpha, mdelta, beta, mu, momega, phi, lmbd=.1, tauF=1.e-2, tauP=1.e-2, eta=1.e-2, epsilon=1.e-8, kind=0, kind1=0, c1=.1, c2=.9, c3=.9, c4=.9, eps=.9, eps1=.9, tolp_abs=1.e-3, tolp_rel=1.e-3, told_abs=1.e-3, told_rel=1.e-3, only_primal=False, max_iter=500, step=10):
        """
        This function solves the problem as described in *** paper ***, parallelizing on a single device (see https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html)
        The optimization procedure stops when the residuals of the primal and dual problems are lower than the provided tolerances.

        Ref.
        [1] Nedić, Angelia, et al. "Parallel and distributed successive convex approximation methods for big-data optimization." Multi-agent Optimization: Cetraro, Italy 2014 (2018): 141-308.
        [2] Boyd, Stephen, et al. "Distributed optimization and statistical learning via the alternating direction method of multipliers." Foundations and Trends® in Machine learning 3.1 (2011): 1-122.

        INPUT
        =====
        - F: jnp.ndarray, intial value for the periodogram, size (N, N, M=T//2+1)
        - P: jnp.ndarray, intial value for the inverse periodogram, size (N, N, M)
        - vF, vP, vU, vX, vV, vL, vW, alpha, mdelta, beta, mu, momega, phi: jnp.ndarray(s) returned by _init_
        - lmbd: strictly positive float, strength of l21-norm regularization.
            If ||V_j||_2<=lmbd/self.theta then the block V_j is set to zero;
        - tauF: strictly positive float, constant used to build the strongly convex surrogate objective (term involving F).
            The larger, the greater the importance wrt F@P-I=0 term;
        - tauP: strictly positive float, constant used to build the strongly convex surrogate objective (term involving P).
            The larger, the greater the importance wrt F@P-I=0 term;
        - eta: strictly positive float, tolerance for the Frobenious norm constraint;
        - epsilon: strictly positive float, closure for the positive definite set of matrices;
        - kind: int, it identifies the diminishing self.stepsize rule for SCA. 
            If 0, uses Eq. (109) in [1] with sqrt at denominator, if 1 uses Eq (109) with linear term at denominator, if 2 uses Eq (108).
            Otherwise the stepsize is kept fixed.
        - kind1: int, it identifies the diminishing self.stepsize rule for ADMM. 
            If 0, uses Eq. (109) in [1] with sqrt at denominator, if 1 uses Eq (109) with linear term at denominator, if 2 uses Eq (108).
            Otherwise the stepsize is kept fixed. 
        - c1: float, alpha in Eq. (109). Used if kind in {0,1} by SCA.
        - c2: float, beta in Eq. (109). Used if kind in {0,1} by SCA.
        - c3: float, alpha in Eq. (109). Used if kind in {0,1} by ADMM.
        - c4: float, beta in Eq. (109). Used if kind in {0,1} by ADMM.
        - eps: float, epsilon in Eq. (108). Used if kind=2 by SCA.
        - eps1: float, epsilon in Eq. (108). Used if kind=2 by ADMM. 
        - tolp_abs: float, absolute tolerance for primal residual, see sec. 3.1.1 in [2].
        - tolp_rel: float, relative tolerance for primal residual, see sec. 3.1.1 in [2].
        - told_abs: float, absolute tolerance for dual residual, see sec. 3.1.1 in [2].
        - told_rel: float, relative tolerance for dual residual, see sec. 3.1.1 in [2].
        - max_iter: int, maximum number of self.iteration. If primal and dual residual is not achieved in max_iter, the algo will stop.
        - only_primal: bool, if True the algo checks only primal conditions for determining convergence.
        - step: int, defines the logging of the algo. It prints opt information each "step" self.iterations.
        """

        iteration=0.
        dims=self.N*jnp.sqrt(self.T1)
        self.lmbd=lmbd
        self.tauP=tauP
        self.tauF=tauF
        self.eta=eta
        self.epsilon=epsilon

        Rp1, Rp2, Rp3, Rp4, Rp5, Rp6 = jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf
        Rd1, Rd2 = jnp.inf, jnp.inf

        Max1p, Max2p, Max3p, Max4p, Max5p, Max6p = 0.,0.,0.,0.,0.,0.
        Max1d, Max2d = 0.,0.

        #store objective and primal and dual
        #feasibility residuals for checking
        #their behaviour along t
        self.objective_values = jnp.nan*zeros([max_iter-1], dtype=jnp.float64)
        self.Rp1s = jnp.nan*zeros([max_iter-1], dtype=jnp.float64)
        self.Rp2s = jnp.nan*zeros([max_iter-1], dtype=jnp.float64)
        self.Rp3s = jnp.nan*zeros([max_iter-1], dtype=jnp.float64)
        self.Rp4s = jnp.nan*zeros([max_iter-1], dtype=jnp.float64)
        self.Rp5s = jnp.nan*zeros([max_iter-1], dtype=jnp.float64)
        self.Rp6s = jnp.nan*zeros([max_iter-1], dtype=jnp.float64)
        self.Rd1s = jnp.nan*zeros([max_iter-1], dtype=jnp.float64)
        self.Rd2s = jnp.nan*zeros([max_iter-1], dtype=jnp.float64)

        def f1():
            return 1.
        def f2():
            return 1/eps - .01
        
        cond_stepsize = lambda x: x!=2
        check_feasibility_p = lambda x,y: x < dims*tolp_abs +  tolp_rel*y
        check_feasibility_d = lambda x,y: x < dims*told_abs +  told_rel*y

        def all_feasibility(Rp1, Rp2, Rp3, Rp4, Rp5, Rp6, Rd1, Rd2, Max1p, Max2p, Max3p, Max4p, Max5p, Max6p, Max1d, Max2d, iteration):
            if only_primal:
                all_cond=not (check_feasibility_p(Rp1, Max1p) and check_feasibility_p(Rp2, Max2p) and check_feasibility_p(Rp3, Max3p) and check_feasibility_p(Rp4, Max4p) and check_feasibility_p(Rp5, Max5p) and check_feasibility_p(Rp6, Max6p)) and iteration<max_iter
            else:
                all_cond = not (check_feasibility_p(Rp1, Max1p) and check_feasibility_p(Rp2, Max2p) and check_feasibility_p(Rp3, Max3p) and check_feasibility_p(Rp4, Max4p) and check_feasibility_p(Rp5, Max5p) and check_feasibility_p(Rp6, Max6p) and check_feasibility_d(Rd1, Max1d) and check_feasibility_d(Rd2, Max2d)) and iteration<max_iter
            return all_cond
        
        stepsize = cond(cond_stepsize(kind), f1, f2)
        stepsize1 = cond(cond_stepsize(kind1), f1, f2)

        rho=stepsize1
        sigma=stepsize1
        theta=stepsize1
        omega=stepsize1
        delta=stepsize1
        tau_prime=1./theta

        vparallel = jax.vmap(self.parallel_update, (2,2,2,2,2,2,2,2,2,2,2,0,2,2,2,2,2,2,2,2,2,2,2,None, None, None, None, None, None, None), (2,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2))
        
        while all_feasibility(Rp1, Rp2, Rp3, Rp4, Rp5, Rp6, Rd1, Rd2, Max1p, Max2p, Max3p, Max4p, Max5p, Max6p, Max1d, Max2d, iteration):
            
            # Store previous values
            oldF = deepcopy(vF)
            oldU = deepcopy(vU)
            oldP = deepcopy(vP)
            oldX = deepcopy(vX)
            oldV = deepcopy(vV)
            oldW = deepcopy(vW)
            oldL = deepcopy(vL)
            
            iteration+=1

            # Check feasibility at the starting point
            if iteration==1:
                assert norm((F-self.bF).flatten(), ord=2)**2-eta<=0., "Initialization violates feasibility condition ||F-F_tilde||_2^2-eta<=0."     

                 
            # Here we parallelize through vectorization
            vF, beta, vU, vL, rp1, alpha, rp5, mdelta, vP, vX, vW, rp2, mu, rp6, momega, rd1, rd2, F, P, max1_, max2_, max7_, maxd1_, maxd2_, maxd3_ = vparallel(F,P,vF,vP,vU,vX,vV,vW,vL,alpha,mdelta,beta,mu,momega,phi, oldF, oldU, self.bF, oldP, oldX, oldV, oldW, oldL, stepsize, rho, sigma, theta, omega, delta, iteration)
                                           
            #V
            vjitted_prox_group_lasso = jax.vmap(self.jitted_prox_group_lasso, (0, None), 0)
            vSP = einsum('ijk,jlk->ilk', self.aS, vP)
            
            # Apply block-soft thresholding using the provided indexes of groups
            for g in range(len(self.K)-1):
                v = vSP[...,self.K[g]:self.K[g+1]] + phi[...,self.K[g]:self.K[g+1]]/theta
                vV = vV.at[...,self.K[g]:self.K[g+1]].set(complex_sign(v)*vjitted_prox_group_lasso(absolute(v), tau_prime))
            
            #phi
            phi += theta*(vSP - vV)
            vSPhi = einsum('ijk,jlk->ilk', self.aS, phi)

            # Update the objective
            obj=self.objective(vF, oldF, vP, oldP, vU, vX, vV)
            
            # Residuals
            max1 = norm(max1_.flatten(), ord=2) #t3
            max2 = norm(max2_.flatten(), ord=2) #t1
            max3 = norm(vU.flatten(), ord=2) #t4
            max4 = norm(vX.flatten(), ord=2) #t2
            max5 = norm(vSP.flatten(), ord=2) #t6
            max6 = norm(vV.flatten(), ord=2) #t7
            max7 = norm(max7_.flatten(), ord=2) #t5
            max8 = norm(vP.flatten(), ord=2) #t8
            max9 = norm(vW.flatten(), ord=2) #t9
            max10 = norm(vF.flatten(), ord=2) #t10
            max11 = norm(vL.flatten(), ord=2) #t11

            maxd12 = norm(momega.flatten(), ord=2) #omega*t12
            maxd1 = norm(maxd1_.flatten(), ord=2) #rho/2*t16
            maxd2 = norm(maxd2_.flatten(), ord=2) #sigma*t13
            maxd4 = norm(vSPhi.flatten(), ord=2) #theta*t14
            maxd15 = .5*norm(mdelta.flatten(), ord=2) #delta/2*t15

            rp3 = norm((F-self.bF).flatten(), ord=2)**2 - eta 
            rp4 = vSP - vV
            
            rd2 += theta/2.*einsum('ijk,jlk->ilk', self.aS, (vV-oldV))

            Rp1, Rp2, Rp3, Rp4, Rp5, Rp6 = norm(rp1.flatten(order='F'), ord=2), norm(rp2.flatten(order='F'),ord=2), abs(rp3), norm(rp4.flatten(order='F'),ord=2), norm(rp5.flatten(order='F'),ord=2), norm(rp6.flatten(order='F'),ord=2) 
            Rd1, Rd2 = norm(rd1.flatten(order='F'),ord=2), norm(rd2.flatten(order='F'),ord=2)

            Max1p, Max2p, Max3p, Max4p, Max5p, Max6p = max(max1, max3, jnp.sqrt(self.T1*self.N)), max(max2, max4, jnp.sqrt(self.T1*self.N)), max(max7,eta/jnp.sqrt(self.T1)), max(max5, max6), max(max10,max11), max(max8, max9) 
            Max1d, Max2d = max(maxd15, maxd1), .5*max(maxd12, maxd2, maxd4)

            #store the values
            self.objective_values = self.objective_values.at[int(iteration)-1].set(obj)
            self.Rp1s = self.Rp1s.at[int(iteration)-1].set(Rp1)
            self.Rp2s = self.Rp2s.at[int(iteration)-1].set(Rp2)
            self.Rp3s = self.Rp3s.at[int(iteration)-1].set(Rp3)
            self.Rp4s = self.Rp4s.at[int(iteration)-1].set(Rp4)
            self.Rp5s = self.Rp5s.at[int(iteration)-1].set(Rp5)
            self.Rp6s = self.Rp6s.at[int(iteration)-1].set(Rp6)
            self.Rd1s = self.Rd1s.at[int(iteration)-1].set(Rd1)
            self.Rd2s = self.Rd2s.at[int(iteration)-1].set(Rd2)

            # Diminishing stepsize of SCA
            if kind==0:
                stepsize = (stepsize+jnp.log(iteration)**(c1))/(1.+ c2*jnp.sqrt(iteration))
            elif kind==1:
                stepsize = (stepsize+jnp.log(iteration)**(c1))/(1.+ c2*iteration)
            elif kind==2:
                stepsize=stepsize*(1-eps*stepsize)
            elif kind==3:
                stepsize = (stepsize+c1)/(1.+ c2*iteration)
            else:
                pass
            
            # Diminishing stepsize of ADMM
            if kind1==0:
                stepsize1 = (stepsize1+jnp.log(iteration)**(c3))/(1.+ c4*jnp.sqrt(iteration))
            elif kind1==1:
                stepsize1 = (stepsize1+jnp.log(iteration)**(c3))/(1.+ c4*iteration)
            elif kind1==2:
                stepsize1=stepsize1*(1-eps1*stepsize1)
            else:
                pass

            rho=stepsize1
            sigma=stepsize1
            theta=stepsize1
            omega=stepsize1
            delta=stepsize1
            tau_prime=1./theta

            if iteration%step==0:
                # Update the conditions for primal and dual residuals
                # primal
                cond1p = check_feasibility_p(Rp1,Max1p)
                cond2p = check_feasibility_p(Rp2,Max2p)
                cond3p = check_feasibility_p(Rp3,Max3p)
                cond4p = check_feasibility_p(Rp4,Max4p)
                cond5p = check_feasibility_p(Rp5,Max5p)
                cond6p = check_feasibility_p(Rp6,Max6p)
                # dual
                cond1d = check_feasibility_d(Rd1,Max1d)
                cond2d = check_feasibility_d(Rd2,Max2d)
                
                print("\n\nIteration: {}".format(iteration))
                print("objective: {}".format(obj))
                print("Residuals for primal feasibility: U {}; X {}; FFT(F) {}; V {}; L {}; W {}".format(jnp.round(Rp1,3), jnp.round(Rp2,3), jnp.round(Rp3,3), jnp.round(Rp4,3), jnp.round(Rp5,3), jnp.round(Rp6,3)))
                print("Residual for dual feasibility: sp F {}vs{}; sp P {}vs{}".format(jnp.round(Rd1,3), jnp.round(dims*told_abs +  told_rel*Max1d,3), jnp.round(Rd2,3),jnp.round(dims*told_abs +  told_rel*Max2d,3)))
                print("Primal feasibility conditions: {}, {}, {}, {}, {}, {}".format(cond1p, cond2p, cond3p, cond4p, cond5p, cond6p))
                print("Dual feasibility conditions: {}, {}".format(cond1d, cond2d))
                print("SCA stepsize value: {}".format(stepsize))
                print("ADMM stepsize value: {}".format(stepsize1))
                print('\n\n')

        # Conditions primal
        cond1p = check_feasibility_p(Rp1,Max1p)
        cond2p = check_feasibility_p(Rp2,Max2p)
        cond3p = check_feasibility_p(Rp3,Max3p)
        cond4p = check_feasibility_p(Rp4,Max4p)
        cond5p = check_feasibility_p(Rp5,Max5p)
        cond6p = check_feasibility_p(Rp6,Max6p)
        # Conditions dual
        cond1d = check_feasibility_d(Rd1,Max1d)
        cond2d = check_feasibility_d(Rd2,Max2d)

        if not all_feasibility(Rp1, Rp2, Rp3, Rp4, Rp5, Rp6, Rd1, Rd2, Max1p, Max2p, Max3p, Max4p, Max5p, Max6p, Max1d, Max2d, iteration) and iteration<max_iter:
            print("################# Optimisation successfully completed #################")
            print("\n\nLast iteration: {}".format(iteration))
            print("objective: {}".format(obj))
            print("Residuals for primal feasibility: U {}; X {}; FFT(F) {}; V {}; L {}; W {}".format(jnp.round(Rp1,3), jnp.round(Rp2,3), jnp.round(Rp3,3), jnp.round(Rp4,3), jnp.round(Rp5,3), jnp.round(Rp6,3)))
            print("Residual for dual feasibility: sp F {}vs{}; sp P {}vs{}".format(jnp.round(Rd1,3), jnp.round(dims*told_abs +  told_rel*Max1d,3), jnp.round(Rd2,3),jnp.round(dims*told_abs +  told_rel*Max2d,3)))
            print("Primal feasibility conditions: {}, {}, {}, {}, {}, {}".format(cond1p, cond2p, cond3p, cond4p, cond5p, cond6p))
            print("Dual feasibility conditions: {}, {}".format(cond1d, cond2d))
            print("SCA stepsize value: {}".format(stepsize))
            print("ADMM stepsize value: {}".format(stepsize1))
            print('\n\n')
        else:
            print("################# Maximum number of iterations reached #################")
            print("\n\nLast iteration: {}".format(iteration))
            print("objective: {}".format(obj))
            print("Residuals for primal feasibility: U {}; X {}; FFT(F) {}; V {}; L {}; W {}".format(jnp.round(Rp1,3), jnp.round(Rp2,3), jnp.round(Rp3,3), jnp.round(Rp4,3), jnp.round(Rp5,3), jnp.round(Rp6,3)))
            print("Residual for dual feasibility: sp F {}vs{}; sp P {}vs{}".format(jnp.round(Rd1,3), jnp.round(dims*told_abs +  told_rel*Max1d,3), jnp.round(Rd2,3),jnp.round(dims*told_abs +  told_rel*Max2d,3)))
            print("Primal feasibility conditions: {}, {}, {}, {}, {}, {}".format(cond1p, cond2p, cond3p, cond4p, cond5p, cond6p))
            print("Dual feasibility conditions: {}, {}".format(cond1d, cond2d))
            print("SCA stepsize value: {}".format(stepsize))
            print("ADMM stepsize value: {}".format(stepsize1))
            print('\n\n')

        return {'F':F,'P':P,'vecF':vF,'vecP':vP,'vecU':vU,'vecX':vX,'vecV':vV,'vecL':vL,'vecW':vW,'alpha': alpha,'delta': mdelta,'beta': beta, 'mu': mu, 'omega': momega, 'phi': phi,'objective value': obj, 'iterations':iteration, 'cond1p':cond1p, 'cond2p':cond2p, 'cond3p':cond3p, 'cond4p':cond4p, 'cond4p':cond5p, 'cond4p':cond6p, 'cond1d':cond1d,'cond2d':cond2d}
    
    def solve_pmap(self, F, P, vF, vP, vU, vX, vV, alpha, beta, mu, phi, lmbd=.1, tauF=1.e-2, tauP=1.e-2, eta=1.e-2, epsilon=1.e-8, kind=0, kind1=0, c1=.1, c2=.9, c3=.9, c4=.9, eps=.9, eps1=.9, tolp_abs=1.e-3, tolp_rel=1.e-3, told_abs=1.e-3, told_rel=1.e-3, only_primal=False, max_iter=500, step=10):
        """
        This function solves the problem as described in *** paper ***, parallelizing on multiple devices (see https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html).
        The optimization procedure stops when the residuals of the primal and dual problems are lower than the provided tolerances.

        Ref.
        [1] Nedić, Angelia, et al. "Parallel and distributed successive convex approximation methods for big-data optimization." Multi-agent Optimization: Cetraro, Italy 2014 (2018): 141-308.
        [2] Boyd, Stephen, et al. "Distributed optimization and statistical learning via the alternating direction method of multipliers." Foundations and Trends® in Machine learning 3.1 (2011): 1-122.

        INPUT
        =====
        - F: jnp.ndarray, intial value for the periodogram, size (N, N, M=T//2+1)
        - P: jnp.ndarray, intial value for the inverse periodogram, size (N, N, M)
        - vF, vP, vU, vX, vV, vL, vW, alpha, mdelta, beta, mu, momega, phi: jnp.ndarray(s) returned by _init_
        - lmbd: strictly positive float, strength of l21-norm regularization.
            If ||V_j||_2<=lmbd/self.theta then the block V_j is set to zero;
        - tauF: strictly positive float, constant used to build the strongly convex surrogate objective (term involving F).
            The larger, the greater the importance wrt F@P-I=0 term;
        - tauP: strictly positive float, constant used to build the strongly convex surrogate objective (term involving P).
            The larger, the greater the importance wrt F@P-I=0 term;
        - eta: strictly positive float, tolerance for the Frobenious norm constraint;
        - epsilon: strictly positive float, closure for the positive definite set of matrices;
        - kind: int, it identifies the diminishing self.stepsize rule for SCA. 
            If 0, uses Eq. (109) in [1] with sqrt at denominator, if 1 uses Eq (109) with linear term at denominator, if 2 uses Eq (108).
            Otherwise the stepsize is kept fixed.
        - kind1: int, it identifies the diminishing self.stepsize rule for ADMM. 
            If 0, uses Eq. (109) in [1] with sqrt at denominator, if 1 uses Eq (109) with linear term at denominator, if 2 uses Eq (108).
            Otherwise the stepsize is kept fixed. 
        - c1: float, alpha in Eq. (109). Used if kind in {0,1} by SCA.
        - c2: float, beta in Eq. (109). Used if kind in {0,1} by SCA.
        - c3: float, alpha in Eq. (109). Used if kind in {0,1} by ADMM.
        - c4: float, beta in Eq. (109). Used if kind in {0,1} by ADMM.
        - eps: float, epsilon in Eq. (108). Used if kind=2 by SCA.
        - eps1: float, epsilon in Eq. (108). Used if kind=2 by ADMM. 
        - tolp_abs: float, absolute tolerance for primal residual, see sec. 3.1.1 in [2].
        - tolp_rel: float, relative tolerance for primal residual, see sec. 3.1.1 in [2].
        - told_abs: float, absolute tolerance for dual residual, see sec. 3.1.1 in [2].
        - told_rel: float, relative tolerance for dual residual, see sec. 3.1.1 in [2].
        - max_iter: int, maximum number of self.iteration. If primal and dual residual is not achieved in max_iter, the algo will stop.
        - only_primal: bool, if True the algo checks only primal conditions for determining convergence.
        - step: int, defines the logging of the algo. It prints opt information each "step" self.iterations.
        """

        iteration=0.
        dims=self.N*jnp.sqrt(self.T1)
        self.lmbd=lmbd
        self.tauP=tauP
        self.tauF=tauF
        self.eta=eta
        self.epsilon=epsilon

        Rp1, Rp2, Rp3, Rp4, Rp5, Rp6 = jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf
        Rd1, Rd2 = jnp.inf, jnp.inf

        Max1p, Max2p, Max3p, Max4p, Max5p, Max6p = 0.,0.,0.,0.,0.,0.
        Max1d, Max2d = 0.,0.

        #store objective and primal and dual
        #feasibility residuals for checking
        #their behaviour along t
        self.objective_values = jnp.nan*zeros([max_iter-1], dtype=jnp.float64)
        self.Rp1s = jnp.nan*zeros([max_iter-1], dtype=jnp.float64)
        self.Rp2s = jnp.nan*zeros([max_iter-1], dtype=jnp.float64)
        self.Rp3s = jnp.nan*zeros([max_iter-1], dtype=jnp.float64)
        self.Rp4s = jnp.nan*zeros([max_iter-1], dtype=jnp.float64)
        self.Rp5s = jnp.nan*zeros([max_iter-1], dtype=jnp.float64)
        self.Rp6s = jnp.nan*zeros([max_iter-1], dtype=jnp.float64)
        self.Rd1s = jnp.nan*zeros([max_iter-1], dtype=jnp.float64)
        self.Rd2s = jnp.nan*zeros([max_iter-1], dtype=jnp.float64)

        def f1():
            return 1.
        def f2():
            return 1/eps - .01
        
        cond_stepsize = lambda x: x!=2
        check_feasibility_p = lambda x,y: x < dims*tolp_abs +  tolp_rel*y
        check_feasibility_d = lambda x,y: x < dims*told_abs +  told_rel*y

        def all_feasibility(Rp1, Rp2, Rp3, Rp4, Rp5, Rp6, Rd1, Rd2, Max1p, Max2p, Max3p, Max4p, Max5p, Max6p, Max1d, Max2d, iteration):
            if only_primal:
                all_cond=not (check_feasibility_p(Rp1, Max1p) and check_feasibility_p(Rp2, Max2p) and check_feasibility_p(Rp3, Max3p) and check_feasibility_p(Rp4, Max4p) and check_feasibility_p(Rp5, Max5p) and check_feasibility_p(Rp6, Max6p)) and iteration<max_iter
            else:
                all_cond = not (check_feasibility_p(Rp1, Max1p) and check_feasibility_p(Rp2, Max2p) and check_feasibility_p(Rp3, Max3p) and check_feasibility_p(Rp4, Max4p) and check_feasibility_p(Rp5, Max5p) and check_feasibility_p(Rp6, Max6p) and check_feasibility_d(Rd1, Max1d) and check_feasibility_d(Rd2, Max2d)) and iteration<max_iter
            return all_cond
        
        stepsize = cond(cond_stepsize(kind), f1, f2)
        stepsize1 = cond(cond_stepsize(kind1), f1, f2)

        rho=stepsize1
        sigma=stepsize1
        theta=stepsize1
        omega=stepsize1
        delta=stepsize1
        tau_prime=1./theta

        vparallel = jax.pmap(self.parallel_update, axis_name=None, in_axes=(2,2,2,2,2,2,2,2,2,2,2,0,2,2,2,2,2,2,2,2,2,2,2,None, None, None, None, None, None, None), out_axes=(2,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2))

        while all_feasibility(Rp1, Rp2, Rp3, Rp4, Rp5, Rp6, Rd1, Rd2, Max1p, Max2p, Max3p, Max4p, Max5p, Max6p, Max1d, Max2d, iteration):
            
            # Store previous values
            oldF = deepcopy(vF)
            oldU = deepcopy(vU)
            oldP = deepcopy(vP)
            oldX = deepcopy(vX)
            oldV = deepcopy(vV)
            oldW = deepcopy(vW)
            oldL = deepcopy(vL)
            
            iteration+=1

            # Check feasibility at the starting point
            if iteration==1:
                assert norm((F-self.bF).flatten(), ord=2)**2-eta<=0., "Initialization violates feasibility condition ||F-F_tilde||_2^2-eta<=0."     

                 
            # Here we parallelize through vectorization
            vF, beta, vU, vL, rp1, alpha, rp5, mdelta, vP, vX, vW, rp2, mu, rp6, momega, rd1, rd2, F, P, max1_, max2_, max7_, maxd1_, maxd2_, maxd3_ = vparallel(F,P,vF,vP,vU,vX,vV,vW,vL,alpha,mdelta,beta,mu,momega,phi, oldF, oldU, self.bF, oldP, oldX, oldV, oldW, oldL, stepsize, rho, sigma, theta, omega, delta, iteration)
                                           
            #V
            vjitted_prox_group_lasso = jax.vmap(self.jitted_prox_group_lasso, (0, None), 0)
            vSP = einsum('ijk,jlk->ilk', self.aS, vP)
            
            # Apply block-soft thresholding using the provided indexes of groups
            for g in range(len(self.K)-1):
                v = vSP[...,self.K[g]:self.K[g+1]] + phi[...,self.K[g]:self.K[g+1]]/theta
                vV = vV.at[...,self.K[g]:self.K[g+1]].set(complex_sign(v)*vjitted_prox_group_lasso(absolute(v), tau_prime))
            
            #phi
            phi += theta*(vSP - vV)
            vSPhi = einsum('ijk,jlk->ilk', self.aS, phi)

            # Update the objective
            obj=self.objective(vF, oldF, vP, oldP, vU, vX, vV)
            
            # Residuals
            max1 = norm(max1_.flatten(), ord=2) #t3
            max2 = norm(max2_.flatten(), ord=2) #t1
            max3 = norm(vU.flatten(), ord=2) #t4
            max4 = norm(vX.flatten(), ord=2) #t2
            max5 = norm(vSP.flatten(), ord=2) #t6
            max6 = norm(vV.flatten(), ord=2) #t7
            max7 = norm(max7_.flatten(), ord=2) #t5
            max8 = norm(vP.flatten(), ord=2) #t8
            max9 = norm(vW.flatten(), ord=2) #t9
            max10 = norm(vF.flatten(), ord=2) #t10
            max11 = norm(vL.flatten(), ord=2) #t11

            maxd12 = norm(momega.flatten(), ord=2) #omega*t12
            maxd1 = norm(maxd1_.flatten(), ord=2) #rho/2*t16
            maxd2 = norm(maxd2_.flatten(), ord=2) #sigma*t13
            maxd4 = norm(vSPhi.flatten(), ord=2) #theta*t14
            maxd15 = .5*norm(mdelta.flatten(), ord=2) #delta/2*t15

            rp3 = norm((F-self.bF).flatten(), ord=2)**2 - eta 
            rp4 = vSP - vV
            
            rd2 += theta/2.*einsum('ijk,jlk->ilk', self.aS, (vV-oldV))

            Rp1, Rp2, Rp3, Rp4, Rp5, Rp6 = norm(rp1.flatten(order='F'), ord=2), norm(rp2.flatten(order='F'),ord=2), abs(rp3), norm(rp4.flatten(order='F'),ord=2), norm(rp5.flatten(order='F'),ord=2), norm(rp6.flatten(order='F'),ord=2) 
            Rd1, Rd2 = norm(rd1.flatten(order='F'),ord=2), norm(rd2.flatten(order='F'),ord=2)

            Max1p, Max2p, Max3p, Max4p, Max5p, Max6p = max(max1, max3, jnp.sqrt(self.T1*self.N)), max(max2, max4, jnp.sqrt(self.T1*self.N)), max(max7,eta/jnp.sqrt(self.T1)), max(max5, max6), max(max10,max11), max(max8, max9) 
            Max1d, Max2d = max(maxd15, maxd1), .5*max(maxd12, maxd2, maxd4)

            #store the values
            self.objective_values = self.objective_values.at[int(iteration)-1].set(obj)
            self.Rp1s = self.Rp1s.at[int(iteration)-1].set(Rp1)
            self.Rp2s = self.Rp2s.at[int(iteration)-1].set(Rp2)
            self.Rp3s = self.Rp3s.at[int(iteration)-1].set(Rp3)
            self.Rp4s = self.Rp4s.at[int(iteration)-1].set(Rp4)
            self.Rp5s = self.Rp5s.at[int(iteration)-1].set(Rp5)
            self.Rp6s = self.Rp6s.at[int(iteration)-1].set(Rp6)
            self.Rd1s = self.Rd1s.at[int(iteration)-1].set(Rd1)
            self.Rd2s = self.Rd2s.at[int(iteration)-1].set(Rd2)

            # Diminishing stepsize of SCA
            if kind==0:
                stepsize = (stepsize+jnp.log(iteration)**(c1))/(1.+ c2*jnp.sqrt(iteration))
            elif kind==1:
                stepsize = (stepsize+jnp.log(iteration)**(c1))/(1.+ c2*iteration)
            elif kind==2:
                stepsize=stepsize*(1-eps*stepsize)
            elif kind==3:
                stepsize = (stepsize+c1)/(1.+ c2*iteration)
            else:
                pass
            
            # Diminishing stepsize of ADMM
            if kind1==0:
                stepsize1 = (stepsize1+jnp.log(iteration)**(c3))/(1.+ c4*jnp.sqrt(iteration))
            elif kind1==1:
                stepsize1 = (stepsize1+jnp.log(iteration)**(c3))/(1.+ c4*iteration)
            elif kind1==2:
                stepsize1=stepsize1*(1-eps1*stepsize1)
            elif kind1==3:
                stepsize1 = (stepsize1+iteration**(c3))/(1.+ c4*iteration)
            else:
                pass

            rho=stepsize1
            sigma=stepsize1
            theta=stepsize1
            omega=stepsize1
            delta=stepsize1
            tau_prime=1./theta

            if iteration%step==0:
                # Update the conditions for primal and dual residuals
                # primal
                cond1p = check_feasibility_p(Rp1,Max1p)
                cond2p = check_feasibility_p(Rp2,Max2p)
                cond3p = check_feasibility_p(Rp3,Max3p)
                cond4p = check_feasibility_p(Rp4,Max4p)
                cond5p = check_feasibility_p(Rp5,Max5p)
                cond6p = check_feasibility_p(Rp6,Max6p)
                # dual
                cond1d = check_feasibility_d(Rd1,Max1d)
                cond2d = check_feasibility_d(Rd2,Max2d)
                
                print("\n\nIteration: {}".format(iteration))
                print("objective: {}".format(obj))
                print("Residuals for primal feasibility: U {}; X {}; FFT(F) {}; V {}; L {}; W {}".format(jnp.round(Rp1,3), jnp.round(Rp2,3), jnp.round(Rp3,3), jnp.round(Rp4,3), jnp.round(Rp5,3), jnp.round(Rp6,3)))
                print("Residual for dual feasibility: sp F {}vs{}; sp P {}vs{}".format(jnp.round(Rd1,3), jnp.round(dims*told_abs +  told_rel*Max1d,3), jnp.round(Rd2,3),jnp.round(dims*told_abs +  told_rel*Max2d,3)))
                print("Primal feasibility conditions: {}, {}, {}, {}, {}, {}".format(cond1p, cond2p, cond3p, cond4p, cond5p, cond6p))
                print("Dual feasibility conditions: {}, {}".format(cond1d, cond2d))
                print("SCA stepsize value: {}".format(stepsize))
                print("ADMM stepsize value: {}".format(stepsize1))
                print('\n\n')

        # Conditions primal
        cond1p = check_feasibility_p(Rp1,Max1p)
        cond2p = check_feasibility_p(Rp2,Max2p)
        cond3p = check_feasibility_p(Rp3,Max3p)
        cond4p = check_feasibility_p(Rp4,Max4p)
        cond5p = check_feasibility_p(Rp5,Max5p)
        cond6p = check_feasibility_p(Rp6,Max6p)
        # Conditions dual
        cond1d = check_feasibility_d(Rd1,Max1d)
        cond2d = check_feasibility_d(Rd2,Max2d)

        if not all_feasibility(Rp1, Rp2, Rp3, Rp4, Rp5, Rp6, Rd1, Rd2, Max1p, Max2p, Max3p, Max4p, Max5p, Max6p, Max1d, Max2d, iteration) and iteration<max_iter:
            print("################# Optimisation successfully completed #################")
            print("\n\nLast iteration: {}".format(iteration))
            print("objective: {}".format(obj))
            print("Residuals for primal feasibility: U {}; X {}; FFT(F) {}; V {}; L {}; W {}".format(jnp.round(Rp1,3), jnp.round(Rp2,3), jnp.round(Rp3,3), jnp.round(Rp4,3), jnp.round(Rp5,3), jnp.round(Rp6,3)))
            print("Residual for dual feasibility: sp F {}vs{}; sp P {}vs{}".format(jnp.round(Rd1,3), jnp.round(dims*told_abs +  told_rel*Max1d,3), jnp.round(Rd2,3),jnp.round(dims*told_abs +  told_rel*Max2d,3)))
            print("Primal feasibility conditions: {}, {}, {}, {}, {}, {}".format(cond1p, cond2p, cond3p, cond4p, cond5p, cond6p))
            print("Dual feasibility conditions: {}, {}".format(cond1d, cond2d))
            print("SCA stepsize value: {}".format(stepsize))
            print("ADMM stepsize value: {}".format(stepsize1))
            print('\n\n')
        else:
            print("################# Maximum number of iterations reached #################")
            print("\n\nLast iteration: {}".format(iteration))
            print("objective: {}".format(obj))
            print("Residuals for primal feasibility: U {}; X {}; FFT(F) {}; V {}; L {}; W {}".format(jnp.round(Rp1,3), jnp.round(Rp2,3), jnp.round(Rp3,3), jnp.round(Rp4,3), jnp.round(Rp5,3), jnp.round(Rp6,3)))
            print("Residual for dual feasibility: sp F {}vs{}; sp P {}vs{}".format(jnp.round(Rd1,3), jnp.round(dims*told_abs +  told_rel*Max1d,3), jnp.round(Rd2,3),jnp.round(dims*told_abs +  told_rel*Max2d,3)))
            print("Primal feasibility conditions: {}, {}, {}, {}, {}, {}".format(cond1p, cond2p, cond3p, cond4p, cond5p, cond6p))
            print("Dual feasibility conditions: {}, {}".format(cond1d, cond2d))
            print("SCA stepsize value: {}".format(stepsize))
            print("ADMM stepsize value: {}".format(stepsize1))
            print('\n\n')

        return {'F':F,'P':P,'vecF':vF,'vecP':vP,'vecU':vU,'vecX':vX,'vecV':vV,'vecL':vL,'vecW':vW,'alpha': alpha,'delta': mdelta,'beta': beta, 'mu': mu, 'omega': momega, 'phi': phi,'objective value': obj, 'iterations':iteration, 'cond1p':cond1p, 'cond2p':cond2p, 'cond3p':cond3p, 'cond4p':cond4p, 'cond4p':cond5p, 'cond4p':cond6p, 'cond1d':cond1d,'cond2d':cond2d}
    