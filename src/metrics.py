import jax.numpy as jnp
from jax.numpy import array, append, insert, zeros
from jax.numpy.linalg import norm
from src.utils import hmean

def blocks_skeleton(X, K=None):
    """
    Returns the boolean tensor of size (N,N,len(K)+1). For each slice (N,N), an entry is equal to 1 if the l2-norm 
    of the corresponding fiber is different from zero at that block of frequencies.

    INPUT
    =====
    X: jnp.ndarray, tensor of size (N,N,M), where M is supposed to be the number of frequencies in [0,0.5].
    K: jnp.ndarray of type jnp.int, splitting points for frequencies to define the blocks.
        Example: M=50, K=[M//10, M//4] --> The groups will be {[0, M//10); [M//10, M//4); [M//4, M]} 

    OUTPUT
    ======
    Y: jnp.ndarray, boolean tensor of shape (N,N,len(K)+1) 
    """

    assert isinstance(X, jnp.ndarray) and X.dtype in (jnp.complex64, jnp.complex128) and X.shape[0]==X.shape[1], 'X must be a tensor size (N,N,M).'
    
    N,_,M=X.shape
    _dtype = X.dtype
    
    if K is None:
            K = array([0,M])
            Y=zeros((N,N,1), dtype=_dtype)
    else:
        assert isinstance(K, jnp.ndarray) and K.dtype in (jnp.int32, jnp.int64) and K.ndim==1, 'indexes of frequencies must be a one dim array of integers.'
        assert ((0<K) & (K<M)).sum()==len(K), 'Splitting point must be in [1,2,...,M-1].' 
        Y=zeros((N,N,len(K)+1), dtype=_dtype)
        K = append(insert(K,0,0), M)

    for g in range(len(K)-1):
        Y=Y.at[...,g].set(norm(X[...,K[g]:K[g+1]], axis=-1, ord=2))

    return Y

def count_accuracy(TX_true, TX_est, K=None, tau=1.e-3, already_blocks=False):
    """Compute various accuracy metrics. 
    We denote with 'd' the number of nodes; with 'M' that of frequencies, and with 'K' the splitting points for frequency bands.

    true positive = predicted association exists in condition 
    false positive = predicted association does not exist in condition
    false negative = missing association exists in condition

    INPUTS
    ======
        TX_true: jnp.ndarray, size [d, d, M or len(K)+1]; ground truth graph
        TX_est: (jnp.ndarray): [d, d, M or len(K)+1] learned graph.
        K: jnp.ndarray of type jnp.int, splitting points for frequencies to define the blocks (Optional).
            Example: M=50, K=[M//10, M//4] --> The groups will be {[0, M//10); [M//10, M//4); [M//4, M]} 
        tau: float, hard-thresholding level (Optional).
        already_blocks: bool, True if you pass the result of 'blocks_skeleton' function, False otherwise.

    OUTPUT
    ======
    Dict containing the values for the following metrics:
        For each block of frequencies (dim -1 of TX_true/est)):
            fdr|(1-precision): false positive / prediction positive
            tpr|recall: true positive / condition positive
            fpr: false positive / condition negative
            f1: 2*((1-fdr)*tpr)/((1-fdr)+tpr)
            nnz: prediction positive
            true_nnz: number of edges in B_true
            ae_nnz: |true_nnz-nnz|
            hamming: false positive + false negative
        Aggregated
            f1_mean: mean f1 along layers
            fpr_mean: mean false positive rate along layers
            hamming_kpcg: sum of layers' hamming distance
    """
    
    assert isinstance(TX_true, jnp.ndarray) and TX_true.dtype in (jnp.complex64, jnp.complex128) and TX_true.shape[0]==TX_true.shape[1], 'TX must be a tensor size (N,N,K).'
    assert isinstance(TX_est, jnp.ndarray) and TX_est.dtype in (jnp.complex64, jnp.complex128) and TX_est.shape[0]==TX_est.shape[1], 'TX must be a tensor size (N,N,K).'
    assert TX_true.shape==TX_est.shape, 'Ground truth and estimate tensors must have the same size.'
    
    if not already_blocks:
        B_est = blocks_skeleton(TX_est,K)
        B_est = jnp.where(B_est>tau, B_est, 0.)
        B_true = blocks_skeleton(TX_true,K)
    else:
        B_est=jnp.where(TX_est>tau, TX_est, 0.)
        B_true=jnp.where(TX_true>tau, TX_true, 0.)

    d,_,g = B_true.shape
    finit_arr = zeros(g, dtype=jnp.float32)
    iinit_arr = zeros(g, dtype=jnp.int16)

    metrics = {'fdr|(1-precision)': finit_arr, 'tpr|recall': finit_arr, 'fpr': finit_arr, 'f1':finit_arr,
            'nnz': iinit_arr, 'true_nnz':iinit_arr, 'ae_nnz':iinit_arr, 'hamming':iinit_arr}

    for bg in range(g):
        # linear index of nonzeros, only lower triangular part
        pred = jnp.flatnonzero(jnp.tril(B_est[...,bg],k=-1))
        cond = jnp.flatnonzero(jnp.tril(B_true[...,bg],k=-1))

        # true pos
        true_pos = jnp.intersect1d(pred, cond, assume_unique=False)
        # false pos
        false_pos = jnp.setdiff1d(pred, cond, assume_unique=False)
        # false neg
        false_neg = jnp.setdiff1d(cond, pred, assume_unique=False)
        
        # extra + missing edges
        hamming = len(false_pos) + len(false_neg)
        metrics['hamming'] = metrics['hamming'].at[bg].set(hamming)

        # compute ratios
        pred_size = len(pred)
        true_nnz = len(cond)
        cond_neg_size = 0.5 * d * (d - 1) - true_nnz
        fdr = len(false_pos)*1. / max(pred_size, 1)
        tpr = len(true_pos)*1. / max(true_nnz, 1)
        fpr = len(false_pos)*1. / max(cond_neg_size, 1)
        
        f1 = hmean(array([1.-fdr, tpr], dtype=jnp.float32))
        
        metrics['fdr|(1-precision)']=metrics['fdr|(1-precision)'].at[bg].set(fdr)
        metrics['tpr|recall']=metrics['tpr|recall'].at[bg].set(tpr)
        metrics['fpr']=metrics['fpr'].at[bg].set(fpr)
        metrics['f1']=metrics['f1'].at[bg].set(f1)
        metrics['nnz']=metrics['nnz'].at[bg].set(pred_size)
        metrics['true_nnz']=metrics['true_nnz'].at[bg].set(true_nnz)
        metrics['ae_nnz']=metrics['ae_nnz'].at[bg].set(jnp.abs(pred_size-true_nnz))
        
    metrics['f1_mean']=metrics['f1'].mean()
    metrics['fpr_mean']=metrics['fpr'].mean()
    metrics['hamming_kpcg']=metrics['hamming'].sum()

    return metrics
