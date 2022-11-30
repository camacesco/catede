#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    (in development)
    Copyright (C) November 2022 Francesco Camaglia, LPENS 
'''

import numpy as np
from scipy import optimize 
from .beta_func_multivar import *

MAX_ITER = 500
TOL = 1e-16
BOUND_DIR = (1e-10, 1e3)
METHOD='L-BFGS-B'
INIT_GUESS = 1.
CUTOFFRATIO = 10
USE_JAC_OPT = False

########################################
#  CONCENTRATION PARAMETER LIKELIHOOD  #
########################################

# LIKELIHOOD

def log_alphaLikelihood( var, *args ) :
    '''log-likelihood for concentration parameter (log of multi-variate Beta).'''
    alpha = var
    compExp = args[0]
    K = compExp.K
    x = compExp.nn + alpha
    X = compExp.N + K * alpha

    # posterior contribution 
    def sumGens( x ) : yield LogGmm( x )
    output = compExp._ffsum( sumGens(x), dim=1 )  - LogGmm( X ) 
    # Dirichelet prior normalization contribution
    output += LogGmm( K*alpha ) - K * LogGmm( alpha )                  

    return output
    
def log_alphaLikelihood_jac( var, *args ) :
    '''1st derivative of the log-likelihood for the concentration parameter.'''
    alpha = var
    compExp = args[0]
    K = compExp.K
    x = compExp.nn + alpha
    X = compExp.N + K * alpha

    # posterior contribution 
    def sumGens( x ) : yield diGmm( x )
    output = compExp._ffsum( sumGens(x), dim=1 ) - K * diGmm( X ) 
    # Dirichelet prior normalization contribution
    output += K * diGmm( K*alpha ) - K * diGmm( alpha )                  

    return output

def log_alphaLikelihood_hess( var, *args ) :
    '''2nd derivative of the log-likelihood for the concentration parameter.'''
    alpha = var
    compExp = args[0]
    K = compExp.K
    x = compExp.nn + alpha
    X = compExp.N + K * alpha

    # posterior contribution 
    def sumGens( x ) : yield triGmm( x )
    output = compExp._ffsum( sumGens(x), dim=1 ) - np.power(K,2) * triGmm( X ) 
    # Dirichelet prior normalization contribution
    output += np.power(K,2) * triGmm( K*alpha ) - K * triGmm( alpha )                  

    return output

# METAPRIOR : ENTROPY

def log_entropyMetaprior( var, *args ) :
    '''log of NSB entropy metaprior'''
    alpha = var
    compExp = args[0]
    K = compExp.K
    output = np.log( K * triGmm(K*alpha+1) - triGmm(alpha+1) )
    return output

def log_entropyMetaprior_jac( var, *args ) :
    '''1st derivative of log of NSB entropy metaprior'''
    alpha = var
    compExp = args[0]
    K = compExp.K
    output = np.power(K,2) * quadriGmm(K*alpha+1) - quadriGmm(alpha+1)
    output /= log_entropyMetaprior( alpha, compExp )
    return output

def log_entropyMetaprior_hess( var, *args ) :
    '''2nd derivative of log of NSB entropy metaprior'''
    alpha = var
    compExp = args[0]
    K = compExp.K
    output = np.power(K,3) * polygamma(3,K*alpha+1) - polygamma(3,alpha+1)
    output /= log_entropyMetaprior( alpha, compExp )
    output -= np.power(log_entropyMetaprior_jac(alpha,compExp),2)
    return output

# METAPRIOR : CROSSENTROPY

def log_crossentropyMetaprior( var, *args ) :
    '''log of NSB crossentropy metaprior'''
    beta = var
    compExp = args[0]
    K = compExp.K
    return np.log( triGmm(beta) - K * triGmm(K*beta) )

def log_crossentropyMetaprior_jac( var, *args ) :
    '''1st derivative of log of NSB crossentropy metaprior'''
    beta = var
    compExp = args[0]
    K = compExp.K
    output = quadriGmm(beta) - np.power(K,2) * quadriGmm(K*beta) 
    output /= log_crossentropyMetaprior(beta, compExp)
    return output # FIXME : it seems wrong...

def log_crossentropyMetaprior_hess( var, *args ) :
    '''2nd derivative of log of NSB crossentropy metaprior'''
    beta = var
    compExp = args[0]
    K = compExp.K
    output = polygamma(3,beta) - np.power(K,3) * polygamma(3,K*beta)
    output /= log_crossentropyMetaprior(beta, compExp)
    output -= np.power(log_crossentropyMetaprior_jac(beta, compExp),2)
    return output

# METAPRIOR : DIVERGENCE

def MetaPrior_DKL( A, B, K, cutoff_ratio ) :
    '''(auxiliary) Phi term in DKL Metaprior.'''

    D = B - A

    # choice of the prior
    # FIXME : implement different strategies
    rho_D = 1. # uniform

    # function by cases 
    if D >= cutoff_ratio * np.log(K) : # cutoff
        output = 0.
    elif D >= np.log(K) : # uniform
        output = rho_D / np.log(K)
    else :
        output = rho_D / D 
    return output

def log_divergenceMetaprior( var, *args ) :
    '''logarithm of Phi term in DKL Metaprior.'''
    alpha, beta = var
    compDiv, cutoff_ratio = args
    K = compDiv.K

    A = prior_entropy_vs_alpha_( alpha, K )
    B = prior_crossentropy_vs_beta_( beta, K )

    return MetaPrior_DKL( A, B, K, cutoff_ratio )

def log_divergenceMetaprior_unif_jac( var, *args ) :
    '''jacobian of logarithm of Phi term in DKL Metaprior.'''
    alpha, beta = var
    compDiv, cutoff_ratio = args
    K = compDiv.K

    A = prior_entropy_vs_alpha_( alpha, K )
    B = prior_crossentropy_vs_beta_( beta, K )
    D = B - A

    # function by cases 
    output = np.zeros(2)
    if D >= cutoff_ratio * np.log(K) : # cutoff
        pass
    elif D >= np.log(K) : # uniform case
        pass
    else :
        output[0] = ( triGmm(alpha+1) - K * triGmm(K*alpha+1) ) / D 
        output[1] = ( K * triGmm(K*beta) - triGmm(beta) ) / D 
    return output

def log_divergenceMetaprior_unif_hess( var, *args ) :
    '''hessian of logarithm of Phi term in DKL Metaprior.'''
    alpha, beta = var
    compDiv, cutoff_ratio = args
    K = compDiv.K
    
    A = prior_entropy_vs_alpha_( alpha, K )
    B = prior_crossentropy_vs_beta_( beta, K )
    D = B - A

    # function by cases 
    output = np.zeros([2,2])
    if D >= cutoff_ratio * np.log(K) : # cutoff
        pass
    elif D >= np.log(K) : # uniform case
        pass
    else :
        # FIXME :
        output[0,0] = np.power( ( triGmm(alpha+1) - K * triGmm(K*alpha+1) ) / D, 2 )
        output[0,0] += ( quadriGmm(alpha+1) - np.power(K,2) * quadriGmm(K*alpha+1) ) / D
        output[0,1] = (triGmm(alpha+1) - K * triGmm(K*alpha+1)) * (triGmm(beta) - K * triGmm(K*beta)) 
        output[0,1] /= np.power(D,2)
        output[1,0] = output[0,1]
        output[1,1] = np.power(( K * triGmm(K*beta) - triGmm(beta) ) / D, 2)
        output[1,1] += ( np.power(K,2) * quadriGmm(K*beta) - quadriGmm(beta) ) / D
    return output

# <<<<<<<<<<<<<<<<<<<<<<
#  MAXIMUM LIKELIHOOD #
# >>>>>>>>>>>>>>>>>>>>>> 

def myMinimizer( myfunc, var, args, jac=None ) :
    # FIXME : extreme cases ?
    results = optimize.minimize(
        myfunc,
        x0=var, args=args,
        jac=jac,
        method=METHOD, bounds=(BOUND_DIR,)*len(var), 
        options={'maxiter': MAX_ITER}, tol=TOL
        )
    return results.x

def optimal_dirichlet_param_( compExp ) :
    '''Return Dirichlet parameter which optimizes entropy posterior (~).''' 
    def myfunc( var, *args ) :
        LogLike = log_alphaLikelihood(var, *args)
        return - LogLike
    def myjac(var, *args) :
        jac_LogLike = log_alphaLikelihood_jac(var, *args)
        return - jac_LogLike
    if USE_JAC_OPT is False : myjac = None
    return myMinimizer( myfunc, [INIT_GUESS], (compExp,), jac=myjac )

def optimal_entropy_param_( compExp ) :
    '''Return NSB parameter Shannon entropy posterior times meta-prior.''' 
    def myfunc(var, *args) :
        LogLike = log_alphaLikelihood(var, *args) + log_entropyMetaprior(var, *args)
        return - LogLike
    def myjac(var, *args) :
        jac_LogLike = log_alphaLikelihood_jac(var, *args) + log_entropyMetaprior_jac(var, *args)
        return - jac_LogLike
    if USE_JAC_OPT is False : myjac = None
    return myMinimizer( myfunc, [INIT_GUESS], (compExp,), jac=myjac )

def optimal_crossentropy_param_( compExp ) :
    '''Return NSB parameter crossentropy posterior times meta-prior. (obsolete)''' 
    def myfunc(var, *args) :
        LogLike = log_alphaLikelihood(var, *args) + log_crossentropyMetaprior(var, *args)
        return - LogLike
    def myjac(var, *args) :
        jac_LogLike = log_alphaLikelihood_jac(var, *args) + log_crossentropyMetaprior_jac(var, *args)
        return - jac_LogLike
    if USE_JAC_OPT is False : myjac = None
    return myMinimizer( myfunc, [INIT_GUESS], (compExp,), jac=myjac )

def optimal_divergence_params_( compDiv ) :
    '''Return NSB parameter Shannon entropy posterior times meta-prior.''' 
    def myfunc(var, *args) :
        LogLike = log_alphaLikelihood(var[0], args[0].compact_1)
        LogLike += log_entropyMetaprior(var[0], args[0].compact_1)
        LogLike += log_alphaLikelihood(var[1], args[0].compact_2) 
        LogLike += log_crossentropyMetaprior(var[1], args[0].compact_2)
        LogLike += log_divergenceMetaprior(var, *args)
        return - LogLike
    def myjac(var, *args) :
        jac_LogLike = np.zeros(2)
        jac_LogLike[0] = log_alphaLikelihood_jac(var[0], args[0].compact_1)
        jac_LogLike[0] += log_entropyMetaprior_jac(var[0], args[0].compact_1)
        jac_LogLike[1] = log_alphaLikelihood_jac(var[1], args[0].compact_2) 
        jac_LogLike[1] += log_crossentropyMetaprior_jac(var[1], args[0].compact_2)
        jac_LogLike += log_divergenceMetaprior_unif_jac(var, *args)
        return - jac_LogLike
    if USE_JAC_OPT is False : myjac = None
    return myMinimizer( myfunc, [INIT_GUESS,INIT_GUESS], (compDiv,CUTOFFRATIO,), jac=myjac )


# -----------------------------------

# IN DEVELOPMENT
# SIMPSON, EQUAL PRIOR

# -----------------------------------

def optimal_simpson_param_( compExp ) :
    '''Return NSB parameter Simpson index posterior times meta-prior.''' 
    def myfunc(var, *args) :
        alpha = var
        compExp = args[0]
        K = compExp.K
        metaprior = (K-1) / np.power( K*alpha + 1, 2 )
        Like = log_alphaLikelihood(var, *args) + np.log(metaprior)
        return - Like
    return myMinimizer( myfunc, [INIT_GUESS], (compExp,) )

def optimal_dirichlet_EP_param_( compDiv ) :
    '''Return Dirchlet parameter which optimizes divergence posterior alpha=beta (~).''' 
    def myfunc( var, *args ) :
        compExp_1, compExp_2, = args[0].compact_1,  args[0].compact_2
        Like = log_alphaLikelihood(var, (compExp_1,)) + log_alphaLikelihood(var, (compExp_2,))
        return - Like
    return myMinimizer( myfunc, [INIT_GUESS], (compDiv,) )

def optimal_divergence_EP_param_( compDiv ) :
    '''Return Dirchlet parameter which optimizes divergence posterior alpha=beta (~).''' 
    def myfunc( var, *args ) :
        compExp_1, compExp_2, = args[0].compact_1,  args[0].compact_2
        K = args[0].K
        logMetaprior = np.log( ( 1. - 1. / K ) / var**2 )
        Like = log_alphaLikelihood(var, (compExp_1,)) + log_alphaLikelihood(var, (compExp_2,)) + logMetaprior
        return - Like
    return myMinimizer( myfunc, [INIT_GUESS], (compDiv,) )
