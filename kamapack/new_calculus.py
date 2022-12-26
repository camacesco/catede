#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    (in development)
    Copyright (C) December 2022 Francesco Camaglia, LPENS 
'''

import numpy as np
from scipy import optimize 
from kamapack.beta_func_multivar import *

MAX_ITER = 8000
TOL = 1.0e-16
BOUND_DIR = (1.0e-5, 1.0e3)
METHOD='L-BFGS-B'
INIT_GUESS = 1.0
CUTOFFRATIO = 5
# Warning : CUTOFFRATIO > 1 
USE_JAC_OPT = True
NUMERICAL_ZERO = 1.0e-14
NUMERICAL_INFTY = 1.0e12

########################################
#  CONCENTRATION PARAMETER LIKELIHOOD  #
########################################

class Likelihood( ) :
    def __init__( self, var, *args ) : 
        self.a = var
        compExp = args[0]
        self.K = compExp.K
        self.x = compExp.nn + self.a
        self.X = compExp.N + self.K * self.a
        self._ffsum = compExp._ffsum
    def log( self ) :
        def sumGens( x ) : yield LogGmm( x )
        output = self._ffsum( sumGens(self.x), dim=1 ) - LogGmm( self.X ) 
        output += LogGmm( self.K*self.a ) - self.K * LogGmm( self.a )                  
        return output
    def log_jac( self ) :
        def sumGens( x ) : yield diGmm( x )
        output = self._ffsum( sumGens(self.x), dim=1 ) - self.K * diGmm( self.X ) 
        output += self.K * diGmm( self.K*self.a ) - self.K * diGmm( self.a )                  
        return output
    def log_hess( self ) :
        def sumGens( x ) : yield triGmm( x )
        output = self._ffsum( sumGens(self.x), dim=1 ) - np.power(self.K,2) * triGmm( self.X ) 
        output += np.power(self.K,2) * triGmm( self.K*self.a ) - self.K * triGmm( self.a )                  
        return output

class DirEntr( ) :
    def __init__(self, var, *args) :
        self.a = np.array(var)
        self.K = args[0] 
    def aPrioriExp( self ) :
        return prior_entropy_vs_alpha_(self.a,self.K)
    def Metapr( self ) :
        return self.K * triGmm(self.K*self.a+1) - triGmm(self.a+1)
    def Metapr_jac( self ) :
        return np.power(self.K,2) * quadriGmm(self.K*self.a+1) - quadriGmm(self.a+1)
    def Metapr_hess( self ) :
        return np.power(self.K,3) * polygamma(3,self.K*self.a+1) - polygamma(3,self.a+1)
    def logMetapr( self ) :
        return np.log(self.Metapr())
    def logMetapr_jac( self ) :
        return self.Metapr_jac() / self.Metapr()
    def logMetapr_hess( self ) :
        return self.Metapr_hess() / self.Metapr() - np.power(self.logMetapr_jac(),2)

class DirCrossEntr( ) :
    def __init__(self, var, *args) :
        self.b = np.array(var)
        self.K = args[0] 
    def aPrioriExp( self ) :
        return prior_crossentropy_vs_beta_(self.b,self.K)
    def Metapr( self ) :
        return triGmm(self.b) - self.K * triGmm(self.K*self.b)
    def Metapr_jac( self ) :
        return quadriGmm(self.b) - np.power(self.K,2) * quadriGmm(self.K*self.b)
    def Metapr_hess( self ) :
        return polygamma(3,self.b) - np.power(self.K,3) * polygamma(3,self.K*self.b)
    def logMetapr( self ) :
        return np.log(self.Metapr())
    def logMetapr_jac( self ) :
        return self.Metapr_jac() / self.Metapr()
    def logMetapr_hess( self ) :
        return self.Metapr_hess() / self.Metapr() - np.power(self.logMetapr_jac(),2)

class DirKLdiv( ) :
    def __init__(self, var, *args ) :
        # note : get around 0-dimensional numpy scalar arrays
        self.a = np.array(var[0]).reshape(-1)
        self.b = np.array(var[1]).reshape(-1)
        self.K, self.choice = args
        self.A = prior_entropy_vs_alpha_( self.a, self.K )
        self.B = prior_crossentropy_vs_beta_( self.b, self.K )
        self.D = self.B - self.A  
        if self.choice not in ["uniform", "log-uniform", "scaled"] :
            raise IOError(f"unrecognized choice `{self.choice}`.")
    def aPrioriExp( self ) :
        return self.D
    def Metapr( self ) :
        output = np.ones( shape = (self.D.size,) )
        # function by cases
        mask = self.D < np.log(self.K)
        output[ mask ] /= self.D[ mask ]
        output[ ~mask ] /= np.log(self.K)
        # choice for rho(D)
        if self.choice in ["uniform"] :
            output[ self.D >= CUTOFFRATIO * np.log(self.K) ] = NUMERICAL_ZERO
            # NOTE : no point in adding the normalization 1. / (CUTOFFRATIO * np.log(self.K))
        elif self.choice in ["log-uniform"] :
            output /= self.D
        elif self.choice in ["scaled"] :
            output *= np.exp( - self.D / self.A )
        return output
    def logMetapr( self ) :
        if self.choice in ["uniform"] :
            output = np.zeros( shape = (self.D.size,) )
            mask = self.D < np.log(self.K) 
            output[ mask ] = - np.log(self.D[mask])
            output[ ~mask ] = - np.log(np.log(self.K))
            output[ self.D >= CUTOFFRATIO * np.log(self.K) ] = - NUMERICAL_INFTY 
            # NOTE CUTOFFRATIO > 1.
        else :
            output = np.log(self.Metapr())
        return output
    def logMetapr_jac(self) :
        output = np.zeros( shape = (self.D.size, 2,) )
        # term function by cases 
        mask = self.D < np.log(self.K)
        output[mask,0] = DirEntr(self.a[mask], self.K).Metapr() / self.D[mask]
        output[mask,1] = DirCrossEntr(self.b[mask], self.K).Metapr() / self.D[mask]
        if self.choice in ["uniform"] :
            output[ ~mask,: ] = NUMERICAL_ZERO
            output[ self.D >= CUTOFFRATIO * np.log(self.K),: ] = - NUMERICAL_INFTY
        elif self.choice in ["log-uniform"] :
            output[:,0] += DirEntr(self.a, self.K).Metapr() / self.D
            output[:,1] += DirCrossEntr(self.b, self.K).Metapr() / self.D
        elif self.choice in ["scaled"] :
            output[:,0] += self.B * DirEntr(self.a, self.K).Metapr() / np.power(self.A,2)
            output[:,1] += (self.A * DirCrossEntr(self.b, self.K).Metapr() + self.D * DirEntr(self.a, self.K).Metapr()) / np.power(self.A,2)
        return output
    def logMetapr_hess( self ) :
        output = np.zeros( shape = (self.D.size, 2, 2,) )
        # function by cases 
        m = self.D < np.log(self.K)
        output[m,0,0] = DirEntr(self.a[m], self.K).Metapr_jac()/self.D[m] + np.power(DirEntr(self.a[m], self.K).Metapr() / self.D[m],2)
        output[m,0,1] = DirEntr(self.a[m], self.K).Metapr() * DirCrossEntr(self.b[m], self.K).Metapr() / np.power(self.D[m],2)
        output[m,1,0] = output[m,0,1]
        output[m,1,1] = DirCrossEntr(self.b[m], self.K).Metapr_jac()/self.D[m] + np.power(DirCrossEntr(self.b[m], self.K).Metapr()/self.D[m],2)
        if self.choice in ["uniform"] :
            output[ ~m,:,: ] = NUMERICAL_ZERO
            output[ self.D >= CUTOFFRATIO * np.log(self.K),:,: ] = - NUMERICAL_INFTY
        elif self.choice in ["log-uniform"] :
            output[:,0,0] += DirEntr(self.a, self.K).Metapr_jac()/self.D + np.power(DirEntr(self.a, self.K).Metapr()/self.D,2)
            output[:,0,1] += DirEntr(self.a, self.K).Metapr() * DirCrossEntr(self.b, self.K).Metapr() / np.power(self.D,2)
            output[:,1,0] += DirEntr(self.a, self.K).Metapr() * DirCrossEntr(self.b, self.K).Metapr() / np.power(self.D,2)
            output[:,1,1] += DirCrossEntr(self.b, self.K).Metapr_jac()/self.D + np.power(DirCrossEntr(self.b, self.K).Metapr()/self.D,2)
        elif self.choice in ["scaled"] :
            raise IOError("Yet to be coded... Sorry...")
        return output

# <<<<<<<<<<<<<<<<<<<<<<
#  MAXIMUM LIKELIHOOD #
# >>>>>>>>>>>>>>>>>>>>>> 

def myMinimizer( myfunc, var, args, jac=None ) :
    '''.'''
    
    if USE_JAC_OPT is False : jac = None
    results = optimize.minimize(
        myfunc,
        x0=var, args=args,
        jac=jac,
        method=METHOD, bounds=(BOUND_DIR,)*len(var), 
        options={'maxiter': MAX_ITER}, tol=TOL
        )
    # FIXME : warning for extreme cases
    return results.x

def optimal_dirichlet_param_( compExp ) :
    '''.'''
    def myfunc( var, *args ) :
        return - Likelihood(var, *args).log()
    def myjac(var, *args) :
        return - Likelihood(var, *args).log_jac()
    return myMinimizer( myfunc, [INIT_GUESS], (compExp,), jac=myjac )

def optimal_entropy_param_( compExp ) :
    '''.'''
    def myfunc(var, *args) :
        return - ( Likelihood(var, *args).log() + DirEntr(var, args[0].K).logMetapr() )
    def myjac(var, *args) :
        return - ( Likelihood(var, *args).log_jac() + DirEntr(var, args[0].K).logMetapr_jac() )
    return myMinimizer( myfunc, [INIT_GUESS], (compExp,), jac=myjac )

def optimal_crossentropy_param_( compExp ) :
    '''(obsolete)''' 
    def myfunc(var, *args) :
        return - ( Likelihood(var, *args).log() + DirCrossEntr(var, args[0].K).logMetapr() )
    def myjac(var, *args) :
        return - ( Likelihood(var, *args).log_jac() + DirCrossEntr(var, args[0].K).logMetapr_jac() )
    return myMinimizer( myfunc, [INIT_GUESS], (compExp,), jac=myjac )

def optimal_divergence_params_( compDiv, choice="uniform" ) :
    '''.'''
    def myfunc(var, *args) :
        LogLike = DirKLdiv(var, args[0].K, args[1]).logMetapr()
        LogLike += Likelihood(var[0], args[0].compact_1).log() + DirEntr(var[0], args[0].K).logMetapr()
        LogLike += Likelihood(var[1], args[0].compact_2).log() + DirCrossEntr(var[1], args[0].K).logMetapr()
        return - LogLike
    def myjac(var, *args) :
        jac_LogLike = DirKLdiv(var, args[0].K, args[1]).logMetapr_jac()
        jac_LogLike[:,0] += Likelihood(var[0], args[0].compact_1).log_jac() + DirEntr(var[0], args[0].K).logMetapr_jac()
        jac_LogLike[:,1] += Likelihood(var[1], args[0].compact_2).log_jac() + DirCrossEntr(var[1], args[0].K).logMetapr_jac()
        return - jac_LogLike
    return myMinimizer( myfunc, [INIT_GUESS,INIT_GUESS], (compDiv,choice), jac=myjac )


# -----------------------------------

# IN DEVELOPMENT
# SIMPSON, EQUAL PRIOR

# -----------------------------------

'''
def optimal_simpson_param_( compExp ) :
    def myfunc(var, *args) :
        alpha = var
        compExp = args[0]
        K = compExp.K
        metaprior = (K-1) / np.power( K*alpha + 1, 2 )
        Like = log_alphaLikelihood(var, *args) + np.log(metaprior)
        return - Like
    return myMinimizer( myfunc, [INIT_GUESS], (compExp,) )

def optimal_dirichlet_EP_param_( compDiv ) :
    def myfunc( var, *args ) :
        compExp_1, compExp_2, = args[0].compact_1,  args[0].compact_2
        Like = log_alphaLikelihood(var, (compExp_1,)) + log_alphaLikelihood(var, (compExp_2,))
        return - Like
    return myMinimizer( myfunc, [INIT_GUESS], (compDiv,) )

def optimal_divergence_EP_param_( compDiv ) :
    def myfunc( var, *args ) :
        compExp_1, compExp_2, = args[0].compact_1,  args[0].compact_2
        K = args[0].K
        logMetaprior = np.log( ( 1. - 1. / K ) / var**2 )
        Like = log_alphaLikelihood(var, (compExp_1,)) + log_alphaLikelihood(var, (compExp_2,)) + logMetaprior
        return - Like
    return myMinimizer( myfunc, [INIT_GUESS], (compDiv,) )
'''