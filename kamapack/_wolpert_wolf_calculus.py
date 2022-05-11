#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Wolpert Wolf Calculus (in development)
    Copyright (C) March 2022 Francesco Camaglia, LPENS 
'''

import numpy as np
from mpmath import mp
from scipy import optimize 
from scipy.special import loggamma, polygamma

##############
#  NOTATION  #
##############

def diGmm(x) :    
    '''Digamma function (polygamma of order 0).'''
    return polygamma(0, x)

def triGmm(x) :    
    '''Trigamma function (polygamma of order 1).'''
    return polygamma(1, x)

def quadriGmm(x) :    
    '''Quadrigamma function (polygamma of order 2).'''
    return polygamma(2, x)

def D_diGmm(x, y):
    '''Difference between digamma functions in `x` and `y`.'''
    return diGmm(x) - diGmm(y)  

def D_triGmm(x, y):
    '''Difference between trigamma functions in `x` and `y`.'''
    return triGmm(x) - triGmm(y)  

def D_quadriGmm(x, y):
    '''Difference between quadrigamma functions in `x` and `y`.'''
    return quadriGmm(x) - quadriGmm(y) 

def LogGmm( x ): 
    ''' alias '''
    return loggamma( x ).real   

################
#  POSTERIORS  #
################

def measureMu_( compACTexp,  alpha ) :
    '''Measure Mu term in the posterior estimators'''     

    return mp.exp( LogMu_(compACTexp, alpha) )

def LogMu_( compACTexp,  alpha ) :
    '''logarithm computation of Measure Mu term.'''
    N, K = compACTexp.N,  compACTexp.K
    nn = compACTexp.nn

    # log(mu) computation  :  
    # posterior contribution 
    output = compACTexp._ffsum( LogGmm(nn+alpha) ) - LogGmm( N + K*alpha ) 
    # Dirichelet prior normalization contribution
    output += LogGmm( K*alpha ) - K * LogGmm( alpha )                  

    return output
    
def integral_with_mu_( mu, func, x ) :
    ''' alias '''   # FIXME: is this really necessary this way?
    return np.trapz(np.multiply(mu, func), x=x)

# <<<<<<<<<<<<<<<<<<<<<<
#  SADDLE-POINT METHOD #
# >>>>>>>>>>>>>>>>>>>>>> 

def optimal_dirichlet_param_( compACTexp, upper=1e-4, lower=1e2 ) :
    '''Return Dirchlet parameter which optimizes entropy posterior (~).''' 
    # FIXME : extreme cases not covered

    def implicit_relation( x, y, compACTexp ):
        N, K = compACTexp.N,  compACTexp.K
        nn = compACTexp.nn

        tmp = K * diGmm(N+K*x) - K * diGmm(K*x) + K * diGmm(x) - compACTexp._ffsum(diGmm(nn+x))
        return tmp - y

    output = get_from_implicit_(implicit_relation, 0, upper, lower, compACTexp )
    return output

def optimal_entropy_param_( compACTexp, upper=1e-4, lower=1e2 ) :
    '''Return NSB parameter entropy posterior times meta-prior.''' 

    def implicit_relation( x, y, compACTexp ):
        N, K = compACTexp.N,  compACTexp.K
        nn = compACTexp.nn

        tmp = K * diGmm(N+K*x) - K * diGmm(K*x) + K * diGmm(x) - compACTexp._ffsum(diGmm(nn+x))
        tmp += ( K**2 * quadriGmm(K*x+1) - quadriGmm(x+1) ) / ( K * triGmm(K*x+1) - triGmm(x+1) )
        return tmp - y

    output = get_from_implicit_(implicit_relation, 0, upper, lower, compACTexp )
    return output

def optimal_crossentropy_param_( compACTexp, upper=1e-4, lower=1e2 ) :
    '''Return NSB parameter crossentropy posterior times meta-prior.''' 

    def implicit_relation( x, y, compACTexp ):
        N, K = compACTexp.N,  compACTexp.K
        nn = compACTexp.nn

        tmp = K * diGmm(N+K*x) - K * diGmm(K*x) + K * diGmm(x) - compACTexp._ffsum(diGmm(nn+x))
        tmp += ( K**2 * quadriGmm(K*x) - quadriGmm(x) ) / ( K * triGmm(K*x) - triGmm(x) )
        return tmp - y

    output = get_from_implicit_(implicit_relation, 0, upper, lower, compACTexp )
    return output

def optimal_ed_param_( compACTdiv, upper=1e-4, lower=1e2 ) :
    '''Return Dirchlet parameter which optimizes divergence posterior alpha=beta (~).''' 

    def implicit_relation( x, y, compACTdiv ):
        N_1, N_2, K  = compACTdiv.N_1, compACTdiv.N_2, compACTdiv.K
        nn_1, nn_2 = compACTdiv.nn_1, compACTdiv.nn_2

        tmp = K * diGmm(N_1+K*x) - K * diGmm(K*x) + K * diGmm(x) - compACTdiv._ffsum(diGmm(nn_1+x))
        tmp += K * diGmm(N_2+K*x) - K * diGmm(K*x) + K * diGmm(x) - compACTdiv._ffsum(diGmm(nn_2+x))

        return tmp - y

    output = get_from_implicit_(implicit_relation, 0, upper, lower, compACTdiv )
    return output


def get_from_implicit_( implicit_relation, y, lower, upper, *args,
                      maxiter=100, xtol=1.e-20 ):
    '''
    Find the root of the implicit relation for x in (0, infty):  
    >    `implicit relation` ( x, *args ) - `y` = 0
    It uses the Brent's algorithm for the root finder in the interval (lower, upper)
    '''   
    # FIXME try with wider lower, upper in case falure

    # NOTE : the implicit_relation must have opposite signs in lower and upper
    output = optimize.brentq( implicit_relation, lower, upper,
                             args=( y , *args ), xtol=xtol, maxiter=maxiter )
    return output
                                                  
def implicit_entropy_vs_alpha_( alpha, entropy, K ):
    ''' implicit relation to be inverted. '''
    return D_diGmm( K * alpha + 1, alpha + 1 ) - entropy

def implicit_crossentropy_vs_beta_( beta, crossentropy, K ):
    ''' implicit relation to be inverted. '''
    return D_diGmm( K * beta , beta ) - crossentropy

def MetaPrior_DKL( A, B, K, cutoff_ratio ) :
    '''Mixture of Dirichlet Prior Meta-Prior for DKL'''

    D = B - A

    # choice of the prior
    rho_D = 1. # uniform

    # function by cases 
    if D >= cutoff_ratio * np.log(K) : # cutoff
        return 0.
    elif D >= np.log(K) : # uniform
        return rho_D / np.log(K)
    else :
        return rho_D / D 

################
#  POSTERIORS  #
################

def Omega_( compACTexp, shift, a) :

    # loading parameters from Experiment Compact        
    N, nn, K = compACTexp.N, compACTexp.nn, compACTexp.K

    if shift == 1 :
        norm = N+K*a
        output = [ nn+a, norm ]
    elif shift == 2 :
        parallel = (nn+a+1) * (nn+a)
        perpendic = np.outer( nn+a , nn+a )
        norm = mp.fmul(N+K*a+1, N+K*a)
        output = [ [parallel, perpendic], norm ]
    elif shift == 3 :
        # f(i) : i==j, j==k
        parallel = (nn+a+2) * (nn+a+1) * (nn+a)
        # f(i,k) : j==k
        cross_1 = np.outer( nn+a, (nn+a+1) * (nn+a) )
        # f(i,j) : k==i
        cross_2 = np.outer( (nn+a+1) * (nn+a), nn+a )
        # f(i,k) : i==j
        cross_3 = cross_2
        # f(i,j,k) : i!=j, j!=k, k!=i
        perpendic = np.array([np.outer( nn+a, r ) for r in np.outer( nn+a, nn+a )])
        norm = mp.fmul( N+K*a+2, mp.fmul(N+K*a+1, N+K*a) )
        output = [ [parallel, cross_1, cross_2, cross_3, perpendic], norm ]
    else :
        raise IOError("FIXME: Developer Error in Omega.")

    return output

def Lambda_( compACTexp, order, shift, a) :
    ''' \frac{ \partial^{o} W }{\partial^{o} x_i } / ( W \Omega ) '''
    N, nn, K = compACTexp.N, compACTexp.nn, compACTexp.K
    if order == 1 :
        if shift == 0 :
            output = D_diGmm(nn+a, N+K*a)    
        elif shift == 1 :
            output = D_diGmm(nn+a+1, N+K*a+1)  
        elif shift == 2 :
            parallel = D_diGmm(nn+a+2, N+K*a+2)
            perpendic = np.tile( D_diGmm(nn+a+1, N+K*a+2), (len(nn),1) )
            output = [ parallel, perpendic ]
        elif shift == 3 :
            # f(i) : i==j, j==k
            parallel = D_diGmm(nn+a+3, N+K*a+3)
            # f(i,k) : j==k
            cross_1 = np.tile( D_diGmm(nn+a+1, N+K*a+3), (len(nn),1) )
            # f(i,j) : k==i
            cross_2 = np.tile( D_diGmm(nn+a+2, N+K*a+3), (len(nn),1) )
            # f(i,k) : i==j
            cross_3 = cross_2
            # f(i,j,k) : i!=j, j!=k, k!=i
            perpendic = np.tile( D_diGmm(nn+a+1, N+K*a+3), (len(nn), len(nn), 1))
            output = [ parallel, cross_1, cross_2, cross_3, perpendic ]
        else :
            raise IOError("FIXME: Developer Error in Lambda.")
    elif order == 2 :
        if shift == 0 : 
            parallel = np.power(D_diGmm(nn+a, N+K*a), 2) + D_triGmm(nn+a, N+K*a)
            perpendic = np.outer(D_diGmm(nn+a, N+K*a), D_diGmm(nn+a, N+K*a)) - triGmm(N+K*a)
        elif shift == 1 : 
            raise IOError("FIXME: Do you ever use this?.")
        elif shift == 2 :
            # i==j
            parallel = np.power(D_diGmm(nn+a+2, N+K*a+2), 2) + D_triGmm(nn+a+2, N+K*a+2)
            # i!=j
            perpendic = np.outer(D_diGmm(nn+a+1, N+K*a+2), D_diGmm(nn+a+1, N+K*a+2)) - triGmm(N+K*a+2)
        else :
            raise IOError("FIXME: Developer Error in Lambda.")
        output = [ parallel, perpendic ]
    elif order == 3 :
        if shift == 0 : 
            # FIXME : to be checked!
            Lambda_1_0 = Lambda_( compACTexp, 1, 0, a)
            Lambda_2_0 = Lambda_( compACTexp, 2, 0, a)
            der_Lambda_2_0 = der_Lambda_( compACTexp, 2, 0, 1, a)
            # f(i) : i==j, j==k
            parallel = Lambda_1_0 * Lambda_2_0[0] + der_Lambda_2_0[0]
            # f(i,k) : j==k
            cross1 = np.outer(Lambda_1_0, Lambda_2_0[0], )
            cross1 += der_Lambda_2_0[1]
            # f(i,j) : k==i
            cross2 = (Lambda_2_0[1] * np.tile(Lambda_1_0[:,None], (1,len(nn))))
            cross2 += der_Lambda_2_0[2]
            # f(i,k) : i==j
            cross3 = (Lambda_2_0[1] * np.tile(Lambda_1_0[:,None], (1,len(nn))))
            cross3 += der_Lambda_2_0[3]
            # f(i,j,k) : i!=j, j!=k, k!=i
            perpendic = np.outer(Lambda_1_0, Lambda_2_0[1]).reshape( (len(nn),)*3 )
            perpendic += der_Lambda_2_0[4]

            output = [parallel, cross1, cross2, cross3, perpendic]
        
        else :
            raise IOError("FIXME: Developer Error in Lambda.")   
    else :
        raise IOError("FIXME: Developer Error in Lambda.")
    return output

def der_Lambda_( compACTexp, order, shift, deriv, a) :
    ''' \partial \Lambda^{(o)}  '''
    N, nn, K = compACTexp.N, compACTexp.nn, compACTexp.K

    if order == 1 :
        if shift == 0 :
            if deriv == 1 :
                parallel = D_triGmm(nn+a, N+K*a)
                perpendic = - np.ones((len(nn),len(nn))) * triGmm(N+K*a)
                output = [parallel, perpendic]
            elif deriv == 2 :
                parallel = D_quadriGmm(nn+a, N+K*a)
                cross = - np.ones((len(nn),len(nn))) * quadriGmm(N+K*a)
                perpendic = - np.ones((len(nn),len(nn),len(nn))) * quadriGmm(N+K*a)
                output = [parallel, cross, cross, cross, perpendic]                
            else :
                raise IOError("FIXME: Developer Error in Lambda.")
        else :
            raise IOError("FIXME: Developer Error in Lambda.")
    elif order == 2 :
        if shift == 0 :
            if deriv == 1 :  
                # FIXME : to be checked!
                Lambda_1_0 = compACTexp._Lambda( 1, 0, a )
                der_Lambda_1_0 = compACTexp._der_Lambda( 1, 0, 1, a )
                der2_Lambda_1_0 = compACTexp._der_Lambda( 1, 0, 2, a )
                # i==j, j==k
                parallel = 2 * Lambda_1_0 * der_Lambda_1_0[0] + der2_Lambda_1_0[0]
                # j==k
                cross_1 = 2 * der_Lambda_1_0[1] * np.tile(Lambda_1_0, (len(nn),1))
                cross_1 += der2_Lambda_1_0[1]
                # k==i
                cross_2 = der_Lambda_1_0[1] * np.tile(Lambda_1_0[:,None], (1,len(nn)))
                cross_2 += np.outer(der_Lambda_1_0[0], Lambda_1_0)
                cross_2 += der2_Lambda_1_0[2]
                # i==j 
                cross_3 = np.outer(der_Lambda_1_0[0], Lambda_1_0, )
                cross_3 += np.tile(Lambda_1_0[:,None], (1,len(nn))) * der_Lambda_1_0[1]
                cross_3 += der2_Lambda_1_0[3]
                # i!=j, j!=k, k!=i
                perpendic = np.array([ x.T for x in np.outer(der_Lambda_1_0[1], Lambda_1_0).reshape((len(nn),)*3 )])
                perpendic += np.outer(der_Lambda_1_0[1], Lambda_1_0).reshape( (len(nn),)*3 )
                perpendic += der2_Lambda_1_0[4]
                output = [parallel, cross_1, cross_2, cross_3, perpendic]
            else :        
                raise IOError("FIXME: Developer Error in Lambda.")              
        else :        
            raise IOError("FIXME: Developer Error in Lambda.")
    else :
        raise IOError("FIXME: Developer Error in Lambda.")

    return output

def right_order_up_(x, initial, final) :
    '''raise order from `initial` to `final` term-term multiplication'''
    if final <= initial :
        raise IOError("FIXME: developer error in right_order_up")
    elif initial == 1 :
        if final == 2 :
            parallel = x
            perpendic = np.tile(x, (len(x), 1)).T
            output = [parallel, perpendic]      
        else :
            raise IOError("FIXME: developer error in right_order_up")
    elif initial == 2 :
        if final == 3 :
            parallel = x[0]
            cross_1 = np.tile(x[0], (len(x[0]), 1)).T
            cross_2 = x[1].T
            cross_3 = x[1]
            perpendic = np.array( [ x[1] ] * x[1].shape[0] )
            output = [parallel, cross_1, cross_2, cross_3, perpendic]  
        else :
            raise IOError("FIXME: developer error in right_order_up")
    else :
        raise IOError("FIXME: developer error in right_order_up")
    return output


#############################################
#  Posterior estimator vs Dirichelet param  #
#############################################

def post_entropy_( compACTexp, a ):
    '''Estimate of the entropy at alpha.'''
    
    Omega_1, Omega_1_norm = compACTexp._Omega( 1, a )
    Lambda_1_1 = compACTexp._Lambda(1, 1, a)

    '''  - q_i ln(q_i) '''
    sumList = - Omega_1 * Lambda_1_1
    output = mp.fdiv( compACTexp._ffsum( sumList ), Omega_1_norm )
    return output

def post_entropy_squared_( compACTexp, a ) :
    '''Estimate of the entropy squared at alpha.'''
    
    Omega_2, Omega_2_norm = compACTexp._Omega( 2, a )
    Lambda_2_2 = compACTexp._Lambda(2, 2, a)

    ''' q_i q_j ln(q_i) ln(q_j) '''
    sumList = [ x * y for x, y in zip(Omega_2, Lambda_2_2) ]
    output = mp.fdiv( compACTexp._ffsum( sumList ), Omega_2_norm )
    return output


#########################################
#  DKL estimation vs Dirichelet params  #
#########################################

def post_crossentropy_( compACTdiv, a, b ):
    '''Estimate of the divergence at alpha and beta.'''
        
    Omega_1_q, Omega_1_norm_q = compACTdiv.compact_1._Omega( 1, a )
    Lambda_1_0_t = compACTdiv.compact_2._Lambda(1, 0, b)

    ''' - q_i ln(t_i) '''
    sumList = - Omega_1_q * Lambda_1_0_t
    output = mp.fdiv( compACTdiv._ffsum( sumList ), Omega_1_norm_q )
    return output

def post_crossentropy_squared_( compACTdiv, a, b ):
    '''Estimate of the divergence at alpha and beta.'''
        
    Omega_2_q, Omega_2_norm_q = compACTdiv.compact_1._Omega( 2, a )
    Lambda_2_0_t = compACTdiv.compact_2._Lambda(2, 0, b)

    ''' q_i q_j ln(t_i) ln(t_j)  '''
    sumList = [ x * y for x, y in zip(Omega_2_q, Lambda_2_0_t) ]
    output = mp.fdiv( compACTdiv._ffsum( sumList ), Omega_2_norm_q )
    return output


def post_divergence_( compACTdiv, a, b ):
    '''Estimate of the divergence at alpha and beta.'''
    
    Omega_1_q, Omega_1_norm_q = compACTdiv.compact_1._Omega( 1, a )
    Lambda_1_1_q = compACTdiv.compact_1._Lambda(1, 1, a)
    Lambda_1_0_t = compACTdiv.compact_2._Lambda(1, 0, b)

    ''' q_i ln(q_i) - q_i ln(t_i) '''
    sumList =  Omega_1_q * ( Lambda_1_1_q - Lambda_1_0_t ) 
    output = mp.fdiv( compACTdiv._ffsum( sumList ), Omega_1_norm_q )
    return output

def post_divergence_x_crossentropy_( compACTdiv, a, b ) :
    ''' Estimate of divergence times crossentropy at alpha and beta.'''

    Omega_2_q, Omega_2_norm_q = compACTdiv.compact_1._Omega( 2, a )
    Lambda_1_2_q = compACTdiv.compact_1._Lambda(1, 2, a)
    Lambda_1_0_t = compACTdiv.compact_2._Lambda(1, 0, b)
    Lambda_1up2_0_t = right_order_up_(Lambda_1_0_t, 1, 2)
    Lambda_2_0_t = compACTdiv.compact_2._Lambda(2, 0, b)

    ''' q_i q_j ln(t_i) ln(t_j) - q_i q_j ln(q_i) ln(t_j) '''
    Lambda_terms = [ x - y * z for x,y,z in zip(Lambda_2_0_t, Lambda_1_2_q, Lambda_1up2_0_t) ]
    sumList = [ x * y for x, y in zip(Omega_2_q, Lambda_terms) ]
    output = mp.fdiv( compACTdiv._ffsum( sumList ), Omega_2_norm_q )
    return output

def post_divergence_squared_( compACTdiv, a, b ) :
    ''' Estimate of the squared divergence at alpha and beta.'''

    Omega_2_q, Omega_2_norm_q = compACTdiv.compact_1._Omega( 2, a )
    Lambda_1_2_q = compACTdiv.compact_1._Lambda(1, 2, a)
    Lambda_2_2_q = compACTdiv.compact_1._Lambda(2, 2, a)
    Lambda_1_0_t = compACTdiv.compact_2._Lambda(1, 0, b)
    Lambda_1up2_0_t = right_order_up_(Lambda_1_0_t, 1, 2)
    Lambda_2_0_t = compACTdiv.compact_2._Lambda(2, 0, b)
        
    ''' q_i q_j ln(q_i) ln(q_j)  - 2 * q_i q_j ln(q_i) ln(t_j) + '''
    Lambda_terms = [ x-2*y*z for x,y,z in zip(Lambda_2_2_q, Lambda_1_2_q, Lambda_1up2_0_t)]
    ''' + q_i q_j ln(t_i) ln(t_j) '''
    Lambda_terms = [ x + y for x, y in zip(Lambda_terms, Lambda_2_0_t) ]
    sumList = [ x * y for x, y in zip(Omega_2_q, Lambda_terms) ]  
    output = mp.fdiv( compACTdiv._ffsum( sumList ), Omega_2_norm_q )
    return output


def post_divergence_x_crossentropy_squared_( compACTdiv, a, b ) :  
    ''' Estimate of divergence times crossentropy squared at alpha and beta.'''

    Omega_3_q, Omega_3_norm_q = compACTdiv.compact_1._Omega( 3, a )
    Lambda_1_3_q = compACTdiv.compact_1._Lambda(1, 3, a)
    Lambda_2_0_t = compACTdiv.compact_2._Lambda(2, 0, b)
    Lambda_2up3_0_t = right_order_up_(Lambda_2_0_t, 2, 3)
    Lambda_3_0_t = compACTdiv.compact_2._Lambda(3, 0, b)

    ''' q_i q_j q_k ln(q_i) ln(t_j) ln(t_k) - q_i q_j q_k ln(t_i) ln(t_j) ln(t_k) '''
    Lambda_terms = [ x*y-z for x,y,z in zip(Lambda_1_3_q, Lambda_2up3_0_t, Lambda_3_0_t) ]
    sumList = [ x*y for x, y in zip(Omega_3_q, Lambda_terms) ]
    output = mp.fdiv( compACTdiv._ffsum( sumList ), Omega_3_norm_q )
    return output