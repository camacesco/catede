
'''
    (in development) Gamam Expansion
    Copyright (C) November 2022 Francesco Camaglia, LPENS 
'''

import numpy as np
from .beta_func_multivar import *
from .new_calculus import *

#####################
#  GAMMA EXPANSION  #
#####################

def Gamma_Exp_( compExp, order, b) :
    ''' Derivative of logGamma  '''
    yvec = compExp.nn + b
    Y = compExp.N + compExp.K * b

    if order == 1 :
        # d_i logGamma 
        # (0) : i
        yield diGmm(yvec)    
    elif order == 2 :
        # d_i d_j logGamma
        # (0,0) : i==j
        yield np.power(diGmm(yvec), 2) + triGmm(yvec)
        # (0,1) : i!=j
        yield outer( diGmm(yvec), diGmm(yvec) )

def devGamma_Exp_( compExp, order, b) :
    ''' double Derivative of logGamma  '''
    yvec = compExp.nn + b
    Y = compExp.N + compExp.K * b

    if order == 1 :
        # d_i logGamma 
        # (0) : i
        yield triGmm(yvec)    
    elif order == 2 :
        # d_i d_j logGamma
        # (0,0) : i==j
        yield 2 * triGmm(yvec) * diGmm(yvec) + quadriGmm(yvec)
        # (0,1) : i!=j
        yield 2 * outer( triGmm(yvec), diGmm(yvec) )    

def devMeasure_Gamma_Exp_( compExp, order, b, gK ) :
    ''' Derivative of logGamma times derivative of altered gamma mesure (~) '''
    yvec = compExp.nn + b
    YgK = compExp.N + compExp.K * b + gK

    if order == 1 :
        # d_i prod(logGamma) / logGmma(sum)
        # (0) : i
        yield D_diGmm( yvec, YgK )

    elif order == 2 :
        # d_i prod(logGamma) / logGmma(sum) * psi(y_j)
        # (0,0) : i==j
        yield D_diGmm( yvec, YgK ) * diGmm(yvec)
        # (0,1) : i!=j
        yield outer( D_diGmm( yvec, YgK ), diGmm(yvec) )
    
    elif order == 3 :
        # d_i prod(logGamma) / logGmma(sum) * ( psi(y_j) * psi(y_k) + (j==k) * psi1(y_j) )
        # (0,0,0) : i==j==k
        yield D_diGmm( yvec, YgK ) * ( np.power(diGmm(yvec), 2) + triGmm(yvec) )
        # (0,1,1) : i!=j==k
        yield D_diGmm( yvec, YgK ) * ( np.power(diGmm(yvec), 2) + triGmm(yvec) )
        # (0,1,0) : j!=k==i
        yield outer( D_diGmm( yvec, YgK ) * diGmm(yvec), diGmm(yvec) )
        # (0,0,1) : k!=i==j
        yield outer( D_diGmm( yvec, YgK ) * diGmm(yvec), diGmm(yvec) )
        # (0,1,2) : i!=j, j!=k, k!=i
        yield outer( D_diGmm( yvec, YgK ), outer( diGmm(yvec), diGmm(yvec) ) )



# >>>>>>>>>>>>>>
#  LIKELIHOOD  #
# <<<<<<<<<<<<<<

def Posterior_Expansion( compDiv, a, b, gK, order=2 ) :
    '''Return the contribution of the expnasion.''' 
    compExp_1, compExp_2, = compDiv.compact_1,  compDiv.compact_2

    # expansion
    expansion = 1.
    # first order expansion
    if order > 0 :
            Omega_1 = compExp_1._Omega( "i", a )
            Psi_1 = Gamma_Exp_( compExp_2, 1, b )
            sumGens = ( Om_1 * Ps_1 for Om_1, Ps_1 in zip( Omega_1, Psi_1 ) )
            expansion += gK * compExp_1._norm_ffsum( sumGens, a, dim=1 )
    # second order expansion
    if order > 1 :
        Omega_2 = compExp_1._Omega( "ij", a )
        Psi_2 = Gamma_Exp_( compExp_2, 2, b )
        sumGens = ( Om_2 * Ps_2 for Om_2, Ps_2 in zip( Omega_2, Psi_2 ) )
        expansion += 0.5 * (gK**2) * compExp_1._norm_ffsum( sumGens, a, dim=2 )

    return expansion

def optimal_divergence_gamma_exp_param_( compDiv ) :
    '''Return Dirchlet parameter which optimizes divergence posterior alpha=beta (~).''' 
    # FIXME : extreme cases ?

    def myfunc( var, *args ) :
        alpha, beta, Kgamma = var
        compDiv = args[0]
        compExp_1, compExp_2, = compDiv.compact_1,  compDiv.compact_2
        K = compDiv.K
        Y = compExp_2.N + K * beta

        Kgamma2 = 0.5 * ( Kgamma**2 )

        # likelihood contribution
        c1 = log_alphaLikelihood( compExp_1, alpha ) + log_alphaLikelihood( compExp_2, beta )

        # NOT expanded denominator of multivar beta
        c1 += LogGmm( Y ) - LogGmm( Y + Kgamma ) - LogGmm( K*beta ) + LogGmm( K*beta + Kgamma )

        # expanded denominator of multivar beta
        #c1 -= np.log( 1. + Kgamma * diGmm(Y) + Kgamma2 * ( triGmm(Y) + np.power(diGmm(Y), 2) ) )
        #c1 += np.log( 1. + Kgamma * diGmm(K*beta) + Kgamma2 * ( triGmm(K*beta) + np.power(diGmm(K*beta), 2) ) )
        
        c1 += np.log( Posterior_Expansion( compDiv, alpha, beta, Kgamma ) )
        # prior contribution to expansion
        expansion = 1. + Kgamma * diGmm(beta)
        expansion += Kgamma2 * ( np.power(diGmm(beta), 2) + triGmm(beta) * (alpha+1)/(K*alpha+1) )
        c2 = np.log( expansion )
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        return - ( c1 - c2 )

    guess_a = optimal_dirichlet_param_( compDiv.compact_1 )[0]
    guess_b = optimal_dirichlet_param_( compDiv.compact_2 )[0]

    return myMinimizer( myfunc, [guess_a, guess_b, 0.1], (compDiv,) )

# >>>>>>>>>>>>>>>>>>>>>>
#  ENTROPY ESTIMATORS  #
# <<<<<<<<<<<<<<<<<<<<<<

def entropy_gmm_exp( compDiv, a, b, gK, order=2 ) :

    output = compDiv.compact_1.entropy( a )
    if order > 0 :
        output += gK * entropy_gmm_exp_order1( compDiv, a, b )
    if order > 1 :
        output += 0.5 * (gK**2) * entropy_gmm_exp_order2( compDiv, a, b )
    return output

def entropy_gmm_exp_order1( compDiv, a, b ) :
    ''' - sum_ij < q_i * q_j * log q_i > * psi(y_j) '''

    compExp_1, compExp_2, = compDiv.compact_1, compDiv.compact_2

    Q2logQ1 = shift_2_deriv_1( compExp_1, a )
    Psi_1 = one_up_SX_( Gamma_Exp_( compExp_2, 1, b ) )

    gens = zip( Q2logQ1, Psi_1 )
    sumGens = ( q2logq * ps1 for q2logq, ps1 in gens )
    sum_value = compDiv.compact_1._norm_ffsum( sumGens, a, dim=2 )
    return - sum_value

def entropy_gmm_exp_order2( compDiv, a, b ) :
    ''' - sum_ijk < q_i * q_j * q_k * log q_i > * ( psi(y_j) * psi(y_k) + (j==k) * psi1(y_j) )'''

    compExp_1, compExp_2, = compDiv.compact_1, compDiv.compact_2

    Q3logQ1 = shift_3_deriv_1( compExp_1, a )
    Psi_2 = one_up_SX_( Gamma_Exp_( compExp_2, 2, b ) )

    gens = zip( Q3logQ1, Psi_2 )
    sumGens = ( q3logq * ps2 for q3logq, ps2 in gens )
    sum_value = compDiv.compact_1._norm_ffsum( sumGens, a, dim=3 )
    return - sum_value

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>
#  CROSS-ENTROPY ESTIMATORS  #
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<

def crossentropy_gmm_exp( compDiv, a, b, gK, order=2 ) :

    output = crossentropy_gmm_exp_order0( compDiv, a, b, gK ) 
    if order > 0 :
        output += gK * crossentropy_gmm_exp_order1( compDiv, a, b, gK )
    if order > 1 :
        output += 0.5 * (gK**2) * crossentropy_gmm_exp_order2( compDiv, a, b, gK )
    return output

def crossentropy_gmm_exp_order0( compDiv, a, b, gK ) :
    ''' - sum_i < q_i > * ( psi(y_i) - psi(YgK) ) '''

    compExp_1, compExp_2, = compDiv.compact_1, compDiv.compact_2
    Omega_1 = compExp_1._Omega( "i", a )
    Lamda_gK = devMeasure_Gamma_Exp_( compExp_2, 1, b, gK )

    gens = zip( Omega_1, Lamda_gK )
    sumGens = ( Om1 * LmgK for Om1, LmgK in gens )
    sum_value = compDiv.compact_1._norm_ffsum( sumGens, a, dim=1 )
    return - sum_value

def crossentropy_gmm_exp_order1( compDiv, a, b, gK ) :
    ''' - sum_ij < q_i * q_j > * ( psi(y_i) - psi(YgK) ) psi(y_j) '''

    compExp_1, compExp_2, = compDiv.compact_1, compDiv.compact_2
    Omega_2 = compExp_1._Omega( "ij", a )
    Lamda_gK = devMeasure_Gamma_Exp_( compExp_2, 2, b, gK )

    gens = zip( Omega_2, Lamda_gK )
    sumGens = ( Om2 * LmgK for Om2, LmgK in gens )
    sum_value = compDiv.compact_1._norm_ffsum( sumGens, a, dim=2 )

    ''' - sum_i < q_i ^ 2 > * psi1(y_i)  '''
    dPsi_1 = devGamma_Exp_( compExp_2, 1, b )
    Omega_2 = compExp_1._Omega( "ii", a )
    gens = zip( Omega_2, dPsi_1 )
    sumGens = ( Om2 * dPs for Om2, dPs in gens )
    sum_value += Normalize_Omega_(compDiv.compact_1, 2, a, compDiv.compact_1._ffsum( sumGens, dim=1 ) ) 

    return - sum_value

def crossentropy_gmm_exp_order2( compDiv, a, b, gK ) :
    ''' - sum_ijk < q_i * q_j * q_k > * ( psi(y_i) - psi(YgK) ) * ( psi(y_j) * psi(y_j) + (j==k) * psi1(y_j) ) '''

    compExp_1, compExp_2, = compDiv.compact_1, compDiv.compact_2
    Omega_3 = compExp_1._Omega( "ijk", a )
    Lamda_gK = devMeasure_Gamma_Exp_( compExp_2, 3, b, gK )

    gens = zip( Omega_3, Lamda_gK )
    sumGens = ( Om3 * LmgK for Om3, LmgK in gens )
    sum_value = compDiv.compact_1._norm_ffsum( sumGens, a, dim=3 )

    ''' < q_i ^ 2 q_k > * ( 2 psi1(y_i) * psi(y_k) + (i==k) * psi2(y_i) ) '''
    dPsi_2 = devGamma_Exp_( compExp_2, 2, b )
    Omega_3 = compExp_1._Omega( "iik", a )
    gens = zip( Omega_3, dPsi_2 )
    sumGens = ( Om3 * dPs for Om3, dPs in gens )
    sum_value += Normalize_Omega_(compDiv.compact_1, 3, a, compDiv.compact_1._ffsum( sumGens, dim=2 ) ) 

    return sum_value