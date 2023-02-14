#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Multivariate Beta Function Calculus (in development)
    Copyright (C) October 2022 Francesco Camaglia, LPENS 
'''
import warnings
import numpy as np
import mpmath as mp
import pandas as pd
import itertools
from scipy.special import loggamma, polygamma
from scipy.special import beta as beta_func   

class Experiment_Compact :
    def __init__( self, source=None, is_div=None, is_comp=False, load=None ) :
        '''
        '''

        if source is None :
                if load is None :
                    raise IOError("One between `source` and `load` must be specified.")
                else :
                    self._load(filename=load)
        elif is_div is None :
            if is_comp is True :
                # experiment compact
                warnings.warn("Trying to create a compact experiment from a a compact experiment.")
                self.N = source.N                                    # total number of counts
                self.K = source.K                                    # user number of categories
                self.Kobs = source.Kobs                              # observed number of categories
                self.nn = source.nn                                  # counts
                self.ff = source.ff                                  # recurrency of counts
            else :
                # experiment
                self.N = source.tot_counts                           # total number of counts
                self.K = source.usr_n_categ                          # user number of categories
                self.Kobs = source.obs_n_categ                       # observed number of categories
                self.nn = source.counts_hist.index.values            # counts
                self.ff = source.counts_hist.values                  # recurrency of counts
        else  :
            if is_comp is True :
                # divergence compact
                self.K = source.K
                self.ff = source.ff                                   
                if (is_div == 1) or (is_div == 'A') :                                          
                    self.Kobs = source.Kobs_1   
                    self.N = source.N_1                               
                    self.nn = source.nn_1    
                elif (is_div == 2) or (is_div == 'B') :                                       
                    self.Kobs = source.Kobs_2
                    self.N = source.N_2                               
                    self.nn = source.nn_2 
                else :
                    raise IOError("Unrecognized identifier for `is_div`.")  
            else :
                # divergence 
                self.K = source.usr_n_categ
                self.ff = source.counts_hist.values                                    
                if (is_div == 1) or (is_div == 'A') :                                          
                    self.Kobs = source.exp_1.obs_n_categ   
                    self.N = source.exp_1.tot_counts                                
                    self.nn = source.exp_1.counts_hist.index.values     
                elif (is_div == 2) or (is_div == 'B') :                                       
                    self.Kobs = source.exp_2.obs_n_categ
                    self.N = source.exp_2.tot_counts                               
                    self.nn = source.exp_2.counts_hist.index.values 
                else :
                    raise IOError("Unrecognized identifier for `is_div`.")                

    def entropy( self, a ) :
        '''Posterior Multinomial-Dirichlet Shannon entropy estimator.'''
        return post_entropy_( self, a )

    def squared_entropy( self, a ) :
        '''Posterior Multinomial-Dirichlet sqaured Shannon entropy estimator.'''
        return post_entropy_sqr_( self, a )

    def simpson( self, a ) :
        '''Posterior Multinomial-Dirichlet Simpson index estimator.'''
        return post_simpson_( self, a )

    def squared_simpson( self, a ) :
        '''Posterior Multinomial-Dirichlet sqaured entropy estimator.'''
        return post_simpson_sqr_( self, a )

    def _Omega( self, terms, a ) :
        '''Shift function Omega.'''
        return Omega_( self, terms, a )

    def _derOmega( self, shift, terms, a ) :
        '''Derivative of shift function Omega.'''
        if shift == 1 :
            return derOmegaS1_( self, terms, a )
        elif shift == 2 :
            return derOmegaS2_( self, terms, a )
        elif shift == 3 :
            return derOmegaS3_( self, terms, a )

    def _Lambda( self, order, a ) :
        '''Derivative function Lambda.'''
        return Lambda_( self, order, a )

    def _ffsum( self, sumGens, dim ) :
        return count_hist_sum_( self.ff, sumGens, dim )

    def _norm_ffsum( self, sumGens, a, dim, dimNorm=None ) :
        if dimNorm is None : dimNorm = dim
        return Normalize_Omega_( self, dimNorm, a, self._ffsum( sumGens, dim ) )

    def _save( self, filename ) : 
        '''Save the Experiment_Compact object to `filename`.'''
        # parameters
        pd.DataFrame(
            [ self.N, self.K, self.Kobs, len(self.ff) ],
            index = ['N', 'K', 'Kobs', 'size_of_ff']
        ).to_csv( filename, sep=' ', mode='w', header=False, index=True )
        # counts hist
        pd.DataFrame(
            { 'nn': self.nn, 'ff': self.ff }
        ).to_csv( filename, sep=' ', mode='a', header=True, index=False )  

    def _load( self, filename ) : 
        '''Load the saved Experiment_Compact object from `filename`.'''
        # parameters
        f = open(filename, "r")
        params = {}
        for _ in range(4) :
            thisline = f.readline().strip().split(' ')
            params[ thisline[0] ] = thisline[1]
        self.N = int(params[ 'N' ])
        self.K = int(params['K'])
        self.Kobs = int(params['Kobs'])
        # count hist
        df = pd.read_csv( filename, header=4, sep=' ' )
        assert len(df) == int(params['size_of_ff'])
        self.nn = df['nn'].values
        self.ff = df['ff'].values  
###

##############################
#  DIVERGENCE COMPACT CLASS  #
##############################
    
class Divergence_Compact :
    def __init__( self, div=None, load=None ) :
        ''' '''
        if div is None :
            if load is None :
                raise IOError("one between `exp` and `load` must be specified.")
            else :
                self._load(filename=load)
        else :
            self.N_1 = div.tot_counts['Exp-1']                           # total number of counts for Exp 1
            self.N_2 = div.tot_counts['Exp-2']                           # total number of counts for Exp 2
            self.K = div.usr_n_categ                                     # user number of categories
            self.Kobs_u = div.obs_n_categ['Union']                       # observed number of categories
            self.Kobs_1 = div.obs_n_categ['Exp-1']                       # observed number of categories for Exp 1
            self.Kobs_2 = div.obs_n_categ['Exp-2']                       # observed number of categories for Exp 2
            temp = np.array(list(map(lambda x: [x[0],x[1]], div.counts_hist.index.values)))
            self.nn_1 = temp[:,0]                                        # counts for Exp 1
            self.nn_2 = temp[:,1]                                        # counts for Exp 2
            self.ff = div.counts_hist.values                             # recurrency of counts
            self.compact_1 = Experiment_Compact( source=div, is_div=1 )  # compact for Exp 1
            self.compact_2 = Experiment_Compact( source=div, is_div=2 )  # compact for Exp 1

    def _ffsum( self, sumList, dim ) :
        return count_hist_sum_( self.ff, sumList, dim )

    def divergence( self, a, b ) :
        '''Posterior independent Multinomial-Dirichlet divergence estimator.'''
        return post_divergence_( self, a, b )

    def squared_divergence( self, a, b ) :
        '''Posterior independent Multinomial-Dirichlet squareddivergence estimator.'''
        return post_divergence_sqr_( self, a, b )

    def _save( self, filename ) : 
        # FIXME : how to implement for compression? (change also load)
        '''Save the Divergence_Compact object to `filename`.'''
        # parameters
        pd.DataFrame(
            [ self.N_1, self.N_2, self.K, self.Kobs_1, self.Kobs_2, self.Kobs_u, len(self.ff) ],
            index = ['N_1', 'N_2', 'K', 'Kobs_1', 'Kobs_2', 'Kobs_u', 'size_of_ff']
        ).to_csv( filename, sep=' ', mode='w', header=False, index=True )
        # counts hist
        pd.DataFrame(
            { 'nn_1' : self.nn_1, 'nn_2' : self.nn_2, 'ff' : self.ff }
        ).to_csv( filename, sep=' ', mode='a', header=True, index=False )


    def _load( self, filename ) : 
        '''Load the saved Divergence_Compact object from `filename`.'''
        # parameters
        f = open(filename, "r")
        params = {}
        dict_size = 7
        for _ in range(dict_size) :
            thisline = f.readline().strip().split(' ')
            params[ thisline[0] ] = thisline[1]
        self.N_1 = int(params['N_1'])
        self.N_2 = int(params['N_2'])
        self.K = int(params['K'])
        self.Kobs_1 = int(params['Kobs_1'])
        self.Kobs_2 = int(params['Kobs_2'])
        self.Kobs_u = int(params['Kobs_u'])
        size_of_ff = int(params['size_of_ff'])
        #count hist
        df = pd.read_csv( filename, header=dict_size, sep=' ' )
        assert len(df) == size_of_ff
        self.nn_1 = df['nn_1'].values
        self.nn_2 = df['nn_2'].values
        self.ff = df['ff'].values
        self.compact_1 = Experiment_Compact( source=self, is_div=1, is_comp=True )
        self.compact_2 = Experiment_Compact( source=self, is_div=2, is_comp=True )
###

################
#  SUMMATIONS  #
################

def count_hist_sum_( ff, sumGens, dim ) :
    ''' Summing methods for histograms of counts.'''

    # FIXME : can I improve with np.diagonal( axis1, axis2)?
    idx = np.arange(len(ff))

    # (0,...) : all ==        
    tmp1D = next(sumGens)

    if dim == 2 :
        # (0,1) : i!=j
        tmp2D = next(sumGens)
        # summing
        tmp1D += tmp2D.dot(ff) - tmp2D.diagonal()

    elif dim == 3 :
        # (0,1,1) : i!=j==k
        tmp2D = next(sumGens)
        # (0,1,0) : j!=k==i
        tmp2D += next(sumGens) 
        # (0,0,1) : k!=i==j
        tmp2D += next(sumGens)
        # (0,1,2) : i!=j, j!=k, k!=i
        tmp3D = next(sumGens)
        # summing
        tmp2D -= tmp3D[:,idx,idx] + tmp3D[idx,:,idx] + tmp3D[idx,idx,:]
        tmp1D += ( tmp2D + tmp3D.dot(ff) ).dot(ff) - tmp2D.diagonal() - tmp3D[idx,idx,idx]
    
    output =tmp1D.dot(ff)

    return output        

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

def quintiGmm(x) :    
    '''Quintigamma function (polygamma of order 3).'''
    return polygamma(3, x)

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
    ''' alias of Log Gamma function'''
    return loggamma( x ).real  

####################################
#  MULTIVARIATE BETA COMPUTATIONS  #
####################################

def outer(x, y):
    return np.multiply.outer(x, y)

def ones(shape):
    return np.ones(shape=shape)

def Normalize_Omega_( compExp, shift, a, value ) :
    '''Returns the normalization for the functions Omega.'''
    # NOTE: this should be a control in case of numerical issue (change numpy to mpmath)
    X = compExp.K * a + compExp.N
    Norm = np.product( X*np.ones(shift) + np.arange(0,shift,1) )
    return np.divide( value, Norm )

def ListIndexes( INDEX ) :
    '''It returns the list of all constrained indexes.'''

    if len( INDEX ) == 1 :
        termsList = [(0)]
    else :
        DICT = {}
        LIST = []
        i = 0
        for k in INDEX :
            if k not in DICT :
                DICT.update({k:i})
                i += 1
            LIST.append(DICT[k])

        all_dict_values = [ x for x in itertools.product( * ( range(DICT[k]+1) for k in DICT ) ) ]
        # note this produces degenerate and disordered indexes
        # since they're not in the if statemets is not an issue
        termsList = []
        for values in all_dict_values :
            tmp_dict = dict(zip(DICT.keys(), values))
            termsList.append( tuple([ tmp_dict[k] for k in INDEX ]) )

    return termsList

def Omega_( compExp, terms, a) :
    '''Generator of the shift functions Omega.'''
    xvec = compExp.nn + a

    termsList = ListIndexes( terms )
    shift = len(terms)
    if shift == 1 :
        # B (x+i) / B
        if (0) in termsList :                               # i
            yield xvec 

    elif shift == 2 :
        # B (x+i+j) / B
        if (0,0) in termsList :                             # i==j
            yield xvec * (xvec+1)
        if (0,1) in termsList :                             # i!=j
            yield outer( xvec , xvec )

    elif shift == 3 :
        # B (x+i+j+k) / B
        if (0,0,0) in termsList :                           # i==j==k
            yield xvec * (xvec+1) * (xvec+2)
        if (0,1,1) in termsList :                           # i!=j==k
            yield outer( xvec, xvec * (xvec+1) ) 
        if (0,1,0) in termsList :                           # j!=k==i
            yield outer( xvec * (xvec+1), xvec)
        if (0,0,1) in termsList :                           # k!=i==j
            yield outer( xvec * (xvec+1), xvec )
        if (0,1,2) in termsList :                           # i!=j, j!=k, k!=i
            yield outer( xvec, outer( xvec, xvec ) )

    elif shift == 4 :
        # B (x+i+j+k+h) / B
        # DIM = 1 (3-edges path joint)
        if (0,0,0,0) in termsList :                         # i==j==k
            yield xvec * (xvec+1) * (xvec+2) * (xvec+3)
        # DIM = 2 (2-edges paths joint)
        if (0,1,1,1) in termsList :                         # i!==j==k==h
            yield outer( xvec, xvec * (xvec+1) * (xvec+2) )
        if (0,1,0,0) in termsList :                         # i==k==h!=j
            yield outer( xvec * (xvec+1) * (xvec+2), xvec )
        if (0,0,1,0) in termsList :                         # i==j==h!=k
            yield outer( xvec * (xvec+1) * (xvec+2), xvec )
        if (0,0,0,1) in termsList :                         # i==j==k!=h
            yield outer( xvec * (xvec+1) * (xvec+2), xvec )
        # DIM = 2 (2-edges paths disjoint)
        if (0,1,1,0) in termsList :                         # i==h!=j==k
            yield outer( xvec * (xvec+1), xvec * (xvec+1) )
        if (0,1,0,1) in termsList :                         # i==k!=h==j
            yield outer( xvec * (xvec+1), xvec * (xvec+1) )
        if (0,0,1,1) in termsList :                         # i==j!=k==h
            yield outer( xvec * (xvec+1), xvec * (xvec+1) )
        # DIM = 3 (1-edge paths joint)
        if (0,1,2,2) in termsList :                         # i!=j, i!=k, j!=k, k==h
            yield outer( xvec, outer( xvec, xvec * (xvec+1) ) )
        if (0,1,2,1) in termsList :                         # i!=j, j==h, i!=h, j!=k
            yield outer( xvec, outer( xvec * (xvec+1), xvec ) )
        if (0,1,2,0) in termsList :                         # i==h, i!=j, i!=k, j!=k
            yield outer( xvec * (xvec+1), outer( xvec, xvec ) )
        if (0,1,1,2) in termsList :                         # i!=j, j==k, i!=h, j!=h
            yield outer( xvec, outer( xvec * (xvec+1), xvec ) )
        if (0,1,0,2) in termsList :                         # i==k, i!=j, i!=h, j!=h
            yield outer( xvec * (xvec+1), outer( xvec, xvec ) )
        if (0,0,1,2) in termsList :                         # i==j, i!=k, i!=h, k!=h
            yield outer( xvec * (xvec+1), outer( xvec, xvec ) )
        # DIM = 4 (0-edge path)
        if (0,1,2,3) in termsList :                         # all != 
            yield outer( xvec, outer( xvec, outer( xvec, xvec ) ) )

def derOmegaS1_( compExp, terms, a) :
    '''Generator of the functions derivative of Omega.'''
    # NOTE : can I take advantage of symmetries ?
    xvec = compExp.nn + a
    X = compExp.N + compExp.K * a
    i_xvec = np.power( xvec, -1. )  
    i_X = np.power( X, -1. )
    xlen = len(i_xvec)

    termsList = ListIndexes( terms )
    order = len(terms) - 1
    if order == 1 :
        # d_j Omega_i / Omega_i
        if (0,0) in termsList :                             # i==j
            yield i_xvec - i_X
        if (0,1) in termsList :                             # i!=j
            yield - i_X * ones((xlen,xlen))
    elif order == 2 :
        # d_j d_k Omega_i / Omega_i
        C = 2. * i_X**2
        if (0,0,0) in termsList :                           # i==j==k
            yield C - 2. * i_xvec * i_X
        if (0,1,1) in termsList :                           # i!=j==k
            yield C * ones((xlen,xlen))
        if (0,1,0) in termsList :                           # j!=k==i
            yield C - i_X * outer( i_xvec, ones(xlen) )
        if (0,0,1) in termsList :                           # k!=i==j
            yield C - i_X * outer( i_xvec, ones(xlen) )
        if (0,1,2) in termsList :                           # i!=j, j!=k, k!=i
            yield C * ones((xlen,xlen,xlen))

def derOmegaS2_( compExp, terms, a ) :
    '''Generator of the functions derivative of Omega.'''
    # NOTE : can I take advantage of symmetries ?
    xvec = compExp.nn + a
    X = compExp.N + compExp.K * a
    i_xvec, i_xvec_p1 = [ np.power( xvec+i, -1. ) for i in range(2) ]
    i_X, i_X_p1 = [ np.power( X+i, -1. ) for i in range(2) ]
    xlen = len(i_xvec)
    C0 = i_X + i_X_p1

    termsList = ListIndexes( terms )
    order = len(terms) - 2
    if order == 1 :
        # d_k Omega_ij / Omega_ij       
        if (0,0,0) in termsList :                           # i==j==k
            yield i_xvec + i_xvec_p1 - C0
        if (0,1,1) in termsList :                           # i!=j==k
            yield outer( ones(xlen), i_xvec ) - C0
        if (0,1,0) in termsList :                           # j!=k==i
            yield outer( i_xvec, ones(xlen) ) - C0
        if (0,0,1) in termsList :                           # k!=i==j
            yield - C0 * np.ones(shape=(xlen,xlen)) 
        if (0,1,2) in termsList :                           # i!=j, j!=k, k!=i
            yield - C0 *  np.ones(shape=(xlen,xlen,xlen))
    elif order == 2 :
        # d_k d_h Omega_ij / Omega_ij
        C1 = i_X**2 + i_X_p1**2 + C0**2
        # DIM = 1 (3-edges path joint)                      
        if (0,0,0,0) in termsList :                         # i==j==k==h
            yield C1 + 2 * i_xvec * i_xvec_p1 - C0 * ( i_xvec + i_xvec_p1 )
        # DIM = 2 (2-edges paths joint)
        if (0,1,1,1) in termsList :                         # i!==j==k==h
            yield C1 - 2 * C0 * outer(ones(xlen),i_xvec)
        if (0,1,0,0) in termsList :                         # i==k==h!=j
            yield C1 - 2 * C0 * outer(i_xvec,ones(xlen))
        if (0,0,1,0) in termsList :                         # i==j==h!=k
            yield C1 - C0 * outer(i_xvec+i_xvec_p1,ones(xlen))
        if (0,0,0,1) in termsList :                         # i==j==k!=h
            yield C1 - C0 * outer(i_xvec+i_xvec_p1,ones(xlen))
        # DIM = 2 (2-edges paths disjoint)
        if (0,1,1,0) in termsList :                         # i==h!=j==k
            yield C1 + outer(i_xvec,i_xvec) - C0 * (outer(i_xvec,ones(xlen)) + outer(ones(xlen),i_xvec))
        if (0,1,0,1) in termsList :                         # i==k!=h==j
            yield C1 + outer(i_xvec,i_xvec) - C0 * (outer(i_xvec,ones(xlen)) + outer(ones(xlen),i_xvec))
        if (0,0,1,1) in termsList :                         # i==j!=k==h
            yield C1
        # DIM = 3 (1-edge paths joint)
        if (0,1,2,2) in termsList :                         # i!=j, i!=k, j!=k, k==h
            yield C1
        if (0,1,2,1) in termsList :                         # i!=j, j==h, i!=h, j!=k
            yield C1 - C0 * outer(ones(xlen),i_xvec)
        if (0,1,2,0) in termsList :                         # i==h, i!=j, i!=k, j!=k
            yield C1 - C0 * outer(i_xvec,ones(xlen))
        if (0,1,1,2) in termsList :                         # i!=j, j==k, i!=h, j!=h
            yield C1 - C0 * outer(ones(xlen),i_xvec)
        if (0,1,0,2) in termsList :                         # i==k, i!=j, i!=h, j!=h
            yield C1 - C0 * outer(i_xvec,ones(xlen))
        if (0,0,1,2) in termsList :                         # i==j, i!=k, i!=h, k!=h
            yield C1
        # DIM = 4 (0-edge path)
        if (0,1,2,3) in termsList :                         # all != 
            yield C1

def derOmegaS3_( compExp, terms, a ) :
    '''Generator of the functions derivative of Omega.'''
    # NOTE : can I take advantage of symmetries ?
    xvec = compExp.nn + a
    X = compExp.N + compExp.K * a
    i_xvec, i_xvec_p1, i_xvec_p2 = [ np.power( xvec+i, -1. ) for i in range(3) ]
    i_X, i_X_p1, i_X_p2 = [ np.power( X+i, -1. ) for i in range(3) ]
    xlen = len(i_xvec)
    C0 = i_X + i_X_p1 + i_X_p2

    termsList = ListIndexes( terms )
    order = len(terms) - 3
    if order == 1 :
        # d_h Omega_ijk / Omega_ijk
        # DIM = 1 (3-edges path joint)
        if (0,0,0,0) in termsList :                         # i==j==k==h
            yield i_xvec + i_xvec_p1 + i_xvec_p2 - C0
        # DIM = 2 (2-edges paths joint)
        if (0,1,1,1) in termsList :                         # i!==j==k==h
            yield outer(ones(xlen),i_xvec + i_xvec_p1) - C0
        if (0,1,0,0) in termsList :                         # i==k==h!=j
            yield outer(i_xvec + i_xvec_p1,ones(xlen)) - C0
        if (0,0,1,0) in termsList :                         # i==j==h!=k
            yield outer(i_xvec + i_xvec_p1,ones(xlen)) - C0
        if (0,0,0,1) in termsList :                         # i==j==k!=h
            yield  - C0
        # DIM = 2 (2-edges paths disjoint)
        if (0,1,1,0) in termsList :                         # i==h!=j==k
            yield outer(i_xvec,ones(xlen)) - C0
        if (0,1,0,1) in termsList :                         # i==k!=h==j
            yield outer(ones(xlen),i_xvec) - C0
        if (0,0,1,1) in termsList :                         # i==j!=k==h
            yield outer(ones(xlen),i_xvec) - C0
        # DIM = 3 (1-edge paths joint)
        if (0,1,2,2) in termsList :                         # i!=j, i!=k, j!=k, k==h
            yield outer(outer(ones(xlen),ones(xlen)),i_xvec) - C0
        if (0,1,2,1) in termsList :                         # i!=j, j==h, i!=h, j!=k
            yield outer(outer(ones(xlen),i_xvec),ones(xlen)) - C0
        if (0,1,2,0) in termsList :                         # i==h, i!=j, i!=k, j!=k
            yield outer(i_xvec,outer(ones(xlen),ones(xlen))) - C0
        if (0,1,1,2) in termsList :                         # i!=j, j==k, i!=h, j!=h
            yield - C0
        if (0,1,0,2) in termsList :                         # i==k, i!=j, i!=h, j!=h
            yield - C0
        if (0,0,1,2) in termsList :                         # i==j, i!=k, i!=h, k!=h
            yield - C0
        # DIM = 4 (0-edge path)
        if (0,1,2,3) in termsList :                         # all != 
            yield - C0

def Lambda_( compExp, order, a) :
    ''' Derivative function Lambda: \frac{ der^o Beta }{ der^o x_i } / Beta  '''
    xvec = compExp.nn + a
    X = compExp.N + compExp.K * a

    if order == 1 :
        # d_i B / B
        # (0) : i
        yield D_diGmm(xvec, X)    
    elif order == 2 :
        # d_i d_j B / B
        # (0,0) : i==j
        yield np.power(D_diGmm(xvec, X), 2) + D_triGmm(xvec, X)
        # (0,1) : i!=j
        yield np.outer(D_diGmm(xvec, X), D_diGmm(xvec, X)) - triGmm(X)

def one_up_SX_( x ) :
    ''' '''
    tmp1D = next(x)                         # DIM == 1
    xlen = len(tmp1D)
    yield tmp1D                             # (0)->(*;0) V (0,0)->(*;0,0)
    yield outer( ones(xlen), tmp1D )        # (0)->(*;0) V (0,0)->(*;0,0)
    try :
        tmp2D = next(x)                     # DIM == 2
        yield tmp2D.T                       # (0,1)->(*;1,0)
        yield tmp2D                         # (0,1)->(*;0,1)
        yield outer( ones(xlen), tmp2D )    # (0,1)->(*;0,1)
    except StopIteration:
        pass    

def one_up_DX_( x ) :
    ''' '''
    tmp1D = next(x)                         # DIM == 1
    xlen = len(tmp1D)
    yield tmp1D                             # (0)->(0,*) V (0,0)->(0,0;*)
    try :
        tmp2D = next(x)                     # DIM == 2
        yield tmp2D                         # (0,1)->(0,1;*)
        yield tmp2D                         # (0,1)->(0,1;*) 
        yield outer( tmp1D, ones(xlen) )    # (0,1)->(0,0;*)
        yield outer( tmp2D, ones(xlen) )    # (0,1)->(0,1;*)
    except StopIteration:
        yield outer( tmp1D, ones(xlen) )    # (0)->(0;*) V (0,0)->(0,0;*)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#  SHIFTS/DERIVATIVES of BETA_FUNC  #
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def shift_1_deriv_1( compExp, a ):
    '''     < q_i ln(q_i) >     '''
    
    Omega_1 = compExp._Omega( "i", a )
    dvOmega_1_1 = compExp._derOmega( 1, "ii", a )
    Lambda_1 = compExp._Lambda( 1, a )

    gens = zip( Omega_1, Lambda_1, dvOmega_1_1 )
    sumGens = ( Om * ( Lm + dvOm ) for Om, Lm, dvOm in gens )
    return sumGens

def shift_2_deriv_1( compExp, a ):
    '''     < q_i q_j ln(q_i) >     '''
    
    Omega_2 = compExp._Omega( "ij", a )
    Lambda_1up = one_up_DX_( compExp._Lambda( 1, a ) )
    dvOmega_2_1 = compExp._derOmega( 2, "iji", a )

    gens = zip( Omega_2, Lambda_1up, dvOmega_2_1 )
    sumGens = ( Om_2 * ( Lm1 + dOm2_1 ) for Om_2, Lm1, dOm2_1 in gens )
    return sumGens

def shift_2_deriv_2( compExp, a ):
    '''     < q_i q_j ln(q_i) ln(q_j) >     '''

    Omega_2 = compExp._Omega( "ij", a )
    dvOmega_2_1 = compExp._derOmega( 2, "iji", a )
    dvOmega_2_2 = compExp._derOmega( 2, "ijij", a )
    Lambda_1up = one_up_SX_( compExp._Lambda( 1, a ) )
    Lambda_2 = compExp._Lambda( 2, a )

    gens = zip( Omega_2, Lambda_2, dvOmega_2_1, Lambda_1up, dvOmega_2_2 )
    sumGens = ( Om_2 * ( Lm2 + 2 * Lm1 * dOm2_1 + dOm2_2 ) for Om_2, Lm2, dOm2_1, Lm1, dOm2_2 in gens )
    return sumGens

def shift_3_deriv_1( compExp, a ):
    '''     < q_i q_j q_k ln(q_i) >     '''
    
    Omega_3 = compExp._Omega( "ijk", a )
    Lambda_1up = one_up_DX_( one_up_DX_( compExp._Lambda( 1, a ) ) )
    dvOmega_3_1 = compExp._derOmega( 3, "ijki", a )

    gens = zip( Omega_3, Lambda_1up, dvOmega_3_1 )
    sumGens = ( Om * ( Lm + dvOm ) for Om, Lm, dvOm in gens )
    return sumGens

def Q_shift_1_T_deriv_1( compDiv, a, b ):
    '''     < q_i ln(t_i) >     '''

    Omega_1_q = compDiv.compact_1._Omega( "i", a )
    Lambda_1_t = compDiv.compact_2._Lambda( 1, b )

    gens = zip( Omega_1_q, Lambda_1_t )
    sumGens = ( Om_q * Lm_t for Om_q, Lm_t in gens )
    return sumGens

def Q_shift_2_deriv_1_T_deriv_1( compDiv, a, b ):
    '''     < q_i q_j ln(q_i) ln(t_j) >     '''

    Omega_2_q = compDiv.compact_1._Omega( "ij", a )
    dvOmega_2_1 = compDiv.compact_1._derOmega( 2, "iji", a )
    Lambda_1up_q = one_up_DX_( compDiv.compact_1._Lambda( 1, a ) )
    Lambda_1up_t = one_up_SX_( compDiv.compact_2._Lambda( 1, b ) )

    gens = zip( Omega_2_q, Lambda_1up_q, dvOmega_2_1, Lambda_1up_t )
    sumGens = ( Om_2_q * ( Lm1_q + dOm2_1_q ) * Lm1_t for Om_2_q, Lm1_q, dOm2_1_q, Lm1_t in gens )
    return sumGens

def Q_shift_2_T_deriv_2( compDiv, a, b ):
    '''     < q_i q_j ln(t_i) ln(t_j) >     '''

    Omega_2_q = compDiv.compact_1._Omega( "ij", a )
    Lambda_2_t = compDiv.compact_2._Lambda( 2, b )

    gens = zip( Omega_2_q, Lambda_2_t )
    sumGens = ( Om2_q * Lm2_t for Om2_q, Lm2_t in gens )
    return sumGens

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#  INDIPENDENT PRIORS ESTIMATORS  #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def post_entropy_( compExp, a ):
    '''Estimate of the entropy at alpha.
        - sum_i < q_i ln(q_i) >
    '''

    sumGens = shift_1_deriv_1( compExp, a )
    sum_value = - compExp._norm_ffsum( sumGens, a, dim=1 )
    return sum_value

def post_entropy_sqr_( compExp, a ):
    '''Estimate of the squared entropy at alpha.
        sum_ij < q_i q_j ln(q_i) ln(q_j) >    
    '''

    sumGens = shift_2_deriv_2( compExp, a )
    sum_value = compExp._norm_ffsum( sumGens, a, dim=2 )
    return sum_value

def post_simpson_( compExp, a ):
    '''Estimate of the entropy at alpha.
        sum_i < q_i^2 >
    '''
    sumGens = compExp._Omega( "ii", a )
    sum_value = compExp._norm_ffsum( sumGens, a, dim=1, dimNorm=2 )
    return sum_value

def post_simpson_sqr_( compExp, a ):
    '''Estimate of the squared entropy at alpha.
        sum_ij < q_i^2 * q_j^2 >    
    '''

    sumGens = compExp._Omega( "iijj", a )
    sum_value = compExp._norm_ffsum( sumGens, a, dim=2, dimNorm=4 )
    return sum_value


def post_divergence_( compDiv, a, b ):
    '''Estimate of the divergence at alpha and beta.
        sum_i < q_i ln(q_i) - q_i ln(t_i) >
    '''

    sumGens_QlogQ = shift_1_deriv_1(compDiv.compact_1, a)
    sumGens_QlogT = Q_shift_1_T_deriv_1(compDiv, a, b)
    gens = zip( sumGens_QlogQ, sumGens_QlogT )
    sumGens = ( QlogQ - QlogT for QlogQ, QlogT in gens )

    return  compDiv.compact_1._norm_ffsum( sumGens, a, dim=1 )

def post_divergence_sqr_( compDiv, a, b ):
    '''Estimate of the squared divergence at alpha and beta.
        sum_i < q_i q_j ln(q_i) ln(q_j) > - 2 * < q_i q_j ln(q_i) ln(t_j) > + < q_i q_j ln(t_i) ln(t_j) >
    '''

    sumGens_Q2logQ2 = shift_2_deriv_2( compDiv.compact_1, a )
    sumGens_Q2logQlogT = Q_shift_2_deriv_1_T_deriv_1( compDiv, a, b )
    sumGens_Q2logT2 = Q_shift_2_T_deriv_2( compDiv, a, b )
    gens = zip( sumGens_Q2logQ2, sumGens_Q2logQlogT, sumGens_Q2logT2 )
    sumGens = ( Q2logQ2 - 2 * Q2logQlogT + Q2logT2 for Q2logQ2, Q2logQlogT, Q2logT2 in gens )

    return compDiv.compact_1._norm_ffsum( sumGens, a, dim=2 )