#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Multivariate Beta Function Calculus
    Copyright (C) July 2023 Francesco Camaglia, LPENS 
'''
import warnings
import numpy as np
import pandas as pd
from numpy import outer
from scipy.special import loggamma, polygamma


# ########## #
#  NOTATION  #
# ########## #

def LogGmm(x): 
    ''' alias of Log Gamma function'''
    return loggamma(x).real  

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


# ########################## #
#  EXPERIMENT COMPACT CLASS  #
# ########################## #

class Experiment_Compact :
    def __init__(self, source=None, is_div=None, is_comp=False, load=None) :
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

    ''' *** Save/Load *** '''

    def save(self, filename) : 
        '''Save the Experiment_Compact object to `filename`.'''
        # parameters
        pd.DataFrame(
            [ self.N, self.K, self.Kobs, len(self.ff) ],
            index = ['N', 'K', 'Kobs', 'size_of_ff']
       ).to_csv(filename, sep=' ', mode='w', header=False, index=True)
        # counts hist
        pd.DataFrame(
            { 'nn': self.nn, 'ff': self.ff }
       ).to_csv(filename, sep=' ', mode='a', header=True, index=False)  

    def _load(self, filename) : 
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
        df = pd.read_csv(filename, header=4, sep=' ')
        assert len(df) == int(params['size_of_ff'])
        self.nn = df['nn'].values
        self.ff = df['ff'].values  

    ''' *** Multidimensional Summations *** '''

    def ffsum(self, sumGens, dim) :
        return count_hist_sum_(self.ff, sumGens, dim)

    def norm_ffsum(self, sumGens, a, dim, dimNorm=None) :
        '''Returns the normalization for the functions Omega.'''
        if dimNorm is None : dimNorm = dim

        X = self.K * a + self.N
        Norm = np.product(X * np.ones(dimNorm) + np.arange(0, dimNorm, 1))
        return np.divide(self.ffsum(sumGens, dim), Norm)


    ''' *** Multivariate Beta Compound : Generators *** '''

    def halfshift_i(self, a) :
        '''     sqrt(q_i)     '''
        xvec = self.nn + a
        X = self.N + self.K * a

        yield np.exp(LogGmm(X) - LogGmm(X+0.5) - LogGmm(xvec) + LogGmm(xvec+0.5)) 

    def halfshift_ij(self, a) :
        '''     sqrt(q_i) * sqrt(q_j)     '''
        xvec = self.nn + a
        X = self.N + self.K * a

        # i == j
        yield xvec

        # i != j
        yield outer(np.exp(LogGmm(xvec+0.5) - LogGmm(xvec)), np.exp(LogGmm(xvec+0.5) - LogGmm(xvec)))

    def shift_ii(self, a) :
        '''     q_i^2     '''
        xvec = self.nn + a

        yield xvec * (xvec+1)

    def shift_iijj(self, a) :
        '''     q_i^2 * q_j^2     '''
        xvec = self.nn + a

        # i == j
        yield xvec * (xvec+1) * (xvec+2) * (xvec+3)

        # i != j
        yield outer(xvec * (xvec+1), xvec * (xvec+1))

    def shift_i_deriv_i(self, a) :
        '''     q_i * ln(q_i)     '''
        xvec = self.nn + a
        X = self.N + self.K * a

        yield xvec * D_diGmm(xvec+1, X+1) 

    def shift_ij_deriv_ij(self, a) :
        '''     q_i * q_j * ln(q_i) * ln(q_j)     '''
        xvec = self.nn + a
        X = self.N + self.K * a

        # i == j
        yield xvec * (xvec+1) * (np.power(D_diGmm(xvec+2, X+2), 2) + D_triGmm(xvec+2, X+2))
        
        # i != j
        yield outer(xvec, xvec) * (outer(D_diGmm(xvec+1, X+2), D_diGmm(xvec+1, X+2)) - triGmm(X+2))
    
    ''' *** Methods *** '''

    def shannon(self, a) :
        '''Expected Shannon entropy under Polya posterior.
            - sum_i < q_i * ln(q_i) | n ; a >
        '''
        sumGens = self.shift_i_deriv_i(a)
        sum_value = - self.norm_ffsum(sumGens, a, dim=1)
        return sum_value

    def squared_shannon(self, a) :
        '''Expected squared Shannon entropy under Polya posterior.
            sum_ij < q_i * q_j * ln(q_i) * ln(q_j) | n ; a > 
        '''
        sumGens = self.shift_ij_deriv_ij(a)
        sum_value = self.norm_ffsum(sumGens, a, dim=2)
        return sum_value

    def simpson(self, a) :
        '''Expected Simpson index under Polya posterior.
            sum_i < q_i^2 | n ; a > 
        '''
        sumGens = self.shift_ii(a)
        sum_value = self.norm_ffsum(sumGens, a, dim=1, dimNorm=2)
        return sum_value

    def squared_simpson(self, a) :
        '''Expected squared Simpson index under Polya posterior.
            sum_ij < q_i^2 * q_j^2 | n ; a >   
        '''
        sumGens = self.shift_iijj(a)
        sum_value = self.norm_ffsum(sumGens, a, dim=2, dimNorm=4)
        return sum_value
###

# ########################## #
#  DIVERGENCE COMPACT CLASS  #
# ########################## #
    
class Divergence_Compact :
    def __init__(self, div=None, load=None) :
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
            self.compact_1 = Experiment_Compact(source=div, is_div=1)  # compact for Exp 1
            self.compact_2 = Experiment_Compact(source=div, is_div=2)  # compact for Exp 1
    
    def reverse(self) :
        '''Switch experiments within a Divergence_compact object.'''

        self.N_1, self.N_2 = self.N_2, self.N_1
        self.Kobs_1, self.Kobs_2 = self.Kobs_2, self.Kobs_1
        self.nn_1, self.nn_2 = self.nn_2, self.nn_1
        self.compact_1, self.compact_2 = self.compact_2, self.compact_1
    
    def save(self, filename) : 
        # FIXME : how to implement for compression? (change also load)
        '''Save the Divergence_Compact object to `filename`.'''
        # parameters
        pd.DataFrame(
            [ self.N_1, self.N_2, self.K, self.Kobs_1, self.Kobs_2, self.Kobs_u, len(self.ff) ],
            index = ['N_1', 'N_2', 'K', 'Kobs_1', 'Kobs_2', 'Kobs_u', 'size_of_ff']
       ).to_csv(filename, sep=' ', mode='w', header=False, index=True)
        # counts hist
        pd.DataFrame(
            { 'nn_1' : self.nn_1, 'nn_2' : self.nn_2, 'ff' : self.ff }
       ).to_csv(filename, sep=' ', mode='a', header=True, index=False)

    def _load(self, filename) : 
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
        df = pd.read_csv(filename, header=dict_size, sep=' ')
        assert len(df) == size_of_ff
        self.nn_1 = df['nn_1'].values
        self.nn_2 = df['nn_2'].values
        self.ff = df['ff'].values
        self.compact_1 = Experiment_Compact(source=self, is_div=1, is_comp=True)
        self.compact_2 = Experiment_Compact(source=self, is_div=2, is_comp=True)

    ''' *** Multivariate Beta Compound : Generators *** '''

    def Q_shift_i_T_deriv_i(self, a, b):
        '''     q_i ln(t_i)     '''
        xvec = self.nn_1 + a
        yvec = self.nn_2 + b
        Y = self.N_2 + self.K * b

        yield xvec * D_diGmm(yvec, Y) 

    def Q_shift_ij_T_deriv_ij(self, a, b):
        '''     q_i q_j ln(t_i) ln(t_j)     '''
        xvec = self.nn_1 + a
        yvec = self.nn_2 + b
        Y = self.N_2 + self.K * b

        # i == j
        yield xvec * (xvec+1) * (np.power(D_diGmm(yvec, Y), 2) + D_triGmm(yvec, Y))
        
        # i != j
        yield outer(xvec, xvec) * (outer(D_diGmm(yvec, Y), D_diGmm(yvec, Y)) - triGmm(Y))
    
    def Q_shift_ij_Q_deriv_i_T_deriv_j(self, a, b):
        '''     q_i q_j ln(q_i) ln(t_j)     '''
        xvec = self.nn_1 + a
        X = self.N_1 + self.K * a
        yvec = self.nn_2 + b
        Y = self.N_2 + self.K * b

        # i == j
        yield xvec * (xvec+1) * D_diGmm(xvec+2, X+2) * D_diGmm(yvec, Y)
        
        # i != j
        yield outer(xvec, xvec) * outer(D_diGmm(xvec+1, X+2), D_diGmm(yvec, Y))

    ''' *** Polya Posterior Methods *** '''

    def kullback_leibler(self, a, b) :
        '''Expected Kullback-Leibler divergence under Polya posterior.
                sum_i < q_i ln(q_i) - q_i ln(t_i) | n, m ; a, b > 
        '''
        sumGens_QlogQ = self.compact_1.shift_i_deriv_i(a)
        sumGens_QlogT = self.Q_shift_i_T_deriv_i(a, b)
        gens = zip(sumGens_QlogQ, sumGens_QlogT)
        sumGens = (QlogQ - QlogT for QlogQ, QlogT in gens)

        return self.compact_1.norm_ffsum(sumGens, a, dim=1)

    def squared_kullback_leibler(self, a, b) :
        '''Expected squared Kullback-Leibler divergence under Polya posterior.
            sum_ij < q_i q_j ln(q_i) ln(q_j) - 2 * < q_i q_j ln(q_i) ln(t_j) + q_i q_j ln(t_i) ln(t_j) | n, m ; a, b > 
        '''

        sumGens_Q2logQ2 = self.compact_1.shift_ij_deriv_ij(a)
        sumGens_Q2logQlogT = self.Q_shift_ij_Q_deriv_i_T_deriv_j(a, b)
        sumGens_Q2logT2 = self.Q_shift_ij_T_deriv_ij(a, b)
        gens = zip(sumGens_Q2logQ2, sumGens_Q2logQlogT, sumGens_Q2logT2)
        sumGens = (Q2logQ2 - 2 * Q2logQlogT + Q2logT2 for Q2logQ2, Q2logQlogT, Q2logT2 in gens)

        return self.compact_1.norm_ffsum(sumGens, a, dim=2)
    
    def bhattacharyya(self, a, b) :
        '''Posterior Bhattacharyya coefficient estimator.
            sum_i < sqrt{q_i} sqrt{t_i} | n, m ; a, b > 
        '''
        sumGens_Qh = self.compact_1.halfshift_i(a)
        sumGens_Th = self.compact_2.halfshift_i(b)
        gens = zip(sumGens_Qh, sumGens_Th)
        sumGens = (Qh * Th for Qh, Th in gens)

        return  self.compact_1.ffsum(sumGens, dim=1)

    def squared_bhattacharyya(self, a, b) :
        '''Posterior Bhattacharyya coefficient estimator.
            sum_ij < sqrt{q_i} sqrt{q_j} sqrt{t_i} sqrt{t_j} | n, m ; a, b > 
        '''
        sumGens_Qh2 = self.compact_1.halfshift_ij(a)
        sumGens_Th2 = self.compact_2.halfshift_ij(b)
        gens = zip(sumGens_Qh2, sumGens_Th2)
        sumGens = (Qh2 * Th2 for Qh2, Th2 in gens)
        X = self.N_1 + self.K * a
        Y = self.N_2 + self.K * b

        return np.divide(self.compact_1.ffsum(sumGens, dim=2), X * Y)
    
################
#  SUMMATIONS  #
################

def count_hist_sum_(ff, sumGens, dim) :
    ''' Summing methods for histograms of counts.'''

    # (0,...) : all ==        
    tmp1D = next(sumGens)

    if dim == 2 :
        # (0,1) : i!=j
        tmp2D = next(sumGens)
        # summing
        tmp1D += tmp2D.dot(ff) - tmp2D.diagonal()
    
    output =tmp1D.dot(ff)

    return output        