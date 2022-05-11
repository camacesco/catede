#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Copyright (C) January 2022 Francesco Camaglia, LPENS 
'''

import warnings
import numpy as np
import pandas as pd
from . import default_entropy, default_divergence
from ._aux_shannon import *
from ._wolpert_wolf_calculus import *

import joblib # FIXME : this approach is not optimal
def load_class( filename ) :
    '''Load Experiment/Divergence objected saved at `filename`.'''
    return joblib.load( filename )

######################
#  EXPERIMENT CLASS  #
######################

class Experiment( Skeleton_Class ) :
    '''The basic class for estimating entropy from dataset distribution.
    
    Methods
    -------
    entropy( method="naive", unit="ln", **kwargs )
        Estimate Shannon entropy.'''

    __doc__ += Skeleton_Class.__doc__
    
    def __init__( self, data, categories=None, iscount=False ):
        '''
        Parameters
        ----------    
        data : Union[dict, pd.DataFrame, pd.Series, list, np.array] 
            pd.DataFrame (deprecated)
        categories : scalar, optional
        iscount : bool
            (default is False)
        '''
        
        #  Load data  #
        if type( data ) == dict :   
            # loading dictionary where values represent keys counts                                    
            data_hist = pd.Series( data )
        elif type( data ) == pd.DataFrame :
            # loading datafame where values represent keys counts 
            data_hist = data[ data.columns[0] ]
            if len( data.columns ) > 1 :
                warnings.warn("The parameter `data` contained multiple columns, only the first one is considered.")
        elif type( data ) == pd.Series :
            # loading series where values represent index counts
            data_hist = data
        elif type( data ) == list or type( data ) == np.ndarray :                              
            if iscount == True :
                # loading list of counts
                data_hist = pd.Series( data ).astype(int)
            else :
                # loading raw list of sequences
                temp = pd.Series( data )
                data_hist = temp.groupby( temp ).size()
        else :
            raise TypeError("The parameter `data` has unsopported type.")

        # check int counts 
        try :
            data_hist = data_hist.astype(int)
        except ValueError :
            # WARNING! : this try/except doesn't looks nice
            raise ValueError("Unrecognized count value in `data`.")    

        self.data_hist = data_hist
        self.tot_counts = np.sum( data_hist.values )  
        self.obs_n_categ = len( data_hist ) # observed categories
        temp = pd.Series( data_hist.values ).astype(int)
        self.counts_hist = temp.groupby( temp ).size()  
        self.counts_hist.name = "freq"

        #  Load categories  #
        self.update_categories( categories )
        if self.usr_n_categ > self.obs_n_categ :
            self.counts_hist[ 0 ] = self.usr_n_categ - self.obs_n_categ
            self.counts_hist = self.counts_hist.sort_index(ascending=True) 

    def entropy( self, method="naive", unit="ln", **kwargs ):
        '''Estimate Shannon entropy.

        Shannon entropy estimation through a chosen `method`.
        The unit (of the logarithm) can be specified with the parameter `unit`.
            
        Parameters
        ----------
        method: str
            the name of the entropy estimation method:
            - "naive": naive estimator (default);
            - "MM": Miller Madow estimator;
            - "CS": Chao Shen estimator;       
            - "shrink": shrinkage estimator;       
            - "Jeffreys": Jeffreys estimator;
            - "Laplace": Laplace estimator;
            - "SG": Schurmann-Grassberger estimator;
            - "minimax": minimax estimator;
            - "NSB": Nemenman Shafee Bialek estimator.
        unit: str, optional
            the entropy logbase unit:
            - "ln": natural logarithm (default);
            - "log2": base 2 logarihtm;
            - "log10":base 10 logarithm.

        return numpy.array
        '''
        
        return default_entropy.switchboard( self.compact(), method=method, unit=unit, **kwargs )
    
    def compact( self ) :
        '''It provides aliases for computations.'''
        return Experiment_Compact( experiment=self )

    def rank_plot( self, figsize=(3,3), color="#ff0054", xlabel="rank", ylabel="frequency", grid=True, logscale=True) :
        '''Rank plot.'''
        return experiment_rank_plot( self, figsize=figsize, color=color, xlabel=xlabel, ylabel=ylabel, logscale=logscale, grid=grid)
     
    def save_compact( self, filename ) :
        '''It saves the compact features of Experiment to `filename`.'''
        self.compact( ).save( filename )

class Experiment_Compact :
    def __init__( self, experiment=None, filename=None ) :
        '''
        '''
        if experiment is not None :
            self.N = experiment.tot_counts                                   # total number of counts
            self.K = experiment.usr_n_categ                                  # user number of categories
            self.Kobs = experiment.obs_n_categ                               # observed number of categories
            self.nn = experiment.counts_hist.index.values                    # counts
            self.ff = experiment.counts_hist.values                          # recurrency of counts
        elif filename is not None :
            self.load( filename )    
        else :
            raise TypeError( 'One between `experiment` and `filename` has to be defined.')

    def save( self, filename ) : 
        '''
        '''
        # parameters
        pd.DataFrame(
            [ self.N, self.K, self.Kobs, len(self.ff) ],
                    index = ['N', 'K', 'Kobs', 'size_of_ff']
        ).to_csv( filename, sep=' ', mode='w', header=False, index=True )
        
        # counts hist
        pd.DataFrame(
            { 'nn' : self.nn, 'ff' : self.ff }
        ).to_csv( filename, sep=' ', mode='a', header=True, index=False )        

    def load( self, filename ) : 
        '''
        '''
        # parameters
        f = open(filename, "r")
        params = {}
        for _ in range(4) :
            thisline = f.readline().strip().split(' ')
            params[ thisline[0] ] = thisline[1]
        self.N = int(params[ 'N' ])
        self.K = int(params['K'])
        self.Kobs = int(params['Kobs'])
        
        #count hist
        df = pd.read_csv( filename, header=4, sep=' ' )
        assert len(df) == int(params['size_of_ff'])
        self.nn = df['nn'].values
        self.ff = df['ff'].values

    def _measureMu( self,  a ) :
        '''Practical alias for the measure mu.'''
        return measureMu_( self,  a )

    def _Omega( self, shift, a) :
        '''Practical alias for Wolpert-Wolf function Omega.'''
        return Omega_( self, shift, a)

    def _Lambda( self, order, shift, a) :
        '''Practical alias for Wolpert-Wolf function Lambda.'''
        return Lambda_( self, order, shift, a)

    def _der_Lambda( self, order, shift, deriv, a) :
        '''Practical alias for Wolpert-Wolf derivative of function Lambda.'''
        return der_Lambda_( self, order, shift, deriv, a)

    def _ffsum( self, sumList ) :
        return ffsum_( self.ff, sumList )

    def _post_entropy( self, a ) :
        '''Practical alias for Posterior Multinomial-Dirichlet entropy estimator.'''
        return post_entropy_( self, a )

    def _post_entropy_squared( self, a ) :
        '''Practical alias for Posterior Multinomial-Dirichlet sqaured entropy estimator.'''
        return post_entropy_squared_( self, a )

######################
#  DIVERGENCE CLASS  #
######################

class Divergence( Skeleton_Class ) :
    
    '''The basic class for estimating divergence between two dataset distributions.

    Methods
    -------
    jensen_shannon( method="naive", unit="ln", **kwargs ):
        Estimate Jensen-Shannon divergence.
    kullback_leibler( method="naive", unit="ln", **kwargs )
        Estimate Kullback-Leibler divergence.'''

    __doc__ += Skeleton_Class.__doc__

    def __init__( self, my_exp_1, my_exp_2 ) :
        '''
        Parameters
        ----------    
        my_exp_1 : class Experiment
            the first dataset.
        my_exp_2 : class Experiment
            the second dataset.
        '''

        # WARNING!: add option to avoid create experiment at first
        
        self.tot_counts = pd.Series(
            {"Exp-1": my_exp_1.tot_counts,
            "Exp-2": my_exp_2.tot_counts}
            )

        df = pd.concat( [my_exp_1.data_hist, my_exp_2.data_hist], axis=1 )
        df = df.replace(np.nan, 0).astype(int)
        df.columns = ["Exp-1", "Exp-2"]
      
        self.data_hist = df
        self.obs_n_categ = len( df )
        
        self.counts_hist = df.groupby(by=["Exp-1", "Exp-2"]).size()
        
        categories = np.max([my_exp_1.usr_n_categ, my_exp_2.usr_n_categ])
        self.update_categories( categories )
        if self.usr_n_categ > self.obs_n_categ :
            self.counts_hist[(0,0)] = self.usr_n_categ - self.obs_n_categ
            self.counts_hist = self.counts_hist.sort_index(ascending=True)
        self.counts_hist.name = "freq"

        # WARNING!: is this a deep copy ?
        tmp = self.counts_hist.reset_index(level=[0,1]).copy()
        # experiment 1 copy
        self.exp_1 = Experiment( my_exp_1.data_hist )
        self.exp_1.update_categories( self.usr_n_categ )
        counts_1 = tmp[["freq", "Exp-1"]].set_index("Exp-1", drop=True)
        self.exp_1.counts_hist = counts_1["freq"] # convert to series
        # experiment 2 copy
        self.exp_2 = Experiment( my_exp_2.data_hist )
        self.exp_2.update_categories( self.usr_n_categ )
        counts_2 = tmp[["freq", "Exp-2"]].set_index("Exp-2", drop=True)
        self.exp_2.counts_hist = counts_2["freq"]

    def kullback_leibler( self, method="naive", unit="ln", **kwargs ):
        '''Estimate Kullback-Leibler divergence.

        Kullback-Leibler divergence estimation through a chosen `method`.
        The unit (of the logarithm) can be specified with the parameter `unit`.
            
        Parameters
        ----------
        method: str
            the name of the estimation method:
            - ["naive", "maximum-likelihood"] : naive estimator (default);
            - "CMW" : Camaglia Mora Walczak estimator.
            - ["Jeffreys", "Krichevsky-Trofimov"] : Jeffreys estimator;
            - ["L", "Laplace", "Bayesian-Laplace"] : Laplace estimator;
            - ["SG", "Schurmann-Grassberger"] : Schurmann-Grassberger estimator;
            - ["minimax", "Trybula"] : minimax estimator; 
            - ["D", "Dirichlet"] : Dirichlet   
        unit: str, optional
            the divergence logbase unit:
            - "ln": natural logarithm (default);
            - "log2": base 2 logarihtm;
            - "log10":base 10 logarithm.

        return numpy.array
        '''
        
        return default_divergence.switchboard( self.compact(), which="Kullback-Leibler", method=method, unit=unit, **kwargs )

    def jensen_shannon( self, method="naive", unit="ln", **kwargs ):
        '''Estimate Jensen-Shannon divergence.

        Jensen-Shannon divergence estimation through a chosen `method`.
        The unit (of the logarithm) can be specified with the parameter `unit`.
            
        Parameters
        ----------
        method: str
            the name of the estimation method:
            - ["naive", "maximum-likelihood"] : naive estimator (default);
            - ["Jeffreys", "Krichevsky-Trofimov"] : Jeffreys estimator;
            - ["L", "Laplace", "Bayesian-Laplace"] : Laplace estimator;
            - ["SG", "Schurmann-Grassberger"] : Schurmann-Grassberger estimator;
            - ["minimax", "Trybula"] : minimax estimator; 
            - ["D", "Dirichlet"] : Dirichlet    
        unit: str, optional
            the divergence logbase unit:
            - "ln": natural logarithm (default);
            - "log2": base 2 logarihtm;
            - "log10":base 10 logarithm.

        return numpy.array
        '''
        
        return default_divergence.switchboard( self.compact(), method=method, which="Jensen-Shannon", unit=unit, **kwargs )

    def rank_plot( self, figsize=(3,3), color1="#ff0054", color2="#0088ff", xlabel="rank", ylabel="frequency", logscale=True, grid=True, by_first=True) :
        '''Rank plot.'''
        return divergence_rank_plot( self, figsize=figsize, color1=color1, color2=color2, xlabel=xlabel, ylabel=ylabel, logscale=logscale, grid=grid, by_first=by_first )

    def compact( self ) :
        '''It provides aliases for computations.'''
        return Divergence_Compact( self )
        
    def save_compact( self, filename ) :
        '''It saves the compact version of Experiment to `filename`.'''
        self.compact( ).save( filename )


##############################
#  DIVERGENCE COMPACT CLASS  #
##############################
    
class Divergence_Compact :
    def __init__( self, divergence=None, filename=None ) :
        '''
        '''

        if divergence is not None :
            self.compact_1 = divergence.exp_1.compact()                      # compact for Exp 1
            self.compact_2 = divergence.exp_2.compact()                      # compact for Exp 2
            
            self.N_1 = divergence.tot_counts['Exp-1']                        # total number of counts for Exp 1
            self.N_2 = divergence.tot_counts['Exp-2']                        # total number of counts for Exp 2
            self.K = divergence.usr_n_categ                                  # user number of categories
            self.Kobs = divergence.obs_n_categ                               # observed number of categories
            temp = np.array(list(map(lambda x: [x[0],x[1]], divergence.counts_hist.index.values)))
            self.nn_1 = temp[:,0]                                            # counts for Exp 1
            self.nn_2 = temp[:,1]                                            # counts for Exp 2
            self.ff = divergence.counts_hist.values                          # recurrency of counts
        
        elif filename is not None :
            self.load( filename )
        
        else :
            raise TypeError( 'One between `experiment` and `filename` has to be defined.')

    def save( self, filename ) : 
        '''
        '''

        # parameters
        pd.DataFrame(
            [ self.N_1, self.N_2, self.K, self.Kobs, len(self.ff) ],
            index = ['N_1', 'N_2', 'K', 'Kobs', 'size_of_ff']
        ).to_csv( filename, sep=' ', mode='w', header=False, index=True )
        
        # counts hist
        pd.DataFrame(
            { 'nn_1' : self.nn_1, 'nn_2' : self.nn_2, 'ff' : self.ff }
        ).to_csv( filename, sep=' ', mode='a', header=True, index=False ) 
    
    def load( self, filename ) : 
        '''
        '''

        # parameters
        f = open(filename, "r")
        params = {}
        for _ in range(5) :
            thisline = f.readline().strip().split(' ')
            params[ thisline[0] ] = thisline[1]
        self.N_1 = params['N_1']
        self.N_2 = params['N_2']
        self.K = params['K']
        self.Kobs = params['Kobs']

        #count hist
        df = pd.read_csv( filename, header=5, sep=' ' )
        assert len(df) == params['size_of_ff']
        self.nn_1 = df['nn_1'].values
        self.nn_2 = df['nn_2'].values
        self.ff = df['ff'].values

    def _ffsum( self, sumList ) :
        return ffsum_( self.ff, sumList )

    def _post_divergence( self, a, b ) :
        return post_divergence_( self, a, b )

    def _post_divergence_squared( self, a, b ) :
        return post_divergence_squared_( self, a, b )