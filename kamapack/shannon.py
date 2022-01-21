#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Copyright (C) January 2022 Francesco Camaglia, LPENS 
'''

import warnings
import numpy as np
import pandas as pd
from . import default_entropy, default_divergence

class _skeleton_ :

    '''
    update_categories( categories )
        Change the number of categories.
    show
        Print a short summary.
    '''

    def update_categories( self, categories ):
        '''Change the number of categories.

        Parameters
        ----------        
        categories : scalar
                The new number of categories of the system (int or float). 
                If the value is lower than the observed number of categories, 
                the observed number is used instead.'''

        if categories is None:
            self.usr_n_categ = self.obs_n_categ    
        else :
            try : 
                categories = int( categories )
            except :        
                raise TypeError("The parameter `categories` must be an integer.")
            if categories > self.obs_n_categ :
                self.usr_n_categ = categories
            else :
                self.usr_n_categ = self.obs_n_categ  
                if categories < self.obs_n_categ :
                    warnings.warn("The parameter `categories` is set equal to the observed number of categories.")
    
    def show( self ):
        '''Print a short summary.'''
        
        print("Total number of counts:")
        if type(self.tot_counts) == pd.Series :
            print(self.tot_counts.to_string())
        else :
            print(self.tot_counts)
            
        print("N. of Categories:")
        print(self.obs_n_categ, ' (observed)')
        print(self.usr_n_categ, ' (a priori)')
        
        print("Recurrencies:")
        print(self.counts_hist)

        
######################
#  EXPERIMENT CLASS  #
######################

class Experiment( _skeleton_ ) :

    '''The basic class for estimating entropy from dataset distribution.
    
    Methods
    -------
    entropy( method="naive", unit="ln", **kwargs )
        Estimate Shannon entropy.'''

    __doc__ += _skeleton_.__doc__
    
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
            { 'nn' : self.nn,
             'ff' : self.ff }
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

######################
#  DIVERGENCE CLASS  #
######################


class Divergence( _skeleton_ ) :
    
    '''The basic class for estimating divergence between two dataset distributions.

    Methods
    -------
    jensen_shannon( method="naive", unit="ln", **kwargs ):
        Estimate Jensen-Shannon divergence.
    kullback_leibler( method="naive", unit="ln", **kwargs )
        Estimate Kullback-Leibler divergence.'''

    __doc__ += _skeleton_.__doc__

    def __init__( self, my_exp_A, my_exp_B ) :
        '''
        Parameters
        ----------    
        my_exp_A : class Experiment
            the first dataset.
        my_exp_B : class Experiment
            the second dataset.
        '''

        # WARNING!: add option to avoid create experiment at first
        
        self.tot_counts = pd.Series({"Exp-A": my_exp_A.tot_counts,
                                     "Exp-B": my_exp_B.tot_counts})

        df = pd.concat( [my_exp_A.data_hist, my_exp_B.data_hist], axis=1 )
        df = df.replace(np.nan, 0).astype(int)
        df.columns = ["Exp-A", "Exp-B"]
      
        self.data_hist = df
        self.obs_n_categ = len( df )
        
        self.counts_hist = df.groupby(by=["Exp-A", "Exp-B"]).size()
        
        categories = np.max([my_exp_A.usr_n_categ, my_exp_B.usr_n_categ])
        self.update_categories( categories )
        if self.usr_n_categ > self.obs_n_categ :
            self.counts_hist[(0,0)] = self.usr_n_categ - self.obs_n_categ
            self.counts_hist = self.counts_hist.sort_index(ascending=True)
        
        # WARNING!: is this a deep copy ?
        
        self.exp_A = my_exp_A
        self.exp_A.update_categories( self.usr_n_categ )
        self.exp_B = my_exp_B
        self.exp_B.update_categories( self.usr_n_categ )
        
    def kullback_leibler( self, method="naive", unit="ln", **kwargs ):
        '''Estimate Kullback-Leibler divergence.

        Kullback-Leibler divergence estimation through a chosen `method`.
        The unit (of the logarithm) can be specified with the parameter `unit`.
            
        Parameters
        ----------
        method: str
            the name of the estimation method:
            - "naive" : naive estimator (default);
            - "CMW" : Camaglia Mora Walczak estimator.
            - "Jeffreys" : Jeffreys estimator;
            - "Laplace" : Laplace estimator;
            - "SG" : Schurmann-Grassberger estimator;
            - "minimax" : minimax estimator;   
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
            - "naive" : naive estimator (default);
            - "Jeffreys" : Jeffreys estimator;
            - "Laplace" : Laplace estimator;
            - "SG" : Schurmann-Grassberger estimator;
            - "minimax" : minimax estimator;   
        unit: str, optional
            the divergence logbase unit:
            - "ln": natural logarithm (default);
            - "log2": base 2 logarihtm;
            - "log10":base 10 logarithm.

        return numpy.array
        '''
        
        return default_divergence.switchboard( self.compact(), method=method, which="Jensen-Shannon", unit=unit, **kwargs )

    def compact( self ) :
        '''It provides aliases for computations.'''
        return Divergence_Compact( self )
        
    def save_compact( self, filename ) :
        '''It saves the count hist features of Experiment to `filename`.'''
        self.compact( ).save( filename )
    
class Divergence_Compact :
    def __init__( self, divergence=None, filename=None ) :
        '''
        '''

        if divergence is not None :
            self.compact_A = divergence.exp_A.compact()                      # compact for Exp A
            self.compact_B = divergence.exp_B.compact()                      # compact for Exp B
            
            self.N_A = divergence.tot_counts['Exp-A']                        # total number of counts for Exp A
            self.N_B = divergence.tot_counts['Exp-B']                        # total number of counts for Exp B
            self.K = divergence.usr_n_categ                                  # user number of categories
            self.Kobs = divergence.obs_n_categ                               # observed number of categories
            temp = np.array(list(map(lambda x: [x[0],x[1]], divergence.counts_hist.index.values)))
            self.nn_A = temp[:,0]                                            # counts for Exp A
            self.nn_B = temp[:,1]                                            # counts for Exp B
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
            [ self.N_A, self.N_B, self.K, self.Kobs, len(self.ff) ],
                    index = ['N_A', 'N_B', 'K', 'Kobs', 'size_of_ff']
        ).to_csv( filename, sep=' ', mode='w', header=False, index=True )
        
        # counts hist
        pd.DataFrame(
            { 'nn_A' : self.nn_A,
             'nn_B' : self.nn_B,
             'ff' : self.ff }
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
        self.N_A = params['N_A']
        self.N_B = params['N_B']
        self.K = params['K']
        self.Kobs = params['Kobs']

        #count hist
        df = pd.read_csv( filename, header=5, sep=' ' )
        assert len(df) == params['size_of_ff']
        self.nn_A = df['nn_A'].values
        self.nn_B = df['nn_B'].values
        self.ff = df['ff'].values