#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Copyright (C) November 2021 Francesco Camaglia, LPENS 
'''

import warnings

import numpy as np
import pandas as pd

from . import default_entropy, default_divergence

class _skeleton_ :

    def update_categories( self, categories ):
        '''
        To change the number of categories.

        Parameters
        ----------        
        categories: scalar
                The new number of categories of the system. If the value is lower than observed number of categories, 
                the observed number is used instead. Default is observed number.
        '''
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
        '''
        To print a short summary.
        '''
        
        print("Total number of counts:")
        if type(self.tot_counts) == pd.Series :
            print(self.tot_counts.to_string())
        else :
            print(self.tot_counts)
            
        print("N. of Categories:")
        print(self.obs_n_categ, ' (observed)')
        print(self.usr_n_categ, ' (a priori)')
        
        print("Recurrencies:")
        if type(self.counts_hist) == pd.Series :
            print(self.counts_hist.to_string())
        else :
            print(self.counts_hist)
        
######################
#  EXPERIMENT CLASS  #
######################

class Experiment( _skeleton_ ) :
    
    def __init__( self, data, categories=None, iscount=False ):
        '''
        TO BE UPDATED
        Parameters
        ----------      
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
                temp = pd.Series( sequences )
                data_hist = output.groupby( temp ).size()
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
    '''
    Methods
    -------
    '''

    def entropy( self, method, unit="ln", **kwargs ):
        '''
        Shannon entropy estimation over a given Experiment class object with a chosen `method`.
        The unit of the logarithm can be specified through the parameter `unit`.
            
        Parameters
        ----------
        method: str
                the name of the entropy estimation method:
                - "ML": Maximum Likelihood estimator;
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
        
        return default_entropy.switchboard( self.compact(), method, unit=unit, **kwargs )
    
    def compact( self ) :
        '''
        It provides aliases useful for computations.
        '''
        return _Experiment_Compact_( self )
    
class _Experiment_Compact_ :

    def __init__( self, experiment ) :
        self.N = experiment.tot_counts                                   # total number of counts
        self.K = experiment.usr_n_categ                                  # user number of categories
        self.Kobs = experiment.obs_n_categ                               # observed number of categories
        self.nn = experiment.counts_hist.index.values                    # counts
        self.ff = experiment.counts_hist.values                          # recurrency of counts
        
######################
#  DIVERGENCE CLASS  #
######################

class Divergence( _skeleton_ ) :
    
    def __init__( self, my_exp_A, my_exp_B ) :
        
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
        
    '''
    Methods
    -------
    '''

    def kullback_leibler( self, method, unit="ln", **kwargs ):
        '''
        Kullback-Leibler divergence estimation over a given Divergence class object through a chosen `method`.
        The unit of the logarithm can be specified through the parameter `unit`.
            
        Parameters
        ----------
        method: str
                the name of the Kullback-Leibler estimation method:
                - "ML": Maximum Likelihood estimator;
                - "NSB": Nemenman Shafee Bialek estimator.
                - "Jeffreys": Jeffreys estimator;
                - "Laplace": Laplace estimator;
                - "SG": Schurmann-Grassberger estimator;
                - "minimax": minimax estimator;                          
        unit: str, optional
                the entropy logbase unit:
                - "ln": natural logarithm (default);
                - "log2": base 2 logarihtm;
                - "log10":base 10 logarithm.

        return numpy.array
        '''
        
        return default_divergence.switchboard( self.compact(), method, unit=unit, **kwargs )
        
    def compact( self ) :
        '''
        It provides aliases useful for computations.
        '''
        return _Divergence_Compact_( self )
    
class _Divergence_Compact_ :

    def __init__( self, divergence ) :

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
